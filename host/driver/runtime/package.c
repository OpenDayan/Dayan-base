/***************************************************************************************
 * Copyright (c) 2026 Institute of Computing Technology, Chinese Academy of Sciences
 *
 * Dayan-base is licensed under the Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *
 *     http://license.coscl.org.cn/MulanPSL2
 *
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 *
 * See the Mulan PSL v2 for more details.
***************************************************************************************/

#include "npu_package.h"

#include <elf.h>
#include <errno.h>
#include <stdlib.h>
#include <string.h>

#define NPU_PACKAGE_ABI_LP64 1U

static int align_size(size_t value, size_t alignment, size_t *result)
{
    size_t mask = alignment - 1U;

    if (value > SIZE_MAX - mask)
        return -EOVERFLOW;
    *result = (value + mask) & ~mask;
    return 0;
}

static int add_size(size_t *value, size_t amount)
{
    if (amount > SIZE_MAX - *value)
        return -EOVERFLOW;
    *value += amount;
    return 0;
}

static int object_valid(const void *object, size_t size)
{
    Elf64_Ehdr header;

    if (!object || size < sizeof(header))
        return 0;
    memcpy(&header, object, sizeof(header));
    return memcmp(header.e_ident, ELFMAG, SELFMAG) == 0 &&
           header.e_ident[EI_CLASS] == ELFCLASS64 &&
           header.e_ident[EI_DATA] == ELFDATA2LSB &&
           header.e_ident[EI_VERSION] == EV_CURRENT &&
           header.e_type == ET_REL && header.e_machine == EM_RISCV &&
           header.e_version == EV_CURRENT;
}

static int argument_valid(const npu_package_argument_t *argument)
{
    if (!argument)
        return 0;
    if (argument->kind == NPU_PACKAGE_ARG_VALUE)
        return (argument->size == 1 || argument->size == 2 ||
                argument->size == 4 || argument->size == 8) &&
               !argument->access;
    return argument->kind == NPU_PACKAGE_ARG_BUFFER &&
           argument->size == 8 && argument->alignment &&
           !(argument->alignment & (argument->alignment - 1U)) &&
           argument->access &&
           !(argument->access & ~(NPU_PACKAGE_ARG_READ |
                                 NPU_PACKAGE_ARG_WRITE));
}

int npu_package_build(const void *riscv_object, size_t object_size,
                      const npu_package_kernel_desc_t *kernels,
                      size_t kernel_count, uint32_t flags,
                      void **package_out, size_t *package_size_out)
{
    static const char isa[] = NPU_PACKAGE_ISA_RV64_DAYAN;
    npu_package_header_t *header;
    npu_package_kernel_t *kernel_table;
    npu_package_argument_t *argument_table;
    npu_package_sync_t *sync_table;
    unsigned char *package;
    size_t total_arguments = 0;
    size_t total_sync = 0;
    size_t strings_size = sizeof(isa);
    size_t cursor;
    size_t string_cursor;
    size_t index;
    int status;

    if (!package_out || !package_size_out)
        return -EINVAL;
    *package_out = 0;
    *package_size_out = 0;
    if (!object_valid(riscv_object, object_size) || !kernels ||
        !kernel_count || kernel_count > NPU_PACKAGE_MAX_KERNELS ||
        (flags & ~NPU_PACKAGE_IMMUTABLE_MODULE))
        return -EINVAL;
    for (index = 0; index < kernel_count; ++index) {
        const npu_package_kernel_desc_t *kernel = &kernels[index];
        size_t argument_index;
        size_t sync_index;

        if (!kernel->name || !*kernel->name || !kernel->entry_symbol ||
            !*kernel->entry_symbol ||
            !kernel->supported_task_modes ||
            (kernel->supported_task_modes & ~NPU_PACKAGE_TASK_MODE_ALL) ||
            (kernel->flags & ~NPU_PACKAGE_KERNEL_FLAGS_ALL) ||
            ((kernel->flags & NPU_PACKAGE_KERNEL_LOCKED_TASK) &&
             kernel->supported_task_modes !=
                 NPU_PACKAGE_TASK_MODE_SINGLE_CORE) ||
            ((kernel->flags & NPU_PACKAGE_KERNEL_LOCKED_TASK) &&
             kernel->max_tasks_per_core > 1U) ||
            kernel->max_child_threads > NPU_PACKAGE_MAX_CHILD_THREADS ||
            kernel->lsu_tcsr_count > NPU_PACKAGE_TCSR_MAX_COUNT ||
            kernel->mat_tcsr_count > NPU_PACKAGE_TCSR_MAX_COUNT ||
            kernel->vec_tcsr_count > NPU_PACKAGE_TCSR_MAX_COUNT ||
            kernel->macro_tcsr_count > NPU_PACKAGE_TCSR_MAX_COUNT ||
            kernel->local_resource_count > NPU_PACKAGE_RESOURCE_MAX_COUNT ||
            kernel->shared_resource_count > NPU_PACKAGE_RESOURCE_MAX_COUNT ||
            (kernel->stack_size &&
             kernel->stack_size < NPU_PACKAGE_PROC_MIN_STACK_SIZE) ||
            kernel->stack_size > NPU_PACKAGE_PROC_STACK_SIZE ||
            kernel->printf_buffer_size > NPU_PACKAGE_PROC_PRINTF_SIZE ||
            (kernel->uram_size & (NPU_PACKAGE_URAM_GRANULE - 1U)) ||
            kernel->argument_count > NPU_PACKAGE_MAX_ARGS ||
            (kernel->argument_count && !kernel->arguments) ||
            (kernel->sync_count && !kernel->sync_resources) ||
            kernel->argument_count > UINT32_MAX - total_arguments ||
            kernel->sync_count > UINT32_MAX - total_sync)
            return -EINVAL;
        for (argument_index = 0; argument_index < kernel->argument_count;
             ++argument_index) {
            if (!argument_valid(&kernel->arguments[argument_index]))
                return -EINVAL;
        }
        for (sync_index = 0; sync_index < kernel->sync_count; ++sync_index) {
            const npu_package_sync_t *sync =
                &kernel->sync_resources[sync_index];

            if (sync->type != NPU_PACKAGE_SYNC_SHARED_RESOURCE ||
                !sync->count || !sync->consumer_count || sync->flags ||
                sync->reserved)
                return -EINVAL;
        }
        total_arguments += kernel->argument_count;
        total_sync += kernel->sync_count;
        status = add_size(&strings_size, strlen(kernel->name) + 1U);
        if (status < 0)
            return status;
        status = add_size(&strings_size, strlen(kernel->entry_symbol) + 1U);
        if (status < 0)
            return status;
    }
    cursor = sizeof(*header);
    if (kernel_count > (SIZE_MAX - cursor) / sizeof(*kernel_table))
        return -EOVERFLOW;
    cursor += kernel_count * sizeof(*kernel_table);
    status = align_size(cursor, 8, &cursor);
    if (status < 0)
        return status;
    if (total_arguments > (SIZE_MAX - cursor) / sizeof(*argument_table))
        return -EOVERFLOW;
    cursor += total_arguments * sizeof(*argument_table);
    status = align_size(cursor, 8, &cursor);
    if (status < 0)
        return status;
    if (total_sync > (SIZE_MAX - cursor) / sizeof(*sync_table))
        return -EOVERFLOW;
    cursor += total_sync * sizeof(*sync_table);
    status = align_size(cursor, 8, &cursor);
    if (status < 0)
        return status;
    if (strings_size > SIZE_MAX - cursor)
        return -EOVERFLOW;
    cursor += strings_size;
    status = align_size(cursor, 8, &cursor);
    if (status < 0 || object_size > SIZE_MAX - cursor)
        return status < 0 ? status : -EOVERFLOW;
    cursor += object_size;
    package = calloc(1, cursor);
    if (!package)
        return -ENOMEM;
    header = (npu_package_header_t *)package;
    memcpy(header->magic, NPU_PACKAGE_MAGIC, sizeof(header->magic));
    header->version = NPU_PACKAGE_VERSION;
    header->header_size = sizeof(*header);
    header->total_size = cursor;
    header->flags = flags;
    header->target_abi = NPU_PACKAGE_ABI_LP64;
    header->kernel_offset = sizeof(*header);
    header->kernel_count = (uint32_t)kernel_count;
    header->argument_offset =
        (header->kernel_offset + kernel_count * sizeof(*kernel_table) + 7U) &
        ~UINT64_C(7);
    header->argument_count = (uint32_t)total_arguments;
    header->sync_offset =
        (header->argument_offset + total_arguments * sizeof(*argument_table) +
         7U) & ~UINT64_C(7);
    header->sync_count = (uint32_t)total_sync;
    header->string_offset =
        (header->sync_offset + total_sync * sizeof(*sync_table) + 7U) &
        ~UINT64_C(7);
    header->string_size = strings_size;
    header->isa_string_offset = (uint32_t)header->string_offset;
    header->isa_string_size = sizeof(isa) - 1U;
    header->object_offset = (header->string_offset + strings_size + 7U) &
                            ~UINT64_C(7);
    header->object_size = object_size;

    kernel_table = (npu_package_kernel_t *)(package + header->kernel_offset);
    argument_table = (npu_package_argument_t *)(package +
                                                header->argument_offset);
    sync_table = (npu_package_sync_t *)(package + header->sync_offset);
    string_cursor = (size_t)header->string_offset;
    memcpy(package + string_cursor, isa, sizeof(isa));
    string_cursor += sizeof(isa);
    total_arguments = 0;
    total_sync = 0;
    for (index = 0; index < kernel_count; ++index) {
        const npu_package_kernel_desc_t *source = &kernels[index];
        npu_package_kernel_t *destination = &kernel_table[index];
        size_t length;

        destination->version = NPU_PACKAGE_KERNEL_VERSION;
        destination->name_offset = (uint32_t)string_cursor;
        length = strlen(source->name) + 1U;
        memcpy(package + string_cursor, source->name, length);
        string_cursor += length;
        destination->entry_offset = (uint32_t)string_cursor;
        length = strlen(source->entry_symbol) + 1U;
        memcpy(package + string_cursor, source->entry_symbol, length);
        string_cursor += length;
        destination->supported_task_modes = source->supported_task_modes;
        destination->first_argument = (uint32_t)total_arguments;
        destination->argument_count = (uint32_t)source->argument_count;
        destination->first_sync = (uint32_t)total_sync;
        destination->sync_count = (uint32_t)source->sync_count;
        destination->max_tasks_per_core = source->max_tasks_per_core ?
            source->max_tasks_per_core : 1U;
        destination->max_child_threads = source->max_child_threads;
        destination->flags = source->flags;
        destination->lsu_tcsr_count = source->lsu_tcsr_count;
        destination->mat_tcsr_count = source->mat_tcsr_count;
        destination->vec_tcsr_count = source->vec_tcsr_count;
        destination->macro_tcsr_count = source->macro_tcsr_count;
        destination->local_resource_count = source->local_resource_count;
        destination->shared_resource_count = source->shared_resource_count;
        destination->stack_size = source->stack_size;
        destination->atomic_cache_size = source->atomic_cache_size;
        destination->printf_buffer_size = source->printf_buffer_size;
        destination->resident_memory = source->resident_memory;
        destination->uram_size = source->uram_size;
        if (source->argument_count)
            memcpy(argument_table + total_arguments, source->arguments,
                   source->argument_count * sizeof(*argument_table));
        if (source->sync_count)
            memcpy(sync_table + total_sync, source->sync_resources,
                   source->sync_count * sizeof(*sync_table));
        total_arguments += source->argument_count;
        total_sync += source->sync_count;
    }
    memcpy(package + header->object_offset, riscv_object, object_size);
    *package_out = package;
    *package_size_out = cursor;
    return 0;
}

void npu_package_free(void *package_image)
{
    free(package_image);
}
