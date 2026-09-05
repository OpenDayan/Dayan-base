#include "package.h"

#include <elf.h>
#include <errno.h>
#include <string.h>

#define NPU_PACKAGE_ABI_LP64 1U

static int range_valid(size_t total, uint64_t offset, uint64_t size)
{
    return offset <= total && size <= (uint64_t)total - offset;
}

static int table_valid(size_t total, uint64_t offset, uint32_t count,
                       size_t element_size)
{
    return count <= UINT64_MAX / element_size &&
           range_valid(total, offset, (uint64_t)count * element_size) &&
           !(offset & 7U);
}

const char *npu_package_string(const struct npu_package_view *view,
                               uint32_t offset)
{
    uint64_t string_end;
    const char *string;

    if (!view || !view->header ||
        view->header->string_offset > UINT64_MAX -
                                      view->header->string_size)
        return 0;
    string_end = view->header->string_offset + view->header->string_size;
    if (offset < view->header->string_offset || offset >= string_end)
        return 0;
    string = (const char *)view->image + offset;
    return memchr(string, '\0', (size_t)(string_end - offset)) ? string : 0;
}

static int object_valid(const unsigned char *object, size_t size)
{
    Elf64_Ehdr header;

    if (size < sizeof(header))
        return 0;
    memcpy(&header, object, sizeof(header));
    return memcmp(header.e_ident, ELFMAG, SELFMAG) == 0 &&
           header.e_ident[EI_CLASS] == ELFCLASS64 &&
           header.e_ident[EI_DATA] == ELFDATA2LSB &&
           header.e_ident[EI_VERSION] == EV_CURRENT &&
           header.e_type == ET_REL && header.e_machine == EM_RISCV &&
           header.e_version == EV_CURRENT &&
           header.e_ehsize == sizeof(header) &&
           header.e_shentsize == sizeof(Elf64_Shdr) && header.e_shnum &&
           header.e_shoff <= size &&
           header.e_shnum <= (size - header.e_shoff) / sizeof(Elf64_Shdr);
}

int npu_package_view_init(struct npu_package_view *view,
                          const void *image, size_t size)
{
    const npu_package_header_t *header;
    const npu_package_kernel_t *kernels;
    const npu_package_argument_t *arguments;
    const npu_package_sync_t *sync_resources;
    size_t index;

    if (!view || !image || size < sizeof(*header))
        return -EINVAL;
    memset(view, 0, sizeof(*view));
    header = image;
    if (memcmp(header->magic, NPU_PACKAGE_MAGIC, sizeof(header->magic)) != 0 ||
        header->version != NPU_PACKAGE_VERSION ||
        header->header_size != sizeof(*header) ||
        header->total_size != size || header->reserved0 || header->reserved1 ||
        header->reserved2 || header->reserved[0] ||
        (header->flags & ~NPU_PACKAGE_IMMUTABLE_MODULE) ||
        header->target_abi != NPU_PACKAGE_ABI_LP64 ||
        !header->kernel_count ||
        header->kernel_count > NPU_PACKAGE_MAX_KERNELS ||
        !table_valid(size, header->kernel_offset, header->kernel_count,
                     sizeof(*kernels)) ||
        !table_valid(size, header->argument_offset, header->argument_count,
                     sizeof(*arguments)) ||
        !table_valid(size, header->sync_offset, header->sync_count,
                     sizeof(*sync_resources)) ||
        !range_valid(size, header->string_offset, header->string_size) ||
        !header->string_size ||
        !range_valid(size, header->object_offset, header->object_size) ||
        !header->object_size ||
        header->isa_string_size != strlen(NPU_PACKAGE_ISA_RV64_DAYAN) ||
        !range_valid(size, header->isa_string_offset,
                     header->isa_string_size) ||
        memcmp((const unsigned char *)image + header->isa_string_offset,
               NPU_PACKAGE_ISA_RV64_DAYAN,
               header->isa_string_size) != 0)
        return -ENOEXEC;
    kernels = (const npu_package_kernel_t *)((const unsigned char *)image +
                                              header->kernel_offset);
    arguments = (const npu_package_argument_t *)((const unsigned char *)image +
                                                  header->argument_offset);
    sync_resources = (const npu_package_sync_t *)((const unsigned char *)image +
                                                   header->sync_offset);
    view->image = image;
    view->size = size;
    view->header = header;
    view->kernels = kernels;
    view->arguments = arguments;
    view->sync_resources = sync_resources;
    view->object = (const unsigned char *)image + header->object_offset;
    if (!object_valid(view->object, (size_t)header->object_size))
        goto invalid;
    for (index = 0; index < header->argument_count; ++index) {
        const npu_package_argument_t *argument = &arguments[index];

        if (argument->kind == NPU_PACKAGE_ARG_VALUE) {
            if ((argument->size != 1 && argument->size != 2 &&
                 argument->size != 4 && argument->size != 8) ||
                argument->access)
                goto invalid;
        } else if (argument->kind != NPU_PACKAGE_ARG_BUFFER ||
                   argument->size != 8 || !argument->alignment ||
                   (argument->alignment & (argument->alignment - 1U)) ||
                   !argument->access ||
                   (argument->access & ~(NPU_PACKAGE_ARG_READ |
                                         NPU_PACKAGE_ARG_WRITE))) {
            goto invalid;
        }
    }
    for (index = 0; index < header->sync_count; ++index) {
        const npu_package_sync_t *sync = &sync_resources[index];

        if (sync->type != NPU_PACKAGE_SYNC_SHARED_RESOURCE || !sync->count ||
            !sync->consumer_count || sync->flags || sync->reserved)
            goto invalid;
    }
    for (index = 0; index < header->kernel_count; ++index) {
        const npu_package_kernel_t *kernel = &kernels[index];

        if (kernel->version != NPU_PACKAGE_KERNEL_VERSION ||
            (kernel->flags & ~NPU_PACKAGE_KERNEL_FLAGS_ALL) ||
            ((kernel->flags & NPU_PACKAGE_KERNEL_LOCKED_TASK) &&
             kernel->supported_task_modes !=
                 NPU_PACKAGE_TASK_MODE_SINGLE_CORE) ||
            ((kernel->flags & NPU_PACKAGE_KERNEL_LOCKED_TASK) &&
             kernel->max_tasks_per_core != 1U) ||
            kernel->reserved0 || kernel->reserved1 ||
            !npu_package_string(view, kernel->name_offset) ||
            !npu_package_string(view, kernel->entry_offset) ||
            !kernel->supported_task_modes ||
            (kernel->supported_task_modes & ~NPU_PACKAGE_TASK_MODE_ALL) ||
            !kernel->max_tasks_per_core ||
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
            kernel->first_argument > header->argument_count ||
            kernel->argument_count >
                header->argument_count - kernel->first_argument ||
            kernel->first_sync > header->sync_count ||
            kernel->sync_count > header->sync_count - kernel->first_sync)
            goto invalid;
    }
    if ((header->flags & NPU_PACKAGE_IMMUTABLE_MODULE) &&
        !npu_package_object_is_immutable(view))
        goto invalid;
    return 0;

invalid:
    memset(view, 0, sizeof(*view));
    return -ENOEXEC;
}

const npu_package_kernel_t *npu_package_find_kernel(
    const struct npu_package_view *view, const char *name)
{
    size_t index;

    if (!view || !view->header || !name || !*name)
        return 0;
    for (index = 0; index < view->header->kernel_count; ++index) {
        const char *candidate = npu_package_string(
            view, view->kernels[index].name_offset);

        if (candidate && strcmp(candidate, name) == 0)
            return &view->kernels[index];
    }
    return 0;
}

int npu_package_object_is_immutable(const struct npu_package_view *view)
{
    Elf64_Ehdr header;
    size_t index;

    if (!view || !view->object || view->header->object_size < sizeof(header))
        return 0;
    memcpy(&header, view->object, sizeof(header));
    for (index = 0; index < header.e_shnum; ++index) {
        Elf64_Shdr section;

        memcpy(&section,
               view->object + header.e_shoff + index * sizeof(section),
               sizeof(section));
        if (!(section.sh_flags & SHF_ALLOC))
            continue;
        if ((section.sh_flags & SHF_WRITE) || section.sh_type == SHT_NOBITS)
            return 0;
    }
    return 1;
}

uint64_t npu_package_hash(const void *image, size_t size)
{
    const unsigned char *bytes = image;
    uint64_t hash = UINT64_C(1469598103934665603);
    size_t index;

    for (index = 0; index < size; ++index) {
        hash ^= bytes[index];
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}
