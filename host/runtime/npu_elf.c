#include "runtime_internal.h"

#include <elf.h>
#include <errno.h>
#include <stdlib.h>
#include <string.h>

#define ELF_ZERO_CHUNK (64U * 1024U)
#define IRQ_ENABLE_OFFSET 0x04U

struct load_segment {
    uint64_t address;
    uint64_t file_offset;
    uint64_t file_size;
    uint64_t memory_size;
    uint32_t flags;
};

static int segment_compare(const void *left, const void *right)
{
    const struct load_segment *a = left;
    const struct load_segment *b = right;

    return a->address < b->address ? -1 : a->address != b->address;
}

static int parse_segments(npu_runtime_t *runtime, const unsigned char *image,
                          size_t size, struct load_segment **segments_out,
                          size_t *count_out, uint64_t *entry_out)
{
    const Elf64_Ehdr *header;
    struct load_segment *segments;
    uint64_t ram_limit;
    size_t count = 0;
    size_t index;
    int entry_executable = 0;

    if (size < sizeof(Elf64_Ehdr))
        return -ENOEXEC;
    header = (const Elf64_Ehdr *)image;
    if (memcmp(header->e_ident, ELFMAG, SELFMAG) != 0 ||
        header->e_ident[EI_CLASS] != ELFCLASS64 ||
        header->e_ident[EI_DATA] != ELFDATA2LSB ||
        header->e_ident[EI_VERSION] != EV_CURRENT ||
        header->e_type != ET_EXEC || header->e_machine != EM_RISCV ||
        header->e_version != EV_CURRENT ||
        header->e_ehsize != sizeof(*header) ||
        header->e_phentsize != sizeof(Elf64_Phdr) || !header->e_phnum ||
        header->e_phnum == PN_XNUM || header->e_phoff > size ||
        header->e_phnum >
            (size - header->e_phoff) / sizeof(Elf64_Phdr))
        return -ENOEXEC;
    segments = calloc(header->e_phnum, sizeof(*segments));
    if (!segments)
        return -ENOMEM;
    ram_limit = runtime->options.ram_base + runtime->options.ram_size;
    for (index = 0; index < header->e_phnum; ++index) {
        const Elf64_Phdr *program = (const Elf64_Phdr *)(
            image + header->e_phoff + index * sizeof(*program));
        uint64_t alignment = program->p_align;

        if (program->p_type != PT_LOAD)
            continue;
        if (!program->p_memsz || program->p_filesz > program->p_memsz ||
            program->p_offset > size ||
            program->p_filesz > size - program->p_offset ||
            program->p_vaddr != program->p_paddr ||
            program->p_paddr < runtime->options.ram_base ||
            program->p_paddr >= ram_limit ||
            program->p_memsz > ram_limit - program->p_paddr ||
            (alignment > 1U &&
             ((alignment & (alignment - 1U)) ||
              (program->p_paddr & (alignment - 1U)) !=
                  (program->p_offset & (alignment - 1U))))) {
            free(segments);
            return -ENOEXEC;
        }
        segments[count].address = program->p_paddr;
        segments[count].file_offset = program->p_offset;
        segments[count].file_size = program->p_filesz;
        segments[count].memory_size = program->p_memsz;
        segments[count].flags = program->p_flags;
        if ((program->p_flags & PF_X) &&
            header->e_entry >= program->p_paddr &&
            header->e_entry < program->p_paddr + program->p_memsz)
            entry_executable = 1;
        ++count;
    }
    if (!count || !entry_executable ||
        (header->e_entry & ~RPC_COMMAND_ARG_MASK)) {
        free(segments);
        return -ENOEXEC;
    }
    qsort(segments, count, sizeof(*segments), segment_compare);
    for (index = 1; index < count; ++index) {
        if (segments[index].address <
            segments[index - 1].address + segments[index - 1].memory_size) {
            free(segments);
            return -ENOEXEC;
        }
    }
    *segments_out = segments;
    *count_out = count;
    *entry_out = header->e_entry;
    return 0;
}

static void release_segments(npu_runtime_t *runtime,
                             const struct load_segment *segments,
                             size_t reserved)
{
    pthread_mutex_lock(&runtime->mutex);
    while (reserved) {
        --reserved;
        (void)npu_arena_free(&runtime->normal, segments[reserved].address,
                             NPU_ALLOC_FIRMWARE, UINT32_MAX);
    }
    pthread_mutex_unlock(&runtime->mutex);
}

static int reserve_segments(npu_runtime_t *runtime,
                            const struct load_segment *segments,
                            size_t count, size_t *reserved_out)
{
    size_t index;
    int status = 0;

    *reserved_out = 0;
    pthread_mutex_lock(&runtime->mutex);
    if (runtime->firmware_loaded || runtime->booted_harts ||
        runtime->active_launches) {
        pthread_mutex_unlock(&runtime->mutex);
        return -EBUSY;
    }
    for (index = 0; index < count; ++index) {
        status = npu_arena_reserve(&runtime->normal,
                                   segments[index].address,
                                   segments[index].memory_size,
                                   NPU_ALLOC_FIRMWARE, UINT32_MAX);
        if (status < 0)
            break;
        ++*reserved_out;
    }
    pthread_mutex_unlock(&runtime->mutex);
    if (status < 0)
        release_segments(runtime, segments, *reserved_out);
    return status;
}

static int load_segments(npu_runtime_t *runtime, const unsigned char *image,
                         const struct load_segment *segments, size_t count)
{
    unsigned char *zeros = calloc(1, ELF_ZERO_CHUNK);
    size_t index;
    int status = 0;

    if (!zeros)
        return -ENOMEM;
    for (index = 0; index < count && status == 0; ++index) {
        uint64_t zeroed = segments[index].file_size;

        if (segments[index].file_size)
            status = npu_device_write(
                &runtime->device, segments[index].address,
                image + segments[index].file_offset,
                (size_t)segments[index].file_size);
        while (status == 0 && zeroed < segments[index].memory_size) {
            uint64_t left = segments[index].memory_size - zeroed;
            size_t chunk = left < ELF_ZERO_CHUNK ?
                (size_t)left : ELF_ZERO_CHUNK;

            status = npu_device_write(&runtime->device,
                                      segments[index].address + zeroed,
                                      zeros, chunk);
            zeroed += chunk;
        }
    }
    free(zeros);
    return status;
}

static int wait_runtime_ready(npu_runtime_t *runtime, uint32_t hart_mask)
{
    uint64_t deadline = npu_now_ms() + runtime->options.default_timeout_ms;

    for (;;) {
        uint32_t enabled = 0;
        int status = npu_device_read_control(&runtime->device,
                                             IRQ_ENABLE_OFFSET,
                                             &enabled, sizeof(enabled));

        if (status < 0)
            return status;
        if ((enabled & hart_mask) == hart_mask)
            return 0;
        if (npu_now_ms() >= deadline)
            return -ETIMEDOUT;
        npu_delay_us(runtime->options.rpc_poll_interval_us);
    }
}

int npu_runtime_load_bl1_elf(npu_runtime_t *runtime,
                             const void *image_data, size_t size,
                             uint32_t hart_mask)
{
    const unsigned char *image = image_data;
    struct load_segment *segments = 0;
    size_t segment_count = 0;
    size_t reserved = 0;
    uint64_t entry = 0;
    uint32_t valid_mask;
    int status;

    valid_mask = runtime->options.hart_count == 32U ? UINT32_MAX :
                 (1U << runtime->options.hart_count) - 1U;
    if (!hart_mask || (hart_mask & ~valid_mask))
        return -EINVAL;
    status = parse_segments(runtime, image, size, &segments,
                            &segment_count, &entry);
    if (status < 0)
        return status;
    status = reserve_segments(runtime, segments, segment_count, &reserved);
    if (status < 0)
        goto out;
    status = load_segments(runtime, image, segments, segment_count);
    if (status < 0) {
        release_segments(runtime, segments, reserved);
        goto out;
    }
    status = npu_device_boot(&runtime->device, hart_mask,
                             runtime->options.hart_count, entry,
                             runtime->options.default_timeout_ms);
    if (status < 0)
        goto out;
    status = wait_runtime_ready(runtime, hart_mask);
    if (status < 0)
        goto out;
    pthread_mutex_lock(&runtime->mutex);
    runtime->firmware_entry = entry;
    runtime->booted_harts = hart_mask;
    runtime->firmware_loaded = 1;
    pthread_mutex_unlock(&runtime->mutex);
out:
    free(segments);
    return status;
}
