#include "service_internal.h"

#include <elf.h>
#include <errno.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "platform_regs.h"

#define ELF_ZERO_CHUNK   (64U * 1024U)
#define IRQ_ENABLE_OFFSET 0x04U
#define IRQ_ENABLE_CLEAR_OFFSET 0x14U
#define ATOMIC_CACHE_STATUS_OFFSET \
    (DAYAN_ATOMIC_CACHE_STATUS - DAYAN_IRQ_STATUS)
#define ATOMIC_CACHE_BACKING_OFFSET \
    (DAYAN_ATOMIC_CACHE_BACKING_BASE - DAYAN_IRQ_STATUS)
#define ATOMIC_CACHE_BUSY            (UINT64_C(1) << 0)
#define ATOMIC_CACHE_DONE            (UINT64_C(1) << 1)
#define ATOMIC_CACHE_WRITEBACK_ERROR (UINT64_C(1) << 2)
#define ATOMIC_CACHE_COMMAND_ERROR   (UINT64_C(1) << 3)
#define ATOMIC_CACHE_ERROR_MASK \
    (ATOMIC_CACHE_WRITEBACK_ERROR | ATOMIC_CACHE_COMMAND_ERROR)
#define ATOMIC_CACHE_BACKING_ALIGNMENT DAYAN_ATOMIC_CACHE_WINDOW_SIZE

struct load_segment {
    uint64_t address;
    uint64_t file_offset;
    uint64_t file_size;
    uint64_t memory_size;
    /*
     * Where the segment actually goes.  @address is what the ELF asks for and
     * what the Device will use, which under the MMU is a virtual address in
     * the firmware window; @physical is the memory behind it, and is what every
     * Host-side reservation and copy must use.  They are equal when the MMU is
     * off.
     */
    uint64_t physical;
    uint32_t flags;
};

static void firmware_log(uint32_t cluster, const char *format, ...)
{
    char message[512];
    va_list arguments;

    va_start(arguments, format);
    (void)vsnprintf(message, sizeof(message), format, arguments);
    va_end(arguments);
    fprintf(stderr, "dayan_server: [firmware C%u] %s\n", cluster,
            message);
    fflush(stderr);
}

static int segment_compare(const void *left, const void *right)
{
    const struct load_segment *a = left;
    const struct load_segment *b = right;

    return a->address < b->address ? -1 : a->address != b->address;
}

/*
 * An ELF program header carries firmware link addresses, which are virtual once
 * the MMU is on.  npu_hal_address_valid() answers about physical addresses, so
 * the link address has to be put through the firmware window first; checking it
 * raw only agrees with reality while the pool offset happens to be zero.
 */
static int segment_address_valid(npu_service_device_t *device,
                                 npu_hal_core_t viewer, uint64_t address,
                                 uint64_t size)
{
    uint64_t physical = 0;
    int status = npu_service_mmu_fw_pa(device, address, size, &physical);

    /* -ENOTSUP is the MMU-off case and has already set physical = address. */
    if (status < 0 && status != -ENOTSUP)
        return 0;
    return npu_hal_address_valid(device->hal, viewer, physical, size);
}

static int parse_segments(npu_service_device_t *device, npu_hal_core_t viewer,
                          const unsigned char *image, size_t size,
                          struct load_segment **segments_out,
                          size_t *count_out, uint64_t *entry_out)
{
    Elf64_Ehdr header;
    struct load_segment *segments;
    size_t count = 0;
    size_t index;
    int entry_executable = 0;

    if (size < sizeof(header))
        return -ENOEXEC;
    memcpy(&header, image, sizeof(header));
    if (memcmp(header.e_ident, ELFMAG, SELFMAG) != 0 ||
        header.e_ident[EI_CLASS] != ELFCLASS64 ||
        header.e_ident[EI_DATA] != ELFDATA2LSB ||
        header.e_ident[EI_VERSION] != EV_CURRENT ||
        header.e_type != ET_EXEC || header.e_machine != EM_RISCV ||
        header.e_version != EV_CURRENT || header.e_ehsize != sizeof(header) ||
        header.e_phentsize != sizeof(Elf64_Phdr) || !header.e_phnum ||
        header.e_phnum == PN_XNUM || header.e_phoff > size ||
        header.e_phnum > (size - header.e_phoff) / sizeof(Elf64_Phdr))
        return -ENOEXEC;
    segments = calloc(header.e_phnum, sizeof(*segments));
    if (!segments)
        return -ENOMEM;
    for (index = 0; index < header.e_phnum; ++index) {
        Elf64_Phdr program;
        uint64_t alignment;

        memcpy(&program,
               image + header.e_phoff + index * sizeof(program),
               sizeof(program));
        if (program.p_type != PT_LOAD)
            continue;
        alignment = program.p_align;
        if (!program.p_memsz || program.p_filesz > program.p_memsz ||
            program.p_offset > size ||
            program.p_filesz > size - program.p_offset ||
            program.p_vaddr != program.p_paddr ||
            !segment_address_valid(device, viewer, program.p_paddr,
                                   program.p_memsz) ||
            (alignment > 1U &&
             ((alignment & (alignment - 1U)) ||
              (program.p_paddr & (alignment - 1U)) !=
                  (program.p_offset & (alignment - 1U))))) {
            free(segments);
            return -ENOEXEC;
        }
        segments[count].address = program.p_paddr;
        segments[count].file_offset = program.p_offset;
        segments[count].file_size = program.p_filesz;
        segments[count].memory_size = program.p_memsz;
        segments[count].flags = program.p_flags;
        if ((program.p_flags & PF_X) && header.e_entry >= program.p_paddr &&
            header.e_entry < program.p_paddr + program.p_memsz)
            entry_executable = 1;
        ++count;
    }
    if (!count || !entry_executable ||
        (header.e_entry & ~RPC_COMMAND_ARG_MASK)) {
        free(segments);
        return -ENOEXEC;
    }
    qsort(segments, count, sizeof(*segments), segment_compare);
    for (index = 1; index < count; ++index) {
        if (segments[index].address <
            segments[index - 1].address +
                segments[index - 1].memory_size) {
            free(segments);
            return -ENOEXEC;
        }
    }
    *segments_out = segments;
    *count_out = count;
    *entry_out = header.e_entry;
    return 0;
}

static int region_contains(const npu_hal_memory_region_t *region,
                           uint64_t address, uint64_t size)
{
    return size <= region->size && address >= region->base &&
           address - region->base <= region->size - size;
}

static struct npu_service_firmware_allocation *find_global_segment(
    npu_service_device_t *device, uint64_t address, uint64_t size)
{
    struct npu_service_firmware_allocation *item;

    for (item = device->firmware_allocations; item; item = item->next) {
        if (item->scope == NPU_HAL_MEMORY_DEVICE &&
            item->address == address && item->size == size)
            return item;
    }
    return 0;
}

static int reserve_segment(npu_service_device_t *device, uint32_t cluster,
                           const struct load_segment *segment)
{
    npu_hal_memory_request_t request;
    npu_hal_allocation_t *allocation;
    struct npu_service_firmware_allocation *tracking;
    npu_hal_memory_region_t region;
    uint32_t pass;
    uint32_t index;
    int status = -ENOMEM;

    npu_hal_memory_request_init(&request);
    request.memory_class = NPU_HAL_MEMORY_NORMAL;
    request.usage = NPU_HAL_MEMORY_FIRMWARE;
    request.owner = UINT32_MAX;
    request.size = segment->memory_size;
    request.alignment = 1;
    request.fixed_address = segment->physical;
    for (pass = 0; pass < 2; ++pass) {
        uint32_t wanted_scope = pass == 0 ? NPU_HAL_MEMORY_CLUSTER :
                                            NPU_HAL_MEMORY_DEVICE;

        for (index = 0; index < device->hal_info.memory_region_count;
             ++index) {
            status = npu_hal_get_memory_region(device->hal, index, &region);
            if (status < 0)
                return status;
            if (region.memory_class != NPU_HAL_MEMORY_NORMAL ||
                region.scope != wanted_scope ||
                (wanted_scope == NPU_HAL_MEMORY_CLUSTER &&
                 region.cluster != cluster) ||
                !region_contains(&region, segment->physical,
                                 segment->memory_size))
                continue;
            if (wanted_scope == NPU_HAL_MEMORY_DEVICE &&
                find_global_segment(device, segment->physical,
                                    segment->memory_size))
                return 0;
            request.scope = wanted_scope;
            request.location.cluster =
                wanted_scope == NPU_HAL_MEMORY_DEVICE ?
                    NPU_HAL_ANY_CLUSTER : cluster;
            request.location.core = NPU_HAL_ANY_CORE;
            status = npu_hal_memory_reserve(device->hal, &request,
                                            &allocation);
            if (status < 0)
                return status;
            tracking = calloc(1, sizeof(*tracking));
            if (!tracking) {
                (void)npu_hal_memory_free(device->hal, allocation);
                return -ENOMEM;
            }
            tracking->allocation = allocation;
            tracking->scope = wanted_scope;
            tracking->cluster = wanted_scope == NPU_HAL_MEMORY_DEVICE ?
                NPU_HAL_ANY_CLUSTER : cluster;
            tracking->address = segment->physical;
            tracking->size = segment->memory_size;
            tracking->next = device->firmware_allocations;
            device->firmware_allocations = tracking;
            return 0;
        }
    }
    return status;
}

/*
 * Fill in @physical for every segment.  The firmware window is mapped in full
 * at init, so this is pure arithmetic - no allocation and no ordering
 * requirement against the mapping itself.
 */
static int resolve_segment_addresses(npu_service_device_t *device,
                                     struct load_segment *segments,
                                     size_t count)
{
    size_t index;

    for (index = 0; index < count; ++index) {
        uint64_t physical = 0;
        int status = npu_service_mmu_fw_pa(device, segments[index].address,
                                          segments[index].memory_size,
                                          &physical);

        /* -ENOTSUP is the MMU-off case and has already set physical = va. */
        if (status < 0 && status != -ENOTSUP)
            return status;
        segments[index].physical = physical;
    }
    return 0;
}

static void rollback_reservations(
    npu_service_device_t *device,
    struct npu_service_firmware_allocation *previous_head)
{
    while (device->firmware_allocations != previous_head) {
        struct npu_service_firmware_allocation *item =
            device->firmware_allocations;

        device->firmware_allocations = item->next;
        (void)npu_hal_memory_free(device->hal, item->allocation);
        free(item);
    }
}

static int load_segments(npu_service_device_t *device, uint32_t cluster,
                         const unsigned char *image,
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
            status = npu_hal_copy_to_address(
                device->hal, cluster, segments[index].physical,
                image + segments[index].file_offset,
                (size_t)segments[index].file_size);
        while (status == 0 && zeroed < segments[index].memory_size) {
            uint64_t left = segments[index].memory_size - zeroed;
            size_t chunk = left < ELF_ZERO_CHUNK ? (size_t)left :
                                                   ELF_ZERO_CHUNK;

            status = npu_hal_copy_to_address(
                device->hal, cluster, segments[index].physical + zeroed,
                zeros, chunk);
            zeroed += chunk;
        }
    }
    free(zeros);
    return status;
}

static int wait_runtime_ready(npu_service_device_t *device, uint32_t cluster,
                              uint32_t core_mask)
{
    uint64_t deadline = npu_now_ms() + device->options.default_timeout_ms;

    for (;;) {
        uint32_t enabled = 0;
        int status = npu_hal_register_read(device->hal, cluster,
                                           IRQ_ENABLE_OFFSET, &enabled,
                                           sizeof(enabled));

        if (status < 0)
            return status;
        if ((enabled & core_mask) == core_mask)
            return 0;
        if (npu_now_ms() >= deadline)
            return -ETIMEDOUT;
        npu_delay_us(device->options.rpc_poll_interval_us);
    }
}

static int runtime_ready_mask(npu_service_device_t *device, uint32_t cluster,
                              uint32_t *mask_out)
{
    return npu_hal_register_read(device->hal, cluster, IRQ_ENABLE_OFFSET,
                                 mask_out, sizeof(*mask_out));
}

static int halt_running_bl1(npu_service_device_t *device, uint32_t cluster,
                            uint32_t valid_mask, uint32_t ready_mask)
{
    uint32_t halt_mask = ready_mask & valid_mask;
    uint32_t remaining = 0;
    uint32_t core;
    int status;

    if (!halt_mask)
        return 0;
    firmware_log(cluster,
                 "running BL1 detected; parking core mask 0x%08x in bootrom",
                 halt_mask);
    for (core = 0; core < device->hal_info.cores_per_cluster; ++core) {
        npu_hal_core_t target = { cluster, core };

        if (!(halt_mask & (UINT32_C(1) << core)))
            continue;
        status = npu_hal_boot_core(device->hal, target,
                                   DAYAN_BOOTROM_RESET_VECTOR,
                                   device->options.default_timeout_ms);
        if (status < 0)
            return status;
        firmware_log(cluster, "Core %u: parked in bootrom", core);
    }
    status = runtime_ready_mask(device, cluster, &remaining);
    if (status < 0)
        return status;
    if (remaining & halt_mask)
        return -EIO;
    return 0;
}

static int wait_atomic_cache(npu_service_device_t *device, uint32_t cluster,
                             int require_done)
{
    uint64_t deadline = npu_now_ms() + device->options.default_timeout_ms;

    for (;;) {
        uint64_t status = 0;
        int result = npu_hal_register_read(
            device->hal, cluster, ATOMIC_CACHE_STATUS_OFFSET,
            &status, sizeof(status));

        if (result < 0)
            return result;
        if (status & ATOMIC_CACHE_ERROR_MASK)
            return -EIO;
        if (!(status & ATOMIC_CACHE_BUSY) &&
            (!require_done || (status & ATOMIC_CACHE_DONE)))
            return 0;
        if (npu_now_ms() >= deadline)
            return -ETIMEDOUT;
        npu_delay_us(device->options.rpc_poll_interval_us);
    }
}

static int allocate_atomic_cache_backing(
    npu_service_device_t *device, uint32_t cluster,
    npu_hal_allocation_t **allocation_out)
{
    static const npu_hal_memory_scope_t scopes[] = {
        NPU_HAL_MEMORY_CLUSTER,
        NPU_HAL_MEMORY_DEVICE,
    };
    npu_hal_memory_request_t request;
    size_t index;
    int status = -ENOMEM;

    npu_hal_memory_request_init(&request);
    request.memory_class = NPU_HAL_MEMORY_NORMAL;
    request.usage = NPU_HAL_MEMORY_ATOMIC_CACHE_BACKING;
    request.owner = UINT32_MAX;
    request.size = DAYAN_ATOMIC_CACHE_WINDOW_SIZE;
    request.alignment = ATOMIC_CACHE_BACKING_ALIGNMENT;
    for (index = 0; index < sizeof(scopes) / sizeof(scopes[0]); ++index) {
        request.scope = scopes[index];
        request.location.cluster = scopes[index] == NPU_HAL_MEMORY_CLUSTER ?
            cluster : NPU_HAL_ANY_CLUSTER;
        request.location.core = NPU_HAL_ANY_CORE;
        status = npu_hal_memory_alloc(device->hal, &request,
                                      allocation_out);
        if (status == 0 || status != -ENOMEM)
            return status;
    }
    return status;
}

static int configure_atomic_cache(npu_service_device_t *device,
                                  uint32_t cluster)
{
    struct npu_service_cluster *state = &device->clusters[cluster];
    npu_hal_allocation_t *allocation = 0;
    npu_hal_core_t viewer = { cluster, 0 };
    uint64_t backing_address = 0;
    int status;

    if (state->atomic_cache_backing)
        return -EBUSY;
    status = allocate_atomic_cache_backing(device, cluster, &allocation);
    if (status < 0)
        return status;
    status = npu_hal_memory_address(device->hal, allocation, viewer,
                                    &backing_address);
    if (status < 0)
        goto fail;
    status = wait_atomic_cache(device, cluster, 0);
    if (status < 0)
        goto fail;
    status = npu_hal_register_write(
        device->hal, cluster, ATOMIC_CACHE_BACKING_OFFSET,
        &backing_address, sizeof(backing_address));
    if (status < 0)
        goto fail;
    status = wait_atomic_cache(device, cluster, 1);
    if (status < 0)
        goto fail;
    state->atomic_cache_backing = allocation;
    state->atomic_cache_backing_address = backing_address;
    return 0;

fail:
    (void)npu_hal_memory_free(device->hal, allocation);
    return status;
}

static void release_atomic_cache_backing(npu_service_device_t *device,
                                         uint32_t cluster)
{
    struct npu_service_cluster *state = &device->clusters[cluster];

    if (state->atomic_cache_backing)
        (void)npu_hal_memory_free(device->hal,
                                  state->atomic_cache_backing);
    state->atomic_cache_backing = 0;
    state->atomic_cache_backing_address = 0;
}

/*
 * Scrub the Cores' D-caches over the firmware region before a new image goes
 * in (ported from npu-firmware 3d0adf4, dayan_firmware_scrub_dcache()).
 *
 * Neither the halt path that sends a Core back to BL0 nor BL0 itself writes
 * back or invalidates the D-cache, and the image is written by DMA, which the
 * Cores' caches do not see.  So a Core can come out of BL0 into a new image
 * still holding the previous firmware's lines over the same addresses: clean
 * ones it then reads instead of the new .data, dirty ones that are later
 * evicted on top of it.
 *
 * Only the Cores can reach their caches.  Each one is booted into a small stub
 * that issues cbo.flush (write back and invalidate) over the range and jumps
 * back to the boot ROM; a second boot request to the reset vector is consumed
 * only once BL0 is polling again, which is the completion.  The stub builds
 * its constants with lui/addi so it never reads through the cache it cleans,
 * and the range is flushed in two parts around the stub's own page.
 *
 * The stub runs at physical addresses, so this needs identity addressing:
 * with enable_mmu the Cluster MMU is already translating and the scrub is
 * skipped.
 */
#define FIRMWARE_SCRUB_SIZE       (UINT64_C(16) << 20)
#define FIRMWARE_SCRUB_PAGE       UINT64_C(4096)
#define FIRMWARE_SCRUB_STUB_WORDS 53U

#define RV_T1 6U
#define RV_T2 7U
#define RV_T3 28U

static uint32_t rv_lui(uint32_t rd, uint32_t imm20)
{
    return ((imm20 & 0xfffffU) << 12) | (rd << 7) | 0x37U;
}

static uint32_t rv_addi(uint32_t rd, uint32_t rs1, int32_t imm)
{
    return (((uint32_t)imm & 0xfffU) << 20) | (rs1 << 15) | (rd << 7) | 0x13U;
}

static uint32_t rv_shift(uint32_t funct3, uint32_t rd, uint32_t rs1,
                         uint32_t shamt)
{
    return (shamt << 20) | (rs1 << 15) | (funct3 << 12) | (rd << 7) | 0x13U;
}

/* rd = v, zero extended; 4 words. */
static uint32_t *rv_li32(uint32_t *p, uint32_t rd, uint32_t v)
{
    uint32_t hi = (uint32_t)(((uint64_t)v + 0x800U) >> 12) & 0xfffffU;

    *p++ = rv_lui(rd, hi);
    *p++ = rv_addi(rd, rd, (int32_t)(v - (hi << 12)));
    *p++ = rv_shift(1, rd, rd, 32);     /* slli */
    *p++ = rv_shift(5, rd, rd, 32);     /* srli */
    return p;
}

/* rd = v using tmp; 10 words. */
static uint32_t *rv_li64(uint32_t *p, uint32_t rd, uint32_t tmp, uint64_t v)
{
    p = rv_li32(p, rd, (uint32_t)(v >> 32));
    *p++ = rv_shift(1, rd, rd, 32);     /* slli */
    p = rv_li32(p, tmp, (uint32_t)v);
    *p++ = (tmp << 20) | (rd << 15) | (rd << 7) | 0x33U;  /* add */
    return p;
}

static uint32_t rv_bltu(uint32_t rs1, uint32_t rs2, int32_t offset)
{
    uint32_t i = (uint32_t)offset;

    return (((i >> 12) & 1U) << 31) | (((i >> 5) & 0x3fU) << 25) |
           (rs2 << 20) | (rs1 << 15) | (6U << 12) |
           (((i >> 1) & 0xfU) << 8) | (((i >> 11) & 1U) << 7) | 0x63U;
}

/* cbo.flush every line of [start, end), end > start; 23 words. */
static uint32_t *rv_flush_range(uint32_t *p, uint64_t start, uint64_t end)
{
    p = rv_li64(p, RV_T1, RV_T3, start);
    p = rv_li64(p, RV_T2, RV_T3, end);
    /* loop: cbo.flush 0(t1), the encoding platform_cache_flush() uses. */
    *p++ = (0x002U << 20) | (RV_T1 << 15) | (2U << 12) | 0x0fU;
    *p++ = rv_addi(RV_T1, RV_T1, 64);
    *p++ = rv_bltu(RV_T1, RV_T2, -8);
    return p;
}

static void build_scrub_stub(uint32_t *code, uint64_t start, uint64_t stub,
                             uint64_t end)
{
    uint64_t a0 = start, a1 = end, b0 = start, b1 = start + 64U;
    uint32_t *p = code;

    if (stub >= start && stub < end) {
        a1 = stub;
        b0 = stub + FIRMWARE_SCRUB_PAGE;
        b1 = end;
        if (a1 <= a0) {         /* stub at the very start */
            a0 = b0;
            a1 = b1;
        }
        if (b1 <= b0) {         /* stub at the very end */
            b0 = a0;
            b1 = a1;
        }
    }
    p = rv_flush_range(p, a0, a1);
    p = rv_flush_range(p, b0, b1);
    *p++ = 0x0ff0000fU;         /* fence iorw, iorw */
    *p++ = 0x0000100fU;         /* fence.i */
    p = rv_li32(p, RV_T1, DAYAN_BOOTROM_RESET_VECTOR);
    *p++ = (RV_T1 << 15) | 0x67U;   /* jr t1 */
}

static int scrub_firmware_dcache(npu_service_device_t *device,
                                 uint32_t cluster, uint32_t core_mask,
                                 const struct load_segment *segments,
                                 size_t count)
{
    npu_hal_memory_request_t request;
    npu_hal_allocation_t *allocation = 0;
    npu_hal_core_t viewer = { cluster, 0 };
    uint32_t code[FIRMWARE_SCRUB_STUB_WORDS];
    unsigned char wire[sizeof(code)];
    uint64_t start = UINT64_MAX, last = 0, end, stub = 0;
    uint32_t core;
    size_t index;
    int status;

    if (device->options.enable_mmu) {
        firmware_log(cluster,
                     "MMU enabled; D-cache scrub skipped (needs identity addressing)");
        return 0;
    }
    for (index = 0; index < count; ++index) {
        if (segments[index].physical < start)
            start = segments[index].physical;
        if (segments[index].physical + segments[index].memory_size > last)
            last = segments[index].physical + segments[index].memory_size;
    }
    if (start == UINT64_MAX)
        return 0;
    start &= ~UINT64_C(63);
    end = start + FIRMWARE_SCRUB_SIZE;
    if (last > end)
        end = (last + 63U) & ~UINT64_C(63);

    npu_hal_memory_request_init(&request);
    request.memory_class = NPU_HAL_MEMORY_NORMAL;
    request.usage = NPU_HAL_MEMORY_BUFFER;
    request.owner = UINT32_MAX;
    request.size = FIRMWARE_SCRUB_PAGE;
    request.alignment = FIRMWARE_SCRUB_PAGE;
    request.location.core = NPU_HAL_ANY_CORE;
    /* Same fallback as the Atomic Cache backing: Cluster, then Device. */
    request.scope = NPU_HAL_MEMORY_CLUSTER;
    request.location.cluster = cluster;
    status = npu_hal_memory_alloc(device->hal, &request, &allocation);
    if (status == -ENOMEM) {
        request.scope = NPU_HAL_MEMORY_DEVICE;
        request.location.cluster = NPU_HAL_ANY_CLUSTER;
        status = npu_hal_memory_alloc(device->hal, &request, &allocation);
    }
    if (status < 0) {
        firmware_log(cluster,
                     "no page for the D-cache scrub (%d); loading without it",
                     status);
        return 0;
    }
    status = npu_hal_memory_address(device->hal, allocation, viewer, &stub);
    if (status < 0)
        goto out;

    build_scrub_stub(code, start, stub, end);
    for (index = 0; index < FIRMWARE_SCRUB_STUB_WORDS; ++index) {
        wire[4 * index + 0] = (unsigned char)(code[index]);
        wire[4 * index + 1] = (unsigned char)(code[index] >> 8);
        wire[4 * index + 2] = (unsigned char)(code[index] >> 16);
        wire[4 * index + 3] = (unsigned char)(code[index] >> 24);
    }
    status = npu_hal_memory_write(device->hal, allocation, viewer, 0, wire,
                                  sizeof(wire));
    if (status < 0)
        goto out;

    /* Start every Core first so the scrubs run side by side. */
    for (core = 0; core < device->hal_info.cores_per_cluster; ++core) {
        npu_hal_core_t target = { cluster, core };

        if (!(core_mask & (UINT32_C(1) << core)))
            continue;
        status = npu_hal_boot_core(device->hal, target, stub,
                                   device->options.default_timeout_ms);
        if (status < 0) {
            firmware_log(cluster, "Core %u did not take the D-cache scrub (%d)",
                         core, status);
            goto out;
        }
    }
    /* Consumed only once the stub has jumped back and BL0 polls again. */
    for (core = 0; core < device->hal_info.cores_per_cluster; ++core) {
        npu_hal_core_t target = { cluster, core };

        if (!(core_mask & (UINT32_C(1) << core)))
            continue;
        status = npu_hal_boot_core(device->hal, target,
                                   DAYAN_BOOTROM_RESET_VECTOR,
                                   device->options.default_timeout_ms);
        if (status < 0) {
            firmware_log(cluster,
                         "Core %u did not return from the D-cache scrub (%d)",
                         core, status);
            goto out;
        }
    }
    firmware_log(cluster, "D-cache scrubbed over 0x%llx+0x%llx",
                 (unsigned long long)start,
                 (unsigned long long)(end - start));
out:
    (void)npu_hal_memory_free(device->hal, allocation);
    return status;
}

int npu_service_load_bl1_elf(npu_service_device_t *device, uint32_t cluster,
                             const void *image_data, size_t size,
                             uint32_t core_mask)
{
    const unsigned char *image = image_data;
    struct npu_service_firmware_allocation *previous_head;
    struct load_segment *segments = 0;
    npu_hal_core_t viewer = { cluster, 0 };
    size_t segment_count = 0;
    uint64_t entry = 0;
    uint32_t ready_mask = 0;
    uint32_t valid_mask;
    uint32_t core;
    const char *stage = "validating the BL1 request";
    int status;

    if (!device || !image || cluster >= device->hal_info.cluster_count)
        return -EINVAL;
    valid_mask = device->hal_info.cores_per_cluster == 32U ? UINT32_MAX :
        (UINT32_C(1) << device->hal_info.cores_per_cluster) - 1U;
    if (!core_mask || (core_mask & ~valid_mask))
        return -EINVAL;
    firmware_log(cluster, "loading BL1 ELF: image=%zu bytes core_mask=0x%08x",
                 size, core_mask);
    stage = "checking the Cluster firmware state";
    pthread_mutex_lock(&device->mutex);
    if (device->clusters[cluster].firmware_loaded ||
        device->clusters[cluster].booted_cores) {
        pthread_mutex_unlock(&device->mutex);
        return -EBUSY;
    }
    pthread_mutex_unlock(&device->mutex);
    stage = "parsing the BL1 ELF";
    firmware_log(cluster, "parsing BL1 ELF");
    status = parse_segments(device, viewer, image, size, &segments,
                            &segment_count, &entry);
    if (status < 0)
        goto out;
    firmware_log(cluster, "ELF parsed: segments=%zu entry=0x%llx",
                 segment_count, (unsigned long long)entry);
    stage = "checking for a running BL1 instance";
    status = runtime_ready_mask(device, cluster, &ready_mask);
    if (status < 0)
        goto out;
    stage = "parking the running BL1 in bootrom";
    status = halt_running_bl1(device, cluster, valid_mask, ready_mask);
    if (status < 0)
        goto out;
    stage = "resolving BL1 segment physical addresses";
    status = resolve_segment_addresses(device, segments, segment_count);
    if (status < 0)
        goto out;
    stage = "scrubbing the Cores' D-caches";
    status = scrub_firmware_dcache(device, cluster, valid_mask, segments,
                                   segment_count);
    if (status < 0)
        goto out;
    previous_head = device->firmware_allocations;
    stage = "reserving BL1 memory segments";
    firmware_log(cluster, "reserving %zu BL1 memory segment(s)",
                 segment_count);
    for (core = 0; core < segment_count; ++core) {
        status = reserve_segment(device, cluster, &segments[core]);
        if (status < 0) {
            rollback_reservations(device, previous_head);
            goto out;
        }
    }
    firmware_log(cluster, "BL1 memory segments reserved");
    stage = "reserving kernel text arena";
    status = npu_service_kernel_text_init(device, cluster);
    if (status < 0) {
        rollback_reservations(device, previous_head);
        goto out;
    }
    stage = "configuring Atomic Cache backing";
    firmware_log(cluster,
                 "configuring Atomic Cache backing (alignment=%llu bytes)",
                 (unsigned long long)ATOMIC_CACHE_BACKING_ALIGNMENT);
    status = configure_atomic_cache(device, cluster);
    if (status < 0) {
        rollback_reservations(device, previous_head);
        goto out;
    }
    firmware_log(cluster,
                 "Atomic Cache backing configured: address=0x%llx size=%llu bytes",
                 (unsigned long long)device->clusters[cluster]
                     .atomic_cache_backing_address,
                 (unsigned long long)DAYAN_ATOMIC_CACHE_WINDOW_SIZE);
    stage = "copying BL1 segments to Device memory";
    firmware_log(cluster, "copying BL1 segments to Device memory");
    status = load_segments(device, cluster, image, segments, segment_count);
    if (status < 0) {
        release_atomic_cache_backing(device, cluster);
        rollback_reservations(device, previous_head);
        goto out;
    }
    firmware_log(cluster, "BL1 segments copied");
    stage = "clearing stale BL1 readiness";
    status = npu_hal_register_write(device->hal, cluster,
                                    IRQ_ENABLE_CLEAR_OFFSET, &core_mask,
                                    sizeof(core_mask));
    if (status < 0)
        goto out;
    for (core = 0; core < device->hal_info.cores_per_cluster; ++core) {
        npu_hal_core_t target = { cluster, core };

        if (!(core_mask & (UINT32_C(1) << core)))
            continue;
        stage = "booting a BL1 Core";
        firmware_log(cluster, "Core %u: issuing BL1 boot at entry 0x%llx",
                     core, (unsigned long long)entry);
        status = npu_hal_boot_core(device->hal, target, entry,
                                   device->options.default_timeout_ms);
        if (status < 0)
            goto out;
        firmware_log(cluster, "Core %u: boot request completed", core);
    }
    stage = "waiting for BL1 ready";
    firmware_log(cluster,
                 "waiting for BL1 ready: core_mask=0x%08x timeout=%u ms",
                 core_mask, device->options.default_timeout_ms);
    status = wait_runtime_ready(device, cluster, core_mask);
    if (status < 0)
        goto out;
    pthread_mutex_lock(&device->mutex);
    device->clusters[cluster].firmware_entry = entry;
    device->clusters[cluster].booted_cores = core_mask;
    device->clusters[cluster].firmware_loaded = 1;
    pthread_mutex_unlock(&device->mutex);
    firmware_log(cluster, "BL1 ready: core_mask=0x%08x", core_mask);

out:
    if (status < 0)
        firmware_log(cluster, "%s failed: %s (%d)", stage,
                     strerror(-status), status);
    free(segments);
    return status;
}
