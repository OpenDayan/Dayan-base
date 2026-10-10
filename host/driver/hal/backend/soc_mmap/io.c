#define _POSIX_C_SOURCE 200809L

#include "io.h"

#include <errno.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <unistd.h>

#include "npu_hal_soc_mmap.h"

/* A 64-bit register must be one access; a 32-bit Host would split it. */
#if UINTPTR_MAX != UINT64_MAX
#error "soc_mmap needs a 64-bit Host for single-beat 64-bit accesses"
#endif

int npu_hal_soc_mmap_map(int fd, uint64_t offset, uint64_t size,
                         struct npu_hal_soc_mmap_mapping *mapping)
{
    long page = sysconf(_SC_PAGESIZE);
    uint64_t delta;
    uint64_t length;
    void *base;

    if (!mapping || fd < 0 || !size || page <= 0 ||
        offset > UINT64_MAX - size)
        return -EINVAL;
    memset(mapping, 0, sizeof(*mapping));
    delta = offset & ((uint64_t)page - 1U);
    length = size + delta;
    if (length < size || length > SIZE_MAX ||
        offset - delta > (uint64_t)INT64_MAX)
        return -EOVERFLOW;
    base = mmap(0, (size_t)length, PROT_READ | PROT_WRITE, MAP_SHARED, fd,
                (off_t)(offset - delta));
    if (base == MAP_FAILED)
        return -errno;
    mapping->base = base;
    mapping->length = (size_t)length;
    mapping->window = (volatile unsigned char *)base + delta;
    mapping->size = size;
    return 0;
}

void npu_hal_soc_mmap_unmap(struct npu_hal_soc_mmap_mapping *mapping)
{
    if (!mapping || !mapping->base)
        return;
    (void)munmap(mapping->base, mapping->length);
    memset(mapping, 0, sizeof(*mapping));
}

void npu_hal_soc_mmap_io_fence(void)
{
#if defined(__riscv)
    __asm__ __volatile__("fence iorw, iorw" ::: "memory");
#else
    __atomic_thread_fence(__ATOMIC_SEQ_CST);
#endif
}

static int reg_access_valid(const volatile unsigned char *target,
                            uint32_t width)
{
    return target && (width == 1U || width == 2U || width == 4U ||
                      width == 8U) &&
           !((uintptr_t)target & (width - 1U));
}

int npu_hal_soc_mmap_reg_write(volatile unsigned char *target,
                               const void *value, uint32_t width)
{
    if (!value || !reg_access_valid(target, width))
        return -EINVAL;
    switch (width) {
    case 8: {
        uint64_t v;

        memcpy(&v, value, sizeof(v));
        *(volatile uint64_t *)(void *)target = v;
        break;
    }
    case 4: {
        uint32_t v;

        memcpy(&v, value, sizeof(v));
        *(volatile uint32_t *)(void *)target = v;
        break;
    }
    case 2: {
        uint16_t v;

        memcpy(&v, value, sizeof(v));
        *(volatile uint16_t *)(void *)target = v;
        break;
    }
    default:
        *target = *(const unsigned char *)value;
        break;
    }
    return 0;
}

int npu_hal_soc_mmap_reg_read(const volatile unsigned char *target,
                              void *value, uint32_t width)
{
    if (!value || !reg_access_valid(target, width))
        return -EINVAL;
    switch (width) {
    case 8: {
        uint64_t v = *(const volatile uint64_t *)(const void *)target;

        memcpy(value, &v, sizeof(v));
        break;
    }
    case 4: {
        uint32_t v = *(const volatile uint32_t *)(const void *)target;

        memcpy(value, &v, sizeof(v));
        break;
    }
    case 2: {
        uint16_t v = *(const volatile uint16_t *)(const void *)target;

        memcpy(value, &v, sizeof(v));
        break;
    }
    default:
        *(unsigned char *)value = *target;
        break;
    }
    return 0;
}

/* Widest power of two up to 8 that the address alignment and size allow. */
static uint32_t mem_access_size(uintptr_t address, size_t size)
{
    uint32_t chosen = 8U;

    while (chosen > 1U && (chosen > size || (address & (chosen - 1U))))
        chosen >>= 1;
    return chosen;
}

void npu_hal_soc_mmap_mem_write(volatile unsigned char *destination,
                                const void *source, size_t size)
{
    const unsigned char *cursor = source;

    while (size) {
        uint32_t chunk = mem_access_size((uintptr_t)destination, size);

        (void)npu_hal_soc_mmap_reg_write(destination, cursor, chunk);
        destination += chunk;
        cursor += chunk;
        size -= chunk;
    }
}

void npu_hal_soc_mmap_mem_read(void *destination,
                               const volatile unsigned char *source,
                               size_t size)
{
    unsigned char *cursor = destination;

    while (size) {
        uint32_t chunk = mem_access_size((uintptr_t)source, size);

        (void)npu_hal_soc_mmap_reg_read(source, cursor, chunk);
        source += chunk;
        cursor += chunk;
        size -= chunk;
    }
}

static int coherence_valid(uint32_t coherence)
{
    return coherence == NPU_HAL_SOC_MMAP_DDR_UNCACHED ||
           coherence == NPU_HAL_SOC_MMAP_DDR_COHERENT;
}

int npu_hal_soc_mmap_ddr_sync_for_device(uint32_t coherence,
                                         volatile unsigned char *address,
                                         size_t size)
{
    (void)address;
    (void)size;
    /* A cache-maintenance model would clean [address, address + size) here. */
    return coherence_valid(coherence) ? 0 : -ENOTSUP;
}

int npu_hal_soc_mmap_ddr_sync_for_cpu(uint32_t coherence,
                                      volatile unsigned char *address,
                                      size_t size)
{
    (void)address;
    (void)size;
    /* A cache-maintenance model would invalidate the range here. */
    return coherence_valid(coherence) ? 0 : -ENOTSUP;
}
