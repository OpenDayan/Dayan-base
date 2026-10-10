#ifndef NPU_HOST_HAL_SOC_MMAP_IO_H
#define NPU_HOST_HAL_SOC_MMAP_IO_H

/* Shared by the Server backend and the Runtime data plane. */

#include <stddef.h>
#include <stdint.h>

/*
 * Export: one descriptor; opaque[0] = layout, opaque[1] = DDR coherence
 * model (npu_hal_soc_mmap_ddr_coherence_t), opaque[2] = DDR Host PA (offset
 * in that descriptor), opaque[3] = DDR size.  This is the only range a
 * Runtime will transfer to; it is fixed at import.
 * Token:  opaque[0] = DDR Host PA, opaque[1] = DDR size (both must repeat
 *         the export), opaque[2] = transfer offset in the DDR,
 *         opaque[3] = zero.
 * Layout 3: layout 2 let the first token choose the range.
 */
#define NPU_HAL_SOC_MMAP_TOKEN_LAYOUT 3U

struct npu_hal_soc_mmap_mapping {
    void *base;              /* What mmap() returned; page aligned. */
    size_t length;           /* What mmap() was given. */
    volatile unsigned char *window; /* base plus the in-page offset. */
    uint64_t size;
};

/* Maps [offset, offset + size) of fd read-write and shared. */
int npu_hal_soc_mmap_map(int fd, uint64_t offset, uint64_t size,
                         struct npu_hal_soc_mmap_mapping *mapping);
void npu_hal_soc_mmap_unmap(struct npu_hal_soc_mmap_mapping *mapping);

/*
 * Orders every earlier Host memory and device access before every later one.
 * On RISC-V this is fence iorw,iorw; a plain C fence does not order device I/O.
 * It is not cache maintenance and makes nothing visible that a cache holds.
 */
void npu_hal_soc_mmap_io_fence(void);

/*
 * One register access of exactly width bytes (1, 2, 4 or 8) at a naturally
 * aligned address; -EINVAL otherwise, with nothing accessed.  No fences.
 */
int npu_hal_soc_mmap_reg_write(volatile unsigned char *target,
                               const void *value, uint32_t width);
int npu_hal_soc_mmap_reg_read(const volatile unsigned char *target,
                              void *value, uint32_t width);

/*
 * Memory copy: naturally aligned volatile accesses of at most 8 bytes, lowest
 * address first, narrower only at unaligned edges.  For memory, never for
 * registers.  No fences.
 */
void npu_hal_soc_mmap_mem_write(volatile unsigned char *destination,
                                const void *source, size_t size);
void npu_hal_soc_mmap_mem_read(void *destination,
                               const volatile unsigned char *source,
                               size_t size);

/*
 * Shared-DDR ownership hand-over, used identically by Server and Runtime:
 * for_device after the Host wrote and before the NPU may read, for_cpu after
 * the NPU wrote and before the Host reads.  Both supported coherence models
 * need no maintenance, so these only validate the model; this is where a
 * kernel cache-maintenance helper would be called.
 */
int npu_hal_soc_mmap_ddr_sync_for_device(uint32_t coherence,
                                         volatile unsigned char *address,
                                         size_t size);
int npu_hal_soc_mmap_ddr_sync_for_cpu(uint32_t coherence,
                                      volatile unsigned char *address,
                                      size_t size);

#endif
