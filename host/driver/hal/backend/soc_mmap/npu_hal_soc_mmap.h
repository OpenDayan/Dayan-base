#ifndef NPU_HOST_HAL_SOC_MMAP_H
#define NPU_HOST_HAL_SOC_MMAP_H

/*
 * SoC backend: the Host CPU reaches one Cluster through /dev/mem (or a
 * restricted node with the same offsets).
 *
 * Four address spaces are involved and kept apart:
 *   - Host PA: what the memory device is mmap()ed at.
 *   - IP offset: a module's position in the Cluster IP aperture
 *     (platform_regs.h).  The NPU sees the aperture at CONFIG_CONTROL_BASE = 0,
 *     so an IP offset is also the NPU-side address of that module.
 *   - NPU-side PA of shared DDR: where the NPU master port sees it
 *     (ddr_device_base).  It need not equal the DDR Host PA.
 *   - ClusterMMU VA: what firmware and kernels use once the MMU is on.  The
 *     HAL never sees it; page tables translate VA to NPU-side PA.
 *
 * Host PA = npu_base + IP offset            for the IP aperture,
 * Host PA = ddr_host_base + (NPU PA - ddr_device_base)  for shared DDR.
 */

#include "npu_hal_config.h"

#ifdef __cplusplus
extern "C" {
#endif

#define NPU_HAL_SOC_MMAP_OPTIONS_VERSION 2U
#define NPU_HAL_SOC_MMAP_DEFAULT_DEVICE  "/dev/mem"

typedef enum {
    /*
     * Clock, reset, power and isolation are owned outside dayan_server
     * (bootloader, BSP or a platform driver) and are ready before it starts.
     * The backend touches none of them.  This is the only mode until the chip
     * integration defines the sequence.
     */
    NPU_HAL_SOC_MMAP_PLATFORM_EXTERNAL = 1,
} npu_hal_soc_mmap_platform_control_t;

/*
 * How Host and NPU agree on shared DDR contents.  There is no default: a
 * memory device does not tell user space which one holds, and neither
 * O_SYNC, volatile accesses nor fences provide either guarantee.
 */
typedef enum {
    /* The Host mapping of the DDR is non-cacheable and the NPU path does not
     * cache it either.  On RISC-V Linux, /dev/mem with O_SYNC goes through
     * pgprot_noncached(), which selects PBMT IO (non-cacheable,
     * non-idempotent, strongly ordered) - not PBMT NC, which is what
     * pgprot_writecombine() selects.  Either takes effect only with Svpbmt in
     * the CPU and CONFIG_RISCV_ISA_SVPBMT in the kernel; otherwise the PMA
     * alone decides, and DRAM PMA is normally cacheable.  Whether the result
     * really is uncached for this DDR has to be established per platform. */
    NPU_HAL_SOC_MMAP_DDR_UNCACHED = 1,
    /* The interconnect keeps NPU accesses coherent with Host caches. */
    NPU_HAL_SOC_MMAP_DDR_COHERENT = 2,
    /* Cache maintenance by a kernel helper is the third model; it has no
     * value until such an interface exists (see soc_ddr_sync_*()). */
} npu_hal_soc_mmap_ddr_coherence_t;

typedef struct {
    npu_hal_backend_options_t base;
    uint32_t platform_control;
    uint32_t ddr_coherence;
    /* Host PA of the Cluster IP aperture.  Page aligned. */
    uint64_t npu_base;
    /* Shared DDR reserved for the NPU: Host PA, size, and NPU-side PA. */
    uint64_t ddr_host_base;
    uint64_t ddr_size;
    uint64_t ddr_device_base;
    /* Device the Server maps from. */
    const char *memory_device;
    /* Device whose descriptor is passed to Runtime; NULL reuses memory_device.
     * It must expose the DDR at the same offsets.  Whoever holds it can map
     * every offset it allows: a token does not limit what it can reach. */
    const char *data_device;
} npu_hal_soc_mmap_options_t;

/* Sets the defaults; npu_base, the DDR Host PA and size, and the coherence
 * model have none and must be set by the caller. */
int npu_hal_soc_mmap_options_init(npu_hal_soc_mmap_options_t *options);

/*
 * Memory regions matching the fixed layout and the DDR configuration: Normal
 * over the whole shared DDR, the Coherent window, and one URAM per core.
 * Pass npu_hal_options_t::uram_size so allocation and access bounds agree.  The
 * backend refuses any other region set, so a Server passes these to
 * npu_hal_options_t::memory_regions.  Returns the region count, or -E2BIG when
 * capacity is too small.
 */
int npu_hal_soc_mmap_memory_regions(const npu_hal_soc_mmap_options_t *options,
                                    uint32_t cores_per_cluster,
                                    uint32_t uram_size,
                                    npu_hal_memory_region_t *regions,
                                    uint32_t capacity);
#define NPU_HAL_SOC_MMAP_MAX_REGIONS (2U + NPU_HAL_MAX_CORES_PER_CLUSTER)

#ifdef __cplusplus
}
#endif

#endif
