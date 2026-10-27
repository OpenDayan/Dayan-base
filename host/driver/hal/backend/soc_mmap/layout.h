#ifndef NPU_HOST_HAL_SOC_MMAP_LAYOUT_H
#define NPU_HOST_HAL_SOC_MMAP_LAYOUT_H

/*
 * The Cluster IP aperture as the soc_mmap backend uses it: every module the
 * Host touches, with its access rule.  Offsets and fixed sizes come from
 * platform_regs.h; URAM size is selected by the Server configuration.
 * IP offsets anywhere else in the aperture (BootROM, the rest of CLINT,
 * reserved space) are not accessible.
 */

#include <stdint.h>

enum soc_layout_kind {
    /* Exactly access_width bytes, naturally aligned, one register per access. */
    SOC_LAYOUT_REGISTER = 1,
    /* Bulk copy with aligned accesses of up to 8 bytes. */
    SOC_LAYOUT_MEMORY = 2,
};

struct soc_layout_entry {
    const char *name;
    uint32_t kind;
    uint32_t access_width;   /* REGISTER only */
    uint64_t offset;         /* IP offset, which is also the NPU address */
    uint64_t size;
    uint32_t memory_class;   /* MEMORY only: NPU_HAL_MEMORY_COHERENT/URAM */
    uint32_t core;           /* URAM owner, NPU_HAL_ANY_CORE otherwise */
};

#define SOC_LAYOUT_MAX_ENTRIES 16U

/*
 * Fills entries for a Cluster with @cores Harts (1..DAYAN_CLUSTER_MAX_HARTS),
 * sorted by offset.  Returns the count, or a negative errno.
 */
int soc_layout_build(uint32_t cores, uint32_t uram_size,
                     struct soc_layout_entry *entries,
                     uint32_t capacity);

#endif
