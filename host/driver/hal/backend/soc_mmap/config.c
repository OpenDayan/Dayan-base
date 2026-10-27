#include "npu_hal_soc_mmap.h"

#include <errno.h>
#include <string.h>

#include "layout.h"
#include "platform_regs.h"

int npu_hal_soc_mmap_options_init(npu_hal_soc_mmap_options_t *options)
{
    if (!options)
        return -EINVAL;
    memset(options, 0, sizeof(*options));
    options->base.backend = NPU_HAL_BACKEND_SOC_MMAP;
    options->base.version = NPU_HAL_SOC_MMAP_OPTIONS_VERSION;
    options->platform_control = NPU_HAL_SOC_MMAP_PLATFORM_EXTERNAL;
    options->memory_device = NPU_HAL_SOC_MMAP_DEFAULT_DEVICE;
    /* Where the firmware is linked; with the ClusterMMU on it may differ. */
    options->ddr_device_base = DAYAN_DRAM_BASE;
    return 0;
}

int npu_hal_soc_mmap_memory_regions(const npu_hal_soc_mmap_options_t *options,
                                    uint32_t cores_per_cluster,
                                    uint32_t uram_size,
                                    npu_hal_memory_region_t *regions,
                                    uint32_t capacity)
{
    struct soc_layout_entry entries[SOC_LAYOUT_MAX_ENTRIES];
    uint32_t count = 0;
    int entry_count;
    int index;

    if (!options || !regions || !options->ddr_size)
        return -EINVAL;
    entry_count = soc_layout_build(cores_per_cluster, uram_size, entries,
                                   SOC_LAYOUT_MAX_ENTRIES);
    if (entry_count < 0)
        return entry_count;
    if (capacity < 1U)
        return -E2BIG;
    regions[count++] = (npu_hal_memory_region_t){
        options->ddr_device_base, options->ddr_size, NPU_HAL_MEMORY_NORMAL,
        NPU_HAL_MEMORY_DEVICE, NPU_HAL_ANY_CLUSTER, NPU_HAL_ANY_CORE
    };
    for (index = 0; index < entry_count; ++index) {
        const struct soc_layout_entry *e = &entries[index];

        if (e->kind != SOC_LAYOUT_MEMORY)
            continue;
        if (count == capacity)
            return -E2BIG;
        /* One Cluster only, so Cluster 0 owns the per-Cluster windows. */
        regions[count++] = e->memory_class == NPU_HAL_MEMORY_URAM ?
            (npu_hal_memory_region_t){
                e->offset, e->size, NPU_HAL_MEMORY_URAM,
                NPU_HAL_MEMORY_CORE, 0, e->core } :
            (npu_hal_memory_region_t){
                e->offset, e->size, e->memory_class,
                NPU_HAL_MEMORY_DEVICE, NPU_HAL_ANY_CLUSTER,
                NPU_HAL_ANY_CORE };
    }
    return (int)count;
}
