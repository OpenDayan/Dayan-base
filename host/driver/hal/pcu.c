#include "hal_internal.h"

#include <errno.h>
#include <string.h>

#include "platform_regs.h"

static int pcu_read64(npu_hal_t *hal, uint32_t cluster, uint64_t address,
                       uint64_t *value)
{
    unsigned char bytes[8];
    unsigned int i;
    int status = npu_hal_copy_from_address(hal, cluster, bytes, address,
                                            sizeof(bytes));

    if (status < 0)
        return status;
    *value = 0;
    for (i = 0; i < sizeof(bytes); ++i)
        *value |= (uint64_t)bytes[i] << (i * 8U);
    return 0;
}

static int pcu_trigger(npu_hal_t *hal, uint32_t cluster, uint32_t cores)
{
    const unsigned char update[8] = { 1 };

    /* The AXI write response waits for every local Hold snapshot to settle.
     * PCU accepts only single-beat accesses, never a bulk counter read. */
    return npu_hal_copy_to_address(hal, cluster, DAYAN_PCU_CONTROL(cores),
                                   update, sizeof(update));
}

/* The snapshot control follows the last Core page of the PCU that was built,
 * so a Cluster built with fewer Cores than the HAL topology has it elsewhere:
 * the 1-core TestCluster at 0x31ff8, not the 4-core 0x37ff8. A trigger at the
 * wrong address latches nothing, and the counters keep whatever they held --
 * zero, or an earlier snapshot. So a layout is proven only when two triggers
 * move a cycle slot; step down from the topology until one does. A full-size
 * Cluster with a running Core is never written below its own control.
 * Returns the Core count, or 0 when no layout moves a cycle. */
static int pcu_detect_cores(npu_hal_t *hal, uint32_t cluster,
                            uint32_t max_cores)
{
    uint32_t cores;

    for (cores = max_cores; cores; --cores) {
        uint64_t first[NPU_PCU_MAX_CORES];
        uint32_t pass;
        uint32_t core;

        for (pass = 0; pass < 2; ++pass) {
            int status = pcu_trigger(hal, cluster, cores);

            if (status < 0)
                return status;
            for (core = 0; core < cores; ++core) {
                uint64_t cycle;

                status = pcu_read64(hal, cluster, DAYAN_PCU_CYCLE(core),
                                    &cycle);
                if (status < 0)
                    return status;
                if (!pass)
                    first[core] = cycle;
                else if (cycle != first[core])
                    return (int)cores;
            }
        }
    }
    return 0;
}

/* The layout is a property of the hardware behind one HAL, probed once per
 * Cluster; the caller already serializes snapshots. With no cycle counter
 * running nothing can be proven, so the topology stands and the next
 * snapshot probes again. */
static int pcu_cores(npu_hal_t *hal, uint32_t cluster, uint32_t max_cores)
{
    static const npu_hal_t *owner;
    static unsigned char known[NPU_HAL_MAX_CLUSTERS];
    int cores;

    if (owner != hal) {
        owner = hal;
        memset(known, 0, sizeof(known));
    }
    if (known[cluster] && known[cluster] <= max_cores)
        return known[cluster];
    cores = pcu_detect_cores(hal, cluster, max_cores);
    if (cores < 0)
        return cores;
    known[cluster] = (unsigned char)cores;
    return cores ? cores : (int)max_cores;
}

int npu_hal_pcu_snapshot(npu_hal_t *hal, uint32_t cluster,
                          npu_pcu_batch_t *batch)
{
    npu_hal_info_t info;
    uint32_t core;
    int have_cycle = 0;
    int status;

    if (!hal || !batch)
        return -EINVAL;
    status = npu_hal_get_info(hal, &info);
    if (status < 0)
        return status;
    if (cluster >= info.cluster_count || cluster >= NPU_HAL_MAX_CLUSTERS)
        return -EINVAL;
    if (!info.cores_per_cluster || info.cores_per_cluster > NPU_PCU_MAX_CORES)
        return -ENOTSUP;
    memset(batch, 0, sizeof(*batch));
    batch->cluster = cluster;
    status = pcu_cores(hal, cluster, info.cores_per_cluster);
    if (status < 0)
        return status;
    batch->core_count = (uint32_t)status;
    status = pcu_trigger(hal, cluster, batch->core_count);
    if (status < 0)
        return status;
    for (core = 0; core < batch->core_count; ++core) {
        uint32_t event;
        npu_pcu_core_sample_t *sample = &batch->cores[core];

        status = pcu_read64(hal, cluster, DAYAN_PCU_CYCLE(core),
                            &sample->cycle);
        if (status < 0)
            return status;
        have_cycle |= sample->cycle != 0;
        for (event = 0; event < NPU_PCU_COUNTER_COUNT; ++event) {
            status = pcu_read64(hal, cluster, DAYAN_PCU_COUNTER(core, event),
                                &sample->counters[event]);
            if (status < 0)
                return status;
        }
    }
    return have_cycle ? 0 : -ENODATA;
}
