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

int npu_hal_pcu_snapshot(npu_hal_t *hal, uint32_t cluster,
                          npu_pcu_batch_t *batch)
{
    const unsigned char update[8] = { 1 };
    npu_hal_info_t info;
    uint32_t core;
    int have_cycle = 0;
    int status;

    if (!hal || !batch)
        return -EINVAL;
    status = npu_hal_get_info(hal, &info);
    if (status < 0)
        return status;
    if (cluster >= info.cluster_count)
        return -EINVAL;
    if (!info.cores_per_cluster || info.cores_per_cluster > NPU_PCU_MAX_CORES)
        return -ENOTSUP;
    memset(batch, 0, sizeof(*batch));
    batch->cluster = cluster;
    batch->core_count = info.cores_per_cluster;
    /* The AXI write response waits for every local Hold snapshot to settle.
     * PCU accepts only single-beat accesses, never a bulk counter read. */
    status = npu_hal_copy_to_address(hal, cluster,
                DAYAN_PCU_CONTROL(info.cores_per_cluster), update,
                sizeof(update));
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
