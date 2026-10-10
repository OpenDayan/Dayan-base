#include <assert.h>
#include <errno.h>
#include <string.h>

#include "hal.h"
#include "platform_regs.h"

/* A PCU built for `built` Cores behind a HAL whose topology says `cores`.
 * Only a trigger at the built control address latches; the slots then hold
 * that snapshot until the next one, and everything else reads zero. */
struct npu_hal {
    unsigned int cores;
    unsigned int built;
    unsigned int reads;
    unsigned int writes;
    unsigned int latches;
    unsigned int stray_writes;
    unsigned int fail_read;
    int frozen;
    int zero_cycle;
    int fail_write;
};

int npu_hal_get_info(npu_hal_t *hal, npu_hal_info_t *info)
{
    memset(info, 0, sizeof(*info));
    info->cluster_count = 2;
    info->cores_per_cluster = hal->cores;
    return 0;
}

int npu_hal_copy_to_address(npu_hal_t *hal, uint32_t cluster,
                            uint64_t address, const void *source, size_t size)
{
    static const unsigned char trigger[8] = { 1 };
    unsigned int layout;

    assert(cluster == 1);
    assert(size == 8 && !memcmp(source, trigger, 8));
    for (layout = 1; layout <= hal->cores; ++layout)
        if (address == 0x30000U + layout * 0x2000U - 8U)
            break;
    assert(layout <= hal->cores);
    ++hal->writes;
    if (hal->fail_write)
        return -EIO;
    if (layout == hal->built)
        ++hal->latches;
    else
        ++hal->stray_writes;
    return 0;
}

int npu_hal_copy_from_address(npu_hal_t *hal, uint32_t cluster,
                              void *destination, uint64_t address, size_t size)
{
    unsigned int core = (unsigned int)((address - 0x30000U) / 0x2000U);
    unsigned int slot = (unsigned int)((address - 0x30000U) % 0x2000U / 8U);
    uint64_t value = 0;
    unsigned char *bytes = destination;
    unsigned int i;

    assert(hal->writes && size == 8 && cluster == 1);
    assert(address >= 0x30000U && !(address % 8U));
    assert(core < hal->cores && (slot < 112U || slot == 1022U));
    if (++hal->reads == hal->fail_read)
        return -EIO;
    if (core < hal->built && hal->latches) {
        if (slot == 1022U)
            value = hal->zero_cycle ? 0 :
                    1000U * (hal->frozen ? 1U : hal->latches);
        else
            value = UINT64_C(0xfedcba9876000000) + core * 1024U + slot;
    }
    for (i = 0; i < 8; ++i)
        bytes[i] = (unsigned char)(value >> (i * 8U));
    return 0;
}

static void recount(struct npu_hal *hal)
{
    hal->reads = 0;
    hal->writes = 0;
    hal->stray_writes = 0;
}

int main(void)
{
    struct npu_hal hal = { .cores = 4, .built = 4 };
    struct npu_hal small = { .cores = 4, .built = 1 };
    struct npu_hal stale = { .cores = 4, .built = 1, .latches = 7 };
    struct npu_hal idle = { .cores = 4, .built = 4, .zero_cycle = 1 };
    struct npu_hal frozen = { .cores = 4, .built = 4, .frozen = 1 };
    struct npu_hal single = { .cores = 1, .built = 1 };
    npu_pcu_batch_t batch;
    unsigned int core;

    /* Full-size Cluster: the probe confirms the topology at once (two
     * triggers, the second stopping at the first moved cycle), then one
     * trigger and 4 * 113 reads per snapshot, as before. */
    assert(npu_hal_pcu_snapshot(&hal, 1, &batch) == 0);
    assert(hal.writes == 3 && !hal.stray_writes);
    assert(hal.reads == 4 + 1 + 4 * 113);
    assert(batch.cluster == 1 && batch.core_count == 4 && !batch.flags);
    for (core = 0; core < 4; ++core) {
        assert(batch.cores[core].cycle == 3000);
        assert(batch.cores[core].counters[109] ==
               UINT64_C(0xfedcba987600006d) + core * 1024U);
    }
    recount(&hal);
    assert(npu_hal_pcu_snapshot(&hal, 1, &batch) == 0);
    assert(hal.writes == 1 && hal.reads == 4 * 113);
    assert(batch.cores[3].cycle == 4000);
    recount(&hal);
    hal.fail_read = 114;
    assert(npu_hal_pcu_snapshot(&hal, 1, &batch) == -EIO);
    assert(hal.reads == 114);
    hal.fail_read = 0;
    recount(&hal);
    hal.fail_write = 1;
    assert(npu_hal_pcu_snapshot(&hal, 1, &batch) == -EIO);
    assert(!hal.reads);
    hal.fail_write = 0;

    /* 1-core PCU under the 4-core topology: layouts 4, 3 and 2 latch
     * nothing, layout 1 does; later snapshots go straight to it. */
    assert(npu_hal_pcu_snapshot(&small, 1, &batch) == 0);
    assert(small.stray_writes == 6 && small.writes == 6 + 2 + 1);
    assert(small.reads == 2 * (4 + 3 + 2) + 2 + 113);
    assert(batch.core_count == 1 && batch.cores[0].cycle == 3000);
    assert(batch.cores[0].counters[4] == UINT64_C(0xfedcba9876000004));
    assert(batch.cores[1].cycle == 0);
    recount(&small);
    assert(npu_hal_pcu_snapshot(&small, 1, &batch) == 0);
    assert(small.writes == 1 && !small.stray_writes && small.reads == 113);
    assert(batch.core_count == 1 && batch.cores[0].cycle == 4000);

    /* Same PCU already holding a snapshot from an earlier owner: a non-zero
     * cycle that does not move must not pass for the 4-core layout. */
    assert(npu_hal_pcu_snapshot(&stale, 1, &batch) == 0);
    assert(batch.core_count == 1 && batch.cores[0].cycle == 10000);

    /* No cycle counter running: no layout can be proven, so the topology
     * stands as before, nothing is remembered, and the next snapshot probes
     * again. */
    assert(npu_hal_pcu_snapshot(&idle, 1, &batch) == -ENODATA);
    assert(idle.writes == 2 * 4 + 1);
    assert(idle.reads == 2 * (4 + 3 + 2 + 1) + 4 * 113);
    assert(batch.core_count == 4);
    idle.zero_cycle = 0;
    recount(&idle);
    assert(npu_hal_pcu_snapshot(&idle, 1, &batch) == 0);
    assert(batch.core_count == 4 && !idle.stray_writes);

    /* A memory-backed stand-in whose cycle never moves still snapshots at
     * the topology's layout. */
    assert(npu_hal_pcu_snapshot(&frozen, 1, &batch) == 0);
    assert(batch.core_count == 4 && batch.cores[3].cycle == 1000);

    assert(npu_hal_pcu_snapshot(&single, 1, &batch) == 0);
    assert(single.reads == 2 + 113 && batch.core_count == 1);
    assert(batch.cores[1].cycle == 0);

    hal.cores = 5;
    assert(npu_hal_pcu_snapshot(&hal, 1, &batch) == -ENOTSUP);
    assert(npu_hal_pcu_snapshot(&hal, 2, &batch) == -EINVAL);
    assert(npu_hal_pcu_snapshot(0, 1, &batch) == -EINVAL);
    assert(npu_hal_pcu_snapshot(&hal, 1, 0) == -EINVAL);
    return 0;
}
