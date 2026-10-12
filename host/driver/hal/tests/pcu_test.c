#include <assert.h>
#include <errno.h>
#include <string.h>

#include "hal.h"
#include "platform_regs.h"

struct npu_hal {
    unsigned int cores;
    unsigned int reads;
    unsigned int writes;
    unsigned int fail_read;
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

    assert(cluster == 1);
    assert(size == 8 && !memcmp(source, trigger, 8));
    assert(address == 0x30000U + hal->cores * 0x2000U - 8U);
    ++hal->writes;
    hal->reads = 0;
    return hal->fail_write ? -EIO : 0;
}

int npu_hal_copy_from_address(npu_hal_t *hal, uint32_t cluster,
                              void *destination, uint64_t address, size_t size)
{
    unsigned int core = hal->reads / 113U;
    unsigned int index = hal->reads % 113U;
    uint64_t value = index ? UINT64_C(0xfedcba9876000000) +
                             core * 1024U + index - 1U :
                             (hal->zero_cycle ? 0 : 1000U * hal->writes);
    unsigned char *bytes = destination;
    unsigned int i;

    assert(hal->writes && size == 8 && cluster == 1);
    assert(core < hal->cores);
    assert(address == 0x30000U + core * 0x2000U +
                      (index ? (index - 1U) * 8U : 0x1ff0U));
    if (++hal->reads == hal->fail_read)
        return -EIO;
    for (i = 0; i < 8; ++i)
        bytes[i] = (unsigned char)(value >> (i * 8U));
    return 0;
}

int main(void)
{
    struct npu_hal hal = { .cores = 4 };
    npu_pcu_batch_t batch;
    unsigned int core;

    assert(npu_hal_pcu_snapshot(&hal, 1, &batch) == 0);
    assert(hal.writes == 1 && hal.reads == 4 * 113);
    assert(batch.cluster == 1 && batch.core_count == 4 && !batch.flags);
    for (core = 0; core < 4; ++core) {
        assert(batch.cores[core].cycle == 1000);
        assert(batch.cores[core].counters[109] ==
               UINT64_C(0xfedcba987600006d) + core * 1024U);
    }
    assert(npu_hal_pcu_snapshot(&hal, 1, &batch) == 0);
    assert(hal.writes == 2 && batch.cores[3].cycle == 2000);
    hal.fail_read = 114;
    assert(npu_hal_pcu_snapshot(&hal, 1, &batch) == -EIO);
    assert(hal.reads == 114);
    hal.fail_read = 0;
    hal.fail_write = 1;
    assert(npu_hal_pcu_snapshot(&hal, 1, &batch) == -EIO);
    assert(!hal.reads);
    hal.fail_write = 0;
    hal.zero_cycle = 1;
    assert(npu_hal_pcu_snapshot(&hal, 1, &batch) == -ENODATA);
    hal.cores = 5;
    assert(npu_hal_pcu_snapshot(&hal, 1, &batch) == -ENOTSUP);
    assert(npu_hal_pcu_snapshot(&hal, 2, &batch) == -EINVAL);
    assert(npu_hal_pcu_snapshot(0, 1, &batch) == -EINVAL);
    assert(npu_hal_pcu_snapshot(&hal, 1, 0) == -EINVAL);
    hal.cores = 1;
    hal.zero_cycle = 0;
    assert(npu_hal_pcu_snapshot(&hal, 1, &batch) == 0);
    assert(hal.reads == 113 && batch.core_count == 1);
    assert(batch.cores[1].cycle == 0);
    return 0;
}
