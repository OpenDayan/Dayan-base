#include <assert.h>
#include <errno.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "service_internal.h"

struct npu_hal_allocation {
    struct npu_hal_allocation *next;
    size_t size;
    unsigned char data[];
};

static npu_service_device_t device;
static struct npu_hal_allocation *allocations, *buffer;
static pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t condition = PTHREAD_COND_INITIALIZER;
static unsigned entered[2], expected_stop = 0xf, active, buffer_frees;
static unsigned fail_start, fail_stop;

int npu_hal_core_valid(npu_hal_t *hal, npu_hal_core_t core)
{
    (void)hal;
    return core.cluster == 0 && core.core < 4;
}

size_t npu_service_core_index(const npu_service_device_t *runtime,
                              npu_hal_core_t core)
{
    (void)runtime;
    return core.core;
}

void npu_hal_memory_request_init(npu_hal_memory_request_t *request)
{
    memset(request, 0, sizeof(*request));
}

static int allocate(size_t size, npu_hal_allocation_t **out)
{
    struct npu_hal_allocation *allocation = calloc(1, sizeof(*allocation) + size);

    assert(allocation);
    allocation->size = size;
    pthread_mutex_lock(&lock);
    allocation->next = allocations;
    allocations = allocation;
    pthread_mutex_unlock(&lock);
    *out = allocation;
    return 0;
}

int npu_hal_memory_alloc(npu_hal_t *hal, const npu_hal_memory_request_t *request,
                         npu_hal_allocation_t **out)
{
    (void)hal;
    assert(request->usage == NPU_HAL_MEMORY_PROFILE);
    allocate(request->size, out);
    buffer = *out;
    return 0;
}

int npu_service_alloc_rpc_descriptor(npu_service_device_t *runtime,
                                     npu_hal_core_t core, size_t size,
                                     npu_hal_allocation_t **out)
{
    (void)runtime;
    (void)core;
    assert(size == sizeof(struct rpc_profile_config));
    return allocate(size, out);
}

int npu_service_mmu_device_address(npu_service_device_t *runtime,
                                   const npu_hal_allocation_t *allocation,
                                   npu_hal_core_t core, uint64_t *out)
{
    (void)runtime;
    (void)core;
    *out = allocation == buffer ? UINT64_C(0x80000000) :
                                 (uintptr_t)allocation->data;
    return 0;
}

int npu_hal_memory_write(npu_hal_t *hal, const npu_hal_allocation_t *allocation,
                         npu_hal_core_t core, uint64_t offset,
                         const void *source, size_t size)
{
    (void)hal;
    (void)core;
    assert(offset + size <= allocation->size);
    memcpy((void *)(allocation->data + offset), source, size);
    return 0;
}

int npu_hal_memory_read(npu_hal_t *hal, const npu_hal_allocation_t *allocation,
                        npu_hal_core_t core, void *destination,
                        uint64_t offset, size_t size)
{
    (void)hal;
    (void)core;
    assert(offset + size <= allocation->size);
    memcpy(destination, allocation->data + offset, size);
    return 0;
}

int npu_hal_memory_free(npu_hal_t *hal, npu_hal_allocation_t *allocation)
{
    struct npu_hal_allocation **link;

    (void)hal;
    pthread_mutex_lock(&lock);
    if (allocation == buffer) {
        assert(!active);
        ++buffer_frees;
        buffer = 0;
    }
    for (link = &allocations; *link != allocation; link = &(*link)->next)
        assert(*link);
    *link = allocation->next;
    pthread_mutex_unlock(&lock);
    free(allocation);
    return 0;
}

int npu_service_rpc_call(npu_service_device_t *runtime, npu_hal_core_t core,
                         uint16_t opcode, uint64_t argument, uint32_t timeout,
                         int *may_be_live)
{
    struct rpc_profile_config *config = (void *)(uintptr_t)argument;
    struct timespec deadline;
    unsigned stopping = config->command == RPC_PROFILE_STOP;
    unsigned mask = stopping ? expected_stop : 0xf;
    int failed = !!((stopping ? fail_stop : fail_start) & (1U << core.core));

    (void)runtime;
    (void)timeout;
    assert(opcode == RPC_PROFILE_CONFIG && config->hart_id == core.core);
    assert(clock_gettime(CLOCK_REALTIME, &deadline) == 0);
    deadline.tv_sec += 5;
    pthread_mutex_lock(&lock);
    if (stopping && fail_start)
        assert(entered[0] == 0xf); /* All STARTs finish before rollback. */
    ++active;
    entered[stopping] |= 1U << core.core;
    pthread_cond_broadcast(&condition);
    while (entered[stopping] != mask)
        assert(pthread_cond_timedwait(&condition, &lock, &deadline) == 0);
    --active;
    pthread_mutex_unlock(&lock);
    *may_be_live = failed;
    config->enabled = !stopping;
    config->count = 0;
    config->status = 0;
    return failed ? -ETIMEDOUT : 0;
}

int main(void)
{
    assert(pthread_mutex_init(&device.mutex, 0) == 0);
    assert(pthread_cond_init(&device.condition, 0) == 0);
    device.clusters = calloc(1, sizeof(*device.clusters));
    assert(device.clusters);
    device.clusters[0].booted_cores = 0xf;
    device.hal_info.cluster_count = 1;
    device.hal_info.cores_per_cluster = 4;

    assert(npu_service_profile_start(&device, 4U * 4096) == 0);
    assert(entered[0] == 0xf && device.profile_active);
    assert(npu_service_profile_stop(&device) == 0);
    assert(entered[1] == 0xf && !device.profile_allocation && buffer_frees == 1);

    entered[0] = entered[1] = 0;
    fail_start = 1U << 1;
    fail_stop = 1U << 2;
    assert(npu_service_profile_start(&device, 4U * 4096) == -ETIMEDOUT);
    assert(entered[0] == 0xf && entered[1] == 0xf);
    assert(!device.profile_active && !device.profile_transition);
    assert(device.profile_allocation && buffer_frees == 1);
    for (unsigned i = 0; i < 4; ++i)
        assert(device.profile_cores[i].configured == (i == 2));
    entered[1] = 0;
    expected_stop = 1U << 2;
    fail_stop = 0;
    assert(npu_service_profile_stop(&device) == 0);
    assert(entered[1] == expected_stop && buffer_frees == 2);
    assert(!device.profile_allocation && !active);

    /* Timed-out descriptors live until the mock HAL closes. */
    while (allocations)
        npu_hal_memory_free(0, allocations);
    free(device.clusters);
    pthread_cond_destroy(&device.condition);
    pthread_mutex_destroy(&device.mutex);
    return 0;
}
