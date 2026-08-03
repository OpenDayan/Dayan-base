#include <stdint.h>

#include <dayan_sys.h>

#include "config.h"
#include "errno.h"
#include "sche.h"

_Static_assert(sizeof(dayan_resource_t) == 64,
               "dayan resource ABI must stay fixed");
_Static_assert(sizeof(dayan_shared_resource_t) == 16,
               "dayan shared resource ABI must stay fixed");
_Static_assert(sizeof(dayan_cluster_barrier_t) == 16,
               "dayan cluster barrier ABI must stay fixed");
_Static_assert(sizeof(dayan_kernel_resource_desc_t) == 32,
               "dayan kernel resource descriptor ABI must stay fixed");

/*
 * BASE Produce/Bar uses unit 0.  Produce reads tp (x4) directly;
 * global pop writes a0 (x10):
 *   LSU:   address 1, re = 0 -> 0x0000220b
 *   MAT:   address 2, re = 0 -> 0x0000420b
 *   VEC:   address 3, re = 0 -> 0x0000620b
 *   MACRO: address 4, re = 0 -> 0x0000820b
 *   global pop: address 0, re = 1 -> 0x0000150b
 */
static inline uintptr_t produce_pop(void)
{
    register uintptr_t a0 __asm__("a0");

    __asm__ volatile(".insn 4, 0x0000150b"
                     : "=r"(a0)
                     :
                     : "memory");
    return a0;
}

static int resource_lock(dayan_resource_t *resource)
{
    int status;

    while (resource->lock) {
        status = proc_wait_event(&resource->wait_lock);
        if (status < 0)
            return status;
    }
    resource->lock = 1;
    return 0;
}

static void resource_unlock(dayan_resource_t *resource)
{
    resource->lock = 0;
    (void)proc_event_wake_one(&resource->wait_lock);
}

static uint32_t resource_state(const dayan_resource_t *resource)
{
    return resource->state;
}

static void shared_resource_lock(dayan_shared_resource_t *resource)
{
    while (__atomic_exchange_n(&resource->lock, 1, __ATOMIC_ACQUIRE)) {
        while (__atomic_load_n(&resource->lock, __ATOMIC_RELAXED))
            __asm__ volatile("" ::: "memory");
    }
}

static void shared_resource_unlock(dayan_shared_resource_t *resource)
{
    __atomic_store_n(&resource->lock, 0, __ATOMIC_RELEASE);
}

static uint32_t shared_resource_state(
    const dayan_shared_resource_t *resource)
{
    return __atomic_load_n(&resource->state, __ATOMIC_ACQUIRE);
}

static int shared_range_valid(const void *object, size_t object_size,
                              size_t alignment)
{
    struct proc *proc = proc_self();
    uintptr_t start = (uintptr_t)CONFIG_COHERENT_BASE;
    uintptr_t size = (uintptr_t)CONFIG_COHERENT_SIZE;
    uintptr_t base;
    uintptr_t capacity;
    uintptr_t addr = (uintptr_t)object;

    if (!proc || !object || !object_size || !alignment ||
        (alignment & (alignment - 1U)))
        return 0;
    base = (uintptr_t)proc->shared_base;
    capacity = proc->shared_size;
    if (!base || shared != proc->shared_base || capacity < object_size ||
        base < start || base - start >= size || capacity > size - (base - start) ||
        addr < base || addr - base > capacity - object_size)
        return 0;
    return !(addr & (alignment - 1U));
}

static int shared_resource_valid(
    const dayan_shared_resource_t *resource)
{
    return shared_range_valid(resource, sizeof(*resource),
                              _Alignof(dayan_shared_resource_t));
}

static int shared_resource_ready(struct proc *proc, void *arg)
{
    (void)proc;
    return shared_resource_state(arg) == DAYAN_RESOURCE_READY;
}

static int shared_resource_free(struct proc *proc, void *arg)
{
    (void)proc;
    return shared_resource_state(arg) == DAYAN_RESOURCE_FREE;
}

void dayan_yield(void)
{
    proc_yield();
}

int dayan_create_proc(dayan_proc_entry_t entry, void *argument)
{
    uint32_t proc_id;
    int status;

    if (!entry)
        return -EINVAL;
    status = proc_create_child(proc_self(), (proc_fn)entry, argument,
                               &proc_id);
    return status < 0 ? status : (int)proc_id;
}

int dayan_join(uint32_t proc_id, int64_t *result)
{
    return proc_join(proc_id, result);
}

int dayan_join_all(void)
{
    return proc_join_all();
}

DAYAN_NORETURN void dayan_proc_exit(int64_t result)
{
    proc_exit_status(result);
}

void *dayan_uram_base(void)
{
    struct proc *proc = proc_self();

    return proc ? proc->uram_base : 0;
}

size_t dayan_uram_size(void)
{
    struct proc *proc = proc_self();

    return proc ? proc->uram_size : 0;
}

struct barrier_wait {
    dayan_cluster_barrier_t *barrier;
    uint32_t generation;
};

static int barrier_advanced(struct proc *proc, void *arg)
{
    const struct barrier_wait *wait = arg;

    (void)proc;
    return __atomic_load_n(&wait->barrier->generation,
                           __ATOMIC_ACQUIRE) != wait->generation;
}

static void barrier_lock(dayan_cluster_barrier_t *barrier)
{
    while (__atomic_exchange_n(&barrier->lock, 1U, __ATOMIC_ACQUIRE)) {
        while (__atomic_load_n(&barrier->lock, __ATOMIC_RELAXED))
            proc_yield();
    }
}

static void barrier_unlock(dayan_cluster_barrier_t *barrier)
{
    __atomic_store_n(&barrier->lock, 0U, __ATOMIC_RELEASE);
}

int dayan_cluster_barrier(uint32_t participants,
                          dayan_cluster_barrier_t *barrier)
{
    struct proc *proc = proc_self();
    struct barrier_wait wait;
    uint32_t count;

    if (!proc || participants != 4U || proc->task_mode != 4U ||
        !shared_range_valid(barrier, sizeof(*barrier),
                            _Alignof(dayan_cluster_barrier_t)))
        return -EINVAL;
    /* LR/SC is unsupported; serialize the state update with AXI AMOSWAP. */
    barrier_lock(barrier);
    count = barrier->count;
    if (count >= participants) {
        barrier_unlock(barrier);
        return -EBUSY;
    }
    wait.barrier = barrier;
    wait.generation = barrier->generation;
    barrier->count = count + 1U;
    if (count + 1U == participants) {
        barrier->count = 0;
        __atomic_store_n(&barrier->generation, wait.generation + 1U,
                         __ATOMIC_RELEASE);
        barrier_unlock(barrier);
        return 0;
    }
    barrier_unlock(barrier);
    while (!barrier_advanced(proc, &wait)) {
        int status = proc_wait_poll(barrier_advanced, &wait);

        if (status < 0)
            return status;
    }
    return 0;
}

uint32_t dayan_task_id(uint32_t axis)
{
    struct proc *proc = proc_self();

    return proc && axis < 3U ? proc->task_id[axis] : 0U;
}

uint32_t dayan_task_dim(uint32_t axis)
{
    struct proc *proc = proc_self();

    return proc && axis < 3U ? proc->task_dim[axis] : 0U;
}

uint32_t dayan_task_mode(void)
{
    struct proc *proc = proc_self();

    return proc ? proc->task_mode : 0U;
}

void dayan_wait_lsu(void)
{
    __asm__ volatile(".insn 4, 0x0000220b" ::: "memory");
    (void)proc_wait_produce();
}

void dayan_wait_mat(void)
{
    __asm__ volatile(".insn 4, 0x0000420b" ::: "memory");
    (void)proc_wait_produce();
}

void dayan_wait_vec(void)
{
    __asm__ volatile(".insn 4, 0x0000620b" ::: "memory");
    (void)proc_wait_produce();
}

void dayan_wait_macro(void)
{
    __asm__ volatile(".insn 4, 0x0000820b" ::: "memory");
    (void)proc_wait_produce();
}

void dayan_wait_dma(void) __attribute__((alias("dayan_wait_lsu")));

void proc_scan_produce(void)
{
    uintptr_t token;

    while ((token = produce_pop()) != 0)
        (void)proc_complete_produce((struct proc *)token);
}

int dayan_resource_init(dayan_resource_t *resource)
{
    if (!resource)
        return -EINVAL;

    resource->lock = 0;
    resource->producer_count = 0;
    resource->consumer_count = 0;
    resource->state = DAYAN_RESOURCE_FREE;
    INIT_LIST_HEAD(&resource->wait_lock);
    INIT_LIST_HEAD(&resource->wait_ready);
    INIT_LIST_HEAD(&resource->wait_free);
    return 0;
}

int dayan_resource_alloc(dayan_resource_t *resource,
                         uint32_t producer_count, uint32_t consumer_count)
{
    int status = 0;

    if (!resource || !consumer_count)
        return -EINVAL;

    status = resource_lock(resource);
    if (status < 0)
        return status;
    if (resource->state != DAYAN_RESOURCE_FREE) {
        status = -EBUSY;
    } else {
        resource->producer_count = producer_count;
        resource->consumer_count = consumer_count;
        resource->state = producer_count ? DAYAN_RESOURCE_BUSY :
                                           DAYAN_RESOURCE_READY;
        if (!producer_count)
            (void)proc_event_wake_all(&resource->wait_ready);
    }
    resource_unlock(resource);
    return status;
}

int dayan_resource_produce(dayan_resource_t *resource)
{
    int status = 0;

    if (!resource)
        return -EINVAL;

    status = resource_lock(resource);
    if (status < 0)
        return status;
    if (resource->state != DAYAN_RESOURCE_BUSY ||
        !resource->producer_count) {
        status = -EBUSY;
    } else if (!--resource->producer_count) {
        resource->state = DAYAN_RESOURCE_READY;
        (void)proc_event_wake_all(&resource->wait_ready);
    }
    resource_unlock(resource);
    return status;
}

int dayan_resource_consume(dayan_resource_t *resource)
{
    int status = 0;

    if (!resource)
        return -EINVAL;

    status = resource_lock(resource);
    if (status < 0)
        return status;
    if (resource->state != DAYAN_RESOURCE_READY ||
        !resource->consumer_count) {
        status = -EBUSY;
    } else if (!--resource->consumer_count) {
        resource->state = DAYAN_RESOURCE_FREE;
        (void)proc_event_wake_all(&resource->wait_free);
    }
    resource_unlock(resource);
    return status;
}

int dayan_resource_wait_ready(dayan_resource_t *resource)
{
    int status;

    if (!resource)
        return -EINVAL;
    while (resource_state(resource) != DAYAN_RESOURCE_READY) {
        status = proc_wait_event(&resource->wait_ready);
        if (status < 0)
            return status;
    }
    return 0;
}

int dayan_resource_wait_free(dayan_resource_t *resource)
{
    int status;

    if (!resource)
        return -EINVAL;
    while (resource_state(resource) != DAYAN_RESOURCE_FREE) {
        status = proc_wait_event(&resource->wait_free);
        if (status < 0)
            return status;
    }
    return 0;
}

int dayan_shared_resource_init(dayan_shared_resource_t *resource)
{
    if (!resource || !shared_resource_valid(resource))
        return -EINVAL;

    __atomic_store_n(&resource->lock, 0, __ATOMIC_RELAXED);
    __atomic_store_n(&resource->producer_count, 0, __ATOMIC_RELAXED);
    __atomic_store_n(&resource->consumer_count, 0, __ATOMIC_RELAXED);
    __atomic_store_n(&resource->state, DAYAN_RESOURCE_FREE,
                     __ATOMIC_RELEASE);
    return 0;
}

int dayan_shared_resource_alloc(dayan_shared_resource_t *resource,
                                uint32_t producer_count,
                                uint32_t consumer_count)
{
    int status = 0;

    if (!resource || !consumer_count)
        return -EINVAL;

    shared_resource_lock(resource);
    if (__atomic_load_n(&resource->state, __ATOMIC_RELAXED) !=
        DAYAN_RESOURCE_FREE) {
        status = -EBUSY;
    } else {
        __atomic_store_n(&resource->producer_count, producer_count,
                         __ATOMIC_RELAXED);
        __atomic_store_n(&resource->consumer_count, consumer_count,
                         __ATOMIC_RELAXED);
        __atomic_store_n(&resource->state,
                         producer_count ? DAYAN_RESOURCE_BUSY :
                                          DAYAN_RESOURCE_READY,
                         __ATOMIC_RELEASE);
    }
    shared_resource_unlock(resource);
    return status;
}

int dayan_shared_resource_produce(dayan_shared_resource_t *resource)
{
    uint32_t producer_count;
    int status = 0;

    if (!resource)
        return -EINVAL;

    shared_resource_lock(resource);
    producer_count = __atomic_load_n(&resource->producer_count,
                                     __ATOMIC_RELAXED);
    if (__atomic_load_n(&resource->state, __ATOMIC_RELAXED) !=
            DAYAN_RESOURCE_BUSY ||
        !producer_count) {
        status = -EBUSY;
    } else {
        --producer_count;
        __atomic_store_n(&resource->producer_count, producer_count,
                         __ATOMIC_RELAXED);
        if (!producer_count)
            __atomic_store_n(&resource->state, DAYAN_RESOURCE_READY,
                             __ATOMIC_RELEASE);
    }
    shared_resource_unlock(resource);
    return status;
}

int dayan_shared_resource_consume(dayan_shared_resource_t *resource)
{
    uint32_t consumer_count;
    int status = 0;

    if (!resource)
        return -EINVAL;

    shared_resource_lock(resource);
    consumer_count = __atomic_load_n(&resource->consumer_count,
                                     __ATOMIC_RELAXED);
    if (__atomic_load_n(&resource->state, __ATOMIC_RELAXED) !=
            DAYAN_RESOURCE_READY ||
        !consumer_count) {
        status = -EBUSY;
    } else {
        --consumer_count;
        __atomic_store_n(&resource->consumer_count, consumer_count,
                         __ATOMIC_RELAXED);
        if (!consumer_count)
            __atomic_store_n(&resource->state, DAYAN_RESOURCE_FREE,
                             __ATOMIC_RELEASE);
    }
    shared_resource_unlock(resource);
    return status;
}

int dayan_shared_resource_wait_ready(dayan_shared_resource_t *resource)
{
    int status;

    if (!resource)
        return -EINVAL;
    while (shared_resource_state(resource) != DAYAN_RESOURCE_READY) {
        status = proc_wait_poll(shared_resource_ready, resource);
        if (status < 0)
            return status;
    }
    return 0;
}

int dayan_shared_resource_wait_free(dayan_shared_resource_t *resource)
{
    int status;

    if (!resource)
        return -EINVAL;
    while (shared_resource_state(resource) != DAYAN_RESOURCE_FREE) {
        status = proc_wait_poll(shared_resource_free, resource);
        if (status < 0)
            return status;
    }
    return 0;
}
