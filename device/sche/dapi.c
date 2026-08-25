#include <stdint.h>

#include <dayan_sys.h>

#include "config.h"
#include "errno.h"
#include "platform.h"
#include "sche.h"
#include "string.h"

_Static_assert(sizeof(dayan_resource_t) == 64,
               "dayan resource ABI must stay fixed");
_Static_assert(sizeof(dayan_shared_resource_t) == 16,
               "dayan shared resource ABI must stay fixed");
_Static_assert(_Alignof(dayan_shared_resource_t) == 8,
               "dayan shared resource ABI alignment changed");
_Static_assert(offsetof(dayan_shared_resource_t, counts) == 0 &&
                   offsetof(dayan_shared_resource_t, reserved) == 8,
               "dayan shared resource ABI layout changed");
_Static_assert(sizeof(dayan_cluster_barrier_t) == 16,
               "dayan cluster barrier ABI must stay fixed");
_Static_assert(offsetof(dayan_cluster_barrier_t, arrivals) == 0 &&
                   offsetof(dayan_cluster_barrier_t, reserved) == 8,
               "dayan cluster barrier ABI layout changed");
_Static_assert(sizeof(dayan_kernel_resource_desc_t) == 32,
               "dayan kernel resource descriptor ABI must stay fixed");
_Static_assert(offsetof(dayan_kernel_resource_desc_t, max_child_threads) == 0 &&
                   offsetof(dayan_kernel_resource_desc_t,
                            max_tasks_per_core) == 2 &&
                   offsetof(dayan_kernel_resource_desc_t, flags) == 4 &&
                   offsetof(dayan_kernel_resource_desc_t,
                            lsu_tcsr_count) == 8 &&
                   offsetof(dayan_kernel_resource_desc_t,
                            atomic_cache_size) == 16 &&
                   offsetof(dayan_kernel_resource_desc_t, uram_size) == 24,
               "dayan kernel resource descriptor layout changed");
_Static_assert(sizeof(dayan_thread_resources_t) == 8,
               "dayan Thread resource ABI must stay fixed");

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

#define PRODUCE_TOKEN_ADDRESS_MASK 0x0000ffffffffffffULL
#define PRODUCE_TOKEN_EVENT_SHIFT  48U

enum produce_token_event {
    PRODUCE_TOKEN_WAKE_THREAD = 0,
    PRODUCE_TOKEN_RESOURCE_PRODUCE = 1,
    PRODUCE_TOKEN_RESOURCE_CONSUME = 2,
    PRODUCE_TOKEN_SHARED_RESOURCE_PRODUCE = 3,
    PRODUCE_TOKEN_SHARED_RESOURCE_CONSUME = 4,
};

enum produce_unit {
    PRODUCE_UNIT_LSU = 1,
    PRODUCE_UNIT_MAT = 2,
    PRODUCE_UNIT_VEC = 3,
    PRODUCE_UNIT_MACRO = 4,
};

static inline void produce_push(uint32_t unit, uintptr_t token)
{
    register uintptr_t a0 __asm__("a0") = token;

    switch (unit) {
    case PRODUCE_UNIT_LSU:
        __asm__ volatile(".insn 4, 0x0000250b"
                         :
                         : "r"(a0)
                         : "memory");
        break;
    case PRODUCE_UNIT_MAT:
        __asm__ volatile(".insn 4, 0x0000450b"
                         :
                         : "r"(a0)
                         : "memory");
        break;
    case PRODUCE_UNIT_VEC:
        __asm__ volatile(".insn 4, 0x0000650b"
                         :
                         : "r"(a0)
                         : "memory");
        break;
    case PRODUCE_UNIT_MACRO:
        __asm__ volatile(".insn 4, 0x0000850b"
                         :
                         : "r"(a0)
                         : "memory");
        break;
    default:
        platform_halt();
    }
}

static inline void resource_lock(dayan_resource_t *resource)
{
    while (resource->lock)
        proc_wait_event(&resource->wait_lock);
    resource->lock = 1;
}

static inline void resource_unlock(dayan_resource_t *resource)
{
    resource->lock = 0;
    (void)proc_event_wake_one(&resource->wait_lock);
}

static inline uint32_t resource_state(const dayan_resource_t *resource)
{
    return resource->state;
}

#define SHARED_RESOURCE_PRODUCER_ONE (1ULL << 32)

static inline uint64_t shared_resource_pack(uint32_t producer_count,
                                            uint32_t consumer_count)
{
    return ((uint64_t)producer_count << 32) | consumer_count;
}

static inline uint32_t shared_resource_producers(uint64_t counts)
{
    return (uint32_t)(counts >> 32);
}

static inline uint32_t shared_resource_consumers(uint64_t counts)
{
    return (uint32_t)counts;
}

static inline uint64_t shared_resource_counts(
    const dayan_shared_resource_t *resource)
{
    return __atomic_load_n(&resource->counts, __ATOMIC_RELAXED);
}

static inline uint64_t shared_resource_fetch_add(
    dayan_shared_resource_t *resource, uint64_t value)
{
    uint64_t old;

    __asm__ volatile("amoadd.d %0, %2, (%1)"
                     : "=r"(old)
                     : "r"(&resource->counts), "r"(value)
                     : "memory");
    return old;
}

static inline uint64_t shared_resource_compare_exchange(
    dayan_shared_resource_t *resource, uint64_t expected, uint64_t desired)
{
    uint64_t observed = expected;

    /* AMOCAS.D: rd is both the comparison operand and returned old value. */
    __asm__ volatile("amocas.d %0, %2, (%1)"
                     : "+r"(observed)
                     : "r"(&resource->counts), "r"(desired)
                     : "memory");
    return observed;
}

static inline uint32_t shared_resource_state(
    const dayan_shared_resource_t *resource)
{
    uint64_t counts = shared_resource_counts(resource);

    if (!counts)
        return DAYAN_RESOURCE_FREE;
    return shared_resource_producers(counts) ? DAYAN_RESOURCE_BUSY :
                                               DAYAN_RESOURCE_READY;
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

static inline int shared_resource_valid(
    const dayan_shared_resource_t *resource)
{
    return shared_range_valid(resource, sizeof(*resource),
                              _Alignof(dayan_shared_resource_t));
}

static int resource_wait_issue(void *resource, size_t alignment,
                               uint16_t event, uint32_t unit,
                               int shared_resource)
{
    uintptr_t address = (uintptr_t)resource;
    uintptr_t token;

    if (!resource || address & ~PRODUCE_TOKEN_ADDRESS_MASK ||
        (!shared_resource && address & (alignment - 1U)) ||
        (shared_resource &&
         !shared_resource_valid((dayan_shared_resource_t *)resource)))
        return -EINVAL;
    token = address | ((uintptr_t)event << PRODUCE_TOKEN_EVENT_SHIFT);
    produce_push(unit, token);
    return 0;
}

static inline int shared_resource_ready(struct proc *proc, void *arg)
{
    (void)proc;
    return shared_resource_state(arg) == DAYAN_RESOURCE_READY;
}

static inline int shared_resource_free(struct proc *proc, void *arg)
{
    (void)proc;
    return shared_resource_counts(arg) == 0;
}

void dayan_yield(void)
{
    proc_yield();
}

void dayan_thread_resources_init(dayan_thread_resources_t *resources)
{
    if (!resources)
        return;
    memset(resources, 0, sizeof(*resources));
}

int dayan_create_thread(const char *name, dayan_thread_entry_t entry,
                        void *argument,
                        const dayan_thread_resources_t *resources)
{
    uint32_t thread_id;
    int status;

    if (!name || !entry)
        return -EINVAL;
    status = thread_create_child(proc_self(), name, (proc_fn)entry, argument,
                                 resources, &thread_id);
    return status < 0 ? status : (int)thread_id;
}

int dayan_thread_join(uint32_t thread_id, int64_t *result)
{
    return thread_join(thread_id, result);
}

int dayan_thread_join_all(void)
{
    return thread_join_all();
}

DAYAN_NORETURN void dayan_thread_exit(int64_t result)
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
    uint64_t target;
};

static inline uint64_t barrier_arrive(dayan_cluster_barrier_t *barrier)
{
    uint64_t previous;
    uint64_t increment = 1;

    __asm__ volatile("amoadd.d %0, %2, (%1)"
                     : "=r"(previous)
                     : "r"(&barrier->arrivals), "r"(increment)
                     : "memory");
    return previous;
}

static inline int barrier_complete(struct proc *proc, void *arg)
{
    const struct barrier_wait *wait = arg;
    uint64_t arrivals;

    (void)proc;
    arrivals = __atomic_load_n(&wait->barrier->arrivals, __ATOMIC_RELAXED);
    return arrivals - wait->target < ((uint64_t)1 << 63);
}

int dayan_cluster_barrier(uint32_t participants,
                          dayan_cluster_barrier_t *barrier)
{
    struct proc *proc = proc_self();
    struct barrier_wait wait;
    uint64_t ticket;

    if (!proc || participants != 4U || proc->task_mode != 4U ||
        !shared_range_valid(barrier, sizeof(*barrier),
                            _Alignof(dayan_cluster_barrier_t)))
        return -EINVAL;
    ticket = barrier_arrive(barrier);
    wait.barrier = barrier;
    wait.target = ticket - ticket % participants + participants;
    while (!barrier_complete(proc, &wait))
        proc_wait_poll(barrier_complete, &wait);
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
    proc_wait_produce();
}

void dayan_wait_mat(void)
{
    __asm__ volatile(".insn 4, 0x0000420b" ::: "memory");
    proc_wait_produce();
}

void dayan_wait_vec(void)
{
    __asm__ volatile(".insn 4, 0x0000620b" ::: "memory");
    proc_wait_produce();
}

void dayan_wait_macro(void)
{
    __asm__ volatile(".insn 4, 0x0000820b" ::: "memory");
    proc_wait_produce();
}

void dayan_wait_dma(void) __attribute__((alias("dayan_wait_lsu")));

#define DEFINE_RESOURCE_WAIT_APIS(unit_name, unit_id)                         \
    int dayan_resource_wait_##unit_name##_produce(                            \
        dayan_resource_t *resource)                                           \
    {                                                                         \
        return resource_wait_issue(                                           \
            resource, _Alignof(dayan_resource_t),                             \
            PRODUCE_TOKEN_RESOURCE_PRODUCE, unit_id, 0);                      \
    }                                                                         \
                                                                              \
    int dayan_resource_wait_##unit_name##_consume(                            \
        dayan_resource_t *resource)                                           \
    {                                                                         \
        return resource_wait_issue(                                           \
            resource, _Alignof(dayan_resource_t),                             \
            PRODUCE_TOKEN_RESOURCE_CONSUME, unit_id, 0);                      \
    }                                                                         \
                                                                              \
    int dayan_shared_resource_wait_##unit_name##_produce(                     \
        dayan_shared_resource_t *resource)                                    \
    {                                                                         \
        return resource_wait_issue(                                           \
            resource, _Alignof(dayan_shared_resource_t),                      \
            PRODUCE_TOKEN_SHARED_RESOURCE_PRODUCE, unit_id, 1);               \
    }                                                                         \
                                                                              \
    int dayan_shared_resource_wait_##unit_name##_consume(                     \
        dayan_shared_resource_t *resource)                                    \
    {                                                                         \
        return resource_wait_issue(                                           \
            resource, _Alignof(dayan_shared_resource_t),                      \
            PRODUCE_TOKEN_SHARED_RESOURCE_CONSUME, unit_id, 1);               \
    }

DEFINE_RESOURCE_WAIT_APIS(lsu, PRODUCE_UNIT_LSU)
DEFINE_RESOURCE_WAIT_APIS(mat, PRODUCE_UNIT_MAT)
DEFINE_RESOURCE_WAIT_APIS(vec, PRODUCE_UNIT_VEC)
DEFINE_RESOURCE_WAIT_APIS(macro, PRODUCE_UNIT_MACRO)

#undef DEFINE_RESOURCE_WAIT_APIS

void scheduler_scan_produce(void)
{
    uintptr_t token;

    while ((token = produce_pop()) != 0) {
        uint16_t event = (uint16_t)(token >> PRODUCE_TOKEN_EVENT_SHIFT);
        void *object = (void *)(token & PRODUCE_TOKEN_ADDRESS_MASK);
        int status;

        switch (event) {
        case PRODUCE_TOKEN_WAKE_THREAD:
            status = proc_wake((struct proc *)object);
            break;
        case PRODUCE_TOKEN_RESOURCE_PRODUCE:
            status = dayan_resource_produce((dayan_resource_t *)object);
            break;
        case PRODUCE_TOKEN_RESOURCE_CONSUME:
            status = dayan_resource_consume((dayan_resource_t *)object);
            break;
        case PRODUCE_TOKEN_SHARED_RESOURCE_PRODUCE:
            status = dayan_shared_resource_produce(
                (dayan_shared_resource_t *)object);
            break;
        case PRODUCE_TOKEN_SHARED_RESOURCE_CONSUME:
            status = dayan_shared_resource_consume(
                (dayan_shared_resource_t *)object);
            break;
        default:
            platform_halt();
        }
        if (status < 0)
            platform_halt();
    }
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

    resource_lock(resource);
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

    resource_lock(resource);
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

    resource_lock(resource);
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
    if (!resource)
        return -EINVAL;
    while (resource_state(resource) != DAYAN_RESOURCE_READY)
        proc_wait_event(&resource->wait_ready);
    return 0;
}

int dayan_resource_wait_free(dayan_resource_t *resource)
{
    if (!resource)
        return -EINVAL;
    while (resource_state(resource) != DAYAN_RESOURCE_FREE)
        proc_wait_event(&resource->wait_free);
    return 0;
}

int dayan_shared_resource_init(dayan_shared_resource_t *resource)
{
    if (!resource || !shared_resource_valid(resource))
        return -EINVAL;

    __atomic_store_n(&resource->counts, 0, __ATOMIC_RELAXED);
    __atomic_store_n(&resource->reserved, 0, __ATOMIC_RELAXED);
    return 0;
}

int dayan_shared_resource_alloc(dayan_shared_resource_t *resource,
                                uint32_t producer_count,
                                uint32_t consumer_count)
{
    uint64_t counts;

    if (!resource || !consumer_count)
        return -EINVAL;
    counts = shared_resource_pack(producer_count, consumer_count);
    return shared_resource_compare_exchange(resource, 0, counts) ?
               -EBUSY : 0;
}

int dayan_shared_resource_produce(dayan_shared_resource_t *resource)
{
    uint64_t counts;

    if (!resource)
        return -EINVAL;
    counts = shared_resource_fetch_add(resource,
                                       -SHARED_RESOURCE_PRODUCER_ONE);
    if (!shared_resource_producers(counts) ||
        !shared_resource_consumers(counts)) {
        (void)shared_resource_fetch_add(resource,
                                        SHARED_RESOURCE_PRODUCER_ONE);
        return -EBUSY;
    }
    return 0;
}

int dayan_shared_resource_consume(dayan_shared_resource_t *resource)
{
    uint64_t counts;
    uint32_t consumer_count;

    if (!resource)
        return -EINVAL;
    counts = shared_resource_fetch_add(resource, ~0ULL);
    consumer_count = shared_resource_consumers(counts);
    if (shared_resource_producers(counts) || !consumer_count) {
        (void)shared_resource_fetch_add(resource, 1ULL);
        return -EBUSY;
    }
    return 0;
}

int dayan_shared_resource_wait_ready(dayan_shared_resource_t *resource)
{
    if (!resource)
        return -EINVAL;
    while (shared_resource_state(resource) != DAYAN_RESOURCE_READY)
        proc_wait_poll(shared_resource_ready, resource);
    return 0;
}

int dayan_shared_resource_wait_free(dayan_shared_resource_t *resource)
{
    if (!resource)
        return -EINVAL;
    while (shared_resource_counts(resource))
        proc_wait_poll(shared_resource_free, resource);
    return 0;
}
