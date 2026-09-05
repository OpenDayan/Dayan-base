#include <stdint.h>

#include <dayan_sys.h>

#include "config.h"
#include "errno.h"
#include "platform.h"
#include "resource_scheduler.h"
#include "sche.h"
#include "string.h"
#include "tensor_context.h"

_Static_assert(sizeof(dayan_resource_t) == 1,
               "dayan Resource value ABI changed");
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

enum produce_unit {
    PRODUCE_UNIT_LSU = 1,
    PRODUCE_UNIT_MAT = 2,
    PRODUCE_UNIT_VEC = 3,
    PRODUCE_UNIT_MACRO = 4,
};

#define RESOURCE_NO_OUTPUT 0xffU

static inline void resource_produce_write(uint32_t unit, uint64_t value)
{
    register uint64_t a0 __asm__("a0") = value;

    switch (unit) {
    case PRODUCE_UNIT_LSU:
        __asm__ volatile(".insn 4, 0x0004250b"
                         :
                         : "r"(a0)
                         : "memory");
        break;
    case PRODUCE_UNIT_MAT:
        __asm__ volatile(".insn 4, 0x0004450b"
                         :
                         : "r"(a0)
                         : "memory");
        break;
    case PRODUCE_UNIT_VEC:
        __asm__ volatile(".insn 4, 0x0004650b"
                         :
                         : "r"(a0)
                         : "memory");
        break;
    case PRODUCE_UNIT_MACRO:
        __asm__ volatile(".insn 4, 0x0004850b"
                         :
                         : "r"(a0)
                         : "memory");
        break;
    default:
        platform_halt();
    }
}

static int shared_range_valid(const void *object, size_t object_size,
                              size_t alignment)
{
    struct task_proc *task = task_proc_self();
    uintptr_t start = (uintptr_t)CONFIG_COHERENT_BASE;
    uintptr_t size = (uintptr_t)CONFIG_COHERENT_SIZE;
    uintptr_t base;
    uintptr_t capacity;
    uintptr_t addr = (uintptr_t)object;

    if (!task || !object || !object_size || !alignment ||
        (alignment & (alignment - 1U)))
        return 0;
    base = (uintptr_t)task_shared_base(task);
    capacity = task_shared_size(task);
    if (!base || shared != (void *)base || capacity < object_size ||
        base < start || base - start >= size || capacity > size - (base - start) ||
        addr < base || addr - base > capacity - object_size)
        return 0;
    return !(addr & (alignment - 1U));
}

static uint64_t resource_access_mask(void)
{
    struct dayan_resource_context_fields fields =
        dayan_resource_context_unpack(thread_self()->resource_context);
    uint64_t local = fields.local.count == 32U ? UINT32_MAX :
        ((uint64_t)1U << fields.local.count) - 1U;
    uint64_t shared_mask = fields.shared.count == 32U ? UINT32_MAX :
        ((uint64_t)1U << fields.shared.count) - 1U;

    return local | (shared_mask << 32);
}

void dayan_yield(void)
{
    thread_yield();
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
    status = thread_create_child(thread_self(), name, (thread_fn)entry, argument,
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
    thread_exit_status(result);
}

void *dayan_uram_base(void)
{
    struct task_proc *task = task_proc_self();

    return task ? task_uram_base(task) : 0;
}

size_t dayan_uram_size(void)
{
    struct task_proc *task = task_proc_self();

    return task ? task_uram_size(task) : 0;
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

static inline int barrier_complete(struct thread *thread, void *arg)
{
    const struct barrier_wait *wait = arg;
    uint64_t arrivals;

    (void)thread;
    arrivals = __atomic_load_n(&wait->barrier->arrivals, __ATOMIC_RELAXED);
    return arrivals - wait->target < ((uint64_t)1 << 63);
}

int dayan_cluster_barrier(uint32_t participants,
                          dayan_cluster_barrier_t *barrier)
{
    const struct task_descriptor *descriptor = task_descriptor_self();
    struct barrier_wait wait;
    uint64_t ticket;

    if (!descriptor || participants != 4U || descriptor->task_mode != 4U ||
        !shared_range_valid(barrier, sizeof(*barrier),
                            _Alignof(dayan_cluster_barrier_t)))
        return -EINVAL;
    ticket = barrier_arrive(barrier);
    wait.barrier = barrier;
    wait.target = ticket - ticket % participants + participants;
    while (!barrier_complete(thread_self(), &wait))
        thread_wait_poll(barrier_complete, &wait);
    return 0;
}

uint32_t dayan_task_id(uint32_t axis)
{
    const struct task_descriptor *descriptor = task_descriptor_self();

    return descriptor && axis < 3U ? descriptor->task_id[axis] : 0U;
}

uint32_t dayan_task_dim(uint32_t axis)
{
    const struct task_descriptor *descriptor = task_descriptor_self();

    return descriptor && axis < 3U ? descriptor->task_dim[axis] : 0U;
}

uint32_t dayan_task_mode(void)
{
    const struct task_descriptor *descriptor = task_descriptor_self();

    return descriptor ? descriptor->task_mode : 0U;
}

void dayan_wait_lsu(void)
{
    __asm__ volatile(".insn 4, 0x0000220b" ::: "memory");
    thread_wait_produce();
}

void dayan_wait_mat(void)
{
    __asm__ volatile(".insn 4, 0x0000420b" ::: "memory");
    thread_wait_produce();
}

void dayan_wait_vec(void)
{
    __asm__ volatile(".insn 4, 0x0000620b" ::: "memory");
    thread_wait_produce();
}

void dayan_wait_macro(void)
{
    __asm__ volatile(".insn 4, 0x0000820b" ::: "memory");
    thread_wait_produce();
}

void dayan_wait_dma(void) __attribute__((alias("dayan_wait_lsu")));

int dayan_resource_wait(dayan_resource_mask_t wait_free,
                        dayan_resource_mask_t wait_ready, int32_t qos)
{
    uint64_t overlap = wait_free & wait_ready;
    uint64_t allowed = resource_access_mask();

    if (qos < -4 || qos > 3 || ((wait_free | wait_ready) & ~allowed) ||
        (overlap && (overlap & (overlap - 1U))))
        return -EINVAL;
    return (int)thread_wait_resources(wait_free, wait_ready, (int8_t)qos);
}

int dayan_resource_wait_free(dayan_resource_t resource)
{
    uint64_t mask;
    int status;

    if (resource >= 64U)
        return -EINVAL;
    mask = DAYAN_RESOURCE_MASK(resource);
    status = dayan_resource_wait(mask, 0, 0);
    return status < 0 ? status : 0;
}

int dayan_resource_wait_ready(dayan_resource_t resource)
{
    uint64_t mask;
    int status;

    if (resource >= 64U)
        return -EINVAL;
    mask = DAYAN_RESOURCE_MASK(resource);
    status = dayan_resource_wait(0, mask, 0);
    return status < 0 ? status : 0;
}

static int resource_produce(uint32_t unit, dayan_resource_t output,
                            uint32_t consumer_count,
                            dayan_resource_mask_t consume_ready)
{
    uint64_t output_mask;
    uint64_t allowed = resource_access_mask();
    uintptr_t interrupt_state;

    if (output >= 64U || !consumer_count || consumer_count > UINT8_MAX ||
        (consume_ready & ~allowed))
        return -EINVAL;
    output_mask = DAYAN_RESOURCE_MASK(output);
    if (!(output_mask & allowed) || (output_mask & consume_ready))
        return -EINVAL;

    interrupt_state = resource_scheduler_interrupt_disable();
    resource_scheduler_write_a0(RESOURCE_CONSUME_READY, consume_ready);
    resource_produce_write(unit, output | (consumer_count << 8));
    resource_scheduler_interrupt_restore(interrupt_state);
    return 0;
}

#define DEFINE_RESOURCE_PRODUCE(unit_name, unit_id)                         \
    int dayan_resource_produce_##unit_name(                                 \
        dayan_resource_t output, uint32_t consumer_count,                   \
        dayan_resource_mask_t consume_ready)                                \
    {                                                                       \
        return resource_produce(unit_id, output, consumer_count,            \
                                consume_ready);                             \
    }

DEFINE_RESOURCE_PRODUCE(lsu, PRODUCE_UNIT_LSU)
DEFINE_RESOURCE_PRODUCE(mat, PRODUCE_UNIT_MAT)
DEFINE_RESOURCE_PRODUCE(vec, PRODUCE_UNIT_VEC)
DEFINE_RESOURCE_PRODUCE(macro, PRODUCE_UNIT_MACRO)

#undef DEFINE_RESOURCE_PRODUCE

static int resource_consume(uint32_t unit,
                            dayan_resource_mask_t consume_ready)
{
    uint64_t allowed = resource_access_mask();
    uintptr_t interrupt_state;

    if (!consume_ready || (consume_ready & ~allowed))
        return -EINVAL;

    interrupt_state = resource_scheduler_interrupt_disable();
    resource_scheduler_write_a0(RESOURCE_CONSUME_READY, consume_ready);
    resource_produce_write(unit, RESOURCE_NO_OUTPUT);
    resource_scheduler_interrupt_restore(interrupt_state);
    return 0;
}

#define DEFINE_RESOURCE_CONSUME(unit_name, unit_id)                         \
    int dayan_resource_consume_##unit_name(                                 \
        dayan_resource_mask_t consume_ready)                               \
    {                                                                       \
        return resource_consume(unit_id, consume_ready);                    \
    }

DEFINE_RESOURCE_CONSUME(lsu, PRODUCE_UNIT_LSU)
DEFINE_RESOURCE_CONSUME(mat, PRODUCE_UNIT_MAT)
DEFINE_RESOURCE_CONSUME(vec, PRODUCE_UNIT_VEC)
DEFINE_RESOURCE_CONSUME(macro, PRODUCE_UNIT_MACRO)

#undef DEFINE_RESOURCE_CONSUME
