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
#define DAYAN_BARRIER_SPIN_COUNT 64U

static inline void resource_produce_write(uint32_t unit, uint64_t value)
{
    switch (unit) {
    case PRODUCE_UNIT_LSU:
        __builtin_riscv_tx_base_tcsr_write(value, 0, RESOURCE_PRODUCE_BASE + 1);
        break;
    case PRODUCE_UNIT_MAT:
        __builtin_riscv_tx_base_tcsr_write(value, 0, RESOURCE_PRODUCE_BASE + 2);
        break;
    case PRODUCE_UNIT_VEC:
        __builtin_riscv_tx_base_tcsr_write(value, 0, RESOURCE_PRODUCE_BASE + 3);
        break;
    case PRODUCE_UNIT_MACRO:
        __builtin_riscv_tx_base_tcsr_write(value, 0, RESOURCE_PRODUCE_BASE + 4);
        break;
    default:
        __builtin_unreachable();
    }
}

static uint64_t resource_access_mask(void)
{
    struct dayan_resource_context_fields fields =
        dayan_resource_context_unpack(thread_self()->resource_context);
    uint64_t local = ((uint64_t)1U << fields.local.count) - 1U;
    uint64_t shared_mask = ((uint64_t)1U << fields.shared.count) - 1U;

    return local | (shared_mask << 32);
}

void dayan_yield(void)
{
    thread_yield();
}

uint64_t dayan_get_time(void)
{
    uint64_t cycle;

    __asm__ volatile("rdcycle %0" : "=r"(cycle));
    return cycle;
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
    return thread_create_child(name, (thread_fn)entry, argument, resources);
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
    struct dayan_context_fields fields;

    if (!task)
        return 0;
    fields = dayan_context_unpack(task->thread.tensor_context);
    if (fields.uram_lo == fields.uram_hi)
        return 0;
    return (void *)((uintptr_t)CONFIG_URAM_BASE +
                    platform_hart_id() * (uintptr_t)CONFIG_URAM_STRIDE +
                    fields.uram_lo * DAYAN_CONTEXT_URAM_GRANULE);
}

size_t dayan_uram_size(void)
{
    struct task_proc *task = task_proc_self();
    struct dayan_context_fields fields;

    if (!task)
        return 0;
    fields = dayan_context_unpack(task->thread.tensor_context);
    return (size_t)(fields.uram_hi - fields.uram_lo) *
           DAYAN_CONTEXT_URAM_GRANULE;
}

int dayan_cluster_barrier(uint32_t participants,
                          dayan_cluster_barrier_t *barrier)
{
    const struct task_descriptor *descriptor = task_descriptor_self();
    uintptr_t address = (uintptr_t)barrier;
    uintptr_t base;
    uint64_t ticket, target;
    uint64_t increment = 1;

    if (!descriptor || !participants || participants > 4U ||
        descriptor->task_mode != 4U)
        return -EINVAL;
    base = CONFIG_COHERENT_BASE + descriptor->shared_offset;
    if (descriptor->shared_size < sizeof(*barrier) ||
        address < base ||
        address - base > descriptor->shared_size - sizeof(*barrier) ||
        (address & (_Alignof(dayan_cluster_barrier_t) - 1U)))
        return -EINVAL;
    __asm__ volatile("amoadd.d %0, %2, (%1)"
                     : "=r"(ticket)
                     : "r"(&barrier->arrivals), "r"(increment)
                     : "memory");
    target = ticket - ticket % participants + participants;
    for (uint32_t spins = 0;;) {
        if (__atomic_load_n(&barrier->arrivals, __ATOMIC_RELAXED) - target <
            ((uint64_t)1 << 63))
            break;
        if (++spins == DAYAN_BARRIER_SPIN_COUNT) {
            thread_busy_yield();
            spins = 0;
        }
    }
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
    __builtin_riscv_tx_base_tcsr_write((uintptr_t)current, 0, 1);
    thread_wait_unit(THREAD_WAIT_LSU);
}

void dayan_wait_mat(void)
{
    __builtin_riscv_tx_base_tcsr_write((uintptr_t)current, 0, 2);
    thread_wait_unit(THREAD_WAIT_MAT);
}

void dayan_wait_vec(void)
{
    __builtin_riscv_tx_base_tcsr_write((uintptr_t)current, 0, 3);
    thread_wait_unit(THREAD_WAIT_VEC);
}

void dayan_wait_macro(void)
{
    __builtin_riscv_tx_base_tcsr_write((uintptr_t)current, 0, 4);
    thread_wait_unit(THREAD_WAIT_MACRO);
}

void dayan_wait_dma(void) __attribute__((alias("dayan_wait_lsu")));

int dayan_resource_wait(dayan_resource_mask_t wait_free,
                        dayan_resource_mask_t wait_ready,
                        dayan_resource_mask_t phase, int32_t qos)
{
    uint64_t overlap = wait_free & wait_ready;
    uint64_t allowed = resource_access_mask();

    if (qos < -4 || qos > 3 || ((wait_free | wait_ready) & ~allowed) ||
        (overlap && (overlap & (overlap - 1U))))
        return -EINVAL;
    return (int)thread_wait_resources(wait_free, wait_ready, phase, (int8_t)qos);
}

int dayan_resource_wait_free(dayan_resource_t resource)
{
    uint64_t mask;
    int status;

    if (resource >= 64U)
        return -EINVAL;
    mask = DAYAN_RESOURCE_MASK(resource);
    status = dayan_resource_wait(mask, 0, 0, 0);
    return status < 0 ? status : 0;
}

int dayan_resource_wait_ready(dayan_resource_t resource, uint32_t phase)
{
    uint64_t mask;
    int status;

    if (resource >= 64U)
        return -EINVAL;
    mask = DAYAN_RESOURCE_MASK(resource);
    if (phase > 1U)
        return -EINVAL;
    status = dayan_resource_wait(0, mask, phase ? mask : 0, 0);
    return status < 0 ? status : 0;
}

static int resource_produce(uint32_t unit, dayan_resource_t output,
                            uint32_t consumer_count,
                            dayan_resource_mask_t consume_ready)
{
    uint64_t output_mask;
    uint64_t allowed = resource_access_mask();

    if (output >= 64U || !consumer_count || consumer_count > UINT8_MAX ||
        (consume_ready & ~allowed))
        return -EINVAL;
    output_mask = DAYAN_RESOURCE_MASK(output);
    if (!(output_mask & allowed) || (output_mask & consume_ready))
        return -EINVAL;

    __builtin_riscv_tx_base_tcsr_write(consume_ready, 0, RESOURCE_CONSUME_READY);
    resource_produce_write(unit, output | (consumer_count << 8));
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

    if (!consume_ready || (consume_ready & ~allowed))
        return -EINVAL;

    __builtin_riscv_tx_base_tcsr_write(consume_ready, 0, RESOURCE_CONSUME_READY);
    resource_produce_write(unit, RESOURCE_NO_OUTPUT);
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
