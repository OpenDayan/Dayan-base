#include <stddef.h>
#include <stdint.h>

#include <dayan_sys.h>

#include "config.h"
#include "context.h"
#include "cpu.h"
#include "errno.h"
#include "platform.h"
#include "profile.h"
#include "resource_scheduler.h"
#include "rpc.h"
#include "rpc_internal.h"
#include "sche.h"
#include "string.h"
#include "tensor_context.h"
#include "tensor_interrupt.h"

extern void proc_bootstrap(void);

#define THREAD_SLOT_COUNT       RPC_PROFILE_SLOT_COUNT
#define TASK_SLOT_FIRST         RPC_PROFILE_SLOT_USER_BASE
#define TASK_SLOT_LAST          CONFIG_PROC_SLOT_COUNT
#define TASK_SLOT_COUNT         CONFIG_PROC_SLOT_COUNT
#define SYSTEM_SLOT_FIRST       RPC_PROFILE_SLOT_RPC_RECEIVE
#define SLOT_ARRAY __attribute__((section(".proc_slots")))

enum slot_state {
    SLOT_FREE = 0,
    SLOT_SYSTEM,
    SLOT_ROOT,
    SLOT_RESERVED,
    SLOT_CHILD,
    SLOT_ZOMBIE,
};

static union thread_slot
    hart_thread_slots[CONFIG_HART_COUNT][THREAD_SLOT_COUNT - 1U]
    __attribute__((section(".proc_slots"), aligned(64)));
static unsigned char
    hart_thread_stacks[CONFIG_HART_COUNT][CONFIG_PROC_SLOT_COUNT]
                      [CONFIG_PROC_STACK_SIZE]
    __attribute__((section(".proc_stack"), aligned(64)));
static unsigned char
    hart_thread_printf[CONFIG_HART_COUNT][CONFIG_PROC_SLOT_COUNT]
                      [CONFIG_PROC_PRINTF_SIZE]
    __attribute__((section(".proc_printf"), aligned(64)));

struct thread *scheduler_rpc_receive_threads[CONFIG_HART_COUNT] SLOT_ARRAY;

_Static_assert(offsetof(struct thread, state) == THREAD_STATE_OFFSET,
               "assembly thread state offset");
_Static_assert(offsetof(struct thread, scheduled) == THREAD_SCHEDULED_OFFSET,
               "assembly thread scheduled offset");
_Static_assert(offsetof(struct thread, sp) == THREAD_SP_OFFSET,
               "assembly thread sp offset");
_Static_assert(offsetof(struct thread, pc) == THREAD_PC_OFFSET,
               "assembly thread pc offset");
_Static_assert(offsetof(struct thread, tensor_context) ==
                   THREAD_TENSOR_CONTEXT_OFFSET,
               "assembly thread Context1 offset");
_Static_assert(offsetof(struct thread, resource_context) ==
                   THREAD_RESOURCE_CONTEXT_OFFSET,
               "assembly thread Context2 offset");
_Static_assert(offsetof(struct thread, token_ext) == THREAD_TOKEN_EXT_OFFSET,
               "assembly thread token extension offset");
_Static_assert(offsetof(struct thread, reap_deferred) ==
                   THREAD_REAP_DEFERRED_OFFSET,
               "assembly deferred reap offset");

/* Each Hart owns one cache-line-isolated structure of per-slot arrays. */
struct thread_runtime {
    void *task_cleanup_arg[TASK_SLOT_COUNT];
    void *system_wait[THREAD_SLOT_COUNT - 1U];
    int64_t result[TASK_SLOT_COUNT];
    struct thread_list *list_owner[THREAD_SLOT_COUNT];
    struct thread_list wait_children[TASK_SLOT_COUNT];
    uint16_t output_size[TASK_SLOT_COUNT];
    uint8_t slot_state[THREAD_SLOT_COUNT];
    uint8_t slot_owner[TASK_SLOT_COUNT];
    uint8_t generation[THREAD_SLOT_COUNT];
} __attribute__((aligned(64)));

static struct thread_runtime hart_thread_runtime[CONFIG_HART_COUNT] SLOT_ARRAY;

#define HART_RUNTIME(hart) hart_thread_runtime[(hart)]
#define TASK_INDEX(slot)   ((slot) - TASK_SLOT_FIRST)
#define WAIT_INDEX(slot)   ((slot) - 1U)

static uintptr_t scheduler_hart(void)
{
    uintptr_t hart = platform_hart_id();

    if (hart >= CONFIG_HART_COUNT)
        platform_halt();
    return hart;
}

struct thread *thread_slot_get(uint8_t slot)
{
    uintptr_t hart = scheduler_hart();

    if (!slot)
        return &cpu_self()->sche.init;
    if (slot >= THREAD_SLOT_COUNT)
        return 0;
    return &hart_thread_slots[hart][slot - 1U].thread;
}

uint8_t thread_slot_id(const struct thread *thread)
{
    uintptr_t hart = scheduler_hart();
    const union thread_slot *base = hart_thread_slots[hart];
    uintptr_t address;
    uintptr_t begin;
    uintptr_t offset;

    if (thread == &cpu_self()->sche.init)
        return 0;
    if (!thread)
        return THREAD_SLOT_NONE;
    address = (uintptr_t)thread;
    begin = (uintptr_t)base;
    if (address < begin || address >= (uintptr_t)(base + THREAD_SLOT_COUNT - 1U))
        return THREAD_SLOT_NONE;
    offset = address - begin;
    if (offset % sizeof(*base))
        return THREAD_SLOT_NONE;
    return (uint8_t)(offset / sizeof(*base) + 1U);
}

static uint8_t checked_slot(const struct thread *thread)
{
    uint8_t slot = thread_slot_id(thread);

    if (slot == THREAD_SLOT_NONE || thread_slot_get(slot) != thread)
        platform_halt();
    return slot;
}

static enum slot_state slot_state_get(uint8_t slot)
{
    return (enum slot_state)HART_RUNTIME(scheduler_hart()).slot_state[slot];
}

static void slot_state_set(uint8_t slot, enum slot_state state)
{
    HART_RUNTIME(scheduler_hart()).slot_state[slot] = (uint8_t)state;
}

void thread_list_init(struct thread_list *list)
{
    if (!list)
        return;
    list->head = THREAD_SLOT_NONE;
    list->tail = THREAD_SLOT_NONE;
}

static int thread_list_empty(const struct thread_list *list)
{
    return !list || list->head == THREAD_SLOT_NONE;
}

static void thread_node_init(struct thread *thread)
{
    thread->node.prev = THREAD_SLOT_UNLINKED;
    thread->node.next = THREAD_SLOT_UNLINKED;
}

static void thread_list_add_tail(struct thread_list *list,
                                 struct thread *thread)
{
    uintptr_t hart = scheduler_hart();
    uint8_t slot = checked_slot(thread);
    uint8_t tail;

    if (!list || HART_RUNTIME(hart).list_owner[slot] ||
        thread->node.prev != THREAD_SLOT_UNLINKED ||
        thread->node.next != THREAD_SLOT_UNLINKED)
        platform_halt();
    tail = list->tail;
    thread->node.prev = tail;
    thread->node.next = THREAD_SLOT_NONE;
    if (tail == THREAD_SLOT_NONE) {
        if (list->head != THREAD_SLOT_NONE)
            platform_halt();
        list->head = slot;
    } else {
        struct thread *previous = thread_slot_get(tail);

        if (!previous || previous->node.next != THREAD_SLOT_NONE)
            platform_halt();
        previous->node.next = slot;
    }
    list->tail = slot;
    HART_RUNTIME(hart).list_owner[slot] = list;
}

static void thread_list_remove(struct thread *thread)
{
    uintptr_t hart = scheduler_hart();
    uint8_t slot = checked_slot(thread);
    struct thread_list *list = HART_RUNTIME(hart).list_owner[slot];
    uint8_t previous = thread->node.prev;
    uint8_t next = thread->node.next;

    if (!list || previous == THREAD_SLOT_UNLINKED ||
        next == THREAD_SLOT_UNLINKED)
        platform_halt();
    if (previous == THREAD_SLOT_NONE) {
        if (list->head != slot)
            platform_halt();
        list->head = next;
    } else {
        struct thread *node = thread_slot_get(previous);

        if (!node || node->node.next != slot)
            platform_halt();
        node->node.next = next;
    }
    if (next == THREAD_SLOT_NONE) {
        if (list->tail != slot)
            platform_halt();
        list->tail = previous;
    } else {
        struct thread *node = thread_slot_get(next);

        if (!node || node->node.prev != slot)
            platform_halt();
        node->node.prev = previous;
    }
    thread_node_init(thread);
    HART_RUNTIME(hart).list_owner[slot] = 0;
}

static struct task_proc *task_proc_for(struct thread *thread)
{
    size_t depth;

    if (!thread || (thread->type != THREAD_TYPE_TASK_PROC &&
                    thread->type != THREAD_TYPE_TASK_THREAD))
        return 0;
    for (depth = 0; depth < CONFIG_PROC_SLOT_COUNT; ++depth) {
        if (thread->type == THREAD_TYPE_TASK_PROC) {
            if (thread->parent != THREAD_SLOT_NONE)
                return 0;
            return (struct task_proc *)thread;
        }
        if (thread->parent == THREAD_SLOT_NONE ||
            thread->parent > TASK_SLOT_LAST)
            return 0;
        thread = thread_slot_get(thread->parent);
        if (!thread)
            return 0;
    }
    return 0;
}

struct thread *thread_self(void)
{
    return current;
}

struct task_proc *task_proc_self(void)
{
    return task_proc_for(current);
}

const struct task_descriptor *task_descriptor_self(void)
{
    struct task_proc *task = task_proc_self();

    return task ? &task->descriptor : 0;
}

void *task_shared_base(const struct task_proc *task)
{
    if (!task || !task->descriptor.shared_size)
        return 0;
    return (void *)((uintptr_t)CONFIG_COHERENT_BASE +
                    task->descriptor.shared_offset);
}

size_t task_shared_size(const struct task_proc *task)
{
    return task ? task->descriptor.shared_size : 0;
}

size_t task_child_stack_size(const struct task_proc *task)
{
    return task ? (size_t)task->descriptor.child_stack_size_m1 + 1U : 0;
}

size_t task_uram_size(const struct task_proc *task)
{
    struct dayan_context_fields fields;

    if (!task)
        return 0;
    fields = dayan_context_unpack(task->thread.tensor_context);
    return (size_t)(fields.uram_hi - fields.uram_lo) *
           DAYAN_CONTEXT_URAM_GRANULE;
}

void *task_uram_base(const struct task_proc *task)
{
    struct dayan_context_fields fields;

    if (!task || !task_uram_size(task))
        return 0;
    fields = dayan_context_unpack(task->thread.tensor_context);
    return (void *)((uintptr_t)CONFIG_URAM_BASE +
                    scheduler_hart() * (uintptr_t)CONFIG_URAM_STRIDE +
                    fields.uram_lo * DAYAN_CONTEXT_URAM_GRANULE);
}

static void thread_state_set(struct thread *thread, uint8_t state)
{
    thread->state = state;
    if (__builtin_expect(cpu_self()->sche.profile_enabled, 0))
        profile_record_thread_state(thread);
}

static int thread_name_copy(char destination[THREAD_NAME_SIZE],
                            const char *source)
{
    size_t length;

    if (!source || !source[0])
        return -EINVAL;
    for (length = 0; length < THREAD_NAME_SIZE; ++length) {
        destination[length] = source[length];
        if (!source[length]) {
            memset(destination + length + 1, 0,
                   THREAD_NAME_SIZE - length - 1);
            return 0;
        }
    }
    memset(destination, 0, THREAD_NAME_SIZE);
    return -ENAMETOOLONG;
}

static void slot_runtime_reset(uint8_t slot)
{
    uintptr_t hart = scheduler_hart();
    struct thread_runtime *runtime = &HART_RUNTIME(hart);

    runtime->list_owner[slot] = 0;
    if (slot)
        runtime->system_wait[WAIT_INDEX(slot)] = 0;
    if (slot >= TASK_SLOT_FIRST && slot <= TASK_SLOT_LAST) {
        uint8_t index = TASK_INDEX(slot);

        runtime->task_cleanup_arg[index] = 0;
        runtime->result[index] = 0;
        runtime->output_size[index] = 0;
        thread_list_init(&runtime->wait_children[index]);
    }
}

static void thread_submit(struct thread *thread, uint64_t wait_free,
                          uint64_t wait_ready, int8_t qos)
{
    if (!thread || thread->scheduled || qos < -4 || qos > 3)
        platform_halt();
    thread->scheduled = 1;
    resource_scheduler_submit(wait_free, wait_ready, qos,
                              (uintptr_t)thread);
}

static int thread_start_common(struct thread *thread, uint8_t slot,
                               uint8_t type, const char *name,
                               thread_fn entry, void *stack,
                               size_t stack_size, void *shared_base,
                               size_t argc, const uintptr_t *args,
                               int schedule)
{
    uintptr_t hart = scheduler_hart();
    struct proc_frame *frame;
    uintptr_t base;
    uintptr_t top;
    uint8_t generation;
    size_t i;
    int status;

    if (!thread || thread != thread_slot_get(slot) || !slot || !entry ||
        !stack || argc > PROC_MAX_ARGS || (argc && !args))
        return -EINVAL;
    if ((type == THREAD_TYPE_SYSTEM &&
         (slot < SYSTEM_SLOT_FIRST || slot_state_get(slot) != SLOT_SYSTEM)) ||
        (type == THREAD_TYPE_TASK_PROC &&
         (slot < TASK_SLOT_FIRST || slot > TASK_SLOT_LAST ||
          slot_state_get(slot) != SLOT_ROOT)) ||
        (type == THREAD_TYPE_TASK_THREAD &&
         (slot < TASK_SLOT_FIRST || slot > TASK_SLOT_LAST ||
          slot_state_get(slot) != SLOT_RESERVED)) ||
        (type != THREAD_TYPE_SYSTEM && type != THREAD_TYPE_TASK_PROC &&
         type != THREAD_TYPE_TASK_THREAD))
        return -EPERM;

    base = (uintptr_t)stack;
    if (stack_size > UINTPTR_MAX - base)
        return -EINVAL;
    top = (base + stack_size) & ~(uintptr_t)0xf;
    if (top < base || top - base < sizeof(*frame))
        return -EINVAL;

    memset(thread, 0, sizeof(*thread));
    status = thread_name_copy(thread->name, name);
    if (status < 0)
        return status;
    generation = (uint8_t)(HART_RUNTIME(hart).generation[slot] + 1U);
    if (!generation)
        generation = 1;
    HART_RUNTIME(hart).generation[slot] = generation;
    thread->type = type;
    thread->state = THREAD_WAIT_START;
    thread->id = (uint16_t)(((uint16_t)generation << THREAD_ID_SLOT_BITS) |
                            slot);
    thread->parent = THREAD_SLOT_NONE;
    thread_node_init(thread);

    frame = (struct proc_frame *)(top - sizeof(*frame));
    frame->entry = (uintptr_t)entry;
    frame->ra = (uintptr_t)thread_exit;
    for (i = 0; i < argc; ++i)
        frame->args[i] = args[i];
    frame->shared = (uintptr_t)shared_base;
    frame->reserved = 0;
    thread->sp = (uintptr_t)frame;
    thread->pc = (uintptr_t)proc_bootstrap + PROC_BOOTSTRAP_PC_OFFSET(argc);

    slot_runtime_reset(slot);
    if (type == THREAD_TYPE_TASK_THREAD)
        slot_state_set(slot, SLOT_CHILD);
    if (cpu_self()->sche.profile_enabled)
        profile_record_thread_start(thread);
    if (slot == RPC_PROFILE_SLOT_RPC_RECEIVE) {
        thread_state_set(thread, THREAD_WAIT_POLL);
    } else if (schedule) {
        thread_state_set(thread, THREAD_READY);
        thread_submit(thread, 0, 0, type == THREAD_TYPE_SYSTEM ? 3 : 0);
    }
    return 0;
}

int thread_start(struct thread *thread, uint8_t slot, uint8_t type,
                 const char *name, thread_fn entry,
                 void *stack, size_t stack_size, void *shared_base,
                 size_t argc, const uintptr_t *args)
{
    return thread_start_common(thread, slot, type, name, entry, stack,
                               stack_size, shared_base, argc, args, 1);
}

int task_proc_set_cleanup_arg(struct task_proc *task, void *arg)
{
    uintptr_t hart = scheduler_hart();
    uint8_t slot;

    if (!task || !arg)
        return -EINVAL;
    slot = thread_slot_id(&task->thread);
    if (slot < TASK_SLOT_FIRST || slot > TASK_SLOT_LAST ||
        task->thread.type != THREAD_TYPE_TASK_PROC)
        return -EINVAL;
    if ((task->thread.state != THREAD_READY &&
         task->thread.state != THREAD_WAIT_START) ||
        HART_RUNTIME(hart).task_cleanup_arg[TASK_INDEX(slot)])
        return -EBUSY;
    HART_RUNTIME(hart).task_cleanup_arg[TASK_INDEX(slot)] = arg;
    return 0;
}

int task_proc_slot_acquire(struct task_proc **task_out)
{
    uintptr_t hart = scheduler_hart();
    uint8_t slot;

    if (!task_out)
        return -EINVAL;
    *task_out = 0;
    for (slot = TASK_SLOT_FIRST; slot <= TASK_SLOT_LAST; ++slot) {
        if (slot_state_get(slot) != SLOT_FREE)
            continue;
        memset(&hart_thread_slots[hart][slot - 1U], 0,
               sizeof(hart_thread_slots[hart][slot - 1U]));
        slot_runtime_reset(slot);
        HART_RUNTIME(hart).slot_owner[TASK_INDEX(slot)] = slot;
        slot_state_set(slot, SLOT_ROOT);
        *task_out = &hart_thread_slots[hart][slot - 1U].task_proc;
        return 0;
    }
    return -ENOMEM;
}

static int task_descriptor_valid(const struct task_descriptor *descriptor)
{
    size_t stack_size;
    size_t axis;

    if (!descriptor)
        return 0;
    stack_size = (size_t)descriptor->child_stack_size_m1 + 1U;
    if (descriptor->max_child_threads > DAYAN_KERNEL_MAX_CHILD_THREADS ||
        stack_size < sizeof(struct proc_frame) ||
        stack_size > CONFIG_PROC_STACK_SIZE ||
        descriptor->child_printf_size > CONFIG_PROC_PRINTF_SIZE ||
        (descriptor->task_mode != 1U && descriptor->task_mode != 4U) ||
        (descriptor->task_mode == 4U && descriptor->task_dim[0] != 4U) ||
        (!descriptor->shared_size && descriptor->shared_offset) ||
        descriptor->shared_offset > CONFIG_COHERENT_SIZE ||
        descriptor->shared_size >
            CONFIG_COHERENT_SIZE - descriptor->shared_offset)
        return 0;
    for (axis = 0; axis < 3; ++axis) {
        if (!descriptor->task_dim[axis] ||
            descriptor->task_id[axis] >= descriptor->task_dim[axis])
            return 0;
    }
    return 1;
}

int task_proc_set_descriptor(struct task_proc *task,
                             const struct task_descriptor *descriptor,
                             uint64_t tensor_context,
                             uint32_t resource_context)
{
    struct dayan_context_fields fields = dayan_context_unpack(tensor_context);
    struct dayan_resource_context_fields resource_fields =
        dayan_resource_context_unpack(resource_context);
    uint8_t slot;

    if (!task || !task_descriptor_valid(descriptor) ||
        (tensor_context >> 56) || !dayan_context_fields_valid(&fields) ||
        (resource_context >> 22) ||
        !dayan_resource_context_fields_valid(&resource_fields) ||
        (uint64_t)fields.uram_hi * DAYAN_CONTEXT_URAM_GRANULE >
            CONFIG_URAM_SIZE ||
        (fields.uram_lo == fields.uram_hi && fields.uram_lo))
        return -EINVAL;
    slot = thread_slot_id(&task->thread);
    if (slot == THREAD_SLOT_NONE || slot_state_get(slot) != SLOT_ROOT ||
        task->thread.type)
        return -EBUSY;
    memcpy(&task->descriptor, descriptor, sizeof(task->descriptor));
    task->thread.tensor_context = tensor_context;
    task->thread.resource_context = resource_context;
    return 0;
}

int task_proc_reserve(struct task_proc *task)
{
    uintptr_t hart = scheduler_hart();
    uint8_t root;
    uint8_t slot;
    uint32_t available = 0;
    uint32_t remaining;

    if (!task)
        return -EINVAL;
    root = thread_slot_id(&task->thread);
    if (root == THREAD_SLOT_NONE || slot_state_get(root) != SLOT_ROOT ||
        !task_descriptor_valid(&task->descriptor))
        return -EPERM;
    for (slot = TASK_SLOT_FIRST; slot <= TASK_SLOT_LAST; ++slot)
        available += slot_state_get(slot) == SLOT_FREE;
    remaining = task->descriptor.max_child_threads;
    if (available < remaining)
        return -ENOMEM;
    for (slot = TASK_SLOT_FIRST; slot <= TASK_SLOT_LAST && remaining; ++slot) {
        if (slot_state_get(slot) != SLOT_FREE)
            continue;
        slot_state_set(slot, SLOT_RESERVED);
        HART_RUNTIME(hart).slot_owner[TASK_INDEX(slot)] = root;
        --remaining;
    }
    return 0;
}

void task_proc_unreserve(struct task_proc *task)
{
    uintptr_t hart = scheduler_hart();
    uint8_t root;
    uint8_t slot;

    if (!task)
        return;
    root = checked_slot(&task->thread);
    for (slot = TASK_SLOT_FIRST; slot <= TASK_SLOT_LAST; ++slot) {
        if (slot == root ||
            HART_RUNTIME(hart).slot_owner[TASK_INDEX(slot)] != root)
            continue;
        if (slot_state_get(slot) != SLOT_RESERVED)
            platform_halt();
        memset(&hart_thread_slots[hart][slot - 1U], 0,
               sizeof(hart_thread_slots[hart][slot - 1U]));
        slot_runtime_reset(slot);
        HART_RUNTIME(hart).slot_owner[TASK_INDEX(slot)] = THREAD_SLOT_NONE;
        slot_state_set(slot, SLOT_FREE);
    }
}

static void task_root_release(struct task_proc *task)
{
    uintptr_t hart = scheduler_hart();
    uint8_t root = checked_slot(&task->thread);

    memset(&hart_thread_slots[hart][root - 1U], 0,
           sizeof(hart_thread_slots[hart][root - 1U]));
    slot_runtime_reset(root);
    HART_RUNTIME(hart).slot_owner[TASK_INDEX(root)] = THREAD_SLOT_NONE;
    slot_state_set(root, SLOT_FREE);
}

void task_proc_slot_release(struct task_proc *task)
{
    uintptr_t hart = scheduler_hart();
    uint8_t root;
    uint8_t slot;

    if (!task)
        platform_halt();
    root = checked_slot(&task->thread);
    if (root < TASK_SLOT_FIRST || root > TASK_SLOT_LAST)
        platform_halt();
    for (slot = TASK_SLOT_FIRST; slot <= TASK_SLOT_LAST; ++slot) {
        if (slot != root &&
            HART_RUNTIME(hart).slot_owner[TASK_INDEX(slot)] == root)
            platform_halt();
    }
    task_root_release(task);
}

int task_proc_start(struct task_proc *task, const char *name,
                    thread_fn entry, size_t argc, const uintptr_t *args)
{
    uintptr_t hart = scheduler_hart();
    uint64_t tensor_context;
    uint32_t resource_context;
    uint8_t slot;
    int status;

    if (!task || !task_descriptor_valid(&task->descriptor))
        return -EINVAL;
    slot = thread_slot_id(&task->thread);
    if (slot == THREAD_SLOT_NONE || slot_state_get(slot) != SLOT_ROOT)
        return -EPERM;
    tensor_context = task->thread.tensor_context;
    resource_context = task->thread.resource_context;
    status = thread_start_common(
        &task->thread, slot, THREAD_TYPE_TASK_PROC, name, entry,
        hart_thread_stacks[hart][slot - TASK_SLOT_FIRST],
        task_child_stack_size(task), task_shared_base(task), argc, args, 0);
    task->thread.tensor_context = tensor_context;
    task->thread.resource_context = resource_context;
    if (status < 0)
        return status;
    return 0;
}

int task_proc_schedule(struct task_proc *task)
{
    uint8_t slot;

    if (!task)
        return -EINVAL;
    slot = thread_slot_id(&task->thread);
    if (slot < TASK_SLOT_FIRST || slot > TASK_SLOT_LAST ||
        slot_state_get(slot) != SLOT_ROOT ||
        task->thread.type != THREAD_TYPE_TASK_PROC)
        return -EPERM;
    if (task->thread.state != THREAD_WAIT_START || task->thread.scheduled)
        return -EBUSY;
    thread_state_set(&task->thread, THREAD_READY);
    thread_submit(&task->thread, 0, 0, 0);
    return 0;
}

static int child_tcsr_reserve(uint8_t *used, struct dayan_tcsr_range parent,
                              uint8_t count, struct dayan_tcsr_range *child)
{
    uint8_t base;

    child->base = 0;
    child->count = 0;
    if (!count)
        return 0;
    if (count > parent.count)
        return -ENOMEM;
    for (base = 0; base <= parent.count - count; ++base) {
        uint8_t mask = (uint8_t)(((1U << count) - 1U) << base);

        if (*used & mask)
            continue;
        *used |= mask;
        child->base = (uint8_t)(parent.base + base);
        child->count = count;
        return 0;
    }
    return -ENOMEM;
}

struct task_tcsr_usage {
    uint8_t lsu;
    uint8_t mat;
    uint8_t vec;
    uint8_t macro;
};

static uint8_t child_tcsr_mask(struct dayan_tcsr_range parent,
                               struct dayan_tcsr_range child)
{
    uint8_t logical_base;

    if (!child.count)
        return 0;
    if (child.base < parent.base ||
        child.base + child.count > parent.base + parent.count)
        platform_halt();
    logical_base = (uint8_t)(child.base - parent.base);
    return (uint8_t)(((1U << child.count) - 1U) << logical_base);
}

static struct task_tcsr_usage task_tcsr_usage(struct task_proc *task)
{
    uintptr_t hart = scheduler_hart();
    struct dayan_context_fields root_context =
        dayan_context_unpack(task->thread.tensor_context);
    struct task_tcsr_usage usage = { 0, 0, 0, 0 };
    uint8_t root = checked_slot(&task->thread);
    uint8_t slot;

    for (slot = TASK_SLOT_FIRST; slot <= TASK_SLOT_LAST; ++slot) {
        struct dayan_context_fields child_context;

        if (HART_RUNTIME(hart).slot_owner[TASK_INDEX(slot)] != root ||
            slot_state_get(slot) != SLOT_CHILD)
            continue;
        child_context = dayan_context_unpack(
            thread_slot_get(slot)->tensor_context);
        usage.lsu |= child_tcsr_mask(root_context.lsu, child_context.lsu);
        usage.mat |= child_tcsr_mask(root_context.mat, child_context.mat);
        usage.vec |= child_tcsr_mask(root_context.vec, child_context.vec);
        usage.macro |= child_tcsr_mask(root_context.macro,
                                       child_context.macro);
    }
    return usage;
}

static struct thread *reserved_child_acquire(struct task_proc *task)
{
    uintptr_t hart = scheduler_hart();
    uint8_t root = checked_slot(&task->thread);
    uint8_t slot;

    for (slot = TASK_SLOT_FIRST; slot <= TASK_SLOT_LAST; ++slot) {
        if (slot_state_get(slot) != SLOT_RESERVED ||
            HART_RUNTIME(hart).slot_owner[TASK_INDEX(slot)] != root)
            continue;
        memset(&hart_thread_slots[hart][slot - 1U], 0,
               sizeof(struct thread));
        return thread_slot_get(slot);
    }
    return 0;
}

static uint32_t task_active_children(uint8_t root)
{
    uintptr_t hart = scheduler_hart();
    uint8_t slot;
    uint32_t count = 0;

    for (slot = TASK_SLOT_FIRST; slot <= TASK_SLOT_LAST; ++slot) {
        enum slot_state state;

        if (HART_RUNTIME(hart).slot_owner[TASK_INDEX(slot)] != root)
            continue;
        state = slot_state_get(slot);
        count += state == SLOT_CHILD;
    }
    return count;
}

static void child_slot_release(struct thread *child)
{
    uintptr_t hart = scheduler_hart();
    uint8_t slot = checked_slot(child);

    if (slot_state_get(slot) != SLOT_ZOMBIE)
        platform_halt();
    memset(&hart_thread_slots[hart][slot - 1U], 0,
           sizeof(struct thread));
    slot_runtime_reset(slot);
    slot_state_set(slot, SLOT_RESERVED);
}

static void child_thread_cleanup(struct thread *thread)
{
    uintptr_t hart = scheduler_hart();
    uint8_t slot = checked_slot(thread);
    uint8_t parent = thread->parent;

    if (slot_state_get(slot) != SLOT_CHILD || parent == THREAD_SLOT_NONE)
        platform_halt();
    slot_state_set(slot, SLOT_ZOMBIE);
    if (HART_RUNTIME(hart).result[TASK_INDEX(slot)] < 0 &&
        HART_RUNTIME(hart).result[TASK_INDEX(parent)] == 0)
        HART_RUNTIME(hart).result[TASK_INDEX(parent)] =
            HART_RUNTIME(hart).result[TASK_INDEX(slot)];
    (void)thread_event_wake_all(
        &HART_RUNTIME(hart).wait_children[TASK_INDEX(parent)]);
}

int thread_create_child(struct thread *parent, const char *name,
                        thread_fn entry, void *argument,
                        const struct dayan_thread_resources *resources,
                        uint32_t *thread_id_out)
{
    uintptr_t hart = scheduler_hart();
    struct task_proc *task;
    struct thread *child;
    struct dayan_context_fields root_context;
    struct dayan_context_fields child_context;
    struct dayan_resource_context_fields root_resources;
    struct dayan_resource_context_fields child_resources;
    struct task_tcsr_usage usage;
    dayan_thread_resources_t empty_resources;
    uintptr_t args[1];
    uint8_t root;
    uint8_t child_slot;
    int status;

    if (!parent || parent != current || !name || !entry || !thread_id_out)
        return -EINVAL;
    task = task_proc_for(parent);
    if (!task)
        return -EPERM;
    root = checked_slot(&task->thread);
    if (slot_state_get(root) != SLOT_ROOT)
        return -EPERM;
    if (!resources) {
        memset(&empty_resources, 0, sizeof(empty_resources));
        resources = &empty_resources;
    }
    if (resources->reserved ||
        resources->lsu_tcsr_count > DAYAN_TCSR_LOGICAL_GROUPS ||
        resources->mat_tcsr_count > DAYAN_TCSR_LOGICAL_GROUPS ||
        resources->vec_tcsr_count > DAYAN_TCSR_LOGICAL_GROUPS ||
        resources->macro_tcsr_count > DAYAN_TCSR_LOGICAL_GROUPS ||
        resources->local_resource_count > DAYAN_RESOURCE_LOGICAL_COUNT ||
        resources->shared_resource_count > DAYAN_RESOURCE_LOGICAL_COUNT)
        return -EINVAL;
    if (task_active_children(root) >= task->descriptor.max_child_threads)
        return -ENOMEM;
    child = reserved_child_acquire(task);
    if (!child)
        return -ENOMEM;
    child_slot = checked_slot(child);
    root_context = dayan_context_unpack(task->thread.tensor_context);
    root_resources = dayan_resource_context_unpack(
        task->thread.resource_context);
    usage = task_tcsr_usage(task);
    child_context = root_context;
    child_context.lsu.count = 0;
    child_context.mat.count = 0;
    child_context.vec.count = 0;
    child_context.macro.count = 0;
    child_resources = root_resources;
    child_resources.local.count = resources->local_resource_count;
    child_resources.shared.count = resources->shared_resource_count;
    if (child_resources.local.count > root_resources.local.count ||
        child_resources.shared.count > root_resources.shared.count) {
        memset(child, 0, sizeof(*child));
        return -ENOMEM;
    }
    status = child_tcsr_reserve(&usage.lsu, root_context.lsu,
                                resources->lsu_tcsr_count,
                                &child_context.lsu);
    if (!status)
        status = child_tcsr_reserve(&usage.mat, root_context.mat,
                                    resources->mat_tcsr_count,
                                    &child_context.mat);
    if (!status)
        status = child_tcsr_reserve(&usage.vec, root_context.vec,
                                    resources->vec_tcsr_count,
                                    &child_context.vec);
    if (!status)
        status = child_tcsr_reserve(&usage.macro, root_context.macro,
                                    resources->macro_tcsr_count,
                                    &child_context.macro);
    if (status < 0) {
        memset(child, 0, sizeof(*child));
        return status;
    }
    args[0] = (uintptr_t)argument;
    status = thread_start(child, child_slot, THREAD_TYPE_TASK_THREAD,
                          name, entry,
                          hart_thread_stacks[hart][child_slot - TASK_SLOT_FIRST],
                          task_child_stack_size(task), task_shared_base(task),
                          1, args);
    if (status < 0) {
        memset(child, 0, sizeof(*child));
        return status;
    }
    child->parent = checked_slot(parent);
    child->tensor_context = dayan_context_pack(&child_context);
    child->resource_context = dayan_resource_context_pack(&child_resources);
    *thread_id_out = child->id;
    return 0;
}

static struct thread *find_direct_child(struct thread *parent,
                                        uint16_t thread_id)
{
    uintptr_t hart = scheduler_hart();
    struct task_proc *task = task_proc_for(parent);
    uint8_t root;
    uint8_t parent_slot;
    uint8_t slot;

    if (!task)
        return 0;
    root = checked_slot(&task->thread);
    parent_slot = checked_slot(parent);
    for (slot = TASK_SLOT_FIRST; slot <= TASK_SLOT_LAST; ++slot) {
        enum slot_state state;
        struct thread *child;

        if (HART_RUNTIME(hart).slot_owner[TASK_INDEX(slot)] != root)
            continue;
        state = slot_state_get(slot);
        if (state != SLOT_CHILD && state != SLOT_ZOMBIE)
            continue;
        child = thread_slot_get(slot);
        if (child->parent == parent_slot && child->id == thread_id)
            return child;
    }
    return 0;
}

static uint32_t direct_live_children(struct thread *parent)
{
    uintptr_t hart = scheduler_hart();
    struct task_proc *task = task_proc_for(parent);
    uint8_t root;
    uint8_t parent_slot;
    uint8_t slot;
    uint32_t count = 0;

    if (!task)
        return 0;
    root = checked_slot(&task->thread);
    parent_slot = checked_slot(parent);
    for (slot = TASK_SLOT_FIRST; slot <= TASK_SLOT_LAST; ++slot) {
        enum slot_state state;
        struct thread *child;

        if (HART_RUNTIME(hart).slot_owner[TASK_INDEX(slot)] != root)
            continue;
        state = slot_state_get(slot);
        if (state != SLOT_CHILD)
            continue;
        child = thread_slot_get(slot);
        count += child->parent == parent_slot;
    }
    return count;
}

int thread_join(uint32_t thread_id, int64_t *result)
{
    uintptr_t hart = scheduler_hart();
    struct thread *parent = current;
    struct thread *child;
    uint8_t parent_slot;

    if (!task_proc_for(parent))
        return -EPERM;
    if (!thread_id || thread_id > UINT16_MAX || thread_id == parent->id)
        return -ECHILD;
    child = find_direct_child(parent, (uint16_t)thread_id);
    if (!child)
        return -ECHILD;
    parent_slot = checked_slot(parent);
    while (slot_state_get(checked_slot(child)) != SLOT_ZOMBIE) {
        thread_wait_event(
            &HART_RUNTIME(hart).wait_children[TASK_INDEX(parent_slot)]);
        child = find_direct_child(parent, (uint16_t)thread_id);
        if (!child)
            return -ECHILD;
    }
    if (result)
        *result = HART_RUNTIME(hart).result[
            TASK_INDEX(checked_slot(child))];
    child_slot_release(child);
    return 0;
}

int thread_join_all(void)
{
    uintptr_t hart = scheduler_hart();
    struct thread *parent = current;
    struct task_proc *task = task_proc_for(parent);
    uint8_t root;
    uint8_t parent_slot;
    uint8_t slot;

    if (!task)
        return -EPERM;
    root = checked_slot(&task->thread);
    parent_slot = checked_slot(parent);
    while (direct_live_children(parent))
        thread_wait_event(
            &HART_RUNTIME(hart).wait_children[TASK_INDEX(parent_slot)]);
    for (slot = TASK_SLOT_FIRST; slot <= TASK_SLOT_LAST; ++slot) {
        struct thread *child;

        if (HART_RUNTIME(hart).slot_owner[TASK_INDEX(slot)] != root ||
            slot_state_get(slot) != SLOT_ZOMBIE)
            continue;
        child = thread_slot_get(slot);
        if (child->parent == parent_slot)
            child_slot_release(child);
    }
    return 0;
}

int64_t thread_result(const struct thread *thread)
{
    uint8_t slot = thread_slot_id(thread);

    if (slot < TASK_SLOT_FIRST || slot > TASK_SLOT_LAST)
        return 0;
    return HART_RUNTIME(scheduler_hart()).result[TASK_INDEX(slot)];
}

void *thread_system_wait(const struct thread *thread)
{
    uint8_t slot = thread_slot_id(thread);

    if (!slot || slot == THREAD_SLOT_NONE)
        return 0;
    return HART_RUNTIME(scheduler_hart()).system_wait[WAIT_INDEX(slot)];
}

void thread_system_wait_set(struct thread *thread, void *wait)
{
    uint8_t slot = checked_slot(thread);

    if (!slot)
        platform_halt();
    HART_RUNTIME(scheduler_hart()).system_wait[WAIT_INDEX(slot)] = wait;
}

int thread_output_enabled(const struct thread *thread)
{
    struct task_proc *task = task_proc_for((struct thread *)thread);

    return task && task->descriptor.child_printf_size;
}

int thread_output_flush(struct thread *thread)
{
    uintptr_t hart = scheduler_hart();
    struct task_proc *task;
    uint16_t *output_size;
    uint16_t capacity;
    uint8_t slot;

    if (!thread)
        return -1;
    slot = thread_slot_id(thread);
    task = task_proc_for(thread);
    if (slot < TASK_SLOT_FIRST || slot > TASK_SLOT_LAST || !task)
        return -1;
    output_size = &HART_RUNTIME(hart).output_size[TASK_INDEX(slot)];
    capacity = task->descriptor.child_printf_size;
    if (*output_size > capacity)
        return -1;
    if (!*output_size)
        return 0;
    if (!capacity ||
        rpc_write_proc(task->thread.id,
                       hart_thread_printf[hart][TASK_INDEX(slot)],
                       *output_size) < 0)
        return -1;
    *output_size = 0;
    return 0;
}

int thread_output_put(struct thread *thread, char ch)
{
    uintptr_t hart = scheduler_hart();
    struct task_proc *task;
    uint16_t *output_size;
    uint16_t capacity;
    uint8_t slot;

    if (!thread || !thread_output_enabled(thread))
        return -1;
    slot = checked_slot(thread);
    task = task_proc_for(thread);
    if (!task || slot < TASK_SLOT_FIRST || slot > TASK_SLOT_LAST)
        return -1;
    output_size = &HART_RUNTIME(hart).output_size[TASK_INDEX(slot)];
    capacity = task->descriptor.child_printf_size;
    if (*output_size > capacity)
        return -1;
    if (*output_size == capacity &&
        thread_output_flush(thread) < 0)
        return -1;
    hart_thread_printf[hart][TASK_INDEX(slot)][(*output_size)++] = ch;
    if (*output_size == capacity &&
        thread_output_flush(thread) < 0)
        return -1;
    return 0;
}

static void snapshot_thread(struct rpc_proc_info *record,
                            struct thread *thread, uint32_t flags,
                            uint32_t slot_index)
{
    struct task_proc *task = task_proc_for(thread);
    struct thread *parent = thread->parent == THREAD_SLOT_NONE ? 0 :
                            thread_slot_get(thread->parent);

    memset(record, 0, sizeof(*record));
    record->proc_id = thread->id;
    record->parent_proc_id = parent ? parent->id : 0;
    record->state = thread->state;
    record->flags = flags;
    record->live_children = direct_live_children(thread);
    record->slot_index = slot_index;
    record->result = thread_result(thread);
    record->pc = thread->pc;
    memcpy(record->name, thread->name, sizeof(record->name));
    if (task) {
        record->task_mode = task->descriptor.task_mode;
        record->host_proc_id = task->thread.id;
        memcpy(record->task_id, task->descriptor.task_id,
               sizeof(record->task_id));
        memcpy(record->task_dim, task->descriptor.task_dim,
               sizeof(record->task_dim));
        record->cookie = task->descriptor.cookie;
    }
}

static int snapshot_slot_flags(uint8_t slot, uint32_t *flags_out)
{
    enum slot_state state = slot_state_get(slot);
    uint32_t flags;

    if (state == SLOT_ROOT && thread_slot_get(slot)->type ==
                              THREAD_TYPE_TASK_PROC) {
        flags = RPC_PROC_INFO_HOST_ROOT;
    } else if (state == SLOT_CHILD) {
        flags = RPC_PROC_INFO_CHILD;
    } else if (state == SLOT_ZOMBIE) {
        flags = RPC_PROC_INFO_CHILD | RPC_PROC_INFO_ZOMBIE;
    } else {
        return 0;
    }
    if (thread_slot_get(slot)->state == THREAD_KILL_WAIT_PRODUCE)
        flags |= RPC_PROC_INFO_KILL_WAIT_PRODUCE;
    if (flags_out)
        *flags_out = flags;
    return 1;
}

int proc_snapshot(struct rpc_proc_ps_snapshot *snapshot)
{
    uint32_t count = 0;
    uint32_t free_slots = 0;
    uint8_t slot;

    if (!snapshot || snapshot->flags ||
        snapshot->capacity != RPC_PROC_PS_MAX_RECORDS || snapshot->reserved)
        return -EINVAL;
    memset(snapshot->records, 0, sizeof(snapshot->records));
    for (slot = SYSTEM_SLOT_FIRST; slot < THREAD_SLOT_COUNT; ++slot) {
        struct thread *thread = thread_slot_get(slot);

        if (thread->id)
            snapshot_thread(&snapshot->records[count++], thread,
                            RPC_PROC_INFO_SYSTEM, UINT32_MAX);
    }
    for (slot = TASK_SLOT_FIRST; slot <= TASK_SLOT_LAST; ++slot) {
        uint32_t flags;

        free_slots += slot_state_get(slot) == SLOT_FREE;
        if (!snapshot_slot_flags(slot, &flags))
            continue;
        if (count >= RPC_PROC_PS_MAX_RECORDS)
            platform_halt();
        snapshot_thread(&snapshot->records[count++], thread_slot_get(slot),
                        flags, slot - TASK_SLOT_FIRST);
    }
    snapshot->hart_id = (uint32_t)scheduler_hart();
    snapshot->count = count;
    snapshot->global_free_slots = free_slots;
    snapshot->reserved_slots = CONFIG_PROC_SLOT_COUNT - free_slots;
    snapshot->status = 0;
    return 0;
}

void proc_profile_snapshot(void)
{
    uint8_t slot;

    for (slot = SYSTEM_SLOT_FIRST; slot < THREAD_SLOT_COUNT; ++slot) {
        struct thread *thread = thread_slot_get(slot);

        if (thread->id)
            profile_record_thread_start(thread);
    }
    for (slot = TASK_SLOT_FIRST; slot <= TASK_SLOT_LAST; ++slot) {
        if (snapshot_slot_flags(slot, 0))
            profile_record_thread_start(thread_slot_get(slot));
    }
}

static void kill_unlink_thread(struct thread *thread)
{
    uintptr_t hart = scheduler_hart();
    uint8_t slot = checked_slot(thread);

    switch (thread->state) {
    case THREAD_READY:
    case THREAD_WAIT_PRODUCE:
        if (!thread->scheduled || HART_RUNTIME(hart).list_owner[slot])
            platform_halt();
        break;
    case THREAD_WAIT_EVENT:
        if (thread->scheduled || !HART_RUNTIME(hart).list_owner[slot])
            platform_halt();
        thread_list_remove(thread);
        break;
    case THREAD_WAIT_POLL:
    case THREAD_WAIT_EXIT:
    case THREAD_WAIT_START:
        if (thread->scheduled || HART_RUNTIME(hart).list_owner[slot])
            platform_halt();
        break;
    default:
        platform_halt();
    }
}

static void kill_mark_thread(struct thread *thread, int64_t status)
{
    uintptr_t hart = scheduler_hart();
    uint8_t slot = checked_slot(thread);
    enum slot_state state = slot_state_get(slot);
    int waiting_token;

    HART_RUNTIME(hart).result[TASK_INDEX(slot)] = status;
    if (state == SLOT_ZOMBIE)
        return;
    if (state != SLOT_ROOT && state != SLOT_CHILD)
        platform_halt();
    waiting_token = thread->scheduled;
    kill_unlink_thread(thread);
    if (waiting_token) {
        thread_state_set(thread, THREAD_KILL_WAIT_PRODUCE);
    } else {
        thread_state_set(thread, THREAD_WAIT_EXIT);
    }
}

static void kill_drain_system_wait(struct thread *thread)
{
    int status;

    while ((status = rpc_proc_wait_cancel(thread)) > 0)
        thread_yield();
    if (status < 0)
        platform_halt();
}

static void thread_run_cleanup(struct thread *thread)
{
    uintptr_t hart = scheduler_hart();
    uint8_t slot = checked_slot(thread);

    if (thread->type == THREAD_TYPE_TASK_THREAD) {
        child_thread_cleanup(thread);
    } else if (thread->type == THREAD_TYPE_TASK_PROC) {
        void **cleanup_arg =
            &HART_RUNTIME(hart).task_cleanup_arg[TASK_INDEX(slot)];
        void *arg = *cleanup_arg;

        *cleanup_arg = 0;
        if (arg)
            rpc_proc_cleanup(thread, arg);
    }
}

static uint32_t family_pending_produce(uint8_t root)
{
    uintptr_t hart = scheduler_hart();
    uint8_t slot;
    uint32_t count = 0;

    for (slot = TASK_SLOT_FIRST; slot <= TASK_SLOT_LAST; ++slot)
        count += HART_RUNTIME(hart).slot_owner[TASK_INDEX(slot)] == root &&
                 slot_state_get(slot) != SLOT_RESERVED &&
                 thread_slot_get(slot)->state == THREAD_KILL_WAIT_PRODUCE;
    return count;
}

static void killed_child_free(struct thread *thread)
{
    uintptr_t hart = scheduler_hart();
    uint8_t slot = checked_slot(thread);

    if ((slot_state_get(slot) != SLOT_CHILD &&
         slot_state_get(slot) != SLOT_ZOMBIE) ||
        thread->state != THREAD_WAIT_EXIT)
        platform_halt();
    memset(&hart_thread_slots[hart][slot - 1U], 0,
           sizeof(struct thread));
    slot_runtime_reset(slot);
    HART_RUNTIME(hart).slot_owner[TASK_INDEX(slot)] = THREAD_SLOT_NONE;
    slot_state_set(slot, SLOT_FREE);
}

static void killed_family_release(uint8_t root)
{
    uintptr_t hart = scheduler_hart();
    struct task_proc *task = (struct task_proc *)thread_slot_get(root);
    uint8_t slot;

    if (family_pending_produce(root))
        return;
    for (slot = TASK_SLOT_FIRST; slot <= TASK_SLOT_LAST; ++slot) {
        if (slot == root ||
            HART_RUNTIME(hart).slot_owner[TASK_INDEX(slot)] != root)
            continue;
        if (slot_state_get(slot) == SLOT_RESERVED) {
            memset(&hart_thread_slots[hart][slot - 1U], 0,
                   sizeof(hart_thread_slots[hart][slot - 1U]));
            slot_runtime_reset(slot);
            HART_RUNTIME(hart).slot_owner[TASK_INDEX(slot)] = THREAD_SLOT_NONE;
            slot_state_set(slot, SLOT_FREE);
        } else if (slot_state_get(slot) == SLOT_CHILD ||
                   slot_state_get(slot) == SLOT_ZOMBIE) {
            struct thread *thread = thread_slot_get(slot);

            if (thread->state != THREAD_WAIT_EXIT)
                platform_halt();
            killed_child_free(thread);
        } else {
            platform_halt();
        }
    }
    if (slot_state_get(root) != SLOT_ROOT ||
        task->thread.state != THREAD_WAIT_EXIT)
        platform_halt();
    task_root_release(task);
}

int proc_kill_all(struct rpc_proc_kill_result *result)
{
    uintptr_t hart = scheduler_hart();
    uint64_t selector;
    int64_t exit_status;
    uint16_t selected_roots = 0;
    uint8_t root;

    if (!result || (result->flags & ~RPC_PROC_KILL_FAMILY))
        return -EINVAL;
    selector = result->reserved;
    if (!!(result->flags & RPC_PROC_KILL_FAMILY) != !!selector ||
        selector > UINT16_MAX)
        return -EINVAL;
    exit_status = result->flags & RPC_PROC_KILL_FAMILY ? result->status :
                                                        -ECANCELED;
    if (exit_status >= 0)
        return -EINVAL;
    result->hart_id = (uint32_t)hart;
    result->status = 0;
    result->killed_roots = 0;
    result->killed_children = 0;
    result->released_slots = 0;
    result->waiting_produce = 0;

    for (root = TASK_SLOT_FIRST; root <= TASK_SLOT_LAST; ++root) {
        struct task_proc *task;
        uint8_t slot;

        if (slot_state_get(root) != SLOT_ROOT)
            continue;
        task = (struct task_proc *)thread_slot_get(root);
        if (task->thread.type != THREAD_TYPE_TASK_PROC ||
            task->thread.state == THREAD_WAIT_EXIT ||
            task->thread.state == THREAD_KILL_WAIT_PRODUCE ||
            (selector && task->thread.id != (uint16_t)selector))
            continue;
        for (slot = TASK_SLOT_FIRST; slot <= TASK_SLOT_LAST; ++slot) {
            enum slot_state state;

            if (slot == root ||
                HART_RUNTIME(hart).slot_owner[TASK_INDEX(slot)] != root)
                continue;
            state = slot_state_get(slot);
            if (state == SLOT_CHILD || state == SLOT_ZOMBIE) {
                ++result->killed_children;
                kill_mark_thread(thread_slot_get(slot), exit_status);
            } else if (state != SLOT_RESERVED) {
                platform_halt();
            }
        }
        ++result->killed_roots;
        selected_roots |= (uint16_t)(1U << (root - TASK_SLOT_FIRST));
        kill_mark_thread(&task->thread, exit_status);
    }
    if (selector && !result->killed_roots)
        return -ESRCH;

    for (root = TASK_SLOT_FIRST; root <= TASK_SLOT_LAST; ++root) {
        struct task_proc *task;
        uint8_t slot;
        uint32_t before;
        uint32_t waiting_produce;

        if (!(selected_roots &
              (uint16_t)(1U << (root - TASK_SLOT_FIRST))))
            continue;
        task = (struct task_proc *)thread_slot_get(root);
        for (slot = TASK_SLOT_FIRST; slot <= TASK_SLOT_LAST; ++slot) {
            if (HART_RUNTIME(hart).slot_owner[TASK_INDEX(slot)] != root)
                continue;
            if (slot_state_get(slot) != SLOT_RESERVED &&
                thread_slot_get(slot)->id)
                kill_drain_system_wait(thread_slot_get(slot));
        }
        for (slot = TASK_SLOT_FIRST; slot <= TASK_SLOT_LAST; ++slot) {
            if (HART_RUNTIME(hart).slot_owner[TASK_INDEX(slot)] != root)
                continue;
            if (slot_state_get(slot) != SLOT_RESERVED)
                (void)thread_output_flush(thread_slot_get(slot));
        }
        thread_run_cleanup(&task->thread);
        before = CONFIG_PROC_SLOT_COUNT;
        for (slot = TASK_SLOT_FIRST; slot <= TASK_SLOT_LAST; ++slot)
            before -= slot_state_get(slot) == SLOT_FREE;
        waiting_produce = family_pending_produce(root);
        (void)rpc_proc_exit(
            task->thread.id, 0,
            HART_RUNTIME(hart).result[TASK_INDEX(root)],
                            task->descriptor.cookie, RPC_PROC_EXIT_ROOT);
        result->waiting_produce += waiting_produce;
        if (slot_state_get(root) == SLOT_ROOT)
            killed_family_release(root);
        for (slot = TASK_SLOT_FIRST; slot <= TASK_SLOT_LAST; ++slot)
            before -= slot_state_get(slot) != SLOT_FREE;
        result->released_slots += before;
    }
    return 0;
}

static void kill_complete_produce(struct thread *thread)
{
    uintptr_t hart = scheduler_hart();
    uint8_t slot = checked_slot(thread);
    uint8_t root = HART_RUNTIME(hart).slot_owner[TASK_INDEX(slot)];

    if ((slot_state_get(slot) != SLOT_ROOT &&
         slot_state_get(slot) != SLOT_CHILD) ||
        thread->state != THREAD_KILL_WAIT_PRODUCE ||
        root < TASK_SLOT_FIRST || root > TASK_SLOT_LAST)
        platform_halt();
    thread_state_set(thread, THREAD_WAIT_EXIT);
    if (slot != root)
        killed_child_free(thread);
    killed_family_release(root);
}

int thread_wake(struct thread *thread)
{
    uintptr_t hart = scheduler_hart();
    uint8_t slot;

    if (!thread)
        return -EINVAL;
    slot = checked_slot(thread);
    if (thread->state == THREAD_WAIT_EVENT) {
        if (thread->scheduled || !HART_RUNTIME(hart).list_owner[slot])
            platform_halt();
        thread_list_remove(thread);
        thread_state_set(thread, THREAD_READY);
        thread_submit(thread, 0, 0,
                      thread->type == THREAD_TYPE_SYSTEM ? 3 : 0);
        return 0;
    }
    return -EBUSY;
}

size_t thread_event_wake_one(struct thread_list *event)
{
    struct thread *thread;

    if (thread_list_empty(event))
        return 0;
    thread = thread_slot_get(event->head);
    return thread && thread_wake(thread) == 0;
}

size_t thread_event_wake_all(struct thread_list *event)
{
    size_t count = 0;

    while (thread_event_wake_one(event))
        ++count;
    return count;
}

void thread_wait_poll(thread_poll_fn poll, void *arg)
{
    if (!poll)
        platform_halt();
    while (!poll(current, arg))
        thread_yield();
}

void thread_wait_event(struct thread_list *event)
{
    struct thread *thread = current;

    if (!event) {
        thread_yield();
        return;
    }
    if (thread->scheduled)
        platform_halt();
    thread_list_add_tail(event, thread);
    thread_state_set(thread, THREAD_WAIT_EVENT);
    switch_context();
}

void thread_wait_produce(void)
{
    struct thread *thread = current;

    if (thread->scheduled)
        platform_halt();
    thread->scheduled = 1;
    thread_state_set(thread, THREAD_WAIT_PRODUCE);
    switch_context();
}

uint32_t thread_wait_resources(uint64_t wait_free, uint64_t wait_ready,
                               int8_t qos)
{
    struct thread *thread = current;

    thread_state_set(thread, THREAD_READY);
    thread_submit(thread, wait_free, wait_ready, qos);
    switch_context();
    return thread->token_ext;
}

void thread_wait_rpc(void)
{
    struct thread *thread = current;

    if (checked_slot(thread) != RPC_PROFILE_SLOT_RPC_RECEIVE ||
        thread->scheduled)
        platform_halt();
    thread_state_set(thread, THREAD_WAIT_POLL);
    switch_context();
}

void thread_yield(void)
{
    struct thread *thread = current;

    thread_state_set(thread, THREAD_READY);
    thread_submit(thread, 0, 0, 0);
    switch_context();
}

_Noreturn void thread_exit_defer(void *wait)
{
    uintptr_t hart = scheduler_hart();
    struct thread *thread = current;
    uint8_t slot;

    if (!wait || !thread || thread->type != THREAD_TYPE_TASK_PROC ||
        thread->state != THREAD_RUNNING || thread->scheduled ||
        thread_system_wait(thread))
        platform_halt();
    slot = checked_slot(thread);
    if (slot_state_get(slot) != SLOT_ROOT ||
        HART_RUNTIME(hart).list_owner[slot] || thread->reap_deferred)
        platform_halt();
    thread_system_wait_set(thread, wait);
    thread->reap_deferred = 1;
    thread_state_set(thread, THREAD_WAIT_EXIT);
    switch_context();
    platform_halt();
}

void thread_exit_deferred_complete(struct thread *thread, void *wait)
{
    uintptr_t hart = scheduler_hart();
    uint8_t slot;

    if (!thread || thread == current || !wait ||
        thread->type != THREAD_TYPE_TASK_PROC ||
        thread->state != THREAD_WAIT_EXIT || thread->scheduled ||
        !thread->reap_deferred || thread_system_wait(thread) != wait)
        platform_halt();
    slot = checked_slot(thread);
    if (slot_state_get(slot) != SLOT_ROOT ||
        HART_RUNTIME(hart).list_owner[slot])
        platform_halt();
    thread_system_wait_set(thread, 0);
    thread->reap_deferred = 0;
    scheduler_reap_thread(thread);
}

_Noreturn void thread_exit_status(int64_t status)
{
    uintptr_t hart = scheduler_hart();
    struct thread *thread = current;
    struct task_proc *task = task_proc_for(thread);
    uint8_t slot;

    if (!thread || thread->type == THREAD_TYPE_BASE || !task)
        platform_halt();
    slot = checked_slot(thread);
    if (status < 0 || HART_RUNTIME(hart).result[TASK_INDEX(slot)] >= 0)
        HART_RUNTIME(hart).result[TASK_INDEX(slot)] = status;
    if (thread_join_all() < 0 &&
        HART_RUNTIME(hart).result[TASK_INDEX(slot)] >= 0)
        HART_RUNTIME(hart).result[TASK_INDEX(slot)] = -EIO;
    (void)thread_output_flush(thread);
    if (thread->type == THREAD_TYPE_TASK_PROC) {
        int rpc_status;

        thread_run_cleanup(thread);
        rpc_status = rpc_proc_exit_current(
            thread->id, 0,
            HART_RUNTIME(hart).result[TASK_INDEX(slot)],
            task->descriptor.cookie, RPC_PROC_EXIT_ROOT);
        if (rpc_status >= 0)
            platform_halt();
        scheduler_notify();
    } else {
        child_thread_cleanup(thread);
    }
    thread_state_set(thread, THREAD_WAIT_EXIT);
    switch_context();
    platform_halt();
}

_Noreturn void thread_exit(void)
{
    thread_exit_status(0);
}

void scheduler_reap_thread(struct thread *thread)
{
    uint8_t slot = checked_slot(thread);

    if (thread->type == THREAD_TYPE_TASK_PROC) {
        struct task_proc *task = (struct task_proc *)thread;

        task_proc_unreserve(task);
        task_proc_slot_release(task);
    } else if (thread->type == THREAD_TYPE_TASK_THREAD) {
        if (slot_state_get(slot) != SLOT_ZOMBIE)
            platform_halt();
    }
}

void scheduler_kill_token(struct thread *thread)
{
    kill_complete_produce(thread);
}

void scheduler_notify(void)
{
    struct thread *init = &cpu_self()->sche.init;

    if (!init->scheduled && init->state != THREAD_WAIT_EVENT)
        thread_submit(init, 0, 0, 3);
}

void sche_init(void)
{
    uintptr_t hart = scheduler_hart();
    struct sche_state *sche = &cpu_self()->sche;
    struct thread *init = current;
    uint8_t slot;

    init->type = THREAD_TYPE_BASE;
    init->state = THREAD_RUNNING;
    init->id = 0;
    init->parent = THREAD_SLOT_NONE;
    init->pc = 0;
    init->tensor_context = 0;
    init->resource_context = 0;
    init->scheduled = 0;
    init->token_ext = 0;
    thread_node_init(init);
    memset(init->name, 0, sizeof(init->name));
    sche->profile_enabled = 0;
    scheduler_rpc_receive_threads[hart] =
        thread_slot_get(RPC_PROFILE_SLOT_RPC_RECEIVE);
    resource_scheduler_local_init();
    memset(hart_thread_slots[hart], 0, sizeof(hart_thread_slots[hart]));
    for (slot = 0; slot < THREAD_SLOT_COUNT; ++slot) {
        slot_runtime_reset(slot);
        if (slot >= TASK_SLOT_FIRST && slot <= TASK_SLOT_LAST)
            HART_RUNTIME(hart).slot_owner[TASK_INDEX(slot)] = THREAD_SLOT_NONE;
        HART_RUNTIME(hart).generation[slot] = 0;
        slot_state_set(slot, slot >= SYSTEM_SLOT_FIRST ? SLOT_SYSTEM :
                                                        SLOT_FREE);
    }
}
