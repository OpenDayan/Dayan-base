#include <stdarg.h>
#include <stddef.h>
#include <stdint.h>

#include <dayan_sys.h>

#include "cpu.h"
#include "context.h"
#include "errno.h"
#include "platform.h"
#include "profile.h"
#include "rpc.h"
#include "sche.h"
#include "stdlib.h"
#include "string.h"
#include "tensor_context.h"
#include "tensor_interrupt.h"

extern void proc_bootstrap(void);

struct proc_family {
    struct proc *root;
    struct proc *owner;
    struct cpu *cpu;
    struct proc_slot *slots;
    struct proc_slot *free_slots;
    struct proc_family *free_next;
    uint32_t max_child_threads;
    uint32_t active_child_threads;
    uint32_t reserved_proc_slots;
    uint32_t killed;
    uint32_t pending_produce;
    size_t child_stack_size;
    size_t child_printf_size;
    int64_t kill_status;
    struct dayan_context_fields root_context;
    uint8_t lsu_tcsr_used;
    uint8_t mat_tcsr_used;
    uint8_t vec_tcsr_used;
    uint8_t macro_tcsr_used;
};

struct proc_slot {
    struct proc proc;
    struct proc_slot *free_next;
    struct proc_slot *family_next;
    struct proc_family *family_owner;
    struct cpu *cpu;
    size_t stack_size;
    size_t printf_size;
    void *stack;
    void *printf_buffer;
    uint32_t state;
};

enum proc_slot_state {
    PROC_SLOT_GLOBAL_FREE = 0,
    PROC_SLOT_ROOT = 1,
    PROC_SLOT_FAMILY_FREE = 2,
    PROC_SLOT_CHILD = 3,
    PROC_SLOT_ZOMBIE = 4,
    PROC_SLOT_KILL_WAIT_PRODUCE = 5,
    PROC_SLOT_KILLED = 6,
};

static struct proc_slot
    hart_proc_slots[CONFIG_HART_COUNT][CONFIG_PROC_SLOT_COUNT]
    __attribute__((section(".proc_slots"), aligned(64)));
static struct proc_family
    hart_proc_families[CONFIG_HART_COUNT][CONFIG_PROC_SLOT_COUNT]
    __attribute__((section(".proc_slots"), aligned(64)));
static unsigned char
    hart_proc_stacks[CONFIG_HART_COUNT][CONFIG_PROC_SLOT_COUNT]
                    [CONFIG_PROC_STACK_SIZE]
    __attribute__((section(".proc_stack"), aligned(64)));
static unsigned char
    hart_proc_printf[CONFIG_HART_COUNT][CONFIG_PROC_SLOT_COUNT]
                    [CONFIG_PROC_PRINTF_SIZE]
    __attribute__((section(".proc_printf"), aligned(64)));

_Static_assert(CONFIG_PROC_SLOT_COUNT == DAYAN_HART_MAX_PROCS,
               "Device and public proc-slot counts must match");
_Static_assert(CONFIG_PROC_SLOT_COUNT == RPC_PROC_SLOTS_PER_HART,
               "Device and RPC proc-slot counts must match");
_Static_assert(CONFIG_PROC_STACK_SIZE == RPC_PROC_MAX_STACK_SIZE,
               "static proc stack capacity must match RPC validation");
_Static_assert(CONFIG_PROC_PRINTF_SIZE == RPC_PROC_MAX_PRINTF_SIZE,
               "static proc printf capacity must match RPC validation");
_Static_assert(PROC_NAME_SIZE == DAYAN_THREAD_NAME_SIZE,
               "scheduler and public Thread name sizes must match");
_Static_assert(PROC_NAME_SIZE == RPC_PROC_NAME_SIZE,
               "scheduler and RPC proc name sizes must match");
_Static_assert(CONFIG_PROC_SLOT_COUNT + RPC_PROFILE_SLOT_USER_BASE ==
                   RPC_PROFILE_SLOT_RPC_RECEIVE,
               "profile user slot range");

static uint8_t proc_slot_profile_id(const struct proc_slot *slot)
{
    uintptr_t hart = platform_hart_id();
    const struct proc_slot *base = hart_proc_slots[hart];
    size_t index;

    if (slot < base || slot >= base + CONFIG_PROC_SLOT_COUNT)
        platform_halt();
    index = (size_t)(slot - base);
    return (uint8_t)(RPC_PROFILE_SLOT_USER_BASE + index);
}

struct proc *proc_self(void)
{
    struct thread *thread = current;

    if (!thread || thread->type != THREAD_TYPE_PROC)
        return 0;
    return container_of(thread, struct proc, thread);
}

static inline void proc_state_set(struct proc *proc, uint32_t state)
{
    proc->thread.state = state;
    if (__builtin_expect(proc->thread.cpu->sche.profile_enabled, 0))
        profile_record_proc_state(proc);
}

static inline void switch_to_scheduler(struct thread *thread)
{
    switch_context(thread, &thread->cpu->sche.init);
}

void sche_init(void)
{
    struct thread *init = current;
    struct sche_state *sche = &init->cpu->sche;
    uintptr_t hart = platform_hart_id();
    uint32_t index;

    if (hart >= CONFIG_HART_COUNT)
        platform_halt();

    init->id = 0;
    init->state = THREAD_RUNNING;
    init->pc = 0;
    init->type = THREAD_TYPE_BASE;
    INIT_LIST_HEAD(&sche->ready);
    INIT_LIST_HEAD(&sche->active);
    INIT_LIST_HEAD(&sche->wait_poll);
    INIT_LIST_HEAD(&sche->wait_event);
    sche->active_dirty = 0;
    sche->profile_enabled = 0;
    sche->next_id = 1;
    sche->free_proc_slots = 0;
    sche->free_proc_slot_count = 0;
    sche->reserved_proc_slot_count = 0;
    sche->free_proc_families = 0;
    sche->free_proc_family_count = 0;
    for (index = 0; index < CONFIG_PROC_SLOT_COUNT; ++index) {
        struct proc_slot *slot = &hart_proc_slots[hart][index];
        struct proc_family *family = &hart_proc_families[hart][index];

        memset(slot, 0, sizeof(*slot));
        slot->cpu = init->cpu;
        slot->stack = hart_proc_stacks[hart][index];
        slot->printf_buffer = hart_proc_printf[hart][index];
        slot->free_next = sche->free_proc_slots;
        sche->free_proc_slots = slot;
        ++sche->free_proc_slot_count;
        memset(family, 0, sizeof(*family));
        family->free_next = sche->free_proc_families;
        sche->free_proc_families = family;
        ++sche->free_proc_family_count;
    }
}

static inline struct proc_slot *global_slot_pop(struct cpu *cpu)
{
    struct sche_state *sche = &cpu->sche;
    struct proc_slot *slot = sche->free_proc_slots;

    if (!slot)
        return 0;
    if (!sche->free_proc_slot_count || slot->state != PROC_SLOT_GLOBAL_FREE ||
        slot->family_owner || slot->cpu != cpu)
        platform_halt();
    sche->free_proc_slots = slot->free_next;
    --sche->free_proc_slot_count;
    slot->free_next = 0;
    return slot;
}

static inline void global_slot_push(struct proc_slot *slot)
{
    struct sche_state *sche;

    if (!slot || !slot->cpu || slot->cpu != cpu_self() ||
        slot->state != PROC_SLOT_GLOBAL_FREE || slot->family_owner ||
        slot->family_next)
        platform_halt();
    sche = &slot->cpu->sche;
    if (sche->free_proc_slot_count >= CONFIG_PROC_SLOT_COUNT)
        platform_halt();
    slot->free_next = sche->free_proc_slots;
    sche->free_proc_slots = slot;
    ++sche->free_proc_slot_count;
}

int proc_slot_acquire(size_t stack_size, size_t printf_size,
                      struct proc **proc_out)
{
    struct cpu *cpu = cpu_self();
    struct proc_slot *slot;

    if (!proc_out || stack_size < sizeof(struct proc_frame) ||
        stack_size > CONFIG_PROC_STACK_SIZE ||
        printf_size > CONFIG_PROC_PRINTF_SIZE)
        return -EINVAL;
    *proc_out = 0;
    slot = global_slot_pop(cpu);
    if (!slot)
        return -ENOMEM;
    memset(&slot->proc, 0, sizeof(slot->proc));
    slot->proc.thread.profile_slot = proc_slot_profile_id(slot);
    slot->stack_size = stack_size;
    slot->printf_size = printf_size;
    slot->state = PROC_SLOT_ROOT;
    *proc_out = &slot->proc;
    return 0;
}

int proc_slot_init_args(struct proc *proc, const char *name, proc_fn entry,
                        void *shared_base, size_t argc,
                        const uintptr_t *args)
{
    struct proc_slot *slot;

    if (!proc)
        return -EINVAL;
    slot = container_of(proc, struct proc_slot, proc);
    if ((slot->state != PROC_SLOT_ROOT &&
         slot->state != PROC_SLOT_CHILD) ||
        slot->cpu != cpu_self())
        return -EPERM;
    return proc_init_args(proc, name, entry, slot->stack, slot->stack_size,
                          shared_base,
                          slot->printf_size ? slot->printf_buffer : 0,
                          slot->printf_size, argc, args);
}

void proc_slot_release(struct proc *proc)
{
    struct proc_slot *slot;

    if (!proc)
        platform_halt();
    slot = container_of(proc, struct proc_slot, proc);
    if (slot->state != PROC_SLOT_ROOT || !slot->cpu ||
        slot->cpu != cpu_self() || slot->family_owner || proc->family ||
        proc->live_children)
        platform_halt();
    memset(&slot->proc, 0, sizeof(slot->proc));
    slot->stack_size = 0;
    slot->printf_size = 0;
    slot->state = PROC_SLOT_GLOBAL_FREE;
    global_slot_push(slot);
}

int thread_init(struct thread *thread, uint32_t type)
{
    struct cpu *cpu = cpu_self();
    struct sche_state *sche = &cpu->sche;

    if (!thread || type == THREAD_TYPE_BASE)
        return -EINVAL;
    if (!sche->next_id || sche->next_id > INT32_MAX)
        return -EOVERFLOW;

    thread->id = sche->next_id++;
    thread->cpu = cpu;
    thread->sp = 0;
    thread->pc = 0;
    thread->tensor_context = 0;
    thread->type = type;
    thread->state = THREAD_WAIT_START;
    thread->profile_defined = 0;
    if (type == THREAD_TYPE_PROC && !thread->profile_slot)
        return -EINVAL;
    if (type == THREAD_TYPE_PROC && sche->profile_enabled)
        profile_record_proc_start(container_of(thread, struct proc, thread));
    return 0;
}

static inline int proc_name_copy(char destination[PROC_NAME_SIZE],
                                 const char *source)
{
    size_t length;

    if (!destination || !source || !source[0])
        return -EINVAL;
    for (length = 0; length < PROC_NAME_SIZE; ++length) {
        destination[length] = source[length];
        if (!source[length]) {
            memset(destination + length + 1, 0,
                   PROC_NAME_SIZE - length - 1);
            return 0;
        }
    }
    memset(destination, 0, PROC_NAME_SIZE);
    return -ENAMETOOLONG;
}

int proc_init_args(struct proc *proc, const char *name, proc_fn entry,
                   void *stack, size_t stack_size, void *shared_base,
                   void *printf_buffer, size_t printf_capacity,
                   size_t argc, const uintptr_t *args)
{
    struct proc_frame *frame;
    uintptr_t base;
    uintptr_t top;
    size_t i;
    int status;

    if (!proc || !entry || !stack || argc > PROC_MAX_ARGS ||
        (argc && !args) ||
        (!printf_buffer != !printf_capacity) ||
        (printf_buffer && printf_capacity >
         UINTPTR_MAX - (uintptr_t)printf_buffer))
        return -EINVAL;

    base = (uintptr_t)stack;
    if (stack_size > UINTPTR_MAX - base)
        return -EINVAL;
    top = (base + stack_size) & ~(uintptr_t)0xf;
    if (top < base || top - base < sizeof(*frame))
        return -EINVAL;

    status = proc_name_copy(proc->name, name);
    if (status < 0)
        return status;
    status = thread_init(&proc->thread, THREAD_TYPE_PROC);
    if (status < 0) {
        memset(proc->name, 0, sizeof(proc->name));
        return status;
    }

    frame = (struct proc_frame *)(top - sizeof(*frame));
    frame->entry = (uintptr_t)entry;
    frame->ra = (uintptr_t)proc_exit;
    for (i = 0; i < argc; ++i)
        frame->args[i] = args[i];
    frame->shared = (uintptr_t)shared_base;
    frame->reserved = 0;

    proc->thread.sp = (uintptr_t)frame;
    proc->thread.pc = (uintptr_t)proc_bootstrap +
                      PROC_BOOTSTRAP_PC_OFFSET(argc);
    INIT_LIST_HEAD(&proc->node);
    proc->poll = 0;
    proc->poll_arg = 0;
    proc->printf_buffer = printf_buffer;
    proc->printf_capacity = printf_capacity;
    proc->printf_size = 0;
    proc->cleanup = 0;
    proc->cleanup_arg = 0;
    proc->system_wait = 0;
    proc->exit_status = 0;
    proc->task_id[0] = 0;
    proc->task_id[1] = 0;
    proc->task_id[2] = 0;
    proc->task_dim[0] = 1;
    proc->task_dim[1] = 1;
    proc->task_dim[2] = 1;
    proc->task_mode = 1;
    proc->parent = 0;
    proc->parent_id = 0;
    proc->family = 0;
    INIT_LIST_HEAD(&proc->wait_children);
    proc->live_children = 0;
    proc->join_kind = 0;
    proc->join_thread_id = 0;
    proc->host_proc_id = 0;
    proc->cookie = 0;
    proc->shared_base = shared_base;
    proc->shared_size = 0;
    proc->uram_base = 0;
    proc->uram_size = 0;
    proc_state_set(proc, THREAD_READY);
    list_add_tail(&proc->node, &proc->thread.cpu->sche.ready);
    return 0;
}

__attribute__((noinline))
int proc_vinit(struct proc *proc, const char *name, proc_fn entry,
               void *stack, size_t stack_size, void *shared_base,
               void *printf_buffer, size_t printf_capacity,
               size_t argc, va_list args)
{
    uintptr_t words[PROC_MAX_ARGS];
    size_t i;

    if (argc > PROC_MAX_ARGS)
        return -EINVAL;
    for (i = 0; i < argc; ++i)
        words[i] = va_arg(args, uintptr_t);
    return proc_init_args(proc, name, entry, stack, stack_size, shared_base,
                          printf_buffer, printf_capacity, argc, words);
}

int proc_init(struct proc *proc, const char *name, proc_fn entry,
              void *stack, size_t stack_size, void *shared_base,
              void *printf_buffer, size_t printf_capacity,
              size_t argc, ...)
{
    va_list args;
    int status;

    va_start(args, argc);
    status = proc_vinit(proc, name, entry, stack, stack_size, shared_base,
                        printf_buffer, printf_capacity, argc, args);
    va_end(args);
    return status;
}

int proc_set_cleanup(struct proc *proc, proc_cleanup_fn cleanup, void *arg)
{
    if (!proc || proc->thread.type != THREAD_TYPE_PROC)
        return -EINVAL;
    if (proc->thread.cpu != cpu_self())
        return -EPERM;
    if (proc->thread.state != THREAD_READY || proc->cleanup)
        return -EBUSY;
    proc->cleanup = cleanup;
    proc->cleanup_arg = arg;
    return 0;
}

int proc_set_task(struct proc *proc, const uint32_t task_id[3],
                  const uint32_t task_dim[3], uint32_t task_mode)
{
    size_t axis;

    if (!proc || !task_id || !task_dim ||
        (task_mode != 1U && task_mode != 4U) ||
        (task_mode == 4U && task_dim[0] != 4U))
        return -EINVAL;
    if (proc->thread.cpu != cpu_self())
        return -EPERM;
    if (proc->thread.type != THREAD_TYPE_PROC ||
        proc->thread.state != THREAD_READY)
        return -EBUSY;
    for (axis = 0; axis < 3; ++axis) {
        if (!task_dim[axis] || task_id[axis] >= task_dim[axis])
            return -EINVAL;
    }
    for (axis = 0; axis < 3; ++axis) {
        proc->task_id[axis] = task_id[axis];
        proc->task_dim[axis] = task_dim[axis];
    }
    proc->task_mode = task_mode;
    return 0;
}

int proc_family_create(struct proc *owner, uint32_t max_child_threads,
                       size_t child_stack_size,
                       size_t child_printf_size,
                       struct proc_family **family_out)
{
    struct proc_slot *root_slot;
    struct proc_family *family;
    struct sche_state *sche;
    uint32_t child_index;
    uint32_t reservation;

    if (!owner || !family_out ||
        max_child_threads > DAYAN_KERNEL_MAX_CHILD_THREADS ||
        child_stack_size < sizeof(struct proc_frame) ||
        child_stack_size > CONFIG_PROC_STACK_SIZE ||
        child_printf_size > CONFIG_PROC_PRINTF_SIZE)
        return -EINVAL;
    *family_out = 0;
    root_slot = container_of(owner, struct proc_slot, proc);
    if (root_slot->state != PROC_SLOT_ROOT || root_slot->family_owner ||
        root_slot->cpu != cpu_self())
        return -EPERM;
    sche = &root_slot->cpu->sche;
    reservation = max_child_threads + 1U;
    if (sche->free_proc_slot_count < max_child_threads ||
        !sche->free_proc_families || !sche->free_proc_family_count ||
        sche->reserved_proc_slot_count > CONFIG_PROC_SLOT_COUNT ||
        reservation > CONFIG_PROC_SLOT_COUNT -
                          sche->reserved_proc_slot_count)
        return -ENOMEM;
    family = sche->free_proc_families;
    sche->free_proc_families = family->free_next;
    --sche->free_proc_family_count;
    memset(family, 0, sizeof(*family));
    sche->reserved_proc_slot_count += reservation;
    family->owner = owner;
    family->cpu = root_slot->cpu;
    family->max_child_threads = max_child_threads;
    family->reserved_proc_slots = reservation;
    family->child_stack_size = child_stack_size;
    family->child_printf_size = child_printf_size;
    root_slot->family_owner = family;
    for (child_index = 0; child_index < max_child_threads; ++child_index) {
        struct proc_slot *child_slot = global_slot_pop(root_slot->cpu);

        if (!child_slot)
            platform_halt();
        child_slot->family_owner = family;
        child_slot->family_next = family->slots;
        family->slots = child_slot;
        child_slot->free_next = family->free_slots;
        family->free_slots = child_slot;
        child_slot->stack_size = child_stack_size;
        child_slot->printf_size = child_printf_size;
        child_slot->state = PROC_SLOT_FAMILY_FREE;
    }
    *family_out = family;
    return 0;
}

static void proc_family_return(struct proc_family *family)
{
    struct sche_state *sche;
    struct proc_slot *slot;

    if (family->active_child_threads || !family->cpu ||
        family->cpu != cpu_self() ||
        family->pending_produce)
        platform_halt();
    sche = &family->cpu->sche;
    if (sche->reserved_proc_slot_count < family->reserved_proc_slots)
        platform_halt();
    for (slot = family->slots; slot;) {
        struct proc_slot *next = slot->family_next;

        if (slot->state != PROC_SLOT_FAMILY_FREE ||
            slot->family_owner != family)
            platform_halt();
        memset(&slot->proc, 0, sizeof(slot->proc));
        slot->family_owner = 0;
        slot->family_next = 0;
        slot->free_next = 0;
        slot->stack_size = 0;
        slot->printf_size = 0;
        slot->state = PROC_SLOT_GLOBAL_FREE;
        global_slot_push(slot);
        slot = next;
    }
    sche->reserved_proc_slot_count -= family->reserved_proc_slots;
    memset(family, 0, sizeof(*family));
    family->free_next = sche->free_proc_families;
    sche->free_proc_families = family;
    if (sche->free_proc_family_count >= CONFIG_PROC_SLOT_COUNT)
        platform_halt();
    ++sche->free_proc_family_count;
}

void proc_family_destroy(struct proc_family *family)
{
    struct proc *owner;
    struct proc_slot *root_slot;

    if (!family)
        return;
    owner = family->owner;
    if (!owner || (family->root && family->root != owner))
        platform_halt();
    root_slot = container_of(owner, struct proc_slot, proc);
    if (root_slot->state != PROC_SLOT_ROOT ||
        root_slot->family_owner != family)
        platform_halt();
    root_slot->family_owner = 0;
    if (owner->family == family)
        owner->family = 0;
    proc_family_return(family);
}

int proc_set_resources(struct proc *proc, struct proc_family *family,
                       size_t shared_size, void *uram_base,
                       size_t uram_size, uint64_t tensor_context)
{
    struct dayan_context_fields fields = dayan_context_unpack(tensor_context);
    uintptr_t uram_address = (uintptr_t)uram_base;
    uintptr_t uram_offset = uram_address &
                            (DAYAN_CONTEXT_URAM_WINDOW - 1U);

    if (!proc || !family || proc->thread.cpu != cpu_self() ||
        proc->thread.type != THREAD_TYPE_PROC ||
        proc->thread.state != THREAD_READY || proc->family || family->root ||
        family->owner != proc || family->cpu != proc->thread.cpu ||
        (!proc->shared_base != !shared_size) ||
        (!uram_base != !uram_size) ||
        (tensor_context >> 56) || !dayan_context_fields_valid(&fields) ||
        (uram_size &&
         ((uram_address & (DAYAN_URAM_ALLOCATION_GRANULE - 1U)) ||
          (uram_size & (DAYAN_URAM_ALLOCATION_GRANULE - 1U)) ||
          uram_size > UINTPTR_MAX - uram_address)) ||
        fields.uram_lo != uram_offset / DAYAN_CONTEXT_URAM_GRANULE ||
        fields.uram_hi != (uram_offset + uram_size) /
                              DAYAN_CONTEXT_URAM_GRANULE)
        return -EINVAL;
    proc->family = family;
    family->root = proc;
    family->root_context = fields;
    proc->shared_size = shared_size;
    proc->uram_base = uram_base;
    proc->uram_size = uram_size;
    proc->thread.tensor_context = tensor_context;
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

static void child_tcsr_release(uint8_t *used,
                               struct dayan_tcsr_range parent,
                               struct dayan_tcsr_range child)
{
    uint8_t logical_base;
    uint8_t mask;

    if (!child.count)
        return;
    if (child.base < parent.base ||
        child.base + child.count > parent.base + parent.count)
        platform_halt();
    logical_base = (uint8_t)(child.base - parent.base);
    mask = (uint8_t)(((1U << child.count) - 1U) << logical_base);
    *used &= (uint8_t)~mask;
}

static void child_resources_release(struct proc *child)
{
    struct proc_family *family;
    struct dayan_context_fields fields;

    if (!child || !child->family)
        return;
    family = child->family;
    fields = dayan_context_unpack(child->thread.tensor_context);
    child_tcsr_release(&family->lsu_tcsr_used, family->root_context.lsu,
                       fields.lsu);
    child_tcsr_release(&family->mat_tcsr_used, family->root_context.mat,
                       fields.mat);
    child_tcsr_release(&family->vec_tcsr_used, family->root_context.vec,
                       fields.vec);
    child_tcsr_release(&family->macro_tcsr_used, family->root_context.macro,
                       fields.macro);
    fields.lsu.count = 0;
    fields.mat.count = 0;
    fields.vec.count = 0;
    fields.macro.count = 0;
    child->thread.tensor_context = dayan_context_pack(&fields);
}

static struct proc_slot *family_slot_acquire(struct proc_family *family)
{
    struct proc_slot *slot;

    if (!family || family->cpu != cpu_self())
        return 0;
    slot = family->free_slots;
    if (!slot)
        return 0;
    if (slot->state != PROC_SLOT_FAMILY_FREE ||
        slot->family_owner != family)
        platform_halt();
    family->free_slots = slot->free_next;
    slot->free_next = 0;
    memset(&slot->proc, 0, sizeof(slot->proc));
    slot->proc.thread.profile_slot = proc_slot_profile_id(slot);
    slot->state = PROC_SLOT_CHILD;
    return slot;
}

static void family_slot_release(struct proc_slot *slot)
{
    struct proc_family *family;

    if (!slot || !slot->family_owner || slot->cpu != cpu_self() ||
        (slot->state != PROC_SLOT_CHILD &&
         slot->state != PROC_SLOT_ZOMBIE &&
         slot->state != PROC_SLOT_KILLED))
        platform_halt();
    family = slot->family_owner;
    memset(&slot->proc, 0, sizeof(slot->proc));
    slot->state = PROC_SLOT_FAMILY_FREE;
    slot->free_next = family->free_slots;
    family->free_slots = slot;
}

static struct proc_slot *find_direct_child(struct proc *parent,
                                           uint32_t proc_id)
{
    struct proc_slot *slot;

    if (!parent || !parent->family)
        return 0;
    for (slot = parent->family->slots; slot; slot = slot->family_next) {
        if ((slot->state == PROC_SLOT_CHILD ||
             slot->state == PROC_SLOT_ZOMBIE ||
             slot->state == PROC_SLOT_KILL_WAIT_PRODUCE) &&
            slot->proc.parent == parent &&
            slot->proc.thread.id == proc_id)
            return slot;
    }
    return 0;
}

static void child_thread_cleanup(struct proc *proc, void *arg)
{
    struct proc_family *family = proc->family;
    struct proc *parent = proc->parent;
    struct proc_slot *slot = container_of(proc, struct proc_slot, proc);

    (void)arg;
    if (!family || !parent || !family->active_child_threads ||
        !parent->live_children || slot->state != PROC_SLOT_CHILD ||
        slot->family_owner != family)
        platform_halt();
    child_resources_release(proc);
    --family->active_child_threads;
    --parent->live_children;
    slot->state = PROC_SLOT_ZOMBIE;
    if (proc->exit_status < 0 && parent->exit_status == 0)
        parent->exit_status = proc->exit_status;
    if ((parent->join_kind == 1U &&
         parent->join_thread_id == proc->thread.id) ||
        (parent->join_kind == 2U && !parent->live_children))
        (void)proc_event_wake_all(&parent->wait_children);
}

int thread_create_child(struct proc *parent, const char *name, proc_fn entry,
                        void *argument,
                        const dayan_thread_resources_t *resources,
                        uint32_t *thread_id_out)
{
    struct proc_family *family;
    struct proc_slot *child_slot;
    struct proc *child;
    struct dayan_context_fields child_context;
    dayan_thread_resources_t empty_resources;
    uintptr_t args[1];
    int status;

    if (!parent || parent != proc_self() || !name || !entry || !thread_id_out)
        return -EINVAL;
    if (!resources) {
        memset(&empty_resources, 0, sizeof(empty_resources));
        resources = &empty_resources;
    }
    if (resources->reserved ||
        resources->lsu_tcsr_count > DAYAN_TCSR_LOGICAL_GROUPS ||
        resources->mat_tcsr_count > DAYAN_TCSR_LOGICAL_GROUPS ||
        resources->vec_tcsr_count > DAYAN_TCSR_LOGICAL_GROUPS ||
        resources->macro_tcsr_count > DAYAN_TCSR_LOGICAL_GROUPS)
        return -EINVAL;
    family = parent->family;
    if (!family || family->root->thread.cpu != cpu_self())
        return -EPERM;
    if (family->active_child_threads >= family->max_child_threads ||
        !family->free_slots)
        return -ENOMEM;
    child_slot = family_slot_acquire(family);
    if (!child_slot)
        return -ENOMEM;
    child = &child_slot->proc;
    child_context = family->root_context;
    child_context.lsu.count = 0;
    child_context.mat.count = 0;
    child_context.vec.count = 0;
    child_context.macro.count = 0;
    status = child_tcsr_reserve(&family->lsu_tcsr_used,
                                family->root_context.lsu,
                                resources->lsu_tcsr_count,
                                &child_context.lsu);
    if (status == 0)
        status = child_tcsr_reserve(&family->mat_tcsr_used,
                                    family->root_context.mat,
                                    resources->mat_tcsr_count,
                                    &child_context.mat);
    if (status == 0)
        status = child_tcsr_reserve(&family->vec_tcsr_used,
                                    family->root_context.vec,
                                    resources->vec_tcsr_count,
                                    &child_context.vec);
    if (status == 0)
        status = child_tcsr_reserve(&family->macro_tcsr_used,
                                    family->root_context.macro,
                                    resources->macro_tcsr_count,
                                    &child_context.macro);
    if (status < 0) {
        child->family = family;
        child->thread.tensor_context = dayan_context_pack(&child_context);
        child_resources_release(child);
        family_slot_release(child_slot);
        return status;
    }
    args[0] = (uintptr_t)argument;
    status = proc_slot_init_args(child, name, entry, parent->shared_base, 1,
                                 args);
    if (status < 0) {
        child->family = family;
        child->thread.tensor_context = dayan_context_pack(&child_context);
        child_resources_release(child);
        family_slot_release(child_slot);
        return status;
    }
    child->parent = parent;
    child->parent_id = parent->thread.id;
    child->family = family;
    child->thread.tensor_context = dayan_context_pack(&child_context);
    child->shared_size = parent->shared_size;
    child->uram_base = parent->uram_base;
    child->uram_size = parent->uram_size;
    child->host_proc_id = parent->host_proc_id;
    child->cookie = parent->cookie;
    status = proc_set_task(child, parent->task_id, parent->task_dim,
                           parent->task_mode);
    if (status < 0)
        platform_halt();
    status = proc_set_cleanup(child, child_thread_cleanup, 0);
    if (status < 0)
        platform_halt();
    ++parent->live_children;
    ++family->active_child_threads;
    *thread_id_out = child->thread.id;
    return 0;
}

int thread_join(uint32_t thread_id, int64_t *result)
{
    struct proc *parent = proc_self();
    struct proc_slot *slot;

    if (!parent || !parent->family)
        return -EPERM;
    if (!thread_id || thread_id == parent->thread.id)
        return -ECHILD;
    slot = find_direct_child(parent, thread_id);
    if (!slot)
        return -ECHILD;
    while (slot->state != PROC_SLOT_ZOMBIE) {
        parent->join_kind = 1U;
        parent->join_thread_id = thread_id;
        proc_wait_event(&parent->wait_children);
        parent->join_kind = 0;
        parent->join_thread_id = 0;
        slot = find_direct_child(parent, thread_id);
        if (!slot)
            return -ECHILD;
    }
    if (result)
        *result = slot->proc.exit_status;
    family_slot_release(slot);
    return 0;
}

int thread_join_all(void)
{
    struct proc *parent = proc_self();
    struct proc_slot *slot;

    if (!parent || !parent->family)
        return -EPERM;
    while (parent->live_children) {
        parent->join_kind = 2U;
        parent->join_thread_id = 0;
        proc_wait_event(&parent->wait_children);
        parent->join_kind = 0;
    }
    for (slot = parent->family->slots; slot; slot = slot->family_next) {
        if (slot->state == PROC_SLOT_ZOMBIE &&
            slot->proc.parent == parent)
            family_slot_release(slot);
    }
    return 0;
}

static void snapshot_proc(struct rpc_proc_info *record,
                          const struct proc *proc, uint32_t flags,
                          uint32_t slot_index)
{
    memset(record, 0, sizeof(*record));
    record->proc_id = proc->thread.id;
    record->parent_proc_id = proc->parent_id;
    record->state = proc->thread.state;
    record->flags = flags;
    record->live_children = proc->live_children;
    record->slot_index = slot_index;
    record->task_mode = proc->task_mode;
    record->host_proc_id = proc->host_proc_id;
    memcpy(record->task_id, proc->task_id, sizeof(record->task_id));
    memcpy(record->task_dim, proc->task_dim, sizeof(record->task_dim));
    record->result = proc->exit_status;
    record->pc = proc->thread.pc;
    record->cookie = proc->cookie;
    memcpy(record->name, proc->name, sizeof(record->name));
}

static int snapshot_slot_flags(const struct proc_slot *slot,
                               uint32_t *flags_out)
{
    uint32_t flags;

    if (slot->state == PROC_SLOT_ROOT) {
        if (!slot->proc.family)
            return 0;
        flags = RPC_PROC_INFO_HOST_ROOT;
    } else if (slot->state == PROC_SLOT_CHILD) {
        flags = RPC_PROC_INFO_CHILD;
    } else if (slot->state == PROC_SLOT_ZOMBIE) {
        flags = RPC_PROC_INFO_CHILD | RPC_PROC_INFO_ZOMBIE;
    } else if (slot->state == PROC_SLOT_KILL_WAIT_PRODUCE) {
        flags = slot->proc.parent_id ? RPC_PROC_INFO_CHILD :
                                       RPC_PROC_INFO_HOST_ROOT;
        flags |= RPC_PROC_INFO_KILL_WAIT_PRODUCE;
    } else {
        return 0;
    }
    if (flags_out)
        *flags_out = flags;
    return 1;
}

int proc_snapshot(struct rpc_proc_ps_snapshot *snapshot)
{
    struct cpu *cpu = cpu_self();
    uintptr_t hart = platform_hart_id();
    uint32_t count = 0;
    uint32_t index;

    if (!snapshot || snapshot->flags ||
        snapshot->capacity != RPC_PROC_PS_MAX_RECORDS ||
        snapshot->reserved)
        return -EINVAL;
    memset(snapshot->records, 0, sizeof(snapshot->records));
    snapshot_proc(&snapshot->records[count++], &cpu->rpc.receive,
                  RPC_PROC_INFO_SYSTEM, UINT32_MAX);
    snapshot_proc(&snapshot->records[count++], &cpu->rpc.send,
                  RPC_PROC_INFO_SYSTEM, UINT32_MAX);
    snapshot_proc(&snapshot->records[count++], profile_proc(),
                  RPC_PROC_INFO_SYSTEM, UINT32_MAX);
    for (index = 0; index < CONFIG_PROC_SLOT_COUNT; ++index) {
        struct proc_slot *slot = &hart_proc_slots[hart][index];
        uint32_t flags;

        if (!snapshot_slot_flags(slot, &flags))
            continue;
        if (count >= RPC_PROC_PS_MAX_RECORDS)
            platform_halt();
        snapshot_proc(&snapshot->records[count++], &slot->proc,
                      flags, index);
    }
    snapshot->hart_id = (uint32_t)hart;
    snapshot->count = count;
    snapshot->global_free_slots = cpu->sche.free_proc_slot_count;
    snapshot->reserved_slots = cpu->sche.reserved_proc_slot_count;
    snapshot->status = 0;
    return 0;
}

void proc_profile_snapshot(void)
{
    struct cpu *cpu = cpu_self();
    uintptr_t hart = platform_hart_id();
    uint32_t index;

    profile_record_proc_start(&cpu->rpc.receive);
    profile_record_proc_start(&cpu->rpc.send);
    profile_record_proc_start(profile_proc());
    for (index = 0; index < CONFIG_PROC_SLOT_COUNT; ++index) {
        struct proc_slot *slot = &hart_proc_slots[hart][index];

        if (snapshot_slot_flags(slot, 0))
            profile_record_proc_start(&slot->proc);
    }
}

static void kill_unlink_proc(struct proc *proc)
{
    switch (proc->thread.state) {
    case THREAD_READY:
    case THREAD_WAIT_POLL:
    case THREAD_WAIT_EVENT:
    case THREAD_WAIT_PRODUCE:
        if (list_empty(&proc->node))
            platform_halt();
        list_del_init(&proc->node);
        break;
    case THREAD_WAIT_EXIT:
    case THREAD_WAIT_START:
        if (!list_empty(&proc->node))
            platform_halt();
        break;
    default:
        platform_halt();
    }
    proc->poll = 0;
    proc->poll_arg = 0;
}

static void kill_mark_slot(struct proc_family *family,
                           struct proc_slot *slot)
{
    struct proc *proc = &slot->proc;
    int waiting_produce;

    if (slot->state == PROC_SLOT_ZOMBIE) {
        slot->state = PROC_SLOT_KILLED;
        proc->exit_status = family->kill_status;
        return;
    }
    if (slot->state != PROC_SLOT_ROOT &&
        slot->state != PROC_SLOT_CHILD)
        platform_halt();
    waiting_produce = proc->thread.state == THREAD_WAIT_PRODUCE;
    kill_unlink_proc(proc);
    proc->exit_status = family->kill_status;
    proc->live_children = 0;
    proc->join_kind = 0;
    proc->join_thread_id = 0;
    if (proc->parent_id)
        proc->cleanup = 0;
    proc->cleanup_arg = proc->parent_id ? 0 : proc->cleanup_arg;
    if (waiting_produce) {
        proc_state_set(proc, THREAD_KILL_WAIT_PRODUCE);
        slot->state = PROC_SLOT_KILL_WAIT_PRODUCE;
        ++family->pending_produce;
    } else {
        proc_state_set(proc, THREAD_WAIT_EXIT);
        slot->state = PROC_SLOT_KILLED;
        if (proc->parent_id) {
            if (!family->active_child_threads)
                platform_halt();
            child_resources_release(proc);
            --family->active_child_threads;
        }
    }
}

static void kill_drain_system_wait(struct proc *proc)
{
    int status;

    while ((status = rpc_proc_wait_cancel(proc)) > 0)
        proc_yield();
    if (status < 0)
        platform_halt();
}

static void kill_release_child_slot(struct proc_family *family,
                                    struct proc_slot *slot)
{
    struct sche_state *sche = &family->cpu->sche;
    struct proc_slot **link = &family->slots;

    if ((slot->state != PROC_SLOT_FAMILY_FREE &&
         slot->state != PROC_SLOT_KILLED) ||
        slot->family_owner != family || !family->reserved_proc_slots ||
        !sche->reserved_proc_slot_count)
        platform_halt();
    while (*link && *link != slot)
        link = &(*link)->family_next;
    if (!*link)
        platform_halt();
    *link = slot->family_next;
    memset(&slot->proc, 0, sizeof(slot->proc));
    slot->family_owner = 0;
    slot->family_next = 0;
    slot->free_next = 0;
    slot->stack_size = 0;
    slot->printf_size = 0;
    slot->state = PROC_SLOT_GLOBAL_FREE;
    --family->reserved_proc_slots;
    --sche->reserved_proc_slot_count;
    global_slot_push(slot);
}

static void kill_release_root_slot(struct proc_family *family)
{
    struct proc *root = family->root;
    struct proc_slot *slot;
    struct sche_state *sche = &family->cpu->sche;

    if (!root || root->parent_id || !family->reserved_proc_slots ||
        !sche->reserved_proc_slot_count)
        platform_halt();
    slot = container_of(root, struct proc_slot, proc);
    if (slot->state != PROC_SLOT_KILLED ||
        slot->family_owner != family)
        platform_halt();
    family->root = 0;
    family->owner = 0;
    --family->reserved_proc_slots;
    --sche->reserved_proc_slot_count;
    slot->family_owner = 0;
    root->family = 0;
    root->live_children = 0;
    slot->state = PROC_SLOT_ROOT;
    proc_slot_release(root);
}

static void kill_flush_output(struct proc *proc)
{
    uint32_t output_proc = proc->host_proc_id ? proc->host_proc_id :
                                                proc->thread.id;

    if (proc->printf_size)
        (void)rpc_write_proc(output_proc, proc->printf_buffer,
                             proc->printf_size);
    proc->printf_size = 0;
    (void)rpc_clear_proc(output_proc);
}

static void kill_complete_root(struct proc_family *family)
{
    struct proc *root = family->root;
    proc_cleanup_fn cleanup;
    void *cleanup_arg;

    if (!root)
        platform_halt();
    kill_flush_output(root);
    cleanup = root->cleanup;
    cleanup_arg = root->cleanup_arg;
    root->cleanup = 0;
    root->cleanup_arg = 0;
    if (!cleanup)
        platform_halt();
    cleanup(root, cleanup_arg);
    (void)rpc_proc_exit(root->thread.id, 0, family->kill_status, root->cookie,
                        RPC_PROC_EXIT_ROOT);
}

static void kill_release_ready_slots(struct proc_family *family,
                                     struct rpc_proc_kill_result *result)
{
    struct proc_slot *slot = family->slots;

    while (slot) {
        struct proc_slot *next = slot->family_next;

        if (slot->state == PROC_SLOT_FAMILY_FREE ||
            slot->state == PROC_SLOT_KILLED) {
            kill_release_child_slot(family, slot);
            ++result->released_slots;
        }
        slot = next;
    }
    if (family->root) {
        struct proc_slot *root_slot = container_of(
            family->root, struct proc_slot, proc);

        if (root_slot->state == PROC_SLOT_KILLED) {
            kill_release_root_slot(family);
            ++result->released_slots;
        }
    }
    if (!family->pending_produce) {
        if (family->active_child_threads || family->root || family->slots)
            platform_halt();
        proc_family_return(family);
    }
}

int proc_kill_all(struct rpc_proc_kill_result *result)
{
    struct cpu *cpu = cpu_self();
    uintptr_t hart = platform_hart_id();
    uint64_t selector;
    int64_t kill_status;
    uint32_t index;

    if (!result || (result->flags & ~RPC_PROC_KILL_FAMILY))
        return -EINVAL;
    selector = result->reserved;
    if (!!(result->flags & RPC_PROC_KILL_FAMILY) != !!selector ||
        selector > UINT32_MAX)
        return -EINVAL;
    kill_status = result->flags & RPC_PROC_KILL_FAMILY ? result->status :
                                                        -ECANCELED;
    if (kill_status >= 0)
        return -EINVAL;
    result->hart_id = (uint32_t)hart;
    result->status = 0;
    result->killed_roots = 0;
    result->killed_children = 0;
    result->released_slots = 0;
    result->waiting_produce = 0;

    /* First make every operator proc unschedulable before outbound RPCs. */
    for (index = 0; index < CONFIG_PROC_SLOT_COUNT; ++index) {
        struct proc_family *family = &hart_proc_families[hart][index];
        struct proc_slot *slot;

        if (family->cpu != cpu || family->killed || !family->root)
            continue;
        if (selector && family->root->thread.id != (uint32_t)selector)
            continue;
        family->killed = 1;
        family->kill_status = kill_status;
        family->free_slots = 0;
        for (slot = family->slots; slot; slot = slot->family_next) {
            if (slot->state == PROC_SLOT_CHILD ||
                slot->state == PROC_SLOT_ZOMBIE) {
                ++result->killed_children;
                kill_mark_slot(family, slot);
            } else if (slot->state != PROC_SLOT_FAMILY_FREE) {
                platform_halt();
            }
        }
        ++result->killed_roots;
        kill_mark_slot(family,
                       container_of(family->root, struct proc_slot, proc));
    }

    if (selector && !result->killed_roots)
        return -ESRCH;

    for (index = 0; index < CONFIG_PROC_SLOT_COUNT; ++index) {
        struct proc_family *family = &hart_proc_families[hart][index];
        struct proc_slot *slot;

        if (family->cpu != cpu || family->killed != 1)
            continue;
        for (slot = family->slots; slot; slot = slot->family_next) {
            if ((slot->state == PROC_SLOT_KILLED ||
                 slot->state == PROC_SLOT_KILL_WAIT_PRODUCE) &&
                slot->proc.thread.id)
                kill_drain_system_wait(&slot->proc);
        }
        kill_drain_system_wait(family->root);
        for (slot = family->slots; slot; slot = slot->family_next) {
            if ((slot->state == PROC_SLOT_KILLED ||
                 slot->state == PROC_SLOT_KILL_WAIT_PRODUCE) &&
                slot->proc.thread.id)
                kill_flush_output(&slot->proc);
        }
        kill_complete_root(family);
        family->killed = 2;
        kill_release_ready_slots(family, result);
    }
    for (index = 0; index < CONFIG_PROC_SLOT_COUNT; ++index) {
        struct proc_family *family = &hart_proc_families[hart][index];

        if (family->cpu == cpu && family->killed)
            result->waiting_produce += family->pending_produce;
    }
    return 0;
}

__attribute__((cold, noinline))
static void kill_complete_produce(struct proc *proc)
{
    struct proc_slot *slot = container_of(proc, struct proc_slot, proc);
    struct proc_family *family = slot->family_owner;
    int child = !!proc->parent_id;

    if (slot->state != PROC_SLOT_KILL_WAIT_PRODUCE || !family ||
        !family->killed || !family->pending_produce)
        platform_halt();
    slot->state = PROC_SLOT_KILLED;
    proc_state_set(proc, THREAD_WAIT_EXIT);
    --family->pending_produce;
    if (child) {
        if (!family->active_child_threads)
            platform_halt();
        child_resources_release(proc);
        --family->active_child_threads;
    }
    if (family->killed == 1)
        return;
    if (child)
        kill_release_child_slot(family, slot);
    else
        kill_release_root_slot(family);
    if (!family->pending_produce) {
        if (family->active_child_threads || family->root || family->slots)
            platform_halt();
        proc_family_return(family);
    }
}

int proc_wake(struct proc *proc)
{
    struct sche_state *sche;
    uint32_t state;

    if (!proc)
        return -EINVAL;
    state = proc->thread.state;
    if (state == THREAD_WAIT_PRODUCE || state == THREAD_WAIT_EVENT) {
        sche = &proc->thread.cpu->sche;
        proc_state_set(proc, THREAD_READY);
        list_move_tail(&proc->node, &sche->ready);
        return 0;
    }
    if (state == THREAD_KILL_WAIT_PRODUCE) {
        kill_complete_produce(proc);
        return 0;
    }
    return -EBUSY;
}

size_t proc_event_wake_one(struct list_head *event)
{
    struct proc *proc;

    if (list_empty(event))
        return 0;
    proc = list_first_entry(event, struct proc, node);
    return proc_wake(proc) == 0;
}

size_t proc_event_wake_all(struct list_head *event)
{
    size_t count = 0;

    while (proc_event_wake_one(event))
        ++count;
    return count;
}

static inline void switch_to_next_or_scheduler(struct proc *proc)
{
    struct sche_state *sche = &proc->thread.cpu->sche;
    struct list_head *next = proc->node.next;

    if (next == &sche->active) {
        switch_to_scheduler(&proc->thread);
    } else {
        struct proc *next_proc = list_entry(next, struct proc, node);

        proc_state_set(next_proc, THREAD_RUNNING);
        switch_context(&proc->thread, &next_proc->thread);
    }
}

void proc_wait_poll(proc_poll_fn poll, void *arg)
{
    struct proc *proc = (struct proc *)current;

    proc->poll = poll;
    proc->poll_arg = arg;
    proc_state_set(proc, THREAD_WAIT_POLL);
    proc->thread.cpu->sche.active_dirty = 1;
    switch_to_next_or_scheduler(proc);
}

void proc_wait_event(struct list_head *event)
{
    struct proc *proc = (struct proc *)current;
    struct sche_state *sche = &proc->thread.cpu->sche;

    proc_state_set(proc, THREAD_WAIT_EVENT);
    if (!event) {
        sche->active_dirty = 1;
        switch_to_next_or_scheduler(proc);
    } else {
        struct list_head *next = proc->node.next;

        list_move_tail(&proc->node, event);
        if (next == &sche->active) {
            switch_to_scheduler(&proc->thread);
        } else {
            struct proc *next_proc = list_entry(next, struct proc, node);

            proc_state_set(next_proc, THREAD_RUNNING);
            switch_context(&proc->thread, &next_proc->thread);
        }
    }
}

void proc_wait_produce(void)
{
    struct proc *proc = (struct proc *)current;

    proc_state_set(proc, THREAD_WAIT_PRODUCE);
    proc->thread.cpu->sche.active_dirty = 1;
    switch_to_next_or_scheduler(proc);
}

void proc_yield(void)
{
    struct proc *proc = (struct proc *)current;

    proc_state_set(proc, THREAD_READY);
    switch_to_next_or_scheduler(proc);
}

static void proc_run_cleanup(struct proc *proc)
{
    proc_cleanup_fn cleanup = proc->cleanup;
    void *cleanup_arg = proc->cleanup_arg;

    proc->cleanup = 0;
    proc->cleanup_arg = 0;
    if (cleanup)
        cleanup(proc, cleanup_arg);
}

_Noreturn void proc_exit_status(int64_t status)
{
    struct proc *proc = proc_self();

    if (proc) {
        /* Preserve a failure already propagated by a completed child. */
        if (status < 0 || proc->exit_status >= 0)
            proc->exit_status = status;
        if (proc->family && thread_join_all() < 0 && proc->exit_status >= 0)
            proc->exit_status = -EIO;
        if (proc->printf_size)
            (void)rpc_write(proc->printf_buffer, proc->printf_size);
        proc->printf_size = 0;
        (void)rpc_clear();
        if (!proc->parent && proc->family) {
            proc_run_cleanup(proc);
            (void)rpc_proc_exit(proc->thread.id, 0, proc->exit_status,
                                proc->cookie, RPC_PROC_EXIT_ROOT);
        }
        proc_state_set(proc, THREAD_WAIT_EXIT);
        proc->thread.cpu->sche.active_dirty = 1;
        switch_to_scheduler(&proc->thread);
    }
    platform_halt();
}

_Noreturn void proc_exit(void)
{
    proc_exit_status(0);
}

__attribute__((cold, noinline))
static void scheduler_cleanup_proc(struct proc *proc)
{
    struct proc_family *family = proc->family;
    int root = !proc->parent && family;

    proc_run_cleanup(proc);
    if (root) {
        proc_family_destroy(family);
        proc_slot_release(proc);
    }
}

static inline void scheduler_activate_ready(struct list_head *ready,
                                            struct list_head *active)
{
    active->prev->next = ready->next;
    ready->next->prev = active->prev;
    ready->prev->next = active;
    active->prev = ready->prev;
    INIT_LIST_HEAD(ready);
}

static inline void scheduler_finish_active_batch(struct sche_state *sche)
{
    struct list_head *node;
    struct list_head *next;

    list_for_each_safe(node, next, &sche->active) {
        struct proc *proc = list_entry(node, struct proc, node);

        switch (proc->thread.state) {
        case THREAD_READY:
            break;
        case THREAD_WAIT_POLL:
            list_move_tail(node, &sche->wait_poll);
            break;
        case THREAD_WAIT_EVENT:
        case THREAD_WAIT_PRODUCE:
            list_move_tail(node, &sche->wait_event);
            break;
        case THREAD_WAIT_EXIT:
            list_del_init(node);
            if (proc->cleanup || (!proc->parent && proc->family))
                scheduler_cleanup_proc(proc);
            break;
        default:
            platform_halt();
        }
    }
}

_Noreturn void scheduler(void)
{
    struct thread *init = current;
    struct sche_state *sche = &init->cpu->sche;

    for (;;) {
        tensor_interrupt_drain();
        struct list_head *node;
        struct list_head *next;

        scheduler_scan_produce();

        list_for_each_safe(node, next, &sche->wait_poll) {
            struct proc *waiting = list_entry(node, struct proc, node);

            if (!waiting->poll(waiting, waiting->poll_arg))
                continue;
            waiting->poll = 0;
            waiting->poll_arg = 0;
            proc_state_set(waiting, THREAD_READY);
            list_move_tail(node, &sche->ready);
        }

        if (!list_empty(&sche->ready))
            scheduler_activate_ready(&sche->ready, &sche->active);
        if (list_empty(&sche->active))
            continue;
        {
            struct proc *proc = list_first_entry(&sche->active,
                                                 struct proc, node);

            proc_state_set(proc, THREAD_RUNNING);
            switch_context(init, &proc->thread);
        }
        if (sche->active_dirty) {
            scheduler_finish_active_batch(sche);
            sche->active_dirty = 0;
        }
    }
}
