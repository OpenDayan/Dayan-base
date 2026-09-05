#include <stdint.h>

#include "cpu.h"
#include "errno.h"
#include "platform.h"
#include "rpc_internal.h"
#include "string.h"
#include "tensor_interrupt.h"
#include "trap.h"

#define TENSOR_INTR_ENCODING(address, read, reg) \
    (0x0bU | (5U << 23) | \
     ((uint32_t)(address) << 13) | ((uint32_t)(read) << 12) | \
     ((uint32_t)(reg) << 7))

struct tensor_interrupt_record_buffer {
    _Alignas(CONFIG_CACHE_LINE_SIZE) struct tensor_interrupt_record record;
};

static struct tensor_interrupt_record_buffer
    tensor_interrupt_records[CONFIG_HART_COUNT][2]
    __attribute__((section(".tensor_interrupt_records")));

static struct tensor_interrupt_record *tensor_interrupt_record_get(uint8_t id)
{
    uintptr_t hart = platform_hart_id();

    if (!id || id > 2U || hart >= CONFIG_HART_COUNT)
        platform_halt();
    return &tensor_interrupt_records[hart][id - 1U].record;
}

void tensor_interrupt_report_complete(void)
{
    struct tensor_interrupt_state *state = &cpu_self()->tensor_interrupt;

    if (!state->outbound)
        platform_halt();
    state->outbound = 0;
}

#define TENSOR_INTR_READ(address)                                             \
    ({                                                                       \
        register uint64_t result __asm__("a0");                              \
        __asm__ volatile(".insn 4, %1"                                      \
                         : "=r"(result)                                      \
                         : "i"(TENSOR_INTR_ENCODING(address, 1, 10))         \
                         : "memory");                                       \
        result;                                                              \
    })

#define TENSOR_BARRIER_ENCODING(unit, reg) \
    (0x0bU | ((uint32_t)(unit) << 13) | (1U << 12) | \
     ((uint32_t)(reg) << 7))

#define TENSOR_BARRIER(unit)                                                  \
    do {                                                                     \
        register uint64_t result __asm__("a0");                             \
        __asm__ volatile(".insn 4, %1"                                      \
                         : "=r"(result)                                     \
                         : "i"(TENSOR_BARRIER_ENCODING(unit, 10))           \
                         : "memory");                                       \
    } while (0)

static inline void tensor_interrupt_barrier(void)
{
    TENSOR_BARRIER(1);
    TENSOR_BARRIER(2);
    TENSOR_BARRIER(3);
    TENSOR_BARRIER(4);
}

static inline void tensor_interrupt_clear_pending(void)
{
    register uint64_t value __asm__("a0") = 1;

    __asm__ volatile(".insn 4, %1"
                     :
                     : "r"(value),
                       "i"(TENSOR_INTR_ENCODING(0, 0, 10))
                     : "memory");
}

void tensor_interrupt_handle(const struct trap_frame *frame)
{
    struct tensor_interrupt_state *state;
    struct tensor_interrupt_record *record;
    uint64_t control;
    size_t i;

    if (!frame || !current)
        platform_halt();
    tensor_interrupt_barrier();
    state = &cpu_self()->tensor_interrupt;
    control = TENSOR_INTR_READ(0);
    if (state->pending) {
        record = tensor_interrupt_record_get(state->pending);
        ++record->dropped_reports;
        tensor_interrupt_clear_pending();
        return;
    }
    state->pending = state->outbound == 1U ? 2U : 1U;
    record = tensor_interrupt_record_get(state->pending);
    memset(record, 0, sizeof(*record));
    record->hart_id = (uint32_t)platform_hart_id();
    record->thread_id = current->id;
    record->thread_type = current->type;
    record->thread_state = current->state;
    record->recent_count = (uint32_t)((control >> 9) & 0x1fU);
    record->tensor_context = current->tensor_context;
    record->control = control;
    record->interrupt_count = TENSOR_INTR_READ(33);
    record->recent[0] = TENSOR_INTR_READ(17);
    record->recent[1] = TENSOR_INTR_READ(18);
    record->recent[2] = TENSOR_INTR_READ(19);
    record->recent[3] = TENSOR_INTR_READ(20);
    record->recent[4] = TENSOR_INTR_READ(21);
    record->recent[5] = TENSOR_INTR_READ(22);
    record->recent[6] = TENSOR_INTR_READ(23);
    record->recent[7] = TENSOR_INTR_READ(24);
    record->recent[8] = TENSOR_INTR_READ(25);
    record->recent[9] = TENSOR_INTR_READ(26);
    record->recent[10] = TENSOR_INTR_READ(27);
    record->recent[11] = TENSOR_INTR_READ(28);
    record->recent[12] = TENSOR_INTR_READ(29);
    record->recent[13] = TENSOR_INTR_READ(30);
    record->recent[14] = TENSOR_INTR_READ(31);
    record->recent[15] = TENSOR_INTR_READ(32);
    record->mstatus = frame->mstatus;
    record->mepc = frame->mepc;
    record->mcause = frame->mcause;
    record->mtval = frame->mtval;
    record->fcsr = frame->fcsr;
    for (i = 0; i < 32; ++i)
        record->x[i] = frame->x[i];
    tensor_interrupt_clear_pending();
    rpc_send_notify();
}

void tensor_interrupt_drain(void)
{
    struct tensor_interrupt_state *state = &cpu_self()->tensor_interrupt;
    struct tensor_interrupt_record *record;
    uint8_t pending;
    int status;

    if (!state->pending || state->outbound ||
        !rpc_tensor_interrupt_ready())
        return;
    pending = state->pending;
    record = tensor_interrupt_record_get(pending);
    state->outbound = pending;
    state->pending = 0;
    status = rpc_report_tensor_interrupt(record);
    if (status < 0)
        platform_halt();
}
