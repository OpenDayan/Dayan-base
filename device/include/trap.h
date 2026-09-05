#ifndef NPU_TRAP_H
#define NPU_TRAP_H

#define TRAP_FRAME_X0          0
#define TRAP_FRAME_X1          8
#define TRAP_FRAME_X2         16
#define TRAP_FRAME_X3         24
#define TRAP_FRAME_X4         32
#define TRAP_FRAME_X5         40
#define TRAP_FRAME_X6         48
#define TRAP_FRAME_X7         56
#define TRAP_FRAME_X8         64
#define TRAP_FRAME_X9         72
#define TRAP_FRAME_X10        80
#define TRAP_FRAME_X11        88
#define TRAP_FRAME_X12        96
#define TRAP_FRAME_X13       104
#define TRAP_FRAME_X14       112
#define TRAP_FRAME_X15       120
#define TRAP_FRAME_X16       128
#define TRAP_FRAME_X17       136
#define TRAP_FRAME_X18       144
#define TRAP_FRAME_X19       152
#define TRAP_FRAME_X20       160
#define TRAP_FRAME_X21       168
#define TRAP_FRAME_X22       176
#define TRAP_FRAME_X23       184
#define TRAP_FRAME_X24       192
#define TRAP_FRAME_X25       200
#define TRAP_FRAME_X26       208
#define TRAP_FRAME_X27       216
#define TRAP_FRAME_X28       224
#define TRAP_FRAME_X29       232
#define TRAP_FRAME_X30       240
#define TRAP_FRAME_X31       248
#define TRAP_FRAME_MSTATUS   256
#define TRAP_FRAME_MEPC      264
#define TRAP_FRAME_MCAUSE    272
#define TRAP_FRAME_MTVAL     280
#define TRAP_FRAME_FCSR      288
#define TRAP_FRAME_RESERVED  296
#define TRAP_FRAME_SIZE      304

#ifndef __ASSEMBLER__

#include <stddef.h>
#include <stdint.h>

struct trap_frame {
    uintptr_t x[32];
    uintptr_t mstatus;
    uintptr_t mepc;
    uintptr_t mcause;
    uintptr_t mtval;
    uintptr_t fcsr;
    uintptr_t reserved;
};

struct trap_record {
    uint32_t hart_id;
    uint32_t thread_id;
    uint32_t thread_type;
    uint32_t thread_state;
    uint64_t mstatus;
    uint64_t mepc;
    uint64_t mcause;
    uint64_t mtval;
    uint64_t fcsr;
    uint64_t x[32];
    uint32_t host_proc_id;
    uint32_t parent_proc_id;
    uint64_t cookie;
};

struct trap_state {
    volatile uint8_t active;
};

_Static_assert(offsetof(struct trap_frame, x[1]) == TRAP_FRAME_X1,
               "trap x1 offset");
_Static_assert(offsetof(struct trap_frame, x[31]) == TRAP_FRAME_X31,
               "trap x31 offset");
_Static_assert(offsetof(struct trap_frame, mstatus) == TRAP_FRAME_MSTATUS,
               "trap mstatus offset");
_Static_assert(offsetof(struct trap_frame, fcsr) == TRAP_FRAME_FCSR,
               "trap fcsr offset");
_Static_assert(sizeof(struct trap_frame) == TRAP_FRAME_SIZE,
               "trap frame size");
_Static_assert((TRAP_FRAME_SIZE & 15) == 0,
               "trap frame must preserve stack alignment");
_Static_assert(sizeof(struct trap_record) == 328,
               "Host trap record ABI must stay fixed");
_Static_assert(offsetof(struct trap_record, hart_id) == 0,
               "trap record hart offset");
_Static_assert(offsetof(struct trap_record, mstatus) == 16,
               "trap record CSR offset");
_Static_assert(offsetof(struct trap_record, x) == 56,
               "trap record GPR offset");
_Static_assert(offsetof(struct trap_record, host_proc_id) == 312,
               "trap record proc identity offset");

void trap_handle(const struct trap_frame *frame);
_Noreturn void trap_exit(void);
void trap_report_complete(void);

#endif
#endif
