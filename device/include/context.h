#ifndef NPU_CONTEXT_H
#define NPU_CONTEXT_H

#include "config.h"

#ifdef __ASSEMBLER__
.set cpu_local_shift, 0
.rept 63
    .if (CONFIG_CPU_LOCAL_SIZE >> cpu_local_shift) > 1
        .set cpu_local_shift, cpu_local_shift + 1
    .endif
.endr
#endif

#define THREAD_STATE_OFFSET     1
#define THREAD_SLOT_OFFSET      0
#define THREAD_SP_OFFSET        8
#define THREAD_PC_OFFSET       16
#define THREAD_TENSOR_CONTEXT_OFFSET 24
#define THREAD_RESOURCE_CONTEXT_OFFSET 32
#define THREAD_TOKEN_EXT_OFFSET 36
#define CPU_PROFILER_OFFSET     0
#define CPU_RPC_RECEIVE_OFFSET  8

#define SWITCH_FRAME_RA         0
#define SWITCH_FRAME_S0         8
#define SWITCH_FRAME_S1        16
#define SWITCH_FRAME_S2        24
#define SWITCH_FRAME_S3        32
#define SWITCH_FRAME_S4        40
#define SWITCH_FRAME_S5        48
#define SWITCH_FRAME_S6        56
#define SWITCH_FRAME_S7        64
#define SWITCH_FRAME_S8        72
#define SWITCH_FRAME_S9        80
#define SWITCH_FRAME_S10       88
#define SWITCH_FRAME_S11       96
#define SWITCH_FRAME_GP       104
#define SWITCH_FRAME_FCSR     112
#define SWITCH_FRAME_RESERVED 120
#define SWITCH_FRAME_SIZE     128

#define PROC_FRAME_ENTRY        0
#define PROC_FRAME_RA           8
#define PROC_FRAME_A0          16
#define PROC_FRAME_A1          24
#define PROC_FRAME_A2          32
#define PROC_FRAME_A3          40
#define PROC_FRAME_A4          48
#define PROC_FRAME_A5          56
#define PROC_FRAME_A6          64
#define PROC_FRAME_A7          72
#define PROC_FRAME_SHARED      80
#define PROC_FRAME_RESERVED    88
#define PROC_FRAME_SIZE        96

#define PROC_MAX_ARGS                      32
#define PROC_REGISTER_ARGS                  8
#define PROC_BOOTSTRAP_ARG_INSN_SIZE        4
#define PROC_BOOTSTRAP_PC_OFFSET(argc) \
    ((PROC_REGISTER_ARGS - ((argc) < PROC_REGISTER_ARGS ? (argc) : \
                             PROC_REGISTER_ARGS)) * PROC_BOOTSTRAP_ARG_INSN_SIZE)

#ifndef __ASSEMBLER__

#include <stddef.h>
#include <stdint.h>

struct switch_frame {
    uintptr_t ra;
    uintptr_t s0;
    uintptr_t s1;
    uintptr_t s2;
    uintptr_t s3;
    uintptr_t s4;
    uintptr_t s5;
    uintptr_t s6;
    uintptr_t s7;
    uintptr_t s8;
    uintptr_t s9;
    uintptr_t s10;
    uintptr_t s11;
    uintptr_t gp;
    uintptr_t fcsr;
    uintptr_t reserved;
};

struct proc_frame {
    uintptr_t entry;
    uintptr_t ra;
    uintptr_t args[PROC_REGISTER_ARGS];
    uintptr_t shared;
    uintptr_t reserved;
    /* Arguments 8..argc-1 follow this frame, at the kernel's entry sp. */
};

_Static_assert(offsetof(struct switch_frame, ra) == SWITCH_FRAME_RA,
               "switch frame ra offset");
_Static_assert(offsetof(struct switch_frame, s0) == SWITCH_FRAME_S0,
               "switch frame s0 offset");
_Static_assert(offsetof(struct switch_frame, s11) == SWITCH_FRAME_S11,
               "switch frame s11 offset");
_Static_assert(offsetof(struct switch_frame, gp) == SWITCH_FRAME_GP,
               "switch frame gp offset");
_Static_assert(offsetof(struct switch_frame, fcsr) == SWITCH_FRAME_FCSR,
               "switch frame fcsr offset");
_Static_assert(sizeof(struct switch_frame) == SWITCH_FRAME_SIZE,
               "switch frame size");
_Static_assert((SWITCH_FRAME_SIZE & 15) == 0,
               "switch frame must preserve stack alignment");
_Static_assert(offsetof(struct proc_frame, entry) == PROC_FRAME_ENTRY,
               "proc frame entry offset");
_Static_assert(offsetof(struct proc_frame, ra) == PROC_FRAME_RA,
               "proc frame ra offset");
_Static_assert(offsetof(struct proc_frame, args[0]) == PROC_FRAME_A0,
               "proc frame a0 offset");
_Static_assert(offsetof(struct proc_frame, args[7]) == PROC_FRAME_A7,
               "proc frame a7 offset");
_Static_assert(offsetof(struct proc_frame, shared) == PROC_FRAME_SHARED,
               "proc frame shared offset");
_Static_assert(sizeof(struct proc_frame) == PROC_FRAME_SIZE,
               "proc frame size");
_Static_assert((PROC_FRAME_SIZE & 15) == 0,
               "proc frame must preserve stack alignment");
_Static_assert(PROC_BOOTSTRAP_PC_OFFSET(0) == 32,
               "zero-argument bootstrap offset");
_Static_assert(PROC_BOOTSTRAP_PC_OFFSET(PROC_MAX_ARGS) == 0,
               "maximum-argument bootstrap offset");

#endif
#endif
