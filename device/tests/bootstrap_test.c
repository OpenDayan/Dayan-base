#include <assert.h>
#include <stdint.h>
#include <string.h>

#include "cpu.h"
#include "rpc.h"
#include "sche.h"

_Alignas(64) uint8_t __cpu_start[CONFIG_CPU_LOCAL_SIZE];
static _Alignas(16) unsigned char stack[1024];
static uintptr_t observed[32];
static uintptr_t observed_sp, observed_gp;

extern void run_bootstrap(uintptr_t sp, uintptr_t pc);

/* Exit RPCs are unreachable: the trampoline supplies a test return address. */
long rpc_write_proc(uint32_t proc_id, const void *data, size_t size)
{
    (void)proc_id; (void)data; (void)size;
    assert(0);
    return -1;
}

long rpc_proc_exit(struct task_proc *task, int64_t result)
{
    (void)task; (void)result;
    assert(0);
    return -1;
}

static void entry0(void)
{
    __asm__ volatile("mv %0, sp\n\tmv %1, gp"
                     : "=r"(observed_sp), "=r"(observed_gp));
}

static void entry1(uintptr_t a0)
{
    __asm__ volatile("mv %0, sp\n\tmv %1, gp"
                     : "=r"(observed_sp), "=r"(observed_gp));
    observed[0] = a0;
}

static void entry8(uintptr_t a0, uintptr_t a1, uintptr_t a2, uintptr_t a3, uintptr_t a4, uintptr_t a5, uintptr_t a6, uintptr_t a7)
{
    __asm__ volatile("mv %0, sp\n\tmv %1, gp"
                     : "=r"(observed_sp), "=r"(observed_gp));
    observed[0] = a0;
    observed[1] = a1;
    observed[2] = a2;
    observed[3] = a3;
    observed[4] = a4;
    observed[5] = a5;
    observed[6] = a6;
    observed[7] = a7;
}

static void entry9(uintptr_t a0, uintptr_t a1, uintptr_t a2, uintptr_t a3, uintptr_t a4, uintptr_t a5, uintptr_t a6, uintptr_t a7, uintptr_t a8)
{
    __asm__ volatile("mv %0, sp\n\tmv %1, gp"
                     : "=r"(observed_sp), "=r"(observed_gp));
    observed[0] = a0;
    observed[1] = a1;
    observed[2] = a2;
    observed[3] = a3;
    observed[4] = a4;
    observed[5] = a5;
    observed[6] = a6;
    observed[7] = a7;
    observed[8] = a8;
}

static void entry32(uintptr_t a0, uintptr_t a1, uintptr_t a2, uintptr_t a3, uintptr_t a4, uintptr_t a5, uintptr_t a6, uintptr_t a7, uintptr_t a8, uintptr_t a9, uintptr_t a10, uintptr_t a11, uintptr_t a12, uintptr_t a13, uintptr_t a14, uintptr_t a15, uintptr_t a16, uintptr_t a17, uintptr_t a18, uintptr_t a19, uintptr_t a20, uintptr_t a21, uintptr_t a22, uintptr_t a23, uintptr_t a24, uintptr_t a25, uintptr_t a26, uintptr_t a27, uintptr_t a28, uintptr_t a29, uintptr_t a30, uintptr_t a31)
{
    __asm__ volatile("mv %0, sp\n\tmv %1, gp"
                     : "=r"(observed_sp), "=r"(observed_gp));
    observed[0] = a0;
    observed[1] = a1;
    observed[2] = a2;
    observed[3] = a3;
    observed[4] = a4;
    observed[5] = a5;
    observed[6] = a6;
    observed[7] = a7;
    observed[8] = a8;
    observed[9] = a9;
    observed[10] = a10;
    observed[11] = a11;
    observed[12] = a12;
    observed[13] = a13;
    observed[14] = a14;
    observed[15] = a15;
    observed[16] = a16;
    observed[17] = a17;
    observed[18] = a18;
    observed[19] = a19;
    observed[20] = a20;
    observed[21] = a21;
    observed[22] = a22;
    observed[23] = a23;
    observed[24] = a24;
    observed[25] = a25;
    observed[26] = a26;
    observed[27] = a27;
    observed[28] = a28;
    observed[29] = a29;
    observed[30] = a30;
    observed[31] = a31;
}

int main(void)
{
    static const size_t counts[] = {0, 1, 8, 9, 32};
    static const thread_fn entries[] = {
        (thread_fn)entry0, (thread_fn)entry1, (thread_fn)entry8,
        (thread_fn)entry9, (thread_fn)entry32,
    };
    uintptr_t args[32];
    struct thread thread;

    for (size_t i = 0; i < 32; ++i)
        args[i] = 0xfedcba9876543200ULL + i;
    for (size_t test = 0; test < sizeof(counts) / sizeof(counts[0]); ++test) {
        size_t argc = counts[test];
        memset(stack, 0xa5, sizeof(stack));
        memset(&thread, 0xa5, sizeof(thread));
        memset(observed, 0, sizeof(observed));
        thread_start(&thread, RPC_PROFILE_SLOT_USER_BASE, THREAD_TYPE_TASK_PROC,
                     "args", entries[test], stack, sizeof(stack),
                     (void *)0x12345000, argc, argc ? args : 0);
        struct proc_frame *frame = (struct proc_frame *)thread.sp;
        uintptr_t *stack_args = (uintptr_t *)(frame + 1);
        assert(!(thread.sp & 15));
        assert(thread.generation == test + 1);
        assert(!thread.tensor_context && !thread.resource_context);
        assert(!thread.token_ext && !thread.profile_defined &&
               thread.parent == THREAD_SLOT_NONE);
        assert(!thread.node.prev && !thread.node.next);
        assert(!strcmp(thread.name, "args") && !thread.name[15]);
        for (size_t i = 0; i < argc; ++i)
            assert((i < 8 ? frame->args[i] : stack_args[i - 8]) == args[i]);
        if (argc <= 8)
            assert((uintptr_t)stack_args == (uintptr_t)(stack + sizeof(stack)));
        assert(stack[0] == 0xa5);
        run_bootstrap(thread.sp, thread.pc);
        assert(!(observed_sp & 15) && observed_gp == 0x12345000);
        for (size_t i = 0; i < argc; ++i)
            assert(observed[i] == args[i]);
        for (size_t i = argc; i < 32; ++i)
            assert(!observed[i]);
        assert(stack[0] == 0xa5);
    }
    return 0;
}
