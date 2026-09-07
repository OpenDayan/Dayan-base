#include <stddef.h>
#include <stdint.h>

#include "config.h"
#include "platform.h"
#include "platform_regs.h"
#include "resource_scheduler.h"

_Static_assert(CONFIG_HART_COUNT <= 32,
               "RPC interrupt controller has only 32 lines");

static volatile uint32_t *rpc_interrupt_reg(uintptr_t offset)
{
    return (volatile uint32_t *)((uintptr_t)CONFIG_CONTROL_BASE + offset);
}

static volatile uint64_t *rpc_mailbox(uintptr_t offset)
{
    uintptr_t hart = platform_hart_id();

    return (volatile uint64_t *)((uintptr_t)CONFIG_CONTROL_BASE + offset +
                                 hart * sizeof(uint64_t));
}

static volatile uint32_t *rpc_msip(void)
{
    uintptr_t hart = platform_hart_id();

    return rpc_interrupt_reg(DAYAN_CLINT_MSIP_BASE +
                             hart * sizeof(uint32_t));
}

static uint32_t rpc_interrupt_mask(void)
{
    return (uint32_t)1U << platform_hart_id();
}

void platform_init(void)
{
    platform_cache_init();
}

uintptr_t platform_hart_id(void)
{
    uintptr_t hartid;

    __asm__ volatile("csrr %0, mhartid" : "=r"(hartid));
    return hartid;
}

__attribute__((weak)) void platform_cache_init(void)
{
}

__attribute__((weak)) void platform_cache_flush(const void *addr, size_t size)
{
    uintptr_t line;
    uintptr_t last;

    if (!size)
        return;
    line = (uintptr_t)addr & ~(uintptr_t)(CONFIG_CACHE_LINE_SIZE - 1);
    last = ((uintptr_t)addr + size - 1) &
           ~(uintptr_t)(CONFIG_CACHE_LINE_SIZE - 1);

    for (;;) {
        __asm__ volatile(".insn i 0x0f, 2, x0, %0, 0x002"
                         : : "r"(line) : "memory");
        if (line == last)
            break;
        line += CONFIG_CACHE_LINE_SIZE;
    }
}

__attribute__((weak)) void platform_cache_invalidate(void *addr, size_t size)
{
    uintptr_t line;
    uintptr_t last;

    if (!size)
        return;
    line = (uintptr_t)addr & ~(uintptr_t)(CONFIG_CACHE_LINE_SIZE - 1);
    last = ((uintptr_t)addr + size - 1) &
           ~(uintptr_t)(CONFIG_CACHE_LINE_SIZE - 1);

    for (;;) {
        __asm__ volatile(".insn i 0x0f, 2, x0, %0, 0x000"
                         : : "r"(line) : "memory");
        if (line == last)
            break;
        line += CONFIG_CACHE_LINE_SIZE;
    }
}

void platform_data_fence(void)
{
    __asm__ volatile("fence rw, rw" ::: "memory");
}

void platform_instruction_fence(void)
{
    __asm__ volatile("fence.i" ::: "memory");
}

void platform_rpc_init(void)
{
    uint32_t mask = rpc_interrupt_mask();

    /* Do not publish BL1 readiness until every Device worker is initialized. */
    __asm__ volatile("fence iorw, iorw" ::: "memory");
    *rpc_interrupt_reg(DAYAN_IRQ_STATUS_CLR) = mask;
    __asm__ volatile("fence iorw, iorw" ::: "memory");
}

void platform_rpc_ready(void)
{
    uint32_t mask = rpc_interrupt_mask();

    /* Publish all ordinary-memory initialization before the Host sees ready. */
    __asm__ volatile("fence iorw, iorw" ::: "memory");
    *rpc_interrupt_reg(DAYAN_IRQ_ENABLE_SET) = mask;
    __asm__ volatile("fence iorw, iorw" ::: "memory");
}

_Noreturn void platform_rpc_halt(void)
{
    uint32_t mask = rpc_interrupt_mask();
    void (*bootrom)(void) = (void (*)(void))DAYAN_BOOTROM_RESET_VECTOR;

    *rpc_interrupt_reg(DAYAN_IRQ_ENABLE_CLR) = mask;
    __asm__ volatile("fence iorw, iorw" ::: "memory");
    __asm__ volatile("csrci mstatus, 8" ::: "memory");
    /*
     * This Hart is about to re-enter BL0 without resetting the tensor block.
     * Its other permanent workers are stopped in switch_context, so discard
     * their already-issued scheduling tokens before initializing the Local
     * Resource table.  The next BL1 instance must not inherit runnable tp
     * values from this one.
     */
    while (__builtin_riscv_tx_base_tcsr_read(0, PRODUCE_BAR_GLOBAL))
        ;
    resource_scheduler_local_init(UINT32_MAX);
    __asm__ volatile("fence.i" ::: "memory");
    /* Bootrom consumes and acknowledges the pending opcode-0 request. */
    bootrom();
    platform_halt();
}

uint64_t platform_rpc_receive(void)
{
    uint64_t command;

    __asm__ volatile("fence iorw, iorw" ::: "memory");
    command = *rpc_mailbox(DAYAN_DEVICE_MAILBOX_BASE);
    __asm__ volatile("fence iorw, iorw" ::: "memory");
    return command;
}

void platform_rpc_receive_ack(void)
{
    __asm__ volatile("fence iorw, iorw" ::: "memory");
    *rpc_msip() = 0;
    __asm__ volatile("fence iorw, iorw" ::: "memory");
}

static int rpc_send_ready(void)
{
    uint32_t status;

    __asm__ volatile("fence iorw, iorw" ::: "memory");
    status = *rpc_interrupt_reg(DAYAN_IRQ_STATUS);
    __asm__ volatile("fence iorw, iorw" ::: "memory");
    return !(status & rpc_interrupt_mask());
}

void platform_rpc_send(uint64_t command)
{
    while (!rpc_send_ready())
        ;
    *rpc_mailbox(DAYAN_IRQ_MAILBOX_BASE) = command;
    __asm__ volatile("fence iorw, iorw" ::: "memory");
    *rpc_interrupt_reg(DAYAN_IRQ_STATUS_SET) = rpc_interrupt_mask();
    __asm__ volatile("fence iorw, iorw" ::: "memory");
    while (!rpc_send_ready())
        ;
}

__attribute__((weak)) void platform_idle(void)
{
    __asm__ volatile("wfi");
}

_Noreturn void platform_halt(void)
{
    for (;;)
        platform_idle();
}
