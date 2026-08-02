#include <stddef.h>
#include <stdint.h>

#include "config.h"
#include "errno.h"
#include "platform.h"

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
    // __asm__ volatile("fence.i" ::: "memory");
    // __asm__ volatile("fence rw, rw" ::: "memory");
}

__attribute__((weak)) void platform_cache_clean(const void *addr, size_t size)
{
    uintptr_t line;
    uintptr_t last;

    if (!size)
        return;
    line = (uintptr_t)addr & ~(uintptr_t)(CONFIG_CACHE_LINE_SIZE - 1);
    if (size - 1 > UINTPTR_MAX - (uintptr_t)addr)
        last = UINTPTR_MAX & ~(uintptr_t)(CONFIG_CACHE_LINE_SIZE - 1);
    else
        last = ((uintptr_t)addr + size - 1) &
               ~(uintptr_t)(CONFIG_CACHE_LINE_SIZE - 1);

    __asm__ volatile("fence rw, rw" ::: "memory");
    for (;;) {
        __asm__ volatile(".insn i 0x0f, 2, x0, %0, 0x002"
                         : : "r"(line) : "memory");
        if (line == last)
            break;
        line += CONFIG_CACHE_LINE_SIZE;
    }
    __asm__ volatile("fence rw, rw" ::: "memory");
}

__attribute__((weak)) void platform_cache_invalidate(void *addr, size_t size)
{
    uintptr_t line;
    uintptr_t last;

    if (!size)
        return;
    line = (uintptr_t)addr & ~(uintptr_t)(CONFIG_CACHE_LINE_SIZE - 1);
    if (size - 1 > UINTPTR_MAX - (uintptr_t)addr)
        last = UINTPTR_MAX & ~(uintptr_t)(CONFIG_CACHE_LINE_SIZE - 1);
    else
        last = ((uintptr_t)addr + size - 1) &
               ~(uintptr_t)(CONFIG_CACHE_LINE_SIZE - 1);

    __asm__ volatile("fence rw, rw" ::: "memory");
    for (;;) {
        __asm__ volatile(".insn i 0x0f, 2, x0, %0, 0x000"
                         : : "r"(line) : "memory");
        if (line == last)
            break;
        line += CONFIG_CACHE_LINE_SIZE;
    }
    __asm__ volatile("fence rw, rw" ::: "memory");
}

__attribute__((weak)) long platform_rpc(uint32_t service,
                                        const void *request,
                                        size_t request_size,
                                        void *response,
                                        size_t response_size)
{
    (void)service;
    (void)request;
    (void)request_size;
    (void)response;
    (void)response_size;
    return -ENOSYS;
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
