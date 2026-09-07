#include <stddef.h>
#include <stdint.h>

#include "config.h"
#include "platform.h"
#include "platform_regs.h"

#ifndef CONFIG_UART_BASE
#define CONFIG_UART_BASE 0x40600000UL
#endif

#define UART_TX_FIFO 0x04U
#define UART_STATUS  0x08U
#define UART_TX_FULL 0x08U

__attribute__((weak)) long platform_print(const void *data, size_t size)
{
    const unsigned char *text = data;
    volatile uint32_t *tx = (void *)(CONFIG_UART_BASE + UART_TX_FIFO);
    volatile uint32_t *status = (void *)(CONFIG_UART_BASE + UART_STATUS);

    for (size_t i = 0; i < size; ++i) {
        while (*status & UART_TX_FULL)
            ;
        *tx = text[i];
        __asm__ volatile("fence iorw, iorw" ::: "memory");
    }
    return (long)size;
}

__attribute__((weak)) void platform_test_init(void *coherent_backing)
{
    volatile uint64_t *backing = (void *)((uintptr_t)CONFIG_CONTROL_BASE +
                                        DAYAN_ATOMIC_CACHE_BACKING_BASE);
    volatile uint32_t *status = (void *)((uintptr_t)CONFIG_CONTROL_BASE +
                                       DAYAN_ATOMIC_CACHE_STATUS);

    *backing = (uintptr_t)coherent_backing;
    __asm__ volatile("fence iorw, iorw" ::: "memory");
    for (;;) {
        uint32_t value = *status;

        if (value & 12U) {
            static const char message[] = "[test] atomic cache setup failed\n";

            (void)platform_print(message, sizeof(message) - 1U);
            platform_halt();
        }
        if (!(value & 1U) && (value & 2U))
            break;
    }
}

__attribute__((weak)) void platform_test_complete(int64_t result)
{
    (void)result;
}
