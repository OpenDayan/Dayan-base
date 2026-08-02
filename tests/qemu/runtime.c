#include <stdint.h>

#define QEMU_UART_BASE       0x10000000UL
#define QEMU_UART_THR        0
#define QEMU_UART_LSR        5
#define QEMU_UART_TX_IDLE    0x20U
#define QEMU_TEST_BASE       0x00100000UL
#define QEMU_TEST_PASS       0x00005555U
#define QEMU_TEST_FAIL(code) (((uint32_t)(code) << 16) | 0x3333U)

static void console_putc(char value)
{
    volatile uint8_t *uart = (volatile uint8_t *)QEMU_UART_BASE;

    while (!(uart[QEMU_UART_LSR] & QEMU_UART_TX_IDLE))
        __asm__ volatile("nop");
    uart[QEMU_UART_THR] = (uint8_t)value;
}

static void console_puts(const char *text)
{
    while (*text)
        console_putc(*text++);
}

static void console_put_unsigned(unsigned value)
{
    char digits[10];
    unsigned count = 0;

    do {
        digits[count++] = (char)('0' + value % 10U);
        value /= 10U;
    } while (value);
    while (count)
        console_putc(digits[--count]);
}

__attribute__((noreturn))
void dayan_test_exit(int status)
{
    volatile uint32_t *test = (volatile uint32_t *)QEMU_TEST_BASE;

    if (status) {
        console_puts("dayan-qemu-test: FAIL\n");
        *test = QEMU_TEST_FAIL((unsigned)status);
    } else {
        console_puts("dayan-qemu-test: PASS\n");
        *test = QEMU_TEST_PASS;
    }
    for (;;)
        __asm__ volatile("wfi");
}

__attribute__((noreturn))
void dayan_test_assert_fail(const char *expression, const char *file,
                            unsigned line)
{
    console_puts("assertion failed: ");
    console_puts(expression);
    console_puts(" at ");
    console_puts(file);
    console_putc(':');
    console_put_unsigned(line);
    console_putc('\n');
    dayan_test_exit(1);
}
