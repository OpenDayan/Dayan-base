#include <assert.h>
#include <stddef.h>
#include <string.h>

#include "rpc.h"
#include "sche.h"
#include "stdio.h"

static struct proc test_proc;
static char sent[128];
static size_t sent_size;
static size_t calls;
static size_t attempts;
static int fail_write;

struct proc *proc_self(void)
{
    return &test_proc;
}

long rpc_write(const void *data, size_t size)
{
    ++attempts;
    if (fail_write)
        return -1;
    assert(sent_size + size <= sizeof(sent));
    memcpy(sent + sent_size, data, size);
    sent_size += size;
    ++calls;
    return 0;
}

int main(void)
{
    char buffer[8];
    char formatted[16];

    test_proc.thread.id = 7;
    test_proc.printf_buffer = buffer;
    test_proc.printf_capacity = 4;
    test_proc.printf_size = 0;

    assert(printf("abc") == 3);
    assert(!calls && test_proc.printf_size == 3);
    assert(putchar('d') == 'd');
    assert(calls == 1 && test_proc.printf_size == 0);
    assert(sent_size == 4 && !memcmp(sent, "abcd", 4));

    assert(puts("xy") == 0);
    assert(calls == 1 && test_proc.printf_size == 3);
    assert(fflush(stdout) == 0);
    assert(calls == 2 && test_proc.printf_size == 0);
    assert(sent_size == 7 && !memcmp(sent + 4, "xy\n", 3));

    assert(printf("%04x", 0x2a) == 4);
    assert(calls == 3 && test_proc.printf_size == 0);
    assert(sent_size == 11 && !memcmp(sent + 7, "002a", 4));

    fail_write = 1;
    assert(printf("FAIL") == -1);
    assert(attempts == 4 && calls == 3);
    assert(test_proc.printf_size == 4);
    fail_write = 0;
    assert(fflush(0) == 0);
    assert(calls == 4 && test_proc.printf_size == 0);
    assert(sent_size == 15 && !memcmp(sent + 11, "FAIL", 4));
    assert(fflush((FILE *)2) == EOF);

    test_proc.printf_capacity = sizeof(buffer);
    assert(printf("a") == 1);
    assert(printf("b") == 1);
    assert(calls == 4 && test_proc.printf_size == 2);
    assert(fflush(stdout) == 0);
    assert(calls == 5 && !memcmp(sent + 15, "ab", 2));

    assert(snprintf(formatted, sizeof(formatted), "%d %s", 12, "ok") == 5);
    assert(!strcmp(formatted, "12 ok"));

    test_proc.printf_buffer = 0;
    test_proc.printf_capacity = 0;
    assert(printf("x") == -1);
    assert(putchar('x') == EOF);
    assert(fflush(stdout) == 0);
    return 0;
}
