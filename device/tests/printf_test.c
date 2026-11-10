/***************************************************************************************
 * Copyright (c) 2026 Institute of Computing Technology, Chinese Academy of Sciences
 *
 * Dayan-base is licensed under the Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *
 *     http://license.coscl.org.cn/MulanPSL2
 *
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 *
 * See the Mulan PSL v2 for more details.
***************************************************************************************/

#include <assert.h>
#include <stddef.h>
#include <string.h>

#include "rpc.h"
#include "sche.h"
#include "stdio.h"

static struct thread test_thread;
static char *output_buffer;
static size_t output_capacity;
static size_t output_size;
static char sent[128];
static size_t sent_size;
static size_t calls;
static size_t attempts;
static int fail_write;

struct thread *thread_self(void)
{
    return &test_thread;
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

int thread_output_enabled(const struct thread *thread)
{
    return thread && output_buffer && output_capacity;
}

int thread_output_flush(struct thread *thread)
{
    if (!thread || output_size > output_capacity ||
        (!output_buffer != !output_capacity))
        return -1;
    if (!output_size)
        return 0;
    if (rpc_write(output_buffer, output_size) < 0)
        return -1;
    output_size = 0;
    return 0;
}

int thread_output_put(struct thread *thread, char ch)
{
    if (!thread_output_enabled(thread) || output_size > output_capacity)
        return -1;
    if (output_size == output_capacity && thread_output_flush(thread) < 0)
        return -1;
    output_buffer[output_size++] = ch;
    if (output_size == output_capacity && thread_output_flush(thread) < 0)
        return -1;
    return 0;
}

int main(void)
{
    char buffer[8];
    char formatted[16];

    test_thread.id = 7;
    output_buffer = buffer;
    output_capacity = 4;
    output_size = 0;

    assert(printf("abc") == 3);
    assert(!calls && output_size == 3);
    assert(putchar('d') == 'd');
    assert(calls == 1 && output_size == 0);
    assert(sent_size == 4 && !memcmp(sent, "abcd", 4));

    assert(puts("xy") == 0);
    assert(calls == 1 && output_size == 3);
    assert(fflush(stdout) == 0);
    assert(calls == 2 && output_size == 0);
    assert(sent_size == 7 && !memcmp(sent + 4, "xy\n", 3));

    assert(printf("%04x", 0x2a) == 4);
    assert(calls == 3 && output_size == 0);
    assert(sent_size == 11 && !memcmp(sent + 7, "002a", 4));

    fail_write = 1;
    assert(printf("FAIL") == -1);
    assert(attempts == 4 && calls == 3);
    assert(output_size == 4);
    fail_write = 0;
    assert(fflush(0) == 0);
    assert(calls == 4 && output_size == 0);
    assert(sent_size == 15 && !memcmp(sent + 11, "FAIL", 4));
    assert(fflush((FILE *)2) == EOF);

    output_capacity = sizeof(buffer);
    assert(printf("a") == 1);
    assert(printf("b") == 1);
    assert(calls == 4 && output_size == 2);
    assert(fflush(stdout) == 0);
    assert(calls == 5 && !memcmp(sent + 15, "ab", 2));

    assert(snprintf(formatted, sizeof(formatted), "%d %s", 12, "ok") == 5);
    assert(!strcmp(formatted, "12 ok"));

    output_buffer = 0;
    output_capacity = 0;
    assert(printf("x") == -1);
    assert(putchar('x') == EOF);
    assert(fflush(stdout) == 0);
    return 0;
}
