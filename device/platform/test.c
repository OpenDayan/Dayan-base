#include <stddef.h>
#include <stdint.h>

#include "config.h"
#include "dayan_test.h"
#include "errno.h"
#include "platform.h"
#include "platform_regs.h"
#include "string.h"

static void test_mailbox_flush(void)
{
    platform_cache_flush((const void *)(uintptr_t)DAYAN_TEST_MAILBOX_BASE,
                         DAYAN_TEST_MAILBOX_SIZE);
    platform_data_fence();
}

__attribute__((weak)) long platform_print(const void *data, size_t size)
{
    const unsigned char *text = data;
    volatile uint64_t *header = (void *)(uintptr_t)DAYAN_TEST_LOG_BASE;
    volatile unsigned char *buffer = (void *)(uintptr_t)DAYAN_TEST_LOG_DATA_ADDR;
    uint64_t written = header[0];
    size_t first = size > DAYAN_TEST_LOG_CAPACITY ? size - DAYAN_TEST_LOG_CAPACITY : 0;

    /* Local tests serialize output on Hart 0; retain the most recent bytes. */
    for (size_t i = first; i < size; ++i)
        buffer[(written + i) % DAYAN_TEST_LOG_CAPACITY] = text[i];
    platform_cache_flush((const void *)(uintptr_t)DAYAN_TEST_LOG_DATA_ADDR,
                         DAYAN_TEST_LOG_CAPACITY);
    platform_data_fence();
    header[0] = written + size;
    platform_cache_flush((const void *)(uintptr_t)DAYAN_TEST_LOG_BASE,
                         DAYAN_TEST_LOG_HEADER_SIZE);
    platform_data_fence();
    return (long)size;
}

__attribute__((weak)) void platform_test_init(void *coherent_backing)
{
    volatile uint64_t *backing = (void *)((uintptr_t)CONFIG_CONTROL_BASE +
                                        DAYAN_ATOMIC_CACHE_BACKING_BASE);
    volatile uint32_t *status = (void *)((uintptr_t)CONFIG_CONTROL_BASE +
                                       DAYAN_ATOMIC_CACHE_STATUS);
    (void)backing;
    (void)status;
    (void)coherent_backing;

    memset((void *)(uintptr_t)DAYAN_TEST_LOG_BASE, 0,
           DAYAN_TEST_LOG_HEADER_SIZE + DAYAN_TEST_LOG_CAPACITY);
    ((volatile uint64_t *)(uintptr_t)DAYAN_TEST_LOG_BASE)[1] = DAYAN_TEST_LOG_MAGIC;
    platform_cache_flush((const void *)(uintptr_t)DAYAN_TEST_LOG_BASE,
                         DAYAN_TEST_LOG_HEADER_SIZE + DAYAN_TEST_LOG_CAPACITY);
    *(volatile uint64_t *)(uintptr_t)DAYAN_TEST_ALIVE_ADDR = 0;
    *(volatile uint64_t *)(uintptr_t)DAYAN_TEST_DONE_ADDR = 0;
    *(volatile uint64_t *)(uintptr_t)DAYAN_TEST_RESULT_ADDR = 0;
    platform_data_fence();
    *(volatile uint64_t *)(uintptr_t)DAYAN_TEST_ALIVE_ADDR = DAYAN_TEST_ALIVE_VALUE;
    test_mailbox_flush();
}

__attribute__((weak)) void platform_test_complete(int64_t result)
{
    /* Publish the aggregate result before the observer can see DONE. */
    platform_data_fence();
    *(volatile uint64_t *)(uintptr_t)DAYAN_TEST_RESULT_ADDR = (uint64_t)result;
    platform_data_fence();
    *(volatile uint64_t *)(uintptr_t)DAYAN_TEST_DONE_ADDR = DAYAN_TEST_DONE_VALUE;
    test_mailbox_flush();
}
