#include <dayan_test.h>
#include <stdio.h>
#include <stdlib.h>

#define ALLOCATION_SIZE (2U * 1024U * 1024U)
#define ELEMENTS 64U

static void check(int condition)
{
    if (!condition)
        dayan_thread_exit(-1);
}

static void transform(void *argument)
{
    const uint32_t *input = argument;
    uint32_t *output = (void *)(uintptr_t)(DAYAN_TEST_DATA_BASE + 4096U);

    for (unsigned i = 0; i < ELEMENTS; ++i)
        output[i] = input[i] * 3U + 7U;
    *(uint32_t *)shared = ELEMENTS;
    printf("child: %u values\n", ELEMENTS);
}

static void first(uintptr_t input_address, uintptr_t marker)
{
    uint32_t *input = (void *)input_address;
    unsigned char *allocation = malloc(ALLOCATION_SIZE);
    uint64_t *saved = (void *)(uintptr_t)(DAYAN_TEST_DATA_BASE + 8192U);
    int child;
    int64_t result;

    check(marker == 0x12345678U && allocation != 0);
    check((uintptr_t)allocation + ALLOCATION_SIZE <= DAYAN_TEST_DATA_BASE);
    check(dayan_uram_size() == DAYAN_URAM_ALLOCATION_GRANULE);
    check(*(uint32_t *)shared == 0);
    allocation[0] = 17;
    allocation[ALLOCATION_SIZE - 1U] = 29;
    *saved = (uintptr_t)allocation;
    for (unsigned i = 0; i < ELEMENTS; ++i)
        input[i] = i * i;
    child = dayan_create_thread("transform", transform, input, 0);
    check(child > 0);
    check(dayan_thread_join((uint32_t)child, &result) == 0 && result == 0);
    check(*(uint32_t *)shared == ELEMENTS);
    check(allocation[0] == 17 && allocation[ALLOCATION_SIZE - 1U] == 29);
    *(volatile uint32_t *)(uintptr_t)(DAYAN_TEST_DATA_BASE +
                                     DAYAN_TEST_DATA_SIZE - 4U) = 0xfeedfaceU;
    free(allocation);
    printf("first: arguments, child, heap and reserved data verified\n");
}

static void second(void)
{
    const uint32_t *output = (void *)(uintptr_t)(DAYAN_TEST_DATA_BASE + 4096U);
    const uint64_t *saved = (void *)(uintptr_t)(DAYAN_TEST_DATA_BASE + 8192U);
    unsigned char *allocation = malloc(ALLOCATION_SIZE);

    check(allocation != 0 && (uintptr_t)allocation > *saved);
    check(*(uint32_t *)shared == 0);
    for (unsigned i = 0; i < ELEMENTS; ++i)
        check(output[i] == i * i * 3U + 7U);
    check(*(volatile uint32_t *)(uintptr_t)(DAYAN_TEST_DATA_BASE +
                                          DAYAN_TEST_DATA_SIZE - 4U) ==
          0xfeedfaceU);
    dayan_cache_flush((const void *)(uintptr_t)DAYAN_TEST_DATA_BASE,
                      ELEMENTS * sizeof(uint32_t));
    dayan_cache_flush(output, ELEMENTS * sizeof(*output));
    dayan_cache_flush((const void *)(uintptr_t)(DAYAN_TEST_DATA_BASE +
                                              DAYAN_TEST_DATA_SIZE - 4U),
                      sizeof(uint32_t));
    free(allocation);
    printf("second: %u reference values and sequential completion verified\n",
           ELEMENTS);
}

DAYAN_TEST(first_test, {
    .entry = (void (*)(void))first,
    .resources = {
        .max_child_threads = 1,
        .max_tasks_per_core = 1,
        .lsu_tcsr_count = 1,
        .atomic_cache_size = 64,
        .uram_size = DAYAN_URAM_ALLOCATION_GRANULE,
    },
    .args = {DAYAN_TEST_DATA_BASE, 0x12345678U},
    .argc = 2,
    .name = "first",
});

DAYAN_TEST(second_test, {
    .entry = second,
    .resources = {.max_tasks_per_core = 1, .atomic_cache_size = 64},
    .name = "second",
});
