#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "cpu.h"
#include "errno.h"
#include "mem.h"
#include "rpc.h"

#define REGION_BYTES 4096
#define SLOT_COUNT   128

struct cpu test_cpu;

static _Alignas(64) unsigned char initial_region[REGION_BYTES];
static _Alignas(64) unsigned char host_region[REGION_BYTES];
static _Alignas(64) unsigned char invalid_region[32];
static int acquired;
static int released;
static int invalid_next;
static int invalid_outstanding;
static int rollback_released;
static int host_cleaned;
static int host_invalidated;

void platform_cache_clean(const void *addr, size_t size)
{
    assert(addr == host_region && size == sizeof(host_region));
    ++host_cleaned;
}

void platform_cache_invalidate(void *addr, size_t size)
{
    assert(addr == host_region && size == sizeof(host_region));
    ++host_invalidated;
}

long rpc_heap_acquire(size_t size, struct rpc_heap_response *result)
{
    if (invalid_next) {
        invalid_next = 0;
        invalid_outstanding = 1;
        result->addr = (uintptr_t)invalid_region;
        result->size = sizeof(invalid_region);
        return 0;
    }
    if (acquired || size > sizeof(host_region))
        return -ENOMEM;
    acquired = 1;
    result->addr = (uintptr_t)host_region;
    result->size = sizeof(host_region);
    return 0;
}

long rpc_heap_release(uintptr_t addr, size_t size)
{
    if (addr == (uintptr_t)invalid_region) {
        assert(invalid_outstanding);
        assert(size == sizeof(invalid_region));
        invalid_outstanding = 0;
        ++rollback_released;
        return 0;
    }
    assert(addr == (uintptr_t)host_region);
    assert(size == sizeof(host_region));
    ++released;
    return 0;
}

static void basic_test(void)
{
    unsigned char *first = mem_alloc(93);
    unsigned char *second = mem_alloc(517);
    unsigned char *resized;
    size_t i;

    assert(first && second);
    assert(((uintptr_t)first & 15U) == 0);
    assert(((uintptr_t)second & 15U) == 0);
    for (i = 0; i < 93; ++i)
        first[i] = (unsigned char)(i ^ 0x5aU);

    resized = mem_resize(first, 701);
    assert(resized);
    for (i = 0; i < 93; ++i)
        assert(resized[i] == (unsigned char)(i ^ 0x5aU));
    mem_free(second);
    mem_free(resized);
}

static void growth_test(size_t initial_capacity)
{
    void *slots[SLOT_COUNT] = { 0 };
    size_t sizes[SLOT_COUNT] = { 0 };
    size_t count = 0;
    size_t i;

    while (count < SLOT_COUNT) {
        size_t size = 73 + count % 11 * 19;
        unsigned char *ptr = mem_alloc(size);

        if (!ptr)
            break;
        assert(mem_size(ptr) >= size);
        memset(ptr, (int)(count + 1), size);
        slots[count] = ptr;
        sizes[count] = size;
        ++count;
    }
    assert(acquired);
    assert(count < SLOT_COUNT);

    for (i = 0; i < count; ++i) {
        size_t index = (i * 37) % count;
        unsigned char *ptr;
        size_t j;

        if (!slots[index])
            continue;
        ptr = slots[index];
        for (j = 0; j < sizes[index]; ++j)
            assert(ptr[j] == (unsigned char)(index + 1));
        mem_free(ptr);
        slots[index] = 0;
    }
    for (i = 0; i < count; ++i) {
        if (slots[i])
            mem_free(slots[i]);
    }

    assert(released == 1);
    assert(host_cleaned == 1);
    assert(host_invalidated == 2);
    assert(mem_capacity() == initial_capacity);
    assert(mem_available() == initial_capacity);
}

static void invalid_growth_test(size_t initial_capacity)
{
    invalid_next = 1;
    assert(!mem_alloc(REGION_BYTES * 2));
    assert(!invalid_outstanding);
    assert(rollback_released == 1);
    assert(!host_cleaned && !host_invalidated);
    assert(mem_capacity() == initial_capacity);
    assert(mem_available() == initial_capacity);
}

int main(void)
{
    size_t initial_capacity;

    assert(mem_init(initial_region, sizeof(initial_region)) == 0);
    assert(mem_alloc(0) == 0);
    initial_capacity = mem_capacity();
    assert(initial_capacity == mem_available());
    invalid_growth_test(initial_capacity);
    basic_test();
    assert(mem_available() == initial_capacity);
    growth_test(initial_capacity);
    return 0;
}
