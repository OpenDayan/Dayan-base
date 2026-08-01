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
static int acquired;
static int released;

long rpc_call(uint32_t service,
              const void *request, size_t request_size,
              void *response, size_t response_size)
{
    if (service == RPC_HEAP_ACQUIRE) {
        const struct rpc_heap_request *get = request;
        struct rpc_heap_response *result = response;

        assert(request_size == sizeof(*get));
        assert(response_size == sizeof(*result));
        if (acquired || get->size > sizeof(host_region))
            return -ENOMEM;
        acquired = 1;
        result->addr = (uintptr_t)host_region;
        result->size = sizeof(host_region);
        return 0;
    }

    if (service == RPC_HEAP_RELEASE) {
        const struct rpc_heap_release *put = request;

        assert(request_size == sizeof(*put));
        assert(!response && !response_size);
        assert(put->addr == (uintptr_t)host_region);
        assert(put->size == sizeof(host_region));
        ++released;
        return 0;
    }
    return -ENOSYS;
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
    basic_test();
    assert(mem_available() == initial_capacity);
    growth_test(initial_capacity);
    return 0;
}
