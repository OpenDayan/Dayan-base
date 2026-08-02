#include <assert.h>
#include <errno.h>
#include <stdint.h>

#include "runtime_internal.h"

int main(void)
{
    struct npu_arena arena;
    uint64_t first;
    uint64_t second;

    npu_arena_init(&arena, 0x1000, 0x1000);
    assert(npu_arena_alloc(&arena, 0x100, 0x100, NPU_ALLOC_USER,
                           UINT32_MAX, &first) == 0);
    assert(first == 0x1000);
    assert(npu_arena_reserve(&arena, 0x1400, 0x100,
                             NPU_ALLOC_FIRMWARE, UINT32_MAX) == 0);
    assert(npu_arena_alloc(&arena, 0x280, 0x80, NPU_ALLOC_HEAP,
                           2, &second) == 0);
    assert(second == 0x1100);
    assert(npu_arena_reserve(&arena, 0x1200, 0x100,
                             NPU_ALLOC_USER, UINT32_MAX) == -EBUSY);
    assert(npu_arena_free(&arena, second, NPU_ALLOC_HEAP, 1) == -EPERM);
    assert(npu_arena_free(&arena, second, NPU_ALLOC_HEAP, 2) == 0);
    assert(npu_arena_free(&arena, first, NPU_ALLOC_USER, UINT32_MAX) == 0);
    assert(npu_arena_free(&arena, 0x1400, NPU_ALLOC_FIRMWARE,
                          UINT32_MAX) == 0);
    assert(!arena.allocations);
    npu_arena_destroy(&arena);
    return 0;
}
