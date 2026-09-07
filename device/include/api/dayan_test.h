#ifndef DAYAN_TEST_H
#define DAYAN_TEST_H

#include "dayan_sys.h"

#define DAYAN_TEST_DATA_BASE 0x84000000ULL
#define DAYAN_TEST_DATA_SIZE 0x40000000ULL
#define DAYAN_TEST_MAX_ARGS 8U

/* One local root Proc, run on Hart 0 in link order. Child Threads are allowed. */
typedef struct dayan_test_desc {
    void (*entry)(void);
    dayan_kernel_resource_desc_t resources;
    uintptr_t args[DAYAN_TEST_MAX_ARGS];
    uint32_t argc;
    char name[DAYAN_THREAD_NAME_SIZE];
} dayan_test_desc_t;

/* Pass a constant aggregate initializer; no C++ constructors are supported. */
#define DAYAN_TEST(symbol, ...) \
    static const dayan_test_desc_t symbol \
    __attribute__((used, section(".npu_test"), aligned(8))) = __VA_ARGS__

#endif
