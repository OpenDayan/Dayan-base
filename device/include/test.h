#ifndef NPU_TEST_H
#define NPU_TEST_H

#include "api/dayan_test.h"
#include "sche.h"

struct test_state {
    uintptr_t heap_next;
    struct thread_list exit_event;
    int64_t result;
    uint32_t proc_id;
    uint32_t completed;
};

int test_present(void);
void test_init(void);
int test_build_request(const dayan_test_desc_t *test,
                       struct rpc_proc_create_request *request);
long test_heap_acquire(size_t size, struct rpc_heap_response *response);
void test_rpc_send(uint64_t command);

#endif
