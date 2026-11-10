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
#include <stdint.h>

#include "cpu.h"
#include "errno.h"
#include "string.h"
#include "test.h"
#include "tensor_context.h"

_Alignas(CONFIG_CACHE_LINE_SIZE) unsigned char __cpu_start[CONFIG_CPU_LOCAL_SIZE];
static unsigned wakes;
static unsigned prints;

int snprintf(char *buffer, size_t size, const char *format, ...)
{
    (void)buffer;
    (void)size;
    (void)format;
    assert(0 && "unexpected fault report");
    return 0;
}

long platform_print(const void *data, size_t size)
{
    assert(size == 4 && memcmp(data, "test", size) == 0);
    ++prints;
    return (long)size;
}

void platform_cache_flush(const void *data, size_t size)
{
    (void)data;
    (void)size;
}

size_t thread_event_wake_all(struct thread_list *event)
{
    assert(event == &cpu_self()->test.exit_event);
    return ++wakes;
}

static void entry(void)
{
}

static void test_descriptors(void)
{
    dayan_test_desc_t test = {.entry = entry, .name = "valid"};
    struct rpc_proc_create_request request;

    test.resources.max_child_threads = 15;
    test.resources.lsu_tcsr_count = 8;
    test.resources.uram_size = CONFIG_URAM_SIZE;
    test.resources.atomic_cache_size = CONFIG_COHERENT_SIZE;
    test.resources.resource_counts = DAYAN_RESOURCE_COUNT_PACK(32, 32);
    test.argc = 32;
    test.args[31] = 1234;
    assert(test_build_request(&test, &request) == 0);
    assert(request.argc == 32 && request.args[31] == 1234);
    assert(request.shared_size == CONFIG_COHERENT_SIZE);
    assert(request.task_mode == 1 && request.task_dim[2] == 1);
    assert(request.max_child_threads == 15);
    assert(dayan_context_unpack(request.tensor_context).uram_hi *
           DAYAN_CONTEXT_URAM_GRANULE == CONFIG_URAM_SIZE);
    test.resources.uram_size = CONFIG_URAM_SIZE + DAYAN_CONTEXT_URAM_GRANULE;
    assert(test_build_request(&test, &request) == -EINVAL);
    test.resources.uram_size = CONFIG_URAM_SIZE;
    test.argc = 33;
    assert(test_build_request(&test, &request) == -EINVAL);
    test.argc = 0;
    test.resources.lsu_tcsr_count = 9;
    assert(test_build_request(&test, &request) == -EINVAL);
    test.resources.lsu_tcsr_count = 0;
    test.resources.uram_size = 1;
    assert(test_build_request(&test, &request) == -EINVAL);
    test.resources.uram_size = 0;
    test.resources.resource_counts = DAYAN_RESOURCE_COUNT_PACK(33, 0);
    assert(test_build_request(&test, &request) == -EINVAL);
    test.resources.resource_counts = 0;
    test.resources.max_child_threads = 16;
    assert(test_build_request(&test, &request) == -EINVAL);
    test.resources.max_child_threads = 0;
    memset(test.name, 'x', sizeof(test.name));
    assert(test_build_request(&test, &request) == -EINVAL);
    test.name[0] = 0;
    assert(test_build_request(&test, &request) == -EINVAL);
}

static long call(uint16_t service, const void *request, size_t request_size,
                 void *response, size_t response_size)
{
    struct rpc_packet packet = {
        .request_addr = (uintptr_t)request, .request_size = request_size,
        .response_addr = (uintptr_t)response, .response_size = response_size,
        .status = -EIO,
    };

    test_rpc_send(rpc_command_pack(service, (uintptr_t)&packet));
    return (long)packet.status;
}

int main(void)
{
    struct test_state *state = &cpu_self()->test;
    struct rpc_heap_request heap = {.size = 1};
    struct rpc_heap_response response;
    struct rpc_heap_release release;
    struct rpc_print_request print = {.data_addr = (uintptr_t)"test", .data_size = 4};
    struct rpc_proc_exit_request exit = {
        .proc_id = 257, .flags = RPC_PROC_EXIT_ROOT, .result = -17,
    };

    test_descriptors();
    state->heap_next = DAYAN_TEST_DATA_BASE - 128;
    assert(call(RPC_HEAP_ACQUIRE, &heap, sizeof(heap),
                &response, sizeof(response)) == 0);
    assert(response.addr == DAYAN_TEST_DATA_BASE - 128 && response.size == 64);
    release.addr = response.addr;
    release.size = response.size;
    assert(call(RPC_HEAP_RELEASE, &release, sizeof(release), 0, 0) == 0);
    assert(call(RPC_HEAP_ACQUIRE, &heap, sizeof(heap),
                &response, sizeof(response)) == 0);
    assert(response.addr == DAYAN_TEST_DATA_BASE - 64 && response.size == 64);
    assert(call(RPC_HEAP_ACQUIRE, &heap, sizeof(heap),
                &response, sizeof(response)) == -ENOMEM);
    assert(!response.addr && !response.size);
    heap.size = SIZE_MAX;
    assert(call(RPC_HEAP_ACQUIRE, &heap, sizeof(heap),
                &response, sizeof(response)) == -ENOMEM);
    assert(state->heap_next == DAYAN_TEST_DATA_BASE);
    heap.size = 0;
    assert(test_heap_acquire(heap.size, &response) == -EINVAL);
    assert(call(RPC_WRITE, &print, sizeof(print), 0, 0) == 4 && prints == 1);
    assert(call(RPC_WRITE, &print, 1, 0, 0) == -EINVAL);
    assert(call(RPC_CLEAR, 0, 0, 0, 0) == -ENOSYS);
    state->proc_id = 257;
    assert(call(RPC_PROC_EXIT, &exit, sizeof(exit), 0, 0) == 0);
    assert(state->completed && state->result == -17 && wakes == 1);
    exit.proc_id = 258;
    assert(call(RPC_PROC_EXIT, &exit, sizeof(exit), 0, 0) == -EINVAL);
    assert(wakes == 1);
    return 0;
}
