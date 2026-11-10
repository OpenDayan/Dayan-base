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
#include <stdint.h>

#include "config.h"
#include "errno.h"
#include "platform_regs.h"
#include "rpc.h"
#include "sche.h"
#include "string.h"
#include "tensor_interrupt.h"
#include "trap.h"

static _Alignas(RPC_DESCRIPTOR_GRANULE) struct rpc_proc_create_request request;
static struct rpc_proc_create_request host_request;
static struct task_proc task;
static unsigned create_step;
static size_t invalidated_size;
static unsigned fence_count;
static int no_slots;

uintptr_t platform_hart_id(void)
{
    return 0;
}

void platform_cache_invalidate(void *address, size_t size)
{
    if (!create_step) {
        assert(address == &request && size == 128);
        create_step = 1;
    } else {
        assert(create_step == 1 && invalidated_size == 128);
        assert(address == (char *)&request + 128);
        assert(size == rpc_proc_create_size(host_request.argc) - 128);
    }
    memcpy(address, (char *)&host_request + invalidated_size, size);
    invalidated_size += size;
}

struct task_proc *task_proc_create(const struct rpc_proc_create_request *input)
{
    assert(create_step++ == 1);
    assert(input == &request);
    assert(invalidated_size == rpc_proc_create_size(host_request.argc));
    assert(input->entry == host_request.entry);
    assert(strcmp(input->name, "create-test") == 0);
    for (unsigned i = 0; i < input->argc; ++i)
        assert(input->args[i] == 0xfedcba9876543200ULL + i);
    task.thread.slot = 1;
    task.thread.generation = 1;
    return no_slots ? 0 : &task;
}

void platform_cache_flush(const void *address, size_t size)
{
    assert(address == &request && size == invalidated_size);
    assert(create_step == 1 || create_step == 2);
    create_step = 3;
    memcpy(&host_request, &request, size);
    /* Neither invalidate nor reply may touch a following allocation. */
    for (size_t i = size; i < sizeof(request); ++i)
        assert(((const unsigned char *)&request)[i] == 0xa5);
}

void platform_instruction_fence(void)
{
    ++fence_count;
}

void thread_schedule(struct thread *thread)
{
    assert(create_step++ == 3);
    assert(thread == &task.thread);
    assert(host_request.proc_id == thread_id(&task.thread));
    assert(host_request.state == RPC_PROC_RUNNING && !host_request.status);
}

int proc_snapshot(struct rpc_proc_ps_snapshot *snapshot)
{
    (void)snapshot;
    return -EINVAL;
}

int proc_kill_all(struct rpc_proc_kill_result *result)
{
    (void)result;
    return -EINVAL;
}

int profile_configure(struct rpc_profile_config *config)
{
    (void)config;
    return -EINVAL;
}

static void test_create(void)
{
    uint64_t response;

    for (unsigned argc = 0; argc <= RPC_PROC_MAX_ARGS; ++argc) {
        memset(&host_request, 0, sizeof(host_request));
        host_request.stack_size_m1 = RPC_PROC_MAX_STACK_SIZE - 1U;
        host_request.argc = argc;
        for (unsigned i = 0; i < argc; ++i)
            host_request.args[i] = 0xfedcba9876543200ULL + i;
        strcpy(host_request.name, "create-test");
        for (unsigned attempt = 0; attempt < 3; ++attempt) {
            /* Device begins with stale cache contents on every descriptor reuse. */
            memset(&request, 0xa5, sizeof(request));
            create_step = 0;
            invalidated_size = 0;
            host_request.entry = 0x80001234;
            no_slots = attempt == 1;
            if (attempt == 2)
                host_request.printf_size = CONFIG_PROC_PRINTF_SIZE + 1U;
            assert(rpc_receive(RPC_PROC_CREATE, (uintptr_t)&request, &response) == 1);
            assert(response == rpc_command_pack(RPC_PROC_CREATE_RESPONSE,
                                                (uintptr_t)&request));
            if (attempt == 0) {
                assert(create_step == 4);
            } else {
                assert(create_step == 3);
                assert(!host_request.proc_id && host_request.state == RPC_PROC_FAILED);
                assert(host_request.status == (no_slots ? -ENOMEM : -EINVAL));
            }
        }
    }
    assert(fence_count == 0);
    assert(rpc_receive(RPC_FENCEI, 0x123456, &response) == 1);
    assert(fence_count == 1);
    assert(response == rpc_command_pack(RPC_FENCEI_RESPONSE, 0x123456));
}

int main(void)
{
    uint64_t command = rpc_command_pack(0x1234, 0xabcdef012345ULL);

    assert(rpc_command_opcode(command) == 0x1234);
    assert(rpc_command_arg(command) == 0xabcdef012345ULL);
    assert(RPC_DESCRIPTOR_GRANULE == 64U);
    assert(sizeof(struct rpc_packet) == 40);
    assert(sizeof(struct rpc_dcache_flush_request) == 40);
    assert(sizeof(struct rpc_print_request) == 24);
    assert(sizeof(struct rpc_heap_request) == 8);
    assert(sizeof(struct rpc_heap_response) == 16);
    assert(sizeof(struct rpc_heap_release) == 16);
    assert(sizeof(struct rpc_proc_exit_request) == 32);
    assert(sizeof(struct rpc_proc_create_request) == 384);
    assert(sizeof(struct rpc_proc_info) == 96);
    assert(sizeof(struct rpc_proc_ps_snapshot) == 1864);
    assert(sizeof(struct rpc_proc_kill_result) == 40);
    assert(RPC_PROC_KILL_FAMILY == 1U);
    assert(sizeof(struct rpc_profile_record) == 8);
    assert(sizeof(struct rpc_profile_slot_define) == 24);
    assert(sizeof(struct rpc_profile_config) == 56);
    assert(sizeof(struct rpc_profile_flush_request) == 32);
    assert(sizeof(struct rpc_profile_flush_response) == 24);
    assert(offsetof(struct rpc_proc_create_request, args) == 88);
    assert(offsetof(struct rpc_proc_create_request, task_id) == 48);
    assert(offsetof(struct rpc_proc_create_request, task_mode) == 29);
    assert(offsetof(struct rpc_proc_create_request, resource_context) == 40);
    assert(offsetof(struct rpc_proc_create_request, tensor_context) == 32);
    assert(offsetof(struct rpc_proc_create_request, name) == 72);
    assert(offsetof(struct rpc_proc_info, name) == 80);
    assert(offsetof(struct rpc_profile_slot_define, name) == 8);
    assert(RPC_PROC_SLOTS_PER_HART == 16);
    assert(RPC_PROC_NAME_SIZE == 16);
    assert(RPC_PROC_MAX_CHILD_THREADS == 15);
    assert(RPC_PROC_CREATE == 0x0100);
    assert(RPC_PROC_CREATE_RESPONSE == (RPC_PROC_CREATE | 0x8000));
    assert(RPC_ICACHE_FLUSH == 0x0101);
    assert(RPC_FENCEI == RPC_ICACHE_FLUSH);
    assert(RPC_FENCEI_RESPONSE == RPC_ICACHE_FLUSH_RESPONSE);
    assert(RPC_ICACHE_FLUSH_RESPONSE == (RPC_ICACHE_FLUSH | 0x8000));
    assert(RPC_PROC_PS == 0x0102);
    assert(RPC_PROC_PS_RESPONSE == (RPC_PROC_PS | 0x8000));
    assert(RPC_PROC_KILL_ALL == 0x0103);
    assert(RPC_PROC_KILL_ALL_RESPONSE ==
           (RPC_PROC_KILL_ALL | 0x8000));
    assert(RPC_TENSOR_INTERRUPT_REPORT == 6);
    assert(RPC_PROC_EXIT == 7);
    assert(RPC_PROFILE_FLUSH == 8);
    assert(RPC_PROFILE_CONFIG == 0x0104);
    assert(RPC_PROFILE_CONFIG_RESPONSE ==
           (RPC_PROFILE_CONFIG | 0x8000));
    assert(RPC_DCACHE_FLUSH == 0x0105);
    assert(RPC_DCACHE_FLUSH_RESPONSE ==
           (RPC_DCACHE_FLUSH | 0x8000));
    assert(RPC_PROFILE_PROC_RUNNING == 0U);
    assert(RPC_PROFILE_PROC_POLLING == 2U);
    assert(RPC_PROFILE_PROC_KILL_WAIT_PRODUCE == 7U);
    assert(RPC_PROFILE_PROC_WAIT_LSU == 8U);
    assert(RPC_PROFILE_PROC_WAIT_MAT == 9U);
    assert(RPC_PROFILE_PROC_WAIT_VEC == 10U);
    assert(RPC_PROFILE_PROC_WAIT_MACRO == 11U);
    assert(RPC_PROFILE_PROC_YIELD == 12U);
    assert(RPC_PROFILE_PROC_WAIT_JOIN == 13U);
    assert(RPC_PROFILE_PROC_WAIT_RESOURCE == 14U);
    assert(RPC_PROFILE_PROC_STATE_MAX == 14U);
    assert(RPC_PROFILE_RECORD_SLOT_DEFINE == 0x80U);
    assert(RPC_PROFILE_SLOT_COUNT == 20U);
    assert(sizeof(struct trap_record) == 328);
    assert(sizeof(struct tensor_interrupt_record) == 472);
    assert(DAYAN_BOOTROM_RESET_VECTOR == 0x10000U);
    assert(DAYAN_DEVICE_MAILBOX(3) == 0x1018U);
    assert(DAYAN_IRQ_MAILBOX(3) == 0x2038U);
    assert(DAYAN_IRQ_ENABLE_CLR == 0x2014U);
    assert(DAYAN_CLINT_MSIP(3) == 0x2000cU);
    test_create();
    return 0;
}
