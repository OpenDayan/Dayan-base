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
static int no_slots;

uintptr_t platform_hart_id(void)
{
    return 0;
}

void platform_cache_invalidate(void *address, size_t size)
{
    assert(address == &request && size == sizeof(request));
    assert(create_step++ == 0);
    request = host_request;
}

struct task_proc *task_proc_create(const struct rpc_proc_create_request *input)
{
    assert(create_step++ == 1);
    assert(input == &request);
    assert(input->args[0] == 0x1234 && strcmp(input->name, "create-test") == 0);
    task.thread.slot = 1;
    task.thread.generation = 1;
    return no_slots ? 0 : &task;
}

void platform_cache_flush(const void *address, size_t size)
{
    assert(address == &request && size == sizeof(request));
    assert(create_step == 1 || create_step == 2);
    create_step = 3;
    host_request = request;
}

void platform_instruction_fence(void)
{
    assert(create_step++ == 3);
}

void thread_schedule(struct thread *thread)
{
    assert(create_step++ == 4);
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

    memset(&host_request, 0, sizeof(host_request));
    host_request.stack_size = RPC_PROC_MIN_STACK_SIZE;
    host_request.argc = 1;
    host_request.args[0] = 0x1234;
    strcpy(host_request.name, "create-test");
    for (unsigned attempt = 0; attempt < 3; ++attempt) {
        /* Device begins with stale cache contents on every descriptor reuse. */
        memset(&request, 0xa5, sizeof(request));
        create_step = 0;
        no_slots = attempt == 1;
        if (attempt == 2)
            host_request.stack_size = CONFIG_PROC_STACK_SIZE + 1U;
        assert(rpc_receive(RPC_PROC_CREATE, (uintptr_t)&request, &response) == 1);
        assert(response == rpc_command_pack(RPC_PROC_CREATE_RESPONSE,
                                            (uintptr_t)&request));
        if (attempt == 0) {
            assert(create_step == 5);
        } else {
            assert(create_step == 3);
            assert(!host_request.proc_id && host_request.state == RPC_PROC_FAILED);
            assert(host_request.status == (no_slots ? -ENOMEM : -EINVAL));
        }
    }
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
    assert(sizeof(struct rpc_proc_create_request) == 224);
    assert(sizeof(struct rpc_proc_info) == 96);
    assert(sizeof(struct rpc_proc_ps_snapshot) == 1864);
    assert(sizeof(struct rpc_proc_kill_result) == 40);
    assert(RPC_PROC_KILL_FAMILY == 1U);
    assert(sizeof(struct rpc_profile_record) == 8);
    assert(sizeof(struct rpc_profile_slot_define) == 24);
    assert(sizeof(struct rpc_profile_config) == 56);
    assert(sizeof(struct rpc_profile_flush_request) == 32);
    assert(sizeof(struct rpc_profile_flush_response) == 24);
    assert(offsetof(struct rpc_proc_create_request, args) == 56);
    assert(offsetof(struct rpc_proc_create_request, status) == 128);
    assert(offsetof(struct rpc_proc_create_request, task_id) == 152);
    assert(offsetof(struct rpc_proc_create_request, task_dim) == 164);
    assert(offsetof(struct rpc_proc_create_request, task_mode) == 120);
    assert(offsetof(struct rpc_proc_create_request, max_child_threads) == 124);
    assert(offsetof(struct rpc_proc_create_request, resource_context) == 180);
    assert(offsetof(struct rpc_proc_create_request, uram_base) == 184);
    assert(offsetof(struct rpc_proc_create_request, tensor_context) == 200);
    assert(offsetof(struct rpc_proc_create_request, name) == 208);
    assert(offsetof(struct rpc_proc_info, name) == 80);
    assert(offsetof(struct rpc_profile_slot_define, name) == 8);
    assert(RPC_PROC_SLOTS_PER_HART == 16);
    assert(RPC_PROC_NAME_SIZE == 16);
    assert(RPC_PROC_MAX_CHILD_THREADS == 15);
    assert(RPC_PROC_CREATE == 0x0100);
    assert(RPC_PROC_CREATE_RESPONSE == (RPC_PROC_CREATE | 0x8000));
    assert(RPC_ICACHE_FLUSH == 0x0101);
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
    assert(RPC_PROFILE_PROC_KILL_WAIT_PRODUCE == 7U);
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
