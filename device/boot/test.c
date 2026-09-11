#include <stddef.h>
#include <stdint.h>

#include "config.h"
#include "cpu.h"
#include "errno.h"
#include "platform.h"
#include "rpc_internal.h"
#include "stdio.h"
#include "string.h"
#include "tensor_context.h"
#include "test.h"

extern const dayan_test_desc_t __npu_test_start[], __npu_test_end[];
extern unsigned char __test_heap_start[], __test_heap_end[];
extern unsigned char __test_backing_start[];

_Static_assert(sizeof(dayan_test_desc_t) == 320,
               "local test descriptors must be densely packed 320-byte records");

int test_present(void)
{
    return (uintptr_t)__npu_test_end != (uintptr_t)__npu_test_start;
}

int test_build_request(const dayan_test_desc_t *test,
                       struct rpc_proc_create_request *request)
{
    const dayan_kernel_resource_desc_t *resources = &test->resources;
    struct dayan_context_fields tensor = {0};
    struct dayan_resource_context_fields named = {0};
    size_t name_size = strnlen(test->name, sizeof(test->name));

    if (!test->entry || ((uintptr_t)test->entry & 3U) ||
        CONFIG_PROC_STACK_SIZE < RPC_PROC_MIN_STACK_SIZE ||
        CONFIG_PROC_STACK_SIZE > RPC_PROC_MAX_STACK_SIZE ||
        CONFIG_PROC_PRINTF_SIZE > RPC_PROC_MAX_PRINTF_SIZE ||
        test->argc > RPC_PROC_MAX_ARGS || !name_size ||
        name_size == sizeof(test->name) ||
        resources->max_child_threads > RPC_PROC_MAX_CHILD_THREADS ||
        resources->flags & ~DAYAN_KERNEL_RESOURCE_FLAGS_ALL ||
        resources->resource_counts & 0xffff0000U ||
        resources->uram_size > CONFIG_URAM_SIZE ||
        resources->uram_size > DAYAN_CONTEXT_URAM_WINDOW ||
        resources->uram_size % DAYAN_CONTEXT_URAM_GRANULE ||
        resources->atomic_cache_size > CONFIG_COHERENT_SIZE ||
        resources->atomic_cache_size > 0x400000U)
        return -EINVAL;

    tensor.uram_hi = resources->uram_size / DAYAN_CONTEXT_URAM_GRANULE;
    tensor.lsu.count = resources->lsu_tcsr_count;
    tensor.mat.count = resources->mat_tcsr_count;
    tensor.vec.count = resources->vec_tcsr_count;
    tensor.macro.count = resources->macro_tcsr_count;
    named.local.count = DAYAN_RESOURCE_LOCAL_COUNT(resources->resource_counts);
    named.shared.count = DAYAN_RESOURCE_SHARED_COUNT(resources->resource_counts);
    if (!dayan_context_fields_valid(&tensor) ||
        !dayan_resource_context_fields_valid(&named))
        return -EINVAL;

    memset(request, 0, sizeof(*request));
    request->entry = (uintptr_t)test->entry;
    request->stack_size = CONFIG_PROC_STACK_SIZE;
    request->printf_buffer_size = CONFIG_PROC_PRINTF_SIZE;
    request->argc = test->argc;
    memcpy(request->args, test->args, sizeof(request->args));
    memcpy(request->name, test->name, sizeof(request->name));
    request->max_child_threads = resources->max_child_threads;
    request->task_mode = 1;
    for (unsigned axis = 0; axis < 3; ++axis)
        request->task_dim[axis] = 1;
    request->shared_size = resources->atomic_cache_size;
    request->shared_base = request->shared_size ? CONFIG_COHERENT_BASE : 0;
    request->uram_size = resources->uram_size;
    request->uram_base = request->uram_size ? CONFIG_URAM_BASE : 0;
    request->tensor_context = dayan_context_pack(&tensor);
    request->resource_context = dayan_resource_context_pack(&named);
    request->cookie = (uintptr_t)test;
    return 0;
}

static void test_report(const char *name, int64_t result)
{
    char text[96];
    int size = snprintf(text, sizeof(text), "[test] %s: %s (%lld)\n",
                        name, result ? "FAIL" : "PASS", (long long)result);

    if (size > 0)
        (void)platform_print(text, (size_t)size < sizeof(text) ?
                             (size_t)size : sizeof(text) - 1U);
}

static void test_main(void)
{
    struct test_state *state = &cpu_self()->test;
    int64_t result = 0;

    if (!platform_hart_id()) {
        for (const dayan_test_desc_t *test = __npu_test_start;
             test < __npu_test_end; ++test) {
            struct rpc_proc_create_request request;
            struct task_proc *task;
            int status = test_build_request(test, &request);

            if (status < 0) {
                test_report("invalid descriptor", status);
                result = status;
                continue;
            }
            if (request.shared_size)
                memset((void *)(uintptr_t)request.shared_base, 0,
                       request.shared_size);
            task = task_proc_create(&request);
            if (!task) {
                test_report(test->name, -ENOMEM);
                result = -ENOMEM;
                continue;
            }
            state->proc_id = thread_id(&task->thread);
            state->completed = 0;
            state->result = 0;
            platform_instruction_fence();
            thread_schedule(&task->thread);
            while (!state->completed)
                thread_wait_event(&state->exit_event);
            test_report(test->name, state->result);
            if (state->result)
                result = state->result;
        }
        test_report("suite", result);
        platform_test_complete(result);
    }
    /* Permanent system workers must never return through the Proc exit path. */
    for (;;)
        thread_wait_event(&state->exit_event);
}

void test_init(void)
{
    struct test_state *state = &cpu_self()->test;

    if (((uintptr_t)__npu_test_end - (uintptr_t)__npu_test_start) %
        sizeof(dayan_test_desc_t))
        platform_halt();
    state->heap_next = platform_hart_id() ? (uintptr_t)__test_heap_end :
                                          (uintptr_t)__test_heap_start;
    thread_list_init(&state->exit_event);
    if (!platform_hart_id())
        platform_test_init(__test_backing_start);
    rpc_start_local(test_main, test_rpc_send);
}
