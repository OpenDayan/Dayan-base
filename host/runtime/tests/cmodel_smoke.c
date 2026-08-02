#define _DEFAULT_SOURCE

#include <assert.h>
#include <errno.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "npu_runtime.h"
#include "trap.h"

static const uint32_t operator_code[] = {
    UINT32_C(0x00b53023), /* sd a1, 0(a0) */
    UINT32_C(0x0330000f), /* fence rw, rw */
    UINT32_C(0x00008067), /* ret */
    UINT32_C(0x00058067), /* tail a1 (puts passed as argument) */
    UINT32_C(0xffffffff), /* deliberate illegal instruction */
};

struct callbacks {
    pthread_mutex_t mutex;
    char output[128];
    size_t output_size;
    struct trap_record trap;
    unsigned trap_count;
};

static void capture_output(void *arg, uint32_t hart, uint32_t proc,
                           const void *data, size_t size, uint32_t flags)
{
    struct callbacks *callbacks = arg;

    (void)hart;
    (void)proc;
    (void)flags;
    pthread_mutex_lock(&callbacks->mutex);
    if (size <= sizeof(callbacks->output) - callbacks->output_size) {
        memcpy(callbacks->output + callbacks->output_size, data, size);
        callbacks->output_size += size;
    }
    pthread_mutex_unlock(&callbacks->mutex);
}

static void capture_trap(void *arg, const struct trap_record *record)
{
    struct callbacks *callbacks = arg;

    pthread_mutex_lock(&callbacks->mutex);
    callbacks->trap = *record;
    ++callbacks->trap_count;
    pthread_mutex_unlock(&callbacks->mutex);
}

static int fail(const char *operation, int status)
{
    fprintf(stderr, "%s failed: %d\n", operation, status);
    return 1;
}

int main(int argc, char **argv)
{
    const uint64_t expected = UINT64_C(0x1122334455667788);
    static const char message[] = "rpc-smoke";
    npu_runtime_options_t options;
    npu_launch_desc_t launch_desc;
    npu_launch_info_t launch_info;
    npu_runtime_t *runtime = 0;
    npu_launch_t *launch = 0;
    uint64_t code_address = 0;
    uint64_t output_address = 0;
    uint64_t message_address = 0;
    uint64_t puts_address;
    uint64_t output = 0;
    struct callbacks callbacks;
    unsigned hart_count = 1;
    uint32_t hart_mask;
    int status;

    if (argc != 3 && argc != 4) {
        fprintf(stderr, "usage: %s BL1.elf PUTS_ADDRESS [HART_COUNT]\n",
                argv[0]);
        return 2;
    }
    puts_address = strtoull(argv[2], 0, 0);
    if (!puts_address)
        return fail("puts symbol", -EINVAL);
    if (argc == 4)
        hart_count = (unsigned)strtoul(argv[3], 0, 0);
    if (!hart_count || hart_count > 4)
        return fail("hart count", -EINVAL);
    hart_mask = (1U << hart_count) - 1U;
    memset(&callbacks, 0, sizeof(callbacks));
    assert(pthread_mutex_init(&callbacks.mutex, 0) == 0);
    npu_runtime_options_init(&options);
    options.hart_count = hart_count;
    options.output = capture_output;
    options.trap = capture_trap;
    options.callback_context = &callbacks;
    status = npu_runtime_open(&runtime, &options);
    if (status < 0)
        return fail("runtime open", status);
    status = npu_runtime_load_bl1_file(runtime, argv[1], hart_mask);
    if (status < 0)
        return fail("BL1 load", status);
    status = npu_runtime_alloc(runtime, sizeof(operator_code), 64,
                               &code_address);
    if (status < 0)
        return fail("code allocation", status);
    status = npu_runtime_alloc(runtime, sizeof(output), 64,
                               &output_address);
    if (status < 0)
        return fail("output allocation", status);
    status = npu_runtime_alloc(runtime, sizeof(message), 64,
                               &message_address);
    if (status < 0)
        return fail("message allocation", status);
    status = npu_runtime_copy_to_device(runtime, code_address,
                                        operator_code,
                                        sizeof(operator_code));
    if (status < 0)
        return fail("operator upload", status);
    status = npu_runtime_copy_to_device(runtime, output_address,
                                        &output, sizeof(output));
    if (status < 0)
        return fail("output clear", status);
    status = npu_runtime_copy_to_device(runtime, message_address,
                                        message, sizeof(message));
    if (status < 0)
        return fail("message upload", status);

    npu_launch_desc_init(&launch_desc);
    launch_desc.hart = 0;
    launch_desc.entry = code_address;
    launch_desc.printf_buffer_size = 0;
    launch_desc.argc = 2;
    launch_desc.args[0] = output_address;
    launch_desc.args[1] = expected;
    status = npu_runtime_launch(runtime, &launch_desc, &launch);
    if (status < 0)
        return fail("operator launch", status);
    status = npu_launch_wait(launch, 5000, &launch_info);
    if (status < 0)
        return fail("operator wait", status);
    if (launch_info.status != 0)
        return fail("operator completion", (int)launch_info.status);
    status = npu_runtime_copy_from_device(runtime, &output,
                                          output_address, sizeof(output));
    if (status < 0)
        return fail("output download", status);
    if (output != expected) {
        fprintf(stderr, "bad operator result: 0x%016llx\n",
                (unsigned long long)output);
        return 1;
    }
    status = npu_launch_release(launch);
    if (status < 0)
        return fail("launch release", status);

    npu_launch_desc_init(&launch_desc);
    launch_desc.hart = 0;
    launch_desc.entry = code_address + 3U * sizeof(uint32_t);
    launch_desc.stack_size = 2U * 1024U * 1024U;
    launch_desc.printf_buffer_size = 64;
    launch_desc.argc = 2;
    launch_desc.args[0] = message_address;
    launch_desc.args[1] = puts_address;
    status = npu_runtime_launch(runtime, &launch_desc, &launch);
    if (status < 0)
        return fail("RPC operator launch", status);
    status = npu_launch_wait(launch, 5000, &launch_info);
    if (status < 0 || launch_info.status != 0)
        return fail("RPC operator wait",
                    status < 0 ? status : (int)launch_info.status);
    status = npu_launch_release(launch);
    if (status < 0)
        return fail("RPC launch release", status);
    pthread_mutex_lock(&callbacks.mutex);
    if (callbacks.output_size != sizeof(message) ||
        memcmp(callbacks.output, "rpc-smoke\n", sizeof(message)) != 0) {
        pthread_mutex_unlock(&callbacks.mutex);
        return fail("RPC output", -EIO);
    }
    pthread_mutex_unlock(&callbacks.mutex);

    npu_launch_desc_init(&launch_desc);
    launch_desc.hart = 0;
    launch_desc.entry = code_address + 4U * sizeof(uint32_t);
    launch_desc.printf_buffer_size = 0;
    status = npu_runtime_launch(runtime, &launch_desc, &launch);
    if (status < 0)
        return fail("trap operator launch", status);
    status = npu_launch_wait(launch, 5000, &launch_info);
    if (status < 0 || launch_info.status != -EIO)
        return fail("trap operator wait",
                    status < 0 ? status : (int)launch_info.status);
    for (unsigned attempt = 0; attempt < 500; ++attempt) {
        unsigned count;

        pthread_mutex_lock(&callbacks.mutex);
        count = callbacks.trap_count;
        pthread_mutex_unlock(&callbacks.mutex);
        if (count)
            break;
        usleep(10000);
    }
    pthread_mutex_lock(&callbacks.mutex);
    if (callbacks.trap_count != 1 ||
        callbacks.trap.thread_id != launch_info.proc_id ||
        callbacks.trap.hart_id != 0) {
        pthread_mutex_unlock(&callbacks.mutex);
        return fail("trap report", -EIO);
    }
    pthread_mutex_unlock(&callbacks.mutex);
    status = npu_launch_release(launch);
    if (status < 0)
        return fail("trap launch release", status);

    status = npu_runtime_free(runtime, message_address);
    if (status < 0)
        return fail("message free", status);
    status = npu_runtime_free(runtime, output_address);
    if (status < 0)
        return fail("output free", status);
    status = npu_runtime_free(runtime, code_address);
    if (status < 0)
        return fail("code free", status);
    status = npu_runtime_close(runtime);
    if (status < 0)
        return fail("runtime close", status);
    pthread_mutex_destroy(&callbacks.mutex);
    puts("cmodel smoke: boot, create, copy, reverse RPC and trap passed");
    return 0;
}
