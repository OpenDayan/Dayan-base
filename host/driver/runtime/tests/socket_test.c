#define _POSIX_C_SOURCE 200809L

#include <assert.h>
#include <elf.h>
#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "npu_package.h"
#include "npu_runtime.h"
#include "server_internal.h"

struct run_state {
    dayan_server_t *server;
    int status;
};

static void create_file(const char *directory, const char *name, size_t size)
{
    char path[512];
    int fd;

    assert(snprintf(path, sizeof(path), "%s/%s", directory, name) > 0);
    fd = open(path, O_CREAT | O_EXCL | O_RDWR, 0600);
    assert(fd >= 0);
    assert(ftruncate(fd, (off_t)size) == 0);
    close(fd);
}

static int open_file(const char *directory, const char *name, int flags)
{
    char path[512];

    assert(snprintf(path, sizeof(path), "%s/%s", directory, name) > 0);
    return open(path, flags);
}

static void *run_server(void *argument)
{
    struct run_state *state = argument;

    state->status = dayan_server_run(state->server, 0);
    return 0;
}

static void make_package_object(unsigned char *object, size_t size)
{
    Elf64_Ehdr header;
    Elf64_Shdr section;

    assert(size == sizeof(header) + sizeof(section));
    memset(&header, 0, sizeof(header));
    memcpy(header.e_ident, ELFMAG, SELFMAG);
    header.e_ident[EI_CLASS] = ELFCLASS64;
    header.e_ident[EI_DATA] = ELFDATA2LSB;
    header.e_ident[EI_VERSION] = EV_CURRENT;
    header.e_type = ET_REL;
    header.e_machine = EM_RISCV;
    header.e_version = EV_CURRENT;
    header.e_ehsize = sizeof(header);
    header.e_shoff = sizeof(header);
    header.e_shentsize = sizeof(section);
    header.e_shnum = 1;
    memset(&section, 0, sizeof(section));
    memcpy(object, &header, sizeof(header));
    memcpy(object + sizeof(header), &section, sizeof(section));
}

static void assert_package_cache(dayan_server_t *server, size_t entries,
                                 size_t immutable_entries,
                                 size_t total_references)
{
    const struct server_package_cache *package;
    size_t observed_entries = 0;
    size_t observed_immutable = 0;
    size_t observed_references = 0;

    pthread_mutex_lock(&server->mutex);
    for (package = server->packages; package; package = package->next) {
        ++observed_entries;
        observed_immutable += !!package->immutable;
        observed_references += package->references;
    }
    pthread_mutex_unlock(&server->mutex);
    assert(observed_entries == entries);
    assert(observed_immutable == immutable_entries);
    assert(observed_references == total_references);
}

int main(void)
{
    static const npu_hal_memory_region_t regions[] = {
        { 0x1000, 0x10000, NPU_HAL_MEMORY_NORMAL, NPU_HAL_MEMORY_DEVICE,
          NPU_HAL_ANY_CLUSTER, NPU_HAL_ANY_CORE },
        { 0x20000, 0x10000, NPU_HAL_MEMORY_COHERENT,
          NPU_HAL_MEMORY_DEVICE, NPU_HAL_ANY_CLUSTER, NPU_HAL_ANY_CORE },
    };
    char directory[] = "/tmp/dayan-socket-test-XXXXXX";
    char socket_path[512];
    static const uint64_t input[] = {
        UINT64_C(0x1122334455667788), UINT64_C(0xaabbccddeeff0011),
    };
    uint64_t output[2] = { 0, 0 };
    dayan_server_options_t server_options;
    npu_runtime_options_t runtime_options;
    npu_context_t *context = 0;
    npu_context_t *foreign_context = 0;
    npu_buffer_t *buffer = 0;
    npu_stream_t *stream = 0;
    npu_stream_t *parallel_stream = 0;
    npu_stream_t *foreign_stream = 0;
    npu_event_t *event = 0;
    npu_event_info_t event_info;
    npu_runtime_t *runtime = 0;
    dayan_server_t *server = 0;
    struct run_state run;
    pthread_t thread;
    int fd;

    assert(mkdtemp(directory));
    create_file(directory, "xdma0_h2c_0", 0x40000);
    create_file(directory, "xdma0_c2h_0", 0x40000);
    create_file(directory, "xdma0_user", 0x400);
    assert(snprintf(socket_path, sizeof(socket_path), "%s/server.sock",
                    directory) > 0);

    dayan_server_options_init(&server_options);
    server_options.socket_path = socket_path;
    server_options.hal.device_prefix = directory;
    server_options.hal.memory_regions = regions;
    server_options.hal.memory_region_count =
        sizeof(regions) / sizeof(regions[0]);
    assert(dayan_server_create(&server, &server_options) == 0);
    run.server = server;
    run.status = -1;
    assert(pthread_create(&thread, 0, run_server, &run) == 0);

    npu_runtime_options_init(&runtime_options);
    runtime_options.server_socket = socket_path;
    assert(npu_runtime_open(&runtime, &runtime_options) == 0);
    assert(npu_context_create(runtime, 0, &context) == 0);
    {
        npu_buffer_desc_t desc;
        npu_buffer_desc_init(&desc, sizeof(input));
        assert(npu_buffer_create(context, &desc, &buffer) == 0);
    }
    assert(npu_stream_create(context, 0, &stream) == 0);

    assert(npu_memcpy_h2d_async(stream, buffer, 0, input, sizeof(input),
                                &event) == 0);
    assert(npu_event_wait(event, 5000, &event_info) == 0);
    assert(event_info.state == NPU_EVENT_COMPLETE && event_info.status == 0);
    assert(npu_event_destroy(event) == 0);
    fd = open_file(directory, "xdma0_h2c_0", O_RDONLY);
    assert(fd >= 0);
    assert(pread(fd, output, sizeof(output), 0x1000) ==
           (ssize_t)sizeof(output));
    close(fd);
    assert(memcmp(output, input, sizeof(input)) == 0);

    fd = open_file(directory, "xdma0_c2h_0", O_WRONLY);
    assert(fd >= 0);
    assert(pwrite(fd, input, sizeof(input), 0x1000) ==
           (ssize_t)sizeof(input));
    close(fd);
    memset(output, 0, sizeof(output));
    assert(npu_memcpy_d2h_async(stream, output, buffer, 0, sizeof(output),
                                &event) == 0);
    assert(npu_event_wait(event, 5000, &event_info) == 0);
    assert(event_info.state == NPU_EVENT_COMPLETE && event_info.status == 0);
    assert(memcmp(output, input, sizeof(input)) == 0);
    assert(npu_event_destroy(event) == 0);
    assert(npu_stream_synchronize(stream, 5000) == 0);

    /* Cross-Stream ordering is explicit; cross-Context object use is illegal. */
    assert(npu_stream_create(context, 0, &parallel_stream) == 0);
    assert(npu_context_create(runtime, 0, &foreign_context) == 0);
    assert(npu_stream_create(foreign_context, 0, &foreign_stream) == 0);
    {
        unsigned char object[sizeof(Elf64_Ehdr) + sizeof(Elf64_Shdr)];
        npu_package_kernel_desc_t kernel;
        npu_module_t *immutable_a = 0;
        npu_module_t *immutable_b = 0;
        npu_module_t *private_a = 0;
        npu_module_t *private_b = 0;
        void *immutable_package = 0;
        void *private_package = 0;
        size_t immutable_size = 0;
        size_t private_size = 0;

        make_package_object(object, sizeof(object));
        memset(&kernel, 0, sizeof(kernel));
        kernel.name = "cache_kernel";
        kernel.entry_symbol = "cache_kernel";
        kernel.supported_task_modes = NPU_PACKAGE_TASK_MODE_ALL;
        kernel.max_tasks_per_core = 1;
        assert(npu_package_build(object, sizeof(object), &kernel, 1,
                                 NPU_PACKAGE_IMMUTABLE_MODULE,
                                 &immutable_package, &immutable_size) == 0);
        assert(npu_package_build(object, sizeof(object), &kernel, 1, 0,
                                 &private_package, &private_size) == 0);

        assert(npu_module_load(context, immutable_package, immutable_size,
                               &immutable_a) == 0);
        assert(npu_module_load(foreign_context, immutable_package,
                               immutable_size, &immutable_b) == 0);
        assert_package_cache(server, 1, 1, 2);

        assert(npu_module_load(context, private_package, private_size,
                               &private_a) == 0);
        assert(npu_module_load(foreign_context, private_package, private_size,
                               &private_b) == 0);
        assert_package_cache(server, 3, 1, 4);

        assert(npu_module_unload(private_b) == 0);
        assert(npu_module_unload(private_a) == 0);
        assert_package_cache(server, 1, 1, 2);
        assert(npu_module_unload(immutable_b) == 0);
        assert(npu_module_unload(immutable_a) == 0);
        assert_package_cache(server, 0, 0, 0);
        npu_package_free(private_package);
        npu_package_free(immutable_package);
    }
    assert(npu_memcpy_h2d_async(stream, buffer, 0, input, sizeof(input),
                                &event) == 0);
    assert(npu_stream_wait_event(foreign_stream, event) == -EINVAL);
    {
        npu_event_t *unexpected = 0;

        assert(npu_memcpy_h2d_async(foreign_stream, buffer, 0, input,
                                    sizeof(input), &unexpected) == -EINVAL);
        assert(!unexpected);
    }
    assert(npu_stream_wait_event(parallel_stream, event) == 0);
    {
        npu_event_t *dependent = 0;
        npu_event_info_t source_info;

        assert(npu_event_record(parallel_stream, &dependent) == 0);
        assert(npu_event_wait(dependent, 5000, &event_info) == 0);
        assert(event_info.state == NPU_EVENT_COMPLETE &&
               event_info.status == 0);
        assert(npu_event_wait(event, 5000, &source_info) == 0);
        assert(source_info.state == NPU_EVENT_COMPLETE &&
               source_info.status == 0 &&
               event_info.sequence > source_info.sequence);
        assert(npu_event_destroy(dependent) == 0);
    }
    assert(npu_event_destroy(event) == 0);
    assert(npu_stream_destroy(foreign_stream) == 0);
    assert(npu_context_destroy(foreign_context) == 0);
    foreign_stream = 0;
    foreign_context = 0;
    assert(npu_stream_destroy(parallel_stream) == 0);
    parallel_stream = 0;

    assert(npu_stream_destroy(stream) == 0);
    assert(npu_buffer_destroy(buffer) == 0);
    assert(npu_context_destroy(context) == 0);
    context = 0;
    buffer = 0;
    stream = 0;

    /* Cancelling a Context must drain granted direct-HAL tokens first. */
    assert(npu_context_create(runtime, 0, &context) == 0);
    {
        npu_buffer_desc_t desc;
        unsigned char *cancel_input = malloc(0x8000);
        unsigned index;

        assert(cancel_input);
        memset(cancel_input, 0x5a, 0x8000);
        npu_buffer_desc_init(&desc, 0x8000);
        assert(npu_buffer_create(context, &desc, &buffer) == 0);
        assert(npu_stream_create(context, 0, &stream) == 0);
        for (index = 0; index < 32; ++index) {
            npu_event_t *pending_event = 0;

            assert(npu_memcpy_h2d_async(stream, buffer, 0, cancel_input,
                                        0x8000, &pending_event) == 0);
            assert(pending_event);
        }
        assert(npu_context_destroy(context) == 0);
        free(cancel_input);
    }
    assert(npu_runtime_close(runtime) == 0);
    dayan_server_request_stop(server);
    assert(pthread_join(thread, 0) == 0);
    assert(run.status == 0);
    dayan_server_destroy(server);
    return 0;
}
