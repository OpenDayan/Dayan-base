#define _POSIX_C_SOURCE 200809L

#include <assert.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "dayan_kernel.h"
#include "runtime_internal.h"

static unsigned char *read_file(const char *path, size_t *size_out)
{
    unsigned char *data;
    long length;
    FILE *file = fopen(path, "rb");

    assert(file);
    assert(fseek(file, 0, SEEK_END) == 0);
    length = ftell(file);
    assert(length > 0);
    assert(fseek(file, 0, SEEK_SET) == 0);
    data = malloc((size_t)length);
    assert(data);
    assert(fread(data, 1, (size_t)length, file) == (size_t)length);
    fclose(file);
    *size_out = (size_t)length;
    return data;
}

int main(int argc, char **argv)
{
    char memory_path[] = "/tmp/dayan-kernel-loader-XXXXXX";
    struct npu_runtime runtime;
    dayan_kernel_descriptor_t *descriptor;
    unsigned char *api_image;
    unsigned char *project_api_image = 0;
    unsigned char *object;
    unsigned char instruction[8];
    size_t api_size;
    size_t project_api_size = 0;
    size_t object_size;
    uint64_t entry;
    int fd;

    assert(argc == 3 || argc == 4);
    api_image = read_file(argv[1], &api_size);
    object = read_file(argv[2], &object_size);
    if (argc == 4)
        project_api_image = read_file(argv[3], &project_api_size);
    fd = mkstemp(memory_path);
    assert(fd >= 0);
    assert(ftruncate(fd, 0x400000) == 0);

    memset(&runtime, 0, sizeof(runtime));
    assert(pthread_mutex_init(&runtime.mutex, 0) == 0);
    assert(pthread_mutex_init(&runtime.kernel_mutex, 0) == 0);
    npu_arena_init(&runtime.normal, 0x10000, 0x100000);
    runtime.options.hart_count = 1;
    runtime.firmware_loaded = 1;
    runtime.device.opened = 1;
    runtime.device.h2c_fd = fd;
    runtime.device.c2h_fd = fd;
    runtime.device.user_fd = -1;
    runtime.device.event_fd = -1;

    if (project_api_image)
        assert(dayan_runtime_set_api_elf(&runtime, project_api_image,
                                         project_api_size) == 0);
    assert(dayan_runtime_set_api_elf(&runtime, api_image, api_size) == 0);
    assert(dayan_kernel_descriptor_create(object, object_size,
                                          "loader_kernel",
                                          &descriptor) == 0);
    assert(npu_kernel_resolve(&runtime, object, object_size,
                              "loader_kernel", &entry) == 0);
    assert(entry >= 0x10000 && entry < 0x110000);
    assert(pread(fd, instruction, sizeof(instruction), (off_t)entry) ==
           (ssize_t)sizeof(instruction));
    assert(memcmp(instruction, "\0\0\0\0\0\0\0\0", sizeof(instruction)) != 0);
    assert(runtime.kernel_modules);
    assert(npu_kernel_resolve(&runtime, object, object_size,
                              "loader_kernel", &entry) == 0);

    dayan_kernel_descriptor_destroy(descriptor);
    npu_kernel_state_destroy(&runtime);
    npu_arena_destroy(&runtime.normal);
    pthread_mutex_destroy(&runtime.kernel_mutex);
    pthread_mutex_destroy(&runtime.mutex);
    close(fd);
    assert(unlink(memory_path) == 0);
    free(object);
    free(api_image);
    free(project_api_image);
    return 0;
}
