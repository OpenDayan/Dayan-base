#define _POSIX_C_SOURCE 200809L

#include <assert.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include "npu_driver.h"

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

static void remove_file(const char *directory, const char *name)
{
    char path[512];

    assert(snprintf(path, sizeof(path), "%s/%s", directory, name) > 0);
    assert(unlink(path) == 0);
}

int main(void)
{
    char directory[] = "/tmp/npu-driver-test-XXXXXX";
    npu_driver_options_t options;
    npu_device_t device;
    uint64_t written = UINT64_C(0x1122334455667788);
    uint64_t observed = 0;
    char path[512];
    int fd;

    assert(mkdtemp(directory));
    create_file(directory, "xdma0_h2c_0", 0x40000);
    create_file(directory, "xdma0_c2h_0", 0x40000);
    create_file(directory, "xdma0_user", 0x200);
    create_file(directory, "xdma0_events_0", 4);
    npu_driver_options_init(&options);
    options.device_prefix = directory;
    assert(npu_device_open(&device, &options) == 0);
    assert(npu_device_write(&device, 0x1230, &written, sizeof(written)) == 0);
    assert(snprintf(path, sizeof(path), "%s/xdma0_h2c_0", directory) > 0);
    fd = open(path, O_RDONLY);
    assert(fd >= 0);
    assert(pread(fd, &observed, sizeof(observed), 0x1230) ==
           (ssize_t)sizeof(observed));
    close(fd);
    assert(observed == written);

    observed = UINT64_C(0xaabbccddeeff0011);
    assert(snprintf(path, sizeof(path), "%s/xdma0_c2h_0", directory) > 0);
    fd = open(path, O_WRONLY);
    assert(fd >= 0);
    assert(pwrite(fd, &observed, sizeof(observed), 0x2340) ==
           (ssize_t)sizeof(observed));
    close(fd);
    observed = 0;
    assert(npu_device_read(&device, 0x2340, &observed, sizeof(observed)) == 0);
    assert(observed == UINT64_C(0xaabbccddeeff0011));
    npu_device_close(&device);

    remove_file(directory, "xdma0_events_0");
    remove_file(directory, "xdma0_user");
    remove_file(directory, "xdma0_c2h_0");
    remove_file(directory, "xdma0_h2c_0");
    assert(rmdir(directory) == 0);
    return 0;
}
