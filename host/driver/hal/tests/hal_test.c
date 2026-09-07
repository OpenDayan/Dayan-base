#define _POSIX_C_SOURCE 200809L

#include <assert.h>
#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#include "hal.h"
#include "backend/xdma/npu_hal_xdma.h"
#include "platform_regs.h"
#include "rpc.h"

static void create_endpoint(const char *directory, uint32_t index)
{
    static const char *suffixes[] = { "h2c_0", "c2h_0", "user" };
    size_t suffix;

    for (suffix = 0; suffix < sizeof(suffixes) / sizeof(suffixes[0]);
         ++suffix) {
        char path[512];
        int fd;

        assert(snprintf(path, sizeof(path), "%s/xdma%u_%s", directory,
                        index, suffixes[suffix]) > 0);
        fd = open(path, O_CREAT | O_EXCL | O_RDWR, 0600);
        assert(fd >= 0);
        assert(ftruncate(fd, 0x40000) == 0);
        close(fd);
    }
}

static int endpoint_fd(const char *directory, uint32_t index,
                       const char *suffix, int flags)
{
    char path[512];

    assert(snprintf(path, sizeof(path), "%s/xdma%u_%s", directory, index,
                    suffix) > 0);
    return open(path, flags);
}

static void create_event_endpoint(const char *directory, uint32_t index)
{
    char path[512];

    assert(snprintf(path, sizeof(path), "%s/xdma%u_events_0", directory,
                    index) > 0);
    assert(mkfifo(path, 0600) == 0);
}

struct irq_writer {
    int status_fd;
    int event_fd;
    uint32_t status;
};

static void *write_irq(void *argument)
{
    const struct timespec delay = { 0, 10 * 1000 * 1000L };
    struct irq_writer *writer = argument;
    uint32_t event_count = 1;

    assert(nanosleep(&delay, 0) == 0);
    assert(pwrite(writer->status_fd, &writer->status,
                  sizeof(writer->status), 0) ==
           (ssize_t)sizeof(writer->status));
    if (writer->event_fd >= 0) {
        assert(write(writer->event_fd, &event_count, sizeof(event_count)) ==
               (ssize_t)sizeof(event_count));
    }
    return 0;
}

int main(void)
{
    static const npu_hal_memory_region_t regions[] = {
        { 0x1000, 0x8000, NPU_HAL_MEMORY_NORMAL, NPU_HAL_MEMORY_DEVICE,
          NPU_HAL_ANY_CLUSTER, NPU_HAL_ANY_CORE },
        { 0x10000, 0x4000, NPU_HAL_MEMORY_NORMAL, NPU_HAL_MEMORY_CLUSTER,
          0, NPU_HAL_ANY_CORE },
        { 0x10000, 0x4000, NPU_HAL_MEMORY_NORMAL, NPU_HAL_MEMORY_CLUSTER,
          1, NPU_HAL_ANY_CORE },
        { 0x20000, 0x4000, NPU_HAL_MEMORY_COHERENT,
          NPU_HAL_MEMORY_DEVICE, NPU_HAL_ANY_CLUSTER, NPU_HAL_ANY_CORE },
        { 0x30000, 0x8000, NPU_HAL_MEMORY_URAM, NPU_HAL_MEMORY_CORE,
          0, 0 },
        { 0x30000, 0x8000, NPU_HAL_MEMORY_URAM, NPU_HAL_MEMORY_CORE,
          1, 3 },
    };
    static const npu_hal_memory_region_t overlapping_regions[] = {
        { 0x1000, 0x2000, NPU_HAL_MEMORY_NORMAL, NPU_HAL_MEMORY_DEVICE,
          NPU_HAL_ANY_CLUSTER, NPU_HAL_ANY_CORE },
        { 0x1800, 0x1000, NPU_HAL_MEMORY_COHERENT,
          NPU_HAL_MEMORY_CLUSTER, 1, NPU_HAL_ANY_CORE },
    };
    char directory[] = "/tmp/dayan-hal-test-XXXXXX";
    npu_hal_allocation_t *global = 0;
    npu_hal_allocation_t *local0 = 0;
    npu_hal_allocation_t *local1 = 0;
    npu_hal_allocation_t *uram0 = 0;
    npu_hal_allocation_t *uram1 = 0;
    npu_hal_allocation_t *invalid = 0;
    npu_hal_memory_request_t request;
    npu_hal_data_export_t exported;
    int handles[NPU_HAL_DATA_MAX_HANDLES];
    npu_hal_data_token_t token;
    npu_hal_options_t options;
    npu_hal_xdma_options_t xdma_options;
    npu_hal_data_plane_t *data_plane = 0;
    npu_hal_info_t info;
    npu_hal_t *hal = 0;
    npu_hal_core_t core0 = { 0, 0 };
    npu_hal_core_t core1 = { 1, 3 };
    uint64_t address0;
    uint64_t address1;
    uint64_t capacity;
    uint64_t value = UINT64_C(0x1122334455667788);
    uint64_t observed = 0;
    uint64_t command;
    uint32_t pending = 0;
    int fd;

    assert(npu_hal_xdma_options_init(
               &xdma_options, NPU_HAL_XDMA_PROFILE_SIMULATOR) == 0);
    assert(strcmp(xdma_options.device_directory,
                  NPU_HAL_XDMA_SIMULATOR_DEVICE_DIRECTORY) == 0);
    assert(xdma_options.mmio_address_offset == 0);
    assert(npu_hal_xdma_options_init(
               &xdma_options, NPU_HAL_XDMA_PROFILE_FPGA) == 0);
    assert(strcmp(xdma_options.device_directory,
                  NPU_HAL_XDMA_FPGA_DEVICE_DIRECTORY) == 0);
    assert(xdma_options.mmio_address_offset ==
           NPU_HAL_XDMA_FPGA_MMIO_OFFSET);
    assert(npu_hal_xdma_options_init(&xdma_options, 0) == -EINVAL);
    assert(npu_hal_xdma_options_init(
               &xdma_options, NPU_HAL_XDMA_PROFILE_SIMULATOR) == 0);

    assert(mkdtemp(directory));
    create_endpoint(directory, 0);
    create_endpoint(directory, 2);
    npu_hal_options_init(&options);
    xdma_options.device_directory = directory;
    xdma_options.mmio_address_offset = 0x10000;
    options.backend_options = &xdma_options.base;
    options.cluster_count = 2;
    options.cores_per_cluster = 4;
    xdma_options.cluster_device_stride = 2;
    options.memory_regions = regions;
    options.memory_region_count = sizeof(regions) / sizeof(regions[0]);
    assert(npu_hal_open(&hal, &options) == 0);
    assert(npu_hal_get_info(hal, &info) == 0);
    assert(info.cluster_count == 2 && info.cores_per_cluster == 4);
    assert(info.data_provider == NPU_HAL_DATA_PROVIDER_XDMA);
    assert(info.data_handle_count == 4 && info.transfer_alignment == 1);
    {
        struct irq_writer writer;
        pthread_t thread;
        uint32_t irq_cluster = 0;

        assert(npu_hal_irq_wait(hal, 0, &irq_cluster, &pending) ==
               -ETIMEDOUT);
        assert(irq_cluster == NPU_HAL_ANY_CLUSTER && pending == 0);
        writer.status_fd = endpoint_fd(directory, 2, "user", O_RDWR);
        writer.event_fd = -1;
        writer.status = UINT32_C(1) << core1.core;
        assert(writer.status_fd >= 0);
        assert(pthread_create(&thread, 0, write_irq, &writer) == 0);
        assert(npu_hal_irq_wait(hal, 100, &irq_cluster, &pending) == 0);
        assert(pthread_join(thread, 0) == 0);
        assert(irq_cluster == core1.cluster && pending == writer.status);
        pending = 0;
        assert(pwrite(writer.status_fd, &pending, sizeof(pending), 0) ==
               (ssize_t)sizeof(pending));
        close(writer.status_fd);
    }

    assert(npu_hal_copy_to_address(hal, 1, 0x1230, &value,
                                   sizeof(value)) == 0);
    fd = endpoint_fd(directory, 2, "h2c_0", O_RDONLY);
    assert(fd >= 0);
    assert(pread(fd, &observed, sizeof(observed), 0x11230) ==
           (ssize_t)sizeof(observed));
    close(fd);
    assert(observed == value);

    value = UINT64_C(0x8877665544332211);
    assert(npu_hal_copy_to_address(hal, 1, UINT64_C(0x80001230),
                                   &value, sizeof(value)) == 0);
    fd = endpoint_fd(directory, 2, "h2c_0", O_RDONLY);
    assert(fd >= 0);
    observed = 0;
    assert(pread(fd, &observed, sizeof(observed), UINT64_C(0x80001230)) ==
           (ssize_t)sizeof(observed));
    close(fd);
    assert(observed == value);

    command = rpc_command_pack(RPC_ICACHE_FLUSH,
                               UINT64_C(0x123456789abc));
    assert(npu_hal_command_submit(hal, core1, command) == 0);
    fd = endpoint_fd(directory, 2, "h2c_0", O_RDONLY);
    assert(fd >= 0);
    observed = 0;
    assert(pread(fd, &observed, sizeof(observed),
                 0x10000 + DAYAN_DEVICE_MAILBOX(core1.core)) ==
           (ssize_t)sizeof(observed));
    assert(observed == command);
    assert(pread(fd, &pending, sizeof(pending),
                 0x10000 + DAYAN_CLINT_MSIP(core1.core)) ==
           (ssize_t)sizeof(pending));
    close(fd);
    assert(pending == 1U);

    npu_hal_memory_request_init(&request);
    request.size = 0x100;
    request.alignment = 0x100;
    assert(npu_hal_memory_alloc(hal, &request, &global) == 0);
    assert(npu_hal_memory_address(hal, global, core0, &address0) == 0);
    assert(npu_hal_memory_address(hal, global, core1, &address1) == 0);
    assert(address0 == 0x1000 && address1 == address0);

    request.scope = NPU_HAL_MEMORY_CLUSTER;
    request.location.cluster = 0;
    request.location.core = NPU_HAL_ANY_CORE;
    assert(npu_hal_memory_alloc(hal, &request, &local0) == 0);
    request.location.cluster = 1;
    assert(npu_hal_memory_alloc(hal, &request, &local1) == 0);
    assert(npu_hal_memory_address(hal, local0, core0, &address0) == 0);
    assert(npu_hal_memory_address(hal, local1, core1, &address1) == 0);
    assert(address0 == 0x10000 && address1 == 0x10000);
    assert(!npu_hal_memory_visible(hal, local0, core1));

    assert(npu_hal_memory_capacity(hal, NPU_HAL_MEMORY_URAM, core0,
                                   &capacity) == 0);
    assert(capacity == 0x8000);
    request.memory_class = NPU_HAL_MEMORY_URAM;
    request.scope = NPU_HAL_MEMORY_CORE;
    request.usage = NPU_HAL_MEMORY_KERNEL_URAM;
    request.location = core0;
    request.size = 0x8000;
    request.alignment = 0x8000;
    assert(npu_hal_memory_alloc(hal, &request, &uram0) == 0);
    assert(npu_hal_memory_address(hal, uram0, core0, &address0) == 0);
    assert(address0 == 0x30000);
    assert(!npu_hal_memory_visible(hal, uram0, core1));
    request.location = core1;
    assert(npu_hal_memory_alloc(hal, &request, &uram1) == 0);
    assert(npu_hal_memory_address(hal, uram1, core1, &address1) == 0);
    assert(address1 == 0x30000);
    request.size = 0x4000;
    assert(npu_hal_memory_alloc(hal, &request, &invalid) == -EINVAL);
    assert(!invalid);
    request.memory_class = NPU_HAL_MEMORY_NORMAL;
    request.scope = NPU_HAL_MEMORY_DEVICE;
    request.usage = NPU_HAL_MEMORY_BUFFER;
    request.location.cluster = NPU_HAL_ANY_CLUSTER;
    request.location.core = NPU_HAL_ANY_CORE;
    request.size = 0x100;
    request.alignment = 0x100;

    assert(npu_hal_data_plane_export(hal, &exported, handles) == 0);
    assert(exported.handle_count == 4);
    assert(npu_hal_data_plane_import(&data_plane, &exported, handles,
                                     exported.handle_count) == 0);
    assert(npu_hal_transfer_prepare(hal, global, core1, 0, sizeof(value),
                                    NPU_HAL_DATA_HOST_TO_DEVICE,
                                    &token) == 0);
    assert(token.provider == info.data_provider &&
           token.direction == NPU_HAL_DATA_HOST_TO_DEVICE);
    assert(token.opaque[0] == 0x11000 && token.opaque[1] == 1);
    value = UINT64_C(0xaabbccddeeff0011);
    assert(npu_hal_data_plane_transfer(data_plane, &token, &value,
                                       sizeof(value)) == 0);
    assert(npu_hal_transfer_finish(hal, &token, 0) == 0);
    fd = endpoint_fd(directory, 2, "h2c_0", O_RDONLY);
    assert(fd >= 0);
    observed = 0;
    assert(pread(fd, &observed, sizeof(observed), 0x11000) ==
           (ssize_t)sizeof(observed));
    close(fd);
    assert(observed == value);

    npu_hal_data_plane_close(data_plane);
    {
        npu_hal_data_export_t unsupported = exported;
        npu_hal_data_plane_t *unexpected = 0;

        unsupported.provider = UINT32_MAX;
        assert(npu_hal_data_plane_import(&unexpected, &unsupported, handles,
                                         unsupported.handle_count) ==
               -ENOTSUP);
        assert(!unexpected);
    }
    /* ClusterMMU register access. */
    {
        uint32_t observed_reg = 0;
        uint64_t observed64 = 0;
        int present = 0;

        /* A LO half bound to a translation window must be 4 KiB aligned.
         * The hardware ignores a misaligned write and reports no error, so
         * the HAL has to reject it or the caller sees a phantom success. */
        assert(npu_hal_mmu_write_reg(hal, 0, DAYAN_MMU_FIX_MAP_BASE_LO,
                                     0x1234) == -EINVAL);
        assert(npu_hal_mmu_write_reg(hal, 0, DAYAN_MMU_REDIR_PAGE_FAULT_LO,
                                     0x0800) == -EINVAL);
        assert(npu_hal_mmu_write_reg(hal, 0, DAYAN_MMU_FIX_MAP_BASE_LO,
                                     0x2000) == 0);
        /* PTW_BASE carries no such rule: page tables are page-aligned by
         * construction and the hardware imposes nothing extra. */
        assert(npu_hal_mmu_write_reg(hal, 0, DAYAN_MMU_PTW_BASE_LO,
                                     0x1240) == 0);
        /* Unaligned and out-of-page offsets are rejected outright. */
        assert(npu_hal_mmu_write_reg(hal, 0, 0x002, 0) == -EINVAL);
        assert(npu_hal_mmu_write_reg(hal, 0, DAYAN_MMU_APB_SIZE, 0) ==
               -EINVAL);

        /*
         * The stub backs writes with xdma0_h2c_0 and reads with
         * xdma0_c2h_0, so a register write is not visible to a register
         * read.  Writes are therefore checked by inspecting the h2c file
         * directly, and read paths by seeding the c2h file.
         *
         * These options set mmio_address_offset, which the backend adds to
         * every address below the MMIO limit, so the MMU page lands that far
         * into the file.
         */
        const off_t NPU_MMU_TEST_BASE = (off_t)DAYAN_MMU_APB_BASE + 0x10000;
        int h2c = endpoint_fd(directory, 0, "h2c_0", O_RDONLY);
        int c2h = endpoint_fd(directory, 0, "c2h_0", O_RDWR);
        uint32_t raw[2];

        assert(h2c >= 0 && c2h >= 0);

        assert(npu_hal_shared_resource_init(hal, 0, 0x00ffff00U) == 0);
        assert(pread(h2c, raw, sizeof(uint32_t),
                     (off_t)DAYAN_RESOURCE_GLOBAL_INIT + 0x10000) ==
               (ssize_t)sizeof(uint32_t));
        assert(raw[0] == 0x00ffff00U);
        assert(npu_hal_shared_resource_init(hal, 0, 0) == 0);
        assert(npu_hal_shared_resource_init(NULL, 0, 1) == -EINVAL);
        assert(npu_hal_shared_resource_init(hal, options.cluster_count, 1) ==
               -EINVAL);

        /* 64-bit writes are LO then HI, with HI holding only bits [55:32]. */
        assert(npu_hal_mmu_write64(hal, 0, DAYAN_MMU_PTW_BASE_LO,
                                   UINT64_C(0x00ab00000012f000)) == 0);
        assert(pread(h2c, raw, sizeof(raw),
                     (off_t)(NPU_MMU_TEST_BASE + DAYAN_MMU_PTW_BASE_LO)) ==
               (ssize_t)sizeof(raw));
        assert(raw[0] == 0x0012f000);
        assert(raw[1] == 0x00ab0000);
        /* Anything above bit 55 cannot be represented; truncating it
         * silently would map to an address nobody asked for. */
        assert(npu_hal_mmu_write64(hal, 0, DAYAN_MMU_PTW_BASE_LO,
                                   UINT64_C(0x0100000000000000)) == -EINVAL);
        assert(pread(h2c, raw, sizeof(raw),
                     (off_t)(NPU_MMU_TEST_BASE + DAYAN_MMU_PTW_BASE_LO)) ==
               (ssize_t)sizeof(raw));
        assert(raw[0] == 0x0012f000 && raw[1] == 0x00ab0000);

        /* read64 reassembles the pair and masks HI to 24 bits. */
        raw[0] = 0x00033000;
        raw[1] = 0xff00cd00;
        assert(pwrite(c2h, raw, sizeof(raw),
                      (off_t)(NPU_MMU_TEST_BASE + DAYAN_MMU_PTW_BASE_LO)) ==
               (ssize_t)sizeof(raw));
        assert(npu_hal_mmu_read64(hal, 0, DAYAN_MMU_PTW_BASE_LO,
                                  &observed64) == 0);
        assert(observed64 == UINT64_C(0x00cd0000033000));

        /* A successful read of the MMU page is sufficient for presence. */
        observed_reg = 0;
        assert(pwrite(c2h, &observed_reg, sizeof(observed_reg),
                      (off_t)(NPU_MMU_TEST_BASE +
                              DAYAN_MMU_FIX_MAP_SIZE_LO)) ==
               (ssize_t)sizeof(observed_reg));
        assert(npu_hal_mmu_present(hal, 0, &present) == 0);
        assert(present == 1);

        /* Only 8 ASIDs exist, so a selective flush past that is a bug. */
        assert(npu_hal_mmu_flush(hal, 0, 1, DAYAN_MMU_ASID_COUNT, 10) ==
               -EINVAL);

        /* Flush handshake, driven by seeding the read-side status words. */
        observed_reg = 0;
        assert(pwrite(c2h, &observed_reg, sizeof(observed_reg),
                      (off_t)(NPU_MMU_TEST_BASE + DAYAN_MMU_FLUSH_BUSY)) ==
               (ssize_t)sizeof(observed_reg));
        assert(pwrite(c2h, &observed_reg, sizeof(observed_reg),
                      (off_t)(NPU_MMU_TEST_BASE +
                              DAYAN_MMU_FLUSH_ERR_CODE)) ==
               (ssize_t)sizeof(observed_reg));
        observed_reg = DAYAN_MMU_FLUSH_DONE_BIT;
        assert(pwrite(c2h, &observed_reg, sizeof(observed_reg),
                      (off_t)(NPU_MMU_TEST_BASE + DAYAN_MMU_FLUSH_DONE)) ==
               (ssize_t)sizeof(observed_reg));
        /*
         * DONE stays set on the read side because nothing clears it here, so
         * the post-ACK recheck reports -EIO.  That recheck is what stops a
         * later flush from mistaking this DONE for its own completion.
         */
        assert(npu_hal_mmu_flush(hal, 0, 0, 0, 100) == -EIO);
        assert(pread(h2c, &observed_reg, sizeof(observed_reg),
                     (off_t)(NPU_MMU_TEST_BASE + DAYAN_MMU_FLUSH_ACK)) ==
               (ssize_t)sizeof(observed_reg));
        assert(observed_reg == DAYAN_MMU_FLUSH_ACK_BIT);
        /* The TLB_FLUSH command word carries mode and ASID. */
        assert(pread(h2c, &observed_reg, sizeof(observed_reg),
                     (off_t)(NPU_MMU_TEST_BASE + DAYAN_MMU_TLB_FLUSH)) ==
               (ssize_t)sizeof(observed_reg));
        assert(observed_reg == DAYAN_MMU_TLB_FLUSH_GO);
        assert(npu_hal_mmu_flush(hal, 0, 1, 5, 100) == -EIO);
        assert(pread(h2c, &observed_reg, sizeof(observed_reg),
                     (off_t)(NPU_MMU_TEST_BASE + DAYAN_MMU_TLB_FLUSH)) ==
               (ssize_t)sizeof(observed_reg));
        assert(observed_reg == (DAYAN_MMU_TLB_FLUSH_GO |
                                DAYAN_MMU_TLB_FLUSH_ASID_SEL |
                                (5U << DAYAN_MMU_TLB_FLUSH_ASID_SHIFT)));
        /* A busy-reject means nothing was invalidated: -EBUSY, not success. */
        observed_reg = DAYAN_MMU_FLUSH_ERR_BUSY_REJECT;
        assert(pwrite(c2h, &observed_reg, sizeof(observed_reg),
                      (off_t)(NPU_MMU_TEST_BASE +
                              DAYAN_MMU_FLUSH_ERR_CODE)) ==
               (ssize_t)sizeof(observed_reg));
        assert(npu_hal_mmu_flush(hal, 0, 0, 0, 100) == -EBUSY);
        /* BUSY never clearing must time out rather than spin forever. */
        observed_reg = DAYAN_MMU_FLUSH_BUSY_BIT;
        assert(pwrite(c2h, &observed_reg, sizeof(observed_reg),
                      (off_t)(NPU_MMU_TEST_BASE + DAYAN_MMU_FLUSH_BUSY)) ==
               (ssize_t)sizeof(observed_reg));
        assert(npu_hal_mmu_flush(hal, 0, 0, 0, 1) == -ETIMEDOUT);

        close(c2h);
        close(h2c);
    }

    assert(npu_hal_memory_free(hal, local1) == 0);
    assert(npu_hal_memory_free(hal, local0) == 0);
    assert(npu_hal_memory_free(hal, uram1) == 0);
    assert(npu_hal_memory_free(hal, uram0) == 0);
    assert(npu_hal_memory_free(hal, global) == 0);
    npu_hal_close(hal);
    hal = 0;
    create_event_endpoint(directory, 0);
    create_event_endpoint(directory, 2);
    options.memory_regions = regions;
    options.memory_region_count = sizeof(regions) / sizeof(regions[0]);
    assert(npu_hal_open(&hal, &options) == 0);
    {
        struct irq_writer writer;
        pthread_t thread;
        uint32_t irq_cluster = 0;
        int event0 = endpoint_fd(directory, 0, "events_0",
                                 O_WRONLY | O_NONBLOCK);
        int event1 = endpoint_fd(directory, 2, "events_0",
                                 O_WRONLY | O_NONBLOCK);

        assert(event0 >= 0 && event1 >= 0);
        assert(npu_hal_irq_wait(hal, 0, &irq_cluster, &pending) ==
               -ETIMEDOUT);
        writer.status_fd = endpoint_fd(directory, 2, "user", O_RDWR);
        writer.event_fd = event1;
        writer.status = UINT32_C(1) << core1.core;
        assert(writer.status_fd >= 0);
        assert(pthread_create(&thread, 0, write_irq, &writer) == 0);
        assert(npu_hal_irq_wait(hal, 100, &irq_cluster, &pending) == 0);
        assert(pthread_join(thread, 0) == 0);
        assert(irq_cluster == core1.cluster && pending == writer.status);
        close(writer.status_fd);
        close(event1);
        close(event0);
    }
    npu_hal_close(hal);
    hal = 0;
    options.memory_regions = 0;
    options.memory_region_count = 0;
    options.normal_memory_size_mib = 4096;
    /*
     * The split costs every non-Buffer usage the Buffer pool it can no longer
     * allocate from, and only the MMU needs it, so it is off unless asked for.
     * With it off the region stays whole and there is no pool to report.
     */
    assert(npu_hal_open(&hal, &options) == 0);
    {
        uint64_t pool_base = 0;
        uint64_t pool_size = 0;
        npu_hal_allocation_t *unsplit = 0;
        uint64_t unsplit_address = 0;

        assert(npu_hal_firmware_pool(hal, &pool_base, &pool_size) ==
               -ENODEV);
        npu_hal_memory_request_init(&request);
        request.memory_class = NPU_HAL_MEMORY_NORMAL;
        request.scope = NPU_HAL_MEMORY_DEVICE;
        request.size = 0x1000;
        request.alignment = 0x1000;
        /* A firmware usage may take the region base, which under the split
         * belongs to the firmware pool and above it to Buffers. */
        request.usage = NPU_HAL_MEMORY_SHARED;
        assert(npu_hal_memory_alloc(hal, &request, &unsplit) == 0);
        assert(npu_hal_memory_address(hal, unsplit, core1,
                                      &unsplit_address) == 0);
        assert(unsplit_address == UINT64_C(0x80000000));
        assert(npu_hal_memory_free(hal, unsplit) == 0);
    }
    npu_hal_close(hal);
    hal = 0;
    options.split_normal_memory = 1;
    assert(npu_hal_open(&hal, &options) == 0);
    assert(npu_hal_memory_capacity(hal, NPU_HAL_MEMORY_NORMAL, core1,
                                   &capacity) == 0);
    /* Capacity is the installed region, not what one request can reach: the
     * split means no single allocation can use all of it. */
    assert(capacity == UINT64_C(0x100000000));
    /*
     * The firmware pool is capped, not proportional: 4 GiB of Normal memory
     * still leaves the firmware 64 MiB and gives the rest to Buffers.  A
     * proportional split would strand 2 GiB behind a firmware that cannot
     * address past its own linker window.
     *
     * It sits at the region base, so the firmware's physical address does not
     * move when this option does.  That is what lets server/mmu.c map it with
     * an offset of zero in the default configuration.
     */
    {
        uint64_t pool_base = 0;
        uint64_t pool_size = 0;
        npu_hal_allocation_t *fw = 0;
        npu_hal_allocation_t *buf = 0;
        uint64_t fw_address = 0;
        uint64_t buf_address = 0;

        assert(npu_hal_firmware_pool(hal, &pool_base, &pool_size) == 0);
        assert(pool_base == UINT64_C(0x80000000));
        assert(pool_size == UINT64_C(0x04000000));

        npu_hal_memory_request_init(&request);
        request.memory_class = NPU_HAL_MEMORY_NORMAL;
        request.scope = NPU_HAL_MEMORY_DEVICE;
        request.size = 0x1000;
        request.alignment = 0x1000;
        /* Firmware-owned memory comes from the pool, Buffers from above it. */
        request.usage = NPU_HAL_MEMORY_SHARED;
        assert(npu_hal_memory_alloc(hal, &request, &fw) == 0);
        assert(npu_hal_memory_address(hal, fw, core1, &fw_address) == 0);
        assert(fw_address >= pool_base &&
               fw_address < pool_base + pool_size);
        request.usage = NPU_HAL_MEMORY_BUFFER;
        request.owner = 1;
        assert(npu_hal_memory_alloc(hal, &request, &buf) == 0);
        assert(npu_hal_memory_address(hal, buf, core1, &buf_address) == 0);
        assert(buf_address >= pool_base + pool_size);
        assert(npu_hal_memory_free(hal, buf) == 0);
        assert(npu_hal_memory_free(hal, fw) == 0);

        /*
         * A fixed-address reservation is confined to its usage's pool too.
         * Bounds-checking against the whole region instead would let firmware
         * memory be pinned inside the Buffer pool, or a range straddle the
         * boundary, and the disjointness would hold for alloc but not reserve.
         */
        request.usage = NPU_HAL_MEMORY_SHARED;
        request.owner = UINT32_MAX;
        request.fixed_address = pool_base + pool_size;
        assert(npu_hal_memory_reserve(hal, &request, &fw) == -ERANGE);
        request.fixed_address = pool_base + pool_size - 0x1000;
        assert(npu_hal_memory_reserve(hal, &request, &fw) == 0);
        assert(npu_hal_memory_free(hal, fw) == 0);
        request.usage = NPU_HAL_MEMORY_BUFFER;
        request.owner = 1;
        request.fixed_address = pool_base;
        assert(npu_hal_memory_reserve(hal, &request, &buf) == -ERANGE);
        request.fixed_address = pool_base + pool_size;
        assert(npu_hal_memory_reserve(hal, &request, &buf) == 0);
        assert(npu_hal_memory_free(hal, buf) == 0);
        request.fixed_address = 0;
        request.owner = 0;
    }
    assert(npu_hal_memory_capacity(hal, NPU_HAL_MEMORY_URAM, core1,
                                   &capacity) == 0);
    assert(capacity == 0x180000);
    npu_hal_memory_request_init(&request);
    request.memory_class = NPU_HAL_MEMORY_URAM;
    request.scope = NPU_HAL_MEMORY_CORE;
    request.usage = NPU_HAL_MEMORY_KERNEL_URAM;
    request.location = core1;
    request.size = 0x8000;
    request.alignment = 0x8000;
    assert(npu_hal_memory_alloc(hal, &request, &uram1) == 0);
    assert(npu_hal_memory_address(hal, uram1, core1, &address1) == 0);
    assert(address1 == 0xe00000);
    assert(npu_hal_memory_free(hal, uram1) == 0);
    uram1 = 0;
    assert(npu_hal_device_status(hal) == 0);
    fd = endpoint_fd(directory, 0, "user", O_WRONLY);
    assert(fd >= 0);
    assert(ftruncate(fd, 0) == 0);
    close(fd);
    pending = 0;
    {
        uint32_t irq_cluster = 0;

        assert(npu_hal_irq_wait(hal, 100, &irq_cluster, &pending) == -EIO);
        assert(irq_cluster == NPU_HAL_ANY_CLUSTER && pending == 0);
    }
    assert(npu_hal_device_status(hal) == -EIO);
    assert(npu_hal_copy_to_address(hal, 0, 0, &value, sizeof(value)) ==
           -EIO);
    npu_hal_close(hal);
    hal = 0;
    options.normal_memory_size_mib = 0;
    assert(npu_hal_open(&hal, &options) == 0);
    {
        /* A region smaller than twice the cap is halved, so the firmware still
         * gets room and Buffers still get a pool. */
        uint64_t pool_base = 0;
        uint64_t pool_size = 0;

        assert(npu_hal_firmware_pool(hal, &pool_base, &pool_size) == 0);
        assert(pool_base == UINT64_C(0x80000000));
        assert(pool_size == UINT64_C(0x02000000));
    }
    npu_hal_close(hal);
    hal = 0;
    options.memory_regions = overlapping_regions;
    options.memory_region_count =
        sizeof(overlapping_regions) / sizeof(overlapping_regions[0]);
    assert(npu_hal_open(&hal, &options) == -EINVAL);
    assert(!hal);
    return 0;
}
