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

#define _POSIX_C_SOURCE 200809L

/*
 * Offline test of the soc_mmap backend and data plane.  A regular file stands
 * in for /dev/mem: its offsets play the Host physical addresses, so every
 * access can be checked with pread/pwrite.  This proves the address
 * translation, the layout and range checks, the HAL routing and the Runtime
 * token path.  It says nothing about bus widths, ordering, caching or
 * register semantics on the QEMU NPU model or a chip.
 */

#include <assert.h>
#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "hal.h"
#include "npu_pcu.h"
#include "backend/soc_mmap/npu_hal_soc_mmap.h"
#include "backend/soc_mmap/io.h"
#include "backend/soc_mmap/layout.h"
#include "platform_regs.h"

/* Host PAs inside the stand-in file.  DDR deliberately sits nowhere near its
 * NPU-side address, so a missing translation shows up. */
#define NPU_BASE   UINT64_C(0x200000)
#define DDR_HOST   UINT64_C(0x1400000)
#define DDR_SIZE   UINT64_C(0x100000)
#define DDR_NPU    UINT64_C(0x80000000)
#define FILE_SIZE  (DDR_HOST + DDR_SIZE + 0x100000)

static int memory_fd;
static char path[] = "/tmp/dayan-soc-mmap-test-XXXXXX";

static void file_read(uint64_t offset, void *data, size_t size)
{
    assert(pread(memory_fd, data, size, (off_t)offset) == (ssize_t)size);
}

static void file_write(uint64_t offset, const void *data, size_t size)
{
    assert(pwrite(memory_fd, data, size, (off_t)offset) == (ssize_t)size);
}

static uint64_t read_u64(uint64_t offset)
{
    uint64_t value = 0;

    file_read(offset, &value, sizeof(value));
    return value;
}

static uint32_t read_u32(uint64_t offset)
{
    uint32_t value = 0;

    file_read(offset, &value, sizeof(value));
    return value;
}

static void write_u32(uint64_t offset, uint32_t value)
{
    file_write(offset, &value, sizeof(value));
}

static void write_u64(uint64_t offset, uint64_t value)
{
    file_write(offset, &value, sizeof(value));
}

/* Fills [offset, offset + size) with a marker and returns it for comparison. */
static void mark(uint64_t offset, unsigned char *saved, size_t size)
{
    memset(saved, 0xa5, size);
    file_write(offset, saved, size);
}

static void assert_unchanged(uint64_t offset, const unsigned char *saved,
                             size_t size)
{
    unsigned char now[64];

    assert(size <= sizeof(now));
    file_read(offset, now, size);
    assert(memcmp(now, saved, size) == 0);
}

static void test_layout(void)
{
    struct soc_layout_entry e[SOC_LAYOUT_MAX_ENTRIES];
    int count = soc_layout_build(4, DAYAN_URAM_SIZE, e, SOC_LAYOUT_MAX_ENTRIES);
    int i;

    assert(count > 0);
    for (i = 0; i < count; ++i) {
        assert(e[i].offset + e[i].size <= DAYAN_CLUSTER_APERTURE_SIZE);
        if (i)
            assert(e[i].offset >= e[i - 1].offset + e[i - 1].size);
    }
    /* Last entries are the per-core URAMs at the platform stride. */
    assert(e[count - 1].memory_class == NPU_HAL_MEMORY_URAM &&
           e[count - 1].core == 3 && e[count - 1].offset == DAYAN_URAM(3));
    assert(soc_layout_build(0, DAYAN_URAM_SIZE, e,
                            SOC_LAYOUT_MAX_ENTRIES) == -EINVAL);
    assert(soc_layout_build(DAYAN_CLUSTER_MAX_HARTS + 1, DAYAN_URAM_SIZE, e,
                            SOC_LAYOUT_MAX_ENTRIES) == -EINVAL);
    assert(soc_layout_build(4, DAYAN_URAM_SIZE, e, 4) == -E2BIG);
    assert(soc_layout_build(4, 0, e, SOC_LAYOUT_MAX_ENTRIES) == -EINVAL);
    assert(soc_layout_build(4, 0x100000, e, SOC_LAYOUT_MAX_ENTRIES) == -EINVAL);
}

static void test_io(void)
{
    _Alignas(8) unsigned char device[64];
    unsigned char source[64];
    unsigned char back[64];
    uint64_t value = UINT64_C(0x0102030405060708);
    size_t index;

    for (index = 0; index < sizeof(source); ++index)
        source[index] = (unsigned char)(index * 7U + 1U);
    memset(device, 0, sizeof(device));
    /* Memory copy may narrow at unaligned edges ... */
    npu_hal_soc_mmap_mem_write(device + 3, source, 45);
    assert(memcmp(device + 3, source, 45) == 0);
    assert(device[2] == 0 && device[48] == 0);
    npu_hal_soc_mmap_mem_read(back, device + 3, 45);
    assert(memcmp(back, source, 45) == 0);
    /* ... registers never do. */
    memset(device, 0, sizeof(device));
    assert(npu_hal_soc_mmap_reg_write(device + 4, &value, 8) == -EINVAL);
    assert(npu_hal_soc_mmap_reg_write(device + 2, &value, 4) == -EINVAL);
    assert(npu_hal_soc_mmap_reg_write(device, &value, 3) == -EINVAL);
    for (index = 0; index < sizeof(device); ++index)
        assert(device[index] == 0);
    assert(npu_hal_soc_mmap_reg_write(device + 8, &value, 8) == 0);
    assert(memcmp(device + 8, &value, 8) == 0);
    /* Only the declared coherence models are accepted. */
    assert(npu_hal_soc_mmap_ddr_sync_for_device(
               NPU_HAL_SOC_MMAP_DDR_UNCACHED, device, 8) == 0);
    assert(npu_hal_soc_mmap_ddr_sync_for_cpu(
               NPU_HAL_SOC_MMAP_DDR_COHERENT, device, 8) == 0);
    assert(npu_hal_soc_mmap_ddr_sync_for_cpu(0, device, 8) == -ENOTSUP);
}

static void soc_defaults(npu_hal_soc_mmap_options_t *soc)
{
    assert(npu_hal_soc_mmap_options_init(soc) == 0);
    assert(soc->ddr_device_base == DAYAN_DRAM_BASE && !soc->npu_base);
    soc->memory_device = path;
    soc->npu_base = NPU_BASE;
    soc->ddr_host_base = DDR_HOST;
    soc->ddr_size = DDR_SIZE;
    soc->ddr_coherence = NPU_HAL_SOC_MMAP_DDR_UNCACHED;
}

static void test_bootrom_discovery(void)
{
    const char id[] = "FpgaSingleTinyClusterConfig-abc123-dirty";
    char empty[sizeof(id)] = {0};
    npu_hal_soc_mmap_options_t soc;
    npu_hal_options_t options;
    npu_hal_info_t info;
    npu_hal_t *hal = 0;
    uint64_t capacity;

    soc_defaults(&soc);
    npu_hal_options_init(&options);
    options.backend_options = &soc.base;
    assert(pwrite(memory_fd, id, sizeof(id),
                  NPU_BASE + DAYAN_BOOTROM_BUILD_ID_ADDRESS) == sizeof(id));
    assert(npu_hal_open(&hal, &options) == 0);
    assert(npu_hal_get_info(hal, &info) == 0);
    assert(info.cores_per_cluster == 1 && info.memory_region_count == 3);
    assert(npu_hal_memory_capacity(hal, NPU_HAL_MEMORY_URAM,
               (npu_hal_core_t){0, 0}, &capacity) == 0);
    assert(capacity == DAYAN_URAM_SIZE_768K);
    /* The SoC access bounds must follow the detected size too. */
    assert(npu_hal_copy_from_address(hal, 0, empty,
               DAYAN_URAM(0) + capacity, 1) == -EFAULT);
    npu_hal_close(hal);
    assert(pwrite(memory_fd, empty, sizeof(empty),
                  NPU_BASE + DAYAN_BOOTROM_BUILD_ID_ADDRESS) == sizeof(empty));
}

static int open_with(npu_hal_t **hal, const npu_hal_soc_mmap_options_t *soc,
                     const npu_hal_memory_region_t *regions, uint32_t count)
{
    npu_hal_options_t options;

    npu_hal_options_init(&options);
    options.cluster_count = 1;
    options.cores_per_cluster = 4;
    options.memory_regions = regions;
    options.memory_region_count = count;
    options.backend_options = &soc->base;
    return npu_hal_open(hal, &options);
}

static void test_configuration(void)
{
    npu_hal_memory_region_t regions[NPU_HAL_SOC_MMAP_MAX_REGIONS];
    npu_hal_memory_region_t bad[NPU_HAL_SOC_MMAP_MAX_REGIONS];
    npu_hal_soc_mmap_options_t soc;
    npu_hal_soc_mmap_options_t broken;
    npu_hal_options_t options;
    npu_hal_t *hal = 0;
    int count;

    soc_defaults(&soc);
    count = npu_hal_soc_mmap_memory_regions(&soc, 4, DAYAN_URAM_SIZE, regions,
                                            NPU_HAL_SOC_MMAP_MAX_REGIONS);
    /* Normal, Coherent, one URAM per core. */
    assert(count == 6);
    assert(regions[0].memory_class == NPU_HAL_MEMORY_NORMAL &&
           regions[0].base == DDR_NPU && regions[0].size == DDR_SIZE);
    assert(regions[1].memory_class == NPU_HAL_MEMORY_COHERENT &&
           regions[1].base == DAYAN_COHERENT_WINDOW_BASE &&
           regions[1].size == DAYAN_ATOMIC_CACHE_WINDOW_SIZE);
    assert(regions[5].memory_class == NPU_HAL_MEMORY_URAM &&
           regions[5].scope == NPU_HAL_MEMORY_CORE && regions[5].core == 3 &&
           regions[5].base == DAYAN_URAM(3) &&
           regions[5].size == DAYAN_URAM_SIZE);
    assert(npu_hal_soc_mmap_memory_regions(&soc, 4, DAYAN_URAM_SIZE,
                                          regions, 3) == -E2BIG);

    /* The regions used to be the 4 GiB PCIe defaults while only the mapped
     * DDR existed: now the backend refuses anything not fully mapped. */
    assert(open_with(&hal, &soc, 0, 0) == 0);
    npu_hal_close(hal);
    hal = 0;
    memcpy(bad, regions, sizeof(regions));
    bad[0].size = UINT64_C(0x100000000);
    assert(open_with(&hal, &soc, bad, (uint32_t)count) == -EINVAL && !hal);
    /* Normal memory cannot be placed on registers or in the Coherent window. */
    memcpy(bad, regions, sizeof(regions));
    bad[0].base = DAYAN_MMU_APB_BASE;
    bad[0].size = 0x1000;
    assert(open_with(&hal, &soc, bad, (uint32_t)count) == -EINVAL && !hal);
    memcpy(bad, regions, sizeof(regions));
    bad[1].memory_class = NPU_HAL_MEMORY_URAM;
    bad[1].scope = NPU_HAL_MEMORY_CORE;
    bad[1].cluster = 0;
    bad[1].core = 0;
    assert(open_with(&hal, &soc, bad, (uint32_t)count) == -EINVAL && !hal);
    /* Core 1's URAM may not be described as core 0's, nor as shared. */
    memcpy(bad, regions, sizeof(regions));
    bad[3].core = 0;
    bad[2].core = 1;
    assert(open_with(&hal, &soc, bad, (uint32_t)count) == -EINVAL && !hal);
    memcpy(bad, regions, sizeof(regions));
    bad[2].scope = NPU_HAL_MEMORY_DEVICE;
    bad[2].cluster = NPU_HAL_ANY_CLUSTER;
    bad[2].core = NPU_HAL_ANY_CORE;
    assert(open_with(&hal, &soc, bad, (uint32_t)count) == -EINVAL && !hal);
    /* A URAM region running past its module. */
    memcpy(bad, regions, sizeof(regions));
    bad[2].size = DAYAN_URAM_SIZE + 0x1000;
    assert(open_with(&hal, &soc, bad, (uint32_t)count) == -EINVAL && !hal);

    /* Option validation, before any device is opened. */
    broken = soc;
    broken.npu_base = 0;
    assert(open_with(&hal, &broken, regions, (uint32_t)count) == -EINVAL);
    broken = soc;
    broken.npu_base = NPU_BASE + 8;
    assert(open_with(&hal, &broken, regions, (uint32_t)count) == -EINVAL);
    broken = soc;
    broken.npu_base = UINT64_MAX & ~UINT64_C(0xfff);
    assert(open_with(&hal, &broken, regions, (uint32_t)count) == -EINVAL);
    broken = soc;
    broken.ddr_size = 0;
    assert(open_with(&hal, &broken, regions, (uint32_t)count) == -EINVAL);
    broken = soc;
    broken.ddr_host_base = UINT64_MAX & ~UINT64_C(0xfff);
    assert(open_with(&hal, &broken, regions, (uint32_t)count) == -EINVAL);
    /* DDR and the aperture may not share Host PAs ... */
    broken = soc;
    broken.ddr_host_base = NPU_BASE + DAYAN_CLUSTER_APERTURE_SIZE - 0x1000;
    assert(open_with(&hal, &broken, regions, (uint32_t)count) == -EINVAL);
    /* ... nor NPU-side addresses. */
    broken = soc;
    broken.ddr_device_base = DAYAN_URAM_BASE;
    assert(open_with(&hal, &broken, regions, (uint32_t)count) == -EINVAL);
    broken = soc;
    broken.ddr_coherence = 0;
    assert(open_with(&hal, &broken, regions, (uint32_t)count) == -EINVAL);
    broken = soc;
    broken.memory_device = "/nonexistent/dayan-mem";
    assert(open_with(&hal, &broken, regions, (uint32_t)count) == -ENOENT);
    assert(!hal);

    /* No platform rule relates two Clusters' addresses. */
    npu_hal_options_init(&options);
    options.cluster_count = 2;
    options.memory_regions = regions;
    options.memory_region_count = (uint32_t)count;
    options.backend_options = &soc.base;
    assert(npu_hal_open(&hal, &options) == -ENOTSUP && !hal);
}

/* Large-range snapshots for the Device-to-Device copy tests. */
static unsigned char *snapshot(uint64_t offset, size_t size)
{
    unsigned char *copy = malloc(size);

    assert(copy);
    file_read(offset, copy, size);
    return copy;
}

/* Hashes the absolute file offset, so no two places in the file hold the
 * same run of bytes: a copy that lands anywhere it should not shows up. */
static void fill(uint64_t offset, size_t size, unsigned char seed)
{
    unsigned char *data = malloc(size);
    size_t index;

    assert(data);
    for (index = 0; index < size; ++index) {
        uint64_t x = (offset + index) ^ ((uint64_t)seed << 40);

        x ^= x >> 29;
        x *= UINT64_C(0x9e3779b97f4a7c15);
        data[index] = (unsigned char)(x >> 56);
    }
    file_write(offset, data, size);
    free(data);
}

static void assert_snapshot(uint64_t offset, unsigned char *saved,
                            size_t size)
{
    unsigned char *now = snapshot(offset, size);

    assert(memcmp(now, saved, size) == 0);
    free(now);
    free(saved);
}

/* Mappings of the stand-in file in this process: the HAL's two, plus one per
 * Runtime provider that has mapped the DDR. */
static int file_mappings(void)
{
    char line[512];
    FILE *maps = fopen("/proc/self/maps", "r");
    int count = 0;

    assert(maps);
    while (fgets(line, sizeof(line), maps))
        count += strstr(line, path) != 0;
    fclose(maps);
    return count;
}

static npu_hal_data_token_t h2d_token(uint64_t host_base, uint64_t size,
                                      uint64_t offset, size_t length)
{
    npu_hal_data_token_t token;

    memset(&token, 0, sizeof(token));
    token.version = NPU_HAL_DATA_TOKEN_VERSION;
    token.provider = NPU_HAL_DATA_PROVIDER_SOC_MMAP;
    token.direction = NPU_HAL_DATA_HOST_TO_DEVICE;
    token.size = length;
    token.opaque[0] = host_base;
    token.opaque[1] = size;
    token.opaque[2] = offset;
    return token;
}

/*
 * A freshly imported provider: the allowed range comes from the export, and
 * even its first token is checked against it before anything is mapped.
 * Layout 2 let that first token choose the range, so one naming the IP
 * aperture wrote the mailbox.
 */
static void test_first_token(npu_hal_t *hal)
{
    npu_hal_data_export_t exported;
    npu_hal_data_export_t bad;
    npu_hal_data_plane_t *data_plane = 0;
    npu_hal_data_token_t token;
    int handles[NPU_HAL_DATA_MAX_HANDLES];
    unsigned char payload[64];
    unsigned char saved[64];
    int spare[1];
    int mapped;

    memset(payload, 0x3c, sizeof(payload));
    assert(npu_hal_data_plane_export(hal, &exported, handles) == 0);
    assert(exported.opaque[0] == NPU_HAL_SOC_MMAP_TOKEN_LAYOUT &&
           exported.opaque[2] == DDR_HOST && exported.opaque[3] == DDR_SIZE);

    /* Import refuses an export whose DDR range is not one the Server could
     * have configured. */
    spare[0] = dup(handles[0]);
    bad = exported;
    bad.opaque[0] = 2;
    assert(npu_hal_data_plane_import(&data_plane, &bad, spare, 1) == -EPROTO);
    bad = exported;
    bad.opaque[3] = 0;
    assert(npu_hal_data_plane_import(&data_plane, &bad, spare, 1) == -EPROTO);
    bad = exported;
    bad.opaque[2] = DDR_HOST + 8;
    assert(npu_hal_data_plane_import(&data_plane, &bad, spare, 1) == -EPROTO);
    bad = exported;
    bad.opaque[2] = UINT64_MAX & ~UINT64_C(0xfff);
    bad.opaque[3] = 0x2000;
    assert(npu_hal_data_plane_import(&data_plane, &bad, spare, 1) == -EPROTO);
    assert(!data_plane);
    close(spare[0]);

    assert(npu_hal_data_plane_import(&data_plane, &exported, handles, 1) == 0);
    mapped = file_mappings();

    /* First token names the IP aperture: the mailbox stays as it was. */
    mark(NPU_BASE + DAYAN_DEVICE_MAILBOX(0), saved, 32);
    token = h2d_token(NPU_BASE, DDR_SIZE, DAYAN_DEVICE_MAILBOX(0),
                      sizeof(payload));
    assert(npu_hal_data_plane_transfer(data_plane, &token, payload, 32) ==
           -EINVAL);  /* token.size must equal the length */
    token.size = 32;
    assert(npu_hal_data_plane_transfer(data_plane, &token, payload, 32) ==
           -EACCES);
    assert_unchanged(NPU_BASE + DAYAN_DEVICE_MAILBOX(0), saved, 32);
    /* Right base, wrong size: would reach past the reserved DDR. */
    mark(DDR_HOST + DDR_SIZE, saved, 64);
    token = h2d_token(DDR_HOST, DDR_SIZE * 2, DDR_SIZE, sizeof(payload));
    assert(npu_hal_data_plane_transfer(data_plane, &token, payload,
                                       sizeof(payload)) == -EACCES);
    assert_unchanged(DDR_HOST + DDR_SIZE, saved, 64);
    /* Inside the DDR but running off its end. */
    mark(DDR_HOST + DDR_SIZE - 8, saved, 8);
    token = h2d_token(DDR_HOST, DDR_SIZE, DDR_SIZE - 8, sizeof(payload));
    assert(npu_hal_data_plane_transfer(data_plane, &token, payload,
                                       sizeof(payload)) == -EACCES);
    assert_unchanged(DDR_HOST + DDR_SIZE - 8, saved, 8);
    /* Offset arithmetic that would wrap. */
    token = h2d_token(DDR_HOST, DDR_SIZE, UINT64_MAX - 7, sizeof(payload));
    assert(npu_hal_data_plane_transfer(data_plane, &token, payload,
                                       sizeof(payload)) == -EACCES);
    token = h2d_token(DDR_HOST, DDR_SIZE, 0, sizeof(payload));
    token.opaque[3] = 1;
    assert(npu_hal_data_plane_transfer(data_plane, &token, payload,
                                       sizeof(payload)) == -EACCES);
    /* None of those mapped anything. */
    assert(file_mappings() == mapped);

    /* A valid first token maps the exported DDR and writes there. */
    token = h2d_token(DDR_HOST, DDR_SIZE, 0x800, sizeof(payload));
    assert(npu_hal_data_plane_transfer(data_plane, &token, payload,
                                       sizeof(payload)) == 0);
    assert(file_mappings() == mapped + 1);
    file_read(DDR_HOST + 0x800, saved, sizeof(saved));
    assert(memcmp(saved, payload, sizeof(payload)) == 0);
    npu_hal_data_plane_close(data_plane);
}

/*
 * npu_hal_copy_address() bounces through 64 KiB chunks.  Each failure below
 * would, per chunk, have written the first 64 KiB before failing; the whole
 * copy is now checked first.
 */
static void test_copy_address(npu_hal_t *hal)
{
    const size_t size = 0x20000;
    const size_t chunk = 0x10000;
    unsigned char saved[64];
    unsigned char *before;
    unsigned char *after;

    fill(DDR_HOST, DDR_SIZE, 0x11);
    fill(NPU_BASE + DAYAN_URAM(0), DAYAN_URAM_SIZE, 0x22);

    /* Destination has only 64 KiB left in the DDR. */
    before = snapshot(DDR_HOST + DDR_SIZE - chunk, chunk);
    assert(npu_hal_copy_address(hal, 0, DDR_NPU + DDR_SIZE - chunk, DDR_NPU,
                                size) == -EFAULT);
    assert_snapshot(DDR_HOST + DDR_SIZE - chunk, before, chunk);
    /* Source has only 64 KiB left. */
    before = snapshot(DDR_HOST + 0x40000, size);
    assert(npu_hal_copy_address(hal, 0, DDR_NPU + 0x40000,
                                DDR_NPU + DDR_SIZE - chunk, size) == -EFAULT);
    assert_snapshot(DDR_HOST + 0x40000, before, size);
    /* Destination runs from URAM0 into the hole after it. */
    before = snapshot(NPU_BASE + DAYAN_URAM(0) + DAYAN_URAM_SIZE - chunk,
                      chunk);
    assert(npu_hal_copy_address(hal, 0,
                                DAYAN_URAM(0) + DAYAN_URAM_SIZE - chunk,
                                DDR_NPU, size) == -EFAULT);
    assert_snapshot(NPU_BASE + DAYAN_URAM(0) + DAYAN_URAM_SIZE - chunk, before,
                    chunk);
    /* Address overflow on either side. */
    assert(npu_hal_copy_address(hal, 0, UINT64_MAX - chunk + 1, DDR_NPU,
                                size) == -EOVERFLOW);
    before = snapshot(DDR_HOST + 0x40000, size);
    assert(npu_hal_copy_address(hal, 0, DDR_NPU + 0x40000,
                                UINT64_MAX - chunk + 1, size) == -EOVERFLOW);
    assert_snapshot(DDR_HOST + 0x40000, before, size);
    /* Registers are not bulk-copy memory, even at their exact width. */
    mark(NPU_BASE + DAYAN_DEVICE_MAILBOX(0), saved, 8);
    assert(npu_hal_copy_address(hal, 0, DAYAN_DEVICE_MAILBOX(0), DDR_NPU, 8) ==
           -EINVAL);
    assert_unchanged(NPU_BASE + DAYAN_DEVICE_MAILBOX(0), saved, 8);
    before = snapshot(DDR_HOST, 8);
    assert(npu_hal_copy_address(hal, 0, DDR_NPU, DAYAN_PCU_BASE, 8) ==
           -EINVAL);
    assert_snapshot(DDR_HOST, before, 8);
    /* Identical ranges are still validated. */
    assert(npu_hal_copy_address(hal, 0, DAYAN_BOOTROM_RESET_VECTOR,
                                DAYAN_BOOTROM_RESET_VECTOR, size) == -EFAULT);
    assert(npu_hal_device_status(hal) == 0);

    /* Valid copies larger than one chunk: DDR to DDR, URAM to DDR. */
    before = snapshot(DDR_HOST, size);
    assert(npu_hal_copy_address(hal, 0, DDR_NPU + 0x80000, DDR_NPU, size) == 0);
    after = snapshot(DDR_HOST + 0x80000, size);
    assert(memcmp(before, after, size) == 0);
    free(after);
    free(before);
    before = snapshot(NPU_BASE + DAYAN_URAM(0), size);
    assert(npu_hal_copy_address(hal, 0, DDR_NPU + 0xc0000, DAYAN_URAM(0),
                                size) == 0);
    after = snapshot(DDR_HOST + 0xc0000, size);
    assert(memcmp(before, after, size) == 0);
    free(after);
    free(before);
}

static void test_uram_capacity(void)
{
    const uint32_t sizes[] = { DAYAN_URAM_SIZE_1536K, DAYAN_URAM_SIZE_768K };
    npu_hal_memory_region_t regions[NPU_HAL_SOC_MMAP_MAX_REGIONS];
    npu_hal_soc_mmap_options_t soc;
    npu_hal_options_t options;
    npu_hal_t *hal = 0;

    soc_defaults(&soc);
    npu_hal_options_init(&options);
    options.backend_options = &soc.base;
    options.memory_regions = regions;
    for (size_t i = 0; i < sizeof(sizes) / sizeof(sizes[0]); ++i) {
        int count;

        options.uram_size = sizes[i];
        /* A stale 1.5 MiB pool must not open on a 768 KiB layout. */
        if (i)
            assert(npu_hal_open(&hal, &options) == -EINVAL && !hal);
        count = npu_hal_soc_mmap_memory_regions(
            &soc, options.cores_per_cluster, options.uram_size, regions,
            NPU_HAL_SOC_MMAP_MAX_REGIONS);
        assert(count == 6);
        options.memory_region_count = (uint32_t)count;
        assert(npu_hal_open(&hal, &options) == 0);
        for (uint32_t core = 0; core < options.cores_per_cluster; ++core) {
            npu_hal_core_t target = {0, core};
            uint64_t end = DAYAN_URAM(core) + sizes[i];
            uint64_t capacity;
            uint64_t value = UINT64_C(0x1122334455667788);
            uint64_t observed = 0;
            unsigned char saved[16];

            assert(regions[2 + core].base == DAYAN_URAM(core));
            assert(regions[2 + core].size == sizes[i]);
            assert(npu_hal_memory_capacity(hal, NPU_HAL_MEMORY_URAM, target,
                                           &capacity) == 0);
            assert(capacity == sizes[i]);
            assert(npu_hal_copy_to_address(hal, 0, end - sizeof(value), &value,
                                           sizeof(value)) == 0);
            assert(npu_hal_copy_from_address(hal, 0, &observed,
                                             end - sizeof(value),
                                             sizeof(value)) == 0);
            assert(observed == value);
            mark(NPU_BASE + end - 8, saved, sizeof(saved));
            assert(npu_hal_copy_to_address(hal, 0, end - 4, &value,
                                           sizeof(value)) == -EFAULT);
            assert(npu_hal_copy_from_address(hal, 0, &observed, end,
                                             sizeof(observed)) == -EFAULT);
            assert(npu_hal_copy_address(hal, 0, end - 4, DDR_NPU,
                                        sizeof(value)) == -EFAULT);
            assert_unchanged(NPU_BASE + end - 8, saved, sizeof(saved));
        }
        npu_hal_close(hal);
        hal = 0;
    }
}

int main(void)
{
    npu_hal_memory_region_t regions[NPU_HAL_SOC_MMAP_MAX_REGIONS];
    npu_hal_soc_mmap_options_t soc;
    npu_hal_t *hal = 0;
    npu_hal_info_t info;
    npu_hal_core_t core2 = { 0, 2 };
    npu_hal_memory_request_t request;
    npu_hal_allocation_t *buffer = 0;
    npu_hal_allocation_t *too_big = 0;
    npu_hal_allocation_t *coherent = 0;
    npu_hal_allocation_t *uram = 0;
    npu_hal_data_export_t exported;
    npu_hal_data_token_t token;
    npu_hal_data_plane_t *data_plane = 0;
    npu_pcu_batch_t batch;
    int handles[NPU_HAL_DATA_MAX_HANDLES];
    unsigned char saved[64];
    unsigned char payload[301];
    unsigned char back[301];
    uint64_t address;
    uint64_t capacity = 0;
    uint64_t value = UINT64_C(0x1122334455667788);
    uint64_t observed = 0;
    uint32_t word = 0x5a5a5a5aU;
    uint32_t cluster = 0;
    uint32_t pending = 0;
    uint32_t core;
    size_t index;
    int count;

    test_layout();
    test_io();
    memory_fd = mkstemp(path);
    assert(memory_fd >= 0);
    assert(ftruncate(memory_fd, (off_t)FILE_SIZE) == 0);
    test_bootrom_discovery();
    test_configuration();
    test_uram_capacity();

    soc_defaults(&soc);
    count = npu_hal_soc_mmap_memory_regions(&soc, 4, DAYAN_URAM_SIZE, regions,
                                            NPU_HAL_SOC_MMAP_MAX_REGIONS);
    assert(open_with(&hal, &soc, regions, (uint32_t)count) == 0);
    assert(npu_hal_get_info(hal, &info) == 0);
    assert(info.backend == NPU_HAL_BACKEND_SOC_MMAP &&
           info.data_provider == NPU_HAL_DATA_PROVIDER_SOC_MMAP &&
           info.data_handle_count == 1);

    /* The allocator sees exactly the mapped DDR. */
    assert(npu_hal_memory_capacity(hal, NPU_HAL_MEMORY_NORMAL, core2,
                                   &capacity) == 0 && capacity == DDR_SIZE);
    npu_hal_memory_request_init(&request);
    request.size = DDR_SIZE + 0x1000;
    assert(npu_hal_memory_alloc(hal, &request, &too_big) < 0 && !too_big);

    /* NPU address -> Host PA: DDR is translated, not offset by npu_base. */
    assert(npu_hal_copy_to_address(hal, 0, DDR_NPU + 0x40, &value,
                                   sizeof(value)) == 0);
    assert(read_u64(DDR_HOST + 0x40) == value);
    assert(npu_hal_copy_from_address(hal, 0, &observed, DDR_NPU + 0x40,
                                     sizeof(observed)) == 0);
    assert(observed == value);
    /* The aperture is npu_base + IP offset. */
    assert(npu_hal_command_submit(hal, core2, UINT64_C(0xabcdef0012345678)) ==
           0);
    assert(read_u64(NPU_BASE + DAYAN_DEVICE_MAILBOX(2)) ==
           UINT64_C(0xabcdef0012345678));
    assert(read_u32(NPU_BASE + DAYAN_CLINT_MSIP(2)) == 1);
    assert(npu_hal_wait_command_consumed(hal, core2, 0) == -ETIMEDOUT);
    write_u32(NPU_BASE + DAYAN_CLINT_MSIP(2), 0);
    assert(npu_hal_wait_command_consumed(hal, core2, 0) == 0);

    /* MMU: 32-bit registers; a 56-bit value is LO then HI. */
    assert(npu_hal_mmu_write_reg(hal, 0, DAYAN_MMU_GLOBAL_ASID, 0x5a) == 0);
    assert(read_u32(NPU_BASE + DAYAN_MMU_APB_BASE + DAYAN_MMU_GLOBAL_ASID) ==
           0x5a);
    assert(npu_hal_mmu_write64(hal, 0, DAYAN_MMU_PTW_BASE_LO,
                               UINT64_C(0x00123456789ab000)) == 0);
    assert(read_u32(NPU_BASE + DAYAN_MMU_APB_BASE + DAYAN_MMU_PTW_BASE_LO) ==
           0x789ab000U);
    assert(read_u32(NPU_BASE + DAYAN_MMU_APB_BASE + DAYAN_MMU_PTW_BASE_HI) ==
           0x00123456U);

    /* Register rules: wrong width, misalignment, or a 64-bit access to a
     * 32-bit register is refused and touches nothing. */
    mark(NPU_BASE + DAYAN_MMU_APB_BASE, saved, 16);
    assert(npu_hal_copy_to_address(hal, 0, DAYAN_MMU_APB_BASE, &value, 8) ==
           -EINVAL);
    assert(npu_hal_copy_to_address(hal, 0, DAYAN_MMU_APB_BASE + 2, &word, 4) ==
           -EINVAL);
    assert(npu_hal_copy_to_address(hal, 0, DAYAN_MMU_APB_BASE, &word, 2) ==
           -EINVAL);
    assert_unchanged(NPU_BASE + DAYAN_MMU_APB_BASE, saved, 16);
    mark(NPU_BASE + DAYAN_DEVICE_MAILBOX(0), saved, 16);
    assert(npu_hal_copy_to_address(hal, 0, DAYAN_DEVICE_MAILBOX(0), &word, 4) ==
           -EINVAL);
    assert(npu_hal_copy_to_address(hal, 0, DAYAN_DEVICE_MAILBOX(0) + 4,
                                   &value, 8) == -EINVAL);
    assert_unchanged(NPU_BASE + DAYAN_DEVICE_MAILBOX(0), saved, 16);
    /* Past the last Hart's MSIP, and offsets no module claims (BootROM). */
    mark(NPU_BASE + DAYAN_CLINT_MSIP(4), saved, 8);
    assert(npu_hal_copy_to_address(hal, 0, DAYAN_CLINT_MSIP(4), &word, 4) ==
           -EFAULT);
    assert_unchanged(NPU_BASE + DAYAN_CLINT_MSIP(4), saved, 8);
    assert(npu_hal_copy_from_address(hal, 0, &word,
                                     DAYAN_BOOTROM_RESET_VECTOR, 4) == -EFAULT);

    /* Range errors never leave half an access behind. */
    mark(DDR_HOST + DDR_SIZE - 4, saved, 4);
    assert(npu_hal_copy_to_address(hal, 0, DDR_NPU + DDR_SIZE - 4, &value,
                                   sizeof(value)) == -EFAULT);
    assert_unchanged(DDR_HOST + DDR_SIZE - 4, saved, 4);
    mark(NPU_BASE + DAYAN_URAM(0) + DAYAN_URAM_SIZE - 4, saved, 4);
    assert(npu_hal_copy_to_address(hal, 0,
                                   DAYAN_URAM(0) + DAYAN_URAM_SIZE - 4,
                                   &value, sizeof(value)) == -EFAULT);
    assert_unchanged(NPU_BASE + DAYAN_URAM(0) + DAYAN_URAM_SIZE - 4, saved, 4);
    mark(NPU_BASE + DAYAN_COHERENT_WINDOW_BASE - 4, saved, 8);
    assert(npu_hal_copy_to_address(hal, 0, DAYAN_COHERENT_WINDOW_BASE - 4,
                                   &value, sizeof(value)) == -EFAULT);
    assert_unchanged(NPU_BASE + DAYAN_COHERENT_WINDOW_BASE - 4, saved, 8);
    assert(npu_hal_copy_to_address(hal, 0, UINT64_MAX - 3, &value,
                                   sizeof(value)) == -EOVERFLOW);
    assert(npu_hal_copy_to_address(hal, 0, DDR_NPU - 4, &value,
                                   sizeof(value)) == -EFAULT);
    /* None of that is a fatal device error. */
    assert(npu_hal_device_status(hal) == 0);

    /* IRQ window: polled status, ack, the read-only ENABLE register. */
    write_u32(NPU_BASE + DAYAN_IRQ_STATUS, 0x4);
    assert(npu_hal_irq_status(hal, 0, &pending) == 0 && pending == 0x4);
    pending = 0;
    assert(npu_hal_irq_wait(hal, 100, &cluster, &pending) == 0);
    assert(cluster == 0 && pending == 0x4);
    assert(npu_hal_irq_ack(hal, 0, 0x4) == 0);
    assert(read_u32(NPU_BASE + DAYAN_IRQ_STATUS_CLR) == 0x4);
    write_u32(NPU_BASE + DAYAN_IRQ_STATUS, 0);
    assert(npu_hal_irq_wait(hal, 0, &cluster, &pending) == -ETIMEDOUT);
    write_u32(NPU_BASE + DAYAN_IRQ_ENABLE, 0xf);
    assert(npu_hal_register_read(hal, 0, DAYAN_IRQ_ENABLE - DAYAN_IRQ_STATUS,
                                 &word, 4) == 0 && word == 0xf);
    write_u64(NPU_BASE + DAYAN_IRQ_MAILBOX(1), UINT64_C(0x77));
    observed = 0;
    assert(npu_hal_irq_mailbox(hal, (npu_hal_core_t){ 0, 1 }, &observed) ==
           0 && observed == 0x77);

    /* Atomic-cache control is 64-bit and reached through register_*. */
    assert(npu_hal_register_write(
               hal, 0, DAYAN_ATOMIC_CACHE_BACKING_BASE - DAYAN_IRQ_STATUS,
               &value, sizeof(value)) == 0);
    assert(read_u64(NPU_BASE + DAYAN_ATOMIC_CACHE_BACKING_BASE) == value);

    /* PCU: latch, then single 64-bit reads per counter. */
    for (core = 0; core < 4; ++core)
        write_u64(NPU_BASE + DAYAN_PCU_CYCLE(core), 1000U + core);
    write_u64(NPU_BASE + DAYAN_PCU_COUNTER(2, 3), UINT64_C(0x123456789));
    assert(npu_hal_pcu_snapshot(hal, 0, &batch) == 0);
    assert(read_u64(NPU_BASE + DAYAN_PCU_CONTROL(4)) == 1);
    assert(batch.cores[3].cycle == 1003 &&
           batch.cores[2].counters[3] == UINT64_C(0x123456789));

    /* Reset is a placeholder until clock/reset are defined. */
    assert(npu_hal_cluster_reset(hal, 0) == -ENOTSUP);
    assert(npu_hal_cluster_reset(hal, 1) == -EINVAL);

    test_first_token(hal);
    test_copy_address(hal);

    /* Runtime moves a Normal Buffer through the translated DDR. */
    npu_hal_memory_request_init(&request);
    request.size = 0x1000;
    request.alignment = 0x1000;
    assert(npu_hal_memory_alloc(hal, &request, &buffer) == 0);
    assert(npu_hal_memory_address(hal, buffer, core2, &address) == 0);
    assert(address >= DDR_NPU && address + 0x1000 <= DDR_NPU + DDR_SIZE);
    assert(npu_hal_data_plane_export(hal, &exported, handles) == 0);
    assert(exported.provider == NPU_HAL_DATA_PROVIDER_SOC_MMAP &&
           exported.handle_count == 1 && handles[0] >= 0 && handles[1] < 0 &&
           exported.opaque[1] == NPU_HAL_SOC_MMAP_DDR_UNCACHED);
    {
        /* A Runtime refuses a coherence model it does not know. */
        npu_hal_data_export_t unknown = exported;
        npu_hal_data_plane_t *refused = 0;
        int spare[1] = { dup(handles[0]) };

        unknown.opaque[1] = 7;
        assert(npu_hal_data_plane_import(&refused, &unknown, spare, 1) ==
               -ENOTSUP && !refused);
        close(spare[0]);
    }
    assert(npu_hal_data_plane_import(&data_plane, &exported, handles,
                                     exported.handle_count) == 0);
    assert(handles[0] == -1);
    for (index = 0; index < sizeof(payload); ++index)
        payload[index] = (unsigned char)(index ^ 0x5aU);
    assert(npu_hal_transfer_prepare(hal, buffer, core2, 3, sizeof(payload),
                                    NPU_HAL_DATA_HOST_TO_DEVICE,
                                    &token) == 0);
    assert(token.opaque[0] == DDR_HOST && token.opaque[1] == DDR_SIZE);
    assert(npu_hal_data_plane_transfer(data_plane, &token, payload,
                                       sizeof(payload)) == 0);
    assert(npu_hal_transfer_finish(hal, &token, 0) == 0);
    memset(back, 0, sizeof(back));
    file_read(DDR_HOST + (address - DDR_NPU) + 3, back, sizeof(back));
    assert(memcmp(back, payload, sizeof(payload)) == 0);
    memset(back, 0, sizeof(back));
    assert(npu_hal_transfer_prepare(hal, buffer, core2, 3, sizeof(back),
                                    NPU_HAL_DATA_DEVICE_TO_HOST,
                                    &token) == 0);
    assert(npu_hal_data_plane_transfer(data_plane, &token, back,
                                       sizeof(back)) == 0);
    assert(npu_hal_transfer_finish(hal, &token, 0) == 0);
    assert(memcmp(back, payload, sizeof(payload)) == 0);
    /* Forged tokens: past the DDR, or naming another physical range. */
    token.opaque[2] = DDR_SIZE - 8;
    assert(npu_hal_data_plane_transfer(data_plane, &token, back,
                                       sizeof(back)) == -EACCES);
    token.opaque[2] = 0;
    token.opaque[0] = NPU_BASE;
    assert(npu_hal_data_plane_transfer(data_plane, &token, back,
                                       sizeof(back)) == -EACCES);
    assert(npu_hal_transfer_prepare(hal, buffer, core2, 0x1000 - 8, 16,
                                    NPU_HAL_DATA_HOST_TO_DEVICE,
                                    &token) == -EINVAL);

    /* Coherent and URAM Buffers are refused for Runtime transfers ... */
    npu_hal_memory_request_init(&request);
    request.memory_class = NPU_HAL_MEMORY_COHERENT;
    request.size = 0x100;
    assert(npu_hal_memory_alloc(hal, &request, &coherent) == 0);
    assert(npu_hal_transfer_prepare(hal, coherent, core2, 0, 0x100,
                                    NPU_HAL_DATA_HOST_TO_DEVICE,
                                    &token) == -ENOTSUP);
    /* ... while the Server still reaches them at npu_base + offset. */
    assert(npu_hal_memory_write(hal, coherent, core2, 0, &value,
                                sizeof(value)) == 0);
    assert(npu_hal_memory_address(hal, coherent, core2, &address) == 0);
    assert(read_u64(NPU_BASE + address) == value);
    npu_hal_memory_request_init(&request);
    request.memory_class = NPU_HAL_MEMORY_URAM;
    request.scope = NPU_HAL_MEMORY_CORE;
    request.usage = NPU_HAL_MEMORY_KERNEL_URAM;
    request.location = core2;
    request.size = 0x8000;
    request.alignment = 0x8000;
    assert(npu_hal_memory_alloc(hal, &request, &uram) == 0);
    assert(npu_hal_memory_address(hal, uram, core2, &address) == 0);
    assert(address >= DAYAN_URAM(2) &&
           address < DAYAN_URAM(2) + DAYAN_URAM_SIZE);
    assert(npu_hal_memory_write(hal, uram, core2, 0, &value,
                                sizeof(value)) == 0);
    assert(read_u64(NPU_BASE + address) == value);

    npu_hal_data_plane_close(data_plane);
    assert(npu_hal_memory_free(hal, uram) == 0);
    assert(npu_hal_memory_free(hal, coherent) == 0);
    assert(npu_hal_memory_free(hal, buffer) == 0);
    npu_hal_close(hal);
    close(memory_fd);
    unlink(path);
    puts("soc_mmap_test: layout, translation, region coverage, register rules, "
         "range prechecks, whole-copy checks, IRQ/PCU, first-token binding, data plane and reset placeholder OK");
    return 0;
}
