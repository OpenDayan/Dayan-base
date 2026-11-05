#define _POSIX_C_SOURCE 200809L

#include "hal_internal.h"
#include "npu_hal_soc_mmap.h"
#include "platform_regs.h"
#include "layout.h"
#include "io.h"

#include <errno.h>
#include <fcntl.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

struct soc_state {
    int fd;
    /* Separate descriptor for Runtime, or -1 to hand out fd. */
    int data_fd;
    uint32_t coherence;
    uint64_t ddr_host_base;
    uint64_t ddr_device_base;
    uint64_t ddr_size;
    struct npu_hal_soc_mmap_mapping aperture;
    struct npu_hal_soc_mmap_mapping ddr;
    uint32_t entry_count;
    struct soc_layout_entry entries[SOC_LAYOUT_MAX_ENTRIES];
};

/* Where one complete HAL access lands; resolved before anything is touched. */
struct soc_target {
    volatile unsigned char *pointer;
    uint32_t kind;
    uint32_t access_width;
    int ddr;
    uint64_t ddr_offset;
};

static void soc_close(void *opaque)
{
    struct soc_state *s = opaque;

    if (!s)
        return;
    npu_hal_soc_mmap_unmap(&s->ddr);
    npu_hal_soc_mmap_unmap(&s->aperture);
    if (s->data_fd >= 0)
        close(s->data_fd);
    if (s->fd >= 0)
        close(s->fd);
    free(s);
}

static int ranges_overlap(uint64_t a, uint64_t a_size, uint64_t b,
                          uint64_t b_size)
{
    return a < b + b_size && b < a + a_size;
}

static int soc_options_valid(const npu_hal_soc_mmap_options_t *o,
                             uint64_t page)
{
    if (!o->memory_device || !*o->memory_device ||
        (o->data_device && !*o->data_device) ||
        (o->ddr_coherence != NPU_HAL_SOC_MMAP_DDR_UNCACHED &&
         o->ddr_coherence != NPU_HAL_SOC_MMAP_DDR_COHERENT) ||
        !o->npu_base || !o->ddr_host_base || !o->ddr_size)
        return 0;
    /* Page alignment keeps every register naturally aligned in the Host
     * mapping and lets the DDR map without a partial page. */
    if ((o->npu_base | o->ddr_host_base | o->ddr_size) & (page - 1U) ||
        (o->ddr_device_base & (DAYAN_MMU_PAGE_SIZE - 1U)))
        return 0;
    if (o->npu_base > UINT64_MAX - DAYAN_CLUSTER_APERTURE_SIZE ||
        o->ddr_host_base > UINT64_MAX - o->ddr_size ||
        o->ddr_device_base > UINT64_MAX - o->ddr_size)
        return 0;
    /* NPU side: DDR cannot alias the IP aperture. */
    if (o->ddr_device_base < DAYAN_CLUSTER_APERTURE_SIZE)
        return 0;
    /* Host side: the two mappings must be distinct physical ranges. */
    return !ranges_overlap(o->npu_base, DAYAN_CLUSTER_APERTURE_SIZE,
                           o->ddr_host_base, o->ddr_size);
}

static const struct soc_layout_entry *find_entry(const struct soc_state *s,
                                                 uint64_t address)
{
    uint32_t i;

    for (i = 0; i < s->entry_count; ++i) {
        const struct soc_layout_entry *e = &s->entries[i];

        if (address >= e->offset && address - e->offset < e->size)
            return e;
    }
    return 0;
}

/*
 * The whole access must fall in one place: the shared DDR, one memory module,
 * or one register.  Anything else fails here, before any device access, so a
 * bad range never leaves half a write behind.  (That covers software range
 * errors only: a bus error on a real chip can still interrupt an access.)
 */
static int soc_resolve(const struct soc_state *s, uint32_t cluster,
                       uint64_t address, uint64_t size,
                       struct soc_target *target)
{
    const struct soc_layout_entry *e;
    uint64_t offset;

    if (cluster != 0 || !size)
        return -EINVAL;
    if (address > UINT64_MAX - size)
        return -EOVERFLOW;
    memset(target, 0, sizeof(*target));
    if (address >= s->ddr_device_base &&
        address - s->ddr_device_base < s->ddr_size) {
        offset = address - s->ddr_device_base;
        if (size > s->ddr_size - offset)
            return -EFAULT;
        target->pointer = s->ddr.window + offset;
        target->kind = SOC_LAYOUT_MEMORY;
        target->ddr = 1;
        target->ddr_offset = offset;
        return 0;
    }
    e = find_entry(s, address);
    if (!e)
        return -EFAULT;
    offset = address - e->offset;
    if (size > e->size - offset)
        return -EFAULT;
    if (e->kind == SOC_LAYOUT_REGISTER &&
        (size != e->access_width || (offset & (e->access_width - 1U))))
        return -EINVAL;
    target->pointer = s->aperture.window + address;
    target->kind = e->kind;
    target->access_width = e->access_width;
    return 0;
}

/* A HAL region is allocatable only where its whole range is the memory the
 * layout says it is, and visible to exactly the Cores the region names. */
static int region_covered(const struct soc_state *s,
                          const npu_hal_memory_region_t *r)
{
    const struct soc_layout_entry *e;

    if (r->memory_class == NPU_HAL_MEMORY_NORMAL)
        return r->scope == NPU_HAL_MEMORY_DEVICE &&
               r->base >= s->ddr_device_base &&
               r->base - s->ddr_device_base < s->ddr_size &&
               r->size <= s->ddr_size - (r->base - s->ddr_device_base);
    e = find_entry(s, r->base);
    if (!e || e->kind != SOC_LAYOUT_MEMORY ||
        e->memory_class != r->memory_class ||
        r->size > e->size - (r->base - e->offset))
        return 0;
    if (r->memory_class == NPU_HAL_MEMORY_URAM)
        return r->scope == NPU_HAL_MEMORY_CORE && r->cluster == 0 &&
               r->core == e->core;
    return r->scope == NPU_HAL_MEMORY_DEVICE ||
           (r->scope == NPU_HAL_MEMORY_CLUSTER && r->cluster == 0);
}

/*
 * Placeholder for platform bring-up.  With PLATFORM_EXTERNAL the bootloader,
 * BSP or a platform driver has already powered the Cluster, released
 * isolation, configured and locked its clocks and released reset, so there is
 * nothing to do.  Once the chip integration defines who owns that sequence and
 * how to confirm it, the status checks or the sequence itself go here.
 */
static int soc_platform_prepare(const npu_hal_soc_mmap_options_t *o)
{
    return o->platform_control == NPU_HAL_SOC_MMAP_PLATFORM_EXTERNAL ?
        0 : -ENOTSUP;
}

static int soc_build_id(void *opaque, uint32_t cluster, char *id, size_t size)
{
    struct soc_state *s = opaque;
    (void)cluster;
    npu_hal_soc_mmap_io_fence();
    npu_hal_soc_mmap_mem_read(id,
        s->aperture.window + DAYAN_BOOTROM_BUILD_ID_ADDRESS, size);
    npu_hal_soc_mmap_io_fence();
    return 0;
}

static int soc_open(npu_hal_options_t *options, void **out)
{
    const npu_hal_soc_mmap_options_t *o;
    long page = sysconf(_SC_PAGESIZE);
    struct soc_state *s;
    uint32_t i;
    int status;

    if (!options || !out || !options->backend_options || page <= 0)
        return -EINVAL;
    *out = 0;
    if (options->backend_options->backend != NPU_HAL_BACKEND_SOC_MMAP ||
        options->backend_options->version != NPU_HAL_SOC_MMAP_OPTIONS_VERSION)
        return -EINVAL;
    o = (const void *)options->backend_options;
    /* The platform defines no address relation between Clusters yet. */
    if (options->cluster_count != 1)
        return -ENOTSUP;
    if (!soc_options_valid(o, (uint64_t)page))
        return -EINVAL;
    s = calloc(1, sizeof(*s));
    if (!s)
        return -ENOMEM;
    s->fd = -1;
    s->data_fd = -1;
    s->coherence = o->ddr_coherence;
    s->ddr_host_base = o->ddr_host_base;
    s->ddr_device_base = o->ddr_device_base;
    s->ddr_size = o->ddr_size;
    status = soc_platform_prepare(o);
    if (status < 0)
        goto fail;
    s->fd = open(o->memory_device, O_RDWR | O_SYNC | O_CLOEXEC);
    if (s->fd < 0) {
        status = -errno;
        goto fail;
    }
    if (o->data_device) {
        s->data_fd = open(o->data_device, O_RDWR | O_SYNC | O_CLOEXEC);
        if (s->data_fd < 0) {
            status = -errno;
            goto fail;
        }
    }
    status = npu_hal_soc_mmap_map(s->fd, o->npu_base,
                                  DAYAN_CLUSTER_APERTURE_SIZE, &s->aperture);
    if (status == 0)
        status = npu_hal_soc_mmap_map(s->fd, o->ddr_host_base, o->ddr_size,
                                      &s->ddr);
    if (status < 0)
        goto fail;
    status = npu_hal_detect_config(options, soc_build_id, s);
    if (status < 0)
        goto fail;
    status = soc_layout_build(options->cores_per_cluster, options->uram_size,
                              s->entries, SOC_LAYOUT_MAX_ENTRIES);
    if (status < 0)
        goto fail;
    s->entry_count = (uint32_t)status;
    for (i = 0; i < options->memory_region_count; ++i) {
        if (!region_covered(s, &options->memory_regions[i])) {
            status = -EINVAL;
            goto fail;
        }
    }
    *out = s;
    return 0;

fail:
    soc_close(s);
    return status;
}

static int soc_info(void *opaque, npu_hal_info_t *info)
{
    if (!opaque || !info)
        return -EINVAL;
    info->data_provider = NPU_HAL_DATA_PROVIDER_SOC_MMAP;
    info->data_handle_count = 1;
    info->max_transfer_size = INT64_MAX;
    info->transfer_alignment = 1;
    return 0;
}

/*
 * Registers are fenced on both sides so a doorbell cannot overtake the data
 * it publishes and a status read cannot be satisfied before earlier writes.
 * DDR writes are handed to the device after the copy, DDR reads taken back
 * before it; module memory (Coherent window, URAM) is device memory inside
 * the aperture and needs only the fences.
 */
static int soc_copy(void *opaque, uint32_t cluster, uint64_t address,
                    void *data, size_t size, int write)
{
    struct soc_state *s = opaque;
    struct soc_target t;
    int status;

    if (!s || (!data && size))
        return -EINVAL;
    status = soc_resolve(s, cluster, address, size, &t);
    if (status < 0)
        return status;
    npu_hal_soc_mmap_io_fence();
    if (t.kind == SOC_LAYOUT_REGISTER) {
        status = write ? npu_hal_soc_mmap_reg_write(t.pointer, data,
                                                    t.access_width) :
                         npu_hal_soc_mmap_reg_read(t.pointer, data,
                                                   t.access_width);
    } else if (write) {
        npu_hal_soc_mmap_mem_write(t.pointer, data, size);
        if (t.ddr)
            status = npu_hal_soc_mmap_ddr_sync_for_device(s->coherence,
                                                          t.pointer, size);
    } else {
        if (t.ddr)
            status = npu_hal_soc_mmap_ddr_sync_for_cpu(s->coherence,
                                                       t.pointer, size);
        if (status == 0)
            npu_hal_soc_mmap_mem_read(data, t.pointer, size);
    }
    npu_hal_soc_mmap_io_fence();
    return status;
}

/*
 * Device-to-Device copies move memory, never registers: both ranges must each
 * lie wholly in the shared DDR, the Coherent window or one URAM.  A register
 * range is refused with -EINVAL instead of being bulk-copied with memory
 * accesses, which would break its width rule.
 */
static int soc_copy_check(void *opaque, uint32_t cluster,
                          uint64_t destination, uint64_t source, size_t size)
{
    struct soc_state *s = opaque;
    struct soc_target t;
    int status;

    if (!s)
        return -EINVAL;
    status = soc_resolve(s, cluster, destination, size, &t);
    if (status == 0 && t.kind != SOC_LAYOUT_MEMORY)
        status = -EINVAL;
    if (status == 0)
        status = soc_resolve(s, cluster, source, size, &t);
    if (status == 0 && t.kind != SOC_LAYOUT_MEMORY)
        status = -EINVAL;
    return status;
}

static int soc_write(void *s, uint32_t c, uint64_t a, const void *v, size_t n)
{
    return soc_copy(s, c, a, (void *)v, n, 1);
}

static int soc_read(void *s, uint32_t c, void *v, uint64_t a, size_t n)
{
    return soc_copy(s, c, a, v, n, 0);
}

/* register_* offsets are relative to DAYAN_IRQ_STATUS, as on every backend. */
static int soc_reg_read(void *s, uint32_t c, uint64_t a, void *v, size_t n)
{
    if (a > UINT64_MAX - DAYAN_IRQ_STATUS)
        return -EOVERFLOW;
    return soc_read(s, c, v, DAYAN_IRQ_STATUS + a, n);
}

static int soc_reg_write(void *s, uint32_t c, uint64_t a, const void *v,
                         size_t n)
{
    if (a > UINT64_MAX - DAYAN_IRQ_STATUS)
        return -EOVERFLOW;
    return soc_write(s, c, DAYAN_IRQ_STATUS + a, v, n);
}

/*
 * Placeholder: no reset sequence exists until the chip integration defines
 * the reset controls, their ordering against the clocks, and how to confirm
 * that every Hart and AXI master has stopped.  Until then a Cluster cannot be
 * reset and memory it may still reach must not be reused.
 */
static int soc_reset_cluster(void *opaque, uint32_t cluster)
{
    if (!opaque || cluster != 0)
        return -EINVAL;
    return -ENOTSUP;
}

static int soc_export(void *opaque, npu_hal_data_export_t *e,
                      int h[NPU_HAL_DATA_MAX_HANDLES])
{
    struct soc_state *s = opaque;

    h[0] = fcntl(s->data_fd >= 0 ? s->data_fd : s->fd, F_DUPFD_CLOEXEC, 0);
    if (h[0] < 0)
        return -errno;
    memset(e, 0, sizeof(*e));
    e->version = NPU_HAL_DATA_EXPORT_VERSION;
    e->provider = NPU_HAL_DATA_PROVIDER_SOC_MMAP;
    e->handle_count = 1;
    e->opaque[0] = NPU_HAL_SOC_MMAP_TOKEN_LAYOUT;
    e->opaque[1] = s->coherence;
    e->opaque[2] = s->ddr_host_base;
    e->opaque[3] = s->ddr_size;
    return 0;
}

/*
 * Runtime transfers reach shared DDR only.  The Coherent window and URAM sit
 * in the IP aperture, whose Host access path for bulk data (and, for the
 * atomic-cache window, its semantics against the backing DDR) is unconfirmed,
 * so Buffers there are refused rather than routed somewhere unverified.
 */
static int soc_prepare(void *opaque, uint32_t c, uint64_t a, size_t n,
                       npu_hal_data_direction_t d, npu_hal_data_token_t *t)
{
    struct soc_state *s = opaque;
    struct soc_target target;
    int status;

    if (!s || !t)
        return -EINVAL;
    status = soc_resolve(s, c, a, n, &target);
    if (status < 0)
        return status;
    if (!target.ddr)
        return -ENOTSUP;
    memset(t, 0, sizeof(*t));
    t->version = NPU_HAL_DATA_TOKEN_VERSION;
    t->provider = NPU_HAL_DATA_PROVIDER_SOC_MMAP;
    t->direction = d;
    t->size = n;
    t->opaque[0] = s->ddr_host_base;
    t->opaque[1] = s->ddr_size;
    t->opaque[2] = target.ddr_offset;
    return 0;
}

static int soc_finish(void *opaque, const npu_hal_data_token_t *t, int status)
{
    if (!opaque || !t || t->version != NPU_HAL_DATA_TOKEN_VERSION ||
        t->provider != NPU_HAL_DATA_PROVIDER_SOC_MMAP)
        return -EINVAL;
    return status;
}

const struct npu_hal_backend_ops npu_hal_soc_mmap_backend_ops = {
    .backend = NPU_HAL_BACKEND_SOC_MMAP,
    .open = soc_open,
    .close = soc_close,
    .get_info = soc_info,
    .copy_to_address = soc_write,
    .copy_from_address = soc_read,
    .register_read = soc_reg_read,
    .register_write = soc_reg_write,
    /* No Host IRQ yet: the HAL polls IRQ status.  UIO can come later. */
    .irq_wait = 0,
    .export_data_plane = soc_export,
    .prepare_transfer = soc_prepare,
    .finish_transfer = soc_finish,
    .copy_address_check = soc_copy_check,
    .reset_cluster = soc_reset_cluster,
};
