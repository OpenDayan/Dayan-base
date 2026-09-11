#include <assert.h>
#include <elf.h>
#include <errno.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "service_internal.h"
#include "rpc.h"

struct npu_hal_allocation {
    uint64_t address;
    size_t size;
    uint32_t usage;
    int freed;
    unsigned char bytes[16U * 1024U * 1024U];
};

static struct npu_hal_allocation arena;
static unsigned allocation_count, upload_count, rpc_count, entered_cores;
static unsigned finished_cores;
static uint64_t last_upload;
static int fail_rpc;
static pthread_mutex_t rpc_mutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t rpc_condition = PTHREAD_COND_INITIALIZER;
static npu_service_device_t device;
static struct test_object {
    Elf64_Ehdr header;
    Elf64_Shdr sections[5];
    Elf64_Sym symbols[2];
    char names[8];
    unsigned char text[65];
} object;

int npu_hal_core_valid(npu_hal_t *hal, npu_hal_core_t core)
{
    (void)hal;
    return core.cluster == 0 && core.core < 4;
}

void npu_hal_memory_request_init(npu_hal_memory_request_t *request)
{
    memset(request, 0, sizeof(*request));
}

int npu_hal_memory_alloc(npu_hal_t *hal, const npu_hal_memory_request_t *request,
                         npu_hal_allocation_t **allocation_out)
{
    (void)hal;
    assert(!device.clusters[0].booted_cores);
    assert(!allocation_count && request->usage == NPU_HAL_MEMORY_KERNEL);
    assert(request->memory_class == NPU_HAL_MEMORY_NORMAL);
    assert(request->size == sizeof(arena.bytes));
    assert(request->location.core == NPU_HAL_ANY_CORE);
    if (request->scope == NPU_HAL_MEMORY_CLUSTER)
        return -ENOMEM; /* Exercise the device-visible fallback. */
    assert(request->scope == NPU_HAL_MEMORY_DEVICE);
    assert(request->location.cluster == NPU_HAL_ANY_CLUSTER);
    ++allocation_count;
    arena.address = UINT64_C(0x80010040);
    arena.size = request->size;
    arena.usage = request->usage;
    memset(arena.bytes, 0xcc, sizeof(arena.bytes));
    *allocation_out = &arena;
    return 0;
}

int npu_hal_memory_address(npu_hal_t *hal,
                           const npu_hal_allocation_t *allocation,
                           npu_hal_core_t viewer, uint64_t *address_out)
{
    (void)hal;
    (void)viewer;
    *address_out = allocation->address;
    return 0;
}

/* The production helper additionally translates through the ClusterMMU.
 * This isolated kernel-cache test has no MMU instance, so model the MMU-off
 * path with the HAL's physical allocation address. */
int npu_service_mmu_device_address(npu_service_device_t *runtime,
                                   const npu_hal_allocation_t *allocation,
                                   npu_hal_core_t viewer, uint64_t *address_out)
{
    return npu_hal_memory_address(runtime ? runtime->hal : 0, allocation,
                                  viewer, address_out);
}

int npu_hal_memory_write(npu_hal_t *hal,
                         const npu_hal_allocation_t *allocation,
                         npu_hal_core_t viewer, uint64_t offset,
                         const void *source, size_t size)
{
    (void)hal;
    (void)viewer;
    assert(!allocation->freed && offset + size <= allocation->size);
    if (allocation->usage == NPU_HAL_MEMORY_KERNEL) {
        assert(entered_cores == 0 && finished_cores == 0);
        assert(size >= sizeof(object.text) && !(offset % 64));
        ++upload_count;
        last_upload = allocation->address + offset;
    }
    memcpy((void *)(allocation->bytes + offset), source, size);
    return 0;
}

int npu_hal_memory_read(npu_hal_t *hal,
                        const npu_hal_allocation_t *allocation,
                        npu_hal_core_t viewer, void *destination,
                        uint64_t offset, size_t size)
{
    (void)hal;
    (void)viewer;
    assert(!allocation->freed && offset + size <= allocation->size);
    memcpy(destination, allocation->bytes + offset, size);
    return 0;
}

int npu_hal_memory_free(npu_hal_t *hal, npu_hal_allocation_t *allocation)
{
    (void)hal;
    assert(!allocation->freed);
    allocation->freed = 1;
    return 0;
}

int npu_service_rpc_call(npu_service_device_t *runtime, npu_hal_core_t core,
                         uint16_t opcode, uint64_t argument, uint32_t timeout,
                         int *argument_may_be_live_out)
{
    struct timespec deadline;

    (void)runtime;
    (void)timeout;
    assert(opcode == RPC_FENCEI && !argument_may_be_live_out);
    assert(upload_count && argument == last_upload);
    assert(memcmp(arena.bytes + argument - arena.address, object.text,
                  sizeof(object.text)) == 0);
    assert(clock_gettime(CLOCK_REALTIME, &deadline) == 0);
    deadline.tv_sec += 5;
    pthread_mutex_lock(&rpc_mutex);
    ++rpc_count;
    entered_cores |= 1U << core.core;
    pthread_cond_broadcast(&rpc_condition);
    /* No response until every Hart has a request: a serial implementation
     * deterministically fails this test instead of merely running slower. */
    while (entered_cores != 0xf) {
        assert(pthread_cond_timedwait(&rpc_condition, &rpc_mutex,
                                     &deadline) == 0);
    }
    finished_cores |= 1U << core.core;
    pthread_mutex_unlock(&rpc_mutex);
    return fail_rpc && core.core == 1 ? -ETIMEDOUT : 0;
}

static void *resolve(void *argument)
{
    uint64_t *entry = argument;

    assert(npu_service_kernel_resolve(&device, (npu_hal_core_t){0, 0},
                                      &object, sizeof(object), "probe", entry) == 0);
    return 0;
}

int main(void)
{
    pthread_t threads[8];
    uint64_t entries[8];
    struct test_object copy;

    memcpy(object.header.e_ident, ELFMAG, SELFMAG);
    object.header.e_ident[EI_CLASS] = ELFCLASS64;
    object.header.e_ident[EI_DATA] = ELFDATA2LSB;
    object.header.e_ident[EI_VERSION] = EV_CURRENT;
    object.header.e_type = ET_REL;
    object.header.e_machine = EM_RISCV;
    object.header.e_version = EV_CURRENT;
    object.header.e_ehsize = sizeof(Elf64_Ehdr);
    object.header.e_shoff = offsetof(struct test_object, sections);
    object.header.e_shentsize = sizeof(Elf64_Shdr);
    object.header.e_shnum = 4;
    object.sections[1] = (Elf64_Shdr){
        .sh_type = SHT_PROGBITS, .sh_flags = SHF_ALLOC | SHF_EXECINSTR,
        .sh_offset = offsetof(struct test_object, text),
        .sh_size = sizeof(object.text), .sh_addralign = 4,
    };
    object.sections[2] = (Elf64_Shdr){
        .sh_type = SHT_SYMTAB, .sh_link = 3,
        .sh_offset = offsetof(struct test_object, symbols),
        .sh_size = sizeof(object.symbols), .sh_entsize = sizeof(Elf64_Sym),
    };
    object.sections[3] = (Elf64_Shdr){
        .sh_type = SHT_STRTAB, .sh_offset = offsetof(struct test_object, names),
        .sh_size = sizeof(object.names),
    };
    memcpy(object.names, "\0probe", 7);
    object.symbols[1] = (Elf64_Sym){
        .st_name = 1, .st_info = ELF64_ST_INFO(STB_GLOBAL, STT_FUNC),
        .st_shndx = 1, .st_size = sizeof(object.text),
    };
    memset(object.text, 0x5a, sizeof(object.text));
    assert(pthread_mutex_init(&device.mutex, 0) == 0);
    assert(pthread_mutex_init(&device.kernel_mutex, 0) == 0);
    device.clusters = calloc(1, sizeof(*device.clusters));
    assert(device.clusters);
    device.hal_info.cluster_count = 1;
    device.hal_info.cores_per_cluster = 4;
    device.api_symbol_count = 1;
    assert(npu_service_kernel_text_init(&device, 0) == 0);
    device.clusters[0].booted_cores = 0xf;

    for (unsigned i = 0; i < 8; ++i)
        assert(pthread_create(&threads[i], 0, resolve, &entries[i]) == 0);
    for (unsigned i = 0; i < 8; ++i) {
        assert(pthread_join(threads[i], 0) == 0);
        assert(entries[i] == arena.address);
    }
    assert(upload_count == 1 && rpc_count == 4 && finished_cores == 0xf);
    assert(memcmp(arena.bytes, object.text, sizeof(object.text)) == 0);
    copy = object;
    assert(npu_service_kernel_resolve(&device, (npu_hal_core_t){0, 3},
                                      &copy, sizeof(copy), "probe", &entries[0]) == 0);
    assert(npu_service_kernel_forget(&device, &object, sizeof(object)) == 0);
    assert(npu_service_kernel_forget(&device, &copy, sizeof(copy)) == 0);
    assert(!arena.freed && allocation_count == 1 && rpc_count == 4);

    /* Unload does not recycle addresses. A failed fence also consumes its
     * range and must join all other Harts before returning the failure. */
    entered_cores = finished_cores = 0;
    fail_rpc = 1;
    assert(npu_service_kernel_resolve(&device, (npu_hal_core_t){0, 0},
                                      &object, sizeof(object), "probe",
                                      &entries[0]) == -ETIMEDOUT);
    assert(last_upload == arena.address + 128 && finished_cores == 0xf);
    assert(!device.kernel_modules && !arena.freed);
    entered_cores = finished_cores = 0;
    fail_rpc = 0;
    object.sections[1].sh_addralign = 4096;
    object.header.e_shnum = 5;
    object.sections[4] = (Elf64_Shdr){.sh_type = SHT_NOBITS,
        .sh_flags = SHF_ALLOC | SHF_WRITE, .sh_size = 65, .sh_addralign = 64};
    resolve(&entries[0]);
    assert(entries[0] == UINT64_C(0x80011000));
    for (size_t i = 65; i < 256; ++i)
        assert(arena.bytes[entries[0] - arena.address + i] == 0);
    assert(npu_service_kernel_forget(&device, &object, sizeof(object)) == 0);
    device.clusters[0].kernel_text_next = arena.size;
    assert(npu_service_kernel_resolve(&device, (npu_hal_core_t){0, 0},
                                      &object, sizeof(object), "probe",
                                      &entries[0]) == -ENOMEM);
    assert(upload_count == 3 && rpc_count == 12 && allocation_count == 1);
    device.api_symbol_count = 0;
    npu_service_kernel_state_destroy(&device);
    assert(arena.freed);
    free(device.clusters);
    pthread_mutex_destroy(&device.kernel_mutex);
    pthread_mutex_destroy(&device.mutex);
    puts("kernel_cache_test: arena lifetime, parallel fencei, cache reuse, alignment and failure ordering OK");
    return 0;
}
