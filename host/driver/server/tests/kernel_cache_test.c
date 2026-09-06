#include <assert.h>
#include <elf.h>
#include <errno.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "service_internal.h"
#include "rpc.h"

struct npu_hal_allocation {
    uint64_t address;
    size_t size;
    uint32_t usage;
    int freed;
    unsigned char bytes[256];
};

static struct npu_hal_allocation allocations[16];
static unsigned allocation_count, upload_count, invalidated_cores;
static int fail_rpc, rpc_may_be_live;
static npu_service_device_t device;
static struct test_object {
    Elf64_Ehdr header;
    Elf64_Shdr sections[4];
    Elf64_Sym symbols[2];
    char names[8];
    unsigned char text[65];
} object;

int npu_hal_core_valid(npu_hal_t *hal, npu_hal_core_t core)
{
    (void)hal;
    return core.cluster == 0 && core.core < 4;
}

int npu_service_alloc_visible(npu_service_device_t *runtime,
                              npu_hal_core_t viewer,
                              npu_hal_memory_class_t memory_class,
                              npu_hal_memory_usage_t usage, uint32_t owner,
                              uint64_t size, uint64_t alignment,
                              npu_hal_allocation_t **allocation_out)
{
    struct npu_hal_allocation *allocation;

    (void)runtime;
    (void)viewer;
    (void)owner;
    assert(memory_class == NPU_HAL_MEMORY_NORMAL);
    assert(size <= 256 && !(size % RPC_DESCRIPTOR_GRANULE));
    assert(alignment >= RPC_DESCRIPTOR_GRANULE);
    assert(allocation_count < 16);
    allocation = &allocations[allocation_count++];
    allocation->address = UINT64_C(0x80010000) + allocation_count * 0x1000;
    allocation->size = size;
    allocation->usage = usage;
    *allocation_out = allocation;
    return 0;
}

int npu_service_alloc_rpc_descriptor(npu_service_device_t *runtime,
                                     npu_hal_core_t viewer, size_t size,
                                     npu_hal_allocation_t **allocation_out)
{
    assert(size <= RPC_DESCRIPTOR_GRANULE);
    return npu_service_alloc_visible(runtime, viewer, NPU_HAL_MEMORY_NORMAL,
                                     NPU_HAL_MEMORY_LAUNCH, 0,
                                     RPC_DESCRIPTOR_GRANULE,
                                     RPC_DESCRIPTOR_GRANULE, allocation_out);
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
        /* Upload must follow completion on every Hart, including the tail line. */
        assert(invalidated_cores == 0xf);
        assert(size == 128 && !offset);
        ++upload_count;
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
    struct npu_hal_allocation *descriptor = &allocations[allocation_count - 1];
    struct rpc_dcache_flush_request request;

    (void)runtime;
    (void)timeout;
    assert(opcode == RPC_DCACHE_FLUSH);
    assert(!upload_count && argument == descriptor->address);
    memcpy(&request, descriptor->bytes, sizeof(request));
    assert(request.hart_id == core.core);
    assert(request.address == allocations[0].address && request.length == 128);
    if (fail_rpc && core.core == 1) {
        *argument_may_be_live_out = rpc_may_be_live;
        return -ETIMEDOUT;
    }
    invalidated_cores |= 1U << core.core;
    request.status = 0;
    memcpy(descriptor->bytes, &request, sizeof(request));
    *argument_may_be_live_out = 0;
    return 0;
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
    device.clusters[0].booted_cores = 0xf;
    device.hal_info.cores_per_cluster = 4;
    device.api_symbol_count = 1;

    for (unsigned i = 0; i < 8; ++i)
        assert(pthread_create(&threads[i], 0, resolve, &entries[i]) == 0);
    for (unsigned i = 0; i < 8; ++i) {
        assert(pthread_join(threads[i], 0) == 0);
        assert(entries[i] == allocations[0].address);
    }
    assert(upload_count == 1);
    assert(memcmp(allocations[0].bytes, object.text, sizeof(object.text)) == 0);
    assert(npu_service_kernel_forget(&device, &object, sizeof(object)) == 0);

    for (rpc_may_be_live = 0; rpc_may_be_live <= 1; ++rpc_may_be_live) {
        memset(allocations, 0, sizeof(allocations));
        allocation_count = upload_count = invalidated_cores = 0;
        fail_rpc = 1;
        assert(npu_service_kernel_resolve(&device, (npu_hal_core_t){0, 0},
                                          &object, sizeof(object), "probe",
                                          &entries[0]) == -ETIMEDOUT);
        assert(!upload_count && !device.kernel_modules);
        assert(allocations[0].freed == !rpc_may_be_live);
        assert(allocations[allocation_count - 1].freed == !rpc_may_be_live);
    }
    free(device.clusters);
    pthread_mutex_destroy(&device.kernel_mutex);
    pthread_mutex_destroy(&device.mutex);
    puts("kernel_cache_test: publication ordering, isolation, concurrent resolve and timeout ownership OK");
    return 0;
}
