#include <assert.h>
#include <elf.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "npu_package.h"
#include "package.h"

static void make_object(unsigned char *object, size_t size,
                        uint64_t section_flags, uint32_t section_type)
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
    section.sh_flags = section_flags;
    section.sh_type = section_type;
    memcpy(object, &header, sizeof(header));
    memcpy(object + sizeof(header), &section, sizeof(section));
}

int main(void)
{
    unsigned char object[sizeof(Elf64_Ehdr) + sizeof(Elf64_Shdr)];
    npu_package_argument_t arguments[2];
    npu_package_kernel_desc_t kernel;
    struct npu_package_view view;
    void *package = 0;
    size_t package_size = 0;

    make_object(object, sizeof(object), 0, SHT_NULL);
    memset(arguments, 0, sizeof(arguments));
    arguments[0].kind = NPU_PACKAGE_ARG_BUFFER;
    arguments[0].size = 8;
    arguments[0].alignment = 8;
    arguments[0].access = NPU_PACKAGE_ARG_READ | NPU_PACKAGE_ARG_WRITE;
    arguments[1].kind = NPU_PACKAGE_ARG_VALUE;
    arguments[1].size = 4;
    memset(&kernel, 0, sizeof(kernel));
    kernel.name = "grid_kernel";
    kernel.entry_symbol = "grid_kernel";
    kernel.supported_task_modes = NPU_PACKAGE_TASK_MODE_ALL;
    kernel.max_tasks_per_core = 2;
    kernel.max_child_procs = 3;
    kernel.atomic_cache_size = 4096;
    kernel.uram_size = 2048;
    kernel.arguments = arguments;
    kernel.argument_count = 2;
    assert(npu_package_build(object, sizeof(object), &kernel, 1,
                             NPU_PACKAGE_IMMUTABLE_MODULE, &package,
                             &package_size) == 0);
    assert(npu_package_view_init(&view, package, package_size) == 0);
    assert(view.kernels[0].supported_task_modes ==
           NPU_PACKAGE_TASK_MODE_ALL);
    assert(view.kernels[0].max_tasks_per_core == 2);
    assert(view.kernels[0].max_child_procs == 3);
    assert(view.kernels[0].atomic_cache_size == 4096);
    assert(view.kernels[0].uram_size == 2048);
    assert(strcmp(npu_package_string(&view, view.kernels[0].name_offset),
                  "grid_kernel") == 0);
    npu_package_free(package);

    package = 0;
    package_size = 0;
    kernel.max_child_procs = NPU_PACKAGE_MAX_CHILD_PROCS + 1U;
    assert(npu_package_build(object, sizeof(object), &kernel, 1, 0,
                             &package, &package_size) < 0);
    assert(!package && !package_size);
    kernel.max_child_procs = 3;
    kernel.uram_size = NPU_PACKAGE_URAM_GRANULE + 1U;
    assert(npu_package_build(object, sizeof(object), &kernel, 1, 0,
                             &package, &package_size) < 0);
    assert(!package && !package_size);
    kernel.uram_size = 2048;

    make_object(object, sizeof(object), SHF_ALLOC | SHF_WRITE, SHT_PROGBITS);
    assert(npu_package_build(object, sizeof(object), &kernel, 1,
                             NPU_PACKAGE_IMMUTABLE_MODULE, &package,
                             &package_size) == 0);
    assert(npu_package_view_init(&view, package, package_size) < 0);
    npu_package_free(package);
    return 0;
}
