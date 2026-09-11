OUTPUT_ARCH(riscv)
ENTRY(_start)
EXTERN(_start)

__hart_count = DEFINED(CONFIG_HART_COUNT) ? CONFIG_HART_COUNT : 4;
__cpu_local_size = DEFINED(CONFIG_CPU_LOCAL_SIZE) ? CONFIG_CPU_LOCAL_SIZE : 0x2000;
__stack_size = DEFINED(CONFIG_STACK_SIZE) ? CONFIG_STACK_SIZE : 0x10000;
__proc_slot_count = DEFINED(CONFIG_PROC_SLOT_COUNT) ? CONFIG_PROC_SLOT_COUNT : 16;
__proc_stack_size = DEFINED(CONFIG_PROC_STACK_SIZE) ? CONFIG_PROC_STACK_SIZE : 0x10000;
__proc_printf_size = DEFINED(CONFIG_PROC_PRINTF_SIZE) ? CONFIG_PROC_PRINTF_SIZE : 0x1000;
__rpc_stack_size = DEFINED(CONFIG_RPC_STACK_SIZE) ? CONFIG_RPC_STACK_SIZE : 0x1000;
__profile_stack_size = DEFINED(CONFIG_PROFILE_STACK_SIZE) ? CONFIG_PROFILE_STACK_SIZE : 0x1000;
__trap_stack_size = DEFINED(CONFIG_TRAP_STACK_SIZE) ? CONFIG_TRAP_STACK_SIZE : 0x1000;
__heap_size = DEFINED(CONFIG_HEAP_SIZE) ? CONFIG_HEAP_SIZE : 0x100000;
__test_data_start = 0x84000000;
__test_data_end = 0xc4000000;
__test_mailbox_start = __test_data_start - 64;
__test_mailbox_end = __test_data_start;
__test_log_start = __test_mailbox_start - 64 - 4096;
__test_log_end = __test_mailbox_start;

MEMORY
{
    RAM (rwx) : ORIGIN = DEFINED(CONFIG_RAM_BASE) ? CONFIG_RAM_BASE : 0x80000000,
                LENGTH = DEFINED(CONFIG_RAM_SIZE) ? CONFIG_RAM_SIZE : 0x04000000
}

/* Reserved CPU-local storage is writable even without any .data/.bss inputs. */
PHDRS
{
    code PT_LOAD FLAGS(5);
    constants PT_LOAD FLAGS(4);
    local PT_LOAD FLAGS(6);
}

SECTIONS
{
    . = ORIGIN(RAM);

    /* Match before named sections, so a writable .rodata.* cannot escape.
     * .cpu and backing pools reserve storage without mutable input objects. */
    .unexpected_writable (NOLOAD) :
    {
        INPUT_SECTION_FLAGS(SHF_WRITE) *(*)
    } > RAM

    .text : ALIGN(16)
    {
        KEEP(*(.text.bl1))
        *(.text)
    } > RAM :code

    .text.api.libc : ALIGN(16)
    {
        KEEP(*(.text.api.libc))
    } > RAM

    .text.api.dapi : ALIGN(16)
    {
        KEEP(*(.text.api.dapi))
    } > RAM

    .text.runtime : ALIGN(16)
    {
        *(.text.*)
    } > RAM

    .rodata : ALIGN(16)
    {
        *(.rodata .rodata.* .srodata .srodata.*)
    } > RAM :constants

    .npu_test : ALIGN(8)
    {
        __npu_test_start = .;
        KEEP(*(.npu_test))
        __npu_test_end = .;
    } > RAM

    .data : ALIGN(16)
    {
        __data_start = .;
        *(.data .data.* .sdata .sdata.* .got .got.*)
        __data_end = .;
    } > RAM :local

    .bss (NOLOAD) : ALIGN(16)
    {
        __bss_start = .;
        *(.bss .bss.* .sbss .sbss.* COMMON)
        __bss_end = .;
    } > RAM

    .init_array : ALIGN(16)
    {
        *(.init_array .init_array.*)
    } > RAM

    __image_end = ALIGN(64);

    .cpu ALIGN(64) (NOLOAD) :
    {
        __cpu_start = .;
        . += __hart_count * __cpu_local_size;
        __cpu_end = .;
    } > RAM

    .stack ALIGN(64) (NOLOAD) :
    {
        __stack_start = .;
        . += __hart_count * __stack_size;
        __stack_end = .;
    } > RAM

    .rpc_stack ALIGN(64) (NOLOAD) :
    {
        __rpc_stack_start = .;
        . += __hart_count * 2 * __rpc_stack_size;
        __rpc_stack_end = .;
    } > RAM

    .profile_stack ALIGN(64) (NOLOAD) :
    {
        __profile_stack_start = .;
        . += __hart_count * __profile_stack_size;
        __profile_stack_end = .;
    } > RAM

    .trap_stack ALIGN(64) (NOLOAD) :
    {
        __trap_stack_start = .;
        . += __hart_count * __trap_stack_size;
        __trap_stack_end = .;
    } > RAM

    .proc_stack ALIGN(64) (NOLOAD) :
    {
        __proc_stack_start = .;
        . += __hart_count * __proc_slot_count * __proc_stack_size;
        __proc_stack_end = .;
    } > RAM

    .proc_printf ALIGN(64) (NOLOAD) :
    {
        __proc_printf_start = .;
        . += __hart_count * __proc_slot_count * __proc_printf_size;
        __proc_printf_end = .;
    } > RAM

    .heap ALIGN(64) (NOLOAD) :
    {
        __heap_start = .;
        . += __hart_count * __heap_size;
        __heap_end = .;
    } > RAM
    .test_backing ALIGN(4096) (NOLOAD) :
    {
        __test_backing_start = .;
        . += SIZEOF(.npu_test) ? 0x400000 : 0;
    } > RAM
    __ram_used_end = .;
    __test_heap_start = ALIGN(64);
    __test_heap_end = MIN(ORIGIN(RAM) + LENGTH(RAM),
                          SIZEOF(.npu_test) ? __test_log_start : __test_data_start);

    /* Coherent storage is Host-owned and must never enter the firmware ELF. */
    .coherent (NOLOAD) :
    {
        *(.coherent .coherent.*)
    } > RAM

    .tdata :
    {
        *(.tdata .tdata.*)
    } > RAM

    .tbss (NOLOAD) :
    {
        *(.tbss .tbss.*)
    } > RAM

    /DISCARD/ :
    {
        *(.eh_frame .eh_frame.*)
        *(.comment)
        *(.note .note.*)
    }

    ASSERT(__hart_count > 0, "at least one hart is required")
    ASSERT(__cpu_local_size >= 64 &&
           (__cpu_local_size & (__cpu_local_size - 1)) == 0,
           "CPU_LOCAL_SIZE must be a power of two and cover whole cache lines")
    ASSERT((__stack_size % 64) == 0,
           "STACK_SIZE must cover whole cache lines")
    ASSERT(__proc_slot_count == 16,
           "the Hart proc-slot ABI requires exactly 16 slots")
    ASSERT((__proc_stack_size % 64) == 0,
           "PROC_STACK_SIZE must cover whole cache lines")
    ASSERT(__proc_printf_size > 0 && (__proc_printf_size % 64) == 0,
           "PROC_PRINTF_SIZE must be nonzero and cover whole cache lines")
    ASSERT((__rpc_stack_size % 64) == 0,
           "RPC_STACK_SIZE must cover whole cache lines")
    ASSERT((__profile_stack_size % 64) == 0,
           "PROFILE_STACK_SIZE must cover whole cache lines")
    ASSERT((__trap_stack_size % 64) == 0,
           "TRAP_STACK_SIZE must cover whole cache lines")
    ASSERT((__heap_size % 64) == 0,
           "HEAP_SIZE must cover whole cache lines")
    ASSERT(SIZEOF(.data) == 0,
           "mutable .data is forbidden; keep state in struct cpu")
    ASSERT(SIZEOF(.bss) == 0,
           "mutable .bss is forbidden; keep state in struct cpu")
    ASSERT(SIZEOF(.unexpected_writable) == 0,
           "mutable input sections are forbidden; keep state in struct cpu")
    ASSERT(SIZEOF(.init_array) == 0,
           "constructors are unsupported")
    ASSERT(SIZEOF(.proc_stack) ==
           __hart_count * __proc_slot_count * __proc_stack_size,
           "per-Hart proc stack reservation has the wrong size")
    ASSERT(SIZEOF(.proc_printf) ==
           __hart_count * __proc_slot_count * __proc_printf_size,
           "per-Hart proc printf reservation has the wrong size")
    ASSERT(__ram_used_end <= ORIGIN(RAM) + LENGTH(RAM),
           "image, per-hart stacks, and heaps exceed RAM")
    ASSERT(!SIZEOF(.npu_test) ||
           (ORIGIN(RAM) == 0x80000000 && __ram_used_end < __test_heap_end),
           "bare-metal image must start at 0x80000000 and leave heap below test log")
    ASSERT((SIZEOF(.npu_test) % 320) == 0,
           "test descriptor table must contain 320-byte records")
    ASSERT(!DEFINED(CONFIG_REQUIRE_TESTS) || SIZEOF(.npu_test),
           "device test must register at least one DAYAN_TEST descriptor")
    ASSERT(SIZEOF(.coherent) == 0,
           "coherent storage is allocated by Host, not the firmware linker")
    ASSERT(SIZEOF(.tdata) == 0 && SIZEOF(.tbss) == 0,
           "tp is current and cannot address ELF TLS")
}
