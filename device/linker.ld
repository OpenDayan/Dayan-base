OUTPUT_ARCH(riscv)
ENTRY(_start)

__hart_count = DEFINED(CONFIG_HART_COUNT) ? CONFIG_HART_COUNT : 4;
__cpu_local_size = DEFINED(CONFIG_CPU_LOCAL_SIZE) ? CONFIG_CPU_LOCAL_SIZE : 0x1000;
__stack_size = DEFINED(CONFIG_STACK_SIZE) ? CONFIG_STACK_SIZE : 0x10000;
__rpc_stack_size = DEFINED(CONFIG_RPC_STACK_SIZE) ? CONFIG_RPC_STACK_SIZE : 0x1000;
__trap_stack_size = DEFINED(CONFIG_TRAP_STACK_SIZE) ? CONFIG_TRAP_STACK_SIZE : 0x1000;
__heap_size = DEFINED(CONFIG_HEAP_SIZE) ? CONFIG_HEAP_SIZE : 0x100000;
__ram_base = DEFINED(CONFIG_RAM_BASE) ? CONFIG_RAM_BASE : 0x80000000;
__ram_size = DEFINED(CONFIG_RAM_SIZE) ? CONFIG_RAM_SIZE : 0x04000000;

MEMORY
{
    RAM (rwx) : ORIGIN = __ram_base, LENGTH = __ram_size
}

SECTIONS
{
    . = ORIGIN(RAM);

    .text : ALIGN(16)
    {
        KEEP(*(.text.bl1))
        *(.text)
    } > RAM

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
    } > RAM

    .data : ALIGN(16)
    {
        __data_start = .;
        *(.data .data.* .sdata .sdata.* .got .got.*)
        __data_end = .;
    } > RAM

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

    .trap_stack ALIGN(64) (NOLOAD) :
    {
        __trap_stack_start = .;
        . += __hart_count * __trap_stack_size;
        __trap_stack_end = .;
    } > RAM

    .heap ALIGN(64) (NOLOAD) :
    {
        __heap_start = .;
        . += __hart_count * __heap_size;
        __heap_end = .;
    } > RAM
    __ram_used_end = .;

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
    ASSERT((__cpu_local_size % 8) == 0,
           "CPU_LOCAL_SIZE must be an 8-byte multiple")
    ASSERT((__stack_size % 16) == 0,
           "STACK_SIZE must be a 16-byte multiple")
    ASSERT((__rpc_stack_size % 16) == 0,
           "RPC_STACK_SIZE must be a 16-byte multiple")
    ASSERT((__trap_stack_size % 16) == 0,
           "TRAP_STACK_SIZE must be a 16-byte multiple")
    ASSERT((__heap_size % 16) == 0,
           "HEAP_SIZE must be a 16-byte multiple")
    ASSERT(SIZEOF(.data) == 0,
           "ordinary mutable .data is unsafe; keep state per hart")
    ASSERT(SIZEOF(.bss) == 0,
           "ordinary mutable .bss is unsafe; keep state per hart")
    ASSERT(SIZEOF(.init_array) == 0,
           "constructors are unsupported")
    ASSERT(__ram_used_end <= ORIGIN(RAM) + LENGTH(RAM),
           "image, per-hart stacks, and heaps exceed RAM")
    ASSERT(SIZEOF(.coherent) == 0,
           "coherent storage is allocated by Host, not the firmware linker")
    ASSERT(SIZEOF(.tdata) == 0 && SIZEOF(.tbss) == 0,
           "tp is current and cannot address ELF TLS")
}
