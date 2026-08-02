OUTPUT_ARCH(riscv)
ENTRY(_start)

__hart_count = DEFINED(CONFIG_HART_COUNT) ? CONFIG_HART_COUNT : 4;
__cpu_local_size = DEFINED(CONFIG_CPU_LOCAL_SIZE) ? CONFIG_CPU_LOCAL_SIZE : 0x1000;
__stack_size = DEFINED(CONFIG_STACK_SIZE) ? CONFIG_STACK_SIZE : 0x10000;
__heap_size = DEFINED(CONFIG_HEAP_SIZE) ? CONFIG_HEAP_SIZE : 0x100000;
__ram_base = DEFINED(CONFIG_RAM_BASE) ? CONFIG_RAM_BASE : 0x80000000;
__ram_size = DEFINED(CONFIG_RAM_SIZE) ? CONFIG_RAM_SIZE : 0x04000000;
__coherent_base = DEFINED(CONFIG_COHERENT_BASE) ? CONFIG_COHERENT_BASE : 0x90000000;
__coherent_size = DEFINED(CONFIG_COHERENT_SIZE) ? CONFIG_COHERENT_SIZE : 0x00400000;

MEMORY
{
    RAM (rwx)      : ORIGIN = __ram_base, LENGTH = __ram_size
    COHERENT (rw)  : ORIGIN = __coherent_base, LENGTH = __coherent_size
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

    .heap ALIGN(64) (NOLOAD) :
    {
        __heap_start = .;
        . += __hart_count * __heap_size;
        __heap_end = .;
    } > RAM
    __ram_used_end = .;

    .coherent ORIGIN(COHERENT) (NOLOAD) :
    {
        __coherent_start = .;
        KEEP(*(.coherent .coherent.*))
        __coherent_used_end = .;
        . = ORIGIN(COHERENT) + LENGTH(COHERENT);
        __coherent_end = .;
    } > COHERENT

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
    ASSERT((__heap_size % 16) == 0,
           "HEAP_SIZE must be a 16-byte multiple")
    ASSERT(SIZEOF(.data) == 0,
           "ordinary mutable .data is unsafe; keep state per hart or coherent")
    ASSERT(SIZEOF(.bss) == 0,
           "ordinary mutable .bss is unsafe; keep state per hart or coherent")
    ASSERT(SIZEOF(.init_array) == 0,
           "constructors are unsupported")
    ASSERT(__ram_used_end <= ORIGIN(RAM) + LENGTH(RAM),
           "image, per-hart stacks, and heaps exceed RAM")
    ASSERT(__coherent_used_end <= ORIGIN(COHERENT) + LENGTH(COHERENT),
           "coherent objects exceed the coherent window")
    ASSERT((ORIGIN(RAM) + LENGTH(RAM) <= ORIGIN(COHERENT)) ||
           (ORIGIN(COHERENT) + LENGTH(COHERENT) <= ORIGIN(RAM)),
           "RAM and coherent windows overlap")
}
