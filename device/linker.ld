OUTPUT_ARCH(riscv)
ENTRY(_start)

__hart_count = DEFINED(CONFIG_HART_COUNT) ? CONFIG_HART_COUNT : 4;
__cpu_local_size = DEFINED(CONFIG_CPU_LOCAL_SIZE) ? CONFIG_CPU_LOCAL_SIZE : 0x1000;
__stack_size = DEFINED(CONFIG_STACK_SIZE) ? CONFIG_STACK_SIZE : 0x10000;
__proc_slot_count = DEFINED(CONFIG_PROC_SLOT_COUNT) ? CONFIG_PROC_SLOT_COUNT : 16;
__proc_stack_size = DEFINED(CONFIG_PROC_STACK_SIZE) ? CONFIG_PROC_STACK_SIZE : 0x10000;
__proc_printf_size = DEFINED(CONFIG_PROC_PRINTF_SIZE) ? CONFIG_PROC_PRINTF_SIZE : 0x1000;
__rpc_stack_size = DEFINED(CONFIG_RPC_STACK_SIZE) ? CONFIG_RPC_STACK_SIZE : 0x1000;
__profile_stack_size = DEFINED(CONFIG_PROFILE_STACK_SIZE) ? CONFIG_PROFILE_STACK_SIZE : 0x1000;
__trap_stack_size = DEFINED(CONFIG_TRAP_STACK_SIZE) ? CONFIG_TRAP_STACK_SIZE : 0x1000;
__heap_size = DEFINED(CONFIG_HEAP_SIZE) ? CONFIG_HEAP_SIZE : 0x100000;

MEMORY
{
    RAM (rwx) : ORIGIN = DEFINED(CONFIG_RAM_BASE) ? CONFIG_RAM_BASE : 0x80000000,
                LENGTH = DEFINED(CONFIG_RAM_SIZE) ? CONFIG_RAM_SIZE : 0x04000000
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

    .profile_state ALIGN(64) (NOLOAD) :
    {
        __profile_state_start = .;
        *(.profile_state)
        __profile_state_end = .;
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

    .trap_records ALIGN(64) (NOLOAD) :
    {
        *(.trap_records)
    } > RAM

    .tensor_interrupt_records ALIGN(64) (NOLOAD) :
    {
        *(.tensor_interrupt_records)
    } > RAM

    .proc_slots ALIGN(64) (NOLOAD) :
    {
        __proc_slots_start = .;
        *(.proc_slots)
        __proc_slots_end = .;
    } > RAM

    .proc_stack ALIGN(64) (NOLOAD) :
    {
        __proc_stack_start = .;
        *(.proc_stack)
        __proc_stack_end = .;
    } > RAM

    .proc_printf ALIGN(64) (NOLOAD) :
    {
        __proc_printf_start = .;
        *(.proc_printf)
        __proc_printf_end = .;
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
    ASSERT(__proc_slot_count == 16,
           "the Hart proc-slot ABI requires exactly 16 slots")
    ASSERT((__proc_stack_size % 16) == 0,
           "PROC_STACK_SIZE must be a 16-byte multiple")
    ASSERT(__proc_printf_size > 0,
           "PROC_PRINTF_SIZE must be nonzero")
    ASSERT((__rpc_stack_size % 16) == 0,
           "RPC_STACK_SIZE must be a 16-byte multiple")
    ASSERT((__profile_stack_size % 16) == 0,
           "PROFILE_STACK_SIZE must be a 16-byte multiple")
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
    ASSERT(SIZEOF(.proc_slots) > 0,
           "per-Hart proc slot metadata is missing")
    ASSERT(SIZEOF(.profile_state) > 0,
           "per-Hart profile state is missing")
    ASSERT(SIZEOF(.proc_stack) ==
           __hart_count * __proc_slot_count * __proc_stack_size,
           "per-Hart proc stack reservation has the wrong size")
    ASSERT(SIZEOF(.proc_printf) ==
           __hart_count * __proc_slot_count * __proc_printf_size,
           "per-Hart proc printf reservation has the wrong size")
    ASSERT(__ram_used_end <= ORIGIN(RAM) + LENGTH(RAM),
           "image, per-hart stacks, and heaps exceed RAM")
    ASSERT(SIZEOF(.coherent) == 0,
           "coherent storage is allocated by Host, not the firmware linker")
    ASSERT(SIZEOF(.tdata) == 0 && SIZEOF(.tbss) == 0,
           "tp is current and cannot address ELF TLS")
}
