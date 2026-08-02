#define MAILBOX_BASE 0x1000
#define CLINT_BASE 0x20000

// boot all cores (only hart 0) and jump to main program execution
.section .text.start, "ax", @progbits
.globl _start
_start:
  csrr a0, mhartid
  li a1, MAILBOX_BASE // base address of mailbox
  li a2, CLINT_BASE   // base address of clint
  sll t0, a0,  3      // get 8B boot-address-reg offset
  sll t1, a0,  2      // get clint offset
  add a1, t0, a1      // get real mailbox
  add a2, t1, a2      // get real clint
#if __riscv_xlen == 32
  lw a3, 0(a1)        // get boot address from mailbox register0
#else
  ld a3, 0(a1)        // get boot address from mailbox register0
#endif
  // Consuming the mailbox before clearing MSIP makes the clear an
  // acknowledgement. The host may reuse the mailbox only after observing it.
  fence iorw, iorw
  sw zero, 0(a2)      // acknowledge the interrupt
  fence iorw, iorw
  csrw mepc, a3       // return from interrupt to start of user program
  li a2, 0x80         // set mstatus MPIE to 0
  csrc mstatus, a2
  mret

// this boot ROM doesn't know about any boot devices, so it just spins,
// waiting for the serial interface to load the program and interrupt it
.section .text.hang, "ax", @progbits
.globl _hang
_hang: // reset vector
  la a0, _start          // on MSIP interrupt, go to _start
  csrw mtvec, a0
  csrr t0, misa          // get MISA mask
  srai t0, t0, 'S' - 'A' // get 'S' supervisor bit
  andi t0, t0, 0x1
  beqz t0, 1f            // only zero mideleg if supported
  csrw mideleg, zero     // no delegation
  li a0, 0x222           // clear spurious seip/stip/ssip left over after reset
  csrc mip, a0
1:
  li a0, 8               // MIE or MSIP bit
  csrw mie, a0           // set only MSIP in MIE CSR
  csrs mstatus, a0       // set MIE in mstatus CSR

wfi_loop: // wait for MSIP interrupt to start program execution
  wfi
  j wfi_loop

// jump to reset vector
.section .text.hang80, "ax", @progbits
.globl _hang80
_hang80:
  j _hang

.align 3
_dtb:
