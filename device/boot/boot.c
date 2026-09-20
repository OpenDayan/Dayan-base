#include <stddef.h>
#include <stdint.h>

#include "boot.h"
#include "config.h"
#include "mem.h"
#include "platform.h"
#include "profile.h"
#include "rpc.h"
#include "sche.h"
#include "test.h"

#define __ASM_STR(x)  #x

#define csr_read(csr)                                       \
({                                                          \
    register unsigned long __v;                             \
    __asm__ __volatile__ ("csrr %0, " __ASM_STR(csr)        \
        : "=r" (__v)                                        \
        :                                                   \
        : "memory");                                        \
    __v;                                                    \
})

#define csr_write(csr, val)                                 \
({                                                          \
    unsigned long __v = (unsigned long)(val);               \
    __asm__ __volatile__ ("csrw " __ASM_STR(csr) ", %0"     \
        :                                                   \
        : "rK" (__v)                                        \
        : "memory");                                        \
})

extern unsigned char __heap_start[];

void init(void)
{
    uintptr_t hart = platform_hart_id();
    void *heap = __heap_start + hart * CONFIG_HEAP_SIZE;

    platform_init();
    sche_init();
    if (mem_init(heap, CONFIG_HEAP_SIZE) < 0)
        platform_halt();
    if (test_present())
        test_init();
    else
        rpc_init();
    profile_init();

    // Set 38-bits full va-range for rts arch
    if(csr_read(0xF11) == 0 && csr_read(0xF12) == 0) {
        csr_write(0x7D6, 0x3FFFFFFFFFUL);
    }
}

void boot_ready(void)
{
    if (!test_present())
        platform_rpc_ready();
}
