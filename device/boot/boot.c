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
}

void boot_ready(void)
{
    if (!test_present())
        platform_rpc_ready();
}
