#include <stddef.h>
#include <stdint.h>

#include "mem.h"
#include "platform.h"
#include "stdlib.h"
#include "string.h"

void *malloc(size_t size)
{
    return mem_alloc(size);
}

void *calloc(size_t count, size_t size)
{
    void *ptr;

    if (count && size > SIZE_MAX / count)
        return 0;
    size *= count;
    ptr = mem_alloc(size);
    if (ptr)
        memset(ptr, 0, size);
    return ptr;
}

void *realloc(void *ptr, size_t size)
{
    return mem_resize(ptr, size);
}

void free(void *ptr)
{
    mem_free(ptr);
}

_Noreturn void abort(void)
{
    platform_halt();
}

_Noreturn void exit(int status)
{
    (void)status;
    platform_halt();
}
