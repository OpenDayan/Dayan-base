#include <stddef.h>
#include <dayan_sys.h>

#include "platform.h"

void dayan_cache_invalidate(const void *address, size_t size)
{
    platform_cache_invalidate((void *)address, size);
}

void dayan_cache_flush(const void *address, size_t size)
{
    platform_cache_clean(address, size);
}
