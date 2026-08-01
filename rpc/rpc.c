#include <stddef.h>
#include <stdint.h>

#include "errno.h"
#include "platform.h"
#include "rpc.h"

long rpc_call(uint32_t service,
              const void *request, size_t request_size,
              void *response, size_t response_size)
{
    long status;

    if ((!request && request_size) || (!response && response_size))
        return -EINVAL;

    if (request_size)
        platform_cache_clean(request, request_size);
    if (response_size)
        platform_cache_clean(response, response_size);

    status = platform_rpc(service, request, request_size,
                          response, response_size);

    if (status >= 0 && response_size)
        platform_cache_invalidate(response, response_size);
    return status;
}

long rpc_write(const void *data, size_t size)
{
    if (!size)
        return 0;
    return rpc_call(RPC_WRITE, data, size, 0, 0);
}

long rpc_clear(void)
{
    return rpc_call(RPC_CLEAR, 0, 0, 0, 0);
}
