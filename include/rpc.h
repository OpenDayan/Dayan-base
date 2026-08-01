#ifndef NPU_RPC_H
#define NPU_RPC_H

#include <stddef.h>
#include <stdint.h>

enum rpc_service {
    RPC_WRITE = 1,
    RPC_CLEAR,
    RPC_HEAP_ACQUIRE,
    RPC_HEAP_RELEASE,
};

struct rpc_heap_request {
    size_t size;
};

struct rpc_heap_response {
    uintptr_t addr;
    size_t size;
};

struct rpc_heap_release {
    uintptr_t addr;
    size_t size;
};

long rpc_call(uint32_t service,
              const void *request, size_t request_size,
              void *response, size_t response_size);
long rpc_write(const void *data, size_t size);
long rpc_clear(void);

#endif
