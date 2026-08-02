#ifndef NPU_RPC_H
#define NPU_RPC_H

#include <stddef.h>
#include <stdint.h>

enum rpc_service {
    RPC_WRITE = 1,
    RPC_CLEAR,
    RPC_HEAP_ACQUIRE,
    RPC_HEAP_RELEASE,
    RPC_TRAP_REPORT,
};

#define RPC_COMMAND_ARG_MASK 0x0000ffffffffffffULL

static inline uint64_t rpc_command_pack(uint16_t opcode, uint64_t arg)
{
    return ((uint64_t)opcode << 48) | (arg & RPC_COMMAND_ARG_MASK);
}

static inline uint16_t rpc_command_opcode(uint64_t command)
{
    return (uint16_t)(command >> 48);
}

static inline uint64_t rpc_command_arg(uint64_t command)
{
    return command & RPC_COMMAND_ARG_MASK;
}

/* Descriptor addressed by the 48-bit command argument for Device requests. */
struct rpc_packet {
    uint64_t request_addr;
    uint64_t request_size;
    uint64_t response_addr;
    uint64_t response_size;
    int64_t status;
};

_Static_assert(offsetof(struct rpc_packet, request_addr) == 0,
               "RPC packet request address offset");
_Static_assert(offsetof(struct rpc_packet, response_addr) == 16,
               "RPC packet response address offset");
_Static_assert(offsetof(struct rpc_packet, status) == 32,
               "RPC packet status offset");
_Static_assert(sizeof(struct rpc_packet) == 40,
               "Host RPC packet ABI must stay fixed");

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

int rpc_init(void);

/*
 * Host-to-Device dispatcher. Return 1 and fill response_command to reply,
 * return 0 for no reply, or a negative error after consuming the request.
 */
int rpc_receive(uint16_t opcode, uint64_t arg,
                uint64_t *response_command);

long rpc_call(uint32_t service,
              const void *request, size_t request_size,
              void *response, size_t response_size);
long rpc_write(const void *data, size_t size);
long rpc_clear(void);

#endif
