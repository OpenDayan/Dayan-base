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
    RPC_PROC_CREATE = 0x0100,
    RPC_PROC_CREATE_RESPONSE = 0x8100,
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

struct rpc_print_request {
    uint32_t hart_id;
    uint32_t proc_id;
    uint64_t data_addr;
    uint64_t data_size;
};

struct rpc_heap_request {
    uint64_t size;
};

struct rpc_heap_response {
    uint64_t addr;
    uint64_t size;
};

struct rpc_heap_release {
    uint64_t addr;
    uint64_t size;
};

#define RPC_PROC_CREATE_VERSION       1U
#define RPC_PROC_MAX_ARGS             8U
#define RPC_PROC_MIN_STACK_SIZE       1024U
#define RPC_PROC_MAX_STACK_SIZE       (16U * 1024U * 1024U)
#define RPC_PROC_MAX_PRINTF_SIZE      (1024U * 1024U)

enum rpc_proc_state {
    RPC_PROC_SUBMITTED = 0,
    RPC_PROC_RUNNING = 1,
    RPC_PROC_EXITED = 2,
    RPC_PROC_FAILED = 3,
};

/*
 * Host-owned create descriptor. Host keeps this storage valid until state is
 * EXITED or FAILED. BL1 updates status/state and cleans the corresponding
 * cache lines before replying or completing the proc.
 */
struct rpc_proc_create_request {
    uint32_t version;
    uint32_t size;
    uint32_t state;
    uint32_t flags;
    uint64_t entry;
    uint64_t stack_size;
    uint64_t shared_base;
    uint64_t shared_size;
    uint64_t printf_buffer_size;
    uint32_t argc;
    uint32_t reserved;
    uint64_t args[RPC_PROC_MAX_ARGS];
    int64_t status;
    uint32_t proc_id;
    uint32_t hart_id;
    uint64_t cookie;
};

_Static_assert(sizeof(struct rpc_print_request) == 24,
               "Host print request ABI must stay fixed");
_Static_assert(offsetof(struct rpc_print_request, proc_id) == 4,
               "Host print proc ID offset");
_Static_assert(offsetof(struct rpc_print_request, data_addr) == 8,
               "Host print data offset");
_Static_assert(offsetof(struct rpc_print_request, data_size) == 16,
               "Host print size offset");
_Static_assert(sizeof(struct rpc_heap_request) == 8,
               "Host heap request ABI must stay fixed");
_Static_assert(sizeof(struct rpc_heap_response) == 16,
               "Host heap response ABI must stay fixed");
_Static_assert(offsetof(struct rpc_heap_response, size) == 8,
               "Host heap response size offset");
_Static_assert(sizeof(struct rpc_heap_release) == 16,
               "Host heap release ABI must stay fixed");
_Static_assert(offsetof(struct rpc_proc_create_request, entry) == 16,
               "Host proc entry offset");
_Static_assert(offsetof(struct rpc_proc_create_request, args) == 64,
               "Host proc argument offset");
_Static_assert(offsetof(struct rpc_proc_create_request, status) == 128,
               "Host proc status offset");
_Static_assert(sizeof(struct rpc_proc_create_request) == 152,
               "Host proc create ABI must stay fixed");

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
long rpc_heap_acquire(size_t size, struct rpc_heap_response *response);
long rpc_heap_release(uintptr_t addr, size_t size);

#endif
