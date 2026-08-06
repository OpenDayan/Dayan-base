#ifndef NPU_RPC_H
#define NPU_RPC_H

#include <stddef.h>
#include <stdint.h>

struct proc;

enum rpc_service {
    RPC_WRITE = 1,
    RPC_CLEAR,
    RPC_HEAP_ACQUIRE,
    RPC_HEAP_RELEASE,
    RPC_TRAP_REPORT,
    RPC_PROC_EXIT,
    RPC_PROFILE_FLUSH,
    RPC_PROC_CREATE = 0x0100,
    RPC_ICACHE_FLUSH = 0x0101,
    RPC_PROC_PS = 0x0102,
    RPC_PROC_KILL_ALL = 0x0103,
    RPC_PROFILE_CONFIG = 0x0104,
    RPC_DCACHE_FLUSH = 0x0105,
    RPC_PROC_CREATE_RESPONSE = 0x8100,
    RPC_ICACHE_FLUSH_RESPONSE = 0x8101,
    RPC_PROC_PS_RESPONSE = 0x8102,
    RPC_PROC_KILL_ALL_RESPONSE = 0x8103,
    RPC_PROFILE_CONFIG_RESPONSE = 0x8104,
    RPC_DCACHE_FLUSH_RESPONSE = 0x8105,
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

#define RPC_DCACHE_FLUSH_VERSION 1U

/* Host-owned descriptor for invalidating a range in one Hart's D-cache. */
struct rpc_dcache_flush_request {
    uint32_t version;
    uint32_t size;
    uint32_t hart_id;
    uint32_t flags;
    uint64_t address;
    uint64_t length;
    int64_t status;
    uint64_t reserved;
};

_Static_assert(offsetof(struct rpc_packet, request_addr) == 0,
               "RPC packet request address offset");
_Static_assert(offsetof(struct rpc_packet, response_addr) == 16,
               "RPC packet response address offset");
_Static_assert(offsetof(struct rpc_packet, status) == 32,
               "RPC packet status offset");
_Static_assert(sizeof(struct rpc_packet) == 40,
               "Host RPC packet ABI must stay fixed");
_Static_assert(offsetof(struct rpc_dcache_flush_request, address) == 16,
               "D-cache flush address offset");
_Static_assert(offsetof(struct rpc_dcache_flush_request, status) == 32,
               "D-cache flush status offset");
_Static_assert(sizeof(struct rpc_dcache_flush_request) == 48,
               "D-cache flush request ABI must stay fixed");

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

#define RPC_PROC_EXIT_VERSION 1U
#define RPC_PROC_EXIT_ROOT    (1U << 0)

struct rpc_proc_exit_request {
    uint32_t version;
    uint32_t size;
    uint32_t hart_id;
    uint32_t proc_id;
    uint32_t parent_proc_id;
    uint32_t flags;
    int64_t result;
    uint64_t cookie;
};

#define RPC_PROC_CREATE_VERSION       4U
#define RPC_PROC_MAX_ARGS             8U
#define RPC_PROC_SLOTS_PER_HART       16U
#define RPC_PROC_NAME_SIZE            16U
#define RPC_PROC_MIN_STACK_SIZE       1024U
#define RPC_PROC_MAX_STACK_SIZE       0x10000U
#define RPC_PROC_MAX_PRINTF_SIZE      0x1000U
#define RPC_PROC_MAX_CHILD_PROCS      (RPC_PROC_SLOTS_PER_HART - 1U)
#define RPC_PROC_URAM_GRANULE         1024U
#define RPC_PROC_PS_VERSION           3U
#define RPC_PROC_KILL_ALL_VERSION     1U
#define RPC_PROC_PS_SYSTEM_PROCS      3U
#define RPC_PROC_PS_MAX_RECORDS       \
    (RPC_PROC_SLOTS_PER_HART + RPC_PROC_PS_SYSTEM_PROCS)

#define RPC_PROC_INFO_SYSTEM            (1U << 0)
#define RPC_PROC_INFO_HOST_ROOT         (1U << 1)
#define RPC_PROC_INFO_CHILD             (1U << 2)
#define RPC_PROC_INFO_ZOMBIE            (1U << 3)
#define RPC_PROC_INFO_KILL_WAIT_PRODUCE (1U << 4)

enum rpc_proc_state {
    RPC_PROC_SUBMITTED = 0,
    RPC_PROC_RUNNING = 1,
    RPC_PROC_EXITED = 2,
    RPC_PROC_FAILED = 3,
};

/*
 * Host-owned create descriptor. Host keeps this storage valid and leaves all
 * request fields unchanged until state is EXITED or FAILED. BL1 updates only
 * result fields and state, then cleans the corresponding cache lines before
 * replying or completing the proc.
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
    uint32_t task_id[3];
    uint32_t task_dim[3];
    uint32_t task_mode;
    uint32_t task_reserved;
    uint32_t max_child_procs;
    uint32_t resource_reserved;
    uint64_t uram_base;
    uint64_t uram_size;
    char name[RPC_PROC_NAME_SIZE];
};

struct rpc_proc_info {
    uint32_t proc_id;
    uint32_t parent_proc_id;
    uint32_t state;
    uint32_t flags;
    uint32_t live_children;
    uint32_t slot_index;
    uint32_t task_mode;
    uint32_t host_proc_id;
    uint32_t task_id[3];
    uint32_t task_dim[3];
    int64_t result;
    uint64_t pc;
    uint64_t cookie;
    char name[RPC_PROC_NAME_SIZE];
};

struct rpc_proc_ps_snapshot {
    uint32_t version;
    uint32_t size;
    uint32_t hart_id;
    uint32_t count;
    uint32_t capacity;
    uint32_t flags;
    uint32_t global_free_slots;
    uint32_t reserved_slots;
    int64_t status;
    uint64_t reserved;
    struct rpc_proc_info records[RPC_PROC_PS_MAX_RECORDS];
};

struct rpc_proc_kill_result {
    uint32_t version;
    uint32_t size;
    uint32_t hart_id;
    uint32_t flags;
    int64_t status;
    uint32_t killed_roots;
    uint32_t killed_children;
    uint32_t released_slots;
    uint32_t waiting_produce;
    uint64_t reserved;
};

#define RPC_PROFILE_VERSION 3U

enum rpc_profile_proc_state {
    RPC_PROFILE_PROC_RUNNING = 0,
    RPC_PROFILE_PROC_READY = 1,
    RPC_PROFILE_PROC_WAIT_POLL = 2,
    RPC_PROFILE_PROC_WAIT_EVENT = 3,
    RPC_PROFILE_PROC_WAIT_START = 4,
    RPC_PROFILE_PROC_WAIT_EXIT = 5,
    RPC_PROFILE_PROC_WAIT_PRODUCE = 6,
    RPC_PROFILE_PROC_KILL_WAIT_PRODUCE = 7,
};

enum rpc_profile_command {
    RPC_PROFILE_START = 1,
    RPC_PROFILE_STOP = 2,
};

struct rpc_profile_event {
    uint32_t cycle;
    uint16_t event_id;
    uint16_t state;
    char name[RPC_PROC_NAME_SIZE];
};

/* Host-owned descriptor used by RPC_PROFILE_CONFIG. */
struct rpc_profile_config {
    uint32_t version;
    uint32_t size;
    uint32_t command;
    uint32_t flags;
    uint32_t hart_id;
    uint32_t capacity;
    uint64_t buffer_addr;
    uint64_t generation;
    uint32_t count;
    uint32_t enabled;
    int64_t status;
    uint64_t reserved;
};

struct rpc_profile_flush_request {
    uint32_t version;
    uint32_t size;
    uint32_t hart_id;
    uint32_t flags;
    uint64_t buffer_addr;
    uint32_t count;
    uint32_t capacity;
    uint64_t generation;
};

struct rpc_profile_flush_response {
    uint32_t version;
    uint32_t size;
    uint32_t continue_profile;
    uint32_t flags;
    int64_t status;
    uint64_t generation;
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
_Static_assert(offsetof(struct rpc_proc_exit_request, result) == 24,
               "Host proc exit result offset");
_Static_assert(offsetof(struct rpc_proc_exit_request, cookie) == 32,
               "Host proc exit cookie offset");
_Static_assert(sizeof(struct rpc_proc_exit_request) == 40,
               "Host proc exit ABI must stay fixed");
_Static_assert(offsetof(struct rpc_proc_create_request, entry) == 16,
               "Host proc entry offset");
_Static_assert(offsetof(struct rpc_proc_create_request, args) == 64,
               "Host proc argument offset");
_Static_assert(offsetof(struct rpc_proc_create_request, status) == 128,
               "Host proc status offset");
_Static_assert(offsetof(struct rpc_proc_create_request, task_id) == 152,
               "Host proc task ID offset");
_Static_assert(offsetof(struct rpc_proc_create_request, task_dim) == 164,
               "Host proc task dimension offset");
_Static_assert(offsetof(struct rpc_proc_create_request, task_mode) == 176,
               "Host proc task mode offset");
_Static_assert(offsetof(struct rpc_proc_create_request, max_child_procs) == 184,
               "Host proc child resource offset");
_Static_assert(offsetof(struct rpc_proc_create_request, uram_base) == 192,
               "Host proc URAM base offset");
_Static_assert(offsetof(struct rpc_proc_create_request, name) == 208,
               "Host proc name offset");
_Static_assert(sizeof(struct rpc_proc_create_request) == 224,
               "Host proc create ABI must stay fixed");
_Static_assert(offsetof(struct rpc_proc_info, result) == 56,
               "Host proc info result offset");
_Static_assert(offsetof(struct rpc_proc_info, cookie) == 72,
               "Host proc info cookie offset");
_Static_assert(offsetof(struct rpc_proc_info, name) == 80,
               "Host proc info name offset");
_Static_assert(sizeof(struct rpc_proc_info) == 96,
               "Host proc info ABI must stay fixed");
_Static_assert(offsetof(struct rpc_proc_ps_snapshot, records) == 48,
               "Host proc snapshot record offset");
_Static_assert(sizeof(struct rpc_proc_ps_snapshot) == 1872,
               "Host proc snapshot ABI must stay fixed");
_Static_assert(offsetof(struct rpc_proc_kill_result, status) == 16,
               "Host proc kill result status offset");
_Static_assert(sizeof(struct rpc_proc_kill_result) == 48,
               "Host proc kill result ABI must stay fixed");
_Static_assert(sizeof(struct rpc_profile_event) == 24,
               "proc profile event ABI must stay fixed");
_Static_assert(offsetof(struct rpc_profile_event, event_id) == 4,
               "proc profile event ID offset");
_Static_assert(offsetof(struct rpc_profile_event, state) == 6,
               "proc profile event state offset");
_Static_assert(offsetof(struct rpc_profile_event, name) == 8,
               "proc profile event name offset");
_Static_assert(sizeof(struct rpc_profile_config) == 64,
               "proc profile config ABI must stay fixed");
_Static_assert(sizeof(struct rpc_profile_flush_request) == 40,
               "proc profile flush request ABI must stay fixed");
_Static_assert(sizeof(struct rpc_profile_flush_response) == 32,
               "proc profile flush response ABI must stay fixed");

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
long rpc_write_proc(uint32_t proc_id, const void *data, size_t size);
long rpc_clear_proc(uint32_t proc_id);
long rpc_heap_acquire(size_t size, struct rpc_heap_response *response);
long rpc_heap_release(uintptr_t addr, size_t size);
long rpc_proc_exit(uint32_t proc_id, uint32_t parent_proc_id,
                   int64_t result, uint64_t cookie, uint32_t flags);
long rpc_profile_flush(const struct rpc_profile_flush_request *request,
                       struct rpc_profile_flush_response *response);
/* Returns 1 while the system send proc still owns the transaction. */
int rpc_proc_wait_cancel(struct proc *proc);

#endif
