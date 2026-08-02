#ifndef NPU_DRIVER_SERVER_INTERNAL_H
#define NPU_DRIVER_SERVER_INTERNAL_H

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif

#include <pthread.h>
#include <stddef.h>
#include <stdint.h>
#include <sys/types.h>

#include "npu_runtime.h"
#include "package.h"
#include "protocol.h"
#include "server.h"

struct server_session;
struct server_context;
struct server_buffer;
struct server_stream;
struct server_module;
struct server_kernel;
struct server_event;
struct server_command;
struct server_launch_group;
struct server_invocation;
struct server_output;

enum server_command_kind {
    SERVER_COMMAND_EVENT = 1,
    SERVER_COMMAND_WAIT,
    SERVER_COMMAND_H2D,
    SERVER_COMMAND_D2H,
    SERVER_COMMAND_D2D,
    SERVER_COMMAND_LAUNCH,
};

struct server_core_slot {
    npu_hal_core_t target;
    uint32_t enabled;
    uint32_t active_tasks;
    uint64_t resident_memory;
    uint32_t reserved_procs;
    uint64_t atomic_cache_memory;
    uint64_t atomic_cache_capacity;
    uint64_t uram_memory;
    uint64_t uram_capacity;
};

struct server_package_cache {
    unsigned char *image;
    size_t size;
    uint64_t hash;
    uint32_t references;
    uint32_t immutable;
    struct npu_package_view view;
    struct server_package_cache *next;
};

struct server_buffer {
    uint64_t id;
    uint64_t size;
    uint32_t flags;
    uint32_t references;
    npu_hal_allocation_t *allocation;
    struct server_context *context;
    struct server_buffer *next;
};

struct server_event {
    uint64_t id;
    uint64_t sequence;
    uint64_t cookie;
    uint32_t state;
    uint32_t command_type;
    int64_t status;
    uint32_t visible;
    uint32_t references;
    struct server_command *command;
    struct server_context *context;
    struct server_event *next;
};

struct server_stream {
    uint64_t id;
    int32_t priority;
    uint32_t flags;
    uint32_t references;
    int64_t last_status;
    struct server_context *context;
    struct server_command *head;
    struct server_command *tail;
    struct server_stream *next;
};

struct server_module {
    uint64_t id;
    uint32_t kernel_handles;
    uint32_t active_commands;
    struct server_context *context;
    struct server_package_cache *package;
    struct server_module *next;
};

struct server_kernel {
    uint64_t id;
    uint32_t references;
    uint32_t *active_per_core;
    struct server_module *module;
    const npu_package_kernel_t *metadata;
    struct server_kernel *next;
};

struct server_invocation {
    npu_service_launch_t *launch;
    npu_hal_allocation_t *uram_allocation;
    size_t slot_index;
    uint64_t resident;
    uint32_t proc_id;
    uint32_t task_id[3];
    uint32_t polling;
    struct server_launch_group *group;
    struct server_invocation *next;
};

struct server_output {
    struct server_output *next;
    size_t size;
    size_t offset;
    uint32_t cluster;
    uint32_t core;
    uint32_t proc;
    uint32_t flags;
    unsigned char data[];
};

struct server_launch_group {
    npu_hal_allocation_t *shared_allocation;
    uint32_t expected;
    uint32_t active;
    uint32_t completed;
    struct server_launch_group *next;
};

struct server_command {
    uint32_t kind;
    uint32_t running;
    uint32_t cancelled;
    uint32_t dispatching;
    struct server_context *context;
    struct server_stream *stream;
    struct server_event *event;
    struct server_command *next;
    union {
        struct {
            struct server_event *dependency;
        } wait;
        struct {
            struct server_buffer *destination;
            struct server_buffer *source;
            uint64_t destination_offset;
            uint64_t source_offset;
            uint64_t size;
            uint64_t transfer_id;
            uint32_t token_ready;
            uint32_t granted;
            uint32_t completed;
            int32_t transfer_status;
            npu_hal_core_t viewer;
            npu_hal_data_token_t token;
        } transfer;
        struct {
            struct server_kernel *kernel;
            npu_ipc_launch_argument_t arguments[NPU_KERNEL_MAX_ARGS];
            struct server_buffer *buffers[NPU_KERNEL_MAX_ARGS];
            uint32_t argument_count;
            NPULaunchConfig config;
            uint64_t next_unit;
            uint64_t total_units;
            uint64_t completed_units;
            int64_t aggregate_status;
            uint64_t cookie;
            struct server_launch_group *groups;
            struct server_invocation *invocations;
        } launch;
    } data;
};

struct server_context {
    uint64_t id;
    uint64_t memory_quota;
    uint64_t memory_used;
    uint32_t owner;
    uint32_t max_pending_commands;
    uint32_t pending_commands;
    uint32_t flags;
    uint32_t closing;
    struct server_session *session;
    struct server_buffer *buffers;
    struct server_stream *streams;
    struct server_module *modules;
    struct server_kernel *kernels;
    struct server_event *events;
    struct server_context *next;
};

struct server_session {
    uint64_t id;
    uid_t uid;
    gid_t gid;
    int control_fd;
    int data_fd;
    uint32_t closing;
    size_t output_bytes;
    struct server_output *output_head;
    struct server_output *output_tail;
    struct server_context *contexts;
    struct server_session *next;
};

struct dayan_server {
    dayan_server_options_t options;
    char *socket_path;
    npu_hal_t *hal;
    npu_service_device_t *device;
    npu_hal_info_t hal_info;
    int listen_fd;
    pthread_mutex_t mutex;
    pthread_cond_t condition;
    pthread_t scheduler;
    uint32_t scheduler_started;
    uint32_t stop;
    uint32_t connection_count;
    uint32_t next_owner;
    size_t next_slot;
    uint64_t next_id;
    uint64_t next_sequence;
    struct server_core_slot *slots;
    size_t slot_count;
    struct server_session *sessions;
    struct server_package_cache *packages;
    npu_service_trap_fn service_trap;
    npu_service_proc_exit_fn service_proc_exit;
    void *service_callback_context;
    size_t pending_output_bytes;
    struct server_output *pending_outputs;
    struct server_output *pending_output_tail;
};

int server_enqueue_locked(dayan_server_t *server,
                          struct server_context *context,
                          struct server_stream *stream,
                          struct server_command *command,
                          uint32_t visible_event,
                          struct server_event **event_out);
void server_complete_command_locked(dayan_server_t *server,
                                    struct server_command *command,
                                    int64_t status);
void server_cancel_context_locked(dayan_server_t *server,
                                  struct server_context *context);
void server_collect_contexts_locked(dayan_server_t *server);
int server_queue_output_locked(dayan_server_t *server,
                               struct server_session *session,
                               uint32_t cluster, uint32_t core,
                               uint32_t proc, const void *data, size_t size,
                               uint32_t flags);
void server_route_pending_outputs_locked(dayan_server_t *server);
int server_launch_resources_valid(const dayan_server_t *server,
                                  const struct server_command *command);
void *dayan_scheduler_main(void *argument);

#endif
