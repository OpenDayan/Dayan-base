#ifndef NPU_RPC_INTERNAL_H
#define NPU_RPC_INTERNAL_H

#include <stddef.h>
#include <stdint.h>

#include "list.h"
#include "rpc.h"
#include "sche.h"

struct rpc_transaction;
struct trap_record;
typedef void (*rpc_cleanup_fn)(struct rpc_transaction *transaction);

struct rpc_transaction {
    struct list_head node;
    uint64_t command;
    rpc_cleanup_fn cleanup;
    long status;
    volatile uint32_t completed;
    struct thread_list wait;
};

struct rpc_state {
    struct list_head send_queue;
    struct rpc_transaction trap;
    struct thread_list send_event;
    void (*send)(uint64_t command);
};

void rpc_start_local(thread_fn launcher, void (*send)(uint64_t command));
void rpc_send_command(uint64_t command);
int rpc_report_trap(const struct trap_record *record);
void rpc_send_notify(void);

#endif
