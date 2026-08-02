#ifndef NPU_RPC_INTERNAL_H
#define NPU_RPC_INTERNAL_H

#include <stddef.h>
#include <stdint.h>

#include "list.h"
#include "sche.h"
#include "trap.h"

struct rpc_transaction;
typedef void (*rpc_cleanup_fn)(struct rpc_transaction *transaction,
                               void *arg);

struct rpc_transaction {
    struct list_head node;
    struct list_head wait;
    uint64_t command;
    rpc_cleanup_fn cleanup;
    void *cleanup_arg;
    volatile uint32_t completed;
    long status;
};

struct rpc_state {
    struct proc receive;
    struct proc send;
    struct list_head send_queue;
    struct list_head send_event;
    struct rpc_transaction trap;
    struct trap_record trap_record;
    uint32_t initialized;
    uint32_t trap_pending;
};

int rpc_report_trap(const void *record, size_t size);

#endif
