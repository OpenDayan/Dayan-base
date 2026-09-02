#ifndef NPU_RPC_INTERNAL_H
#define NPU_RPC_INTERNAL_H

#include <stddef.h>
#include <stdint.h>

#include "list.h"
#include "rpc.h"
#include "sche.h"

struct rpc_transaction;
struct tensor_interrupt_record;
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
    struct rpc_transaction tensor_interrupt;
    struct thread_list send_event;
    uint8_t initialized;
};

int rpc_report_trap(const void *record, size_t size);
int rpc_tensor_interrupt_ready(void);
int rpc_report_tensor_interrupt(const struct tensor_interrupt_record *record);

#endif
