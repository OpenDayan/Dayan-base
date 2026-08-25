#ifndef NPU_RPC_INTERNAL_H
#define NPU_RPC_INTERNAL_H

#include <stddef.h>
#include <stdint.h>

#include "list.h"
#include "rpc.h"
#include "sche.h"
#include "tensor_interrupt.h"
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
    _Alignas(RPC_DESCRIPTOR_GRANULE) struct trap_record trap_record;
    struct rpc_transaction tensor_interrupt;
    _Alignas(RPC_DESCRIPTOR_GRANULE)
        struct tensor_interrupt_record tensor_interrupt_record;
    uint32_t initialized;
    uint32_t trap_pending;
    uint32_t tensor_interrupt_pending;
};

int rpc_report_trap(const void *record, size_t size);
int rpc_report_tensor_interrupt(const struct tensor_interrupt_record *record);

#endif
