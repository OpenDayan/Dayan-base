#ifndef NPU_PLATFORM_H
#define NPU_PLATFORM_H

#include <stddef.h>
#include <stdint.h>

void platform_init(void);
uintptr_t platform_hart_id(void);

void platform_cache_init(void);
void platform_cache_flush(const void *addr, size_t size);
void platform_cache_invalidate(void *addr, size_t size);
void platform_data_fence(void);
void platform_instruction_fence(void);

/* Standalone test hooks; boards may override the default AXI UARTLite output. */
void platform_test_init(void *coherent_backing);
long platform_print(const void *data, size_t size);
void platform_test_complete(int64_t result);

void platform_rpc_init(void);
void platform_rpc_ready(void);
_Noreturn void platform_rpc_halt(void);
uint64_t platform_rpc_receive(void);
void platform_rpc_receive_ack(void);
/* Return after the Host acknowledges the mailbox. */
void platform_rpc_send(uint64_t command);

void platform_idle(void);
_Noreturn void platform_halt(void);

#endif
