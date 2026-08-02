#ifndef NPU_PLATFORM_H
#define NPU_PLATFORM_H

#include <stddef.h>
#include <stdint.h>

void platform_init(void);
uintptr_t platform_hart_id(void);

void platform_cache_init(void);
void platform_cache_clean(const void *addr, size_t size);
void platform_cache_invalidate(void *addr, size_t size);

void platform_rpc_init(void);
int platform_rpc_receive_pending(void);
uint64_t platform_rpc_receive(void);
void platform_rpc_receive_ack(void);
int platform_rpc_send_ready(void);
void platform_rpc_send(uint64_t command);

void platform_idle(void);
_Noreturn void platform_halt(void);

#endif
