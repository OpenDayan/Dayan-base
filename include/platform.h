#ifndef NPU_PLATFORM_H
#define NPU_PLATFORM_H

#include <stddef.h>
#include <stdint.h>

void platform_init(void);
uintptr_t platform_hart_id(void);

void platform_cache_init(void);
void platform_cache_clean(const void *addr, size_t size);
void platform_cache_invalidate(void *addr, size_t size);

long platform_rpc(uint32_t service,
                  const void *request, size_t request_size,
                  void *response, size_t response_size);

void platform_idle(void);
_Noreturn void platform_halt(void);

#endif
