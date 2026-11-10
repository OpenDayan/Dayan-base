/***************************************************************************************
 * Copyright (c) 2026 Institute of Computing Technology, Chinese Academy of Sciences
 *
 * Dayan-base is licensed under the Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *
 *     http://license.coscl.org.cn/MulanPSL2
 *
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 *
 * See the Mulan PSL v2 for more details.
***************************************************************************************/

#ifndef NPU_PLATFORM_H
#define NPU_PLATFORM_H

#include <stddef.h>
#include <stdint.h>

void platform_init(void);
uintptr_t platform_hart_id(void);

void platform_cache_init(void);
void platform_cache_flush(const void *addr, size_t size);
void platform_cache_invalidate(void *addr, size_t size);
static inline void platform_prefetch_read(const void *addr)
{
    /* Zicbop prefetch.r 0(addr); a harmless ORI-to-x0 hint without Zicbop. */
    __asm__ volatile (".insn i 0x13, 6, x0, %0, 1" :: "r"(addr) : "memory");
}
void platform_data_fence(void);
void platform_instruction_fence(void);

/* Standalone hooks publish ALIVE/RESULT/DONE and append output to a DRAM ring. */
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
