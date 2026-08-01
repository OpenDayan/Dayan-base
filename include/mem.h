#ifndef NPU_MEM_H
#define NPU_MEM_H

#include <stddef.h>

int mem_init(void *base, size_t size);
void *mem_alloc(size_t size);
void *mem_resize(void *ptr, size_t size);
void mem_free(void *ptr);
size_t mem_size(const void *ptr);
size_t mem_capacity(void);
size_t mem_available(void);

#endif
