#ifndef NPU_STDLIB_H
#define NPU_STDLIB_H

#include <stddef.h>

void *malloc(size_t size);
void *calloc(size_t count, size_t size);
void *realloc(void *ptr, size_t size);
void free(void *ptr);

_Noreturn void abort(void);
_Noreturn void exit(int status);

#endif
