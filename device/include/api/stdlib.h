#ifndef DAYAN_STDLIB_H
#define DAYAN_STDLIB_H

#include <stddef.h>

#include "dayan_attrs.h"

#ifdef __cplusplus
extern "C" {
#endif

DAYAN_LIBC_API void *malloc(size_t size);
DAYAN_LIBC_API void *calloc(size_t count, size_t size);
DAYAN_LIBC_API void *realloc(void *ptr, size_t size);
DAYAN_LIBC_API void free(void *ptr);

DAYAN_LIBC_API DAYAN_NORETURN void abort(void);
DAYAN_LIBC_API DAYAN_NORETURN void exit(int status);

#ifdef __cplusplus
}
#endif

#endif
