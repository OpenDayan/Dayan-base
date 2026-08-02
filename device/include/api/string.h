#ifndef DAYAN_STRING_H
#define DAYAN_STRING_H

#include <stddef.h>

#include "dayan_attrs.h"

#ifdef __cplusplus
extern "C" {
#endif

DAYAN_LIBC_API void *memcpy(void *DAYAN_RESTRICT dst,
                            const void *DAYAN_RESTRICT src, size_t size);
DAYAN_LIBC_API void *memmove(void *dst, const void *src, size_t size);
DAYAN_LIBC_API void *memset(void *dst, int value, size_t size);
DAYAN_LIBC_API int memcmp(const void *left, const void *right, size_t size);

DAYAN_LIBC_API size_t strlen(const char *str);
DAYAN_LIBC_API size_t strnlen(const char *str, size_t maxlen);
DAYAN_LIBC_API char *strcpy(char *DAYAN_RESTRICT dst,
                            const char *DAYAN_RESTRICT src);
DAYAN_LIBC_API char *strncpy(char *DAYAN_RESTRICT dst,
                             const char *DAYAN_RESTRICT src, size_t size);
DAYAN_LIBC_API int strcmp(const char *left, const char *right);
DAYAN_LIBC_API int strncmp(const char *left, const char *right, size_t size);
DAYAN_LIBC_API char *strchr(const char *str, int ch);
DAYAN_LIBC_API char *strrchr(const char *str, int ch);

#ifdef __cplusplus
}
#endif

#endif
