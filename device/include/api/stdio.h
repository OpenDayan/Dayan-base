#ifndef DAYAN_STDIO_H
#define DAYAN_STDIO_H

#include <stdarg.h>
#include <stddef.h>

#include "dayan_attrs.h"

#define EOF (-1)

typedef struct dayan_file FILE;

/* Dayan currently exposes one proc-local output stream and no FILE object. */
#define stdout ((FILE *)1)

#ifdef __cplusplus
extern "C" {
#endif

DAYAN_LIBC_API int putchar(int ch);
DAYAN_LIBC_API int puts(const char *str);
DAYAN_LIBC_API int fflush(FILE *stream);
DAYAN_LIBC_API int printf(const char *DAYAN_RESTRICT format, ...);
DAYAN_LIBC_API int vprintf(const char *DAYAN_RESTRICT format, va_list args);
DAYAN_LIBC_API int snprintf(char *DAYAN_RESTRICT dst, size_t size,
                            const char *DAYAN_RESTRICT format, ...);
DAYAN_LIBC_API int vsnprintf(char *DAYAN_RESTRICT dst, size_t size,
                             const char *DAYAN_RESTRICT format, va_list args);

#ifdef __cplusplus
}
#endif

#endif
