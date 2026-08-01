#ifndef NPU_STDIO_H
#define NPU_STDIO_H

#include <stdarg.h>
#include <stddef.h>

#define EOF (-1)

int putchar(int ch);
int puts(const char *str);
int printf(const char *restrict format, ...);
int vprintf(const char *restrict format, va_list args);
int snprintf(char *restrict dst, size_t size,
             const char *restrict format, ...);
int vsnprintf(char *restrict dst, size_t size,
              const char *restrict format, va_list args);

#endif
