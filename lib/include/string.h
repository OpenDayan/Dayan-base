#ifndef NPU_STRING_H
#define NPU_STRING_H

#include <stddef.h>

void *memcpy(void *restrict dst, const void *restrict src, size_t size);
void *memmove(void *dst, const void *src, size_t size);
void *memset(void *dst, int value, size_t size);
int memcmp(const void *left, const void *right, size_t size);

size_t strlen(const char *str);
size_t strnlen(const char *str, size_t maxlen);
char *strcpy(char *restrict dst, const char *restrict src);
char *strncpy(char *restrict dst, const char *restrict src, size_t size);
int strcmp(const char *left, const char *right);
int strncmp(const char *left, const char *right, size_t size);
char *strchr(const char *str, int ch);
char *strrchr(const char *str, int ch);

#endif
