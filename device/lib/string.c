#include <stddef.h>
#include <stdint.h>

#include "string.h"

void *memcpy(void *restrict dst, const void *restrict src, size_t size)
{
    unsigned char *out = dst;
    const unsigned char *in = src;

    /*
     * Once both pointers have the same alignment, a short byte prefix brings
     * them to a machine-word boundary.  The may_alias typedef keeps this
     * valid for arbitrary byte storage while allowing the compiler to emit
     * native word loads/stores.
     */
    if ((((uintptr_t)out ^ (uintptr_t)in) & (sizeof(uintptr_t) - 1U)) == 0) {
        typedef uintptr_t memcpy_word_t __attribute__((__may_alias__));

        while (size && ((uintptr_t)out & (sizeof(uintptr_t) - 1U))) {
            *out++ = *in++;
            --size;
        }
        while (size >= sizeof(uintptr_t)) {
            *(memcpy_word_t *)out = *(const memcpy_word_t *)in;
            out += sizeof(uintptr_t);
            in += sizeof(uintptr_t);
            size -= sizeof(uintptr_t);
        }
    }

    while (size--)
        *out++ = *in++;
    return dst;
}

void *memmove(void *dst, const void *src, size_t size)
{
    unsigned char *out = dst;
    const unsigned char *in = src;

    if (out < in) {
        while (size--)
            *out++ = *in++;
    } else if (out > in) {
        out += size;
        in += size;
        while (size--)
            *--out = *--in;
    }
    return dst;
}

void *memset(void *dst, int value, size_t size)
{
    unsigned char *out = dst;

    while (size--)
        *out++ = (unsigned char)value;
    return dst;
}

int memcmp(const void *left, const void *right, size_t size)
{
    const unsigned char *a = left;
    const unsigned char *b = right;

    while (size--) {
        if (*a != *b)
            return *a - *b;
        ++a;
        ++b;
    }
    return 0;
}

size_t strlen(const char *str)
{
    const char *end = str;

    while (*end)
        ++end;
    return (size_t)(end - str);
}

size_t strnlen(const char *str, size_t maxlen)
{
    size_t size = 0;

    while (size < maxlen && str[size])
        ++size;
    return size;
}

char *strcpy(char *restrict dst, const char *restrict src)
{
    char *out = dst;

    do {
        *out++ = *src;
    } while (*src++);
    return dst;
}

char *strncpy(char *restrict dst, const char *restrict src, size_t size)
{
    size_t i = 0;

    while (i < size && src[i]) {
        dst[i] = src[i];
        ++i;
    }
    while (i < size)
        dst[i++] = '\0';
    return dst;
}

int strcmp(const char *left, const char *right)
{
    while (*left && *left == *right) {
        ++left;
        ++right;
    }
    return (unsigned char)*left - (unsigned char)*right;
}

int strncmp(const char *left, const char *right, size_t size)
{
    while (size && *left && *left == *right) {
        ++left;
        ++right;
        --size;
    }
    if (!size)
        return 0;
    return (unsigned char)*left - (unsigned char)*right;
}

char *strchr(const char *str, int ch)
{
    char value = (char)ch;

    do {
        if (*str == value)
            return (char *)str;
    } while (*str++);
    return 0;
}

char *strrchr(const char *str, int ch)
{
    const char *found = 0;
    char value = (char)ch;

    do {
        if (*str == value)
            found = str;
    } while (*str++);
    return (char *)found;
}
