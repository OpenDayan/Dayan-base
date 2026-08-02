#include <stdarg.h>
#include <stddef.h>
#include <stdint.h>

#include "limits.h"
#include "rpc.h"
#include "sche.h"
#include "stdio.h"
#include "string.h"

struct output {
    int (*put)(void *context, char ch);
    void *context;
    size_t count;
    int failed;
};

static void out_char(struct output *out, char ch)
{
    if (!out->failed && out->put(out->context, ch) < 0)
        out->failed = 1;
    ++out->count;
}

static void out_repeat(struct output *out, char ch, size_t count)
{
    while (count--)
        out_char(out, ch);
}

static void out_string(struct output *out, const char *str, size_t size)
{
    while (size--)
        out_char(out, *str++);
}

static void out_number(struct output *out, uint64_t value, unsigned base,
                       int upper, int negative, int prefix,
                       size_t width, int left, int zero)
{
    const char *alphabet = upper ?
        "0123456789ABCDEF" : "0123456789abcdef";
    char digits[32];
    size_t count = 0;
    size_t lead = (negative ? 1U : 0U) + (prefix ? 2U : 0U);
    size_t padding;

    do {
        digits[count++] = alphabet[value % base];
        value /= base;
    } while (value);

    padding = width > lead + count ? width - lead - count : 0;
    if (!left && !zero)
        out_repeat(out, ' ', padding);
    if (negative)
        out_char(out, '-');
    if (prefix) {
        out_char(out, '0');
        out_char(out, upper ? 'X' : 'x');
    }
    if (!left && zero)
        out_repeat(out, '0', padding);
    while (count)
        out_char(out, digits[--count]);
    if (left)
        out_repeat(out, ' ', padding);
}

enum length {
    LEN_INT,
    LEN_LONG,
    LEN_LLONG,
    LEN_SIZE,
};

static uint64_t unsigned_arg(va_list *args, enum length length)
{
    switch (length) {
    case LEN_LONG:
        return va_arg(*args, unsigned long);
    case LEN_LLONG:
        return va_arg(*args, unsigned long long);
    case LEN_SIZE:
        return va_arg(*args, size_t);
    default:
        return va_arg(*args, unsigned int);
    }
}

static int64_t signed_arg(va_list *args, enum length length)
{
    switch (length) {
    case LEN_LONG:
        return va_arg(*args, long);
    case LEN_LLONG:
        return va_arg(*args, long long);
    case LEN_SIZE:
        return va_arg(*args, ptrdiff_t);
    default:
        return va_arg(*args, int);
    }
}

static int format(struct output *out, const char *format, va_list input)
{
    va_list args;

    va_copy(args, input);
    while (*format) {
        size_t width = 0;
        int left = 0;
        int zero = 0;
        enum length length = LEN_INT;
        char spec;

        if (*format != '%') {
            out_char(out, *format++);
            continue;
        }
        ++format;
        if (*format == '-') {
            left = 1;
            ++format;
        }
        if (*format == '0') {
            zero = 1;
            ++format;
        }
        while (*format >= '0' && *format <= '9') {
            width = width * 10 + (size_t)(*format - '0');
            ++format;
        }
        if (*format == 'l') {
            length = LEN_LONG;
            if (*++format == 'l') {
                length = LEN_LLONG;
                ++format;
            }
        } else if (*format == 'z') {
            length = LEN_SIZE;
            ++format;
        }

        spec = *format;
        if (!spec)
            break;
        ++format;
        switch (spec) {
        case '%':
            out_char(out, '%');
            break;
        case 'c': {
            char ch = (char)va_arg(args, int);
            size_t padding = width > 1 ? width - 1 : 0;

            if (!left)
                out_repeat(out, ' ', padding);
            out_char(out, ch);
            if (left)
                out_repeat(out, ' ', padding);
            break;
        }
        case 's': {
            const char *str = va_arg(args, const char *);
            size_t size;
            size_t padding;

            if (!str)
                str = "(null)";
            size = strlen(str);
            padding = width > size ? width - size : 0;
            if (!left)
                out_repeat(out, ' ', padding);
            out_string(out, str, size);
            if (left)
                out_repeat(out, ' ', padding);
            break;
        }
        case 'd':
        case 'i': {
            int64_t value = signed_arg(&args, length);
            int negative = value < 0;
            uint64_t magnitude = negative ?
                (uint64_t)(-(value + 1)) + 1 : (uint64_t)value;

            out_number(out, magnitude, 10, 0, negative, 0,
                       width, left, zero);
            break;
        }
        case 'u':
            out_number(out, unsigned_arg(&args, length), 10, 0, 0, 0,
                       width, left, zero);
            break;
        case 'x':
        case 'X':
            out_number(out, unsigned_arg(&args, length), 16,
                       spec == 'X', 0, 0, width, left, zero);
            break;
        case 'p':
            out_number(out, (uintptr_t)va_arg(args, void *), 16, 0, 0, 1,
                       width, left, zero);
            break;
        default:
            out_char(out, '%');
            out_char(out, spec);
            break;
        }
    }
    va_end(args);

    if (out->failed || out->count > INT_MAX)
        return -1;
    return (int)out->count;
}

struct string_output {
    char *dst;
    size_t size;
    size_t pos;
};

static int string_put(void *context, char ch)
{
    struct string_output *out = context;

    if (out->size && out->pos < out->size - 1)
        out->dst[out->pos] = ch;
    ++out->pos;
    return 0;
}

int vsnprintf(char *restrict dst, size_t size,
              const char *restrict fmt, va_list args)
{
    struct string_output string = { dst, size, 0 };
    struct output out = { string_put, &string, 0, 0 };
    int result;

    if (!fmt || (!dst && size))
        return -1;
    result = format(&out, fmt, args);
    if (size)
        dst[string.pos < size ? string.pos : size - 1] = '\0';
    return result;
}

int snprintf(char *restrict dst, size_t size,
             const char *restrict fmt, ...)
{
    va_list args;
    int result;

    va_start(args, fmt);
    result = vsnprintf(dst, size, fmt, args);
    va_end(args);
    return result;
}

static int proc_flush(struct proc *proc)
{
    if (!proc || proc->printf_size > proc->printf_capacity ||
        (!proc->printf_buffer != !proc->printf_capacity))
        return -1;
    if (!proc->printf_size)
        return 0;
    if (rpc_write(proc->printf_buffer, proc->printf_size) < 0)
        return -1;
    proc->printf_size = 0;
    return 0;
}

static int proc_put(void *context, char ch)
{
    struct proc *proc = context;

    if (!proc || !proc->printf_buffer || !proc->printf_capacity ||
        proc->printf_size > proc->printf_capacity)
        return -1;
    if (proc->printf_size == proc->printf_capacity &&
        proc_flush(proc) < 0)
        return -1;
    proc->printf_buffer[proc->printf_size++] = ch;
    if (proc->printf_size == proc->printf_capacity &&
        proc_flush(proc) < 0)
        return -1;
    return 0;
}

int vprintf(const char *restrict fmt, va_list args)
{
    struct proc *proc = proc_self();
    struct output out = { proc_put, proc, 0, 0 };

    if (!fmt || !proc || !proc->printf_buffer || !proc->printf_capacity)
        return -1;
    return format(&out, fmt, args);
}

int printf(const char *restrict fmt, ...)
{
    va_list args;
    int result;

    va_start(args, fmt);
    result = vprintf(fmt, args);
    va_end(args);
    return result;
}

int putchar(int ch)
{
    struct proc *proc = proc_self();

    if (!proc || proc_put(proc, (char)ch) < 0)
        return EOF;
    return (unsigned char)ch;
}

int puts(const char *str)
{
    struct proc *proc = proc_self();

    if (!str || !proc)
        return EOF;
    while (*str) {
        if (proc_put(proc, *str++) < 0)
            return EOF;
    }
    if (proc_put(proc, '\n') < 0)
        return EOF;
    return 0;
}

int fflush(FILE *stream)
{
    if (stream && stream != stdout)
        return EOF;
    return proc_flush(proc_self()) < 0 ? EOF : 0;
}
