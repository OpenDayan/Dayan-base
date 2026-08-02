#ifndef DAYAN_STDARG_H
#define DAYAN_STDARG_H

typedef __builtin_va_list va_list;

#define va_start(args, last) __builtin_va_start(args, last)
#define va_arg(args, type)   __builtin_va_arg(args, type)
#define va_end(args)         __builtin_va_end(args)
#define va_copy(dst, src)    __builtin_va_copy(dst, src)

#endif
