#ifndef DAYAN_STDDEF_H
#define DAYAN_STDDEF_H

typedef __SIZE_TYPE__ size_t;
typedef __PTRDIFF_TYPE__ ptrdiff_t;
#ifndef __cplusplus
typedef __WCHAR_TYPE__ wchar_t;
#endif

typedef struct {
    long long __ll;
    long double __ld;
} max_align_t;

#ifdef __cplusplus
#define NULL __null
#else
#define NULL ((void *)0)
#endif

#define offsetof(type, member) __builtin_offsetof(type, member)

#endif
