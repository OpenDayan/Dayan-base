#ifndef DAYAN_API_ATTRS_H
#define DAYAN_API_ATTRS_H

#if defined(__clang__) || defined(__GNUC__)
#define DAYAN_LIBC_API \
    __attribute__((visibility("default"), section(".text.api.libc")))
#define DAYAN_API \
    __attribute__((visibility("default"), section(".text.api.dapi")))
#define DAYAN_NORETURN __attribute__((noreturn))
#else
#define DAYAN_LIBC_API
#define DAYAN_API
#define DAYAN_NORETURN
#endif

#ifdef __cplusplus
#define DAYAN_RESTRICT __restrict
#else
#define DAYAN_RESTRICT restrict
#endif

#endif
