#ifndef NPU_CONTAINER_OF_H
#define NPU_CONTAINER_OF_H

#include <stddef.h>

#define container_of(ptr, type, member) __extension__ ({ \
    const __typeof__(((type *)0)->member) *__member = (ptr); \
    (type *)((char *)__member - offsetof(type, member)); \
})

#define container_of_const(ptr, type, member) __extension__ ({ \
    const __typeof__(((type *)0)->member) *__member = (ptr); \
    (const type *)((const char *)__member - offsetof(type, member)); \
})

#endif
