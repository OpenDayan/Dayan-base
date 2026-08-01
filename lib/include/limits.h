#ifndef NPU_LIMITS_H
#define NPU_LIMITS_H

#define CHAR_BIT  __CHAR_BIT__
#define SCHAR_MIN (-__SCHAR_MAX__ - 1)
#define SCHAR_MAX __SCHAR_MAX__
#define UCHAR_MAX (__SCHAR_MAX__ * 2 + 1)
#define SHRT_MIN  (-__SHRT_MAX__ - 1)
#define SHRT_MAX  __SHRT_MAX__
#define USHRT_MAX (__SHRT_MAX__ * 2U + 1U)
#define INT_MIN   (-__INT_MAX__ - 1)
#define INT_MAX   __INT_MAX__
#define UINT_MAX  (__INT_MAX__ * 2U + 1U)
#define LONG_MIN  (-__LONG_MAX__ - 1L)
#define LONG_MAX  __LONG_MAX__
#define ULONG_MAX (__LONG_MAX__ * 2UL + 1UL)

#endif
