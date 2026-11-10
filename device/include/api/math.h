/***************************************************************************************
 * Copyright (c) 2026 Institute of Computing Technology, Chinese Academy of Sciences
 *
 * Dayan-base is licensed under the Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *
 *     http://license.coscl.org.cn/MulanPSL2
 *
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 *
 * See the Mulan PSL v2 for more details.
***************************************************************************************/

#ifndef DAYAN_MATH_H
#define DAYAN_MATH_H

#include "dayan_attrs.h"

#define INFINITY  (__builtin_inff())
#define NAN       (__builtin_nanf(""))
#define HUGE_VALF INFINITY

#define isnan(value)    __builtin_isnan(value)
#define isinf(value)    __builtin_isinf(value)
#define isfinite(value) __builtin_isfinite(value)
#define signbit(value)  __builtin_signbit(value)

#ifdef __cplusplus
extern "C" {
#endif

DAYAN_LIBC_API float fabsf(float value);
DAYAN_LIBC_API float copysignf(float value, float sign);
DAYAN_LIBC_API float fminf(float left, float right);
DAYAN_LIBC_API float fmaxf(float left, float right);
DAYAN_LIBC_API float sqrtf(float value);
DAYAN_LIBC_API float rsqrtf(float value);
DAYAN_LIBC_API float expf(float value);
DAYAN_LIBC_API float logf(float value);

#ifdef __cplusplus
}
#endif

#endif
