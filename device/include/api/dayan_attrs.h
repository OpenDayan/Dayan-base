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
