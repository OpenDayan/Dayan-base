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

#ifndef DAYAN_QEMU_ASSERT_H
#define DAYAN_QEMU_ASSERT_H

#ifdef __cplusplus
extern "C" {
#endif

__attribute__((noreturn))
void dayan_test_assert_fail(const char *expression, const char *file,
                            unsigned line);

#ifdef __cplusplus
}
#endif

#define assert(expression)                                                   \
    ((expression) ? (void)0 :                                                \
     dayan_test_assert_fail(#expression, __FILE__, __LINE__))

#endif
