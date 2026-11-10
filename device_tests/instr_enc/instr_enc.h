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

#ifndef TXENC_INSTR_ENC_H
#define TXENC_INSTR_ENC_H

/* 指令编码库伞形头：五个部分一次引入。
 *   device_tests/VEC/xxx_test/main.cpp:
 *     #include "../../instr_enc/instr_enc.h"
 *     txenc::emit32_a0_a1<txenc::lsu_dma_enc()>(dram, uram);
 */

#include "tx_base_enc.h"
#include "tx_lsu_enc.h"
#include "tx_vec_enc.h"
#include "tx_mat_enc.h"
#include "tx_macro_enc.h"

#endif
