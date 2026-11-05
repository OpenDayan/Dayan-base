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
