// Adapted from llama.cpp ggml-dayan-gated-delta-net.h (chunked scalar-gate path).
/*
MIT License

Copyright (c) 2023-2026 The ggml authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/
#pragma once
#include <stdint.h>

struct DayanGatedDeltaNetConfig {
    uint32_t head_dim;
    uint32_t heads;
    uint32_t tokens;
    uint32_t sequences;
    uint32_t snapshots;
    uint32_t gate_width;
    uint32_t scale_bits;
    uint32_t q_heads;
    uint32_t k_heads;
    uint32_t q_sequences;
    uint32_t k_sequences;
    uint32_t reserved;
    uint64_t q_nb[4];
    uint64_t k_nb[4];
    uint64_t v_nb[4];
    uint64_t g_nb[4];
    uint64_t beta_nb[4];
};


constexpr uint32_t kGdnUramBytes = 768U * 1024U;

#if defined(__DAYAN_DEVICE__)
#include <dayan_intrinsics.h>
#include <dayan_sys.h>

namespace gdn_detail {
constexpr uint32_t kHeadDim = 128;
constexpr long kLanes = 64;
constexpr long kVecBytes = kLanes * sizeof(float);
constexpr long kScalarStride = 32;
constexpr long kRowGroups = 2;

enum VecTcsr {
    kVecControl = 0,
    kVecSync    = 1,
    kVecDim0    = 2,
    kVecDim1    = 3,
    kVecDim2    = 4,
    kVecDim3    = 5,
    kVecDim4    = 6,
    kVecDim5    = 7,
    kVecS0Base  = 9,
    kVecV0Base  = 37,
    kVecV1Base  = 44,
    kVecV2Base  = 51,
    kVecV3Base  = 58,
};

static inline uint64_t uram_address(long offset) {
    return (uintptr_t) dayan_uram_base() + offset;
}

template <typename T> static inline T * uram_pointer(long offset) {
    return reinterpret_cast<T *>(uram_address(offset));
}

template <long Base>
static inline void configure_vec_iterator(long dim0_stride, long dim1_stride, long dim2_stride, long dim3_stride) {
#if !GDN_VEC_TGMT
    // Rows execute sequentially inside one task, using DIM1 instead of DIM2.
    // DIM0 remains the inner recurrence/half-row loop.
    dim1_stride = dim2_stride;
    dim2_stride = 0;
    dim3_stride = 0;
#endif
    __tx_vec_tcsr_write<Base>(0);
    __tx_vec_tcsr_write<Base + 1>(dim0_stride);
    __tx_vec_tcsr_write<Base + 2>(dim1_stride);
    __tx_vec_tcsr_write<Base + 3>(dim2_stride);
    __tx_vec_tcsr_write<Base + 4>(dim3_stride);
    __tx_vec_tcsr_write<Base + 5>(0);
    __tx_vec_tcsr_write<Base + 6>(0);
    if constexpr (Base == kVecV0Base) {
        __tx_vec_tcsr_write<VEC_V0_URAM_CHANNEL_STRIDE>(32);
    } else if constexpr (Base == kVecV1Base) {
        __tx_vec_tcsr_write<VEC_V1_URAM_CHANNEL_STRIDE>(32);
    } else if constexpr (Base == kVecV2Base) {
        __tx_vec_tcsr_write<VEC_V2_URAM_CHANNEL_STRIDE>(32);
    } else if constexpr (Base == kVecV3Base) {
        __tx_vec_tcsr_write<VEC_V3_URAM_CHANNEL_STRIDE>(32);
    }
}

static inline long vec_control(long prologue0, long prologue1, long loop, long epilogue1, long epilogue0) {
    return static_cast<long>((static_cast<uint64_t>(prologue0) & 0xffU) |
                             ((static_cast<uint64_t>(prologue1) & 0xffU) << 8) |
                             ((static_cast<uint64_t>(loop) & 0xffU) << 16) |
                             ((static_cast<uint64_t>(epilogue1) & 0xffU) << 24) |
                             ((static_cast<uint64_t>(epilogue0) & 0xffU) << 32));
}

static inline void begin_vec_block(long instructions) {
    long free_entries;
    do {
        __tx_vec_tcsr_read<kVecControl>(free_entries);
    } while (free_entries < instructions);
}

static inline void start_vec_block(long prologue0, long prologue1, long loop, long epilogue1, long epilogue0) {
#if !GDN_VEC_TGMT
    // Repeat each row's prologue/epilogue in DIM1, retaining its DIM0 loop.
    prologue1 = prologue0;
    prologue0 = 0;
    epilogue1 = epilogue0;
#endif
    // Keep CPU URAM stores before VEC submission.
    __asm__ volatile("" ::: "memory");
    __tx_vec_tcsr_write<kVecSync>(0);
    __tx_vec_tcsr_write<kVecControl>(vec_control(prologue0, prologue1, loop, epilogue1, epilogue0));
}

template <long Group>
static inline void configure_lsu_rows(long rows, long row_bytes, long input_stride, long output_stride) {
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_BLOCK>(0);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_DIM0>(row_bytes / 64 - 1);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_DIM1>(rows - 1);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_DIM2>(0);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_INBLOCKSTRIDE>(64);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_INDIM0STRIDE>(64);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_INDIM1STRIDE>(input_stride);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_INDIM2STRIDE>(0);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_OUTBLOCKSTRIDE>(64);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_OUTDIM0STRIDE>(64);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_OUTDIM1STRIDE>(output_stride);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_OUTDIM2STRIDE>(0);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_BLOCKTRANSPOSE>(6);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_BLOCKPADDING>(0);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_REQUESTREADSIZE>(63);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_PADDINGVALUE>(0);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_URAMWRITESWIZZLEMBS>(0);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_URAMREADSWIZZLEMBS>(0);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_REQUESTWRITESIZE>(63);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_TABLESIZE>(0);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_TABLESTRIDE>(64);
}

static inline void configure_vec_outer(long rows) {
#if GDN_VEC_TGMT
    __tx_vec_tcsr_write<kVecDim1>(0);
    __tx_vec_tcsr_write<kVecDim2>(rows - 1);
#else
    // Only DIM2..5 fork hardware tasks. Zero bounds mean exactly one task.
    __tx_vec_tcsr_write<kVecDim1>(rows - 1);
    __tx_vec_tcsr_write<kVecDim2>(0);
#endif
    __tx_vec_tcsr_write<kVecDim0>(0);
    __tx_vec_tcsr_write<kVecDim3>(0);
    __tx_vec_tcsr_write<kVecDim4>(0);
    __tx_vec_tcsr_write<kVecDim5>(0);
}

static inline const float * gdn_row(const float *  base,
                                    const uint64_t nb[4],
                                    uint32_t       head,
                                    uint32_t       token,
                                    uint32_t       sequence) {
    return reinterpret_cast<const float *>(reinterpret_cast<const char *>(base) + (uint64_t) head * nb[1] +
                                           (uint64_t) token * nb[2] + (uint64_t) sequence * nb[3]);
}

#if !GDN_SOFTWARE_PIPELINE
static inline void load_gdn_pair(uint64_t destination0, uint64_t destination1,
                                  const float * source0, const float * source1,
                                  uint64_t stride0, uint64_t stride1) {
    const uint64_t source_delta = (uint64_t) source1 - (uint64_t) source0;
    // DIM2 joins two tensors with equal row strides; hardware strides are 28 bits.
    if (stride0 == stride1 && source_delta < (uint64_t{1} << 28)) {
        __tx_lsu_tcsr_write<LSU_DIM2>(1);
        __tx_lsu_tcsr_write<LSU_INDIM2STRIDE>(source_delta);
        __tx_lsu_tcsr_write<LSU_OUTDIM2STRIDE>(destination1 - destination0);
        __tx_lsu_dma_d2u<0>(destination0, (uint64_t) source0);
    } else {
        __tx_lsu_tcsr_write<LSU_DIM2>(0);
        __tx_lsu_dma_d2u<0>(destination0, (uint64_t) source0);
        __tx_lsu_tcsr_write<LSU_INDIM1STRIDE>(stride1);
        __tx_lsu_dma_d2u<0>(destination1, (uint64_t) source1);
    }
}
#endif

constexpr long kChunkTokens = 64;
constexpr long kSolveBlockTokens = 16;
constexpr long kHeadStride = kHeadDim * sizeof(float) + 64;
constexpr long kChunkStride = kChunkTokens * sizeof(float) + 64;

constexpr long kInputQ = 0;
constexpr long kInputK = kInputQ + kChunkTokens * kHeadStride;
constexpr long kInputV = kInputK + kChunkTokens * kHeadStride;
constexpr long kInputG = kInputV + kChunkTokens * kHeadStride;
constexpr long kInputBeta = kInputG + kVecBytes;
constexpr long kInputBytes = kInputBeta + kVecBytes;

// U and the temporary W transpose share one triangular solve.
constexpr long kPreparedU = 0;
constexpr long kPreparedKT = kPreparedU + kHeadDim * kChunkStride;
constexpr long kPreparedW = kPreparedKT + kHeadDim * kChunkStride;
constexpr long kPreparedQ = kPreparedW + kChunkTokens * kHeadStride;
constexpr long kPreparedA = kPreparedQ + kChunkTokens * kHeadStride;
constexpr long kPreparedG = kPreparedA + kChunkTokens * kChunkStride;
constexpr long kPreparedDecay = kPreparedG + kVecBytes;
constexpr long kPreparedBytes = kPreparedDecay + kVecBytes;

constexpr long kChunkPrepared = 2 * kInputBytes;
constexpr long kChunkState = kChunkPrepared + 2 * kPreparedBytes;
constexpr long kChunkR = kChunkState + kHeadDim * kHeadStride;
constexpr long kChunkO = kChunkR + kHeadDim * kChunkStride;
constexpr long kChunkKK = kChunkO + kHeadDim * kChunkStride;
constexpr long kChunkKQ = kChunkKK + kChunkTokens * kChunkStride;
constexpr long kChunkGCum = kChunkKQ + kChunkTokens * kChunkStride;
constexpr long kChunkTailDecay = kChunkGCum + kVecBytes;
constexpr long kChunkLower = kChunkTailDecay + kVecBytes;
constexpr long kChunkDiag = kChunkLower + kChunkTokens * kScalarStride;
constexpr long kChunkOneHot = kChunkDiag + kChunkTokens * kScalarStride;
constexpr long kChunkEnd = kChunkOneHot + kChunkTokens * kScalarStride;
static_assert(kChunkEnd <= kGdnUramBytes, "GDN pipeline exceeds URAM budget");

struct GdnPipeline {
    const DayanGatedDeltaNetConfig * config;
    const float * q;
    const float * k;
    const float * v;
    const float * g;
    const float * beta;
    uint64_t chunks;
    uint64_t generations;
    uint32_t task;
    uint32_t tasks;
};

struct GdnChunk {
    uint32_t head;
    uint32_t sequence;
    uint32_t token;
    long rows;
    long input;
    long prepared;
};

static inline GdnChunk gdn_chunk(const GdnPipeline & pipeline, uint64_t generation) {
    const auto * config = pipeline.config;
    const uint64_t work = pipeline.task + generation / pipeline.chunks * pipeline.tasks;
    const uint32_t token = (uint32_t) (generation % pipeline.chunks * kChunkTokens);
    return { (uint32_t) (work % config->heads), (uint32_t) (work / config->heads), token,
             config->tokens - token < kChunkTokens ? config->tokens - token : kChunkTokens,
             (long) (generation & 1) * kInputBytes,
             kChunkPrepared + (long) (generation & 1) * kPreparedBytes };
}

static inline void gdn_check(int status) {
    if (status < 0) {
        dayan_thread_exit(status);
    }
}

static inline void configure_chunk_mat(long m, long n, long k, long a_stride, long b_stride, long c_stride) {
    __tx_mat_tcsr_write<MAT_M>(m - 1);
    __tx_mat_tcsr_write<MAT_N>(n - 1);
    __tx_mat_tcsr_write<MAT_K>(k - 1);
    __tx_mat_tcsr_write<MAT_ASTRIDE>(a_stride);
    __tx_mat_tcsr_write<MAT_BSTRIDE>(b_stride);
    __tx_mat_tcsr_write<MAT_CSTRIDE>(c_stride);
}

static inline void initialize_chunk_mat() {
    __tx_mat_tcsr_write<MAT_ASWIZZLEMBS>(0);
    __tx_mat_tcsr_write<MAT_BSWIZZLEMBS>(0);
    __tx_mat_tcsr_write<MAT_CSWIZZLEMBS>(0);
    __tx_mat_tcsr_write<MAT_PERSISTENTMODE>(0);
    __tx_mat_tcsr_write<MAT_DATATYPE>(MAT_DATATYPE_PAIR(BaseNPUInstBuilder::DT_MATIN_TF32, BaseNPUInstBuilder::DT_MATIN_TF32));
}

static inline void issue_chunk_prefix(const GdnChunk & chunk) {
    constexpr long instructions = 38;
    configure_vec_outer(1);
    configure_vec_iterator<kVecV0Base>(0, 0, 0, 0);
    begin_vec_block(instructions);
    __tx_vec_lsu_u2v<0, 0>(uram_address(chunk.input + kInputG));
    for (long shift = 1; shift < kChunkTokens; shift *= 2) {
        __tx_vec_lsu_g2s<0>((long) (~uint64_t{0} << shift));
        __tx_vec_shuffle_ror<0, 0>(kChunkTokens - shift);
        __tx_vec_vv_fp32_add<0, 0, 1, 0>();
        __tx_vec_vv_mux2<0, 2, 0, 0>();
    }
    __tx_vec_lsu_v2u<0, 0>(uram_address(kChunkGCum));
    __tx_vec_lsu_g2v_broadcast32<1, 0>(0x3fb8aa3bU);
    __tx_vec_vv_fp32_mul<0, 0, 16, 0>();
    __tx_vec_v_fp32_sfu<0, 0, 1>();
    __tx_vec_lsu_v2u<0, 0>(uram_address(chunk.prepared + kPreparedG));
    __tx_vec_shuffle_broadcast<0, 0>(chunk.rows - 1);
    __tx_vec_lsu_v2u<0, 0>(uram_address(chunk.prepared + kPreparedDecay));
    __tx_vec_lsu_u2v<0, 0>(uram_address(kChunkGCum));
    __tx_vec_shuffle_broadcast<0, 0>(chunk.rows - 1);
    __tx_vec_vv_fp32_sub<0, 0, 1, 0>();
    __tx_vec_vv_fp32_mul<0, 0, 16, 0>();
    __tx_vec_v_fp32_sfu<0, 0, 1>();
    __tx_vec_lsu_v2u<0, 0>(uram_address(kChunkTailDecay));
    start_vec_block(instructions, instructions, instructions, instructions, instructions);
}

static inline void issue_chunk_coefficients(const GdnChunk & chunk, uint32_t scale_bits) {
    constexpr long instructions = 25;
    configure_vec_outer(chunk.rows);
    configure_vec_iterator<kVecS0Base>(0, 0, 1, 0);
    configure_vec_iterator<kVecV0Base>(0, 0, kChunkStride, 0);
    configure_vec_iterator<kVecV1Base>(0, 0, 0, 0);
    configure_vec_iterator<kVecV3Base>(0, 0, kScalarStride, 0);
    begin_vec_block(instructions);
    __tx_vec_lsu_u2v<1, 1>(uram_address(kChunkGCum));
    __tx_vec_lsu_g2v_broadcast32<1, 0>(0);
    __tx_vec_s_iter_gets<0, 0, 0, 0>();
    __tx_vec_shuffle_broadcast_dyn<0, 17, 0>();
    __tx_vec_vv_fp32_sub<0, 17, 0, 0>();
    // Mask unused entries before exp to avoid overflow.
    __tx_vec_lsu_u2s<0, 3>(uram_address(kChunkDiag));
    __tx_vec_vv_mux2<0, 16, 0, 0>();
    __tx_vec_lsu_g2v_broadcast32<0, 0>(0x3fb8aa3bU);
    __tx_vec_vv_fp32_mul<0, 0, 1, 0>();
    __tx_vec_v_fp32_sfu<0, 0, 1>();
    __tx_vec_shuffle_xor<1, 0>(0);
    __tx_vec_lsu_u2s<0, 3>(uram_address(kChunkLower));
    __tx_vec_lsu_u2v<0, 0>(uram_address(kChunkKK));
    __tx_vec_lsu_u2v<0, 1>(uram_address(chunk.input + kInputBeta));
    __tx_vec_vv_fp32_mul<0, 0, 1, 0>();
    __tx_vec_vv_fp32_mul<0, 0, 16, 0>();
    __tx_vec_vv_mux2<0, 17, 0, 0>();
    __tx_vec_lsu_v2u<0, 0>(uram_address(kChunkKK));
    __tx_vec_lsu_u2s<0, 3>(uram_address(kChunkDiag));
    __tx_vec_lsu_u2v<0, 0>(uram_address(kChunkKQ));
    __tx_vec_vv_fp32_mul<0, 0, 16, 0>();
    __tx_vec_lsu_g2v_broadcast32<0, 0>(scale_bits);
    __tx_vec_vv_fp32_mul<0, 0, 1, 0>();
    __tx_vec_vv_mux2<0, 17, 0, 0>();
    __tx_vec_lsu_v2u<0, 0>(uram_address(kChunkKQ));
    start_vec_block(instructions, instructions, instructions, instructions, instructions);
}

static inline void issue_chunk_rhs(const GdnChunk & chunk) {
    configure_vec_outer(kHeadDim);
    configure_vec_iterator<kVecV0Base>(0, 0, kChunkStride, 0);
    configure_vec_iterator<kVecV1Base>(0, 0, 0, 0);
    for (long operand = 0; operand < 2; ++operand) {
        const long instructions = operand == 0 ? 4 : 6;
        const uint64_t address = uram_address(chunk.prepared + operand * kHeadDim * kChunkStride);
        begin_vec_block(instructions);
        __tx_vec_lsu_u2v<1, 1>(uram_address(chunk.input + kInputBeta));
        if (operand != 0) {
            __tx_vec_lsu_u2v<1, 1>(uram_address(chunk.prepared + kPreparedG));
            __tx_vec_vv_fp32_mul<1, 16, 17, 0>();
        }
        __tx_vec_lsu_u2v<0, 0>(address);
        __tx_vec_vv_fp32_mul<0, 0, 16, 0>();
        __tx_vec_lsu_v2u<0, 0>(address);
        start_vec_block(instructions, instructions, instructions, instructions, instructions);
        dayan_wait_vec();
    }
}

static inline void issue_chunk_identity(long output) {
    constexpr long instructions = 5;
    configure_vec_outer(kChunkTokens);
    configure_vec_iterator<kVecV0Base>(0, 0, 0, 0);
    configure_vec_iterator<kVecV1Base>(0, 0, 0, 0);
    configure_vec_iterator<kVecV2Base>(0, 0, kScalarStride, 0);
    configure_vec_iterator<kVecV3Base>(0, 0, kChunkStride, 0);
    begin_vec_block(instructions);
    __tx_vec_lsu_u2s<0, 2>(uram_address(kChunkOneHot));
    __tx_vec_lsu_g2v_broadcast32<0, 0>(0x3f800000U);
    __tx_vec_lsu_g2v_broadcast32<1, 0>(0);
    __tx_vec_vv_mux2<0, 16, 0, 0>();
    __tx_vec_lsu_v2u<0, 3>(uram_address(output));
    start_vec_block(instructions, instructions, instructions, instructions, instructions);
}

static inline void issue_chunk_inverse(const GdnChunk & chunk, long output) {
    if (chunk.rows <= 1) {
        return;
    }
    configure_vec_outer(kChunkTokens);
    __tx_vec_tcsr_write<kVecDim0>(chunk.rows - 2);
    configure_vec_iterator<kVecS0Base>(1, 0, 0, 0);
    configure_vec_iterator<kVecV0Base>(kChunkStride, 0, 0, 0);
    configure_vec_iterator<kVecV1Base>(0, 0, kChunkStride, 0);
    begin_vec_block(7);
    __tx_vec_lsu_u2v<1, 1>(uram_address(output));
    __tx_vec_s_iter_gets<0, 0, 0, 0>();
    __tx_vec_shuffle_broadcast_dyn<0, 16, 0>();
    __tx_vec_lsu_u2v<0, 0>(uram_address(kChunkKK));
    __tx_vec_vv_fp32_mul<0, 0, 1, 0>();
    __tx_vec_vv_fp32_sub<1, 16, 0, 0>();
    __tx_vec_lsu_v2u<16, 1>(uram_address(output));
    start_vec_block(1, 1, 6, 6, 7);
}

static inline void solve_chunk_mat(const GdnChunk & chunk, long & checksum) {
    const long inverse = chunk.prepared + kPreparedW;

    // Invert with 64 VEC RHS rows, then apply it to 256 RHS rows with MAT.
    issue_chunk_identity(inverse);
    dayan_wait_vec();
    issue_chunk_inverse(chunk, inverse);
    dayan_wait_vec();
    set_lsu_matrix<0, true, 2>(checksum, kChunkTokens, kChunkTokens * sizeof(float), kChunkStride, kChunkStride);
    __tx_lsu_dma_u2u<0>(uram_address(kChunkKQ), uram_address(inverse));
    dayan_wait_lsu();

    const long blocks = (chunk.rows + kSolveBlockTokens - 1) / kSolveBlockTokens;
    // Write high columns first so later blocks do not read replaced RHS columns.
    for (long block = blocks; block-- > 0;) {
        const long column = block * kSolveBlockTokens;
        const long columns = chunk.rows - column < kSolveBlockTokens ? chunk.rows - column : kSolveBlockTokens;
        const long rhs = chunk.prepared + kPreparedU + column * sizeof(float);

        configure_chunk_mat(2 * kHeadDim, columns, column + columns,
                            kChunkStride, kChunkStride, kChunkStride);
        __tx_mat_mm<0>(uram_address(rhs), uram_address(chunk.prepared + kPreparedU),
                       uram_address(kChunkKQ + column * kChunkStride));
    }
}

static inline void issue_chunk_residual(long prepared, uint32_t scale_bits) {
    constexpr long instructions = 10;
    configure_vec_outer(kHeadDim);
    configure_vec_iterator<kVecV0Base>(0, 0, kChunkStride, 0);
    configure_vec_iterator<kVecV1Base>(0, 0, 0, 0);
    begin_vec_block(instructions);
    __tx_vec_lsu_u2v<1, 1>(uram_address(prepared + kPreparedG));
    __tx_vec_lsu_u2v<0, 0>(uram_address(kChunkR));
    __tx_vec_lsu_u2v<0, 0>(uram_address(prepared + kPreparedU));
    __tx_vec_vv_fp32_sub<0, 0, 1, 0>();
    __tx_vec_lsu_v2u<0, 0>(uram_address(kChunkR));
    __tx_vec_lsu_u2v<0, 0>(uram_address(kChunkO));
    __tx_vec_vv_fp32_mul<0, 0, 16, 0>();
    __tx_vec_lsu_g2v_broadcast32<0, 0>(scale_bits);
    __tx_vec_vv_fp32_mul<0, 0, 1, 0>();
    __tx_vec_lsu_v2u<0, 0>(uram_address(kChunkO));
    start_vec_block(instructions, instructions, instructions, instructions, instructions);
}

#if !GDN_SOFTWARE_PIPELINE
static inline void load_one(const GdnPipeline & pipeline, const GdnChunk & chunk) {
    const auto * config = pipeline.config;
    if (chunk.rows != kChunkTokens) {
        configure_vec_outer(kInputBytes / kVecBytes);
        configure_vec_iterator<kVecV0Base>(0, 0, kVecBytes, 0);
        begin_vec_block(2);
        __tx_vec_lsu_g2v_broadcast32<0, 0>(0);
        __tx_vec_lsu_v2u<0, 0>(uram_address(chunk.input));
        start_vec_block(2, 2, 2, 2, 2);
        dayan_wait_vec();
    }

    const uint32_t q_sequence = chunk.sequence / (config->sequences / config->q_sequences);
    const uint32_t k_sequence = chunk.sequence / (config->sequences / config->k_sequences);
    configure_lsu_rows<0>(chunk.rows, kHeadDim * sizeof(float), config->q_nb[2], kHeadStride);
    load_gdn_pair(uram_address(chunk.input + kInputQ), uram_address(chunk.input + kInputK),
                   gdn_row(pipeline.q, config->q_nb, chunk.head % config->q_heads, chunk.token, q_sequence),
                   gdn_row(pipeline.k, config->k_nb, chunk.head % config->k_heads, chunk.token, k_sequence),
                   config->q_nb[2], config->k_nb[2]);
    configure_lsu_rows<0>(chunk.rows, kHeadDim * sizeof(float), config->v_nb[2], kHeadStride);
    __tx_lsu_dma_d2u<0>(uram_address(chunk.input + kInputV),
                        (uint64_t) gdn_row(pipeline.v, config->v_nb, chunk.head, chunk.token, chunk.sequence));
    configure_lsu_rows<0>(chunk.rows, 64, config->g_nb[2], sizeof(float));
    __tx_lsu_tcsr_write<LSU_REQUESTREADSIZE>(sizeof(float) - 1);
    __tx_lsu_tcsr_write<LSU_REQUESTWRITESIZE>(sizeof(float) - 1);
    load_gdn_pair(uram_address(chunk.input + kInputG), uram_address(chunk.input + kInputBeta),
                   gdn_row(pipeline.g, config->g_nb, chunk.head, chunk.token, chunk.sequence),
                   gdn_row(pipeline.beta, config->beta_nb, chunk.head, chunk.token, chunk.sequence),
                   config->g_nb[2], config->beta_nb[2]);
}
#endif

template <bool SignalInput>
static inline void prepare_one(const GdnPipeline & pipeline, const GdnChunk & chunk, uint64_t input, long & checksum) {
    // Materialize all zero-padded columns when a slot follows a full chunk.
    configure_chunk_mat(kChunkTokens, kChunkTokens, kHeadDim, kHeadStride, kHeadStride, kChunkStride);
    __tx_mat_mm<0>(uram_address(kChunkKK), uram_address(chunk.input + kInputK), uram_address(chunk.input + kInputK));
    __tx_mat_mm<0>(uram_address(kChunkKQ), uram_address(chunk.input + kInputK), uram_address(chunk.input + kInputQ));
    set_lsu_matrix<0, true, 2>(checksum, kChunkTokens, kHeadDim * sizeof(float), kHeadStride, kChunkStride);
    __tx_lsu_dma_u2u<0>(uram_address(chunk.prepared + kPreparedU), uram_address(chunk.input + kInputV));
    __tx_lsu_dma_u2u<0>(uram_address(chunk.prepared + kPreparedKT), uram_address(chunk.input + kInputK));
    configure_lsu_rows<0>(chunk.rows, kHeadDim * sizeof(float), kHeadStride, kHeadStride);
    __tx_lsu_dma_u2u<0>(uram_address(chunk.prepared + kPreparedQ), uram_address(chunk.input + kInputQ));
    issue_chunk_prefix(chunk);
    dayan_wait_mat();
    dayan_wait_lsu();
    dayan_wait_vec();

    issue_chunk_coefficients(chunk, pipeline.config->scale_bits);
    dayan_wait_vec();
    set_lsu_matrix<0, true, 2>(checksum, kChunkTokens, kChunkTokens * sizeof(float), kChunkStride, kChunkStride);
    __tx_lsu_dma_u2u<0>(uram_address(chunk.prepared + kPreparedA), uram_address(kChunkKQ));
    issue_chunk_rhs(chunk);
    dayan_wait_lsu();
    solve_chunk_mat(chunk, checksum);
    dayan_wait_mat();

    set_lsu_matrix<0, true, 2>(checksum, kHeadDim, kChunkTokens * sizeof(float), kChunkStride, kHeadStride);
    __tx_lsu_dma_u2u<0>(uram_address(chunk.prepared + kPreparedW), uram_address(chunk.prepared + kPreparedKT));
    set_lsu_matrix<0, true, 2>(checksum, kChunkTokens, kHeadDim * sizeof(float), kHeadStride, kChunkStride);
    __tx_lsu_dma_u2u<0>(uram_address(chunk.prepared + kPreparedKT), uram_address(chunk.input + kInputK));
    if constexpr (SignalInput) {
        gdn_check(dayan_resource_consume_lsu(input));
    }
    dayan_wait_lsu();

    configure_vec_outer(kHeadDim);
    configure_vec_iterator<kVecV0Base>(0, 0, kChunkStride, 0);
    configure_vec_iterator<kVecV1Base>(0, 0, 0, 0);
    begin_vec_block(4);
    __tx_vec_lsu_u2v<1, 1>(uram_address(kChunkTailDecay));
    __tx_vec_lsu_u2v<0, 0>(uram_address(chunk.prepared + kPreparedKT));
    __tx_vec_vv_fp32_mul<0, 0, 16, 0>();
    __tx_vec_lsu_v2u<0, 0>(uram_address(chunk.prepared + kPreparedKT));
    start_vec_block(4, 4, 4, 4, 4);
}

#if !GDN_SOFTWARE_PIPELINE
static void load_chunks(void * opaque) {
    const auto & pipeline = *static_cast<const GdnPipeline *>(opaque);
    for (uint64_t generation = 0; generation < pipeline.generations; ++generation) {
        const auto resource = DAYAN_LOCAL_RESOURCE(generation & 1);
        gdn_check(dayan_resource_wait_free(resource));
        load_one(pipeline, gdn_chunk(pipeline, generation));
        gdn_check(dayan_resource_produce_lsu(resource, 1, 0));
    }
    dayan_wait_lsu();
}

static void prepare_chunks(void * opaque) {
    const auto & pipeline = *static_cast<const GdnPipeline *>(opaque);
    uint64_t phases = 0;
    long checksum = 0;
    initialize_chunk_mat();
    for (uint64_t generation = 0; generation < pipeline.generations; ++generation) {
        const auto input = DAYAN_RESOURCE_MASK(DAYAN_LOCAL_RESOURCE(generation & 1));
        const auto prepared = DAYAN_LOCAL_RESOURCE(2 + (generation & 1));
        gdn_check(dayan_resource_wait(DAYAN_RESOURCE_MASK(prepared), input, phases, 0));
        prepare_one<true>(pipeline, gdn_chunk(pipeline, generation), input, checksum);
        phases ^= input;
        gdn_check(dayan_resource_produce_vec(prepared, 1, 0));
    }
    dayan_wait_vec();
}
#endif

static long run_chunked(const DayanGatedDeltaNetConfig * config, const float * q, const float * k,
                        const float * v, const float * g, const float * beta, const float * state, float * output) {
    if (dayan_uram_size() < kGdnUramBytes) {
        return -22;
    }
    const bool four_core = dayan_task_mode() == 4;
    if ((!four_core && dayan_task_mode() != 1) || !dayan_task_dim(DAYAN_TASK_X) ||
        (four_core ? dayan_task_dim(DAYAN_TASK_X) != 4 : dayan_task_dim(DAYAN_TASK_Y) != 1) ||
        !dayan_task_dim(DAYAN_TASK_Y) || dayan_task_dim(DAYAN_TASK_Z) != 1) return -22;
    const uint32_t task = dayan_task_id(DAYAN_TASK_X) +
                          (four_core ? 4 * dayan_task_id(DAYAN_TASK_Y) : 0);
    const uint32_t tasks = dayan_task_dim(DAYAN_TASK_X) * dayan_task_dim(DAYAN_TASK_Y);
    const uint64_t work_items = (uint64_t) config->heads * config->sequences;
    const uint64_t chunks = ((uint64_t) config->tokens + kChunkTokens - 1) / kChunkTokens;
    GdnPipeline pipeline = { config, q, k, v, g, beta, chunks,
                             task < work_items ? (work_items - task + tasks - 1) / tasks * chunks : 0, task, tasks };
    // Idle lanes still reach both barriers in the kernel wrapper.
    if (!pipeline.generations) return 0;
    for (long j = 0; j < kChunkTokens; ++j) {
        *uram_pointer<uint64_t>(kChunkLower + j * kScalarStride) = j == 63 ? 0 : ~uint64_t{0} << (j + 1);
        *uram_pointer<uint64_t>(kChunkDiag + j * kScalarStride) = ~uint64_t{0} << j;
        *uram_pointer<uint64_t>(kChunkOneHot + j * kScalarStride) = uint64_t{1} << j;
    }

#if !GDN_SOFTWARE_PIPELINE
    dayan_thread_resources_t resources = {};
    resources.lsu_tcsr_count = 1;
    resources.vec_tcsr_count = 1;
    resources.local_resource_count = 4;
    gdn_check(dayan_create_thread("gdn-load", load_chunks, &pipeline, &resources));
    resources.mat_tcsr_count = 1;
    gdn_check(dayan_create_thread("gdn-prepare", prepare_chunks, &pipeline, &resources));
#endif
    initialize_chunk_mat();

#if !GDN_SOFTWARE_PIPELINE
    uint64_t phases = 0;
#endif
    long checksum = 0;
    const uint64_t attn_count = (uint64_t) config->sequences * config->tokens * config->heads * kHeadDim;
#if GDN_SOFTWARE_PIPELINE
    // Prologue: the same two input/prepared slots, one VEC context.
    // Every VEC block is drained before submitting the next one. With
    // GDN_VEC_TGMT=0, this also limits the whole kernel to one VEC task.
    {
        const GdnChunk chunk = gdn_chunk(pipeline, 0);
        const auto * config = pipeline.config;
        if (chunk.rows != kChunkTokens) {
            configure_vec_outer(kInputBytes / kVecBytes);
            configure_vec_iterator<kVecV0Base>(0, 0, kVecBytes, 0);
            begin_vec_block(2);
            __tx_vec_lsu_g2v_broadcast32<0, 0>(0);
            __tx_vec_lsu_v2u<0, 0>(uram_address(chunk.input));
            start_vec_block(2, 2, 2, 2, 2);
            dayan_wait_vec();
        }

        const uint32_t q_sequence = chunk.sequence / (config->sequences / config->q_sequences);
        const uint32_t k_sequence = chunk.sequence / (config->sequences / config->k_sequences);
        configure_lsu_rows<0>(chunk.rows, kHeadDim * sizeof(float), config->q_nb[2], kHeadStride);
        {
            const uint64_t destination0 = uram_address(chunk.input + kInputQ);
            const uint64_t destination1 = uram_address(chunk.input + kInputK);
            const float * source0 = gdn_row(pipeline.q, config->q_nb, chunk.head % config->q_heads, chunk.token, q_sequence);
            const float * source1 = gdn_row(pipeline.k, config->k_nb, chunk.head % config->k_heads, chunk.token, k_sequence);
            const uint64_t stride0 = config->q_nb[2];
            const uint64_t stride1 = config->k_nb[2];
            const uint64_t source_delta = (uint64_t) source1 - (uint64_t) source0;
            // DIM2 joins two tensors with equal row strides; hardware strides are 28 bits.
            if (stride0 == stride1 && source_delta < (uint64_t{1} << 28)) {
                __tx_lsu_tcsr_write<LSU_DIM2>(1);
                __tx_lsu_tcsr_write<LSU_INDIM2STRIDE>(source_delta);
                __tx_lsu_tcsr_write<LSU_OUTDIM2STRIDE>(destination1 - destination0);
                __tx_lsu_dma_d2u<0>(destination0, (uint64_t) source0);
            } else {
                __tx_lsu_tcsr_write<LSU_DIM2>(0);
                __tx_lsu_dma_d2u<0>(destination0, (uint64_t) source0);
                __tx_lsu_tcsr_write<LSU_INDIM1STRIDE>(stride1);
                __tx_lsu_dma_d2u<0>(destination1, (uint64_t) source1);
            }
        }
        configure_lsu_rows<0>(chunk.rows, kHeadDim * sizeof(float), config->v_nb[2], kHeadStride);
        __tx_lsu_dma_d2u<0>(uram_address(chunk.input + kInputV),
                            (uint64_t) gdn_row(pipeline.v, config->v_nb, chunk.head, chunk.token, chunk.sequence));
        configure_lsu_rows<0>(chunk.rows, 64, config->g_nb[2], sizeof(float));
        __tx_lsu_tcsr_write<LSU_REQUESTREADSIZE>(sizeof(float) - 1);
        __tx_lsu_tcsr_write<LSU_REQUESTWRITESIZE>(sizeof(float) - 1);
        {
            const uint64_t destination0 = uram_address(chunk.input + kInputG);
            const uint64_t destination1 = uram_address(chunk.input + kInputBeta);
            const float * source0 = gdn_row(pipeline.g, config->g_nb, chunk.head, chunk.token, chunk.sequence);
            const float * source1 = gdn_row(pipeline.beta, config->beta_nb, chunk.head, chunk.token, chunk.sequence);
            const uint64_t stride0 = config->g_nb[2];
            const uint64_t stride1 = config->beta_nb[2];
            const uint64_t source_delta = (uint64_t) source1 - (uint64_t) source0;
            // DIM2 joins two tensors with equal row strides; hardware strides are 28 bits.
            if (stride0 == stride1 && source_delta < (uint64_t{1} << 28)) {
                __tx_lsu_tcsr_write<LSU_DIM2>(1);
                __tx_lsu_tcsr_write<LSU_INDIM2STRIDE>(source_delta);
                __tx_lsu_tcsr_write<LSU_OUTDIM2STRIDE>(destination1 - destination0);
                __tx_lsu_dma_d2u<0>(destination0, (uint64_t) source0);
            } else {
                __tx_lsu_tcsr_write<LSU_DIM2>(0);
                __tx_lsu_dma_d2u<0>(destination0, (uint64_t) source0);
                __tx_lsu_tcsr_write<LSU_INDIM1STRIDE>(stride1);
                __tx_lsu_dma_d2u<0>(destination1, (uint64_t) source1);
            }
        }
    }
    dayan_wait_lsu();
    prepare_one<false>(pipeline, gdn_chunk(pipeline, 0), 0, checksum);
    dayan_wait_vec();
#endif
    for (uint64_t generation = 0; generation < pipeline.generations; ++generation) {
        const GdnChunk chunk = gdn_chunk(pipeline, generation);
#if !GDN_SOFTWARE_PIPELINE
        const auto input = DAYAN_RESOURCE_MASK(DAYAN_LOCAL_RESOURCE(2 + (generation & 1)));
#endif
        const uint64_t state_index = ((uint64_t) chunk.sequence * config->heads + chunk.head) * kHeadDim * kHeadDim;
        if (chunk.token == 0) {
            configure_lsu_rows<0>(kHeadDim, kHeadDim * sizeof(float), kHeadDim * sizeof(float), kHeadStride);
            __tx_lsu_dma_d2u<0>(uram_address(kChunkState), (uint64_t) (state + state_index));
            dayan_wait_lsu();
        }
#if !GDN_SOFTWARE_PIPELINE
        gdn_check(dayan_resource_wait(0, input, phases, 0));
#endif

        configure_chunk_mat(kHeadDim, kChunkTokens, kHeadDim, kHeadStride, kHeadStride, kChunkStride);
        __tx_mat_mm<0>(uram_address(kChunkR), uram_address(kChunkState), uram_address(chunk.prepared + kPreparedW));
        __tx_mat_mm<0>(uram_address(kChunkO), uram_address(kChunkState), uram_address(chunk.prepared + kPreparedQ));
#if GDN_SOFTWARE_PIPELINE
        // Prefetch the next input slot while current state projections run.
        if (generation + 1 < pipeline.generations) {
            const GdnChunk chunk = gdn_chunk(pipeline, generation + 1);
            const auto * config = pipeline.config;
            if (chunk.rows != kChunkTokens) {
                configure_vec_outer(kInputBytes / kVecBytes);
                configure_vec_iterator<kVecV0Base>(0, 0, kVecBytes, 0);
                begin_vec_block(2);
                __tx_vec_lsu_g2v_broadcast32<0, 0>(0);
                __tx_vec_lsu_v2u<0, 0>(uram_address(chunk.input));
                start_vec_block(2, 2, 2, 2, 2);
                dayan_wait_vec();
            }

            const uint32_t q_sequence = chunk.sequence / (config->sequences / config->q_sequences);
            const uint32_t k_sequence = chunk.sequence / (config->sequences / config->k_sequences);
            configure_lsu_rows<0>(chunk.rows, kHeadDim * sizeof(float), config->q_nb[2], kHeadStride);
            {
                const uint64_t destination0 = uram_address(chunk.input + kInputQ);
                const uint64_t destination1 = uram_address(chunk.input + kInputK);
                const float * source0 = gdn_row(pipeline.q, config->q_nb, chunk.head % config->q_heads, chunk.token, q_sequence);
                const float * source1 = gdn_row(pipeline.k, config->k_nb, chunk.head % config->k_heads, chunk.token, k_sequence);
                const uint64_t stride0 = config->q_nb[2];
                const uint64_t stride1 = config->k_nb[2];
                const uint64_t source_delta = (uint64_t) source1 - (uint64_t) source0;
                // DIM2 joins two tensors with equal row strides; hardware strides are 28 bits.
                if (stride0 == stride1 && source_delta < (uint64_t{1} << 28)) {
                    __tx_lsu_tcsr_write<LSU_DIM2>(1);
                    __tx_lsu_tcsr_write<LSU_INDIM2STRIDE>(source_delta);
                    __tx_lsu_tcsr_write<LSU_OUTDIM2STRIDE>(destination1 - destination0);
                    __tx_lsu_dma_d2u<0>(destination0, (uint64_t) source0);
                } else {
                    __tx_lsu_tcsr_write<LSU_DIM2>(0);
                    __tx_lsu_dma_d2u<0>(destination0, (uint64_t) source0);
                    __tx_lsu_tcsr_write<LSU_INDIM1STRIDE>(stride1);
                    __tx_lsu_dma_d2u<0>(destination1, (uint64_t) source1);
                }
            }
            configure_lsu_rows<0>(chunk.rows, kHeadDim * sizeof(float), config->v_nb[2], kHeadStride);
            __tx_lsu_dma_d2u<0>(uram_address(chunk.input + kInputV),
                                (uint64_t) gdn_row(pipeline.v, config->v_nb, chunk.head, chunk.token, chunk.sequence));
            configure_lsu_rows<0>(chunk.rows, 64, config->g_nb[2], sizeof(float));
            __tx_lsu_tcsr_write<LSU_REQUESTREADSIZE>(sizeof(float) - 1);
            __tx_lsu_tcsr_write<LSU_REQUESTWRITESIZE>(sizeof(float) - 1);
            {
                const uint64_t destination0 = uram_address(chunk.input + kInputG);
                const uint64_t destination1 = uram_address(chunk.input + kInputBeta);
                const float * source0 = gdn_row(pipeline.g, config->g_nb, chunk.head, chunk.token, chunk.sequence);
                const float * source1 = gdn_row(pipeline.beta, config->beta_nb, chunk.head, chunk.token, chunk.sequence);
                const uint64_t stride0 = config->g_nb[2];
                const uint64_t stride1 = config->beta_nb[2];
                const uint64_t source_delta = (uint64_t) source1 - (uint64_t) source0;
                // DIM2 joins two tensors with equal row strides; hardware strides are 28 bits.
                if (stride0 == stride1 && source_delta < (uint64_t{1} << 28)) {
                    __tx_lsu_tcsr_write<LSU_DIM2>(1);
                    __tx_lsu_tcsr_write<LSU_INDIM2STRIDE>(source_delta);
                    __tx_lsu_tcsr_write<LSU_OUTDIM2STRIDE>(destination1 - destination0);
                    __tx_lsu_dma_d2u<0>(destination0, (uint64_t) source0);
                } else {
                    __tx_lsu_tcsr_write<LSU_DIM2>(0);
                    __tx_lsu_dma_d2u<0>(destination0, (uint64_t) source0);
                    __tx_lsu_tcsr_write<LSU_INDIM1STRIDE>(stride1);
                    __tx_lsu_dma_d2u<0>(destination1, (uint64_t) source1);
                }
            }
        }
#endif
        dayan_wait_mat();
        issue_chunk_residual(chunk.prepared, config->scale_bits);
        dayan_wait_vec();
        configure_chunk_mat(kHeadDim, kChunkTokens, kChunkTokens, kChunkStride, kChunkStride, kChunkStride);
        __tx_mat_mma<0>(uram_address(kChunkO), uram_address(kChunkR), uram_address(chunk.prepared + kPreparedA));

#if GDN_VEC_TGMT
        configure_vec_outer(kRowGroups);
        __tx_vec_tcsr_write<kVecDim3>(kHeadDim - 1);
        configure_vec_iterator<kVecV0Base>(0, 0, kVecBytes, kHeadStride);
#else
        configure_vec_outer(kHeadDim);
        __tx_vec_tcsr_write<kVecDim0>(kRowGroups - 1);
        configure_vec_iterator<kVecV0Base>(kVecBytes, 0, kHeadStride, 0);
#endif
        configure_vec_iterator<kVecV1Base>(0, 0, 0, 0);
        begin_vec_block(4);
        __tx_vec_lsu_u2v<0, 0>(uram_address(kChunkState));
        __tx_vec_lsu_u2v_broadcast32<1, 1>(uram_address(chunk.prepared + kPreparedDecay));
        __tx_vec_vv_fp32_mul<0, 0, 16, 0>();
        __tx_vec_lsu_v2u<0, 0>(uram_address(kChunkState));
#if GDN_VEC_TGMT
        start_vec_block(4, 4, 4, 4, 4);
#else
        // DIM1 selects the state row; DIM0 visits its two 64-lane halves.
        start_vec_block(0, 0, 4, 4, 4);
#endif
        dayan_wait_mat();
        dayan_wait_vec();

        configure_chunk_mat(kHeadDim, kHeadDim, kChunkTokens, kChunkStride, kChunkStride, kHeadStride);
        __tx_mat_mma<0>(uram_address(kChunkState), uram_address(kChunkR), uram_address(chunk.prepared + kPreparedKT));
        // Q is no longer read; reuse its slot for the transposed output.
        set_lsu_matrix<0, true, 2>(checksum, kHeadDim, kChunkTokens * sizeof(float), kChunkStride, kHeadStride);
        __tx_lsu_dma_u2u<0>(uram_address(chunk.prepared + kPreparedQ), uram_address(kChunkO));
        dayan_wait_lsu();
        const uint64_t value_index = (((uint64_t) chunk.sequence * config->tokens + chunk.token) * config->heads + chunk.head) * kHeadDim;
        configure_lsu_rows<0>(chunk.rows, kHeadDim * sizeof(float), kHeadStride, (uint64_t) config->heads * kHeadDim * sizeof(float));
        __tx_lsu_dma_u2d<0>((uint64_t) (output + value_index), uram_address(chunk.prepared + kPreparedQ));
#if !GDN_SOFTWARE_PIPELINE
        dayan_wait_mat();
#endif
        if (chunk.token + chunk.rows == config->tokens) {
#if GDN_SOFTWARE_PIPELINE
            dayan_wait_mat();
#endif
            configure_lsu_rows<0>(kHeadDim, kHeadDim * sizeof(float), kHeadStride, kHeadDim * sizeof(float));
            __tx_lsu_dma_u2d<0>((uint64_t) (output + attn_count + state_index), uram_address(kChunkState));
        }
#if GDN_SOFTWARE_PIPELINE
        // LSU -> next Prepare: finish its prefetched inputs and current
        // output stores before slot reuse. wait_mat() does not cover LSU.
        // Current state MAT update may still overlap the next Prepare.
        dayan_wait_lsu();
        if (generation + 1 < pipeline.generations) {
            prepare_one<false>(pipeline, gdn_chunk(pipeline, generation + 1), 0, checksum);
            dayan_wait_vec();
        }
#else
        gdn_check(dayan_resource_consume_lsu(input));
        phases ^= input;
#endif
    }
    dayan_wait_lsu();
#if GDN_SOFTWARE_PIPELINE
    dayan_wait_mat();
    return 0;
#else
    return dayan_thread_join_all();
#endif
}

} // namespace gdn_detail
#endif
