#include <dayan_kernel.h>

#if defined(__DAYAN_DEVICE__)
#include <dayan_intrinsics.h>
#include <dayan_sys.h>
#endif

#include <stddef.h>
#include <stdint.h>

/*
 * Fixed FlashAttention pipeline:
 *
 *   FP16: Q[128][128] x KT[4][256][128] x VT[4][128][256]
 *       -> O[128][128] FP32
 *   FP8: Q[128][128] x KT[4][256][128]
 *       -> P[4][128][256] FP32, final M/L[128]
 *
 * The KV-tile loop is outermost, so each KT tile is loaded once and consumed
 * by one 128-row MAT command. FlexVec then processes the score through 16
 * resident eight-row tasks. KT is stored by key row. Q must include the
 * attention scale and log2(e), because the FlexVec flow evaluates
 * exp2(score - newM). The FP16 path converts P to FP16, accumulates P*V into
 * an FP32 O tile, then normalizes O by L. M and L use the physical Vec layout:
 * eight copies of each row scalar.
 */
enum : uint64_t {
    kAttentionRows = 8,
    kHeadD = 128,
    kKvTileCols = 256,
    kContextLength = 1024,
    kHardwareVecGroups = 2,
    kVecThreadsPerGroup = 8,
    kVecTaskCount = kHardwareVecGroups * kVecThreadsPerGroup,
    kQueryBlocks = kVecTaskCount,
    kQueryRows = kQueryBlocks * kAttentionRows,
    kKvTiles = kContextLength / kKvTileCols,
    kKtBufferCount = 2,
    kScoreBufferCount = 3,
    kStateLanesPerRow = 8,
    kQTileElements = kAttentionRows * kHeadD,
    kQElements = kQueryBlocks * kQTileElements,
    kKtTileElements = kKvTileCols * kHeadD,
    kKtElements = kKvTiles * kKtTileElements,
    kVtTileElements = kHeadD * kKvTileCols,
    kVtElements = kKvTiles * kVtTileElements,
    kProbabilityTaskElements = kAttentionRows * kKvTileCols,
    kScoreTaskBytes = kProbabilityTaskElements * sizeof(float),
    kProbabilityElements =
        kKvTiles * kQueryBlocks * kProbabilityTaskElements,
    kProbabilityBytes = kProbabilityElements * sizeof(float),
    kStateTileElements = kAttentionRows * kStateLanesPerRow,
    kStateTileBytes = kStateTileElements * sizeof(float),
    kStateElements = kQueryBlocks * kStateTileElements,
    kStateBytes = kStateElements * sizeof(float),
    kOutputElements = kQueryRows * kHeadD,
    kOutputBytes = kOutputElements * sizeof(float),
    kOutputTaskBytes = kAttentionRows * kHeadD * sizeof(float),
    kOutputRowBytes = kHeadD * sizeof(float),
    kP16TaskBytes = kAttentionRows * kKvTileCols * sizeof(uint16_t),
    kP16TileBytes = kQueryRows * kKvTileCols * sizeof(uint16_t),
    kFp8UramBytes = 480 * 1024,
    kFp16UramBytes = 960 * 1024,
};

enum : uint64_t {
    kFlashAttentionPerfMagic = 0x4641314350455246ULL,
};

struct FlashAttentionPerf {
    uint64_t magic;
    uint64_t kernel_cycles;
    uint64_t pipeline_cycles;
};

#if defined(__DAYAN_DEVICE__)

namespace flash_attention_1c_detail {

__device__ constexpr uint64_t kLinearSwizzle = MAT_BATCH_SWIZZLE(5, 0, 1);
__device__ constexpr uint64_t kFp16WideSwizzle = MAT_BATCH_SWIZZLE(7, 2, 3);

template <typename Element, bool FullAttention, uint64_t InputSwizzle,
          uint64_t InputDataType, uint64_t InputReorderWindowBytes,
          uint64_t UramBytes>
struct PipelineTraits {
    using Input = Element;

    enum : uint64_t {
        kFullAttention = FullAttention,
        kInputSwizzle = InputSwizzle,
        kInputDataType = InputDataType,
        kInputReorderWindowBytes = InputReorderWindowBytes,
        kUramBytes = UramBytes,
        kQTileBytes = kQTileElements * sizeof(Input),
        kQBytes = kQElements * sizeof(Input),
        kKtTileBytes = kKtTileElements * sizeof(Input),
        kKtBytes = kKvTiles * kKtTileBytes,
        kVtTileBytes = kVtTileElements * sizeof(Input),
        kVtBytes = kKvTiles * kVtTileBytes,
        kQOffset = 0,
        kKtOffset = kQOffset + kQBytes,
        kVtOffset = kKtOffset + kKtBufferCount * kKtTileBytes,
        kScoreOffset =
            FullAttention ? kVtOffset + kKtBufferCount * kVtTileBytes
                          : kKtOffset + kKtBufferCount * kKtTileBytes,
        kScoreSlotBytes = kQueryRows * kKvTileCols * sizeof(float),
        kP16Offset = kScoreOffset + kScoreBufferCount * kScoreSlotBytes,
        kMOffset =
            FullAttention ? kP16Offset + kScoreBufferCount * kP16TileBytes
                          : kScoreOffset + kScoreBufferCount * kScoreSlotBytes,
        kLOffset = kMOffset + kStateBytes,
        kOutputOffset = kLOffset + kStateBytes,
        kLayoutBytes = FullAttention ? kOutputOffset + kOutputBytes
                                     : kLOffset + kStateBytes,
        kProbabilityConvertInstructionCount =
            2 + kAttentionRows * (kKvTileCols / 64) * 4,
        kVecInstructionCount =
            7 * (kKvTileCols / 8) + 19 +
            (FullAttention ? 3 * (kHeadD / 8) +
                                 kProbabilityConvertInstructionCount
                           : 0),
        kNormalizeInstructionCount = 2 + 3 * (kHeadD / 8),
    };
};

using Fp16Pipeline = PipelineTraits<
    uint16_t, true, MAT_BATCH_SWIZZLE(7, 1, 3),
    BaseNPUInstBuilder::DT_FP16, 256, kFp16UramBytes>;
using Fp8Pipeline = PipelineTraits<
    uint8_t, false, kLinearSwizzle,
    BaseNPUInstBuilder::DT_E4M3, 32, kFp8UramBytes>;

static_assert(Fp16Pipeline::kLayoutBytes <= Fp16Pipeline::kUramBytes,
              "FlashAttention tile exceeds declared URAM");
static_assert(Fp8Pipeline::kLayoutBytes <= Fp8Pipeline::kUramBytes,
              "FlashAttention tile exceeds declared URAM");
static_assert(Fp16Pipeline::kLayoutBytes == 936 * 1024,
              "FP16 FlashAttention URAM layout size changed");
static_assert(Fp8Pipeline::kLayoutBytes == 472 * 1024,
              "FlashAttention URAM layout size changed");
static_assert(kVecTaskCount == 16,
              "two Vec groups need sixteen eight-row tasks");
static_assert(kContextLength % kKvTileCols == 0,
              "context length must contain whole KV tiles");
static_assert(kKvTileCols % 64 == 0 && kHeadD % 8 == 0,
              "Vec flow requires aligned KV and head dimensions");
static_assert(kOutputTaskBytes == kP16TaskBytes,
              "output and P16 task strides must match");
static_assert(Fp16Pipeline::kVecInstructionCount <= 512 &&
                  Fp16Pipeline::kNormalizeInstructionCount <= 512,
              "FP16 FlexVec flow exceeds the instruction queue");
static_assert(Fp16Pipeline::kVecInstructionCount == 421,
              "KV=256 FP16 FlexVec flow instruction count changed");
static_assert(Fp8Pipeline::kVecInstructionCount == 243,
              "KV=256 FP8 FlexVec flow instruction count changed");
static_assert(Fp16Pipeline::kNormalizeInstructionCount == 50,
              "D=128 normalization instruction count changed");

static inline uint64_t read_cycle()
{
    uint64_t cycle;

    __asm__ volatile("csrr %0, mcycle" : "=r"(cycle) : : "memory");
    return cycle;
}

static_assert(Fp16Pipeline::kQOffset %
                          Fp16Pipeline::kInputReorderWindowBytes == 0 &&
                  Fp16Pipeline::kKtOffset %
                          Fp16Pipeline::kInputReorderWindowBytes == 0 &&
                  Fp16Pipeline::kKtTileBytes %
                          Fp16Pipeline::kInputReorderWindowBytes == 0 &&
                  Fp16Pipeline::kVtOffset % 512 == 0 &&
                  Fp16Pipeline::kVtTileBytes % 512 == 0 &&
                  Fp8Pipeline::kQOffset %
                          Fp8Pipeline::kInputReorderWindowBytes == 0 &&
                  Fp8Pipeline::kKtOffset %
                          Fp8Pipeline::kInputReorderWindowBytes == 0 &&
                  Fp8Pipeline::kKtTileBytes %
                          Fp8Pipeline::kInputReorderWindowBytes == 0,
              "input swizzle must stay within Q/KT buffer boundaries");

struct TileBuffer {
    dayan_resource_t resource;
    volatile uint64_t generation;
    uint64_t uram;
};

template <typename Traits>
struct PipelineState {
    uint64_t q_dram;
    uint64_t kt_dram;
    uint64_t m_dram;
    uint64_t l_dram;
    uint64_t vt_dram;
    uint64_t output_dram;
    uint64_t probability_dram;
    uint64_t q_uram;
    uint64_t m_uram;
    uint64_t l_uram;
    uint64_t output_uram;
    TileBuffer kt[kKtBufferCount];
    uint64_t vt_uram[kKtBufferCount];
    TileBuffer score[kScoreBufferCount];
    TileBuffer probability[kScoreBufferCount];
    dayan_resource_t q_ready;
    dayan_resource_t state_ready;
    dayan_resource_t output_for_vec[kKvTiles + 1];
    dayan_resource_t output_for_mat[kKvTiles];
    dayan_resource_t output_done;
    volatile uint32_t start;
    volatile uint32_t abort;
};

static inline void require_status(int status)
{
    if (status < 0)
        dayan_thread_exit(status);
}

template <typename Traits>
static inline int worker_started(const PipelineState<Traits> *state)
{
    while (!state->start)
        dayan_yield();
    return !state->abort;
}

static inline int wait_for_generation(TileBuffer *buffer,
                                      uint64_t generation)
{
    while (buffer->generation != generation)
        dayan_yield();
    return dayan_resource_wait_ready(&buffer->resource);
}

template <long Group>
static inline void configure_lsu(uint64_t write_swizzle,
                                 uint64_t read_swizzle)
{
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_BLOCK>(0);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_DIM2>(0);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_INBLOCKSTRIDE>(0);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_INDIM0STRIDE>(64);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_INDIM2STRIDE>(0);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_OUTBLOCKSTRIDE>(0);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_OUTDIM0STRIDE>(64);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_OUTDIM2STRIDE>(0);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_BLOCKTRANSPOSE>(6);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_BLOCKPADDING>(0);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_REQUESTREADSIZE>(63);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_PADDINGVALUE>(0);
    __tx_lsu_tcsr_write<
        Group * LSU_GROUPSZ + LSU_URAMWRITESWIZZLEMBS>(write_swizzle);
    __tx_lsu_tcsr_write<
        Group * LSU_GROUPSZ + LSU_URAMREADSWIZZLEMBS>(read_swizzle);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_REQUESTWRITESIZE>(63);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_TABLESIZE>(0);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_TABLESTRIDE>(64);
}

template <long Group>
static inline void configure_lsu_shape(uint64_t rows, uint64_t row_bytes,
                                       uint64_t input_stride,
                                       uint64_t output_stride)
{
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_DIM0>(row_bytes / 64U - 1U);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_DIM1>(rows - 1U);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_INDIM1STRIDE>(input_stride);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_OUTDIM1STRIDE>(output_stride);
}

template <long Base> static inline void configure_vec_iterator()
{
    __tx_vec_tcsr_write<Base + 0>(0);
    __tx_vec_tcsr_write<Base + 1>(0);
    __tx_vec_tcsr_write<Base + 2>(0);
    __tx_vec_tcsr_write<Base + 3>(0);
    __tx_vec_tcsr_write<Base + 4>(0);
    __tx_vec_tcsr_write<Base + 5>(0);
    __tx_vec_tcsr_write<Base + 6>(0);
}

template <typename Traits>
static inline void configure_vec()
{
    __tx_vec_tcsr_write<VEC_SYNC>(0);
    __tx_vec_tcsr_write<VEC_DIM0>(0);
    __tx_vec_tcsr_write<VEC_DIM1>(0);
    __tx_vec_tcsr_write<VEC_DIM2>(kVecTaskCount - 1U);
    __tx_vec_tcsr_write<VEC_DIM3>(0);
    __tx_vec_tcsr_write<VEC_DIM4>(0);
    __tx_vec_tcsr_write<VEC_DIM5>(0);
    __tx_vec_tcsr_write<VEC_IMPL_DEF_HINT>(0);
    configure_vec_iterator<VEC_V0_BASE>();
    configure_vec_iterator<VEC_V1_BASE>();
    if constexpr (Traits::kFullAttention) {
        configure_vec_iterator<VEC_V2_BASE>();
        configure_vec_iterator<VEC_V3_BASE>();
    }
    __tx_vec_tcsr_write<VEC_V0_DIM2_STRIDE>(kScoreTaskBytes);
    __tx_vec_tcsr_write<VEC_V1_DIM2_STRIDE>(kStateTileBytes);
    if constexpr (Traits::kFullAttention) {
        __tx_vec_tcsr_write<VEC_V2_DIM2_STRIDE>(kScoreTaskBytes);
        __tx_vec_tcsr_write<VEC_V3_DIM2_STRIDE>(kOutputTaskBytes);
    }
    __tx_vec_tcsr_write<VEC_V0_URAM_CHANNEL_STRIDE>(
        kKvTileCols * sizeof(float));
    __tx_vec_tcsr_write<VEC_V1_URAM_CHANNEL_STRIDE>(32);
    if constexpr (Traits::kFullAttention) {
        __tx_vec_tcsr_write<VEC_V2_URAM_CHANNEL_STRIDE>(32);
        __tx_vec_tcsr_write<VEC_V3_URAM_CHANNEL_STRIDE>(kOutputRowBytes);
    }
}

template <bool ScaleOutput>
static inline void queue_online_softmax(uint64_t score, uint64_t m,
                                        uint64_t l, uint64_t output)
{
    constexpr uint32_t kScoreSlices = kKvTileCols / 8;
    constexpr uint32_t kOutputSlices = kHeadD / 8;

    (void)output;

    /* Scan 16 eight-element slices, retaining one max per lane group. */
    __tx_vec_lsu_u2v<0, 0>((long)score);
    for (uint32_t slice = 1; slice < kScoreSlices; ++slice) {
        __tx_vec_lsu_u2v<0, 0>((long)(score + slice * 32U));
        __tx_vec_vv_fp32_max<0, 1, 0, 0>();
    }
    for (uint32_t amount = 4; amount != 0; amount >>= 1) {
        __tx_vec_shuffle_xor<1, 0>((long)amount);
        __tx_vec_vv_fp32_max<0, 0, 16, 0>();
    }

    __tx_vec_lsu_u2v<1, 1>((long)m);
    __tx_vec_vv_fp32_max<1, 0, 16, 0>();

    /* Re-scan score, write P in place, and retain the running tile sum. */
    for (uint32_t slice = 0; slice < kScoreSlices; ++slice) {
        const uint64_t address = score + slice * 32U;

        __tx_vec_lsu_u2v<0, 0>((long)address);
        __tx_vec_vv_fp32_sub<0, 0, 16, 0>();
        __tx_vec_v_fp32_sfu<0, 0, 1>();
        __tx_vec_lsu_v2u<0, 0>((long)address);
        if (slice != 0)
            __tx_vec_vv_fp32_add<0, 3, 0, 0>();
    }
    for (uint32_t amount = 4; amount != 0; amount >>= 1) {
        __tx_vec_shuffle_xor<1, 0>((long)amount);
        __tx_vec_vv_fp32_add<0, 0, 16, 0>();
    }

    /* oldM/newM are queue-1 ages 4/3 at this point. */
    __tx_vec_vv_fp32_sub<0, 20, 19, 0>();
    __tx_vec_v_fp32_sfu<0, 0, 1>();
    if constexpr (ScaleOutput) {
        /* Commit newM before output scaling reuses queue-1 registers. */
        __tx_vec_lsu_v2u<19, 1>((long)m);
        for (uint32_t slice = 0; slice < kOutputSlices; ++slice) {
            const uint64_t address = output + slice * 32U;

            __tx_vec_lsu_u2v<1, 3>((long)address);
            __tx_vec_vv_fp32_mul<1, 0, 16, 0>();
            __tx_vec_lsu_v2u<16, 3>((long)address);
        }
    }
    __tx_vec_lsu_u2v<1, 1>((long)l);
    __tx_vec_vv_fp32_mul<0, 0, 16, 0>();
    __tx_vec_vv_fp32_add<0, 0, 3, 0>();
    __tx_vec_lsu_v2u<0, 1>((long)l);
    if constexpr (!ScaleOutput)
        __tx_vec_lsu_v2u<20, 1>((long)m);

    /* Dynamic mode consumes the compile-time-checked queued flow above. */
}

static inline void queue_probability_fp16(uint64_t score,
                                          uint64_t probability)
{
    constexpr uint64_t kFp32ChunkBytes = 64 * sizeof(float);
    constexpr uint64_t kFp16ChunkBytes = 64 * sizeof(uint16_t);
    constexpr uint32_t kChunksPerRow = kKvTileCols / 64;

    /* Mask the duplicate high half produced by compress(ctrl=1). */
    __tx_vec_s_bitwise<0, 0, 0, 1>();
    __tx_vec_s_int32_rol<1, 0, 0>();
    for (uint32_t row = 0; row < kAttentionRows; ++row) {
        for (uint32_t chunk = 0; chunk < kChunksPerRow; ++chunk) {
            const uint64_t source =
                score + row * kKvTileCols * sizeof(float) +
                chunk * kFp32ChunkBytes;
            const uint64_t destination =
                probability + row * kKvTileCols * sizeof(uint16_t) +
                chunk * kFp16ChunkBytes;

            __tx_vec_lsu_u2v<0, 2>((long)source);
            __tx_vec_cvt_fp32_fp16<0, 0, 0>();
            __tx_vec_shuffle_compress<0, 0>(0);
            __tx_vec_lsu_v2u_mask<0, 3, 1>((long)destination);
        }
    }
}

static inline void queue_normalize_output(uint64_t output, uint64_t l)
{
    constexpr uint32_t kSlices = kHeadD / 8;

    __tx_vec_lsu_u2v<1, 1>((long)l);
    __tx_vec_v_fp32_sfu<1, 16, 5>();
    for (uint32_t slice = 0; slice < kSlices; ++slice) {
        const uint64_t address = output + slice * 32U;

        __tx_vec_lsu_u2v<0, 3>((long)address);
        __tx_vec_vv_fp32_mul<0, 0, 16, 0>();
        __tx_vec_lsu_v2u<0, 3>((long)address);
    }
}

static inline void start_online_softmax()
{
    __tx_vec_tcsr_write<VEC_CONTROL>(1L << 40);
}

template <long Group, typename Traits>
static inline void configure_mat(uint64_t rows, uint64_t columns,
                                 uint64_t inner, uint64_t a_stride,
                                 uint64_t b_stride, uint64_t c_stride,
                                 uint64_t a_swizzle, uint64_t b_swizzle)
{
    __tx_mat_tcsr_write<Group * MAT_GROUPSZ + MAT_M>(rows - 1U);
    __tx_mat_tcsr_write<Group * MAT_GROUPSZ + MAT_N>(columns - 1U);
    __tx_mat_tcsr_write<Group * MAT_GROUPSZ + MAT_K>(inner - 1U);
    __tx_mat_tcsr_write<Group * MAT_GROUPSZ + MAT_ASTRIDE>(a_stride);
    __tx_mat_tcsr_write<Group * MAT_GROUPSZ + MAT_BSTRIDE>(b_stride);
    __tx_mat_tcsr_write<Group * MAT_GROUPSZ + MAT_CSTRIDE>(c_stride);
    __tx_mat_tcsr_write<Group * MAT_GROUPSZ + MAT_ASWIZZLEMBS>(a_swizzle);
    __tx_mat_tcsr_write<Group * MAT_GROUPSZ + MAT_BSWIZZLEMBS>(b_swizzle);
    __tx_mat_tcsr_write<Group * MAT_GROUPSZ + MAT_CSWIZZLEMBS>(
        kLinearSwizzle);
    __tx_mat_tcsr_write<Group * MAT_GROUPSZ + MAT_PERSISTENTMODE>(0);
    __tx_mat_tcsr_write<Group * MAT_GROUPSZ + MAT_DATATYPE>(
        Traits::kInputDataType);
}

template <typename Traits>
static inline void store_probability(PipelineState<Traits> *state,
                                     uint32_t kv_tile)
{
    TileBuffer *probability =
        &state->probability[kv_tile % kScoreBufferCount];

    require_status(wait_for_generation(probability, kv_tile + 1U));
    __tx_lsu_dma_u2d<1>(
        (long)(state->probability_dram +
               (uint64_t)kv_tile * Traits::kScoreSlotBytes),
        (long)probability->uram);
    require_status(
        dayan_resource_wait_lsu_consume(&probability->resource));
}

template <typename Traits>
static void lsu_worker(void *argument)
{
    PipelineState<Traits> *state = (PipelineState<Traits> *)argument;

    if (!worker_started(state))
        return;

    configure_lsu<0>(Traits::kInputSwizzle, Traits::kInputSwizzle);
    configure_lsu<1>(kLinearSwizzle, kLinearSwizzle);
    if constexpr (Traits::kFullAttention)
        configure_lsu<2>(kFp16WideSwizzle, kFp16WideSwizzle);

    /* Q is loaded once; KT and VT keep fixed shapes for the tile loop. */
    configure_lsu_shape<0>(kQueryRows,
                           kHeadD * sizeof(typename Traits::Input),
                           kHeadD * sizeof(typename Traits::Input),
                           kHeadD * sizeof(typename Traits::Input));
    __tx_lsu_dma_d2u<0>((long)state->q_uram, (long)state->q_dram);
    require_status(dayan_resource_wait_lsu_produce(&state->q_ready));

    configure_lsu_shape<0>(kKvTileCols,
                           kHeadD * sizeof(typename Traits::Input),
                           kHeadD * sizeof(typename Traits::Input),
                           kHeadD * sizeof(typename Traits::Input));
    if constexpr (Traits::kFullAttention) {
        configure_lsu_shape<2>(kHeadD,
                               kKvTileCols * sizeof(typename Traits::Input),
                               kKvTileCols * sizeof(typename Traits::Input),
                               kKvTileCols * sizeof(typename Traits::Input));
    }

    /* M/L are contiguous physical state tiles, each scalar copied 8 times. */
    configure_lsu_shape<1>(1, kStateBytes, kStateBytes, kStateBytes);
    __tx_lsu_dma_d2u<1>((long)state->m_uram, (long)state->m_dram);
    __tx_lsu_dma_d2u<1>((long)state->l_uram, (long)state->l_dram);
    require_status(dayan_resource_wait_lsu_produce(&state->state_ready));

    if constexpr (Traits::kFullAttention) {
        configure_lsu_shape<1>(kQueryRows, kHeadD * sizeof(float),
                               kHeadD * sizeof(float),
                               kHeadD * sizeof(float));
        __tx_lsu_dma_d2u<1>((long)state->output_uram,
                            (long)state->output_dram);
        require_status(
            dayan_resource_wait_lsu_produce(&state->output_for_vec[0]));
    } else {
        configure_lsu_shape<1>(kQueryRows, kKvTileCols * sizeof(float),
                               kKvTileCols * sizeof(float),
                               kKvTileCols * sizeof(float));
    }

    for (uint32_t kv_tile = 0; kv_tile < kKvTiles; ++kv_tile) {
        TileBuffer *kt = &state->kt[kv_tile % kKtBufferCount];

        require_status(dayan_resource_wait_free(&kt->resource));
        require_status(dayan_resource_alloc(
            &kt->resource, 1, Traits::kFullAttention ? 2 : 1));
        kt->generation = kv_tile + 1U;
        __tx_lsu_dma_d2u<0>(
            (long)kt->uram,
            (long)(state->kt_dram +
                   (uint64_t)kv_tile * Traits::kKtTileBytes));
        if constexpr (Traits::kFullAttention) {
            __tx_lsu_dma_d2u<2>(
                (long)state->vt_uram[kv_tile % kKtBufferCount],
                (long)(state->vt_dram +
                       (uint64_t)kv_tile * Traits::kVtTileBytes));
        }
        require_status(dayan_resource_wait_lsu_produce(&kt->resource));

        if constexpr (!Traits::kFullAttention) {
            /* A two-tile delay leaves Vec and MAT room to overlap the store. */
            if (kv_tile >= 2U)
                store_probability(state, kv_tile - 2U);
        }
        dayan_yield();
    }

    if constexpr (Traits::kFullAttention) {
        require_status(dayan_resource_wait_ready(&state->output_done));
        configure_lsu_shape<1>(kQueryRows, kHeadD * sizeof(float),
                               kHeadD * sizeof(float),
                               kHeadD * sizeof(float));
        __tx_lsu_dma_u2d<1>((long)state->output_dram,
                            (long)state->output_uram);
        require_status(
            dayan_resource_wait_lsu_consume(&state->output_done));
    } else {
        store_probability(state, kKvTiles - 2U);
        store_probability(state, kKvTiles - 1U);
    }

    require_status(dayan_resource_wait_free(&state->state_ready));
    configure_lsu_shape<1>(1, kStateBytes, kStateBytes, kStateBytes);
    __tx_lsu_dma_u2d<1>((long)state->m_dram, (long)state->m_uram);
    __tx_lsu_dma_u2d<1>((long)state->l_dram, (long)state->l_uram);
    dayan_wait_lsu();
}

template <typename Traits>
static void mat_worker(void *argument)
{
    PipelineState<Traits> *state = (PipelineState<Traits> *)argument;

    if (!worker_started(state))
        return;

    require_status(dayan_resource_wait_ready(&state->q_ready));
    configure_mat<0, Traits>(
        kQueryRows, kKvTileCols, kHeadD,
        kHeadD * sizeof(typename Traits::Input),
        kHeadD * sizeof(typename Traits::Input),
        kKvTileCols * sizeof(float),
        Traits::kInputSwizzle, Traits::kInputSwizzle);
    if constexpr (Traits::kFullAttention) {
        configure_mat<1, Traits>(
            kQueryRows, kHeadD, kKvTileCols,
            kKvTileCols * sizeof(typename Traits::Input),
            kKvTileCols * sizeof(typename Traits::Input),
            kHeadD * sizeof(float),
            kLinearSwizzle, kFp16WideSwizzle);
    }

    for (uint32_t kv_tile = 0; kv_tile < kKvTiles; ++kv_tile) {
        TileBuffer *kt = &state->kt[kv_tile % kKtBufferCount];
        TileBuffer *score =
            &state->score[kv_tile % kScoreBufferCount];

        require_status(wait_for_generation(kt, kv_tile + 1U));
        require_status(dayan_resource_wait_free(&score->resource));
        if constexpr (!Traits::kFullAttention) {
            TileBuffer *probability =
                &state->probability[kv_tile % kScoreBufferCount];
            require_status(dayan_resource_wait_free(&probability->resource));
        }
        require_status(dayan_resource_alloc(
            &score->resource, 1, 1));
        score->generation = kv_tile + 1U;

        __tx_mat_mm<0>((long)score->uram, (long)state->q_uram,
                       (long)kt->uram);
        require_status(dayan_resource_wait_mat_consume(&kt->resource));
        require_status(
            dayan_resource_wait_mat_produce(&score->resource));
        if constexpr (Traits::kFullAttention) {
            /* Keep MAT busy while Vec converts the next score tile. */
            if (kv_tile != 0U) {
                const uint32_t pv_tile = kv_tile - 1U;
                TileBuffer *pv_kt = &state->kt[pv_tile % kKtBufferCount];
                TileBuffer *probability =
                    &state->probability[pv_tile % kScoreBufferCount];

                require_status(
                    wait_for_generation(probability, pv_tile + 1U));
                require_status(dayan_resource_wait_ready(
                    &state->output_for_mat[pv_tile]));
                require_status(wait_for_generation(pv_kt, pv_tile + 1U));
                __tx_mat_mma<1>(
                    (long)state->output_uram, (long)probability->uram,
                    (long)state->vt_uram[pv_tile % kKtBufferCount]);
                require_status(dayan_resource_wait_mat_consume(
                    &probability->resource));
                require_status(dayan_resource_wait_mat_consume(
                    &state->output_for_mat[pv_tile]));
                require_status(
                    dayan_resource_wait_mat_consume(&pv_kt->resource));
                require_status(dayan_resource_wait_mat_produce(
                    &state->output_for_vec[pv_tile + 1U]));
            }
        }
        dayan_yield();
    }

    require_status(dayan_resource_wait_mat_consume(&state->q_ready));
    if constexpr (Traits::kFullAttention) {
        constexpr uint32_t pv_tile = kKvTiles - 1U;
        TileBuffer *kt = &state->kt[pv_tile % kKtBufferCount];
        TileBuffer *probability =
            &state->probability[pv_tile % kScoreBufferCount];

        require_status(wait_for_generation(probability, pv_tile + 1U));
        require_status(
            dayan_resource_wait_ready(&state->output_for_mat[pv_tile]));
        require_status(wait_for_generation(kt, pv_tile + 1U));
        __tx_mat_mma<1>(
            (long)state->output_uram, (long)probability->uram,
            (long)state->vt_uram[pv_tile % kKtBufferCount]);
        require_status(
            dayan_resource_wait_mat_consume(&probability->resource));
        require_status(dayan_resource_wait_mat_consume(
            &state->output_for_mat[pv_tile]));
        require_status(dayan_resource_wait_mat_consume(&kt->resource));
        require_status(dayan_resource_wait_mat_produce(
            &state->output_for_vec[pv_tile + 1U]));
    }
    dayan_wait_mat();
}

template <typename Traits>
static void vec_worker(void *argument)
{
    PipelineState<Traits> *state = (PipelineState<Traits> *)argument;

    if (!worker_started(state))
        return;

    configure_vec<Traits>();
    require_status(dayan_resource_wait_ready(&state->state_ready));

    for (uint32_t kv_tile = 0; kv_tile < kKvTiles; ++kv_tile) {
        TileBuffer *score =
            &state->score[kv_tile % kScoreBufferCount];
        TileBuffer *probability =
            &state->probability[kv_tile % kScoreBufferCount];

        require_status(wait_for_generation(score, kv_tile + 1U));
        require_status(dayan_resource_wait_free(&probability->resource));
        require_status(
            dayan_resource_alloc(&probability->resource, 1, 1));
        probability->generation = kv_tile + 1U;
        if constexpr (Traits::kFullAttention)
            require_status(dayan_resource_wait_ready(
                &state->output_for_vec[kv_tile]));

        queue_online_softmax<Traits::kFullAttention>(
            score->uram, state->m_uram, state->l_uram, state->output_uram);
        if constexpr (Traits::kFullAttention)
            queue_probability_fp16(score->uram, probability->uram);
        start_online_softmax();
        require_status(
            dayan_resource_wait_vec_consume(&score->resource));
        if constexpr (Traits::kFullAttention) {
            require_status(dayan_resource_wait_vec_consume(
                &state->output_for_vec[kv_tile]));
            require_status(dayan_resource_wait_vec_produce(
                &state->output_for_mat[kv_tile]));
            require_status(
                dayan_resource_wait_vec_produce(&probability->resource));
        } else {
            require_status(
                dayan_resource_wait_vec_produce(&probability->resource));
            if (kv_tile + 1U == kKvTiles)
                require_status(
                    dayan_resource_wait_vec_consume(&state->state_ready));
        }

        /* M/L from this KV tile are inputs to the next online update. */
        dayan_wait_vec();
    }

    if constexpr (Traits::kFullAttention) {
        require_status(
            dayan_resource_wait_ready(&state->output_for_vec[kKvTiles]));
        queue_normalize_output(state->output_uram, state->l_uram);
        start_online_softmax();
        require_status(dayan_resource_wait_vec_consume(
            &state->output_for_vec[kKvTiles]));
        require_status(
            dayan_resource_wait_vec_consume(&state->state_ready));
        require_status(
            dayan_resource_wait_vec_produce(&state->output_done));
        dayan_wait_vec();
    }
}

} // namespace flash_attention_1c_detail

#endif

#if defined(__DAYAN_DEVICE__)
namespace flash_attention_1c_detail {

template <typename Traits>
static void run_attention(const typename Traits::Input *q,
                          const typename Traits::Input *kt,
                          const typename Traits::Input *vt,
                          float *output, float *probability,
                          FlashAttentionPerf *perf, float *m, float *l)
{
    const uint64_t kernel_begin =
        Traits::kFullAttention ? read_cycle() : 0;
    const uint64_t uram = (uint64_t)(uintptr_t)dayan_uram_base();
    PipelineState<Traits> state;
    uintptr_t pointer_bits = (uintptr_t)q | (uintptr_t)kt |
                             (uintptr_t)m | (uintptr_t)l;
    int status;

    if constexpr (Traits::kFullAttention)
        pointer_bits |= (uintptr_t)vt | (uintptr_t)output | (uintptr_t)perf;
    else
        pointer_bits |= (uintptr_t)probability;

    if (!q || !kt || !m || !l || !uram ||
        (Traits::kFullAttention && (!vt || !output || !perf)) ||
        (!Traits::kFullAttention && !probability) ||
        dayan_uram_size() < Traits::kUramBytes || dayan_task_mode() != 1U ||
        dayan_task_dim(DAYAN_TASK_X) != 1U ||
        dayan_task_dim(DAYAN_TASK_Y) != 1U ||
        dayan_task_dim(DAYAN_TASK_Z) != 1U ||
        dayan_task_id(DAYAN_TASK_X) != 0U ||
        dayan_task_id(DAYAN_TASK_Y) != 0U ||
        dayan_task_id(DAYAN_TASK_Z) != 0U || (pointer_bits & 63U) != 0U)
        dayan_thread_exit(-22);

    state.q_dram = (uint64_t)(uintptr_t)q;
    state.kt_dram = (uint64_t)(uintptr_t)kt;
    if constexpr (Traits::kFullAttention) {
        state.vt_dram = (uint64_t)(uintptr_t)vt;
        state.output_dram = (uint64_t)(uintptr_t)output;
    } else {
        state.probability_dram = (uint64_t)(uintptr_t)probability;
    }
    state.m_dram = (uint64_t)(uintptr_t)m;
    state.l_dram = (uint64_t)(uintptr_t)l;
    state.q_uram = uram + Traits::kQOffset;
    state.m_uram = uram + Traits::kMOffset;
    state.l_uram = uram + Traits::kLOffset;
    state.output_uram = Traits::kFullAttention
                            ? uram + Traits::kOutputOffset
                            : 0;
    state.start = 0;
    state.abort = 0;

    status = dayan_resource_init(&state.q_ready);
    if (status >= 0)
        status = dayan_resource_alloc(&state.q_ready, 1, 1);
    if (status >= 0)
        status = dayan_resource_init(&state.state_ready);
    if (status >= 0)
        status = dayan_resource_alloc(&state.state_ready, 1, 1);
    for (uint32_t index = 0; status >= 0 && index < kKtBufferCount;
         ++index) {
        status = dayan_resource_init(&state.kt[index].resource);
        state.kt[index].generation = 0;
        state.kt[index].uram =
            uram + Traits::kKtOffset +
            (uint64_t)index * Traits::kKtTileBytes;
        if constexpr (Traits::kFullAttention)
            state.vt_uram[index] =
                uram + Traits::kVtOffset +
                (uint64_t)index * Traits::kVtTileBytes;
    }
    for (uint32_t index = 0; status >= 0 && index < kScoreBufferCount;
         ++index) {
        status = dayan_resource_init(&state.score[index].resource);
        if (status >= 0)
            status = dayan_resource_init(
                &state.probability[index].resource);
        state.score[index].generation = 0;
        state.probability[index].generation = 0;
        state.score[index].uram =
            uram + Traits::kScoreOffset +
            (uint64_t)index * Traits::kScoreSlotBytes;
        if constexpr (Traits::kFullAttention)
            state.probability[index].uram =
                uram + Traits::kP16Offset +
                (uint64_t)index * kP16TileBytes;
        else
            state.probability[index].uram = state.score[index].uram;
    }
    if constexpr (Traits::kFullAttention) {
        for (uint32_t index = 0; status >= 0 && index <= kKvTiles; ++index) {
            status = dayan_resource_init(&state.output_for_vec[index]);
            if (status >= 0)
                status = dayan_resource_alloc(
                    &state.output_for_vec[index], 1, 1);
        }
        for (uint32_t index = 0; status >= 0 && index < kKvTiles; ++index) {
            status = dayan_resource_init(&state.output_for_mat[index]);
            if (status >= 0)
                status = dayan_resource_alloc(
                    &state.output_for_mat[index], 1, 1);
        }
        if (status >= 0)
            status = dayan_resource_init(&state.output_done);
        if (status >= 0)
            status = dayan_resource_alloc(&state.output_done, 1, 1);
    }
    if (status < 0)
        dayan_thread_exit(status);

    dayan_thread_resources_t resources;
    dayan_thread_resources_init(&resources);
    resources.lsu_tcsr_count = Traits::kFullAttention ? 3 : 2;
    status = dayan_create_thread("lsu-worker", lsu_worker<Traits>, &state,
                                 &resources);

    dayan_thread_resources_init(&resources);
    resources.mat_tcsr_count = Traits::kFullAttention ? 2 : 1;
    if (status >= 0)
        status = dayan_create_thread("mat-worker", mat_worker<Traits>, &state,
                                     &resources);

    dayan_thread_resources_init(&resources);
    resources.vec_tcsr_count = 1;
    if (status >= 0)
        status = dayan_create_thread("vec-worker", vec_worker<Traits>, &state,
                                     &resources);

    if (status < 0) {
        state.abort = 1;
        state.start = 1;
        (void)dayan_thread_join_all();
        dayan_thread_exit(status);
    }

    const uint64_t pipeline_begin =
        Traits::kFullAttention ? read_cycle() : 0;
    state.start = 1;
    status = dayan_thread_join_all();
    if (status < 0)
        dayan_thread_exit(status);
    if constexpr (Traits::kFullAttention) {
        const uint64_t pipeline_end = read_cycle();

        perf->magic = kFlashAttentionPerfMagic;
        perf->pipeline_cycles = pipeline_end - pipeline_begin;
        perf->kernel_cycles = read_cycle() - kernel_begin;
    }
}

} // namespace flash_attention_1c_detail
#endif

__kernel__ void flash_attention_fp16_1c(const uint16_t *q,
                                        const uint16_t *kt,
                                        const uint16_t *vt, float *output,
                                        FlashAttentionPerf *perf,
                                        float *m, float *l)
{
#if defined(__DAYAN_DEVICE__)
    flash_attention_1c_detail::run_attention<
        flash_attention_1c_detail::Fp16Pipeline>(
        q, kt, vt, output, nullptr, perf, m, l);
#endif
}

DAYAN_KERNEL_TASK_DESC(flash_attention_fp16_1c, 1, 3, 0,
                       kFp16UramBytes, 3, 2, 1, 0);

__kernel__ void flash_attention_fp8_1c(const uint8_t *q,
                                       const uint8_t *kt,
                                       float *probability,
                                       float *m, float *l)
{
#if defined(__DAYAN_DEVICE__)
    flash_attention_1c_detail::run_attention<
        flash_attention_1c_detail::Fp8Pipeline>(
        q, kt, nullptr, nullptr, probability, nullptr, m, l);
#endif
}

DAYAN_KERNEL_TASK_DESC(flash_attention_fp8_1c, 1, 3, 0,
                       kFp8UramBytes, 2, 1, 1, 0);

#if !defined(__DAYAN_DEVICE__)

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>

extern "C" __host__ char *getenv(const char *name);

namespace flash_attention_1c_host {

constexpr uint32_t kTimeoutMs = 300000;

template <typename Input>
static float decode_input(Input raw)
{
    static_assert(sizeof(Input) == 1 || sizeof(Input) == 2,
                  "unsupported attention input type");

    if constexpr (sizeof(Input) == sizeof(uint16_t)) {
        const bool negative = (raw & 0x8000U) != 0;
        const uint32_t exponent = (raw >> 10) & 0x1fU;
        const uint32_t mantissa = raw & 0x03ffU;
        float value;

        if (exponent == 0U) {
            value = mantissa == 0U
                        ? 0.0f
                        : std::ldexp((float)mantissa / 1024.0f, -14);
        } else if (exponent == 0x1fU) {
            return mantissa == 0U
                       ? (negative ? -std::numeric_limits<float>::infinity()
                                   : std::numeric_limits<float>::infinity())
                       : std::numeric_limits<float>::quiet_NaN();
        } else {
            value = std::ldexp(1.0f + (float)mantissa / 1024.0f,
                               (int)exponent - 15);
        }
        return negative ? -value : value;
    } else {
        const bool negative = (raw & 0x80U) != 0;
        const uint32_t exponent = (raw >> 3) & 0x0fU;
        const uint32_t mantissa = raw & 0x07U;
        float value;

        if (exponent == 0U) {
            value = mantissa == 0U
                        ? 0.0f
                        : std::ldexp((float)mantissa / 8.0f, -6);
        } else if (exponent == 0x0fU && mantissa == 0x07U) {
            return std::numeric_limits<float>::quiet_NaN();
        } else {
            value = std::ldexp(1.0f + (float)mantissa / 8.0f,
                               (int)exponent - 7);
        }
        return negative ? -value : value;
    }
}

template <typename Input>
static Input encode_small_integer(int value)
{
    static_assert(sizeof(Input) == 1 || sizeof(Input) == 2,
                  "unsupported attention input type");
    const bool negative = value < 0;
    const uint32_t magnitude = (uint32_t)(negative ? -value : value);
    Input raw = 0;

    switch (magnitude) {
    case 0:
        raw = 0x00;
        break;
    case 1:
        raw = sizeof(Input) == sizeof(uint16_t) ? 0x3c00 : 0x38;
        break;
    case 2:
        raw = sizeof(Input) == sizeof(uint16_t) ? 0x4000 : 0x40;
        break;
    case 3:
        raw = sizeof(Input) == sizeof(uint16_t) ? 0x4200 : 0x44;
        break;
    default:
        return sizeof(Input) == sizeof(uint16_t) ? 0x7e00 : 0x7f;
    }
    if (!negative || magnitude == 0U)
        return raw;
    return (Input)(raw | (sizeof(Input) == sizeof(uint16_t)
                              ? 0x8000U
                              : 0x80U));
}

static int wait_and_destroy(npu_event_t *event)
{
    npu_event_info_t info;
    int status;
    int destroy_status;

    if (!event)
        return -1;
    status = npu_event_wait(event, kTimeoutMs, &info);
    destroy_status = npu_event_destroy(event);
    if (status < 0)
        return status;
    if (info.status < std::numeric_limits<int>::min() ||
        info.status > std::numeric_limits<int>::max())
        return -1;
    if (info.status < 0)
        return (int)info.status;
    return destroy_status;
}

static int copy_h2d(npu_stream_t *stream, npu_buffer_t *buffer,
                    const void *source, size_t size)
{
    npu_event_t *event = nullptr;
    const int status = npu_memcpy_h2d_async(stream, buffer, 0,
                                            source, size, &event);
    return status < 0 ? status : wait_and_destroy(event);
}

static int copy_d2h(npu_stream_t *stream, void *destination,
                    npu_buffer_t *buffer, size_t size)
{
    npu_event_t *event = nullptr;
    const int status = npu_memcpy_d2h_async(stream, destination, buffer, 0,
                                            size, &event);
    return status < 0 ? status : wait_and_destroy(event);
}

static int create_buffer(npu_context_t *context, size_t size,
                         npu_buffer_t **buffer)
{
    npu_buffer_desc_t desc;

    npu_buffer_desc_init(&desc, size);
    desc.alignment = 64;
    return npu_buffer_create(context, &desc, buffer);
}

static void preserve_error(int candidate, int *status)
{
    if (*status == 0 && candidate < 0)
        *status = candidate;
}

static uint16_t fp32_to_fp16(float value)
{
    uint32_t bits;

    std::memcpy(&bits, &value, sizeof(bits));
    uint16_t result = (uint16_t)(((bits >> 16) & 0x8000U) |
        ((((bits & 0x7f800000U) - 0x38000000U) >> 13) & 0x7c00U) |
        ((bits >> 13) & 0x03ffU));

    if ((bits & 0x7f800000U) == 0x7f800000U) {
        result = (uint16_t)((result & 0x8000U) | 0x7c00U |
                            ((bits >> 13) & 0x03ffU));
    } else if ((bits & 0x7f800000U) > 0x477fe000U) {
        result = (uint16_t)((result & 0x8000U) | 0x7c00U);
    } else if ((bits & 0x7f800000U) < 0x38000000U) {
        uint32_t mantissa = (bits & 0x007fffffU) | 0x00800000U;
        const int32_t exponent = (int32_t)((bits >> 23) & 0xffU) -
                                 127 - 14;

        if (exponent < -24)
            result = (uint16_t)(bits >> 16) & 0x8000U;
        else
            result = (uint16_t)(((bits >> 16) & 0x8000U) |
                                (mantissa >> (-exponent + 13)));
    }
    return result;
}

static void initialize_fp16(uint16_t *q, uint16_t *kt, uint16_t *vt,
                            float *output, float *m, float *l)
{
    static constexpr int tile_bias[kKvTiles] = {
        -2, 1, -1, 2,
    };

    std::memset(q, 0, kQElements * sizeof(*q));
    std::memset(kt, 0, kKtElements * sizeof(*kt));
    std::memset(vt, 0, kVtElements * sizeof(*vt));
    std::memset(output, 0, kOutputBytes);
    for (uint32_t row = 0; row < kQueryRows; ++row) {
        const int value = 1 + (int)((row / kAttentionRows) & 1U);

        q[row * kHeadD + row] = encode_small_integer<uint16_t>(value);
    }
    for (uint32_t kv_tile = 0; kv_tile < kKvTiles; ++kv_tile) {
        for (uint32_t key = 0; key < kKvTileCols; ++key) {
            for (uint32_t component = 0; component < kHeadD;
                 ++component) {
                const int value = tile_bias[kv_tile] +
                                  (int)((key * 2U + component) % 3U) - 1;

                kt[(kv_tile * kKvTileCols + key) * kHeadD +
                   component] = encode_small_integer<uint16_t>(value);
            }
        }
    }
    for (uint32_t kv_tile = 0; kv_tile < kKvTiles; ++kv_tile) {
        for (uint32_t component = 0; component < kHeadD; ++component) {
            for (uint32_t key = 0; key < kKvTileCols; ++key) {
                const int value =
                    (int)((key * 3U + component + kv_tile) % 5U) - 2;

                vt[(kv_tile * kHeadD + component) * kKvTileCols + key] =
                    encode_small_integer<uint16_t>(value);
            }
        }
    }
    for (uint32_t row = 0; row < kQueryRows; ++row) {
        for (uint32_t lane = 0; lane < kStateLanesPerRow; ++lane) {
            const uint32_t index = row * kStateLanesPerRow + lane;

            m[index] = -4.0f;
            l[index] = 0.0f;
        }
    }
}

static int validate_fp16(const uint16_t *q, const uint16_t *kt,
                         const uint16_t *vt, const float *output,
                         const float *old_m, const float *old_l,
                         const float *m, const float *l)
{
    float expected_o[kOutputElements] = {};
    float expected_m[kQueryRows];
    float expected_l[kQueryRows];
    float max_absolute_error = 0.0f;
    float max_relative_error = 0.0f;
    uint32_t mismatches = 0;
    uint32_t output_mismatches = 0;
    uint32_t m_mismatches = 0;
    uint32_t l_mismatches = 0;

    for (uint32_t row = 0; row < kQueryRows; ++row) {
        expected_m[row] = old_m[row * kStateLanesPerRow];
        expected_l[row] = old_l[row * kStateLanesPerRow];
    }

    for (uint32_t kv_tile = 0; kv_tile < kKvTiles; ++kv_tile) {
        const uint16_t *kt_tile =
            kt + kv_tile * kKtTileElements;

        for (uint32_t row = 0; row < kQueryRows; ++row) {
            float scores[kKvTileCols];
            float tile_max = -std::numeric_limits<float>::infinity();
            float tile_sum = 0.0f;

            for (uint32_t key = 0; key < kKvTileCols; ++key) {
                float score = 0.0f;

                for (uint32_t component = 0; component < kHeadD;
                     ++component) {
                    score +=
                        decode_input(q[row * kHeadD + component]) *
                        decode_input(
                            kt_tile[key * kHeadD + component]);
                }
                scores[key] = score;
                tile_max = std::fmax(tile_max, score);
            }
            const float new_m = std::fmax(expected_m[row], tile_max);
            const float alpha = std::exp2(expected_m[row] - new_m);
            for (uint32_t component = 0; component < kHeadD;
                 ++component)
                expected_o[row * kHeadD + component] *= alpha;
            for (uint32_t key = 0; key < kKvTileCols; ++key) {
                const float p = std::exp2(scores[key] - new_m);
                const float p16 = decode_input(fp32_to_fp16(p));

                tile_sum += p;
                for (uint32_t component = 0; component < kHeadD;
                     ++component) {
                    expected_o[row * kHeadD + component] +=
                        p16 * decode_input(
                            vt[(kv_tile * kHeadD + component) *
                                   kKvTileCols +
                               key]);
                }
            }
            expected_l[row] =
                std::exp2(expected_m[row] - new_m) * expected_l[row] +
                tile_sum;
            expected_m[row] = new_m;
        }
    }

    for (uint32_t row = 0; row < kQueryRows; ++row)
        for (uint32_t component = 0; component < kHeadD; ++component)
            expected_o[row * kHeadD + component] /= expected_l[row];

    const auto check = [&](float actual, float expected,
                           float absolute_tolerance,
                           float relative_tolerance,
                           const char *tensor, uint32_t index,
                           uint32_t *tensor_mismatches) {
        const float absolute_error = std::fabs(actual - expected);
        const float relative_error =
            absolute_error / std::fmax(std::fabs(expected), 1.0e-6f);

        max_absolute_error = std::fmax(max_absolute_error, absolute_error);
        max_relative_error = std::fmax(max_relative_error, relative_error);
        if (!std::isfinite(actual) ||
            absolute_error > absolute_tolerance +
                                 relative_tolerance * std::fabs(expected)) {
            if (*tensor_mismatches < 2U)
                std::printf("mismatch %s[%u]: actual=%g expected=%g\n",
                            tensor, index, actual, expected);
            ++mismatches;
            ++*tensor_mismatches;
        }
    };

    for (uint32_t index = 0; index < kOutputElements; ++index)
        check(output[index], expected_o[index], 0.08f, 0.05f, "O", index,
              &output_mismatches);
    for (uint32_t row = 0; row < kQueryRows; ++row) {
        for (uint32_t lane = 0; lane < kStateLanesPerRow; ++lane) {
            const uint32_t index = row * kStateLanesPerRow + lane;
            check(m[index], expected_m[row], 1.0e-6f, 0.0f, "M", index,
                  &m_mismatches);
            check(l[index], expected_l[row], 0.8f, 0.02f, "L", index,
                  &l_mismatches);
        }
    }

    std::printf("[host] rows=%llu D=%llu kv_tiles=%llu input=FP16 "
                "elements=%llu mismatches=%u (O=%u M=%u L=%u) "
                "max_abs_error=%g "
                "max_rel_error=%g\n",
                (unsigned long long)kQueryRows,
                (unsigned long long)kHeadD,
                (unsigned long long)kKvTiles,
                (unsigned long long)kOutputElements,
                mismatches, output_mismatches, m_mismatches, l_mismatches,
                max_absolute_error, max_relative_error);
    return mismatches == 0U ? 0 : -1;
}
static void initialize_fp8(uint8_t *q, uint8_t *kt, float *m, float *l)
{
    static constexpr int tile_bias[kKvTiles] = {
        -2, 1, -1, 2,
    };

    std::memset(q, 0, kQElements * sizeof(*q));
    std::memset(kt, 0, kKtElements * sizeof(*kt));
    for (uint32_t row = 0; row < kQueryRows; ++row)
        q[row * kHeadD + row] = encode_small_integer<uint8_t>(1);
    for (uint32_t kv_tile = 0; kv_tile < kKvTiles; ++kv_tile) {
        for (uint32_t key = 0; key < kKvTileCols; ++key) {
            for (uint32_t component = 0; component < kHeadD;
                 ++component) {
                const int value = tile_bias[kv_tile] -
                                  (int)((key + component) & 1U);

                kt[(kv_tile * kKvTileCols + key) * kHeadD +
                   component] = encode_small_integer<uint8_t>(value);
            }
        }
    }
    for (uint32_t row = 0; row < kQueryRows; ++row) {
        for (uint32_t lane = 0; lane < kStateLanesPerRow; ++lane) {
            const uint32_t index = row * kStateLanesPerRow + lane;

            m[index] = -4.0f;
            l[index] = 0.0f;
        }
    }
}

static int validate_fp8(const uint8_t *q, const uint8_t *kt,
                        const float *old_m, const float *old_l,
                        const float *probability, const float *m,
                        const float *l)
{
    float expected_p[kProbabilityElements];
    float expected_m[kQueryRows];
    float expected_l[kQueryRows];
    float max_absolute_error = 0.0f;
    float max_relative_error = 0.0f;
    uint32_t mismatches = 0;

    for (uint32_t row = 0; row < kQueryRows; ++row) {
        expected_m[row] = old_m[row * kStateLanesPerRow];
        expected_l[row] = old_l[row * kStateLanesPerRow];
    }

    for (uint32_t kv_tile = 0; kv_tile < kKvTiles; ++kv_tile) {
        const uint8_t *kt_tile =
            kt + kv_tile * kKtTileElements;

        for (uint32_t row = 0; row < kQueryRows; ++row) {
            float scores[kKvTileCols];
            float tile_max = -std::numeric_limits<float>::infinity();
            float tile_sum = 0.0f;

            for (uint32_t key = 0; key < kKvTileCols; ++key) {
                float score = 0.0f;

                for (uint32_t component = 0; component < kHeadD;
                     ++component) {
                    score +=
                        decode_input(q[row * kHeadD + component]) *
                        decode_input(
                            kt_tile[key * kHeadD + component]);
                }
                scores[key] = score;
                tile_max = std::fmax(tile_max, score);
            }
            const float new_m = std::fmax(expected_m[row], tile_max);
            for (uint32_t key = 0; key < kKvTileCols; ++key) {
                const float p = std::exp2(scores[key] - new_m);
                const uint64_t index =
                    ((uint64_t)kv_tile * kQueryRows + row) *
                        kKvTileCols +
                    key;

                expected_p[index] = p;
                tile_sum += p;
            }
            expected_l[row] =
                std::exp2(expected_m[row] - new_m) * expected_l[row] +
                tile_sum;
            expected_m[row] = new_m;
        }
    }

    const auto check = [&](float actual, float expected,
                           float absolute_tolerance,
                           float relative_tolerance,
                           const char *tensor, uint32_t index) {
        const float absolute_error = std::fabs(actual - expected);
        const float relative_error =
            absolute_error / std::fmax(std::fabs(expected), 1.0e-6f);

        max_absolute_error = std::fmax(max_absolute_error, absolute_error);
        max_relative_error = std::fmax(max_relative_error, relative_error);
        if (!std::isfinite(actual) ||
            absolute_error > absolute_tolerance +
                                 relative_tolerance * std::fabs(expected)) {
            if (mismatches < 6U)
                std::printf("mismatch %s[%u]: actual=%g expected=%g\n",
                            tensor, index, actual, expected);
            ++mismatches;
        }
    };

    for (uint32_t index = 0; index < kProbabilityElements; ++index)
        check(probability[index], expected_p[index], 0.04f, 0.02f,
              "P", index);
    for (uint32_t row = 0; row < kQueryRows; ++row) {
        for (uint32_t lane = 0; lane < kStateLanesPerRow; ++lane) {
            const uint32_t index = row * kStateLanesPerRow + lane;
            check(m[index], expected_m[row], 1.0e-6f, 0.0f, "M", index);
            check(l[index], expected_l[row], 0.8f, 0.02f, "L", index);
        }
    }

    std::printf("[host] rows=%llu D=%llu kv_tiles=%llu input=FP8-E4M3 "
                "elements=%llu mismatches=%u max_abs_error=%g "
                "max_rel_error=%g\n",
                (unsigned long long)kQueryRows,
                (unsigned long long)kHeadD,
                (unsigned long long)kKvTiles,
                (unsigned long long)kProbabilityElements,
                mismatches, max_absolute_error, max_relative_error);
    return mismatches == 0U ? 0 : -1;
}
} // namespace flash_attention_1c_host

template <typename Input, bool FullAttention>
__host__ static int run_attention_demo()
{
    using namespace flash_attention_1c_host;

    static_assert((FullAttention && sizeof(Input) == sizeof(uint16_t)) ||
                      (!FullAttention && sizeof(Input) == sizeof(uint8_t)),
                  "precision and input type disagree");
    Input q[kQElements];
    Input kt[kKtElements];
    Input vt[kVtElements];
    float output[kOutputElements];
    float probability[kProbabilityElements] = {};
    FlashAttentionPerf perf = {};
    float m[kStateElements];
    float l[kStateElements];
    float old_m[kStateElements];
    float old_l[kStateElements];
    npu_runtime_options_t options;
    npu_runtime_t *runtime = nullptr;
    npu_context_t *context = nullptr;
    npu_stream_t *stream = nullptr;
    npu_buffer_t *q_buffer = nullptr;
    npu_buffer_t *kt_buffer = nullptr;
    npu_buffer_t *vt_buffer = nullptr;
    npu_buffer_t *output_buffer = nullptr;
    npu_buffer_t *perf_buffer = nullptr;
    npu_buffer_t *probability_buffer = nullptr;
    npu_buffer_t *m_buffer = nullptr;
    npu_buffer_t *l_buffer = nullptr;
    const char *socket_path;
    int status = 0;

    if constexpr (FullAttention)
        initialize_fp16(q, kt, vt, output, m, l);
    else
        initialize_fp8(q, kt, m, l);
    std::memcpy(old_m, m, sizeof(m));
    std::memcpy(old_l, l, sizeof(l));

    npu_runtime_options_init(&options);
    socket_path = getenv("DAYAN_SERVER_SOCKET");
    if (socket_path && *socket_path)
        options.server_socket = socket_path;
    status = npu_runtime_open(&runtime, &options);
    if (status == 0)
        status = npu_context_create(runtime, nullptr, &context);
    if (status == 0)
        status = npu_stream_create(context, nullptr, &stream);
    if (status == 0)
        status = create_buffer(context, sizeof(q), &q_buffer);
    if (status == 0)
        status = create_buffer(context, sizeof(kt), &kt_buffer);
    if constexpr (FullAttention) {
        if (status == 0)
            status = create_buffer(context, sizeof(vt), &vt_buffer);
        if (status == 0)
            status = create_buffer(context, sizeof(output), &output_buffer);
        if (status == 0)
            status = create_buffer(context, sizeof(perf), &perf_buffer);
    } else if (status == 0) {
        status = create_buffer(context, sizeof(probability),
                               &probability_buffer);
    }
    if (status == 0)
        status = create_buffer(context, sizeof(m), &m_buffer);
    if (status == 0)
        status = create_buffer(context, sizeof(l), &l_buffer);
    if (status == 0)
        status = copy_h2d(stream, q_buffer, q, sizeof(q));
    if (status == 0)
        status = copy_h2d(stream, kt_buffer, kt, sizeof(kt));
    if constexpr (FullAttention) {
        if (status == 0)
            status = copy_h2d(stream, vt_buffer, vt, sizeof(vt));
        if (status == 0)
            status = copy_h2d(stream, output_buffer, output, sizeof(output));
    }
    if (status == 0)
        status = copy_h2d(stream, m_buffer, m, sizeof(m));
    if (status == 0)
        status = copy_h2d(stream, l_buffer, l, sizeof(l));
    if (status == 0) {
        npu_kernel_launch_attrs_t attrs;

        constexpr const char *kTaskName =
            FullAttention ? "flash-fp16-1c" : "flash-fp8-1c";
        npu_kernel_launch_attrs_init(&attrs, context, stream, kTaskName);
        attrs.config.task_dim = NPUTaskDim3{1, 1, 1};
        attrs.config.task_mode = NPUTaskMode::kSingleCore;
        attrs.timeout_ms = kTimeoutMs;
        if constexpr (FullAttention)
            status = flash_attention_fp16_1c(
                q_buffer, kt_buffer, vt_buffer, output_buffer, perf_buffer,
                m_buffer, l_buffer, &attrs);
        else
            status = flash_attention_fp8_1c(
                q_buffer, kt_buffer, probability_buffer,
                m_buffer, l_buffer, &attrs);
    }
    if constexpr (FullAttention) {
        if (status == 0)
            status = copy_d2h(stream, &perf, perf_buffer, sizeof(perf));
        if (status == 0 &&
            (perf.magic != kFlashAttentionPerfMagic || !perf.kernel_cycles ||
             !perf.pipeline_cycles ||
             perf.pipeline_cycles > perf.kernel_cycles))
            status = -1;
        if (status == 0) {
            constexpr double kMatPeakFlopsPerCycle = 4096.0;
            const double mat_flops =
                4.0 * (double)kKvTiles * (double)kQueryRows *
                (double)kKvTileCols * (double)kHeadD;
            const double pipeline_flops_per_cycle =
                mat_flops / (double)perf.pipeline_cycles;

            std::printf("[host] perf: kernel_cycles=%llu "
                        "pipeline_cycles=%llu mat_flops_per_cycle=%.3f "
                        "mat_peak_utilization=%.2f%%\n",
                        (unsigned long long)perf.kernel_cycles,
                        (unsigned long long)perf.pipeline_cycles,
                        pipeline_flops_per_cycle,
                        pipeline_flops_per_cycle * 100.0 /
                            kMatPeakFlopsPerCycle);
        }
        if (status == 0)
            status = copy_d2h(stream, output, output_buffer, sizeof(output));
    } else if (status == 0) {
        status = copy_d2h(stream, probability, probability_buffer,
                          sizeof(probability));
    }
    if (status == 0)
        status = copy_d2h(stream, m, m_buffer, sizeof(m));
    if (status == 0)
        status = copy_d2h(stream, l, l_buffer, sizeof(l));
    if (status == 0) {
        if constexpr (FullAttention)
            status = validate_fp16(q, kt, vt, output,
                                   old_m, old_l, m, l);
        else
            status = validate_fp8(q, kt, old_m, old_l,
                                  probability, m, l);
    }

    constexpr const char *kInputName = FullAttention ? "FP16" : "FP8";
    constexpr const char *kOperation =
        FullAttention ? "QK-softmax-PV" : "online-softmax";
    if (status == 0)
        std::printf("[host] PASS: %s FlashAttention %s 1c\n",
                    kInputName, kOperation);
    else
        std::printf("[host] FAIL: %s FlashAttention 1c, status=%d\n",
                    kInputName, status);

    if (l_buffer)
        preserve_error(npu_buffer_destroy(l_buffer), &status);
    if (m_buffer)
        preserve_error(npu_buffer_destroy(m_buffer), &status);
    if (perf_buffer)
        preserve_error(npu_buffer_destroy(perf_buffer), &status);
    if (output_buffer)
        preserve_error(npu_buffer_destroy(output_buffer), &status);
    if (vt_buffer)
        preserve_error(npu_buffer_destroy(vt_buffer), &status);
    if (probability_buffer)
        preserve_error(npu_buffer_destroy(probability_buffer), &status);
    if (kt_buffer)
        preserve_error(npu_buffer_destroy(kt_buffer), &status);
    if (q_buffer)
        preserve_error(npu_buffer_destroy(q_buffer), &status);
    if (stream)
        preserve_error(npu_stream_destroy(stream), &status);
    if (context)
        preserve_error(npu_context_destroy(context), &status);
    if (runtime)
        preserve_error(npu_runtime_close(runtime), &status);
    return status == 0 ? 0 : 1;
}

__host__ static int run_fp16()
{
    return run_attention_demo<uint16_t, true>();
}

__host__ static int run_fp8()
{
    return run_attention_demo<uint8_t, false>();
}

__host__ int main(int argc, char **argv)
{
    if (argc == 2 && std::strcmp(argv[1], "--help") == 0) {
        std::printf("usage: %s [--fp8]\n", argv[0]);
        return 0;
    }
    if (argc == 1)
        return run_fp16();
    if (argc == 2 && std::strcmp(argv[1], "--fp8") == 0)
        return run_fp8();
    return 2;
}

#endif
