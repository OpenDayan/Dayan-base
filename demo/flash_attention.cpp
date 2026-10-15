#include <dayan_kernel.h>
#include <stdint.h>

// Ported from llama.cpp/ggml/src/ggml-dayan/ggml-dayan-flash-attention.h.
// Two 128-row Q slices share double-buffered 128-key K/V tiles. Six child
// Threads load Q/K/V, evaluate online softmax, multiply P*V, and update O;
// the root issues Q*K. P stays FP32 in the score allocation and MAT reads it
// as TF32, avoiding the old demo's FP16 conversion and separate P allocation.
// Four-core prefill shares K/V loads; Split-KV decode merges per-core M/L/O.
// Q/K/V are 16-bit strided [batch, head, row, D]; O is FP32 [batch, row, head, D].
// scale_bits contains the attention scale (normally 1/sqrt(D)); Q is not pre-scaled.
struct FlashAttentionConfig {
    uint32_t batch;
    uint32_t q_heads;
    uint32_t kv_heads;
    uint32_t q_len;
    uint32_t kv_len;
    uint32_t head_dim;
    uint32_t scale_bits;
    uint32_t causal_offset;
    uint32_t flags;
    uint64_t q_row_stride;
    uint64_t q_head_stride;
    uint64_t q_batch_stride;
    uint64_t k_row_stride;
    uint64_t k_head_stride;
    uint64_t k_batch_stride;
    uint64_t v_row_stride;
    uint64_t v_head_stride;
    uint64_t v_batch_stride;
};

// Device harts have noncoherent 64-byte cache lines; each status owns a line.
struct alignas(64) FlashAttentionResult {
    int64_t status;
};
static_assert(sizeof(FlashAttentionResult) == 64, "Task status must own a cache line");

#if defined(__DAYAN_DEVICE__)
#include <dayan_intrinsics.h>
#include <dayan_sys.h>

namespace flash_attention_detail {

struct Bf16Format { enum : long { kMatDataType = BaseNPUInstBuilder::DT_BF16 }; };
struct Fp16Format { enum : long { kMatDataType = BaseNPUInstBuilder::DT_FP16 }; };

static inline uint32_t float_as_u32(float value) {
    union {
        float    f;
        uint32_t u;
    } bits;

    bits.f = value;
    return bits.u;
}

static inline float u32_as_float(uint32_t value) {
    union {
        float    f;
        uint32_t u;
    } bits;

    bits.u = value;
    return bits.f;
}

enum VecTcsr {
    kVecControl = 0,
    kVecSync    = 1,
    kVecDim0    = 2,
    kVecDim1    = 3,
    kVecDim2    = 4,
    kVecDim3    = 5,
    kVecDim4    = 6,
    kVecDim5    = 7,
    kVecV0Base  = 37,
    kVecV1Base  = 44,
    kVecV2Base  = 51,
    kVecV3Base  = 58,
};

template <long Base>
static inline void configure_vec_iterator(long dim0_stride,
                                          long dim1_stride,
                                          long dim2_stride,
                                          long dim3_stride,
                                          long channel_stride = 32) {
    __tx_vec_tcsr_write<Base>(0);
    __tx_vec_tcsr_write<Base + 1>(dim0_stride);
    __tx_vec_tcsr_write<Base + 2>(dim1_stride);
    __tx_vec_tcsr_write<Base + 3>(dim2_stride);
    __tx_vec_tcsr_write<Base + 4>(dim3_stride);
    __tx_vec_tcsr_write<Base + 5>(0);
    __tx_vec_tcsr_write<Base + 6>(0);
    if constexpr (Base == kVecV0Base) {
        __tx_vec_tcsr_write<VEC_V0_URAM_CHANNEL_STRIDE>(channel_stride);
    } else if constexpr (Base == kVecV1Base) {
        __tx_vec_tcsr_write<VEC_V1_URAM_CHANNEL_STRIDE>(channel_stride);
    } else if constexpr (Base == kVecV2Base) {
        __tx_vec_tcsr_write<VEC_V2_URAM_CHANNEL_STRIDE>(channel_stride);
    } else if constexpr (Base == kVecV3Base) {
        __tx_vec_tcsr_write<VEC_V3_URAM_CHANNEL_STRIDE>(channel_stride);
    }
}

static inline void configure_vec_grid(long dim2, long dim3 = 1) {
    __tx_vec_tcsr_write<kVecDim0>(0);
    __tx_vec_tcsr_write<kVecDim1>(0);
    __tx_vec_tcsr_write<kVecDim2>(dim2 - 1);
    __tx_vec_tcsr_write<kVecDim3>(dim3 - 1);
    __tx_vec_tcsr_write<kVecDim4>(0);
    __tx_vec_tcsr_write<kVecDim5>(0);
}

static inline long vec_control(long instructions) {
    const uint64_t count = (uint64_t) instructions & 0xffU;
    return (long) (count | (count << 8) | (count << 16) | (count << 24) | (count << 32));
}

static inline void begin_vec_block(long instructions) {
    long free_entries;

    do {
        __tx_vec_tcsr_read<kVecControl>(free_entries);
    } while (free_entries < instructions);
}

static inline void start_vec_block(long instructions) {
    // Keep CPU URAM stores before VEC submission.
    __asm__ volatile("" ::: "memory");
    __tx_vec_tcsr_write<kVecSync>(0);
    __tx_vec_tcsr_write<kVecControl>(vec_control(instructions));
}

__device__ constexpr long kSlices = 2;
__device__ constexpr long kSlots = 2;
__device__ constexpr long kVecLanes = 64;
__device__ constexpr uint32_t kFlagCausal = 1;
__device__ constexpr long kBr = 128;
__device__ constexpr long kBc = 128;
__device__ constexpr long kMaxHeadDim = 256;
__device__ constexpr long kVecBytes = 256;
__device__ constexpr long kLocalResources = kSlices + 6 * kSlices * kSlots + 4;
__device__ constexpr long kSharedResources = 4;
__device__ constexpr long kMatQK = 1;
__device__ constexpr long kAtomicBytes = 128;
__device__ constexpr long kUramBytes = 1536 * 1024;
__device__ constexpr long kQBytes = kBr * kMaxHeadDim * sizeof(uint16_t);
__device__ constexpr long kOBytes = kBr * kMaxHeadDim * sizeof(float);
__device__ constexpr long kPVBytes = kOBytes;
__device__ constexpr long kScoreBytes = kBr * kBc * sizeof(float);
__device__ constexpr long kStateVectorBytes = kBr * 32;
__device__ constexpr long kMaskBytes = (kBr / 8) * (kBc / 8) * 32;
__device__ constexpr long kCtxQ = 0;
__device__ constexpr long kCtxO = kQBytes;
__device__ constexpr long kCtxTiles = kCtxO + kOBytes;
__device__ constexpr long kTileScore = 0;
__device__ constexpr long kTilePV = kTileScore + kScoreBytes;
__device__ constexpr long kTileM = kTilePV + kPVBytes;
__device__ constexpr long kTileL = kTileM + kStateVectorBytes;
__device__ constexpr long kTileAlpha = kTileL + kStateVectorBytes;
__device__ constexpr long kTileMask = kTileAlpha + kStateVectorBytes;
__device__ constexpr long kTileBytes = kTileMask + kMaskBytes;
__device__ constexpr long kContextBytes = kCtxTiles + kSlots * kTileBytes;
__device__ constexpr long kKBytes = kBc * kMaxHeadDim * sizeof(uint16_t);
__device__ constexpr long kVtBytes = kKBytes;
__device__ constexpr long kUramK0 = kSlices * kContextBytes;
__device__ constexpr long kUramVt0 = kUramK0 + 2 * kKBytes;
__device__ constexpr long kUramZero = kUramVt0 + 2 * kVtBytes;
__device__ constexpr long kUramMaskRows = kUramZero + kVecBytes;
__device__ constexpr long kUramMaskCols = kUramMaskRows + kVecBytes;
__device__ constexpr long kUramMaskDelta = kUramMaskCols + kVecBytes;
static_assert(kUramMaskDelta + kVecBytes <= kUramBytes, "Attention URAM exceeds allocation");
static_assert(kLocalResources <= 32, "Attention exceeds the Local Resource window");

static inline long min_long(long a, long b) { return a < b ? a : b; }
static inline long extent(long n, long size) { return n <= 0 ? 0 : min_long(n, size); }
static inline void check(int status) {
    if (status < 0) {
        dayan_thread_exit(status);
    }
}

static inline dayan_resource_t q_resource(long slice) { return DAYAN_LOCAL_RESOURCE(slice); }
static inline dayan_resource_t score_resource(long slice, long slot) { return DAYAN_LOCAL_RESOURCE(kSlices + slice * kSlots + slot); }
static inline dayan_resource_t p_resource(long slice, long slot) { return DAYAN_LOCAL_RESOURCE(kSlices + kSlices * kSlots + slice * kSlots + slot); }
static inline dayan_resource_t pv_resource(long slice, long slot) { return DAYAN_LOCAL_RESOURCE(kSlices + 2 * kSlices * kSlots + slice * kSlots + slot); }
static inline dayan_resource_t softmax_state_resource(long slice, long slot) { return DAYAN_LOCAL_RESOURCE(kSlices + 3 * kSlices * kSlots + slice * kSlots + slot); }
static inline dayan_resource_t o_resource(long slice, long slot) { return DAYAN_LOCAL_RESOURCE(kSlices + 4 * kSlices * kSlots + slice * kSlots + slot); }
static inline dayan_resource_t kv_resource(bool v, long slot) { return DAYAN_LOCAL_RESOURCE(kSlices + 5 * kSlices * kSlots + 2 * v + slot); }
static inline dayan_resource_t output_state_resource(long slice, long slot) { return DAYAN_LOCAL_RESOURCE(kLocalResources - kSlices * kSlots + slice * kSlots + slot); }
static inline long tile_offset(long slice, long slot) {
    return slice * kContextBytes + kCtxTiles + slot * kTileBytes;
}

template <long Group> static inline void configure_lsu_rows(long row_bytes, long input_stride, long output_stride) {
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_BLOCK>(0);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_DIM0>(row_bytes / 64 - 1);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_DIM1>(0);
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

template <long Group> static inline void set_lsu_rows(long rows) {
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_DIM1>(rows - 1);
}

template <long Group> static inline void configure_v_transpose(long head_dim, uint64_t row_stride) {
    constexpr long side = 32;

    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_BLOCK>(side - 1);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_DIM0>(head_dim / side - 1);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_DIM1>(0);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_DIM2>(0);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_INBLOCKSTRIDE>(row_stride);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_INDIM0STRIDE>(64);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_INDIM1STRIDE>(0);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_INDIM2STRIDE>(0);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_OUTBLOCKSTRIDE>(kBc * sizeof(uint16_t));
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_OUTDIM0STRIDE>(kBc * side * sizeof(uint16_t));
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_OUTDIM1STRIDE>(0);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_OUTDIM2STRIDE>(0);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_BLOCKTRANSPOSE>(1);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_BLOCKPADDING>(side - 1);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_REQUESTREADSIZE>(63);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_PADDINGVALUE>(0);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_URAMWRITESWIZZLEMBS>(0);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_URAMREADSWIZZLEMBS>(0);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_REQUESTWRITESIZE>(63);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_TABLESIZE>(0);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_TABLESTRIDE>(64);
}

template <long Group>
static inline void load_v_transposed(uint64_t v_dram, long vt_base, long rows, uint64_t row_stride) {
    constexpr long side   = 32;
    const long     panels = (rows + side - 1) / side;

    for (long panel = 0; panel < panels; ++panel) {
        const long panel_rows = min_long(side, rows - panel * side);
        __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_BLOCK>(panel_rows - 1);
        __tx_lsu_dma_d2u<Group>(vt_base + panel * side * sizeof(uint16_t),
                                v_dram + static_cast<uint64_t>(panel * side) * row_stride);
    }
}

template <typename Format, long Group>
static inline void configure_mat(long m, long n, long k, long a_stride, long b_stride, long c_stride) {
    __tx_mat_tcsr_write<Group * MAT_GROUPSZ + MAT_M>(m - 1);
    __tx_mat_tcsr_write<Group * MAT_GROUPSZ + MAT_N>(n - 1);
    __tx_mat_tcsr_write<Group * MAT_GROUPSZ + MAT_K>(k - 1);
    __tx_mat_tcsr_write<Group * MAT_GROUPSZ + MAT_ASTRIDE>(a_stride);
    __tx_mat_tcsr_write<Group * MAT_GROUPSZ + MAT_BSTRIDE>(b_stride);
    __tx_mat_tcsr_write<Group * MAT_GROUPSZ + MAT_CSTRIDE>(c_stride);
    __tx_mat_tcsr_write<Group * MAT_GROUPSZ + MAT_ASWIZZLEMBS>(0);
    __tx_mat_tcsr_write<Group * MAT_GROUPSZ + MAT_BSWIZZLEMBS>(0);
    __tx_mat_tcsr_write<Group * MAT_GROUPSZ + MAT_CSWIZZLEMBS>(0);
    __tx_mat_tcsr_write<Group * MAT_GROUPSZ + MAT_PERSISTENTMODE>(0);
    __tx_mat_tcsr_write<Group * MAT_GROUPSZ + MAT_DATATYPE>(Format::kMatDataType);
}


static inline bool checked_mul_u64(uint64_t a, uint64_t b, uint64_t * result) {
    constexpr uint64_t max_u64 = ~uint64_t{ 0 };
    if (a != 0 && b > max_u64 / a) {
        return false;
    }
    *result = a * b;
    return true;
}

static inline bool checked_add_u64(uint64_t a, uint64_t b, uint64_t * result) {
    if (b > (~uint64_t{ 0 }) - a) {
        return false;
    }
    *result = a + b;
    return true;
}

static inline bool checked_strided_tensor_span(uint32_t   batches,
                                               uint32_t   heads,
                                               uint32_t   rows,
                                               uint64_t   row_bytes,
                                               uint64_t   batch_stride,
                                               uint64_t   head_stride,
                                               uint64_t   row_stride,
                                               uint64_t * result) {
    uint64_t batch_offset;
    uint64_t head_offset;
    uint64_t row_offset;
    uint64_t span;
    return checked_mul_u64(batches - 1U, batch_stride, &batch_offset) &&
           checked_mul_u64(heads - 1U, head_stride, &head_offset) &&
           checked_mul_u64(rows - 1U, row_stride, &row_offset) && checked_add_u64(batch_offset, head_offset, &span) &&
           checked_add_u64(span, row_offset, &span) && checked_add_u64(span, row_bytes, result);
}

static inline bool checked_tensor_bytes(uint32_t   d0,
                                        uint32_t   d1,
                                        uint32_t   d2,
                                        uint32_t   d3,
                                        uint64_t   element_bytes,
                                        uint64_t * result) {
    uint64_t value;
    return checked_mul_u64(d0, d1, &value) && checked_mul_u64(value, d2, &value) &&
           checked_mul_u64(value, d3, &value) && checked_mul_u64(value, element_bytes, result);
}

static inline bool valid_address_range(uint64_t base, uint64_t bytes) {
    return bytes <= (~uint64_t{ 0 }) - base;
}

static inline bool ranges_overlap(uint64_t a, uint64_t a_bytes, uint64_t b, uint64_t b_bytes) {
    return a < b + b_bytes && b < a + a_bytes;
}



struct Cluster {
    dayan_cluster_barrier_t barrier;
    uint64_t uram[4];
    uint32_t owner[4];
};
static_assert(sizeof(Cluster) <= kAtomicBytes, "Attention shared metadata exceeds allocation");

struct State {
    const FlashAttentionConfig * cfg;
    uint64_t q, k, v, out, uram;
    uint64_t blocks;
    uint32_t task, tasks, cores, lane, scale;
    long groups;
    bool split_kv;

    uint64_t work(uint64_t block) const { return task + block / groups * tasks; }
    uint32_t batch(uint64_t block) const { return (uint32_t) (work(block) / (split_kv ? cfg->kv_heads : cfg->q_heads)); }
    uint32_t head(uint64_t block) const {
        return split_kv ? (work(block) % cfg->kv_heads) * (cfg->q_heads / cfg->kv_heads) + block % groups * kBr : work(block) % cfg->q_heads;
    }
    long row(uint64_t block, long slice, long core) const {
        if (split_kv) {
            return 0;
        }
        return ((block % groups * kSlices + slice) * cores + core) * kBr;
    }
    long rows(uint64_t block, long slice) const {
        if (split_kv) {
            return slice ? 0 : extent(cfg->q_heads / cfg->kv_heads - block % groups * kBr, kBr);
        }
        return extent((long) cfg->q_len - row(block, slice, lane), kBr);
    }
    long key_begin(long core) const {
        const long visible = (cfg->flags & kFlagCausal) ? cfg->causal_offset + 1L : cfg->kv_len;
        return split_kv ? visible * core / 4 : 0;
    }
    long key_count(long core) const { return split_kv ? key_begin(core + 1) - key_begin(core) : cfg->kv_len; }
    long cols(long ki) const { return extent(key_count(lane) - ki * kBc, kBc); }
    long keys(uint64_t block, long slice, long core) const {
        if (split_kv) {
            return slice ? 0 : (key_count(core) + kBc - 1) / kBc;
        }
        const long first = row(block, slice, core);
        const long count = extent((long) cfg->q_len - first, kBr);
        if (!count) {
            return 0;
        }
        const long end = (cfg->flags & 1) ? cfg->causal_offset + first + count : cfg->kv_len;
        return (end + kBc - 1) / kBc;
    }
    long shared_keys(uint64_t block) const {
        if (split_kv) {
            return keys(block, 0, lane);
        }
        long count = 0;
        for (long slice = 0; slice < kSlices; ++slice) {
            for (long core = 0; core < cores; ++core) {
                const long n = keys(block, slice, core);
                count = n > count ? n : count;
            }
        }
        return count;
    }
    long users(uint64_t block, long ki) const {
        long count = 0;
        for (long slice = 0; slice < kSlices; ++slice) {
            count += ki < keys(block, slice, lane);
        }
        return count;
    }
    uint64_t q_address(uint64_t block, long slice) const {
        return q + batch(block) * cfg->q_batch_stride + head(block) * cfg->q_head_stride +
               row(block, slice, lane) * cfg->q_row_stride;
    }
    uint64_t kv_address(uint64_t block, long ki, bool value) const {
        const long kh = head(block) / (cfg->q_heads / cfg->kv_heads);
        const long first = key_begin(lane) + ki * kBc;
        return value ? v + batch(block) * cfg->v_batch_stride + kh * cfg->v_head_stride + first * cfg->v_row_stride :
                       k + batch(block) * cfg->k_batch_stride + kh * cfg->k_head_stride + first * cfg->k_row_stride;
    }
    uint64_t output_address(uint64_t block, long slice) const {
        return out + ((uint64_t) batch(block) * cfg->q_len * cfg->q_heads +
                      row(block, slice, lane) * cfg->q_heads + head(block)) * cfg->head_dim * sizeof(float);
    }
};

static inline long qk_swizzle(long head_dim) {
    // VisualSwizzle, 32 banks, 8 rows/cycle: D=64/128/192/256 has linear
    // peak 1/2/1/4; the source mappings below all reach peak 1. Vt has
    // stride 256 and reaches peak 1 from 2. Keep score/P and PV identity:
    // FlexVec consumes them without a swizzle iterator (peaks 4 and 2/4/2/8).
    return head_dim == 128 ? MAT_BATCH_SWIZZLE(5, 1, 5) :
           head_dim == 256 ? MAT_BATCH_SWIZZLE(5, 2, 5) : 0;
}
__device__ constexpr long kVSwizzle = MAT_BATCH_SWIZZLE(5, 1, 5);

static void load_q(void * argument) {
    const State & s = *(State *) argument;
    const long uram = (long) dayan_uram_base();
    configure_lsu_rows<0>(s.cfg->head_dim * 2, s.split_kv ? s.cfg->q_head_stride : s.cfg->q_row_stride, s.cfg->head_dim * 2);
    __tx_lsu_tcsr_write<LSU_URAMWRITESWIZZLEMBS>(qk_swizzle(s.cfg->head_dim));
    for (uint64_t block = 0; block < s.blocks; ++block) {
        for (long slice = 0; slice < kSlices; ++slice) {
            const long rows = s.rows(block, slice);
            if (!rows) {
                continue;
            }
            check(dayan_resource_wait_free(q_resource(slice)));
            set_lsu_rows<0>(rows);
            __tx_lsu_dma_d2u<0>(uram + slice * kContextBytes + kCtxQ, s.q_address(block, slice));
            check(dayan_resource_produce_lsu(q_resource(slice), 1, 0));
        }
    }
    dayan_wait_lsu();
}

template <int Cores, bool Value> static void load_kv(void * argument) {
    const State & s = *(State *) argument;
    const long uram = (long) dayan_uram_base();
    uint64_t phases = 0;
    uint64_t generation = 0;
    if constexpr (Value) {
        configure_v_transpose<0>(s.cfg->head_dim, s.cfg->v_row_stride);
        __tx_lsu_tcsr_write<LSU_URAMWRITESWIZZLEMBS>(kVSwizzle);
    } else {
        configure_lsu_rows<0>(s.cfg->head_dim * 2, s.cfg->k_row_stride, s.cfg->head_dim * 2);
        __tx_lsu_tcsr_write<LSU_URAMWRITESWIZZLEMBS>(qk_swizzle(s.cfg->head_dim));
    }
    for (uint64_t block = 0; block < s.blocks; ++block) {
        const long keys = s.shared_keys(block);
        const uint64_t kv_base = s.kv_address(block, 0, Value);
        const uint64_t row_stride = Value ? s.cfg->v_row_stride : s.cfg->k_row_stride;
        for (long ki = 0; ki < keys; ++ki, ++generation) {
            const long slot = generation & 1;
            const long users = s.users(block, ki);
            const long cols = s.cols(ki);
            const long offset = Value ? kUramVt0 + slot * kVtBytes : kUramK0 + slot * kKBytes;
            const auto local = kv_resource(Value, slot);
            const auto ready = DAYAN_SHARED_RESOURCE(2 * Value + slot);
            const uint64_t shared_mask = DAYAN_RESOURCE_MASK(ready);
            int role = DAYAN_RESOURCE_ROLE_PRODUCER;
            if constexpr (Cores == 4) {
                role = dayan_resource_wait(users ? DAYAN_RESOURCE_MASK(local) | shared_mask : 0, shared_mask, phases, 0);
            } else if (users) {
                check(dayan_resource_wait_free(local));
            }
            check(role);
            if (!users) {
                if constexpr (Cores == 4) {
                    check(dayan_resource_consume_lsu(shared_mask));
                }
            } else if (role == DAYAN_RESOURCE_ROLE_PRODUCER) {
                if constexpr (Cores == 4) {
                    ((Cluster *) shared)->owner[2 * Value + slot] = s.lane;
                }
                if constexpr (Value) {
                    const long end = (cols + 31) / 32 * 64;
                    if (end < kBc * 2) {
                        configure_lsu_rows<1>(kBc * 2 - end, 0, kBc * 2);
                        set_lsu_rows<1>(s.cfg->head_dim);
                        __tx_lsu_tcsr_write<LSU_GROUPSZ + LSU_URAMWRITESWIZZLEMBS>(kVSwizzle);
                        __tx_lsu_dma_u2u<1>(uram + offset + end, uram + kUramZero);
                    }
                    load_v_transposed<0>(kv_base + ki * kBc * row_stride, uram + offset, cols, row_stride);
                } else {
                    set_lsu_rows<0>(cols);
                    __tx_lsu_dma_d2u<0>(uram + offset, kv_base + ki * kBc * row_stride);
                }
                check(dayan_resource_produce_lsu(local, users, 0));
                if constexpr (Cores == 4) {
                    check(dayan_resource_produce_lsu(ready, Cores - 1, 0));
                }
            } else if (role == DAYAN_RESOURCE_ROLE_CONSUMER) {
                const Cluster & cluster = *(Cluster *) shared;
                const long owner = cluster.owner[2 * Value + slot];
                const long bytes = Value ? s.cfg->head_dim * kBc * 2 : cols * s.cfg->head_dim * 2;
                configure_lsu_rows<1>(bytes, bytes, bytes);
                __tx_lsu_dma_d2u<1>(uram + offset, cluster.uram[owner] + offset);
                check(dayan_resource_produce_lsu(local, users, shared_mask));
            } else {
                dayan_thread_exit(-22);
            }
            phases ^= shared_mask;
        }
    }
    dayan_wait_lsu();
}

template <typename Format> static void calculate_qk(const State & s) {
    uint64_t phases = 0;
    uint64_t generation = 0;
    // Children allocate TCSR groups from zero; the root uses the remaining MAT group.
    configure_mat<Format, kMatQK>(kBr, kBc, s.cfg->head_dim, s.cfg->head_dim * 2, s.cfg->head_dim * 2, kBc * 4);
    __tx_mat_tcsr_write<kMatQK * MAT_GROUPSZ + MAT_ASWIZZLEMBS>(qk_swizzle(s.cfg->head_dim));
    __tx_mat_tcsr_write<kMatQK * MAT_GROUPSZ + MAT_BSWIZZLEMBS>(qk_swizzle(s.cfg->head_dim));
    for (uint64_t block = 0; block < s.blocks; ++block) {
        const long keys = s.shared_keys(block);
        for (long ki = 0; ki < keys; ++ki, ++generation) {
            const auto k = kv_resource(false, generation & 1);
            const long slot = ki & 1;
            const long users = s.users(block, ki);
            for (long slice = 0, used = 0; slice < kSlices; ++slice) {
                const long slice_keys = s.keys(block, slice, s.lane);
                if (ki >= slice_keys) {
                    continue;
                }
                const auto q = q_resource(slice);
                const uint64_t inputs = DAYAN_RESOURCE_MASK(q) | DAYAN_RESOURCE_MASK(k);
                // Both tokens protect the in-place score/P tile through the PV read.
                const uint64_t outputs = DAYAN_RESOURCE_MASK(score_resource(slice, slot)) | DAYAN_RESOURCE_MASK(p_resource(slice, slot));
                check(dayan_resource_wait(outputs, inputs, phases, 0));
                __tx_mat_tcsr_write<kMatQK * MAT_GROUPSZ + MAT_M>(s.rows(block, slice) - 1);
                __tx_mat_tcsr_write<kMatQK * MAT_GROUPSZ + MAT_N>(s.cols(ki) - 1);
                const long base = s.uram + slice * kContextBytes;
                __tx_mat_mm<kMatQK>(s.uram + tile_offset(slice, slot) + kTileScore, base + kCtxQ, s.uram + kUramK0 + (generation & 1) * kKBytes);
                const bool last = ki + 1 == slice_keys;
                // Q and K are reusable after their last MAT read, before softmax or PV.
                check(dayan_resource_produce_mat(score_resource(slice, slot), 1, DAYAN_RESOURCE_MASK(k) | (last ? DAYAN_RESOURCE_MASK(q) : 0)));
                if (last) {
                    phases ^= DAYAN_RESOURCE_MASK(q);
                }
                if (++used == users) {
                    phases ^= DAYAN_RESOURCE_MASK(k);
                }
            }
        }
    }
    dayan_wait_mat();
}

static inline void initialize_vec() {
    configure_vec_grid(1);
    configure_vec_iterator<kVecV0Base>(0, 0, 8 * kBc * 4, 0, kBc * 4);
    configure_vec_iterator<kVecV1Base>(0, 0, kVecBytes, 0);
    configure_vec_iterator<kVecV2Base>(0, 0, kVecBytes, 0);
    configure_vec_iterator<kVecV3Base>(0, 0, 16 * 32, 0);
    // Iterator fields are unsigned; subtract row and column positions in VEC.
    configure_vec_iterator<VEC_S0_BASE>(0, 0, 8, 0);
    configure_vec_iterator<VEC_S1_BASE>(0, 0, 0, 8);
    __tx_vec_tcsr_write<VEC_S1_BASE>(kBr);
}

template <bool Mask, bool First> static inline void online_softmax(long base, long previous, long rows, uint32_t scale) {
    // FlexVecAttentionSpec: 131 instructions, plus scale, constants, alpha store, and optional masks.
    constexpr long instructions = Mask ? 231 : 167;
    __tx_vec_tcsr_write<kVecDim2>((rows + 7) / 8 - 1);
    begin_vec_block(instructions);
    __tx_vec_lsu_g2v_broadcast32<1, 0>(scale);
    __tx_vec_lsu_g2v_broadcast32<1, 0>(0xff7fffffU);
    __tx_vec_lsu_g2v_broadcast32<1, 0>(0);
    for (long part = 0; part < 16; ++part) {
        __tx_vec_lsu_u2v<0, 0>(base + kTileScore + part * 32);
        __tx_vec_vv_fp32_mul<0, 0, 18, 0>();
        if constexpr (Mask) {
            __tx_vec_lsu_u2s<0, 3>(base + kTileMask + part * 32);
            __tx_vec_vv_mux2<0, 17, 0, 0>();
        }
        if (part) {
            __tx_vec_vv_fp32_max<0, Mask ? 3 : 2, 0, 0>();
        }
    }
    for (long amount = 4; amount; amount /= 2) {
        __tx_vec_shuffle_xor<1, 0>(amount);
        __tx_vec_vv_fp32_max<0, 0, 16, 0>();
    }
    if constexpr (First) {
        __tx_vec_lsu_g2v_broadcast32<1, 0>(0xff7fffffU);
    } else {
        __tx_vec_lsu_u2v<1, 1>(previous + kTileM);
    }
    __tx_vec_vv_fp32_max<1, 0, 16, 0>();
    // Q1 ages: new M=0, old M=1, zero=5, floor=6, scale=7.
    for (long part = 0; part < 16; ++part) {
        __tx_vec_lsu_u2v<0, 0>(base + kTileScore + part * 32);
        __tx_vec_vv_fp32_mul<0, 0, 23, 0>();
        __tx_vec_vv_fp32_sub<0, 0, 16, 0>();
        __tx_vec_v_fp32_eru<0, 0, 0>();
        if constexpr (Mask) {
            __tx_vec_lsu_u2s<0, 3>(base + kTileMask + part * 32);
            __tx_vec_vv_mux2<0, 21, 0, 0>();
        }
        __tx_vec_lsu_v2u<0, 0>(base + kTileScore + part * 32);
        if (part) {
            __tx_vec_vv_fp32_add<0, Mask ? 5 : 4, 0, 0>();
        }
    }
    for (long amount = 4; amount; amount /= 2) {
        __tx_vec_shuffle_xor<1, 0>(amount);
        __tx_vec_vv_fp32_add<0, 0, 16, 0>();
    }
    __tx_vec_vv_fp32_sub<0, 20, 19, 0>();
    __tx_vec_v_fp32_eru<0, 0, 0>();
    __tx_vec_lsu_v2u<0, 2>(base + kTileAlpha);
    if constexpr (First) {
        __tx_vec_lsu_g2v_broadcast32<1, 0>(0);
    } else {
        __tx_vec_lsu_u2v<1, 1>(previous + kTileL);
    }
    __tx_vec_vv_fp32_mul<0, 0, 16, 0>();
    __tx_vec_vv_fp32_add<0, 3, 0, 0>();
    __tx_vec_lsu_v2u<0, 1>(base + kTileL);
    __tx_vec_lsu_v2u<20, 1>(base + kTileM);
    __asm__ volatile("" ::: "memory");
    __tx_vec_tcsr_write<kVecControl>(vec_control(instructions));
}

static bool prepare_mask(const State & s, long uram, uint64_t block, long slice, long ki) {
    const long rows = s.rows(block, slice);
    const long cols = s.cols(ki);
    const bool causal = !s.split_kv && (s.cfg->flags & kFlagCausal);
    const long first = s.row(block, slice, s.lane) + s.cfg->causal_offset - ki * kBc;
    if (rows % 8 == 0 && cols == kBc && (!causal || first >= kBc - 1)) {
        return false;
    }
    const bool tail = rows % 8 || cols < kBc;
    const long row_base = extent(first + kBr, kBr + kBc);
    const long instructions = (causal ? 6 : 1) + (rows % 8 ? 7 : 0) + (cols < kBc ? 7 : 0) + (tail ? 4 : 0) + 1;
    configure_vec_grid((rows + 7) / 8, kBc / 8);
    __tx_vec_tcsr_write<kVecV2Base + 3>(0);
    __tx_vec_tcsr_write<kVecV3Base + 4>(32);
    __tx_vec_tcsr_write<VEC_S0_BASE>(row_base);
    begin_vec_block(instructions);
    if (causal) {
        // Compare lane_col - lane_row with first + 8 * group - 8 * part.
        __tx_vec_s_iter_gets<0, 0, 0, 0>();
        __tx_vec_s_iter_gets<1, 0, 0, 1>();
        __tx_vec_s_int32_sub<0, 0, 1>();
        __tx_vec_lsu_s2v_broadcast32<0, 2, 0>();
        __tx_vec_lsu_u2v<1, 2>(uram + kUramMaskDelta);
        __tx_vec_vv_int32_cmp<16, 0, 0, 0b110>();
    } else {
        __tx_vec_lsu_g2s<0>(-1);
    }
    if (tail) {
        __tx_vec_lsu_g2v_broadcast32<0, 2>(0);
        __tx_vec_lsu_g2v_mask_broadcast32<0, 0, 0>(1);
    }
    if (rows % 8) {
        __tx_vec_s_iter_gets<1, 0, 0, 0>();
        __tx_vec_lsu_g2s<2>(rows + row_base);
        __tx_vec_s_int32_sub<1, 2, 1>();
        __tx_vec_lsu_s2v_broadcast32<1, 2, 1>();
        __tx_vec_lsu_u2v<1, 2>(uram + kUramMaskRows);
        __tx_vec_vv_int32_cmp<16, 17, 1, 0b011>();
        __tx_vec_lsu_g2v_mask_broadcast32<0, 0, 1>(0);
    }
    if (cols < kBc) {
        __tx_vec_s_iter_gets<1, 0, 0, 1>();
        __tx_vec_lsu_g2s<2>(cols + kBr);
        __tx_vec_s_int32_sub<1, 2, 1>();
        __tx_vec_lsu_s2v_broadcast32<1, 2, 1>();
        __tx_vec_lsu_u2v<1, 2>(uram + kUramMaskCols);
        __tx_vec_vv_int32_cmp<16, 17, 1, 0b011>();
        __tx_vec_lsu_g2v_mask_broadcast32<0, 0, 1>(0);
    }
    if (tail) {
        __tx_vec_lsu_g2v_broadcast32<1, 2>(0);
        __tx_vec_vv_int32_cmp<0, 16, 0, 0b001>();
    }
    __tx_vec_lsu_s2u<0, 3>(uram + tile_offset(slice, ki & 1) + kTileMask);
    __asm__ volatile("" ::: "memory");
    __tx_vec_tcsr_write<kVecControl>(vec_control(instructions));
    // Both softmax passes read the packed masks from URAM.
    dayan_wait_vec();
    __tx_vec_tcsr_write<kVecDim3>(0);
    __tx_vec_tcsr_write<kVecV2Base + 3>(kVecBytes);
    __tx_vec_tcsr_write<kVecV3Base + 4>(0);
    return true;
}

static void softmax(void * argument) {
    const State & s = *(State *) argument;
    const long uram = (long) dayan_uram_base();
    uint64_t phases = 0;
    initialize_vec();
    for (uint64_t block = 0; block < s.blocks; ++block) {
        for (long ki = 0; ki < s.shared_keys(block); ++ki) {
            for (long slice = 0; slice < kSlices; ++slice) {
                if (ki >= s.keys(block, slice, s.lane)) {
                    continue;
                }
                const long slot = ki & 1;
                const uint64_t score = DAYAN_RESOURCE_MASK(score_resource(slice, slot));
                const auto state = softmax_state_resource(slice, slot);
                const uint64_t current_state = DAYAN_RESOURCE_MASK(state);
                const auto output_state = output_state_resource(slice, slot);
                const uint64_t previous_state = ki == 0 ? 0 : DAYAN_RESOURCE_MASK(softmax_state_resource(slice, slot ^ 1));
                // Both readers must release M/L/Alpha before this storage is reused.
                check(dayan_resource_wait(current_state | DAYAN_RESOURCE_MASK(output_state), score | previous_state, phases, 0));
                const bool masked = prepare_mask(s, uram, block, slice, ki);
                const long base = uram + tile_offset(slice, slot);
                const long previous = uram + tile_offset(slice, slot ^ 1);
                const long rows = s.rows(block, slice);
                if (ki == 0) {
                    if (masked) online_softmax<true, true>(base, previous, rows, s.scale);
                    else online_softmax<false, true>(base, previous, rows, s.scale);
                } else {
                    if (masked) online_softmax<true, false>(base, previous, rows, s.scale);
                    else online_softmax<false, false>(base, previous, rows, s.scale);
                }
                const bool last = ki + 1 == s.keys(block, slice, s.lane);
                check(dayan_resource_produce_vec(p_resource(slice, slot), 1, score));
                check(dayan_resource_produce_vec(state, 1, previous_state));
                check(dayan_resource_produce_vec(output_state, 1, 0));
                phases ^= score | previous_state;
                if (last) {
                    // Consume every softmax generation, including the tail with no next tile.
                    check(dayan_resource_wait_ready(state, (phases & current_state) ? 1 : 0));
                    check(dayan_resource_consume_vec(current_state));
                    phases ^= current_state;
                }
            }
        }
    }
    dayan_wait_vec();
}

template <typename Format> static void calculate_pv(void * argument) {
    const State & s = *(State *) argument;
    uint64_t phases = 0;
    uint64_t generation = 0;
    configure_mat<Format, 0>(kBr, s.cfg->head_dim, kBc, kBc * 4, kBc * 2, s.cfg->head_dim * 4);
    constexpr long value_type = Format::kMatDataType == BaseNPUInstBuilder::DT_BF16 ?
        BaseNPUInstBuilder::DT_MATIN_BF16 : BaseNPUInstBuilder::DT_MATIN_FP16;
    __tx_mat_tcsr_write<MAT_DATATYPE>(MAT_DATATYPE_PAIR(BaseNPUInstBuilder::DT_MATIN_TF32, value_type));
    __tx_mat_tcsr_write<MAT_BSWIZZLEMBS>(kVSwizzle);
    for (uint64_t block = 0; block < s.blocks; ++block) {
        const long keys = s.shared_keys(block);
        for (long ki = 0; ki < keys; ++ki, ++generation) {
            const auto v = kv_resource(true, generation & 1);
            const long slot = ki & 1;
            const long users = s.users(block, ki);
            for (long slice = 0, used = 0; slice < kSlices; ++slice) {
                if (ki >= s.keys(block, slice, s.lane)) {
                    continue;
                }
                const uint64_t p = DAYAN_RESOURCE_MASK(p_resource(slice, slot));
                const uint64_t inputs = p | DAYAN_RESOURCE_MASK(v);
                check(dayan_resource_wait(DAYAN_RESOURCE_MASK(pv_resource(slice, slot)), inputs, phases, 0));
                __tx_mat_tcsr_write<MAT_M>(s.rows(block, slice) - 1);
                const long base = s.uram + tile_offset(slice, slot);
                __tx_mat_mm<0>(base + kTilePV, base + kTileScore, s.uram + kUramVt0 + (generation & 1) * kVtBytes);
                check(dayan_resource_produce_mat(pv_resource(slice, slot), 1, inputs));
                phases ^= p;
                if (++used == users) {
                    phases ^= DAYAN_RESOURCE_MASK(v);
                }
            }
        }
    }
    dayan_wait_mat();
}

template <bool First, bool Last> static inline void update_output(long base, long out) {
    constexpr long instructions = (First ? 2 : 6) + (Last ? 3 : 0);
    begin_vec_block(instructions);
    __tx_vec_lsu_u2v<0, 0>(base + kTilePV);
    if constexpr (!First) {
        __tx_vec_lsu_u2v<1, 0>(out);
        __tx_vec_lsu_u2v_broadcast32<1, 2>(base + kTileAlpha);
        __tx_vec_vv_fp32_mul<1, 17, 16, 0>();
        __tx_vec_vv_fp32_add<0, 0, 16, 0>();
    }
    if constexpr (Last) {
        __tx_vec_lsu_u2v_broadcast32<1, 2>(base + kTileL);
        __tx_vec_v_fp32_eru<1, 16, 1>();
        __tx_vec_vv_fp32_mul<0, 0, 16, 0>();
    }
    __tx_vec_lsu_v2u<0, 0>(out);
    __asm__ volatile("" ::: "memory");
    __tx_vec_tcsr_write<kVecControl>(vec_control(instructions));
}

static void merge_decode(const State & s, long uram, long base, long rows) {
    Cluster & cluster = *(Cluster *) shared;
    // The unused second Q slice holds peer state and merge weights.
    const long peer = uram + tile_offset(1, 0);
    const long alpha = peer + kTileAlpha;
    const long beta = peer + kTileMask;
    const long state_bytes = (rows + 7) / 8 * kVecBytes;
    check(dayan_cluster_barrier(4, &cluster.barrier));
    if (s.lane == 0) {
        for (long core = 1; core < 4; ++core) {
            const long slot = (s.keys(0, 0, core) - 1) & 1;
            const uint64_t source = cluster.uram[core];
            configure_lsu_rows<0>(state_bytes, state_bytes, state_bytes);
            __tx_lsu_dma_d2u<0>(peer + kTileM, source + tile_offset(0, slot) + kTileM);
            __tx_lsu_dma_d2u<0>(peer + kTileL, source + tile_offset(0, slot) + kTileL);
            configure_lsu_rows<0>(rows * s.cfg->head_dim * 4, 0, 0);
            __tx_lsu_dma_d2u<0>(uram + kContextBytes + kCtxO, source + kCtxO);
            dayan_wait_lsu();

            configure_vec_grid((rows + 7) / 8);
            configure_vec_iterator<kVecV0Base>(0, 0, kVecBytes, 0);
            begin_vec_block(16);
            __tx_vec_lsu_u2v<0, 0>(base + kTileM);
            __tx_vec_lsu_u2v<1, 0>(peer + kTileM);
            __tx_vec_vv_fp32_max<0, 0, 16, 0>();
            __tx_vec_lsu_v2u<0, 0>(base + kTileM);
            __tx_vec_vv_fp32_sub<0, 1, 0, 0>();
            __tx_vec_v_fp32_eru<0, 0, 0>();
            __tx_vec_lsu_v2u<0, 0>(alpha);
            __tx_vec_vv_fp32_sub<1, 16, 2, 0>();
            __tx_vec_v_fp32_eru<1, 16, 0>();
            __tx_vec_lsu_v2u<16, 0>(beta);
            __tx_vec_lsu_u2v<0, 0>(base + kTileL);
            __tx_vec_vv_fp32_mul<0, 0, 1, 0>();
            __tx_vec_lsu_u2v<1, 0>(peer + kTileL);
            __tx_vec_vv_fp32_mul<1, 16, 17, 0>();
            __tx_vec_vv_fp32_add<0, 0, 16, 0>();
            __tx_vec_lsu_v2u<0, 0>(base + kTileL);
            start_vec_block(16);
            dayan_wait_vec();

            configure_vec_grid(rows, s.cfg->head_dim / 64);
            configure_vec_iterator<kVecV0Base>(0, 0, s.cfg->head_dim * 4, kVecBytes);
            configure_vec_iterator<kVecV2Base>(0, 0, 32, 0);
            begin_vec_block(8);
            __tx_vec_lsu_u2v<0, 0>(uram + kCtxO);
            __tx_vec_lsu_u2v_broadcast32<1, 2>(alpha);
            __tx_vec_vv_fp32_mul<0, 0, 16, 0>();
            __tx_vec_lsu_u2v<1, 0>(uram + kContextBytes + kCtxO);
            __tx_vec_lsu_u2v_broadcast32<1, 2>(beta);
            __tx_vec_vv_fp32_mul<1, 16, 17, 0>();
            __tx_vec_vv_fp32_add<0, 0, 16, 0>();
            __tx_vec_lsu_v2u<0, 0>(uram + kCtxO);
            start_vec_block(8);
            dayan_wait_vec();
        }
        begin_vec_block(5);
        __tx_vec_lsu_u2v<0, 0>(uram + kCtxO);
        __tx_vec_lsu_u2v_broadcast32<1, 2>(base + kTileL);
        __tx_vec_v_fp32_eru<1, 16, 1>();
        __tx_vec_vv_fp32_mul<0, 0, 16, 0>();
        __tx_vec_lsu_v2u<0, 0>(uram + kCtxO);
        start_vec_block(5);
        dayan_wait_vec();
    }
    // Peer O and softmax state stay live until all remote reads finish.
    check(dayan_cluster_barrier(4, &cluster.barrier));
}

static void output(void * argument) {
    const State & s = *(State *) argument;
    const long uram = (long) dayan_uram_base();
    uint64_t phases = 0;
    uint64_t o_phases = 0;
    for (uint64_t block = 0; block < s.blocks; ++block) {
        if (block == 0 || s.split_kv) {
            configure_vec_grid(1, s.cfg->head_dim / 64);
            configure_vec_iterator<kVecV0Base>(0, 0, s.cfg->head_dim * 4, kVecBytes);
            configure_vec_iterator<kVecV2Base>(0, 0, 32, 0);
            configure_lsu_rows<0>(s.cfg->head_dim * 4, s.cfg->head_dim * 4, s.cfg->q_heads * s.cfg->head_dim * 4);
        }
        for (long ki = 0; ki < s.shared_keys(block); ++ki) {
            for (long slice = 0; slice < kSlices; ++slice) {
                const long keys = s.keys(block, slice, s.lane);
                if (ki >= keys) {
                    continue;
                }
                const long slot = ki & 1;
                const auto o = o_resource(slice, slot);
                const uint64_t current_o = DAYAN_RESOURCE_MASK(o);
                const uint64_t previous_o = ki == 0 ? 0 : DAYAN_RESOURCE_MASK(o_resource(slice, slot ^ 1));
                const uint64_t inputs = DAYAN_RESOURCE_MASK(pv_resource(slice, slot)) | DAYAN_RESOURCE_MASK(output_state_resource(slice, slot));
                // Two version tokens serialize updates to the single O allocation.
                check(dayan_resource_wait(current_o, inputs | previous_o, phases | (previous_o & ~o_phases), 0));
                const long rows = s.rows(block, slice);
                const long base = uram + tile_offset(slice, slot);
                const long out = uram + slice * kContextBytes + kCtxO;
                const bool last = ki + 1 == keys;
                const bool normalize = last && !s.split_kv;
                __tx_vec_tcsr_write<kVecDim2>(rows - 1);
                if (ki == 0) {
                    if (normalize) update_output<true, true>(base, out);
                    else update_output<true, false>(base, out);
                } else {
                    if (normalize) update_output<false, true>(base, out);
                    else update_output<false, false>(base, out);
                }
                const uint64_t merge_state = last && s.split_kv ? DAYAN_RESOURCE_MASK(output_state_resource(slice, slot)) : 0;
                check(dayan_resource_produce_vec(o, 1, (inputs & ~merge_state) | previous_o));
                phases ^= inputs;
                o_phases ^= current_o;
                if (last) {
                    check(dayan_resource_wait_ready(o, (o_phases & current_o) ? 0 : 1));
                    if (s.split_kv) {
                        merge_decode(s, uram, base, rows);
                        configure_lsu_rows<0>(s.cfg->head_dim * 4, s.cfg->head_dim * 4, s.cfg->head_dim * 4);
                    }
                    if (!s.split_kv || s.lane == 0) {
                        set_lsu_rows<0>(rows);
                        __tx_lsu_dma_u2d<0>(s.output_address(block, slice), out);
                    }
                    check(dayan_resource_consume_lsu(current_o | merge_state));
                    dayan_wait_lsu();
                }
            }
        }
    }
    dayan_wait_vec();
    dayan_wait_lsu();
}

enum FlashAttentionError {
    FA_INVALID_ARGUMENT = -1,
    FA_UNSUPPORTED_HEAD_DIM = -2,
    FA_UNSUPPORTED_HEAD_MAPPING = -3,
    FA_SIZE_OVERFLOW = -4,
    FA_UNALIGNED_V = -5,
    FA_OUTPUT_ALIAS = -6,
};

template <typename Format, int Cores, bool SplitKV = false>
static int run(const FlashAttentionConfig * cfg, const uint16_t * q, const uint16_t * k, const uint16_t * v, float * out) {
    static_assert(Cores == 1 || Cores == 4, "Attention supports one or four cores");
    if (!cfg) {
        return FA_INVALID_ARGUMENT;
    }
    const uint32_t batch = cfg->batch;
    const uint32_t q_heads = cfg->q_heads;
    const uint32_t kv_heads = cfg->kv_heads;
    const uint32_t q_len = cfg->q_len;
    const uint32_t kv_len = cfg->kv_len;
    const uint32_t head_dim = cfg->head_dim;
    const uint32_t scale_bits = cfg->scale_bits;
    const uint32_t causal_offset = cfg->causal_offset;
    const uint32_t flags = cfg->flags;
    const uint64_t q_row_stride = cfg->q_row_stride;
    const uint64_t q_head_stride = cfg->q_head_stride;
    const uint64_t q_batch_stride = cfg->q_batch_stride;
    const uint64_t k_row_stride = cfg->k_row_stride;
    const uint64_t k_head_stride = cfg->k_head_stride;
    const uint64_t k_batch_stride = cfg->k_batch_stride;
    const uint64_t v_row_stride = cfg->v_row_stride;
    const uint64_t v_head_stride = cfg->v_head_stride;
    const uint64_t v_batch_stride = cfg->v_batch_stride;
    const uint32_t task            = dayan_task_id(DAYAN_TASK_X);
    const uint32_t tasks           = dayan_task_dim(DAYAN_TASK_X);
    const uint32_t scale_magnitude = scale_bits & 0x7fffffffU;
    if (!q || !k || !v || !out || batch == 0 || q_heads == 0 || kv_heads == 0 || q_len == 0 || kv_len == 0 ||
        (flags & ~kFlagCausal) != 0 || ((flags & kFlagCausal) == 0 && causal_offset != 0) ||
        ((flags & kFlagCausal) != 0 && static_cast<uint64_t>(causal_offset) + q_len > kv_len) ||
        (scale_bits & 0x7f800000U) == 0x7f800000U || ((scale_bits & 0x80000000U) != 0 && scale_magnitude != 0) ||
        dayan_task_mode() != Cores ||
        (Cores == 4 && tasks != 4) || tasks == 0U || task >= tasks ||
        (Cores == 1 && dayan_task_dim(DAYAN_TASK_Y) != 1U) || !dayan_task_dim(DAYAN_TASK_Y) || dayan_task_dim(DAYAN_TASK_Z) != 1U) {
        return FA_INVALID_ARGUMENT;
    }
    if (head_dim == 0 || head_dim > kMaxHeadDim || (head_dim % kVecLanes) != 0) {
        return FA_UNSUPPORTED_HEAD_DIM;
    }
    const uint64_t row_bytes = static_cast<uint64_t>(head_dim) * sizeof(uint16_t);
    if (q_row_stride < row_bytes || k_row_stride < row_bytes || v_row_stride < row_bytes ||
        ((q_row_stride | q_head_stride | q_batch_stride | k_row_stride | k_head_stride | k_batch_stride | v_row_stride |
          v_head_stride | v_batch_stride) &
         63U) != 0U) {
        return FA_INVALID_ARGUMENT;
    }
    if ((q_heads % kv_heads) != 0) {
        return FA_UNSUPPORTED_HEAD_MAPPING;
    }
    if constexpr (SplitKV) {
        static_assert(Cores == 4, "Split-KV uses four cores");
        if (q_len != 1 || ((flags & kFlagCausal) ? causal_offset + 1L : kv_len) < 4) {
            return FA_INVALID_ARGUMENT;
        }
    }

    const float scale_log2e = u32_as_float(scale_bits) * 1.4426950408889634f;
    if ((float_as_u32(scale_log2e) & 0x7f800000U) == 0x7f800000U) {
        return FA_INVALID_ARGUMENT;
    }
    if ((reinterpret_cast<uint64_t>(v) & 63U) != 0) {
        return FA_UNALIGNED_V;
    }

    const uint64_t q_base_addr   = reinterpret_cast<uint64_t>(q);
    const uint64_t k_base_addr   = reinterpret_cast<uint64_t>(k);
    const uint64_t v_base_addr   = reinterpret_cast<uint64_t>(v);
    const uint64_t out_base_addr = reinterpret_cast<uint64_t>(out);
    uint64_t       q_span;
    uint64_t       k_span;
    uint64_t       v_span;
    uint64_t       out_bytes;
    uint64_t       traversal_generations;
    const uint64_t q_tiles  = (static_cast<uint64_t>(q_len) + kBr - 1) / kBr;
    const uint64_t q_groups = (q_tiles + (kSlices * Cores) - 1) / (kSlices * Cores);
    const uint64_t kv_tiles = (static_cast<uint64_t>(kv_len) + kBc - 1) / kBc;
    if (!checked_strided_tensor_span(batch, q_heads, q_len, row_bytes, q_batch_stride, q_head_stride, q_row_stride,
                                     &q_span) ||
        !checked_strided_tensor_span(batch, kv_heads, kv_len, row_bytes, k_batch_stride, k_head_stride, k_row_stride,
                                     &k_span) ||
        !checked_strided_tensor_span(batch, kv_heads, kv_len, row_bytes, v_batch_stride, v_head_stride, v_row_stride,
                                     &v_span) ||
        !checked_tensor_bytes(batch, q_heads, q_len, head_dim, sizeof(float), &out_bytes) ||
        !checked_mul_u64(batch, q_heads, &traversal_generations) ||
        !checked_mul_u64(traversal_generations, q_groups, &traversal_generations) ||
        !checked_mul_u64(traversal_generations, kv_tiles, &traversal_generations) ||
        !valid_address_range(q_base_addr, q_span) || !valid_address_range(k_base_addr, k_span) ||
        !valid_address_range(v_base_addr, v_span) || !valid_address_range(out_base_addr, out_bytes)) {
        return FA_SIZE_OVERFLOW;
    }
    if (ranges_overlap(out_base_addr, out_bytes, q_base_addr, q_span) ||
        ranges_overlap(out_base_addr, out_bytes, k_base_addr, k_span) ||
        ranges_overlap(out_base_addr, out_bytes, v_base_addr, v_span)) {
        return FA_OUTPUT_ALIAS;
    }
    const long uram = (long) dayan_uram_base();
    if (!uram || dayan_uram_size() < kUramBytes) {
        return FA_INVALID_ARGUMENT;
    }

    State s = {};
    s.cfg = cfg;
    s.split_kv = SplitKV;
    s.q = q_base_addr;
    s.k = k_base_addr;
    s.v = v_base_addr;
    s.out = out_base_addr;
    s.uram = (uint64_t) uram & (2 * 1024 * 1024 - 1);
    if (s.uram + kUramBytes > 2 * 1024 * 1024) {
        return FA_INVALID_ARGUMENT;
    }
    s.cores = Cores;
    s.lane = Cores == 4 ? task : 0;
    s.task = Cores == 4 ? dayan_task_id(DAYAN_TASK_Y) : task;
    s.tasks = Cores == 4 ? dayan_task_dim(DAYAN_TASK_Y) : tasks;
    s.scale = float_as_u32(scale_log2e);
    s.groups = SplitKV ? (q_heads / kv_heads + kBr - 1) / kBr : q_groups;
    const uint64_t work = (uint64_t) batch * (SplitKV ? kv_heads : q_heads);
    s.blocks = (work > s.task ? (work - s.task + s.tasks - 1) / s.tasks : 0) * s.groups;
    for (long i = 0; i < kVecLanes; ++i) {
        ((uint32_t *) (uram + kUramZero))[i] = 0;
        ((int32_t *) (uram + kUramMaskRows))[i] = i / 8;
        ((int32_t *) (uram + kUramMaskCols))[i] = i % 8;
        ((int32_t *) (uram + kUramMaskDelta))[i] = i % 8 - i / 8;
    }
    if constexpr (Cores == 4) {
        if (!shared) {
            return FA_INVALID_ARGUMENT;
        }
        Cluster & cluster = *(Cluster *) shared;
        cluster.uram[s.lane] = uram;
        check(dayan_cluster_barrier(4, &cluster.barrier));
    }
    dayan_thread_resources_t resources = {};
    resources.local_resource_count = kLocalResources;
    resources.shared_resource_count = Cores == 4 && !SplitKV ? kSharedResources : 0;
    resources.lsu_tcsr_count = 1;
    check(dayan_create_thread("attn-load-q", load_q, &s, &resources));
    resources.lsu_tcsr_count = 2;
    check(dayan_create_thread("attn-load-k", load_kv<SplitKV ? 1 : Cores, false>, &s, &resources));
    check(dayan_create_thread("attn-load-v", load_kv<SplitKV ? 1 : Cores, true>, &s, &resources));
    resources.lsu_tcsr_count = 0;
    resources.vec_tcsr_count = 1;
    check(dayan_create_thread("attn-softmax", softmax, &s, &resources));
    resources.vec_tcsr_count = 0;
    resources.mat_tcsr_count = 1;
    check(dayan_create_thread("attn-pv", calculate_pv<Format>, &s, &resources));
    resources.mat_tcsr_count = 0;
    resources.vec_tcsr_count = 1;
    resources.lsu_tcsr_count = 1;
    check(dayan_create_thread("attn-output", output, &s, &resources));
    calculate_qk<Format>(s);
    check(dayan_thread_join_all());
    if constexpr (Cores == 4) {
        check(dayan_cluster_barrier(4, &((Cluster *) shared)->barrier));
    }
    return 0;
}


} // namespace flash_attention_detail

#endif // __DAYAN_DEVICE__

__kernel__ void flash_attention_fp16_1c(const FlashAttentionConfig * config,
                                                const uint16_t * q,
                                                const uint16_t * k,
                                                const uint16_t * v,
                                                float * output,
                                                FlashAttentionResult * result) {
#if defined(__DAYAN_DEVICE__)
    if (config) {
        dayan_cache_invalidate(config, sizeof(*config));
    }
    const int status = flash_attention_detail::run<flash_attention_detail::Fp16Format, 1, false>(
        config, q, k, v, output);
    const uint32_t task = dayan_task_id(DAYAN_TASK_Y) * dayan_task_dim(DAYAN_TASK_X) + dayan_task_id(DAYAN_TASK_X);
    result[task].status = status;
    dayan_cache_flush(&result[task], sizeof(result[task]));
#endif
}

__kernel__ void flash_attention_bf16_1c(const FlashAttentionConfig * config,
                                                const uint16_t * q,
                                                const uint16_t * k,
                                                const uint16_t * v,
                                                float * output,
                                                FlashAttentionResult * result) {
#if defined(__DAYAN_DEVICE__)
    if (config) {
        dayan_cache_invalidate(config, sizeof(*config));
    }
    const int status = flash_attention_detail::run<flash_attention_detail::Bf16Format, 1, false>(
        config, q, k, v, output);
    const uint32_t task = dayan_task_id(DAYAN_TASK_Y) * dayan_task_dim(DAYAN_TASK_X) + dayan_task_id(DAYAN_TASK_X);
    result[task].status = status;
    dayan_cache_flush(&result[task], sizeof(result[task]));
#endif
}

__kernel__ void flash_attention_fp16_4c(const FlashAttentionConfig * config,
                                                const uint16_t * q,
                                                const uint16_t * k,
                                                const uint16_t * v,
                                                float * output,
                                                FlashAttentionResult * result) {
#if defined(__DAYAN_DEVICE__)
    if (config) {
        dayan_cache_invalidate(config, sizeof(*config));
    }
    const int status = flash_attention_detail::run<flash_attention_detail::Fp16Format, 4, false>(
        config, q, k, v, output);
    const uint32_t task = dayan_task_id(DAYAN_TASK_Y) * dayan_task_dim(DAYAN_TASK_X) + dayan_task_id(DAYAN_TASK_X);
    result[task].status = status;
    dayan_cache_flush(&result[task], sizeof(result[task]));
#endif
}

__kernel__ void flash_attention_bf16_4c(const FlashAttentionConfig * config,
                                                const uint16_t * q,
                                                const uint16_t * k,
                                                const uint16_t * v,
                                                float * output,
                                                FlashAttentionResult * result) {
#if defined(__DAYAN_DEVICE__)
    if (config) {
        dayan_cache_invalidate(config, sizeof(*config));
    }
    const int status = flash_attention_detail::run<flash_attention_detail::Bf16Format, 4, false>(
        config, q, k, v, output);
    const uint32_t task = dayan_task_id(DAYAN_TASK_Y) * dayan_task_dim(DAYAN_TASK_X) + dayan_task_id(DAYAN_TASK_X);
    result[task].status = status;
    dayan_cache_flush(&result[task], sizeof(result[task]));
#endif
}

__kernel__ void flash_attention_fp16_decode_4c(const FlashAttentionConfig * config,
                                                const uint16_t * q,
                                                const uint16_t * k,
                                                const uint16_t * v,
                                                float * output,
                                                FlashAttentionResult * result) {
#if defined(__DAYAN_DEVICE__)
    if (config) {
        dayan_cache_invalidate(config, sizeof(*config));
    }
    const int status = flash_attention_detail::run<flash_attention_detail::Fp16Format, 4, true>(
        config, q, k, v, output);
    const uint32_t task = dayan_task_id(DAYAN_TASK_Y) * dayan_task_dim(DAYAN_TASK_X) + dayan_task_id(DAYAN_TASK_X);
    result[task].status = status;
    dayan_cache_flush(&result[task], sizeof(result[task]));
#endif
}

__kernel__ void flash_attention_bf16_decode_4c(const FlashAttentionConfig * config,
                                                const uint16_t * q,
                                                const uint16_t * k,
                                                const uint16_t * v,
                                                float * output,
                                                FlashAttentionResult * result) {
#if defined(__DAYAN_DEVICE__)
    if (config) {
        dayan_cache_invalidate(config, sizeof(*config));
    }
    const int status = flash_attention_detail::run<flash_attention_detail::Bf16Format, 4, true>(
        config, q, k, v, output);
    const uint32_t task = dayan_task_id(DAYAN_TASK_Y) * dayan_task_dim(DAYAN_TASK_X) + dayan_task_id(DAYAN_TASK_X);
    result[task].status = status;
    dayan_cache_flush(&result[task], sizeof(result[task]));
#endif
}

DAYAN_KERNEL_NAMED_RESOURCE_TASK_DESC(flash_attention_fp16_1c, 1, 6,
    0, 1536U * 1024U, 6, 2, 2, 0, 30, 0);
DAYAN_KERNEL_NAMED_RESOURCE_TASK_DESC(flash_attention_bf16_1c, 1, 6,
    0, 1536U * 1024U, 6, 2, 2, 0, 30, 0);
DAYAN_KERNEL_NAMED_RESOURCE_TASK_DESC(flash_attention_fp16_4c, 1, 6,
    128, 1536U * 1024U, 6, 2, 2, 0, 30, 4);
DAYAN_KERNEL_NAMED_RESOURCE_TASK_DESC(flash_attention_bf16_4c, 1, 6,
    128, 1536U * 1024U, 6, 2, 2, 0, 30, 4);
DAYAN_KERNEL_NAMED_RESOURCE_TASK_DESC(flash_attention_fp16_decode_4c, 1, 6,
    128, 1536U * 1024U, 6, 2, 2, 0, 30, 0);
DAYAN_KERNEL_NAMED_RESOURCE_TASK_DESC(flash_attention_bf16_decode_4c, 1, 6,
    128, 1536U * 1024U, 6, 2, 2, 0, 30, 0);

#if !defined(__DAYAN_DEVICE__)

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

static constexpr uint32_t kTimeoutMs = NPU_TIMEOUT_INFINITE;

static int wait_and_destroy(npu_event_t *event) {
    if (!event) return -1;
    npu_event_info_t info = {};
    const int status = npu_event_wait(event, kTimeoutMs, &info);
    const int destroyed = npu_event_destroy(event);
    if (status < 0) return status;
    return info.status < 0 ? -1 : destroyed;
}

static int upload(npu_stream_t *stream, npu_buffer_t *buffer, const void *data, size_t bytes) {
    npu_event_t *event = nullptr;
    const int status = npu_memcpy_h2d_async(stream, buffer, 0, data, bytes, &event);
    return status < 0 ? status : wait_and_destroy(event);
}

static int download(npu_stream_t *stream, void *data, npu_buffer_t *buffer, size_t bytes) {
    npu_event_t *event = nullptr;
    const int status = npu_memcpy_d2h_async(stream, data, buffer, 0, bytes, &event);
    return status < 0 ? status : wait_and_destroy(event);
}

static void preserve_error(int candidate, int *status) {
    if (*status == 0 && candidate < 0) *status = candidate;
}

// Inputs are exact multiples of 1/16 in [-2, 2], representable in both formats.
// This keeps the CPU reference independent of a host half-conversion library.
static uint16_t encode_input(float value, bool bf16) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    if (bf16) return static_cast<uint16_t>(bits >> 16);
    if (value == 0) return 0;
    return static_cast<uint16_t>(((bits >> 16) & 0x8000U) |
                                ((((bits >> 23) & 0xffU) - 112U) << 10) |
                                ((bits >> 13) & 0x3ffU));
}

static float decode_input(uint16_t value, bool bf16) {
    if (bf16) {
        const uint32_t bits = static_cast<uint32_t>(value) << 16;
        float result;
        std::memcpy(&result, &bits, sizeof(result));
        return result;
    }
    const float magnitude = std::ldexp(static_cast<float>(1024 + (value & 1023)),
                                       static_cast<int>((value >> 10) & 31) - 25);
    return value == 0 ? 0 : (value & 0x8000U) ? -magnitude : magnitude;
}

static void initialize_tensor(std::vector<uint16_t> &tensor, uint32_t batches,
                              uint32_t heads, uint32_t rows, uint32_t dim,
                              uint64_t batch_stride, uint64_t head_stride,
                              uint64_t row_stride, uint32_t seed, bool bf16) {
    for (uint32_t b = 0; b < batches; ++b) {
        for (uint32_t h = 0; h < heads; ++h) {
            for (uint32_t r = 0; r < rows; ++r) {
                const size_t base = (b * batch_stride + h * head_stride + r * row_stride) / 2;
                for (uint32_t d = 0; d < dim; ++d) {
                    seed = seed * 1664525U + 1013904223U;
                    const float value = (static_cast<int>((seed >> 16) % 65) - 32) / 16.0f;
                    tensor[base + d] = encode_input(value, bf16);
                }
            }
        }
    }
}

static std::vector<float> reference_attention(const FlashAttentionConfig &cfg,
                                             const std::vector<uint16_t> &q,
                                             const std::vector<uint16_t> &k,
                                             const std::vector<uint16_t> &v, bool bf16) {
    std::vector<float> output(static_cast<size_t>(cfg.batch) * cfg.q_len * cfg.q_heads * cfg.head_dim);
    std::vector<double> scores(cfg.kv_len);
    float scale;
    std::memcpy(&scale, &cfg.scale_bits, sizeof(scale));
    for (uint32_t b = 0; b < cfg.batch; ++b) {
        for (uint32_t h = 0; h < cfg.q_heads; ++h) {
            const uint32_t kh = h / (cfg.q_heads / cfg.kv_heads);
            const size_t kb = (b * cfg.k_batch_stride + kh * cfg.k_head_stride) / 2;
            const size_t vb = (b * cfg.v_batch_stride + kh * cfg.v_head_stride) / 2;
            for (uint32_t row = 0; row < cfg.q_len; ++row) {
                const size_t qb = (b * cfg.q_batch_stride + h * cfg.q_head_stride + row * cfg.q_row_stride) / 2;
                const uint32_t visible = cfg.flags ? cfg.causal_offset + row + 1 : cfg.kv_len;
                double maximum = -std::numeric_limits<double>::infinity();
                for (uint32_t key = 0; key < visible; ++key) {
                    double dot = 0;
                    for (uint32_t d = 0; d < cfg.head_dim; ++d) {
                        dot += static_cast<double>(decode_input(q[qb + d], bf16)) *
                               decode_input(k[kb + key * cfg.k_row_stride / 2 + d], bf16);
                    }
                    scores[key] = dot * scale;
                    maximum = std::max(maximum, scores[key]);
                }
                double sum = 0;
                for (uint32_t key = 0; key < visible; ++key) {
                    scores[key] = std::exp(scores[key] - maximum);
                    sum += scores[key];
                }
                const size_t ob = ((static_cast<size_t>(b) * cfg.q_len + row) * cfg.q_heads + h) * cfg.head_dim;
                for (uint32_t d = 0; d < cfg.head_dim; ++d) {
                    double weighted = 0;
                    for (uint32_t key = 0; key < visible; ++key) {
                        weighted += scores[key] * decode_input(v[vb + key * cfg.v_row_stride / 2 + d], bf16);
                    }
                    output[ob + d] = static_cast<float>(weighted / sum);
                }
            }
        }
    }
    return output;
}

static int validate(const std::vector<float> &actual, const std::vector<float> &expected) {
    constexpr double abs_tolerance = 3e-2;
    constexpr double rel_tolerance = 3e-2;
    double max_abs = 0, max_rel = 0, squared_error = 0;
    size_t mismatches = 0;
    for (size_t i = 0; i < expected.size(); ++i) {
        const double error = std::fabs(static_cast<double>(actual[i]) - expected[i]);
        max_abs = std::max(max_abs, error);
        max_rel = std::max(max_rel, error / std::max(std::fabs(static_cast<double>(expected[i])), 1e-6));
        squared_error += error * error;
        // TF32 P operands and the VEC exp2/reciprocal approximations are compared
        // with an independent, double-precision full-attention reference.
        // Use relaxed tolerances for repeated Resource stability testing.
        if (!std::isfinite(actual[i]) || error > abs_tolerance + rel_tolerance * std::fabs(expected[i])) {
            if (mismatches < 6) {
                std::printf("mismatch O[%zu]: actual=%g expected=%g\n", i, actual[i], expected[i]);
            }
            ++mismatches;
        }
    }
    std::printf("[host] elements=%zu max_abs=%.8g max_rel=%.8g rmse=%.8g mismatches=%zu atol=%g rtol=%g\n",
                expected.size(), max_abs, max_rel, std::sqrt(squared_error / expected.size()), mismatches,
                abs_tolerance, rel_tolerance);
    return mismatches ? -1 : 0;
}

static void usage(const char *program) {
    std::printf("usage: %s [--bf16] [--cores 1|4] [--q-len N] [--kv-len N]\n"
                "       [--head-dim 64|128|192|256] [--q-heads N] [--kv-heads N]\n"
                "       [--batch N] [--causal] [--causal-offset N] [--split-kv]\n"
                "       [--padded] [--tasks N] [--repeat N]\n"
                "defaults: FP16, 1 core, B=1, Q=256, KV=512, Hq=4, Hkv=2, D=128\n"
                "causal offset defaults to KV-Q; four-core Q=1, KV>=512 uses Split-KV.\n", program);
}

__host__ int main(int argc, char **argv) {
    FlashAttentionConfig cfg = {};
    cfg.batch = 1;
    cfg.q_heads = 4;
    cfg.kv_heads = 2;
    cfg.q_len = 256;
    cfg.kv_len = 512;
    cfg.head_dim = 128;
    uint32_t cores = 1, tasks = 1, repeat = 1;
    bool bf16 = false, padded = false, split_kv = false, offset_set = false;
    for (int i = 1; i < argc; ++i) {
        const char *arg = argv[i];
        if (!std::strcmp(arg, "--help")) { usage(argv[0]); return 0; }
        if (!std::strcmp(arg, "--bf16")) { bf16 = true; continue; }
        if (!std::strcmp(arg, "--padded")) { padded = true; continue; }
        if (!std::strcmp(arg, "--split-kv")) { split_kv = true; continue; }
        if (!std::strcmp(arg, "--causal")) { cfg.flags = 1; continue; }
        uint32_t *value = nullptr;
        if (!std::strcmp(arg, "--cores")) value = &cores;
        else if (!std::strcmp(arg, "--tasks")) value = &tasks;
        else if (!std::strcmp(arg, "--repeat")) value = &repeat;
        else if (!std::strcmp(arg, "--q-len")) value = &cfg.q_len;
        else if (!std::strcmp(arg, "--kv-len")) value = &cfg.kv_len;
        else if (!std::strcmp(arg, "--head-dim")) value = &cfg.head_dim;
        else if (!std::strcmp(arg, "--q-heads")) value = &cfg.q_heads;
        else if (!std::strcmp(arg, "--kv-heads")) value = &cfg.kv_heads;
        else if (!std::strcmp(arg, "--batch")) value = &cfg.batch;
        else if (!std::strcmp(arg, "--causal-offset")) {
            value = &cfg.causal_offset;
            offset_set = true;
            cfg.flags = 1;
        }
        if (!value || ++i == argc || argv[i][0] < '0' || argv[i][0] > '9') {
            usage(argv[0]);
            return 2;
        }
        errno = 0;
        char *end = nullptr;
        const unsigned long parsed = std::strtoul(argv[i], &end, 10);
        if (errno || *end || parsed > 65536 || (parsed == 0 && value != &cfg.causal_offset)) {
            std::fprintf(stderr, "invalid value for %s: %s\n", arg, argv[i]);
            return 2;
        }
        *value = static_cast<uint32_t>(parsed);
    }
    if ((cores != 1 && cores != 4) || cfg.head_dim > 256 || cfg.head_dim % 64 ||
        cfg.q_heads % cfg.kv_heads || (cfg.flags && cfg.q_len > cfg.kv_len)) {
        std::fprintf(stderr, "unsupported core count, head dimension, head mapping or causal shape\n");
        return 2;
    }
    if (cfg.flags && !offset_set) cfg.causal_offset = cfg.kv_len - cfg.q_len;
    if (cfg.flags && cfg.causal_offset + cfg.q_len > cfg.kv_len) {
        std::fprintf(stderr, "causal-offset + q-len must not exceed kv-len\n");
        return 2;
    }
    split_kv |= cores == 4 && cfg.q_len == 1 && cfg.kv_len >= 512;
    if (split_kv && (cores != 4 || cfg.q_len != 1 || (cfg.flags ? cfg.causal_offset + 1 : cfg.kv_len) < 4)) {
        std::fprintf(stderr, "Split-KV requires four cores, q-len=1 and at least four visible keys\n");
        return 2;
    }
    const uint64_t padding = padded ? 64 : 0;
    cfg.q_row_stride = cfg.k_row_stride = cfg.v_row_stride = cfg.head_dim * 2 + padding;
    cfg.q_head_stride = cfg.q_len * cfg.q_row_stride + padding;
    cfg.k_head_stride = cfg.v_head_stride = cfg.kv_len * cfg.k_row_stride + padding;
    cfg.q_batch_stride = cfg.q_heads * cfg.q_head_stride + padding;
    cfg.k_batch_stride = cfg.v_batch_stride = cfg.kv_heads * cfg.k_head_stride + padding;
    const float scale = 1.0f / std::sqrt(static_cast<float>(cfg.head_dim));
    std::memcpy(&cfg.scale_bits, &scale, sizeof(scale));
    const uint64_t q_bytes = cfg.batch * cfg.q_batch_stride;
    const uint64_t kv_bytes = cfg.batch * cfg.k_batch_stride;
    const uint64_t output_elements = static_cast<uint64_t>(cfg.batch) * cfg.q_len * cfg.q_heads * cfg.head_dim;
    if (q_bytes + 2 * kv_bytes + output_elements * sizeof(float) > 128 * 1024 * 1024) {
        std::fprintf(stderr, "shape exceeds the demo's 128 MiB tensor limit\n");
        return 2;
    }

    std::vector<uint16_t> q(q_bytes / 2, encode_input(2.0f, bf16));
    std::vector<uint16_t> k(kv_bytes / 2, encode_input(2.0f, bf16));
    std::vector<uint16_t> v(kv_bytes / 2, encode_input(2.0f, bf16));
    initialize_tensor(q, cfg.batch, cfg.q_heads, cfg.q_len, cfg.head_dim,
                      cfg.q_batch_stride, cfg.q_head_stride, cfg.q_row_stride, 11, bf16);
    initialize_tensor(k, cfg.batch, cfg.kv_heads, cfg.kv_len, cfg.head_dim,
                      cfg.k_batch_stride, cfg.k_head_stride, cfg.k_row_stride, 23, bf16);
    initialize_tensor(v, cfg.batch, cfg.kv_heads, cfg.kv_len, cfg.head_dim,
                      cfg.v_batch_stride, cfg.v_head_stride, cfg.v_row_stride, 37, bf16);
    std::printf("[host] FlashAttention %s %uc B=%u Q=%u KV=%u Hq=%u Hkv=%u D=%u "
                "causal=%u offset=%u split_kv=%u padded=%u tasks=%u repeat=%u\n",
                bf16 ? "BF16" : "FP16", cores, cfg.batch, cfg.q_len, cfg.kv_len,
                cfg.q_heads, cfg.kv_heads, cfg.head_dim, cfg.flags, cfg.causal_offset,
                split_kv, padded, tasks, repeat);
    const std::vector<float> reference = reference_attention(cfg, q, k, v, bf16);
    std::vector<float> output(output_elements + 16, 12345.0f);
    std::vector<FlashAttentionResult> results(cores * tasks, FlashAttentionResult{-999});
    npu_runtime_options_t options;
    npu_runtime_options_init(&options);
    options.default_timeout_ms = kTimeoutMs;
    const char *socket = std::getenv("DAYAN_SERVER_SOCKET");
    if (socket && *socket) options.server_socket = socket;
    npu_runtime_t *runtime = nullptr;
    npu_context_t *context = nullptr;
    npu_stream_t *stream = nullptr;
    // config, Q, K, V, output (including a guard), per-task status
    npu_buffer_t *buffers[6] = {};
    const size_t bytes[] = {sizeof(cfg), q_bytes, kv_bytes, kv_bytes,
                           output.size() * sizeof(float), results.size() * sizeof(FlashAttentionResult)};
    const void *inputs[] = {&cfg, q.data(), k.data(), v.data()};
    int status = npu_runtime_open(&runtime, &options);
    if (status == 0) status = npu_context_create(runtime, nullptr, &context);
    if (status == 0) status = npu_stream_create(context, nullptr, &stream);
    for (size_t i = 0; i < 6 && status == 0; ++i) {
        npu_buffer_desc_t desc;
        npu_buffer_desc_init(&desc, bytes[i]);
        desc.alignment = 64;
        status = npu_buffer_create(context, &desc, &buffers[i]);
    }
    for (size_t i = 0; i < 4 && status == 0; ++i) {
        status = upload(stream, buffers[i], inputs[i], bytes[i]);
    }
    auto launch = bf16 ? flash_attention_bf16_1c : flash_attention_fp16_1c;
    if (cores == 4) launch = bf16 ? flash_attention_bf16_4c : flash_attention_fp16_4c;
    if (split_kv) launch = bf16 ? flash_attention_bf16_decode_4c : flash_attention_fp16_decode_4c;
    npu_kernel_launch_attrs_t attrs;
    npu_kernel_launch_attrs_init(&attrs, context, stream, "flash-attention");
    attrs.config.task_mode = cores == 4 ? NPUTaskMode::kFourCore : NPUTaskMode::kSingleCore;
    attrs.config.task_dim = cores == 4 ? NPUTaskDim3{4, tasks, 1} : NPUTaskDim3{tasks, 1, 1};
    attrs.timeout_ms = kTimeoutMs;
    for (uint32_t iteration = 0; iteration < repeat && status == 0; ++iteration) {
        std::fill(output.begin(), output.end(), 12345.0f);
        std::fill(results.begin(), results.end(), FlashAttentionResult{-999});
        status = upload(stream, buffers[4], output.data(), bytes[4]);
        if (status == 0) status = upload(stream, buffers[5], results.data(), bytes[5]);
        if (status == 0) status = launch(buffers[0], buffers[1], buffers[2], buffers[3], buffers[4], buffers[5], &attrs);
        if (status == 0) status = download(stream, results.data(), buffers[5], bytes[5]);
        if (status == 0) {
            for (size_t task = 0; task < results.size(); ++task) {
                if (results[task].status != 0) {
                    std::fprintf(stderr, "task %zu returned %lld\n", task, static_cast<long long>(results[task].status));
                    status = -1;
                }
            }
        }
        if (status == 0) status = download(stream, output.data(), buffers[4], bytes[4]);
        if (status == 0) status = validate(output, reference);
        if (status == 0) {
            for (size_t i = output_elements; i < output.size(); ++i) {
                if (output[i] != 12345.0f) {
                    std::fprintf(stderr, "output guard overwritten at %zu\n", i);
                    status = -1;
                }
            }
        }
    }
    for (auto *buffer : buffers) {
        if (buffer) preserve_error(npu_buffer_destroy(buffer), &status);
    }
    if (stream) preserve_error(npu_stream_destroy(stream), &status);
    if (context) preserve_error(npu_context_destroy(context), &status);
    if (runtime) preserve_error(npu_runtime_close(runtime), &status);
    std::printf("[host] %s: FlashAttention, status=%d\n", status == 0 ? "PASS" : "FAIL", status);
    return status == 0 ? 0 : 1;
}

#endif // !__DAYAN_DEVICE__
