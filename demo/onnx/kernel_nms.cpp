#include <dayan_kernel.h>

#if defined(__DAYAN_DEVICE__)
#include <dayan_intrinsics.h>
#include <dayan_sys.h>
#endif

#include <stdint.h>

#ifndef RVNPU_NMS_CACHE_SELECTED
#define RVNPU_NMS_CACHE_SELECTED 3
#endif
#ifndef RVNPU_NMS_LAZY_INVALIDATE
#define RVNPU_NMS_LAZY_INVALIDATE 1
#endif
#ifndef RVNPU_NMS_SINGLE_IOU_WAIT
#define RVNPU_NMS_SINGLE_IOU_WAIT 1
#endif

namespace {
#if defined(__DAYAN_DEVICE__)

__device__ constexpr uint32_t kLsuTileElements = 4096;
__device__ constexpr uint32_t kReduceTileElements = kLsuTileElements;
__device__ constexpr uint32_t kUramBytes = 256 * 1024;
__device__ constexpr uint32_t kIouBatch = 64;
__device__ constexpr uint32_t kIouUramBytes = 8 * 1024;
__device__ constexpr uint32_t kIouUramOffset = kUramBytes - kIouUramBytes;
__device__ constexpr uint32_t kMaxCachedBoxes =
    (kIouUramOffset - 128U) / sizeof(float);
__device__ constexpr uint32_t kWinnerBlock = 1024U;
__device__ constexpr uint32_t kWinnerCount =
    (kMaxCachedBoxes + kWinnerBlock - 1U) / kWinnerBlock;
__device__ constexpr long kLsuGroup = 0;
__device__ constexpr long kMacroGroup = 0;
static_assert(19U * kIouBatch * sizeof(float) <= kIouUramBytes,
              "NMS IoU buffers exceed reserved URAM");

static inline float negative_infinity() {
    union FloatBits {
        uint32_t bits;
        float value;
    } value = {0xff800000U};
    return value.value;
}

static inline uint32_t float_as_u32(float value) {
    union {
        uint32_t u;
        float f;
    } bits;
    bits.f = value;
    return bits.u;
}

static inline uint32_t align_up_64(uint32_t value) {
    return (value + 63U) & ~63U;
}

static inline void configure_lsu_linear(uint32_t block_count,
                                        uint32_t request_bytes) {
    __tx_lsu_tcsr_write<kLsuGroup * LSU_GROUPSZ + LSU_BLOCK>(0);
    __tx_lsu_tcsr_write<kLsuGroup * LSU_GROUPSZ + LSU_DIM0>(block_count - 1);
    __tx_lsu_tcsr_write<kLsuGroup * LSU_GROUPSZ + LSU_DIM1>(0);
    __tx_lsu_tcsr_write<kLsuGroup * LSU_GROUPSZ + LSU_DIM2>(0);
    __tx_lsu_tcsr_write<kLsuGroup * LSU_GROUPSZ + LSU_INBLOCKSTRIDE>(64);
    __tx_lsu_tcsr_write<kLsuGroup * LSU_GROUPSZ + LSU_INDIM0STRIDE>(64);
    __tx_lsu_tcsr_write<kLsuGroup * LSU_GROUPSZ + LSU_INDIM1STRIDE>(0);
    __tx_lsu_tcsr_write<kLsuGroup * LSU_GROUPSZ + LSU_INDIM2STRIDE>(0);
    __tx_lsu_tcsr_write<kLsuGroup * LSU_GROUPSZ + LSU_OUTBLOCKSTRIDE>(64);
    __tx_lsu_tcsr_write<kLsuGroup * LSU_GROUPSZ + LSU_OUTDIM0STRIDE>(64);
    __tx_lsu_tcsr_write<kLsuGroup * LSU_GROUPSZ + LSU_OUTDIM1STRIDE>(0);
    __tx_lsu_tcsr_write<kLsuGroup * LSU_GROUPSZ + LSU_OUTDIM2STRIDE>(0);
    __tx_lsu_tcsr_write<kLsuGroup * LSU_GROUPSZ + LSU_BLOCKTRANSPOSE>(6);
    __tx_lsu_tcsr_write<kLsuGroup * LSU_GROUPSZ + LSU_BLOCKPADDING>(0);
    __tx_lsu_tcsr_write<kLsuGroup * LSU_GROUPSZ + LSU_REQUESTPADDING>(
        request_bytes - 1);
    __tx_lsu_tcsr_write<kLsuGroup * LSU_GROUPSZ + LSU_REQUESTWRITESIZE>(request_bytes - 1);
    __tx_lsu_tcsr_write<kLsuGroup * LSU_GROUPSZ + LSU_PADDINGVALUE>(0);
    __tx_lsu_tcsr_write<kLsuGroup * LSU_GROUPSZ +
                        LSU_URAMWRITESWIZZLEMBS>(0);
    __tx_lsu_tcsr_write<kLsuGroup * LSU_GROUPSZ +
                        LSU_URAMREADSWIZZLEMBS>(0);
}

static inline void copy_aligned_to_uram(const float *source,
                                        uintptr_t destination,
                                        uint32_t elements) {
    const uint32_t bytes = elements * sizeof(float);
    const uint32_t full_blocks = bytes / 64U;
    const uint32_t tail_bytes = bytes % 64U;
    if (full_blocks != 0) {
        configure_lsu_linear(full_blocks, 64);
        __tx_lsu_dma_d2u<kLsuGroup>((long)destination, (long)source);
        dayan_wait_lsu();
    }
    if (tail_bytes != 0) {
        configure_lsu_linear(1, tail_bytes);
        __tx_lsu_dma_d2u<kLsuGroup>(
            ((long)(destination + full_blocks * 64U)),
            (long)((const char *)source + full_blocks * 64U));
        dayan_wait_lsu();
    }
}

static void cache_scores(const float *source, float *scratch,
                         uint32_t count) {
    for (uint32_t offset = 0; offset < count;
         offset += kLsuTileElements) {
        const uint32_t remaining = count - offset;
        const uint32_t length = remaining < kLsuTileElements
                                    ? remaining
                                    : kLsuTileElements;
        const float *source_tile = source + offset;
        if ((((uintptr_t)source_tile) & 63U) == 0) {
            copy_aligned_to_uram(source_tile, (uintptr_t)(scratch + offset), length);
        } else {
            for (uint32_t index = 0; index < length; ++index)
                scratch[offset + index] = source_tile[index];
        }
    }
}

static inline void configure_reduce(uint32_t elements) {
    __tx_macro_tcsr_write<kMacroGroup * MACRO_GROUPSZ + MACRO_DIM0>(
        elements - 1);
    __tx_macro_tcsr_write<kMacroGroup * MACRO_GROUPSZ + MACRO_DIM1>(0);
    __tx_macro_tcsr_write<kMacroGroup * MACRO_GROUPSZ + MACRO_DIM2>(0);
    __tx_macro_tcsr_write<kMacroGroup * MACRO_GROUPSZ + MACRO_DIM3>(0);
    __tx_macro_tcsr_write<kMacroGroup * MACRO_GROUPSZ + MACRO_SRC0SWIZZLEMBS>(
        0);
    __tx_macro_tcsr_write<kMacroGroup * MACRO_GROUPSZ + MACRO_SRC1SWIZZLEMBS>(
        0);
    __tx_macro_tcsr_write<kMacroGroup * MACRO_GROUPSZ + MACRO_DSTSWIZZLEMBS>(
        0);
}

static bool reduce_best(float *scratch, uint32_t count,
                        float *tile_value, uint32_t *tile_index,
                        float &best_value, uint32_t &best_index) {
    bool found = false;
    for (uint32_t offset = 0; offset < count;
         offset += kReduceTileElements) {
        const uint32_t remaining = count - offset;
        const uint32_t length = remaining < kReduceTileElements
                                    ? remaining
                                    : kReduceTileElements;
        configure_reduce(length);
        __tx_macro_reduce_max_fp32<kMacroGroup, 1>(
            (long)tile_value, (long)(scratch + offset),
            (long)tile_index);
        dayan_wait_macro();
        const float candidate = *tile_value;
        const uint32_t candidate_index = offset + *tile_index;
        if (!found || candidate > best_value) {
            found = true;
            best_value = candidate;
            best_index = candidate_index;
        }
    }
    return found;
}

static bool reduce_best_streamed(const float *scores, uint32_t count,
                                 float *scratch, float *tile_value,
                                 uint32_t *tile_index, float &best_value,
                                 uint32_t &best_index) {
    bool found = false;
    for (uint32_t offset = 0; offset < count;
         offset += kReduceTileElements) {
        const uint32_t remaining = count - offset;
        const uint32_t length = remaining < kReduceTileElements
                                    ? remaining
                                    : kReduceTileElements;
        cache_scores(scores + offset, scratch, length);
        configure_reduce(length);
        __tx_macro_reduce_max_fp32<kMacroGroup, 1>(
            (long)tile_value, (long)scratch, (long)tile_index);
        dayan_wait_macro();
        const float candidate = *tile_value;
        const uint32_t candidate_index = offset + *tile_index;
        if (!found || candidate > best_value) {
            found = true;
            best_value = candidate;
            best_index = candidate_index;
        }
    }
    return found;
}

template <long group>
static inline void macro_immediate_max(float *output, float value,
                                       float *input) {
    __tx_macro_elemwise_max_fp32<group, 2>(
        (long)output, (long)float_as_u32(value), (long)input);
}

template <long group>
static inline void macro_immediate_min(float *output, float value,
                                       float *input) {
    __tx_macro_elemwise_min_fp32<group, 2>(
        (long)output, (long)float_as_u32(value), (long)input);
}

struct IouBuffers {
    float *raw0;
    float *raw1;
    float *raw2;
    float *raw3;
    float *rhs_y1;
    float *rhs_x1;
    float *rhs_y2;
    float *rhs_x2;
    float *inter_y1;
    float *inter_x1;
    float *inter_h;
    float *inter_w;
    float *rhs_h;
    float *rhs_w;
    float *rhs_area;
    float *intersection;
    float *union_area;
    float *reciprocal;
    float *iou;
};

static IouBuffers iou_buffers() {
    float *base = (float *)((uint64_t)(uintptr_t)dayan_uram_base() + kIouUramOffset);
    return {
        base + 0 * kIouBatch,  base + 1 * kIouBatch,
        base + 2 * kIouBatch,  base + 3 * kIouBatch,
        base + 4 * kIouBatch,  base + 5 * kIouBatch,
        base + 6 * kIouBatch,  base + 7 * kIouBatch,
        base + 8 * kIouBatch,  base + 9 * kIouBatch,
        base + 10 * kIouBatch, base + 11 * kIouBatch,
        base + 12 * kIouBatch, base + 13 * kIouBatch,
        base + 14 * kIouBatch, base + 15 * kIouBatch,
        base + 16 * kIouBatch, base + 17 * kIouBatch,
        base + 18 * kIouBatch};
}

static void cache_selected_box(const float *box, bool center, float *out,
                               uint32_t stride) {
    float y1, x1, y2, x2;
    if (center) {
        y1 = box[1] - box[3] * 0.5f;
        x1 = box[0] - box[2] * 0.5f;
        y2 = box[1] + box[3] * 0.5f;
        x2 = box[0] + box[2] * 0.5f;
    } else {
        y1 = box[0] < box[2] ? box[0] : box[2];
        x1 = box[1] < box[3] ? box[1] : box[3];
        y2 = box[0] > box[2] ? box[0] : box[2];
        x2 = box[1] > box[3] ? box[1] : box[3];
    }
    out[0] = y1;
    out[stride] = x1;
    out[2U * stride] = y2;
    out[3U * stride] = x2;
    out[4U * stride] = (y2 - y1) * (x2 - x1);
}

static bool batch_iou_above_threshold(const float *candidate,
                                      const float *boxes,
                                      const int64_t *selected_indices,
                                      uint64_t selected_begin,
                                      uint32_t batch_size,
                                      float iou_threshold,
                                      bool center_point_box,
                                      const float *selected_cache,
                                      uint32_t cache_capacity,
                                      uint64_t class_begin) {
    IouBuffers buffer = iou_buffers();
    const uint64_t relative_begin = selected_begin - class_begin;
    const bool cached_batch = relative_begin + batch_size <= cache_capacity;
#if RVNPU_NMS_CACHE_SELECTED == 3
    if (cached_batch) {
        // SoA cache: directly consumed by MACRO. No scalar gather/copy into
        // IoU staging on every candidate. Each plane/64-box slice is aligned.
        float *base = const_cast<float *>(selected_cache) + relative_begin;
        buffer.rhs_y1 = base;
        buffer.rhs_x1 = base + cache_capacity;
        buffer.rhs_y2 = base + 2U * cache_capacity;
        buffer.rhs_x2 = base + 3U * cache_capacity;
        buffer.rhs_area = base + 4U * cache_capacity;
    } else
#endif
    for (uint32_t index = 0; index < batch_size; ++index) {
        if (cached_batch) {
            const float *box = selected_cache + (relative_begin + index) * 5U;
            buffer.rhs_y1[index] = box[0];
            buffer.rhs_x1[index] = box[1];
            buffer.rhs_y2[index] = box[2];
            buffer.rhs_x2[index] = box[3];
            buffer.rhs_area[index] = box[4];
            continue;
        }
        const uint32_t box_index = static_cast<uint32_t>(
            selected_indices[(selected_begin + index) * 3 + 2]);
        const float *box = boxes + (uint64_t)box_index * 4;
        if (center_point_box) {
            const float half_w = box[2] * 0.5f;
            const float half_h = box[3] * 0.5f;
            buffer.rhs_y1[index] = box[1] - half_h;
            buffer.rhs_x1[index] = box[0] - half_w;
            buffer.rhs_y2[index] = box[1] + half_h;
            buffer.rhs_x2[index] = box[0] + half_w;
        } else {
            buffer.raw0[index] = box[0];
            buffer.raw1[index] = box[1];
            buffer.raw2[index] = box[2];
            buffer.raw3[index] = box[3];
        }
    }

    float candidate_y1 = 0.0f;
    float candidate_x1 = 0.0f;
    float candidate_y2 = 0.0f;
    float candidate_x2 = 0.0f;
    if (center_point_box) {
        const float half_w = candidate[2] * 0.5f;
        const float half_h = candidate[3] * 0.5f;
        candidate_y1 = candidate[1] - half_h;
        candidate_x1 = candidate[0] - half_w;
        candidate_y2 = candidate[1] + half_h;
        candidate_x2 = candidate[0] + half_w;
    } else {
        candidate_y1 = candidate[0] < candidate[2]
                           ? candidate[0]
                           : candidate[2];
        candidate_x1 = candidate[1] < candidate[3]
                           ? candidate[1]
                           : candidate[3];
        candidate_y2 = candidate[0] > candidate[2]
                           ? candidate[0]
                           : candidate[2];
        candidate_x2 = candidate[1] > candidate[3]
                           ? candidate[1]
                           : candidate[3];
    }
    const float candidate_area =
        (candidate_y2 - candidate_y1) * (candidate_x2 - candidate_x1);

    // Exact broad phase: if every selected rectangle is disjoint, IoU is
    // zero and none can suppress at a nonnegative threshold. Avoid the
    // entire MACRO/SFU chain, without TopK truncation or changing borderline
    // IoU arithmetic. Keep the original path if any pair may intersect.
    if (iou_threshold >= 0.0f &&
        !(RVNPU_NMS_CACHE_SELECTED == 3 && cached_batch)) {
        bool may_intersect = false;
        const bool canonical = center_point_box || cached_batch;
        for (uint32_t index = 0; index < batch_size; ++index) {
            const float y0 = canonical ? buffer.rhs_y1[index] : buffer.raw0[index];
            const float x0 = canonical ? buffer.rhs_x1[index] : buffer.raw1[index];
            const float y1 = canonical ? buffer.rhs_y2[index] : buffer.raw2[index];
            const float x1 = canonical ? buffer.rhs_x2[index] : buffer.raw3[index];
            const bool disjoint =
                (y0 <= candidate_y1 && y1 <= candidate_y1) ||
                (y0 >= candidate_y2 && y1 >= candidate_y2) ||
                (x0 <= candidate_x1 && x1 <= candidate_x1) ||
                (x0 >= candidate_x2 && x1 >= candidate_x2);
            if (!disjoint) {
                may_intersect = true;
                break;
            }
        }
        if (!may_intersect)
            return false;
    }

    configure_reduce(batch_size);
    if (!center_point_box && !cached_batch) {
        __tx_macro_elemwise_min_fp32<kMacroGroup, 0>(
            (long)buffer.rhs_y1, (long)buffer.raw0, (long)buffer.raw2);
        __tx_macro_elemwise_min_fp32<kMacroGroup, 0>(
            (long)buffer.rhs_x1, (long)buffer.raw1, (long)buffer.raw3);
        __tx_macro_elemwise_max_fp32<kMacroGroup, 0>(
            (long)buffer.rhs_y2, (long)buffer.raw0, (long)buffer.raw2);
        __tx_macro_elemwise_max_fp32<kMacroGroup, 0>(
            (long)buffer.rhs_x2, (long)buffer.raw1, (long)buffer.raw3);
    }

    macro_immediate_max<kMacroGroup>(buffer.inter_y1, candidate_y1,
                                     buffer.rhs_y1);
    macro_immediate_max<kMacroGroup>(buffer.inter_x1, candidate_x1,
                                     buffer.rhs_x1);
    macro_immediate_min<kMacroGroup>(buffer.inter_h, candidate_y2,
                                     buffer.rhs_y2);
    macro_immediate_min<kMacroGroup>(buffer.inter_w, candidate_x2,
                                     buffer.rhs_x2);
    if (!cached_batch) {
        __tx_macro_elemwise_sub_fp32<kMacroGroup, 0>(
            (long)buffer.rhs_h, (long)buffer.rhs_y2, (long)buffer.rhs_y1);
        __tx_macro_elemwise_sub_fp32<kMacroGroup, 0>(
            (long)buffer.rhs_w, (long)buffer.rhs_x2, (long)buffer.rhs_x1);
    }

    __tx_macro_elemwise_sub_fp32<kMacroGroup, 0>(
        (long)buffer.inter_h, (long)buffer.inter_h,
        (long)buffer.inter_y1);
    __tx_macro_elemwise_sub_fp32<kMacroGroup, 0>(
        (long)buffer.inter_w, (long)buffer.inter_w,
        (long)buffer.inter_x1);
    if (!cached_batch)
        __tx_macro_elemwise_mul_fp32<kMacroGroup, 0>(
            (long)buffer.rhs_area, (long)buffer.rhs_h, (long)buffer.rhs_w);

    macro_immediate_max<kMacroGroup>(buffer.inter_h, 0.0f,
                                     buffer.inter_h);
    macro_immediate_max<kMacroGroup>(buffer.inter_w, 0.0f,
                                     buffer.inter_w);

    __tx_macro_elemwise_mul_fp32<kMacroGroup, 0>(
        (long)buffer.intersection, (long)buffer.inter_h,
        (long)buffer.inter_w);
    __tx_macro_elemwise_add_fp32<kMacroGroup, 2>(
        (long)buffer.union_area, (long)float_as_u32(candidate_area), (long)buffer.rhs_area);

    __tx_macro_elemwise_sub_fp32<kMacroGroup, 0>(
        (long)buffer.union_area, (long)buffer.union_area,
        (long)buffer.intersection);
#if !RVNPU_NMS_SINGLE_IOU_WAIT
    // Same-engine operations above are ordered. Only wait at the actual
    // MACRO -> CPU dependency, not between every arithmetic instruction.
    dayan_wait_macro();

    // 退化框只在这里修正 SFU 输入，其他 IoU 算术均由 NPU 完成。
    for (uint32_t index = 0; index < batch_size; ++index) {
        if (buffer.union_area[index] <= 0.0f) {
            buffer.union_area[index] = 1.0f;
            buffer.intersection[index] = 0.0f;
        }
    }
#endif

    __tx_macro_sfu_recip_fp32<kMacroGroup>(
        (long)buffer.reciprocal, (long)buffer.union_area);
    __tx_macro_elemwise_mul_fp32<kMacroGroup, 0>(
        (long)buffer.iou, (long)buffer.intersection,
        (long)buffer.reciprocal);
    dayan_wait_macro();

    for (uint32_t index = 0; index < batch_size; ++index) {
#if RVNPU_NMS_SINGLE_IOU_WAIT
        // Preserve the original zero-IoU rule for degenerate unions.
        // Any intermediate Inf/NaN from recip(0) is not used as an IoU.
        // This path is separately qualified on RTL with crossing zero-area
        // segments; no threshold approximation or epsilon is introduced.
        if (buffer.union_area[index] <= 0.0f) {
            if (0.0f > iou_threshold) return true;
            continue;
        }
#endif
        if (buffer.iou[index] > iou_threshold)
            return true;
    }
    return false;
}

static bool is_suppressed(const float *candidate, const float *boxes,
                          const int64_t *selected_indices,
                          uint64_t selected_begin, uint64_t selected_end,
                          float iou_threshold, bool center_point_box,
                          const float *selected_cache, uint32_t cache_capacity) {
    for (uint64_t begin = selected_begin; begin < selected_end;
         begin += kIouBatch) {
        const uint64_t remaining = selected_end - begin;
        const uint32_t batch_size = remaining < kIouBatch
                                        ? static_cast<uint32_t>(remaining)
                                        : kIouBatch;
        if (batch_iou_above_threshold(candidate, boxes, selected_indices,
                                      begin, batch_size, iou_threshold,
                                      center_point_box, selected_cache,
                                      cache_capacity, selected_begin))
            return true;
    }
    return false;
}

#endif
} // namespace

__kernel__ void rvnpu_nms_fp32(
    const float *boxes, float *scores, int64_t *selected_indices,
    uint64_t *selected_count, uint32_t num_boxes, uint32_t num_classes,
    uint32_t max_output_per_class, uint32_t iou_threshold_bits,
    uint32_t score_threshold_bits, uint32_t center_point_box) {
    union FloatBits {
        uint32_t bits;
        float value;
    } iou_value = {iou_threshold_bits}, score_value = {score_threshold_bits};
    const float iou_threshold = iou_value.value;
    const float score_threshold = score_value.value;
    if (center_point_box > 1)
        dayan_thread_exit(-2);
    if (max_output_per_class == 0 || num_boxes == 0 || num_classes == 0) {
        *selected_count = 0;
        return;
    }

#if !RVNPU_NMS_LAZY_INVALIDATE
    dayan_cache_invalidate(boxes, (uint64_t)num_boxes*4*sizeof(float));
    dayan_cache_invalidate(scores, (uint64_t)num_classes*num_boxes*sizeof(float));
#endif
    uint64_t count = 0;

    const bool cache_whole_class = num_boxes <= kMaxCachedBoxes;
    const uint32_t scratch_elements =
        cache_whole_class ? num_boxes : kReduceTileElements;
    float *scratch = (float *)(uintptr_t)dayan_uram_base();
    const uint32_t scratch_bytes =
        align_up_64(scratch_elements * sizeof(float));
    float *tile_value = (float *)((uint64_t)(uintptr_t)dayan_uram_base() + scratch_bytes);
    uint32_t *tile_index =
        (uint32_t *)((uint64_t)(uintptr_t)dayan_uram_base() + scratch_bytes + 64U);
    const bool uram_valid = scratch_bytes + 128U <= kIouUramOffset;
    if (!uram_valid)
        dayan_thread_exit(-1);
#if RVNPU_NMS_CACHE_SELECTED == 2
    // Experimental control: the same normalized records in the CPU-cached
    // stack, within the runtime's 64 KiB default stack. This distinguishes
    // arithmetic reuse from URAM placement/bank effects.
    float local_selected_cache[512U * 5U];
    float *selected_cache = local_selected_cache;
    uint32_t cache_capacity = 512U;
#else
    // Use the already reserved gap between scores/results and IoU scratch.
    // Never shrink the score-cache capacity or overlap the streamed buffer.
    float *selected_cache = (float *)((uintptr_t)scratch + scratch_bytes + 128U);
    const uint32_t free_cache_bytes = kIouUramOffset - scratch_bytes - 128U;
    uint32_t cache_capacity = RVNPU_NMS_CACHE_SELECTED
        ? free_cache_bytes / (5U * sizeof(float)) : 0U;
    if (cache_capacity > 512U) cache_capacity = 512U;
#if RVNPU_NMS_CACHE_SELECTED == 3
    cache_capacity &= ~15U;
    const uint32_t wanted = max_output_per_class >= 512U
        ? 512U : (max_output_per_class + 15U) & ~15U;
    if (cache_capacity > wanted) cache_capacity = wanted;
#endif
#endif

    for (uint32_t class_index = 0; class_index < num_classes;
         ++class_index) {
        const uint64_t class_scores_base =
            (uint64_t)class_index * num_boxes;
#if RVNPU_NMS_LAZY_INVALIDATE
        // Aligned scores are read by LSU, not by the scalar cache.
        // Only an unaligned class takes cache_scores' scalar fallback.
        if (((uintptr_t)(scores + class_scores_base) & 63U) != 0)
            dayan_cache_invalidate(scores + class_scores_base,
                                   (uint64_t)num_boxes * sizeof(float));
#endif
        if (cache_whole_class && num_boxes != 0)
            cache_scores(scores + class_scores_base, scratch, num_boxes);

        // Cache block winners. Removing one candidate changes only its
        // block; do not rescan the entire class on every NMS attempt.
        float winner_scores[kWinnerCount];
        uint32_t winner_indices[kWinnerCount];
        const uint32_t winner_count = cache_whole_class
            ? (num_boxes + kWinnerBlock - 1U) / kWinnerBlock : 0U;
        bool winners_ready = false;
        uint32_t dirty_block = winner_count;
        const uint64_t class_output_begin = count;
        uint32_t emitted = 0;
        for (uint32_t attempt = 0;
             attempt < num_boxes && emitted < max_output_per_class;
             ++attempt) {
            bool found = false;
            uint32_t best_box = 0;
            float best_score = negative_infinity();
            if (cache_whole_class) {
                // Most YOLO classes may have no score above threshold.
                // Test the first maximum with the original wide reduction;
                // do not build fine-grained winners for empty/Top1 classes.
                if (attempt == 0 || winner_count <= 1U) {
                    found = reduce_best(scratch, num_boxes, tile_value, tile_index,
                                        best_score, best_box);
                } else {
                    if (!winners_ready) {
                        for (uint32_t block = 0; block < winner_count; ++block) {
                            const uint32_t begin = block * kWinnerBlock;
                            const uint32_t length = num_boxes - begin < kWinnerBlock
                                ? num_boxes - begin : kWinnerBlock;
                            float value = negative_infinity();
                            uint32_t index = 0;
                            reduce_best(scratch + begin, length, tile_value, tile_index,
                                        value, index);
                            winner_scores[block] = value;
                            winner_indices[block] = begin + index;
                        }
                        winners_ready = true;
                    } else if (dirty_block < winner_count) {
                        const uint32_t begin = dirty_block * kWinnerBlock;
                        const uint32_t length = num_boxes - begin < kWinnerBlock
                            ? num_boxes - begin : kWinnerBlock;
                        float value = negative_infinity();
                        uint32_t index = 0;
                        reduce_best(scratch + begin, length, tile_value, tile_index,
                                    value, index);
                        winner_scores[dirty_block] = value;
                        winner_indices[dirty_block] = begin + index;
                    }
                    for (uint32_t block = 0; block < winner_count; ++block) {
                        if (!found || winner_scores[block] > best_score) {
                            found = true;
                            best_score = winner_scores[block];
                            best_box = winner_indices[block];
                            dirty_block = block;
                        }
                    }
                }
            } else {
                found = reduce_best_streamed(
                    scores + class_scores_base, num_boxes, scratch,
                    tile_value, tile_index, best_score, best_box);
            }
            found = found && best_score > score_threshold;
            if (!found)
                break;

            if (cache_whole_class) {
                scratch[best_box] = negative_infinity();
#if !RVNPU_NMS_LAZY_INVALIDATE
                scores[class_scores_base + best_box] = negative_infinity();
#endif
            } else {
                float *score_slot = scores + class_scores_base + best_box;
#if RVNPU_NMS_LAZY_INVALIDATE
                // The next streamed argmax uses DMA. Publish the mutation
                // before that DMA and preserve neighbouring score values.
                dayan_cache_invalidate(score_slot, sizeof(float));
#endif
                *score_slot = negative_infinity();
#if RVNPU_NMS_LAZY_INVALIDATE
                dayan_cache_flush(score_slot, sizeof(float));
#endif
            }

            const float *candidate = boxes + (uint64_t)best_box * 4;
#if RVNPU_NMS_LAZY_INVALIDATE
            // A selected box was already a candidate, so every box read
            // later by IoU has first passed through this invalidation.
            dayan_cache_invalidate(candidate, 4U * sizeof(float));
#endif
            if (is_suppressed(candidate, boxes, selected_indices,
                              class_output_begin, count, iou_threshold,
                              center_point_box != 0, selected_cache,
                              cache_capacity))
                continue;

            if (emitted < cache_capacity)
                cache_selected_box(candidate, center_point_box != 0,
                    selected_cache + (RVNPU_NMS_CACHE_SELECTED == 3 ? emitted : emitted * 5U),
                    RVNPU_NMS_CACHE_SELECTED == 3 ? cache_capacity : 1U);
            selected_indices[count * 3] = 0;
            selected_indices[count * 3 + 1] = class_index;
            selected_indices[count * 3 + 2] = best_box;
            ++count;
            ++emitted;
        }
    }
    *selected_count = count;
    return;
}

DAYAN_KERNEL_TASK_DESC(rvnpu_nms_fp32, 4, 0, 0, 256U * 1024U, 1, 0, 0, 1);
