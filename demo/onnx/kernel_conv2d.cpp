#include <dayan_kernel.h>
#include <stdint.h>

struct Conv2dTileConfig {
    uint32_t ci, co, wi, hi, wo, ho, wk, hk;
    uint32_t pad_l, pad_u, stride_w, stride_h, dilation_w, dilation_h;
    uint32_t input_stride_w, input_stride_h;
    uint32_t weight_stride_w, weight_stride_h, weight_stride_co;
    uint32_t output_stride_w, output_stride_h;
    uint32_t tile_h, tile_w, tile_ci, tile_co;
    uint32_t pipeline;
};
static_assert(sizeof(Conv2dTileConfig) == 26U * sizeof(uint32_t),
              "Conv host/device ABI mismatch");

#if defined(RVNPU_CONV_PROFILE)
// Profiling-only ABI; task records are cache-line isolated and published after timing ends.
struct alignas(64) Conv2dProfileRecord {
    uint64_t start, end, call_id, task, hart, completed, phase[16];
};
static_assert(sizeof(Conv2dProfileRecord) == 192, "Conv timing ABI mismatch");
enum ConvPhase {
    kSetup, kInputCache, kWeightCache, kOutputCache, kClear, kInputIssue,
    kWeightLoad, kMatConfig, kMatWait, kOutputStore, kOutputFlush, kLsuWait,
    kBody, kTiles, kMatCalls, kWeightHits
};
#endif

enum : uint32_t { kConvUramBytes = 1536U * 1024U };

#if !defined(__DAYAN_DEVICE__) || defined(CONV_HOST_MODEL)
#include <algorithm>
#include <array>
#include <limits>

namespace conv2d_host_detail {
static uint64_t tile_bytes(const Conv2dTileConfig &c)
{
    const auto align = [](uint64_t x) { return (x + 63) & ~uint64_t(63); };
    const uint64_t ih = (uint64_t(c.tile_h) - 1) * c.stride_h +
                         (uint64_t(c.hk) - 1) * c.dilation_h + 1;
    const uint64_t iw = (uint64_t(c.tile_w) - 1) * c.stride_w +
                         (uint64_t(c.wk) - 1) * c.dilation_w + 1;
    return 2 * (align(ih * iw * c.tile_ci * 2) +
        align(uint64_t(c.hk) * c.wk * c.tile_ci * c.tile_co * 2) +
        align(uint64_t(c.tile_h) * c.tile_w * c.tile_co * 4));
}

// Static ranking only, not predicted cycles or bandwidth. Penalize padded
// channels and the heaviest core; retain the full Ci reduction on one core.
static long double tile_score(const Conv2dTileConfig &c)
{
    struct Work { uint64_t bytes = 0, calls = 0, padded = 0, useful = 0; };
    std::array<Work, 4> cores{};
    const uint64_t nh = (uint64_t(c.ho) + c.tile_h - 1) / c.tile_h;
    const uint64_t nw = (uint64_t(c.wo) + c.tile_w - 1) / c.tile_w;
    const uint64_t nc = (uint64_t(c.co) + c.tile_co - 1) / c.tile_co;
    const uint64_t nk = (uint64_t(c.ci) + c.tile_ci - 1) / c.tile_ci;
    if (nh * nw > 65536 || nc > 65536 / (nh * nw))
        return std::numeric_limits<long double>::infinity();
    uint64_t id = 0;
    for (uint64_t y = 0; y < nh; ++y) {
        const uint64_t y0 = y * c.ho / nh, y1 = (y + 1) * c.ho / nh;
        for (uint64_t x = 0; x < nw; ++x) {
            const uint64_t x0 = x * c.wo / nw, x1 = (x + 1) * c.wo / nw;
            const int64_t iy0 = int64_t(y0 * c.stride_h) - c.pad_u;
            const int64_t ix0 = int64_t(x0 * c.stride_w) - c.pad_l;
            const int64_t iy1 = int64_t((y1 - 1) * c.stride_h +
                                       (c.hk - 1) * c.dilation_h + 1) - c.pad_u;
            const int64_t ix1 = int64_t((x1 - 1) * c.stride_w +
                                       (c.wk - 1) * c.dilation_w + 1) - c.pad_l;
            const uint64_t ih = std::max<int64_t>(
                0, std::min<int64_t>(iy1, c.hi) - std::max<int64_t>(iy0, 0));
            const uint64_t iw = std::max<int64_t>(
                0, std::min<int64_t>(ix1, c.wi) - std::max<int64_t>(ix0, 0));
            for (uint64_t z = 0; z < nc; ++z, ++id) {
                auto &work = cores[id % 4];
                const uint64_t channels = std::min<uint64_t>(c.tile_co, c.co - z * c.tile_co);
                const uint64_t pixels = (y1 - y0) * (x1 - x0);
                work.bytes += ih * iw * c.ci * 2 +
                    uint64_t(c.hk) * c.wk * c.ci * channels * 2 + pixels * channels * 4;
                work.calls += nk;
                work.padded += pixels * c.tile_co * nk * c.tile_ci * c.hk * c.wk;
                work.useful += pixels * channels * c.ci * c.hk * c.wk;
            }
        }
    }
    uint64_t bytes = 0, calls = 0, padded = 0, useful = 0;
    for (const auto &work : cores) {
        bytes = std::max(bytes, work.bytes); calls = std::max(calls, work.calls);
        padded = std::max(padded, work.padded); useful += work.useful;
    }
    // 4 KiB/issue is an uncalibrated control-cost proxy, not bus traffic.
    return (bytes + calls * 4096.L) * (4.L * padded / useful);
}
} // namespace conv2d_host_detail

// Host helper is packaged with the operator, not a separate tuning tool.
bool conv2d_plan_tiles(Conv2dTileConfig *config)
{
    if (!config || !config->ci || !config->co || !config->ho || !config->wo ||
        !config->hk || config->hk > 7 || !config->wk || config->wk > 7 ||
        !config->stride_h || config->stride_h > 3 ||
        !config->stride_w || config->stride_w > 3 ||
        !config->dilation_h || config->dilation_h > 3 ||
        !config->dilation_w || config->dilation_w > 3)
        return false;
    Conv2dTileConfig best = *config;
    long double score = std::numeric_limits<long double>::infinity();
    for (uint32_t ci : {32U, 64U, 128U}) {
        if (ci > 32 && ci / 2 >= config->ci) continue;
        for (uint32_t co : {16U, 32U, 64U, 128U}) {
            if (co > 16 && co / 2 >= config->co) continue;
            for (uint32_t h : {4U, 8U, 12U, 16U, 20U, 24U, 28U, 32U})
                for (uint32_t w : {4U, 8U, 12U, 16U, 20U, 24U, 28U, 32U}) {
                    Conv2dTileConfig c = *config;
                    c.tile_h = std::min(h, c.ho); c.tile_w = std::min(w, c.wo);
                    c.tile_ci = ci; c.tile_co = co; c.pipeline = 1;
                    const uint64_t bytes = conv2d_host_detail::tile_bytes(c);
                    if (bytes > kConvUramBytes) continue;
                    const long double candidate = conv2d_host_detail::tile_score(c);
                    if (candidate < score ||
                        (candidate == score && bytes < conv2d_host_detail::tile_bytes(best))) {
                        best = c; score = candidate;
                    }
                }
        }
    }
    if (score == std::numeric_limits<long double>::infinity()) return false;
    *config = best;
    return true;
}
#endif

#if defined(__DAYAN_DEVICE__)
#include <dayan_intrinsics.h>
#include <dayan_sys.h>

namespace conv2d_tile_detail {

#if defined(RVNPU_CONV_PROFILE)
static __attribute__((always_inline)) inline uint64_t cycle_stamp()
{
    uint64_t cycle;
    __asm__ volatile("rdcycle %0" : "=r"(cycle) :: "memory");
    return cycle;
}

struct PhaseScope {
    Conv2dProfileRecord *record;
    unsigned phase;
    uint64_t start;
    PhaseScope(Conv2dProfileRecord *r, unsigned p)
        : record(r), phase(p), start(cycle_stamp()) {}
    ~PhaseScope() { record->phase[phase] += cycle_stamp() - start; }
};
#define CONV_PHASE(s, id) PhaseScope phase_scope((s).profile, id)
#define CONV_COUNT(s, id) (++(s).profile->phase[id])

static void report_cycles(Conv2dProfileRecord *records, uint64_t call_id,
                          Conv2dProfileRecord &value)
{
    const uint32_t task = dayan_task_id(DAYAN_TASK_X);
    uint64_t hart;
    __asm__ volatile("csrr %0, mhartid" : "=r"(hart));
    Conv2dProfileRecord *record = records + task;
    // The buffer was cleared by Host. Invalidate before this core's private write.
    dayan_cache_invalidate(record, sizeof(*record));
    value.call_id = call_id; value.task = task; value.hart = hart; value.completed = 1;
    *record = value;
    dayan_cache_flush(record, sizeof(*record));
}
#else
#define CONV_PHASE(s, id) ((void)0)
#define CONV_COUNT(s, id) ((void)0)
#endif

static void check(int status)
{
    if (status < 0)
        dayan_thread_exit(status);
}
static void configure_lsu_linear(uint32_t bytes)
{
    __tx_lsu_tcsr_write<LSU_BLOCK>(0);
    __tx_lsu_tcsr_write<LSU_DIM0>(0);
    __tx_lsu_tcsr_write<LSU_DIM1>(0);
    __tx_lsu_tcsr_write<LSU_DIM2>(0);
    __tx_lsu_tcsr_write<LSU_INBLOCKSTRIDE>(64);
    __tx_lsu_tcsr_write<LSU_INDIM0STRIDE>(64);
    __tx_lsu_tcsr_write<LSU_INDIM1STRIDE>(64);
    __tx_lsu_tcsr_write<LSU_OUTBLOCKSTRIDE>(64);
    __tx_lsu_tcsr_write<LSU_OUTDIM0STRIDE>(64);
    __tx_lsu_tcsr_write<LSU_OUTDIM1STRIDE>(64);
    __tx_lsu_tcsr_write<LSU_BLOCKTRANSPOSE>(6);
    __tx_lsu_tcsr_write<LSU_BLOCKPADDING>(0);
    __tx_lsu_tcsr_write<LSU_REQUESTREADSIZE>(bytes - 1U);
    __tx_lsu_tcsr_write<LSU_REQUESTWRITESIZE>(bytes - 1U);
}

static bool input_needs_clear(const Conv2dTileConfig &c,
                              int64_t input_y0, int64_t input_x0,
                              uint32_t input_h, uint32_t input_w,
                              uint32_t ci_count, uint32_t tile_ci)
{
    const int64_t y0 = input_y0 - c.pad_u, x0 = input_x0 - c.pad_l;
    // Skip only when the DMA will overwrite every byte read by MAT.
    return ci_count != tile_ci || y0 < 0 || x0 < 0 ||
           y0 + input_h > c.hi || x0 + input_w > c.wi;
}

// Each element is one <=64-byte channel slice; the two dimensions gather or
// scatter pixels/channels and rows/kernel positions without implicit padding.
static void configure_lsu_rect(uint32_t bytes, uint32_t columns, uint32_t rows,
                               uint32_t source_column, uint32_t source_row,
                               uint32_t destination_column, uint32_t destination_row)
{
    configure_lsu_linear(bytes);
    __tx_lsu_tcsr_write<LSU_DIM0>(columns - 1U);
    __tx_lsu_tcsr_write<LSU_DIM1>(rows - 1U);
    __tx_lsu_tcsr_write<LSU_INDIM0STRIDE>(source_column);
    __tx_lsu_tcsr_write<LSU_INDIM1STRIDE>(source_row);
    __tx_lsu_tcsr_write<LSU_OUTDIM0STRIDE>(destination_column);
    __tx_lsu_tcsr_write<LSU_OUTDIM1STRIDE>(destination_row);
}

static void start_channel_rect(uint64_t destination, uint64_t source,
                               uint32_t bytes, uint32_t columns, uint32_t rows,
                               uint32_t source_column, uint32_t source_row,
                               uint32_t destination_column, uint32_t destination_row,
                               bool to_uram)
{
    // Wide-channel path: DIM0=64B channel chunks,
    // DIM1=pixels/output channels, DIM2=rows/kernel positions.
    // A partial last chunk uses an explicit second transfer, never overreads.
    if (bytes > 64U) {
        const uint32_t chunks = bytes / 64U;
        configure_lsu_linear(64);
        __tx_lsu_tcsr_write<LSU_DIM0>(chunks - 1U);
        __tx_lsu_tcsr_write<LSU_DIM1>(columns - 1U);
        __tx_lsu_tcsr_write<LSU_DIM2>(rows - 1U);
        __tx_lsu_tcsr_write<LSU_INDIM0STRIDE>(64);
        __tx_lsu_tcsr_write<LSU_INDIM1STRIDE>(source_column);
        __tx_lsu_tcsr_write<LSU_INDIM2STRIDE>(source_row);
        __tx_lsu_tcsr_write<LSU_OUTDIM0STRIDE>(64);
        __tx_lsu_tcsr_write<LSU_OUTDIM1STRIDE>(destination_column);
        __tx_lsu_tcsr_write<LSU_OUTDIM2STRIDE>(destination_row);
        if (to_uram)
            __tx_lsu_dma_d2u<0>((long)destination, (long)source);
        else
            __tx_lsu_dma_u2d<0>((long)destination, (long)source);
        bytes %= 64U;
        if (!bytes)
            return;
        dayan_wait_lsu();
        source += chunks * 64U;
        destination += chunks * 64U;
    }
    configure_lsu_rect(bytes, columns, rows, source_column, source_row,
                       destination_column, destination_row);
    if (to_uram)
        __tx_lsu_dma_d2u<0>((long)destination, (long)source);
    else
        __tx_lsu_dma_u2d<0>((long)destination, (long)source);
}

alignas(64) __device__ static const uint8_t kZeroDmaBlock[64] = {};

static void clear_uram_rect(uint64_t destination, uint32_t row_bytes,
                            uint32_t rows, uint32_t row_stride,
                            uint32_t copies = 1, uint32_t copy_stride = 0)
{
    const uint64_t base = (uint64_t)(uintptr_t)dayan_uram_base();
    const uint64_t size = dayan_uram_size();
    const uint64_t extent = (uint64_t)(rows ? rows - 1U : 0U) * row_stride +
                            (uint64_t)(copies ? copies - 1U : 0U) * copy_stride + row_bytes;
    if (!row_bytes || !rows || !copies || copies > 2 ||
        (row_bytes & 63U) || (destination & 63U) ||
        (row_stride & 63U) || (copy_stride & 63U) ||
        destination < base || destination - base > size ||
        extent > size - (destination - base))
        check(-22);
    // Re-read the immutable zero block. No implicit padding or DRAM overread.
    configure_lsu_rect(64, row_bytes / 64U, rows, 0, 0, 64, row_stride);
    if (copies > 1) {
        __tx_lsu_tcsr_write<LSU_DIM2>(copies - 1U);
        __tx_lsu_tcsr_write<LSU_INDIM2STRIDE>(0);
        __tx_lsu_tcsr_write<LSU_OUTDIM2STRIDE>(copy_stride);
    }
    __tx_lsu_dma_d2u<0>((long)destination, (long)(uintptr_t)kZeroDmaBlock);
    dayan_wait_lsu();
}

static void clear_uram_dma(uint64_t destination, uint32_t bytes)
{
    // Slots are independently rounded to 64 bytes; never clear the next slot.
    clear_uram_rect(destination, (bytes + 63U) & ~63U, 1, 0);
}

static bool start_input_rect(const Conv2dTileConfig &c, const uint16_t *input,
                             uint64_t destination, int64_t input_y0, int64_t input_x0,
                             uint32_t input_h, uint32_t input_w, uint32_t tile_ci,
                             uint32_t ci0, uint32_t ci_count)
{
    const int64_t y0 = input_y0 - c.pad_u, x0 = input_x0 - c.pad_l;
    const int64_t y1 = y0 + input_h, x1 = x0 + input_w;
    const int64_t vy0 = y0 < 0 ? 0 : y0, vx0 = x0 < 0 ? 0 : x0;
    const int64_t vy1 = y1 > c.hi ? c.hi : y1;
    const int64_t vx1 = x1 > c.wi ? c.wi : x1;
    if (vy0 >= vy1 || vx0 >= vx1)
        return false; // Explicitly cleared halo, no DMA and no zero dimensions.
    const uint8_t *source = (const uint8_t *)input +
        (uint64_t)vy0 * c.input_stride_h + (uint64_t)vx0 * c.input_stride_w + ci0 * 2U;
    destination += ((uint64_t)(vy0 - y0) * input_w + (uint64_t)(vx0 - x0)) * tile_ci * 2U;
    start_channel_rect(destination, (uint64_t)(uintptr_t)source, ci_count * 2U,
                       (uint32_t)(vx1 - vx0), (uint32_t)(vy1 - vy0),
                       c.input_stride_w, c.input_stride_h,
                       tile_ci * 2U, input_w * tile_ci * 2U, true);
    return true;
}

static void load_weight_rect(const Conv2dTileConfig &c, const uint16_t *weight,
                             uint64_t destination, uint32_t tile_ci, uint32_t tile_co,
                             uint32_t ci0, uint32_t ci_count,
                             uint32_t co0, uint32_t co_count)
{
    const uint8_t *source = (const uint8_t *)weight +
        (uint64_t)co0 * c.weight_stride_co + ci0 * 2U;
    start_channel_rect(destination, (uint64_t)(uintptr_t)source,
                       ci_count * 2U, co_count, c.hk * c.wk,
                       c.weight_stride_co, c.weight_stride_w,
                       tile_ci * 2U, tile_ci * tile_co * 2U, true);
    dayan_wait_lsu();
}

static bool packed_pointwise(const Conv2dTileConfig &c)
{
    return c.hk == 1 && c.wk == 1 && c.stride_h == 1 && c.stride_w == 1;
}

static void configure_mat(const Conv2dTileConfig &c, uint32_t ci,
                          uint32_t co, uint32_t wi, uint32_t hi,
                          uint32_t wo, uint32_t ho)
{
    // A packed 1x1/stride-1 tile is a single dense pixel axis. Avoid restarting
    // MAT's Ho subtasks for every short row (and padding each row separately).
    // Input DMA already packs rows; post() scatters the packed output rows.
    if (packed_pointwise(c)) {
        wi = wo = wo * ho;
        hi = ho = 1;
    }
    // RTL dimensions are terminal indices. Config/layout dimensions stay
    // actual counts; only these eight TCSR writes subtract one.
    __tx_mat_tcsr_write<CONV_CI>(ci - 1U);
    __tx_mat_tcsr_write<CONV_WI>(wi - 1U);
    __tx_mat_tcsr_write<CONV_HI>(hi - 1U);
    __tx_mat_tcsr_write<CONV_WO>(wo - 1U);
    __tx_mat_tcsr_write<CONV_HO>(ho - 1U);
    __tx_mat_tcsr_write<CONV_WK>(c.wk - 1U);
    __tx_mat_tcsr_write<CONV_HK>(c.hk - 1U);
    __tx_mat_tcsr_write<CONV_CO>(co - 1U);
    __tx_mat_tcsr_write<CONV_WISTRIDE>(ci * 2U);
    __tx_mat_tcsr_write<CONV_HISTRIDE>(wi * ci * 2U);
    __tx_mat_tcsr_write<CONV_WKSTRIDE>(co * ci * 2U);
    __tx_mat_tcsr_write<CONV_HKSTRIDE>(c.wk * co * ci * 2U);
    __tx_mat_tcsr_write<CONV_COSTRIDE>(ci * 2U);
    __tx_mat_tcsr_write<CONV_WOSTRIDE>(co * 4U);
    __tx_mat_tcsr_write<CONV_HOSTRIDE>(c.tile_w * co * 4U);
    __tx_mat_tcsr_write<CONV_ISWIZZLEMBS>(0);
    __tx_mat_tcsr_write<CONV_KSWIZZLEMBS>(0);
    __tx_mat_tcsr_write<CONV_OSWIZZLEMBS>(0);
    __tx_mat_tcsr_write<CONV_DATATYPE>(0x02);
    __tx_mat_tcsr_write<CONV_PADL>(0);
    __tx_mat_tcsr_write<CONV_PADU>(0);
    __tx_mat_tcsr_write<CONV_DILATIONW>(c.dilation_w);
    __tx_mat_tcsr_write<CONV_DILATIONH>(c.dilation_h);
    __tx_mat_tcsr_write<CONV_STRIDEW>(c.stride_w);
    __tx_mat_tcsr_write<CONV_STRIDEH>(c.stride_h);
    __tx_mat_tcsr_write<CONV_STRIDEDILATIONW>(c.dilation_w * ci * 2U);
    __tx_mat_tcsr_write<CONV_STRIDEDILATIONH>(c.dilation_h * wi * ci * 2U);
}

static bool config_valid(const Conv2dTileConfig *config)
{
    if (!config)
        return false;
    if (
        !config->ci || !config->co || !config->wi || !config->hi ||
        !config->wo || !config->ho || !config->wk || !config->hk ||
        !config->tile_h || !config->tile_w || !config->tile_ci ||
        !config->tile_co || config->tile_ci < 32 || config->tile_ci > 128 ||
        config->tile_co < 16 || config->tile_co > 128 ||
        (config->tile_ci & (config->tile_ci - 1U)) ||
        (config->tile_co & (config->tile_co - 1U)) ||
        config->tile_h > 32 || config->tile_w > 32 ||
        !config->stride_h || !config->stride_w ||
        !config->dilation_h || !config->dilation_w ||
        config->hk > 7 || config->wk > 7 ||
        config->stride_h > 3 || config->stride_w > 3 ||
        config->dilation_h > 3 || config->dilation_w > 3 ||
        config->ci > 0x7fffffffU || config->co > 0x7fffffffU ||
        config->hi > 0x7fffffffU || config->wi > 0x7fffffffU ||
        config->ho > 0x7fffffffU || config->wo > 0x7fffffffU ||
        config->pipeline > 1)
        return false;
    if (config->input_stride_w != (uint64_t)config->ci * 2 ||
        config->input_stride_h != (uint64_t)config->wi * config->input_stride_w ||
        config->weight_stride_co != (uint64_t)config->ci * 2 ||
        config->weight_stride_w != (uint64_t)config->co * config->weight_stride_co ||
        config->weight_stride_h != (uint64_t)config->wk * config->weight_stride_w ||
        config->output_stride_w != (uint64_t)config->co * 4 ||
        config->output_stride_h != (uint64_t)config->wo * config->output_stride_w)
        return false;
    return true;
}

struct Node {
    uint32_t oh, ow, co, rows, cols, channels;
    uint32_t input_h, input_w, output_at;
    int64_t input_y, input_x;
};

struct State {
    const Conv2dTileConfig *cfg;
    const uint16_t *input, *weight;
    uint32_t *output;
    uint64_t uram;
    uint32_t input_span, weight_span, output_span, weight_base, output_base;
    uint32_t buffers, tiles_w, tiles_h, tiles_co;
    // Slot tags are private to this invocation/core. A hit also preserves the
    // padded zeros; clearing a resident tail would destroy its valid weights.
    uint32_t weight_ci[2], weight_co[2];
    bool weight_valid[2];
    Node pending;
    bool has_output;
#if defined(RVNPU_CONV_PROFILE)
    Conv2dProfileRecord *profile;
#endif
};

static uint32_t extent(uint32_t remaining, uint32_t size)
{
    return remaining < size ? remaining : size;
}

// Each Task owns a complete output tile, including its entire Ci reduction.
static Node node(const State &s, uint64_t tile_id, uint64_t local_tile)
{
    const Conv2dTileConfig &c = *s.cfg;
    Node n = {};
    const uint64_t spatial = tile_id / s.tiles_co;
    n.co = (uint32_t)(tile_id % s.tiles_co) * c.tile_co;
    // Split the complete axis evenly instead of producing e.g. 20+1.
    const uint64_t y = spatial / s.tiles_w, x = spatial % s.tiles_w;
    n.oh = (uint32_t)(y * c.ho / s.tiles_h);
    n.ow = (uint32_t)(x * c.wo / s.tiles_w);
    n.rows = (uint32_t)((y + 1) * c.ho / s.tiles_h) - n.oh;
    n.cols = (uint32_t)((x + 1) * c.wo / s.tiles_w) - n.ow;
    n.channels = extent(c.co - n.co, c.tile_co);
    n.input_y = (int64_t)n.oh * c.stride_h;
    n.input_x = (int64_t)n.ow * c.stride_w;
    n.input_h = (n.rows - 1U) * c.stride_h + (c.hk - 1U) * c.dilation_h + 1U;
    n.input_w = (n.cols - 1U) * c.stride_w + (c.wk - 1U) * c.dilation_w + 1U;
    n.output_at = s.output_base + (local_tile % s.buffers) * s.output_span;
    return n;
}

static bool weight_resident(const State &s, const Node &n, uint32_t ci, uint32_t slot)
{
    return s.weight_valid[slot] && s.weight_ci[slot] == ci &&
           s.weight_co[slot] == n.co;
}

static void clear_input_slot(const State &s, const Node &n, uint32_t ci, uint32_t slot)
{
    CONV_PHASE(s, kClear);
    const Conv2dTileConfig &c = *s.cfg;
    const uint32_t count = extent(c.ci - ci, c.tile_ci);
    if (input_needs_clear(c, n.input_y, n.input_x, n.input_h, n.input_w,
                         count, c.tile_ci)) {
        const uint64_t destination = s.uram + slot * s.input_span;
        const uint32_t pixel_bytes = c.tile_ci * 2U;
        const uint32_t row_bytes = n.input_w * pixel_bytes;
        const int64_t y0 = n.input_y - c.pad_u, x0 = n.input_x - c.pad_l;
        const auto clip = [](int64_t value, uint32_t limit) -> uint32_t {
            return value < 0 ? 0U : value > limit ? limit : (uint32_t)value;
        };
        const uint32_t top = clip(-y0, n.input_h);
        const uint32_t bottom = clip((int64_t)c.hi - y0, n.input_h);
        const uint32_t left = clip(-x0, n.input_w);
        const uint32_t right = clip((int64_t)c.wi - x0, n.input_w);
        if (count != c.tile_ci || top >= bottom || left >= right) {
            // Channel tails also need zeros inside every valid pixel.
            clear_uram_dma(destination, n.input_h * row_bytes);
        } else {
            // Only the halo survives the real-input DMA. Do not clear the
            // interior merely to overwrite it. Pair symmetric opposite strips
            // in DIM2, avoiding four issue/wait round trips on a padded tile.
            if (top && top == n.input_h - bottom) {
                clear_uram_rect(destination, row_bytes, top, row_bytes,
                                2, bottom * row_bytes);
            } else {
                if (top)
                    clear_uram_rect(destination, row_bytes, top, row_bytes);
                if (bottom < n.input_h)
                    clear_uram_rect(destination + (uint64_t)bottom * row_bytes,
                                    row_bytes, n.input_h - bottom, row_bytes);
            }
            if (left && left == n.input_w - right) {
                clear_uram_rect(destination + (uint64_t)top * row_bytes,
                                left * pixel_bytes, bottom - top, row_bytes,
                                2, right * pixel_bytes);
            } else {
                if (left)
                    clear_uram_rect(destination + (uint64_t)top * row_bytes,
                                    left * pixel_bytes, bottom - top, row_bytes);
                if (right < n.input_w)
                    clear_uram_rect(destination + (uint64_t)top * row_bytes +
                                    right * pixel_bytes, (n.input_w - right) * pixel_bytes,
                                    bottom - top, row_bytes);
            }
        }
    }
}

static void clear_weight_slot(const State &s, const Node &n, uint32_t ci, uint32_t slot)
{
    const Conv2dTileConfig &c = *s.cfg;
    if (!weight_resident(s, n, ci, slot) &&
        (extent(c.ci - ci, c.tile_ci) != c.tile_ci || n.channels != c.tile_co)) {
        CONV_PHASE(s, kClear);
        clear_uram_dma(s.uram + s.weight_base + slot * s.weight_span,
                       c.hk * c.wk * c.tile_ci * c.tile_co * 2U);
    }
}

static void clear_slot(const State &s, const Node &n, uint32_t ci, uint32_t slot)
{
    clear_input_slot(s, n, ci, slot);
    clear_weight_slot(s, n, ci, slot);
}

static bool start_input(const State &s, const Node &n, uint32_t ci, uint32_t slot)
{
    CONV_PHASE(s, kInputIssue);
    const Conv2dTileConfig &c = *s.cfg;
    return start_input_rect(c, s.input, s.uram + slot * s.input_span,
                            n.input_y, n.input_x, n.input_h, n.input_w,
                            c.tile_ci, ci, extent(c.ci - ci, c.tile_ci));
}

static void load_weight(State &s, const Node &n, uint32_t ci, uint32_t slot)
{
    if (weight_resident(s, n, ci, slot)) {
        CONV_COUNT(s, kWeightHits);
        return;
    }
    CONV_PHASE(s, kWeightLoad);
    const Conv2dTileConfig &c = *s.cfg;
    load_weight_rect(c, s.weight, s.uram + s.weight_base + slot * s.weight_span,
                     c.tile_ci, c.tile_co, ci, extent(c.ci - ci, c.tile_ci),
                     n.co, n.channels);
    s.weight_ci[slot] = ci;
    s.weight_co[slot] = n.co;
    s.weight_valid[slot] = true;
}

static void wait_lsu(const State &s)
{
    (void)s; // 未启用分段计时时，CONV_PHASE不会引用s。
    CONV_PHASE(s, kLsuWait);
    dayan_wait_lsu();
}

static void wait_mat(const State &s)
{
    (void)s;
    CONV_PHASE(s, kMatWait);
    dayan_wait_mat();
}

// Synchronous initial/serial load; the pipeline prefetch uses the same pieces.
static void loader(State &s, const Node &n, uint32_t ci, uint32_t slot)
{
    clear_slot(s, n, ci, slot);
    if (start_input(s, n, ci, slot))
        wait_lsu(s);
    load_weight(s, n, ci, slot);
}

// An asynchronous post keeps the pending slot owned until the caller waits.
static bool post(State &s, bool wait = true)
{
    if (!s.has_output)
        return false;
    CONV_PHASE(s, kOutputStore);
    const Conv2dTileConfig &c = *s.cfg;
    const Node &n = s.pending;
    uint8_t *destination = (uint8_t *)s.output +
        (uint64_t)n.oh * c.output_stride_h + (uint64_t)n.ow * c.output_stride_w +
        n.co * 4U;
    start_channel_rect((uint64_t)(uintptr_t)destination, s.uram + n.output_at,
                       n.channels * 4U, n.cols, n.rows,
                       c.tile_co * 4U,
                       (packed_pointwise(c) ? n.cols : c.tile_w) * c.tile_co * 4U,
                       c.output_stride_w, c.output_stride_h, false);
    if (wait) {
        wait_lsu(s);
        s.has_output = false;
    }
    return true;
}

  // Unlike MatMul's worker threads, Conv uses one ordered LSU/MAT scheduler.
  // Preserve the existing wait/issue order when overlapping the two engines.
static void calc(State &s, const Node &n)
{
    const Conv2dTileConfig &c = *s.cfg;
    if (c.pipeline)
        loader(s, n, 0, 0);
    for (uint32_t ci = 0; ci < c.ci; ci += c.tile_ci) {
        const uint32_t slot = (ci / c.tile_ci) % s.buffers;
        const uint32_t next_slot = 1U - slot;
        const uint32_t next_ci = ci + c.tile_ci;
        const bool has_next = c.pipeline && next_ci < c.ci;
        if (c.pipeline) {
            // Finish the MAT that may still read the next slot before reuse.
            if (ci != 0)
                wait_mat(s);
            if (has_next)
                clear_slot(s, n, next_ci, next_slot);
        } else {
            loader(s, n, ci, slot);
        }
        {
            CONV_PHASE(s, kMatConfig);
            configure_mat(c, c.tile_ci, c.tile_co, n.input_w, n.input_h,
                          n.cols, n.rows);
        }
        CONV_COUNT(s, kMatCalls);
        // One LSU group: previous output takes priority over input prefetch.
        const bool storing = c.pipeline && ci == 0 && post(s, false);
        const bool prefetching = !storing && has_next &&
                                 start_input(s, n, next_ci, next_slot);
        if (ci == 0)
            __tx_mat_conv2d<0>((long)(s.uram + n.output_at),
                               (long)(s.uram + slot * s.input_span),
                               (long)(s.uram + s.weight_base + slot * s.weight_span));
        else
            __tx_mat_conv2da<0>((long)(s.uram + n.output_at),
                                (long)(s.uram + slot * s.input_span),
                                (long)(s.uram + s.weight_base + slot * s.weight_span));
        if (storing || prefetching)
            wait_lsu(s);
        if (storing)
            s.has_output = false;
        if (has_next) {
            if (storing && start_input(s, n, next_ci, next_slot))
                wait_lsu(s);
            load_weight(s, n, next_ci, next_slot);
        }
        if (!c.pipeline)
            wait_mat(s);
    }
    if (c.pipeline)
        wait_mat(s);
}

// With one Ci block there is no next-Ci prefetch. Instead prepare the next
// output tile while MAT owns this tile. Input/output slots alternate, but a
// resident weight slot is shared read-only rather than duplicated. On a miss,
// only the OTHER weight slot can be overwritten until the current MAT ends.
static void calc_spatial_pipeline(State &s, const Node &n, uint32_t input_slot,
                                  uint32_t &weight_slot, const Node *next)
{
    const Conv2dTileConfig &c = *s.cfg;
    {
        CONV_PHASE(s, kMatConfig);
        configure_mat(c, c.tile_ci, c.tile_co, n.input_w, n.input_h,
                      n.cols, n.rows);
    }
    CONV_COUNT(s, kMatCalls);
    const bool storing = post(s, false);
    __tx_mat_conv2d<0>((long)(s.uram + n.output_at),
                       (long)(s.uram + input_slot * s.input_span),
                       (long)(s.uram + s.weight_base + weight_slot * s.weight_span));
    if (storing) {
        wait_lsu(s);
        s.has_output = false;
    }
    if (next) {
        const uint32_t next_input = 1U - input_slot;
        clear_input_slot(s, *next, 0, next_input);
        if (start_input(s, *next, 0, next_input))
            wait_lsu(s);
        uint32_t next_weight = 1U - weight_slot;
        for (uint32_t slot = 0; slot < 2; ++slot)
            if (weight_resident(s, *next, 0, slot)) {
                next_weight = slot;
                break;
            }
        clear_weight_slot(s, *next, 0, next_weight);
        load_weight(s, *next, 0, next_weight);
        weight_slot = next_weight;
    }
    wait_mat(s);
}

static void run(const Conv2dTileConfig *config, const uint16_t *input,
                const uint16_t *weight, uint32_t *output, bool four_core
#if defined(RVNPU_CONV_PROFILE)
                , Conv2dProfileRecord &profile
#endif
                )
{
    if (!config || !input || !weight || !output)
        check(-22);
    dayan_cache_invalidate(config, sizeof(*config));
    if (!config_valid(config))
        check(-22);
    const uint32_t task = dayan_task_id(DAYAN_TASK_X);
    const uint32_t tasks = dayan_task_dim(DAYAN_TASK_X);
    if (dayan_task_mode() != (four_core ? 4 : 1) ||
        tasks != (four_core ? 4U : 1U))
        check(-22);

    __tx_lsu_tcsr_write<LSU_URAMWRITESWIZZLEMBS>(0);
    __tx_lsu_tcsr_write<LSU_URAMREADSWIZZLEMBS>(0);

    const Conv2dTileConfig &c = *config;
    State s = {};
#if defined(RVNPU_CONV_PROFILE)
    s.profile = &profile;
#endif
    s.cfg = config;
    s.input = input;
    s.weight = weight;
    s.output = output;
    s.uram = (uint64_t)(uintptr_t)dayan_uram_base();
    s.tiles_w = (c.wo + c.tile_w - 1U) / c.tile_w;
    s.tiles_co = (c.co + c.tile_co - 1U) / c.tile_co;
    const uint32_t tiles_h = (c.ho + c.tile_h - 1U) / c.tile_h;
    s.tiles_h = tiles_h;
    const uint64_t spatial_tiles = (uint64_t)tiles_h * s.tiles_w;
    if (spatial_tiles > UINT64_MAX / s.tiles_co)
        check(-22);
    const uint64_t tile_count = spatial_tiles * s.tiles_co;
    const uint32_t ih = (c.tile_h - 1U) * c.stride_h + (c.hk - 1U) * c.dilation_h + 1U;
    const uint32_t iw = (c.tile_w - 1U) * c.stride_w + (c.wk - 1U) * c.dilation_w + 1U;
    s.buffers = c.pipeline ? 2U : 1U;
    s.input_span = (ih * iw * c.tile_ci * 2U + 63U) & ~63U;
    s.weight_span = (c.hk * c.wk * c.tile_ci * c.tile_co * 2U + 63U) & ~63U;
    s.output_span = (c.tile_h * c.tile_w * c.tile_co * 4U + 63U) & ~63U;
    s.weight_base = s.input_span * s.buffers;
    s.output_base = s.weight_base + s.weight_span * s.buffers;
    const uint32_t required = s.output_base + s.output_span * s.buffers;
    if (!s.uram || required > kConvUramBytes || dayan_uram_size() < required)
        check(-12);
#if defined(RVNPU_CONV_PROFILE)
    profile.phase[kSetup] = cycle_stamp() - profile.start;
#endif
    // Tensor data is accessed ONLY by LSU/MAT, never by scalar loads/stores.
    // The caller hands buffers to DMA with no dirty scalar-cache aliases,
    // including aliases of a reused output, and no concurrent scalar access.
    // The producer must publish input/weights to DMA-visible memory before
    // launch (e.g. completed H2D, or a CPU producer's flush + synchronization).
    // A scalar consumer must invalidate before reading DMA-produced output.
    // Invalidating tensors here cannot publish a producer's dirty cache, and
    // flushing output here cannot publish DMA writes: this Hart dirties none
    // of these lines. Keep config/record cache maintenance and all engine waits.
    {
        CONV_PHASE(s, kBody);
        uint64_t local_tile = 0;
        uint32_t weight_slot = 0;
        // RTL 1x1 measurements regress with next-tile input prefetch even when
        // MAT wait shrinks: LSU/MAT overlap can increase transfer contention.
        // Keep its resident-weight/Ci pipeline, and enable outer prefetch only
        // for the targeted 3x3 path rather than treating overlap as a free win.
        const bool spatial_pipeline = c.pipeline && c.hk == 3 && c.wk == 3 &&
            c.ci <= c.tile_ci && tile_count > tasks;
        for (uint64_t id = task; id < tile_count; id += tasks, ++local_tile) {
            CONV_COUNT(s, kTiles);
            const Node n = node(s, id, local_tile);
            if (spatial_pipeline) {
                if (!local_tile)
                    loader(s, n, 0, 0);
                const bool has_next = tile_count - id > tasks;
                const Node next = has_next ? node(s, id + tasks, local_tile + 1) : Node{};
                calc_spatial_pipeline(s, n, local_tile % s.buffers,
                                      weight_slot, has_next ? &next : nullptr);
            } else {
                calc(s, n);
            }
            s.pending = n;
            s.has_output = true;
            if (!c.pipeline)
                post(s);
        }
        post(s); // Drain output before completion, including the last tile.
    }
}
} // namespace conv2d_tile_detail
#endif

__kernel__ void conv2d_tile_1c(const Conv2dTileConfig *config,
                               const uint16_t *input, const uint16_t *weight,
                               uint32_t *output
#if defined(RVNPU_CONV_PROFILE)
                               , Conv2dProfileRecord *records, uint64_t call_id
#endif
                               )
{
#if defined(__DAYAN_DEVICE__)
#if defined(RVNPU_CONV_PROFILE)
    Conv2dProfileRecord timing{};
    timing.start = conv2d_tile_detail::cycle_stamp();
    conv2d_tile_detail::run(config, input, weight, output, false, timing);
    timing.end = conv2d_tile_detail::cycle_stamp();
    conv2d_tile_detail::report_cycles(records, call_id, timing);
#else
    conv2d_tile_detail::run(config, input, weight, output, false);
#endif
#endif
}
DAYAN_KERNEL_TASK_DESC(conv2d_tile_1c, 1, 0, 0, 1572864, 1, 1, 0, 0);

__kernel__ void conv2d_tile_4c(const Conv2dTileConfig *config,
                               const uint16_t *input, const uint16_t *weight,
                               uint32_t *output
#if defined(RVNPU_CONV_PROFILE)
                               , Conv2dProfileRecord *records, uint64_t call_id
#endif
                               )
{
#if defined(__DAYAN_DEVICE__)
#if defined(RVNPU_CONV_PROFILE)
    Conv2dProfileRecord timing{};
    timing.start = conv2d_tile_detail::cycle_stamp();
    conv2d_tile_detail::run(config, input, weight, output, true, timing);
    timing.end = conv2d_tile_detail::cycle_stamp();
    conv2d_tile_detail::report_cycles(records, call_id, timing);
#else
    conv2d_tile_detail::run(config, input, weight, output, true);
#endif
#endif
}
DAYAN_KERNEL_TASK_DESC(conv2d_tile_4c, 1, 0, 0, 1572864, 1, 1, 0, 0);
