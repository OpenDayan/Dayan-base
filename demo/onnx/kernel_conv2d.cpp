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

struct Conv2dTileShape {
    enum : uint32_t {
        kUramBytes = 64U * 1024U,
        kOutputTileH = 8U,
        kOutputTileW = 8U,
        kInputCiTile = 32U,
        kOutputCoTile = 16U,
    };
};

#if defined(__DAYAN_DEVICE__)
#include <dayan_intrinsics.h>
#include <dayan_sys.h>

namespace conv2d_tile_detail {

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

alignas(64) __device__ static const uint8_t kZeroDmaBlock[64] = {};

static void clear_uram_dma(uint64_t destination, uint32_t bytes)
{
    // Every input/weight slot is independently rounded up to 64 bytes.
    // Round only the clear length: never write into the adjacent slot.
    if (!bytes || bytes > Conv2dTileShape::kUramBytes || (destination & 63U))
        check(-22);
    const uint32_t blocks = (bytes + 63U) / 64U;
    // Re-read an immutable, aligned DRAM block, with explicit equal 64B
    // reads/writes. This does NOT depend on implicit DMA padding.
    // A zero source stride needs independent RTL qualification.
    configure_lsu_rect(64, blocks, 1, 0, 0, 64, 0);
    __tx_lsu_dma_d2u<0>((long)destination, (long)(uintptr_t)kZeroDmaBlock);
    dayan_wait_lsu(); // Same LSU group is reused for the real input/weight.
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
    configure_lsu_rect(ci_count * 2U, (uint32_t)(vx1 - vx0), (uint32_t)(vy1 - vy0),
                       c.input_stride_w, c.input_stride_h,
                       tile_ci * 2U, input_w * tile_ci * 2U);
    __tx_lsu_dma_d2u<0>((long)destination, (long)source);
    return true;
}

static void load_weight_rect(const Conv2dTileConfig &c, const uint16_t *weight,
                             uint64_t destination, uint32_t tile_ci, uint32_t tile_co,
                             uint32_t ci0, uint32_t ci_count,
                             uint32_t co0, uint32_t co_count)
{
    const uint8_t *source = (const uint8_t *)weight +
        (uint64_t)co0 * c.weight_stride_co + ci0 * 2U;
    configure_lsu_rect(ci_count * 2U, co_count, c.hk * c.wk,
                       c.weight_stride_co, c.weight_stride_w,
                       tile_ci * 2U, tile_ci * tile_co * 2U);
    __tx_lsu_dma_d2u<0>((long)destination, (long)source);
    dayan_wait_lsu();
}

static void configure_mat(const Conv2dTileConfig &c, uint32_t ci,
                          uint32_t co, uint32_t wi, uint32_t hi,
                          uint32_t wo, uint32_t ho)
{
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
        !config->tile_co || config->tile_ci > Conv2dTileShape::kInputCiTile ||
        config->tile_co > Conv2dTileShape::kOutputCoTile ||
        (config->tile_ci & (config->tile_ci - 1U)) ||
        (config->tile_co & (config->tile_co - 1U)) ||
        config->tile_h > Conv2dTileShape::kOutputTileH || config->tile_w > Conv2dTileShape::kOutputTileW ||
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
    uint32_t buffers, tiles_w, tiles_co;
    Node pending;
    bool has_output;
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
    n.ow = (uint32_t)(spatial % s.tiles_w) * c.tile_w;
    n.oh = (uint32_t)(spatial / s.tiles_w) * c.tile_h;
    n.rows = extent(c.ho - n.oh, c.tile_h);
    n.cols = extent(c.wo - n.ow, c.tile_w);
    n.channels = extent(c.co - n.co, c.tile_co);
    n.input_y = (int64_t)n.oh * c.stride_h;
    n.input_x = (int64_t)n.ow * c.stride_w;
    n.input_h = (n.rows - 1U) * c.stride_h + (c.hk - 1U) * c.dilation_h + 1U;
    n.input_w = (n.cols - 1U) * c.stride_w + (c.wk - 1U) * c.dilation_w + 1U;
    n.output_at = s.output_base + (local_tile % s.buffers) * s.output_span;
    return n;
}

static void clear_slot(const State &s, const Node &n, uint32_t ci, uint32_t slot)
{
    const Conv2dTileConfig &c = *s.cfg;
    const uint32_t count = extent(c.ci - ci, c.tile_ci);
    if (input_needs_clear(c, n.input_y, n.input_x, n.input_h, n.input_w,
                          count, c.tile_ci))
        clear_uram_dma(s.uram + slot * s.input_span,
                       n.input_h * n.input_w * c.tile_ci * 2U);
    if (count != c.tile_ci || n.channels != c.tile_co)
        clear_uram_dma(s.uram + s.weight_base + slot * s.weight_span,
                       c.hk * c.wk * c.tile_ci * c.tile_co * 2U);
}

static bool start_input(const State &s, const Node &n, uint32_t ci, uint32_t slot)
{
    const Conv2dTileConfig &c = *s.cfg;
    return start_input_rect(c, s.input, s.uram + slot * s.input_span,
                            n.input_y, n.input_x, n.input_h, n.input_w,
                            c.tile_ci, ci, extent(c.ci - ci, c.tile_ci));
}

static void load_weight(const State &s, const Node &n, uint32_t ci, uint32_t slot)
{
    const Conv2dTileConfig &c = *s.cfg;
    load_weight_rect(c, s.weight, s.uram + s.weight_base + slot * s.weight_span,
                     c.tile_ci, c.tile_co, ci, extent(c.ci - ci, c.tile_ci),
                     n.co, n.channels);
}

// Synchronous initial/serial load; the pipeline prefetch uses the same pieces.
static void loader(const State &s, const Node &n, uint32_t ci, uint32_t slot)
{
    clear_slot(s, n, ci, slot);
    if (start_input(s, n, ci, slot))
        dayan_wait_lsu();
    load_weight(s, n, ci, slot);
}

// An asynchronous post keeps the pending slot owned until the caller waits.
static bool post(State &s, bool wait = true)
{
    if (!s.has_output)
        return false;
    const Conv2dTileConfig &c = *s.cfg;
    const Node &n = s.pending;
    uint8_t *destination = (uint8_t *)s.output +
        (uint64_t)n.oh * c.output_stride_h + (uint64_t)n.ow * c.output_stride_w +
        n.co * 4U;
    configure_lsu_rect(n.channels * 4U, n.cols, n.rows,
                       c.tile_co * 4U, c.tile_w * c.tile_co * 4U,
                       c.output_stride_w, c.output_stride_h);
    __tx_lsu_dma_u2d<0>((long)destination, (long)(s.uram + n.output_at));
    if (wait) {
        dayan_wait_lsu();
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
                dayan_wait_mat();
            if (has_next)
                clear_slot(s, n, next_ci, next_slot);
        } else {
            loader(s, n, ci, slot);
        }
        configure_mat(c, c.tile_ci, c.tile_co, n.input_w, n.input_h,
                      n.cols, n.rows);
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
            dayan_wait_lsu();
        if (storing)
            s.has_output = false;
        if (has_next) {
            if (storing && start_input(s, n, next_ci, next_slot))
                dayan_wait_lsu();
            load_weight(s, n, next_ci, next_slot);
        }
        if (!c.pipeline)
            dayan_wait_mat();
    }
    if (c.pipeline)
        dayan_wait_mat();
}

static void run(const Conv2dTileConfig *config, const uint16_t *input,
                const uint16_t *weight, uint32_t *output, bool four_core)
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

    const Conv2dTileConfig &c = *config;
    State s = {};
    s.cfg = config;
    s.input = input;
    s.weight = weight;
    s.output = output;
    s.uram = (uint64_t)(uintptr_t)dayan_uram_base();
    s.tiles_w = (c.wo + c.tile_w - 1U) / c.tile_w;
    s.tiles_co = (c.co + c.tile_co - 1U) / c.tile_co;
    const uint32_t tiles_h = (c.ho + c.tile_h - 1U) / c.tile_h;
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
    if (!s.uram || required > Conv2dTileShape::kUramBytes || dayan_uram_size() < required)
        check(-12);
    dayan_cache_invalidate(input, (uint64_t)c.hi * c.input_stride_h);
    dayan_cache_invalidate(weight, (uint64_t)c.hk * c.weight_stride_h);
    dayan_cache_invalidate(output, (uint64_t)c.ho * c.output_stride_h);

    uint64_t local_tile = 0;
    for (uint64_t id = task; id < tile_count; id += tasks, ++local_tile) {
        const Node n = node(s, id, local_tile);
        calc(s, n);
        s.pending = n;
        s.has_output = true;
        if (!c.pipeline)
            post(s);
    }
    post(s); // Drain the final output; also safe for a Task with no tiles.
    dayan_cache_flush(output, (uint64_t)c.ho * c.output_stride_h);
}
} // namespace conv2d_tile_detail
#endif

__kernel__ void conv2d_tile_1c(const Conv2dTileConfig *config,
                               const uint16_t *input, const uint16_t *weight,
                               uint32_t *output)
{
#if defined(__DAYAN_DEVICE__)
    conv2d_tile_detail::run(config, input, weight, output, false);
#endif
}
DAYAN_KERNEL_TASK_DESC(conv2d_tile_1c, 1, 0, 0, 65536, 1, 1, 0, 0);

__kernel__ void conv2d_tile_4c(const Conv2dTileConfig *config,
                               const uint16_t *input, const uint16_t *weight,
                               uint32_t *output)
{
#if defined(__DAYAN_DEVICE__)
    conv2d_tile_detail::run(config, input, weight, output, true);
#endif
}
DAYAN_KERNEL_TASK_DESC(conv2d_tile_4c, 1, 0, 0, 65536, 1, 1, 0, 0);
