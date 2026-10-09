// Fused NCHW im2col + FP32->BF16 + MAT Conv.
//
// The host provides the original NCHW FP32 input and OIHW weights converted to
// BF16 (OIHW is already BT[N][K] for the GEMM view).  Each logical task owns
// complete output tiles, generates only its current BF16 A tile in URAM, and
// immediately feeds that tile to MAT.  No full im2col tensor and no K/N padding
// are materialized.  The same indexing handles 1x1, 3x3, and 7x7 kernels.
#include <dayan_kernel.h>
#include <stdint.h>

struct ConvIm2colMatBf16Config {
    uint64_t m, n, k;
    uint64_t input_channels, input_h, input_w;
    uint64_t output_h, output_w;
    uint64_t kernel_h, kernel_w;
    uint64_t stride_h, stride_w;
    uint64_t pad_h, pad_w;
    uint64_t bt_stride, c_stride;
};

struct ConvIm2colMatBf16Shape {
    enum : int {
        kTileM = 512, kTileN = 384, kTileK = 256, kSlices = 4,
        kUramBytes = 1536 * 1024,
    };
};

#if defined(__DAYAN_DEVICE__)

#include <dayan_intrinsics.h>
#include <dayan_sys.h>

namespace conv_im2col_mat_bf16_detail {

enum : int {
    TM = ConvIm2colMatBf16Shape::kTileM,
    TN = ConvIm2colMatBf16Shape::kTileN,
    TK = ConvIm2colMatBf16Shape::kTileK,
    Slices = ConvIm2colMatBf16Shape::kSlices,
};
enum : uint64_t {
    InputSwizzle = MAT_BATCH_SWIZZLE(7, 2, 3),
    WindowStride = 2 * 1024 * 1024,
};
static_assert(TM % Slices == 0 && TN % (Slices * 16) == 0);
static_assert(TM * TK * 2 + TN * TK * 2 + TM * TN * 4 <=
              ConvIm2colMatBf16Shape::kUramBytes);

struct Buffer {
    uint64_t uram;
    uint64_t proc;
};
struct Tile {
    uint64_t row;
    int rows, inner, step;
};

static int extent(int64_t remaining, int size)
{
    return remaining <= 0 ? 0 : (remaining < size ? (int)remaining : size);
}

static uint16_t fp32_to_bf16(float value)
{
    union {
        float f;
        uint32_t u;
    } bits = {value};
    if ((bits.u & 0x7f800000U) == 0x7f800000U)
        return (uint16_t)((bits.u >> 16) | 0x007fU);
    if ((bits.u & 0x7f800000U) == 0)
        return (uint16_t)(bits.u >> 16);
    uint32_t truncated = bits.u >> 16;
    const uint32_t round_bit = (bits.u >> 15) & 1U;
    const uint32_t sticky = bits.u & 0x7fffU;
    if (round_bit && (sticky || (truncated & 1U)))
        ++truncated;
    if ((truncated & 0x7f80U) == 0x7f80U)
        truncated = (truncated & 0x8000U) | 0x7f80U;
    return (uint16_t)truncated;
}

static void generate_a(const ConvIm2colMatBf16Config &cfg,
                       const float *input, const Tile &tile,
                       const Buffer &dst)
{
    uint16_t *uram_tile = (uint16_t *)(uintptr_t)dst.proc;
    const uint64_t kernel_plane = cfg.kernel_h * cfg.kernel_w;
    const uint64_t input_plane = cfg.input_h * cfg.input_w;
    const uint64_t inner_begin = (uint64_t)tile.step * TK;
    for (int local_row = 0; local_row < tile.rows; ++local_row) {
        const uint64_t position = tile.row + (uint64_t)local_row;
        const uint64_t oy = position / cfg.output_w;
        const uint64_t ox = position % cfg.output_w;
        uint16_t *destination = uram_tile + (uint64_t)local_row * TK;
        for (int local_k = 0; local_k < tile.inner; ++local_k) {
            const uint64_t inner = inner_begin + (uint64_t)local_k;
            const uint64_t channel = inner / kernel_plane;
            const uint64_t kernel_index = inner % kernel_plane;
            const uint64_t ky = kernel_index / cfg.kernel_w;
            const uint64_t kx = kernel_index % cfg.kernel_w;
            const int64_t iy = (int64_t)(oy * cfg.stride_h + ky) -
                               (int64_t)cfg.pad_h;
            const int64_t ix = (int64_t)(ox * cfg.stride_w + kx) -
                               (int64_t)cfg.pad_w;
            float value = 0.0f;
            if (iy >= 0 && iy < (int64_t)cfg.input_h &&
                ix >= 0 && ix < (int64_t)cfg.input_w) {
                const uint64_t index = channel * input_plane +
                                       (uint64_t)iy * cfg.input_w +
                                       (uint64_t)ix;
                value = input[index];
            }
            destination[local_k] = fp32_to_bf16(value);
        }
    }
}

static bool config_valid(const ConvIm2colMatBf16Config *c, bool mat)
{
    constexpr uint64_t limit = 0x7ffff000ULL;
    if (!c || !c->m || !c->n || !c->k || c->m > limit || c->n > limit ||
        c->k > limit || !c->input_channels || !c->input_h ||
        !c->input_w || !c->output_h || !c->output_w || !c->kernel_h ||
        !c->kernel_w || !c->stride_h || !c->stride_w)
        return false;
    if (c->m != c->output_h * c->output_w ||
        c->k != c->input_channels * c->kernel_h * c->kernel_w ||
        c->bt_stride < c->k * 2 || c->c_stride < c->n * 4 ||
        (c->bt_stride & 1))
        return false;
    return mat ? !(c->n & 15) && !(c->c_stride & 63) : c->n < 16;
}

} // namespace conv_im2col_mat_bf16_detail
#endif

__kernel__ void rvnpu_conv_im2col_mat_bf16(
    const ConvIm2colMatBf16Config *config, const float *input,
    const uint16_t *bt, float *c)
{
#if defined(__DAYAN_DEVICE__)
    using namespace conv_im2col_mat_bf16_detail;
    dayan_cache_invalidate(config, sizeof(*config));
    if (!config_valid(config, true))
        dayan_thread_exit(-22);
    if (!input || !bt || !c || ((uintptr_t)input & 3) ||
        ((uintptr_t)bt & 1) || ((uintptr_t)c & 63))
        dayan_thread_exit(-22);
    if ((dayan_task_mode() != 1 && dayan_task_mode() != 4) ||
        !dayan_task_dim(DAYAN_TASK_X) ||
        dayan_task_dim(DAYAN_TASK_Y) != 1 || dayan_task_dim(DAYAN_TASK_Z) != 1)
        dayan_thread_exit(-22);
    dayan_cache_invalidate(input, config->input_channels * config->input_h *
                                      config->input_w * sizeof(float));
    dayan_cache_invalidate(bt, config->n * config->k * sizeof(uint16_t));
    const uint32_t task = dayan_task_id(DAYAN_TASK_X);
    const uint32_t tasks = dayan_task_dim(DAYAN_TASK_X);
    const uint64_t nt = (config->n + TN - 1) / TN;
    const uint64_t outputs = (config->m + TM - 1) / TM * nt;
    if (task >= tasks)
        dayan_thread_exit(-22);
    if (task >= outputs)
        return;

    constexpr uint64_t a_bytes = (uint64_t)TM * TK * 2;
    constexpr uint64_t b_bytes = (uint64_t)TN * TK * 2;
    constexpr uint64_t c_bytes = (uint64_t)TM * TN * 4;
    constexpr uint64_t required = a_bytes + b_bytes + c_bytes;
    const uint64_t physical = (uintptr_t)dayan_uram_base();
    uint64_t address = physical & (WindowStride - 1);
    if (!physical || required > ConvIm2colMatBf16Shape::kUramBytes ||
        dayan_uram_size() < required || address > WindowStride - required)
        dayan_thread_exit(-12);
    const Buffer a = {address, physical};
    const Buffer b = {address + a_bytes, physical + a_bytes};
    const Buffer output = {address + a_bytes + b_bytes,
                           physical + a_bytes + b_bytes};
    long checksum = 0;
    __tx_mat_tcsr_write<MAT_ASTRIDE>(TK * 2);
    __tx_mat_tcsr_write<MAT_BSTRIDE>(TK * 2);
    __tx_mat_tcsr_write<MAT_CSTRIDE>(TN * 4);
    __tx_mat_tcsr_write<MAT_ASWIZZLEMBS>(0);
    __tx_mat_tcsr_write<MAT_BSWIZZLEMBS>(InputSwizzle);
    __tx_mat_tcsr_write<MAT_CSWIZZLEMBS>(InputSwizzle);
    __tx_mat_tcsr_write<MAT_PERSISTENTMODE>(0);
    __tx_mat_tcsr_write<MAT_DATATYPE>(BaseNPUInstBuilder::DT_BF16);

    for (uint64_t tile = task; tile < outputs; tile += tasks) {
        const uint64_t row = tile / nt * TM;
        const uint64_t col = tile % nt * TN;
        const int rows = extent((int64_t)config->m - (int64_t)row, TM);
        const int cols = extent((int64_t)config->n - (int64_t)col, TN);
        for (uint64_t step = 0; step * TK < config->k; ++step) {
            const int inner = extent((int64_t)config->k - (int64_t)step * TK,
                                     TK);
            const Tile a_tile = {row, rows, inner, (int)step};
            generate_a(*config, input, a_tile, a);

            set_lsu_matrix<0>(checksum, cols, TK * 2, config->bt_stride,
                              TK * 2);
            __tx_lsu_tcsr_write<LSU_URAMWRITESWIZZLEMBS>(InputSwizzle);
            const uint64_t source = (uintptr_t)bt + col * config->bt_stride +
                                    step * TK * 2;
            const int bytes = inner * 2;
            for (int pos = 0; pos < bytes;) {
                const int count = bytes - pos >= 64 ? (bytes - pos) / 64 : 1;
                const int request = bytes - pos >= 64 ? 64 : bytes - pos;
                __tx_lsu_tcsr_write<LSU_DIM0>(count - 1);
                __tx_lsu_tcsr_write<LSU_REQUESTREADSIZE>(request - 1);
                __tx_lsu_dma_d2u<0>((long)(b.uram + pos),
                                    (long)(source + pos));
                pos += count * request;
            }
            dayan_wait_lsu();

            for (int row_offset = 0; row_offset < rows; row_offset += TM / Slices) {
                const int slice_rows = extent(rows - row_offset, TM / Slices);
                __tx_mat_tcsr_write<MAT_M>(slice_rows - 1);
                __tx_mat_tcsr_write<MAT_N>(cols - 1);
                __tx_mat_tcsr_write<MAT_K>(inner - 1);
                const uint64_t a_slice = a.uram + (uint64_t)row_offset * TK * 2;
                const uint64_t c_slice =
                    output.uram + (uint64_t)row_offset * TN * 4;
                if (step == 0)
                    __tx_mat_mm<0>((long)c_slice, (long)a_slice, (long)b.uram);
                else
                    __tx_mat_mma<0>((long)c_slice, (long)a_slice, (long)b.uram);
            }
            dayan_wait_mat();
        }

        set_lsu_matrix<0>(checksum, rows, cols * 4, TN * 4,
                          config->c_stride);
        __tx_lsu_tcsr_write<LSU_URAMREADSWIZZLEMBS>(InputSwizzle);
        __tx_lsu_dma_u2d<0>(
            (long)((uintptr_t)c + row * config->c_stride + col * 4),
            (long)output.uram);
        dayan_wait_lsu();
    }
#endif
}

DAYAN_KERNEL_TASK_DESC(
    rvnpu_conv_im2col_mat_bf16, 4, 0, 0,
    ConvIm2colMatBf16Shape::kUramBytes, 1, 1, 0, 0);

// Exact 1..15 output-channel tail.  This avoids padding N to a MAT block.
__kernel__ void rvnpu_conv_tail_bf16(
    const ConvIm2colMatBf16Config *config, const float *input,
    const uint16_t *bt, float *c)
{
#if defined(__DAYAN_DEVICE__)
    using namespace conv_im2col_mat_bf16_detail;
    dayan_cache_invalidate(config, sizeof(*config));
    if (!config_valid(config, false) || !input || !bt || !c ||
        (dayan_task_mode() != 1 && dayan_task_mode() != 4) ||
        !dayan_task_dim(DAYAN_TASK_X) ||
        dayan_task_dim(DAYAN_TASK_Y) != 1 || dayan_task_dim(DAYAN_TASK_Z) != 1)
        dayan_thread_exit(-22);
    dayan_cache_invalidate(input, config->input_channels * config->input_h *
                                      config->input_w * sizeof(float));
    dayan_cache_invalidate(bt, config->n * config->k * sizeof(uint16_t));
    const uint32_t task = dayan_task_id(DAYAN_TASK_X);
    const uint32_t tasks = dayan_task_dim(DAYAN_TASK_X);
    const uint64_t outputs = config->m * config->n;
    const uint64_t blocks = (outputs + 15) / 16;
    const uint64_t kernel_plane = config->kernel_h * config->kernel_w;
    const uint64_t input_plane = config->input_h * config->input_w;
    // Give each task complete 64-byte output cache lines. Round-robin scalar
    // elements would let different Cores dirty the same line.
    for (uint64_t block = task; block < blocks; block += tasks) {
        const uint64_t begin = block * 16;
        const uint64_t end = begin + 16 < outputs ? begin + 16 : outputs;
        for (uint64_t index = begin; index < end; ++index) {
            const uint64_t row = index / config->n;
            const uint64_t column = index % config->n;
            const uint64_t oy = row / config->output_w;
            const uint64_t ox = row % config->output_w;
            float sum = 0.0f;
            for (uint64_t inner = 0; inner < config->k; ++inner) {
                const uint64_t channel = inner / kernel_plane;
                const uint64_t kernel_index = inner % kernel_plane;
                const uint64_t ky = kernel_index / config->kernel_w;
                const uint64_t kx = kernel_index % config->kernel_w;
                const int64_t iy = (int64_t)(oy * config->stride_h + ky) -
                                   (int64_t)config->pad_h;
                const int64_t ix = (int64_t)(ox * config->stride_w + kx) -
                                   (int64_t)config->pad_w;
                if (iy < 0 || iy >= (int64_t)config->input_h || ix < 0 ||
                    ix >= (int64_t)config->input_w)
                    continue;
                const uint64_t input_index = channel * input_plane +
                                             (uint64_t)iy * config->input_w +
                                             (uint64_t)ix;
                union {
                    uint32_t u;
                    float f;
                } av = {(uint32_t)fp32_to_bf16(input[input_index]) << 16};
                union {
                    uint32_t u;
                    float f;
                } bv = {(uint32_t)bt[column * config->k + inner] << 16};
                sum += av.f * bv.f;
            }
            c[index] = sum;
        }
    }
#endif
}

DAYAN_KERNEL_TASK_DESC(rvnpu_conv_tail_bf16, 4, 0, 0, 0, 0, 0, 0, 0);
