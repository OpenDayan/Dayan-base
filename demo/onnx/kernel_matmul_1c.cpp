#include <dayan_kernel.h>
#include <stdint.h>

// A[M][K] BF16 * BT[N][K] BF16 -> C[M][N] FP32; strides are bytes.
// Each Core's root Task owns complete output tiles and loads both operands.
// One operand has four slices, the other two physical buffers. Each slice
// has its own C tile. Two Loader Threads, the root Calc Proc, and one Post
// Thread coordinate through Local Resources only; Tasks never exchange data.
struct MatmulBf16Config {
    uint64_t m, n, k;
    uint64_t a_stride, bt_stride, c_stride;
};

struct MatmulBf16Shape {
    enum : int {
        kTileM = 512, kTileN = 384, kTileK = 256, kSlices = 4,
        // Keep row strides; reduce A/C buffers to fit both pipeline layouts.
        kSmallTileM = 192, kSmallUramBytes = 768 * 1024,
        kWorkers = 3, kUramBytes = 1536 * 1024,
    };
};

#if defined(__DAYAN_DEVICE__)

#include <dayan_intrinsics.h>
#include <dayan_sys.h>

namespace matmul_bf16_1c_detail {

// One Local Resource per sliced input, buffered input, and C slice.
enum : int {
    SmallTM = MatmulBf16Shape::kSmallTileM,
    TM = MatmulBf16Shape::kTileM, TN = MatmulBf16Shape::kTileN,
    TK = MatmulBf16Shape::kTileK, Slices = MatmulBf16Shape::kSlices,
    LocalResources = 2 * Slices + 2,
};
enum : uint64_t {
    // VisualSwizzle, 32 banks, 8 x 32B per beat: A/BT's 512B stride
    // and C's 1536B stride improve from 4x to 1x with M7/B2/S3.
    // C's 384B stride is already 1x with identity. All cover 32/32 banks
    // over four horizontal blocks; LSU and MAT use the same mapping.
    InputSwizzle = MAT_BATCH_SWIZZLE(7, 2, 3),
    WindowStride = 2 * 1024 * 1024,
};
static_assert(TM % Slices == 0 && TN % (Slices * 16) == 0);
static_assert(LocalResources <= 32);
static_assert(2 * TM * TK * 2 + TN * TK * 2 + TM * TN * 4 <=
              MatmulBf16Shape::kUramBytes);
static_assert(TM * TK * 2 + 2 * TN * TK * 2 + TM * TN * 4 <=
              MatmulBf16Shape::kUramBytes);

static_assert(SmallTM % Slices == 0);
static_assert((2ULL * SmallTM + TN) * TK * 2 +
              4ULL * SmallTM * TN <=
              MatmulBf16Shape::kSmallUramBytes);
static_assert((SmallTM + 2ULL * TN) * TK * 2 +
              4ULL * SmallTM * TN <=
              MatmulBf16Shape::kSmallUramBytes);

struct Buffer {
    uint64_t uram;
    dayan_resource_t resource;
};
struct State {
    const MatmulBf16Config *cfg;
    uint64_t input[2], output, generations, c_swizzle;
    uint32_t task, tasks;
    int tile_m, nt, kt, slice_m, slice_n;
    bool buffer_a;
    Buffer a[Slices], b[Slices], c[Slices];
};
struct LoadArgs {
    State *state;
    bool b;
};
struct Node {
    uint64_t row, col;
    int rows, cols, inner, step, a_index, b_index, users;
};

static int extent(int64_t remaining, int size)
{
    return remaining <= 0 ? 0 : (remaining < size ? (int)remaining : size);
}

// All stages use the same (output tile, K tile, slice) order. phase selects
// the physical double buffer; Resource lifecycle phases are tracked separately.
static Node node(const State &s, uint64_t generation, int slice)
{
    const uint64_t out = s.task + generation / s.kt * s.tasks;
    const int step = (int)(generation % s.kt);
    const int phase = (int)(generation & 1);
    const uint64_t row = out / s.nt * s.tile_m;
    const uint64_t col = out % s.nt * TN;
    const int rows = extent((int64_t)s.cfg->m - (int64_t)row, s.tile_m);
    const int cols = extent((int64_t)s.cfg->n - (int64_t)col, TN);
    const int row_offset = s.buffer_a ? 0 : slice * s.slice_m;
    const int col_offset = s.buffer_a ? slice * s.slice_n : 0;
    const int users = s.buffer_a ? (cols + s.slice_n - 1) / s.slice_n
                                 : (rows + s.slice_m - 1) / s.slice_m;
    return {row + row_offset, col + col_offset,
            extent(rows - row_offset, s.slice_m),
            extent(cols - col_offset, s.slice_n),
            extent((int64_t)s.cfg->k - (int64_t)step * TK, TK), step,
            s.buffer_a ? phase : slice, s.buffer_a ? slice : phase, users};
}

static void check(int status)
{
    if (status < 0)
        dayan_thread_exit(status);
}

static void loader(void *argument)
{
    const LoadArgs &args = *(LoadArgs *)argument;
    const State &s = *args.state;
    const bool b = args.b;
    const bool buffered = b ? !s.buffer_a : s.buffer_a;
    long checksum = 0;

    for (uint64_t gen = 0; gen < s.generations; ++gen) {
        for (int slice = 0; slice < (buffered ? 1 : Slices); ++slice) {
            const Node n = node(s, gen, slice);
            if (!n.rows || !n.cols)
                continue;
            const Buffer &dst = b ? s.b[n.b_index] : s.a[n.a_index];
            const int rows = b ? n.cols : n.rows;
            const int bytes = n.inner * 2;
            const uint32_t users = buffered ? n.users : 1;
            const uint64_t stride = b ? s.cfg->bt_stride : s.cfg->a_stride;
            const uint64_t src = s.input[b] + (b ? n.col : n.row) * stride +
                                 (uint64_t)n.step * TK * 2;

            check(dayan_resource_wait_free(dst.resource));
            set_lsu_matrix<0>(checksum, rows, TK * 2, stride, TK * 2);
            __tx_lsu_tcsr_write<LSU_URAMWRITESWIZZLEMBS>(InputSwizzle);
            // One full-burst DMA and, for a K tail, one short DMA.
            for (int pos = 0; pos < bytes;) {
                const int count = bytes - pos >= 64 ? (bytes - pos) / 64 : 1;
                const int request = bytes - pos >= 64 ? 64 : bytes - pos;
                __tx_lsu_tcsr_write<LSU_DIM0>(count - 1);
                __tx_lsu_tcsr_write<LSU_REQUESTREADSIZE>(request - 1);
                __tx_lsu_dma_d2u<0>((long)(dst.uram + pos), (long)(src + pos));
                pos += count * request;
            }
            check(dayan_resource_produce_lsu(dst.resource, users, 0));
        }
    }
    dayan_wait_lsu();
}

static void calc(void *argument)
{
    const State &s = *(State *)argument;
    uint64_t phases = 0;
    __tx_mat_tcsr_write<MAT_ASTRIDE>(TK * 2);
    __tx_mat_tcsr_write<MAT_BSTRIDE>(TK * 2);
    __tx_mat_tcsr_write<MAT_CSTRIDE>(s.slice_n * 4);
    __tx_mat_tcsr_write<MAT_ASWIZZLEMBS>(InputSwizzle);
    __tx_mat_tcsr_write<MAT_BSWIZZLEMBS>(InputSwizzle);
    __tx_mat_tcsr_write<MAT_CSWIZZLEMBS>(s.c_swizzle);
    __tx_mat_tcsr_write<MAT_PERSISTENTMODE>(0);
    __tx_mat_tcsr_write<MAT_DATATYPE>(BaseNPUInstBuilder::DT_BF16);
    for (uint64_t gen = 0; gen < s.generations; ++gen) {
        for (int slice = 0; slice < Slices; ++slice) {
            const Node n = node(s, gen, slice);
            if (!n.rows || !n.cols)
                continue;
            const Buffer &a = s.a[n.a_index], &b = s.b[n.b_index], &c = s.c[slice];
            const auto inputs = DAYAN_RESOURCE_MASK(a.resource) |
                                DAYAN_RESOURCE_MASK(b.resource);
            check(dayan_resource_wait(
                n.step == 0 ? DAYAN_RESOURCE_MASK(c.resource) : 0,
                inputs, phases, 0));
            __tx_mat_tcsr_write<MAT_M>(n.rows - 1);
            __tx_mat_tcsr_write<MAT_N>(n.cols - 1);
            __tx_mat_tcsr_write<MAT_K>(n.inner - 1);
            if (n.step == 0)
                __tx_mat_mm<0>((long)c.uram, (long)a.uram, (long)b.uram);
            else
                __tx_mat_mma<0>((long)c.uram, (long)a.uram, (long)b.uram);
            if (n.step + 1 == s.kt)
                check(dayan_resource_produce_mat(c.resource, 1, inputs));
            else
                check(dayan_resource_consume_mat(inputs));
            // Sliced inputs advance once per use; the buffered input is shared
            // by all active slices of this generation.
            phases ^= DAYAN_RESOURCE_MASK(s.buffer_a ? b.resource : a.resource);
            if (slice + 1 == n.users)
                phases ^= DAYAN_RESOURCE_MASK(s.buffer_a ? a.resource : b.resource);
        }
    }
    dayan_wait_mat();
}

static void post(void *argument)
{
    const State &s = *(State *)argument;
    uint64_t phases = 0;
    long checksum = 0;
    set_lsu_matrix<0>(checksum, s.slice_m, s.slice_n * 4,
                      s.slice_n * 4, s.cfg->c_stride);
    __tx_lsu_tcsr_write<LSU_URAMREADSWIZZLEMBS>(s.c_swizzle);
    for (uint64_t gen = s.kt - 1; gen < s.generations; gen += s.kt) {
        for (int slice = 0; slice < Slices; ++slice) {
            const Node n = node(s, gen, slice);
            if (!n.rows || !n.cols)
                continue;
            const Buffer &c = s.c[slice];
            const auto mask = DAYAN_RESOURCE_MASK(c.resource);
            check(dayan_resource_wait(0, mask, phases, 0));
            // Release this C slice only after its final store completes.
            __tx_lsu_tcsr_write<LSU_DIM0>(n.cols * 4 / 64 - 1);
            __tx_lsu_tcsr_write<LSU_DIM1>(n.rows - 1);
            const uint64_t dst = s.output + n.row * s.cfg->c_stride + n.col * 4;
            __tx_lsu_dma_u2d<0>((long)dst, (long)c.uram);
            check(dayan_resource_consume_lsu(mask));
            phases ^= mask;
        }
    }
    dayan_wait_lsu();
}

static bool config_valid(const MatmulBf16Config *c)
{
    constexpr uint64_t limit = 0x7ffff000ULL;
    if (!c || !c->m || !c->n || !c->k || c->m > limit ||
        c->n > limit || c->k > limit || c->n % 16)
        return false;
    const uint64_t rows[] = {c->m, c->n, c->m};
    const uint64_t widths[] = {c->k * 2, c->k * 2, c->n * 4};
    const uint64_t strides[] = {c->a_stride, c->bt_stride, c->c_stride};
    for (int i = 0; i < 3; ++i)
        if (strides[i] < widths[i] || strides[i] > INT64_MAX / rows[i])
            return false;
    return !(c->a_stride & 1) && !(c->bt_stride & 1) && !(c->c_stride & 63);
}

static void run(const MatmulBf16Config *config,
                const uint16_t *a, const uint16_t *bt, float *c, int tile_m)
{
    const bool four_core = dayan_task_mode() == 4;
    // Host DMA may reuse a config address still cached by this Hart.
    if (config)
        dayan_cache_invalidate(config, sizeof(*config));
    if (!config_valid(config) || !a || !bt || !c ||
        ((uintptr_t)a & 1) || ((uintptr_t)bt & 1) || ((uintptr_t)c & 63) ||
        (!four_core && dayan_task_mode() != 1) || !dayan_task_dim(DAYAN_TASK_X) ||
        (four_core ? dayan_task_dim(DAYAN_TASK_X) != 4 : dayan_task_dim(DAYAN_TASK_Y) != 1) ||
        !dayan_task_dim(DAYAN_TASK_Y) || dayan_task_dim(DAYAN_TASK_Z) != 1)
        dayan_thread_exit(-22);
    State s = {};
    s.tile_m = tile_m;
    s.cfg = config;
    s.input[0] = (uintptr_t)a;
    s.input[1] = (uintptr_t)bt;
    s.output = (uintptr_t)c;
    s.task = dayan_task_id(DAYAN_TASK_X) + (four_core ? 4 * dayan_task_id(DAYAN_TASK_Y) : 0);
    s.tasks = dayan_task_dim(DAYAN_TASK_X) * dayan_task_dim(DAYAN_TASK_Y);
    s.nt = (int)((config->n + TN - 1) / TN);
    s.kt = (int)((config->k + TK - 1) / TK);
    const uint64_t outputs = (config->m + s.tile_m - 1) / s.tile_m * s.nt;
    if (s.task >= s.tasks)
        dayan_thread_exit(-22);
    if (s.task >= outputs)
        return;
    // Round-robin complete output tiles; every Task owns its entire K loop.
    const uint64_t tiles = (outputs - 1 - s.task) / s.tasks + 1;
    if (tiles > UINT64_MAX / s.kt)
        dayan_thread_exit(-22);
    s.generations = tiles * s.kt;
    s.buffer_a = config->m < config->n;
    s.slice_m = s.buffer_a ? s.tile_m : s.tile_m / Slices;
    s.slice_n = s.buffer_a ? TN / Slices : TN;
    s.c_swizzle = s.buffer_a ? MAT_BATCH_SWIZZLE(5, 0, 1)
                            : MAT_BATCH_SWIZZLE(7, 2, 3);
    const int counts[] = {s.buffer_a ? 2 : Slices, s.buffer_a ? Slices : 2, Slices};
    const uint64_t bytes[] = {(uint64_t)s.slice_m * TK * 2,
                              (uint64_t)s.slice_n * TK * 2,
                              (uint64_t)s.slice_m * s.slice_n * 4};
    const uint64_t required = counts[0] * bytes[0] + counts[1] * bytes[1] +
                              counts[2] * bytes[2];
    const uint64_t physical = (uintptr_t)dayan_uram_base();
    // Tensor U addresses are offsets in this Core's 2 MiB window.
    // Task coordinates do not identify the Core selected by the scheduler.
    uint64_t address = physical & (WindowStride - 1);
    if (!physical || required > MatmulBf16Shape::kUramBytes ||
        dayan_uram_size() < required || address > WindowStride - required)
        dayan_thread_exit(-12);
    int resource = 0;
    Buffer *buffers[] = {s.a, s.b, s.c};
    for (int operand = 0; operand < 3; ++operand) {
        for (int i = 0; i < counts[operand]; ++i) {
            buffers[operand][i] = {address, DAYAN_LOCAL_RESOURCE(resource++)};
            address += bytes[operand];
        }
    }
    LoadArgs loads[] = {{&s, false}, {&s, true}};
    dayan_thread_resources_t resources = {};
    resources.lsu_tcsr_count = 1;
    resources.local_resource_count = LocalResources;
    check(dayan_create_thread("load-a", loader, &loads[0], &resources));
    check(dayan_create_thread("load-b", loader, &loads[1], &resources));
    check(dayan_create_thread("post", post, &s, &resources));
    calc(&s);
    check(dayan_thread_join_all());
}

// Independent complete output tiles; the only inter-core operation is barrier.
static void launch(const MatmulBf16Config *config, const uint16_t *a,
                   const uint16_t *bt, float *c, int tile_m)
{
    const bool four_core = dayan_task_mode() == 4;
    auto *barrier = (dayan_cluster_barrier_t *)shared;
    if (four_core && dayan_cluster_barrier(4, barrier) < 0) dayan_thread_exit(-22);
    run(config, a, bt, c, tile_m);
    if (four_core && dayan_cluster_barrier(4, barrier) < 0) dayan_thread_exit(-22);
}

} // namespace matmul_bf16_1c_detail
#endif

__kernel__ void matmul_bf16_1c(const MatmulBf16Config *config,
                              const uint16_t *a, const uint16_t *bt, float *c)
{
    matmul_bf16_1c_detail::launch(config, a, bt, c, MatmulBf16Shape::kTileM);
}

__kernel__ void matmul_bf16_1c_small(const MatmulBf16Config *config,
                              const uint16_t *a, const uint16_t *bt, float *c)
{
    matmul_bf16_1c_detail::launch(config, a, bt, c, MatmulBf16Shape::kSmallTileM);
}

DAYAN_KERNEL_NAMED_RESOURCE_TASK_DESC(
    matmul_bf16_1c, 1, MatmulBf16Shape::kWorkers,
    64, MatmulBf16Shape::kUramBytes, 3, 1, 0, 0,
    2 * MatmulBf16Shape::kSlices + 2, 0);

DAYAN_KERNEL_NAMED_RESOURCE_TASK_DESC(
    matmul_bf16_1c_small, 1, MatmulBf16Shape::kWorkers,
    64, MatmulBf16Shape::kSmallUramBytes, 3, 1, 0, 0,
    2 * MatmulBf16Shape::kSlices + 2, 0);


// MAT requires N to be a multiple of 16. The host sends only the remaining
// 1..15 columns here, so no padded columns are calculated. Logical Tasks split
// exact output elements; this path is intentionally small and tail-only.
__kernel__ void dayan_gemm_tail_bf16(const uint16_t *a, const uint16_t *bt,
                                     float *c, uint32_t m, uint32_t n,
                                     uint32_t k)
{
#if defined(__DAYAN_DEVICE__)
    if (!a || !bt || !c || !m || !n || n >= 16 || !k ||
        dayan_task_mode() != 1 || !dayan_task_dim(DAYAN_TASK_X) ||
        dayan_task_dim(DAYAN_TASK_Y) != 1 ||
        dayan_task_dim(DAYAN_TASK_Z) != 1)
        dayan_thread_exit(-22);
    dayan_cache_invalidate(a, (uint64_t)m * k * sizeof(uint16_t));
    dayan_cache_invalidate(bt, (uint64_t)n * k * sizeof(uint16_t));
    dayan_cache_invalidate(c, (uint64_t)m * n * sizeof(float));
    const uint32_t task = dayan_task_id(DAYAN_TASK_X);
    const uint32_t tasks = dayan_task_dim(DAYAN_TASK_X);
    const uint64_t outputs = (uint64_t)m * n;
    for (uint64_t index = task; index < outputs; index += tasks) {
        const uint32_t row = (uint32_t)(index / n);
        const uint32_t column = (uint32_t)(index % n);
        float sum = 0.0f;
        for (uint32_t inner = 0; inner < k; ++inner) {
            union { uint32_t u; float f; } av = {
                (uint32_t)a[(uint64_t)row * k + inner] << 16};
            union { uint32_t u; float f; } bv = {
                (uint32_t)bt[(uint64_t)column * k + inner] << 16};
            sum += av.f * bv.f;
        }
        c[index] = sum;
    }
#endif
}

DAYAN_KERNEL_TASK_DESC(dayan_gemm_tail_bf16, 4, 0, 0, 0, 1, 0, 0, 0);
