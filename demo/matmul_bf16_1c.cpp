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

#if !defined(__DAYAN_DEVICE__)

#include "uram_config.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

extern "C" __host__ char *getenv(const char *name);

__host__ static void usage(const char *program)
{
    std::printf("usage: %s [M N K [TASKS [CORES]]]\n", program);
    std::printf("  defaults: M=1024 N=1536 K=1024 TASKS=1 CORES=1\n");
    std::printf("  N must be a multiple of 16.\n");
    std::printf("  CORES=1|4; TASKS is the number of launch groups; four-Core groups use barriers only.\n");
    std::printf("  B is generated/stored as BT[N][K], and C is FP32.\n");
}

__host__ static int parse_dimension(const char *text, uint64_t *value)
{
    uint64_t parsed = 0;

    if (!text || !*text || !value)
        return -1;
    while (*text) {
        unsigned digit;

        if (*text < '0' || *text > '9')
            return -1;
        digit = (unsigned)(*text - '0');
        if (parsed > (UINT64_MAX - digit) / 10U)
            return -1;
        parsed = parsed * 10U + digit;
        ++text;
    }
    if (!parsed)
        return -1;
    *value = parsed;
    return 0;
}

__host__ static uint16_t float_to_bf16(float value)
{
    uint32_t bits;

    std::memcpy(&bits, &value, sizeof(bits));
    bits += UINT32_C(0x7fff) + ((bits >> 16) & 1U);
    return (uint16_t)(bits >> 16);
}

__host__ static float bf16_to_float(uint16_t value)
{
    uint32_t bits = (uint32_t)value << 16;
    float result;

    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

__host__ static int wait_and_destroy(npu_event_t *event, uint32_t timeout_ms)
{
    npu_event_info_t info;
    int status;
    int destroy_status;

    if (!event)
        return -1;
    status = npu_event_wait(event, timeout_ms, &info);
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

__host__ static int create_buffer(npu_context_t *context, uint64_t size,
                                  npu_buffer_t **buffer)
{
    npu_buffer_desc_t desc;

    npu_buffer_desc_init(&desc, size);
    return npu_buffer_create(context, &desc, buffer);
}

__host__ static int upload(npu_stream_t *stream, npu_buffer_t *buffer,
                           const void *source, size_t size,
                           uint32_t timeout_ms)
{
    npu_event_t *event = nullptr;
    int status = npu_memcpy_h2d_async(stream, buffer, 0, source, size,
                                      &event);

    return status < 0 ? status : wait_and_destroy(event, timeout_ms);
}

__host__ static int download(npu_stream_t *stream, void *destination,
                             npu_buffer_t *buffer, size_t size,
                             uint32_t timeout_ms)
{
    npu_event_t *event = nullptr;
    int status = npu_memcpy_d2h_async(stream, destination, buffer, 0, size,
                                      &event);

    return status < 0 ? status : wait_and_destroy(event, timeout_ms);
}

__host__ static void preserve_error(int candidate, int *status)
{
    if (*status == 0 && candidate < 0)
        *status = candidate;
}

__host__ static bool checked_elements(uint64_t rows, uint64_t columns,
                                      size_t element_size,
                                      size_t *elements, size_t *bytes)
{
    if (!rows || !columns || rows > SIZE_MAX / columns)
        return false;
    const uint64_t count = rows * columns;

    if (count > SIZE_MAX || count > SIZE_MAX / element_size)
        return false;
    *elements = (size_t)count;
    *bytes = (size_t)count * element_size;
    return true;
}

__host__ static void initialize_inputs(std::vector<uint16_t> *a,
                                       std::vector<uint16_t> *bt,
                                       size_t m, size_t n, size_t k)
{
    for (size_t row = 0; row < m; ++row) {
        for (size_t inner = 0; inner < k; ++inner) {
            const int code = (int)((row * 13U + inner * 3U) % 9U) - 4;

            (*a)[row * k + inner] = float_to_bf16((float)code * 0.125f);
        }
    }
    for (size_t column = 0; column < n; ++column) {
        for (size_t inner = 0; inner < k; ++inner) {
            const int code =
                (int)((column * 11U + inner * 5U) % 7U) - 3;

            (*bt)[column * k + inner] =
                float_to_bf16((float)code * 0.125f);
        }
    }
}

__host__ static void reference_matmul(const std::vector<uint16_t> &a,
                                      const std::vector<uint16_t> &bt,
                                      std::vector<float> *reference,
                                      size_t m, size_t n, size_t k)
{
    for (size_t row = 0; row < m; ++row) {
        for (size_t column = 0; column < n; ++column) {
            float sum = 0.0f;

            for (size_t inner = 0; inner < k; ++inner) {
                sum += bf16_to_float(a[row * k + inner]) *
                       bf16_to_float(bt[column * k + inner]);
            }
            (*reference)[row * n + column] = sum;
        }
    }
}

__host__ static int validate_output(const std::vector<float> &actual,
                                    const std::vector<float> &reference,
                                    size_t n)
{
    float max_absolute = 0.0f;
    float max_relative = 0.0f;
    size_t worst = 0;
    size_t failures = 0;

    for (size_t index = 0; index < actual.size(); ++index) {
        const float absolute = std::fabs(actual[index] - reference[index]);
        const float relative = absolute /
            (std::fabs(reference[index]) > 1.0e-6f ?
                 std::fabs(reference[index]) : 1.0f);

        if (!std::isfinite(actual[index]) ||
            absolute > 1.0e-4f + 1.0e-4f * std::fabs(reference[index])) {
            if (failures < 8) {
                std::printf("mismatch C[%zu,%zu]: got=%g expected=%g "
                            "abs=%g\n",
                            index / n, index % n, actual[index],
                            reference[index], absolute);
            }
            ++failures;
        }
        if (absolute > max_absolute) {
            max_absolute = absolute;
            worst = index;
        }
        if (relative > max_relative)
            max_relative = relative;
    }

    std::printf("[host] max_abs=%g max_rel=%g at C[%zu,%zu]\n",
                max_absolute, max_relative, worst / n, worst % n);
    if (failures) {
        std::printf("[host] FAIL: %zu of %zu outputs differ\n", failures,
                    actual.size());
        return -1;
    }
    std::printf("[host] PASS: all %zu outputs match the BF16 reference\n",
                actual.size());
    return 0;
}

__host__ int main(int argc, char **argv)
{
    uint64_t m = 1024;
    uint64_t n = 1536;
    uint64_t k = 1024;
    uint64_t tasks = 1;
    uint64_t cores = 1;
    size_t a_elements = 0;
    size_t bt_elements = 0;
    size_t c_elements = 0;
    size_t a_bytes = 0;
    size_t bt_bytes = 0;
    size_t c_bytes = 0;
    const uint32_t timeout_ms = 3600000;

    if (argc == 2 && std::strcmp(argv[1], "--help") == 0) {
        usage(argv[0]);
        return 0;
    }
    if (argc != 1 && (argc < 4 || argc > 6)) {
        usage(argv[0]);
        return 2;
    }
    if (argc >= 4 &&
        (parse_dimension(argv[1], &m) < 0 ||
         parse_dimension(argv[2], &n) < 0 ||
         parse_dimension(argv[3], &k) < 0)) {
        std::printf("M, N and K must be positive decimal integers\n");
        return 2;
    }
    if ((argc >= 5 && parse_dimension(argv[4], &tasks) < 0) ||
        (argc >= 6 && parse_dimension(argv[5], &cores) < 0) ||
        (cores != 1 && cores != 4) || tasks > UINT32_MAX / cores) {
        std::printf("require CORES=1|4 and 0<TASKS<=UINT32_MAX/CORES\n");
        return 2;
    }
    if (m > UINT64_C(0x7ffff000) || n > UINT64_C(0x7ffff000) ||
        k > UINT64_C(0x7ffff000) || (n % 16U) != 0) {
        std::printf("unsupported shape: require M>0, N%%16==0 and "
                    "32-bit dimensions\n");
        return 2;
    }
    if (!checked_elements(m, k, sizeof(uint16_t), &a_elements, &a_bytes) ||
        !checked_elements(n, k, sizeof(uint16_t), &bt_elements,
                          &bt_bytes) ||
        !checked_elements(m, n, sizeof(float), &c_elements, &c_bytes) ||
        a_bytes > 64U * 1024U * 1024U ||
        bt_bytes > 64U * 1024U * 1024U ||
        c_bytes > 64U * 1024U * 1024U) {
        std::printf("shape is too large for this demo\n");
        return 2;
    }

    std::vector<uint16_t> a(a_elements);
    std::vector<uint16_t> bt(bt_elements);
    std::vector<float> c(c_elements, 0.0f);
    std::vector<float> reference(c_elements, 0.0f);
    initialize_inputs(&a, &bt, (size_t)m, (size_t)n, (size_t)k);
    reference_matmul(a, bt, &reference, (size_t)m, (size_t)n, (size_t)k);

    MatmulBf16Config config = {
        m, n, k, k * sizeof(uint16_t), k * sizeof(uint16_t),
        n * sizeof(float),
    };
    npu_runtime_options_t runtime_options;
    npu_kernel_launch_attrs_t launch_attrs;
    npu_runtime_t *runtime = nullptr;
    npu_context_t *context = nullptr;
    npu_stream_t *stream = nullptr;
    npu_buffer_t *config_buffer = nullptr;
    npu_buffer_t *a_buffer = nullptr;
    npu_buffer_t *bt_buffer = nullptr;
    npu_buffer_t *c_buffer = nullptr;
    const char *socket_path;
    uint64_t uram_capacity = 0;
    int status = 0;

    npu_runtime_options_init(&runtime_options);
    runtime_options.default_timeout_ms = timeout_ms;
    socket_path = getenv("DAYAN_SERVER_SOCKET");
    if (socket_path && *socket_path)
        runtime_options.server_socket = socket_path;

    status = npu_runtime_open(&runtime, &runtime_options);
    if (status == 0)
        status = demo_uram_capacity(runtime, MatmulBf16Shape::kSmallUramBytes,
                                    &uram_capacity);
    const bool small_uram = uram_capacity < MatmulBf16Shape::kUramBytes;
    const unsigned tile_m = small_uram ? MatmulBf16Shape::kSmallTileM :
                                         MatmulBf16Shape::kTileM;
    if (status == 0)
        status = npu_context_create(runtime, nullptr, &context);
    if (status == 0)
        status = npu_stream_create(context, nullptr, &stream);
    if (status == 0)
        status = create_buffer(context, sizeof(config), &config_buffer);
    if (status == 0)
        status = create_buffer(context, a_bytes, &a_buffer);
    if (status == 0)
        status = create_buffer(context, bt_bytes, &bt_buffer);
    if (status == 0)
        status = create_buffer(context, c_bytes, &c_buffer);
    if (status == 0)
        status = upload(stream, config_buffer, &config, sizeof(config),
                        timeout_ms);
    if (status == 0)
        status = upload(stream, a_buffer, a.data(), a_bytes, timeout_ms);
    if (status == 0)
        status = upload(stream, bt_buffer, bt.data(), bt_bytes, timeout_ms);

    if (status == 0) {
        const uint64_t tiles_m =
            (m + tile_m - 1U) / tile_m;
        const uint64_t tiles_n =
            (n + MatmulBf16Shape::kTileN - 1U) / MatmulBf16Shape::kTileN;

        npu_kernel_launch_attrs_init(&launch_attrs, context, stream,
                                     "matmul-bf16-1c");
        launch_attrs.config.task_dim = cores == 4 ? NPUTaskDim3{4, (uint32_t)tasks, 1}
                                                 : NPUTaskDim3{(uint32_t)tasks, 1, 1};
        launch_attrs.config.task_mode = cores == 4 ? NPUTaskMode::kFourCore : NPUTaskMode::kSingleCore;
        std::printf("[host] cores=%llu groups=%llu, independent tiles, inter-core barrier only\n",
                    (unsigned long long)cores, (unsigned long long)tasks);
        launch_attrs.timeout_ms = timeout_ms;
        std::printf("[host] launch matmul_bf16_1c: A[%llu,%llu] x "
                    "BT[%llu,%llu] -> C[%llu,%llu], tile=%ux%ux%u, "
                    "slices=%u, tile-grid=%llux%llu, tasks=%llu, buffered=%s, loaders=A+B\n",
                    (unsigned long long)m, (unsigned long long)k,
                    (unsigned long long)n, (unsigned long long)k,
                    (unsigned long long)m, (unsigned long long)n,
                    (unsigned)tile_m,
                    (unsigned)MatmulBf16Shape::kTileN,
                    (unsigned)MatmulBf16Shape::kTileK,
                    (unsigned)MatmulBf16Shape::kSlices,
                    (unsigned long long)tiles_m,
                    (unsigned long long)tiles_n, (unsigned long long)tasks, m < n ? "A" : "B");
        auto launch = small_uram ? matmul_bf16_1c_small : matmul_bf16_1c;
        status = launch(config_buffer, a_buffer, bt_buffer,
                                c_buffer, &launch_attrs);
    }
    if (status == 0)
        status = download(stream, c.data(), c_buffer, c_bytes, timeout_ms);
    if (status == 0)
        status = validate_output(c, reference, (size_t)n);
    if (status < 0)
        std::printf("matmul_bf16_1c demo failed: %d\n", status);

    if (c_buffer)
        preserve_error(npu_buffer_destroy(c_buffer), &status);
    if (bt_buffer)
        preserve_error(npu_buffer_destroy(bt_buffer), &status);
    if (a_buffer)
        preserve_error(npu_buffer_destroy(a_buffer), &status);
    if (config_buffer)
        preserve_error(npu_buffer_destroy(config_buffer), &status);
    if (stream)
        preserve_error(npu_stream_destroy(stream), &status);
    if (context)
        preserve_error(npu_context_destroy(context), &status);
    if (runtime)
        preserve_error(npu_runtime_close(runtime), &status);
    return status < 0 ? 1 : 0;
}

#endif /* !__DAYAN_DEVICE__ */
