#include <dayan_kernel.h>
#include <stdint.h>

// A[M][K] E2M1 * BT[N][K] E2M1 -> C[M][N] FP32; strides are bytes.
// Pack consecutive K values low nibble first; each row uses ceil(K/2) bytes.
// For odd K the final high nibble is padding and is not included by MAT.
// The 2x2 Core grid shares A horizontally and BT vertically.
// Each Core has a 512x512 tile and two slices along the private dimension.
// One input is double-buffered; private inputs and C each have one buffer per slice.
// Two Loader Threads, the root Calc Proc, and one Post Thread run per Core.
struct MatmulFp4Config {
    uint64_t m, n, k;
    uint64_t a_stride, bt_stride, c_stride;
};

struct MatmulFp4Shape {
    enum : int {
        kTileM = 512, kTileN = 512, kTileK = 640, kSlices = 2,
        kInputRowBytes = kTileK / 2,
        kWorkers = 3, kAtomicBytes = 64, kUramBytes = 1536 * 1024,
    };
};

#if defined(__DAYAN_DEVICE__)

#include <dayan_intrinsics.h>
#include <dayan_sys.h>

namespace matmul_fp4_4c_detail {

// Local: sliced operand + two buffered tiles + C slices. Shared: each pair's
// input-ready resources, consumed once after the peer's copy completes.
enum : int {
    TM = MatmulFp4Shape::kTileM, TN = MatmulFp4Shape::kTileN,
    TK = MatmulFp4Shape::kTileK,
    InputRowBytes = MatmulFp4Shape::kInputRowBytes,
    Slices = MatmulFp4Shape::kSlices,
    LocalResources = 2 * Slices + 2, SharedResources = 2 * (Slices + 2),
};
enum : uint64_t {
    // FP4 A/B row stride is K/2 bytes. Previous K=512: 256B -> (7,1,3).
    // Original K=1024: 512B -> (7,2,3).
    // K=640: 320B -> identity (5,0,1), already 1x peak in VisualSwizzle.
    // Its unconstrained best (9,1,1) moves data between rows, incompatible
    // with the peer's raw copy of exactly real_rows * InputRowBytes at tails.
    InputSwizzle = MAT_BATCH_SWIZZLE(5, 0, 1),
    // C depends on slice_n, not K. With TN=512 and A shared:
    // previous 4 slices -> stride 512B -> (7,2,3);
    // current 2 slices -> stride 1024B -> (7,3,3), addr[9:7] ^= addr[12:10].
    NarrowCSwizzle = MAT_BATCH_SWIZZLE(7, 3, 3),
    // B shared: C stride stays 2048B -> (7,3,4), addr[9:7] ^= addr[13:11].
    WideCSwizzle = MAT_BATCH_SWIZZLE(7, 3, 4),
    WindowBase = 0x00800000, WindowStride = 2 * 1024 * 1024,
};
static_assert(TM % Slices == 0 && TN % (Slices * 16) == 0);
static_assert(Slices == 2 && InputRowBytes == 320);
static_assert((2 * TM + TN) * InputRowBytes + TM * TN * 4 <=
              MatmulFp4Shape::kUramBytes);
static_assert((TM + 2 * TN) * InputRowBytes + TM * TN * 4 <=
              MatmulFp4Shape::kUramBytes);
static_assert(SharedResources <= 32);

struct Cluster {
    dayan_cluster_barrier_t barrier;
    uint64_t uram[4];
};
static_assert(sizeof(Cluster) <= MatmulFp4Shape::kAtomicBytes);

struct Buffer {
    uint64_t uram;
    dayan_resource_t resource;
};
struct State {
    const MatmulFp4Config *cfg;
    uint64_t input[2], output, uram, generations;
    int lane, nt, kt, slice_m, slice_n;
    bool buffer_a;
    Buffer a[Slices], b[Slices], c[Slices];
};
struct LoadArgs {
    State *state;
    bool b;
};
struct Node {
    uint64_t row, col;
    int rows, cols, inner, step, phase, a_index, b_index, users;
};

static int extent(int64_t remaining, int size)
{
    return remaining <= 0 ? 0 : (remaining < size ? (int)remaining : size);
}

// All stages use the same (output tile, K tile, slice) order. phase selects
// the physical double buffer; Resource lifecycle phases are tracked separately.
static Node node(const State &s, uint64_t generation, int slice, int lane)
{
    const uint64_t out = generation / s.kt;
    const int step = (int)(generation % s.kt);
    const int phase = (int)(generation & 1);
    const uint64_t begin = s.cfg->m * (lane >> 1) / 2;
    const uint64_t end = s.cfg->m * ((lane >> 1) + 1) / 2;
    const uint64_t row = begin + out / s.nt * TM;
    const uint64_t col = (out % s.nt * 2 + (lane & 1)) * TN;
    const int rows = extent((int64_t)end - (int64_t)row, TM);
    const int cols = extent((int64_t)s.cfg->n - (int64_t)col, TN);
    const int row_offset = s.buffer_a ? 0 : slice * s.slice_m;
    const int col_offset = s.buffer_a ? slice * s.slice_n : 0;
    const int users = s.buffer_a ? (cols + s.slice_n - 1) / s.slice_n
                                 : (rows + s.slice_m - 1) / s.slice_m;
    return {row + row_offset, col + col_offset,
            extent(rows - row_offset, s.slice_m),
            extent(cols - col_offset, s.slice_n),
            extent((int64_t)s.cfg->k - (int64_t)step * TK, TK), step, phase,
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
    const int peer = s.lane ^ (b ? 2 : 1);
    const int pair = b ? s.lane & 1 : s.lane >> 1;
    const Cluster &cluster = *(Cluster *)shared;
    uint64_t phases = 0;
    long checksum = 0;

    for (uint64_t gen = 0; gen < s.generations; ++gen) {
        for (int slice = 0; slice < (buffered ? 1 : Slices); ++slice) {
            const Node n = node(s, gen, slice, s.lane);
            const Node other = node(s, gen, slice, peer);
            const bool active = n.rows && n.cols;
            const bool peer_active = other.rows && other.cols;
            if (!active && !peer_active)
                continue;
            const Buffer &dst = b ? s.b[n.b_index] : s.a[n.a_index];
            const int index = buffered ? 2 * Slices + pair * 2 + n.phase
                                       : pair * Slices + slice;
            const auto ready = DAYAN_SHARED_RESOURCE(index);
            const auto ready_mask = DAYAN_RESOURCE_MASK(ready);
            const int rows = b ? n.cols : n.rows;
            const int bytes = (n.inner + 1) / 2;
            const uint32_t users = buffered ? n.users : 1;
            const uint64_t stride = b ? s.cfg->bt_stride : s.cfg->a_stride;
            uint64_t src = s.input[b] + (b ? n.col : n.row) * stride +
                           (uint64_t)n.step * InputRowBytes;

            // Reuse needs both local MAT completion and the prior peer copy.
            // An inactive peer still consumes Shared so tail transitions retain
            // the same lifecycle protection; it never reserves a local buffer.
            const int role = dayan_resource_wait(
                active ? DAYAN_RESOURCE_MASK(dst.resource) | ready_mask : 0,
                ready_mask, phases, 0);
            check(role);
            if (!active) {
                check(dayan_resource_consume_lsu(ready_mask));
            } else if (role == DAYAN_RESOURCE_ROLE_PRODUCER) {
                set_lsu_matrix<0>(checksum, rows, InputRowBytes,
                                  stride, InputRowBytes);
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
                check(dayan_resource_produce_lsu(ready, 1, 0));
            } else if (role == DAYAN_RESOURCE_ROLE_CONSUMER) {
                // Remote URAM is mapped in the D address space. Copy padded
                // rows verbatim, preserving the swizzle already applied by DRAM load.
                src = cluster.uram[peer] + dst.uram - s.uram;
                set_lsu_matrix<0>(checksum, rows, InputRowBytes,
                                  InputRowBytes, InputRowBytes);
                __tx_lsu_dma_d2u<0>((long)dst.uram, (long)src);
                check(dayan_resource_produce_lsu(dst.resource, users, ready_mask));
            } else {
                dayan_thread_exit(-22);
            }
            phases ^= ready_mask;
        }
    }
    dayan_wait_lsu();
}

static void calc(void *argument)
{
    const State &s = *(State *)argument;
    uint64_t phases = 0;
    __tx_mat_tcsr_write<MAT_ASTRIDE>(InputRowBytes);
    __tx_mat_tcsr_write<MAT_BSTRIDE>(InputRowBytes);
    __tx_mat_tcsr_write<MAT_CSTRIDE>(s.slice_n * 4);
    __tx_mat_tcsr_write<MAT_ASWIZZLEMBS>(InputSwizzle);
    __tx_mat_tcsr_write<MAT_BSWIZZLEMBS>(InputSwizzle);
    __tx_mat_tcsr_write<MAT_CSWIZZLEMBS>(
        s.buffer_a ? NarrowCSwizzle : WideCSwizzle);
    __tx_mat_tcsr_write<MAT_PERSISTENTMODE>(0);
    __tx_mat_tcsr_write<MAT_DATATYPE>(BaseNPUInstBuilder::DT_FP4_E2M1);
    for (uint64_t gen = 0; gen < s.generations; ++gen) {
        for (int slice = 0; slice < Slices; ++slice) {
            const Node n = node(s, gen, slice, s.lane);
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
    __tx_lsu_tcsr_write<LSU_URAMREADSWIZZLEMBS>(
        s.buffer_a ? NarrowCSwizzle : WideCSwizzle);
    for (uint64_t gen = s.kt - 1; gen < s.generations; gen += s.kt) {
        for (int slice = 0; slice < Slices; ++slice) {
            const Node n = node(s, gen, slice, s.lane);
            if (!n.rows || !n.cols)
                continue;
            const Buffer &c = s.c[slice];
            const auto mask = DAYAN_RESOURCE_MASK(c.resource);
            check(dayan_resource_wait(0, mask, phases, 0));
            // No activation yet: store C directly. A future compute stage needs
            // its own Output Buffer; C can be released once that compute completes.
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

static bool config_valid(const MatmulFp4Config *c)
{
    constexpr uint64_t limit = 0x7ffff000ULL;
    if (!c || !c->m || !c->n || !c->k || c->m > limit ||
        c->n > limit || c->k > limit || c->n % 16)
        return false;
    const uint64_t rows[] = {c->m, c->n, c->m};
    const uint64_t widths[] = {(c->k + 1) / 2, (c->k + 1) / 2, c->n * 4};
    const uint64_t strides[] = {c->a_stride, c->bt_stride, c->c_stride};
    for (int i = 0; i < 3; ++i)
        if (strides[i] < widths[i] || strides[i] > INT64_MAX / rows[i])
            return false;
    return !(c->c_stride & 63);
}

} // namespace matmul_fp4_4c_detail
#endif

__kernel__ void matmul_fp4_4c(const MatmulFp4Config *config,
                              const uint8_t *a, const uint8_t *bt, float *c)
{
    using namespace matmul_fp4_4c_detail;
    const int lane = dayan_task_id(DAYAN_TASK_X);
    if (!config_valid(config) || !a || !bt || !c || !shared ||
        ((uintptr_t)c & 63) ||
        dayan_task_mode() != 4 || lane >= 4 ||
        dayan_task_dim(DAYAN_TASK_X) != 4 ||
        dayan_task_dim(DAYAN_TASK_Y) != 1 || dayan_task_dim(DAYAN_TASK_Z) != 1)
        dayan_thread_exit(-22);
    State s = {};
    s.cfg = config;
    s.input[0] = (uintptr_t)a;
    s.input[1] = (uintptr_t)bt;
    s.output = (uintptr_t)c;
    s.lane = lane;
    s.nt = (int)((config->n + 2 * TN - 1) / (2 * TN));
    s.kt = (int)((config->k + TK - 1) / TK);
    const uint64_t outputs = (config->m + 2 * TM - 1) / (2 * TM) * s.nt;
    if (outputs > UINT64_MAX / s.kt)
        dayan_thread_exit(-22);
    s.generations = outputs * s.kt;
    s.buffer_a = config->m < config->n;
    s.slice_m = s.buffer_a ? TM : TM / Slices;
    s.slice_n = s.buffer_a ? TN / Slices : TN;
    const int counts[] = {s.buffer_a ? 2 : Slices, s.buffer_a ? Slices : 2, Slices};
    const uint64_t bytes[] = {(uint64_t)s.slice_m * InputRowBytes,
                              (uint64_t)s.slice_n * InputRowBytes,
                              (uint64_t)s.slice_m * s.slice_n * 4};
    const uint64_t required = counts[0] * bytes[0] + counts[1] * bytes[1] +
                              counts[2] * bytes[2];
    const uint64_t physical = (uintptr_t)dayan_uram_base();
    const uint64_t window = WindowBase + lane * WindowStride;
    if (required > MatmulFp4Shape::kUramBytes || dayan_uram_size() < required ||
        physical < window || physical - window > WindowStride - required)
        dayan_thread_exit(-12);
    s.uram = physical - window;
    uint64_t address = s.uram;
    int resource = 0;
    Buffer *buffers[] = {s.a, s.b, s.c};
    for (int operand = 0; operand < 3; ++operand) {
        for (int i = 0; i < counts[operand]; ++i) {
            buffers[operand][i] = {address, DAYAN_LOCAL_RESOURCE(resource++)};
            address += bytes[operand];
        }
    }
    Cluster &cluster = *(Cluster *)shared;
    cluster.uram[lane] = physical;
    check(dayan_cluster_barrier(4, &cluster.barrier));
    LoadArgs loads[] = {{&s, false}, {&s, true}};
    dayan_thread_resources_t resources = {};
    resources.lsu_tcsr_count = 1;
    resources.local_resource_count = LocalResources;
    resources.shared_resource_count = SharedResources;
    check(dayan_create_thread("load-a", loader, &loads[0], &resources));
    check(dayan_create_thread("load-b", loader, &loads[1], &resources));
    check(dayan_create_thread("post", post, &s, &resources));
    calc(&s);
    check(dayan_thread_join_all());
    check(dayan_cluster_barrier(4, &cluster.barrier));
}

DAYAN_KERNEL_NAMED_RESOURCE_DESC(
    matmul_fp4_4c, MatmulFp4Shape::kWorkers,
    MatmulFp4Shape::kAtomicBytes, MatmulFp4Shape::kUramBytes,
    3, 1, 0, 0, 2 * MatmulFp4Shape::kSlices + 2,
    2 * (MatmulFp4Shape::kSlices + 2));

#if !defined(__DAYAN_DEVICE__)

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

extern "C" __host__ char *getenv(const char *name);

__host__ static void usage(const char *program)
{
    std::printf("usage: %s [M N K]\n", program);
    std::printf("  defaults: M=1024 N=2048 K=6144\n");
    std::printf("  N must be a multiple of 16; odd K is supported.\n");
    std::printf("  E2M1 inputs pack two values per byte, low nibble first.\n");
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

__host__ static float fp4_to_float(uint8_t value)
{
    static constexpr float magnitude[] = {0.0f, 0.5f, 1.0f, 1.5f,
                                         2.0f, 3.0f, 4.0f, 6.0f};
    return value & 8 ? -magnitude[value & 7] : magnitude[value & 7];
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

__host__ static void initialize_inputs(std::vector<uint8_t> *a,
                                       std::vector<uint8_t> *bt,
                                       size_t m, size_t n, size_t k)
{
    const size_t stride = (k + 1) / 2;
    // Exercise all E2M1 encodings. Odd-K padding is nonzero to catch overreads.
    for (size_t row = 0; row < m; ++row) {
        const size_t seed = row * 13U + row / 17U;
        for (size_t inner = 0; inner < k; inner += 2) {
            const uint8_t lo = (seed + inner * 3U + inner / 17U) % 16U;
            const uint8_t hi = inner + 1 < k ?
                (seed + (inner + 1) * 3U + (inner + 1) / 17U) % 16U : 7;
            (*a)[row * stride + inner / 2] = lo | (hi << 4);
        }
    }
    for (size_t column = 0; column < n; ++column) {
        const size_t seed = column * 11U + column / 19U;
        for (size_t inner = 0; inner < k; inner += 2) {
            const uint8_t lo = (seed + inner * 5U + inner / 19U) % 16U;
            const uint8_t hi = inner + 1 < k ?
                (seed + (inner + 1) * 5U + (inner + 1) / 19U) % 16U : 7;
            (*bt)[column * stride + inner / 2] = lo | (hi << 4);
        }
    }
}

__host__ static void reference_matmul(const std::vector<uint8_t> &a,
                                      const std::vector<uint8_t> &bt,
                                      std::vector<float> *reference,
                                      size_t m, size_t n, size_t k)
{
    const size_t stride = (k + 1) / 2;
    std::vector<float> decoded_a(m * k), decoded_bt(n * k);
    for (size_t row = 0; row < m; ++row)
        for (size_t inner = 0; inner < k; ++inner)
            decoded_a[row * k + inner] =
                fp4_to_float(a[row * stride + inner / 2] >> ((inner & 1) * 4));
    for (size_t column = 0; column < n; ++column)
        for (size_t inner = 0; inner < k; ++inner)
            decoded_bt[column * k + inner] =
                fp4_to_float(bt[column * stride + inner / 2] >> ((inner & 1) * 4));
    for (size_t row = 0; row < m; ++row) {
        for (size_t column = 0; column < n; ++column) {
            float sum = 0.0f;
            for (size_t inner = 0; inner < k; ++inner)
                sum += decoded_a[row * k + inner] * decoded_bt[column * k + inner];
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
    std::printf("[host] PASS: all %zu outputs match the FP4 reference\n",
                actual.size());
    return 0;
}

__host__ int main(int argc, char **argv)
{
    uint64_t m = 1024;
    uint64_t n = 2048;
    uint64_t k = 6144;
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
    if (argc != 1 && argc != 4) {
        usage(argv[0]);
        return 2;
    }
    if (argc == 4 &&
        (parse_dimension(argv[1], &m) < 0 ||
         parse_dimension(argv[2], &n) < 0 ||
         parse_dimension(argv[3], &k) < 0)) {
        std::printf("M, N and K must be positive decimal integers\n");
        return 2;
    }
    if (m > UINT64_C(0x7ffff000) || n > UINT64_C(0x7ffff000) ||
        k > UINT64_C(0x7ffff000) || (n % 16U) != 0) {
        std::printf("unsupported shape: require M>0, N%%16==0 and "
                    "32-bit dimensions\n");
        return 2;
    }
    if (!checked_elements(m, (k + 1) / 2, sizeof(uint8_t), &a_elements, &a_bytes) ||
        !checked_elements(n, (k + 1) / 2, sizeof(uint8_t), &bt_elements,
                          &bt_bytes) ||
        !checked_elements(m, n, sizeof(float), &c_elements, &c_bytes) ||
        a_bytes > 64U * 1024U * 1024U ||
        bt_bytes > 64U * 1024U * 1024U ||
        c_bytes > 64U * 1024U * 1024U) {
        std::printf("shape is too large for this demo\n");
        return 2;
    }

    std::vector<uint8_t> a(a_elements);
    std::vector<uint8_t> bt(bt_elements);
    std::vector<float> c(c_elements, 0.0f);
    std::vector<float> reference(c_elements, 0.0f);
    initialize_inputs(&a, &bt, (size_t)m, (size_t)n, (size_t)k);
    reference_matmul(a, bt, &reference, (size_t)m, (size_t)n, (size_t)k);

    MatmulFp4Config config = {
        m, n, k, (k + 1) / 2, (k + 1) / 2,
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
    int status = 0;

    npu_runtime_options_init(&runtime_options);
    runtime_options.default_timeout_ms = timeout_ms;
    socket_path = getenv("DAYAN_SERVER_SOCKET");
    if (socket_path && *socket_path)
        runtime_options.server_socket = socket_path;

    status = npu_runtime_open(&runtime, &runtime_options);
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
        const uint64_t cluster_tiles_m =
            (m + 2U * MatmulFp4Shape::kTileM - 1U) /
            (2U * MatmulFp4Shape::kTileM);
        const uint64_t cluster_tiles_n =
            (n + 2U * MatmulFp4Shape::kTileN - 1U) /
            (2U * MatmulFp4Shape::kTileN);

        npu_kernel_launch_attrs_init(&launch_attrs, context, stream,
                                     "matmul-fp4-4c");
        launch_attrs.config.task_dim = NPUTaskDim3{4, 1, 1};
        launch_attrs.config.task_mode = NPUTaskMode::kFourCore;
        launch_attrs.timeout_ms = timeout_ms;
        std::printf("[host] launch matmul_fp4_4c: A[%llu,%llu] x "
                    "BT[%llu,%llu] -> C[%llu,%llu], tile=%ux%ux%u, "
                    "slices=%u, cluster-grid=%llux%llu, buffered=%s, loaders=A+B\n",
                    (unsigned long long)m, (unsigned long long)k,
                    (unsigned long long)n, (unsigned long long)k,
                    (unsigned long long)m, (unsigned long long)n,
                    (unsigned)MatmulFp4Shape::kTileM,
                    (unsigned)MatmulFp4Shape::kTileN,
                    (unsigned)MatmulFp4Shape::kTileK,
                    (unsigned)MatmulFp4Shape::kSlices,
                    (unsigned long long)cluster_tiles_m,
                    (unsigned long long)cluster_tiles_n, m < n ? "A" : "B");
        status = matmul_fp4_4c(config_buffer, a_buffer, bt_buffer,
                                c_buffer, &launch_attrs);
    }
    if (status == 0)
        status = download(stream, c.data(), c_buffer, c_bytes, timeout_ms);
    if (status == 0)
        status = validate_output(c, reference, (size_t)n);
    if (status < 0)
        std::printf("matmul_fp4_4c demo failed: %d\n", status);

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
