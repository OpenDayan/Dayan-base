#include <dayan_kernel.h>
#include <stdint.h>

#ifndef MATMUL_BF16_4C_SLICES
#define MATMUL_BF16_4C_SLICES 3
#endif

#ifndef MATMUL_BF16_4C_URAM_BYTES
#define MATMUL_BF16_4C_URAM_BYTES (1024U * 1024U)
#endif

#ifndef MATMUL_BF16_4C_ATOMIC_BYTES
#define MATMUL_BF16_4C_ATOMIC_BYTES 1024U
#endif

#ifndef MATMUL_BF16_4C_TRACE
#define MATMUL_BF16_4C_TRACE 0
#endif

#ifndef MATMUL_BF16_4C_TILE_M
#define MATMUL_BF16_4C_TILE_M 512
#endif

#ifndef MATMUL_BF16_4C_TILE_N
#define MATMUL_BF16_4C_TILE_N 256
#endif

#ifndef MATMUL_BF16_4C_TILE_K
#define MATMUL_BF16_4C_TILE_K 256
#endif

#ifndef MATMUL_BF16_4C_SLICE_ROW_ALIGNMENT
#define MATMUL_BF16_4C_SLICE_ROW_ALIGNMENT 32
#endif

/*
 * B is supplied in transposed storage: A[M][K] * BT[N][K] -> C[M][N].
 * Strides are measured in bytes.  The four cluster lanes form this grid:
 *
 *                 N-even tile       N-odd tile
 *   M-even tile      core 00           core 01
 *   M-odd tile       core 10           core 11
 *
 * A is sourced from DRAM by cores 00/11 and shared horizontally.  B is
 * sourced from DRAM by cores 01/10 and shared vertically.  Thus each A and B
 * tile is fetched from external memory once per pair instead of once per core.
 */
struct MatmulBf16Config {
    uint64_t m;
    uint64_t n;
    uint64_t k;
    uint64_t a_stride;
    uint64_t bt_stride;
    uint64_t c_stride;
};

#if defined(__DAYAN_DEVICE__)

#include <dayan_sys.h>
#include <stdio.h>
/* Only raw instruction builders and the public DAPI are used. */
#define _NPU_ALIAS_HPP_
#include <npu_instruction.hpp>
#include <npu_lsu_csr.hpp>
#include <npu_mat_csr.hpp>

/* See the corresponding explanation in matmul_bf16_1c.cpp. */
__asm__(".globl z\n.set z, 0");

namespace matmul_bf16_4c_detail {

#if MATMUL_BF16_4C_TRACE
#define MATMUL_4C_TRACE(...)                                                   \
    do {                                                                       \
        printf(__VA_ARGS__);                                                   \
        fflush(stdout);                                                        \
    } while (0)
#else
#define MATMUL_4C_TRACE(...) ((void)0)
#endif

struct Constants {
    enum : int {
        kLsuA = 0,
        kLsuB = 1,
        kLsuCStore = 3,
        kTileM = MATMUL_BF16_4C_TILE_M,
        kTileN = MATMUL_BF16_4C_TILE_N,
        kTileK = MATMUL_BF16_4C_TILE_K,
        kSliceRowAlignment = MATMUL_BF16_4C_SLICE_ROW_ALIGNMENT,
        kClusterRows = 2,
        kClusterColumns = 2,
        kClusterCores = 4,
        kUramAlignment = 32,
        kUramWindowBase = 0x00800000,
        kUramWindowStride = 2 * 1024 * 1024,
    };
};

static_assert(MATMUL_BF16_4C_SLICES == 1 ||
                  MATMUL_BF16_4C_SLICES == 3,
              "MATMUL_BF16_4C_SLICES must be 1 or 3");
static_assert(Constants::kSliceRowAlignment > 0 &&
                  (Constants::kSliceRowAlignment &
                   (Constants::kSliceRowAlignment - 1)) == 0,
              "slice row alignment must be a power of two");
static_assert(Constants::kTileM % Constants::kSliceRowAlignment == 0,
              "tile M must be aligned to the slice row alignment");
static_assert((Constants::kUramWindowStride &
               (Constants::kUramWindowStride - 1)) == 0,
              "URAM window stride must be a power of two");

static constexpr uint64_t align_up(uint64_t value, uint64_t alignment)
{
    return (value + alignment - 1U) & ~(alignment - 1U);
}

static inline int min_int(int left, int right)
{
    return left < right ? left : right;
}

template <int SliceCount> struct UramLayout {
    static constexpr int slice_offset(int slice)
    {
        if (slice <= 0)
            return 0;
        if (slice >= SliceCount)
            return Constants::kTileM;

        const int ideal =
            (Constants::kTileM * slice + SliceCount / 2) / SliceCount;
        return ((ideal + Constants::kSliceRowAlignment / 2) /
                Constants::kSliceRowAlignment) *
               Constants::kSliceRowAlignment;
    }

    static constexpr int slice_rows(int slice)
    {
        return slice_offset(slice + 1) - slice_offset(slice);
    }

    static constexpr uint64_t a_bytes(int slice)
    {
        return align_up((uint64_t)slice_rows(slice) * Constants::kTileK *
                            sizeof(uint16_t),
                        Constants::kUramAlignment);
    }

    static constexpr uint64_t b_bytes()
    {
        return align_up((uint64_t)Constants::kTileN * Constants::kTileK *
                            sizeof(uint16_t),
                        Constants::kUramAlignment);
    }

    static constexpr uint64_t c_bytes(int slice)
    {
        return align_up((uint64_t)slice_rows(slice) * Constants::kTileN *
                            sizeof(float),
                        Constants::kUramAlignment);
    }

    static constexpr uint64_t a_base(int slice)
    {
        uint64_t base = 0;

        for (int index = 0; index < slice; ++index)
            base = align_up(base + a_bytes(index),
                            Constants::kUramAlignment);
        return base;
    }

    static constexpr uint64_t a_end()
    {
        return align_up(a_base(SliceCount - 1) + a_bytes(SliceCount - 1),
                        Constants::kUramAlignment);
    }

    static constexpr uint64_t b_base(int buffer)
    {
        return align_up(a_end() + (uint64_t)buffer * b_bytes(),
                        Constants::kUramAlignment);
    }

    static constexpr uint64_t c_start()
    {
        return align_up(b_base(1) + b_bytes(), Constants::kUramAlignment);
    }

    static constexpr uint64_t c_base(int slice)
    {
        uint64_t base = c_start();

        for (int index = 0; index < slice; ++index)
            base = align_up(base + c_bytes(index),
                            Constants::kUramAlignment);
        return base;
    }

    static constexpr uint64_t total_bytes()
    {
        return align_up(c_base(SliceCount - 1) + c_bytes(SliceCount - 1),
                        Constants::kUramAlignment);
    }
};

using Layout = UramLayout<MATMUL_BF16_4C_SLICES>;

static_assert(Layout::slice_rows(0) % Constants::kSliceRowAlignment == 0,
              "slice 0 must be row aligned");
static_assert(MATMUL_BF16_4C_SLICES == 1 ||
                  (Layout::slice_rows(1) %
                       Constants::kSliceRowAlignment == 0 &&
                   Layout::slice_rows(2) %
                       Constants::kSliceRowAlignment == 0),
              "all three slices must be row aligned");
static_assert(Layout::total_bytes() <= MATMUL_BF16_4C_URAM_BYTES,
              "output-stationary matmul URAM layout is too large");

struct SharedBuffer {
    dayan_shared_resource_t resource;
    uint64_t generation;
    uint64_t reserved;
};

struct ClusterShared {
    dayan_cluster_barrier_t barrier;
    int32_t status;
    uint32_t status_lock;
    uint64_t uram_base[Constants::kClusterCores];
    SharedBuffer a[Constants::kClusterRows][MATMUL_BF16_4C_SLICES];
    SharedBuffer b[Constants::kClusterColumns][2];
};

static_assert(sizeof(ClusterShared) <= MATMUL_BF16_4C_ATOMIC_BYTES,
              "cluster matmul shared state exceeds its atomic-cache request");

struct BBuffer {
    dayan_resource_t resource;
    volatile uint64_t generation;
    uint64_t uram;
};

struct MatmulState {
    int m;
    int n;
    int k;
    int count_m;
    int count_n;
    int count_k;
    uint32_t lane;
    uint32_t row_lane;
    uint32_t column_lane;
    uint64_t a_base;
    uint64_t bt_base;
    uint64_t c_base;
    uint64_t a_stride;
    uint64_t bt_stride;
    uint64_t c_stride;
    uint64_t uram_base;
    ClusterShared *cluster;
    BBuffer b[2];
    volatile int status;
    volatile uint32_t start;
    volatile uint32_t abort;
};

static inline uint32_t a_source_lane(uint32_t row_lane)
{
    return row_lane == 0U ? 0U : 3U;
}

static inline uint32_t b_source_lane(uint32_t column_lane)
{
    return column_lane == 0U ? 2U : 1U;
}

static inline int rows_for_slice(const MatmulState *state, int tile_m,
                                 uint32_t row_lane, int slice)
{
    const int global_tile_m = tile_m * Constants::kClusterRows +
                              (int)row_lane;
    const int row_base = global_tile_m * Constants::kTileM +
                         Layout::slice_offset(slice);

    if (state->m <= row_base)
        return 0;
    return min_int(state->m - row_base, Layout::slice_rows(slice));
}

static inline int active_slices(const MatmulState *state, int tile_m,
                                uint32_t row_lane)
{
    int count = 0;

    for (int slice = 0; slice < MATMUL_BF16_4C_SLICES; ++slice) {
        if (rows_for_slice(state, tile_m, row_lane, slice) > 0)
            ++count;
    }
    return count;
}

static inline int cols_for_tile(const MatmulState *state, int tile_n,
                                uint32_t column_lane)
{
    const int global_tile_n = tile_n * Constants::kClusterColumns +
                              (int)column_lane;
    const int column_base = global_tile_n * Constants::kTileN;

    if (state->n <= column_base)
        return 0;
    return min_int(state->n - column_base, Constants::kTileN);
}

static inline int k_for_tile(const MatmulState *state, int tile_k)
{
    const int k_base = tile_k * Constants::kTileK;

    if (state->k <= k_base)
        return 0;
    return min_int(state->k - k_base, Constants::kTileK);
}

static inline uint64_t generation_for(const MatmulState *state, int tile_m,
                                      int tile_n, int tile_k)
{
    return ((uint64_t)tile_m * (uint64_t)state->count_n +
            (uint64_t)tile_n) * (uint64_t)state->count_k +
           (uint64_t)tile_k;
}

static inline int remember_error(MatmulState *state, int status)
{
    if (status < 0) {
        if (state->status == 0)
            state->status = status;
        while (__atomic_exchange_n(&state->cluster->status_lock, 1U,
                                    __ATOMIC_ACQUIRE)) {
            while (__atomic_load_n(&state->cluster->status_lock,
                                   __ATOMIC_RELAXED))
                dayan_yield();
        }
        if (state->cluster->status == 0)
            state->cluster->status = (int32_t)status;
        __atomic_store_n(&state->cluster->status_lock, 0U,
                         __ATOMIC_RELEASE);
    }
    return status;
}

/* major is the contiguous byte count in a row; minor is the row count. */
template <long TcsrId>
static inline void load_matrix(uint64_t major_left, uint64_t minor_left,
                               uint64_t major_tile, uint64_t minor_tile,
                               uint64_t uram_base, uint64_t source_base)
{
    if (minor_left == 0 || major_left == 0)
        return;

    const uint64_t real_minor =
        minor_left < minor_tile ? minor_left : minor_tile;
    __tx_lsu_tcsr_write<TcsrId * LSU_GROUPSZ + LSU_DIM1>(real_minor - 1U);

    if (major_left >= major_tile) {
        __tx_lsu_tcsr_write<TcsrId * LSU_GROUPSZ + LSU_DIM0>(
            major_tile / 64U - 1U);
        __tx_lsu_tcsr_write<TcsrId * LSU_GROUPSZ + LSU_REQUESTREADSIZE>(63);
        __tx_lsu_dma_d2u<TcsrId>((long)uram_base, (long)source_base);
        return;
    }

    while (major_left) {
        uint64_t handled_bytes = major_left & ~0x3fULL;
        const uint64_t remaining_bytes = major_left & 0x3fULL;

        if (handled_bytes == 0 && remaining_bytes > 0) {
            __tx_lsu_tcsr_write<TcsrId * LSU_GROUPSZ + LSU_DIM0>(0);
            __tx_lsu_tcsr_write<
                TcsrId * LSU_GROUPSZ + LSU_REQUESTREADSIZE>(
                remaining_bytes - 1U);
            handled_bytes = remaining_bytes;
        } else {
            __tx_lsu_tcsr_write<TcsrId * LSU_GROUPSZ + LSU_DIM0>(
                handled_bytes / 64U - 1U);
            __tx_lsu_tcsr_write<
                TcsrId * LSU_GROUPSZ + LSU_REQUESTREADSIZE>(63);
        }
        __tx_lsu_dma_d2u<TcsrId>((long)uram_base, (long)source_base);
        uram_base += handled_bytes;
        source_base += handled_bytes;
        major_left -= handled_bytes;
    }
}

template <long TcsrId>
static inline void store_matrix(uint64_t major_left, uint64_t minor_left,
                                uint64_t major_tile, uint64_t minor_tile,
                                uint64_t uram_base, uint64_t dram_base)
{
    if (minor_left == 0 || major_left == 0)
        return;

    const uint64_t real_minor =
        minor_left < minor_tile ? minor_left : minor_tile;
    __tx_lsu_tcsr_write<TcsrId * LSU_GROUPSZ + LSU_DIM1>(real_minor - 1U);

    if (major_left >= major_tile) {
        __tx_lsu_tcsr_write<TcsrId * LSU_GROUPSZ + LSU_DIM0>(
            major_tile / 64U - 1U);
        __tx_lsu_tcsr_write<TcsrId * LSU_GROUPSZ + LSU_REQUESTREADSIZE>(63);
        __tx_lsu_dma_u2d<TcsrId>((long)dram_base, (long)uram_base);
        return;
    }

    while (major_left) {
        uint64_t handled_bytes = major_left & ~0x3fULL;
        const uint64_t remaining_bytes = major_left & 0x3fULL;

        if (handled_bytes == 0 && remaining_bytes > 0) {
            __tx_lsu_tcsr_write<TcsrId * LSU_GROUPSZ + LSU_DIM0>(0);
            __tx_lsu_tcsr_write<
                TcsrId * LSU_GROUPSZ + LSU_REQUESTREADSIZE>(
                remaining_bytes - 1U);
            handled_bytes = remaining_bytes;
        } else {
            __tx_lsu_tcsr_write<TcsrId * LSU_GROUPSZ + LSU_DIM0>(
                handled_bytes / 64U - 1U);
            __tx_lsu_tcsr_write<
                TcsrId * LSU_GROUPSZ + LSU_REQUESTREADSIZE>(63);
        }
        __tx_lsu_dma_u2d<TcsrId>((long)dram_base, (long)uram_base);
        uram_base += handled_bytes;
        dram_base += handled_bytes;
        major_left -= handled_bytes;
    }
}

static inline int wait_for_b_generation(BBuffer *buffer,
                                        uint64_t expected_generation)
{
    while (buffer->generation != expected_generation)
        dayan_yield();
    return dayan_resource_wait_ready(&buffer->resource);
}

static inline int wait_for_shared_generation(
    SharedBuffer *buffer, uint64_t expected_generation)
{
    /* Do not wait for FREE after observing an older generation.  A source
     * hart can allocate the resource and publish expected_generation between
     * those two observations; sleeping for FREE would then deadlock because
     * this hart is one of the consumers that must release that generation. */
    while (__atomic_load_n(&buffer->generation, __ATOMIC_ACQUIRE) !=
           expected_generation)
        dayan_yield();
    return dayan_shared_resource_wait_ready(&buffer->resource);
}

template <long GroupBase, int Slice>
static inline void configure_consumer(const MatmulState *state)
{
    long checksum = 0;
    constexpr int rows = Layout::slice_rows(Slice);
    constexpr uint64_t swizzle = MAT_BATCH_SWIZZLE(5, 3, 5);
    const bool a_is_remote = state->lane != a_source_lane(state->row_lane);
    const long a_input_stride = a_is_remote ?
        Constants::kTileK * (long)sizeof(uint16_t) :
        (long)state->a_stride;

    set_lsu_matrix<GroupBase + Constants::kLsuA>(
        checksum, rows, Constants::kTileK * (int)sizeof(uint16_t),
        a_input_stride, Constants::kTileK * (int)sizeof(uint16_t));
    set_lsu_matrix<GroupBase + Constants::kLsuCStore>(
        checksum, rows, Constants::kTileN * (int)sizeof(float),
        Constants::kTileN * (int)sizeof(float), (long)state->c_stride);
    __tx_lsu_tcsr_write<
        (GroupBase + Constants::kLsuA) * LSU_GROUPSZ +
        LSU_URAMWRITESWIZZLEMBS>(a_is_remote ? 0U : swizzle);
    __tx_lsu_tcsr_write<
        (GroupBase + Constants::kLsuA) * LSU_GROUPSZ +
        LSU_URAMREADSWIZZLEMBS>(swizzle);
    __tx_lsu_tcsr_write<
        (GroupBase + Constants::kLsuCStore) * LSU_GROUPSZ +
        LSU_URAMREADSWIZZLEMBS>(swizzle);

    __tx_mat_tcsr_write<GroupBase * MAT_GROUPSZ + MAT_M>(rows - 1);
    __tx_mat_tcsr_write<GroupBase * MAT_GROUPSZ + MAT_N>(
        Constants::kTileN - 1);
    __tx_mat_tcsr_write<GroupBase * MAT_GROUPSZ + MAT_K>(
        Constants::kTileK - 1);
    __tx_mat_tcsr_write<GroupBase * MAT_GROUPSZ + MAT_ASTRIDE>(
        Constants::kTileK * sizeof(uint16_t));
    __tx_mat_tcsr_write<GroupBase * MAT_GROUPSZ + MAT_BSTRIDE>(
        Constants::kTileK * sizeof(uint16_t));
    __tx_mat_tcsr_write<GroupBase * MAT_GROUPSZ + MAT_CSTRIDE>(
        Constants::kTileN * sizeof(float));
    __tx_mat_tcsr_write<GroupBase * MAT_GROUPSZ + MAT_ASWIZZLEMBS>(swizzle);
    __tx_mat_tcsr_write<GroupBase * MAT_GROUPSZ + MAT_BSWIZZLEMBS>(swizzle);
    __tx_mat_tcsr_write<GroupBase * MAT_GROUPSZ + MAT_CSWIZZLEMBS>(swizzle);
    __tx_mat_tcsr_write<GroupBase * MAT_GROUPSZ + MAT_PERSISTENTMODE>(0);
    __tx_mat_tcsr_write<GroupBase * MAT_GROUPSZ + MAT_DATATYPE>(
        BaseNPUInstBuilder::DT_BF16);
}

template <long GroupBase, int Slice>
static void consume_slice(MatmulState *state)
{
    constexpr int slice_offset = Layout::slice_offset(Slice);
    constexpr int slice_rows = Layout::slice_rows(Slice);
    const uint64_t a_uram = state->uram_base + Layout::a_base(Slice);
    const uint64_t c_uram = state->uram_base + Layout::c_base(Slice);
    const uint32_t source_lane = a_source_lane(state->row_lane);
    const uint32_t other_column = state->column_lane ^ 1U;
    const bool source = state->lane == source_lane;
    SharedBuffer *shared_a =
        &state->cluster->a[state->row_lane][Slice];

    while (!state->start)
        dayan_yield();
    if (state->abort)
        return;

    MATMUL_4C_TRACE("[4c] lane=%u slice=%d consumer-start\n",
                    (unsigned)state->lane, Slice);

    configure_consumer<GroupBase, Slice>(state);

    for (int tile_m = 0; tile_m < state->count_m; ++tile_m) {
        const int real_rows = rows_for_slice(
            state, tile_m, state->row_lane, Slice);

        if (real_rows == 0)
            continue;
        for (int tile_n = 0; tile_n < state->count_n; ++tile_n) {
            const int real_cols = cols_for_tile(
                state, tile_n, state->column_lane);
            const int other_cols = cols_for_tile(
                state, tile_n, other_column);

            for (int tile_k = 0; tile_k < state->count_k; ++tile_k) {
                const uint64_t generation = generation_for(
                    state, tile_m, tile_n, tile_k);
                const int real_k = k_for_tile(state, tile_k);
                const bool first_k = tile_k == 0;
                const bool last_k = tile_k + 1 == state->count_k;
                int status;

                if (source) {
                    const uint32_t users = (real_cols > 0 ? 1U : 0U) +
                                           (other_cols > 0 ? 1U : 0U);

                    if (!users)
                        continue;
                    status = dayan_shared_resource_wait_free(
                        &shared_a->resource);
                    if (status < 0) {
                        remember_error(state, status);
                        return;
                    }
                    status = dayan_shared_resource_alloc(
                        &shared_a->resource, 1U, users);
                    if (status < 0) {
                        remember_error(state, status);
                        return;
                    }
                    __atomic_store_n(&shared_a->generation, generation,
                                     __ATOMIC_RELEASE);

                    const int global_tile_m =
                        tile_m * Constants::kClusterRows +
                        (int)state->row_lane;
                    const uint64_t a_dram =
                        state->a_base +
                        (uint64_t)tile_k * Constants::kTileK *
                            sizeof(uint16_t) +
                        ((uint64_t)global_tile_m * Constants::kTileM +
                         slice_offset) * state->a_stride;

                    load_matrix<GroupBase + Constants::kLsuA>(
                        (uint64_t)real_k * sizeof(uint16_t), real_rows,
                        Constants::kTileK * sizeof(uint16_t), slice_rows,
                        a_uram, a_dram);
                    dayan_wait_lsu();
                    status = dayan_shared_resource_produce(
                        &shared_a->resource);
                    if (status < 0) {
                        remember_error(state, status);
                        return;
                    }
                    if (real_cols == 0)
                        continue;
                } else {
                    if (real_cols == 0)
                        continue;
                    status = wait_for_shared_generation(shared_a,
                                                        generation);
                    if (status < 0) {
                        remember_error(state, status);
                        return;
                    }
                    const uint64_t remote_a =
                        __atomic_load_n(
                            &state->cluster->uram_base[source_lane],
                            __ATOMIC_ACQUIRE) + Layout::a_base(Slice);

                    /*
                     * Remote URAM is a tensor-xbar (D-side) address.  Copy
                     * its complete physical slice without applying a second
                     * swizzle; MAT later reads the preserved layout with the
                     * normal A swizzle.
                     */
                    load_matrix<GroupBase + Constants::kLsuA>(
                        Constants::kTileK * sizeof(uint16_t), slice_rows,
                        Constants::kTileK * sizeof(uint16_t), slice_rows,
                        a_uram, remote_a);
                    dayan_wait_lsu();
                    status = dayan_shared_resource_consume(
                        &shared_a->resource);
                    if (status < 0) {
                        remember_error(state, status);
                        return;
                    }
                }

                MATMUL_4C_TRACE("[4c] lane=%u slice=%d gen=%llu A-ready\n",
                                (unsigned)state->lane, Slice,
                                (unsigned long long)generation);

                BBuffer *b = &state->b[generation & 1U];

                status = wait_for_b_generation(b, generation);
                if (status < 0) {
                    remember_error(state, status);
                    return;
                }

                MATMUL_4C_TRACE("[4c] lane=%u slice=%d gen=%llu B-ready\n",
                                (unsigned)state->lane, Slice,
                                (unsigned long long)generation);

                __tx_mat_tcsr_write<GroupBase * MAT_GROUPSZ + MAT_M>(
                    real_rows - 1);
                __tx_mat_tcsr_write<GroupBase * MAT_GROUPSZ + MAT_N>(
                    real_cols - 1);
                if (first_k)
                    __tx_mat_mm<GroupBase>((long)c_uram, (long)a_uram,
                                           (long)b->uram);
                else
                    __tx_mat_mma<GroupBase>((long)c_uram, (long)a_uram,
                                            (long)b->uram);
                dayan_wait_mat();
                MATMUL_4C_TRACE("[4c] lane=%u slice=%d gen=%llu MAT-done\n",
                                (unsigned)state->lane, Slice,
                                (unsigned long long)generation);

                status = dayan_resource_consume(&b->resource);
                if (status < 0) {
                    remember_error(state, status);
                    return;
                }
                if (source) {
                    status = dayan_shared_resource_consume(
                        &shared_a->resource);
                    if (status < 0) {
                        remember_error(state, status);
                        return;
                    }
                }

                if (last_k) {
                    const int global_tile_m =
                        tile_m * Constants::kClusterRows +
                        (int)state->row_lane;
                    const int global_tile_n =
                        tile_n * Constants::kClusterColumns +
                        (int)state->column_lane;
                    const uint64_t c_dram =
                        state->c_base +
                        (uint64_t)global_tile_n * Constants::kTileN *
                            sizeof(float) +
                        ((uint64_t)global_tile_m * Constants::kTileM +
                         slice_offset) * state->c_stride;

                    store_matrix<GroupBase + Constants::kLsuCStore>(
                        (uint64_t)real_cols * sizeof(float), real_rows,
                        Constants::kTileN * sizeof(float), slice_rows,
                        c_uram, c_dram);
                } else {
                    dayan_yield();
                }
            }
        }
    }
    dayan_wait_lsu();
    MATMUL_4C_TRACE("[4c] lane=%u slice=%d consumer-done\n",
                    (unsigned)state->lane, Slice);
}

static void consumer_0(void *argument)
{
    consume_slice<0, 0>((MatmulState *)argument);
}

#if MATMUL_BF16_4C_SLICES == 3
static void consumer_1(void *argument)
{
    consume_slice<4, 1>((MatmulState *)argument);
}

static void consumer_2(void *argument)
{
    consume_slice<2, 2>((MatmulState *)argument);
}
#endif

static inline void configure_producer(const MatmulState *state)
{
    long checksum = 0;
    constexpr uint64_t swizzle = MAT_BATCH_SWIZZLE(5, 3, 5);
    const bool remote = state->lane != b_source_lane(state->column_lane);
    const long input_stride = remote ?
        Constants::kTileK * (long)sizeof(uint16_t) :
        (long)state->bt_stride;

    set_lsu_matrix<Constants::kLsuB>(
        checksum, Constants::kTileN,
        Constants::kTileK * (int)sizeof(uint16_t), input_stride,
        Constants::kTileK * (int)sizeof(uint16_t));
    __tx_lsu_tcsr_write<
        Constants::kLsuB * LSU_GROUPSZ + LSU_URAMWRITESWIZZLEMBS>(
        remote ? 0U : swizzle);
    __tx_lsu_tcsr_write<
        Constants::kLsuB * LSU_GROUPSZ + LSU_URAMREADSWIZZLEMBS>(
        swizzle);
}

static int produce_b(MatmulState *state)
{
    const uint32_t source_lane = b_source_lane(state->column_lane);
    const uint32_t other_row = state->row_lane ^ 1U;
    const bool source = state->lane == source_lane;

    configure_producer(state);
    MATMUL_4C_TRACE("[4c] lane=%u B-producer-start source=%u\n",
                    (unsigned)state->lane, (unsigned)source);

    for (int tile_m = 0; tile_m < state->count_m; ++tile_m) {
        const int local_users = active_slices(
            state, tile_m, state->row_lane);
        const int remote_users = active_slices(state, tile_m, other_row);

        for (int tile_n = 0; tile_n < state->count_n; ++tile_n) {
            const int real_cols = cols_for_tile(
                state, tile_n, state->column_lane);

            if (real_cols == 0)
                continue;
            for (int tile_k = 0; tile_k < state->count_k; ++tile_k) {
                const uint64_t generation = generation_for(
                    state, tile_m, tile_n, tile_k);
                BBuffer *b = &state->b[generation & 1U];
                SharedBuffer *shared_b =
                    &state->cluster->b[state->column_lane][generation & 1U];
                const int real_k = k_for_tile(state, tile_k);
                int status;

                if (source) {
                    if (!local_users && !remote_users)
                        continue;
                    /* Both resources may still describe an older generation
                     * of this physical double buffer, even when the current
                     * generation has no user on that side. */
                    status = dayan_resource_wait_free(&b->resource);
                    if (status < 0)
                        return remember_error(state, status);
                    status = dayan_shared_resource_wait_free(
                        &shared_b->resource);
                    if (status < 0)
                        return remember_error(state, status);
                    if (remote_users) {
                        status = dayan_shared_resource_alloc(
                            &shared_b->resource, 1U, 1U);
                        if (status < 0)
                            return remember_error(state, status);
                        __atomic_store_n(&shared_b->generation, generation,
                                         __ATOMIC_RELEASE);
                    }
                    if (local_users) {
                        status = dayan_resource_alloc(
                            &b->resource, 1U, (uint32_t)local_users);
                        if (status < 0)
                            return remember_error(state, status);
                        b->generation = generation;
                    }

                    const int global_tile_n =
                        tile_n * Constants::kClusterColumns +
                        (int)state->column_lane;
                    const uint64_t bt_dram =
                        state->bt_base +
                        (uint64_t)tile_k * Constants::kTileK *
                            sizeof(uint16_t) +
                        (uint64_t)global_tile_n * Constants::kTileN *
                            state->bt_stride;

                    load_matrix<Constants::kLsuB>(
                        (uint64_t)real_k * sizeof(uint16_t), real_cols,
                        Constants::kTileK * sizeof(uint16_t),
                        Constants::kTileN, b->uram, bt_dram);
                    dayan_wait_lsu();

                    if (local_users) {
                        status = dayan_resource_produce(&b->resource);
                        if (status < 0)
                            return remember_error(state, status);
                    }
                    if (remote_users) {
                        status = dayan_shared_resource_produce(
                            &shared_b->resource);
                        if (status < 0)
                            return remember_error(state, status);
                    }
                } else {
                    if (!local_users)
                        continue;
                    status = dayan_resource_wait_free(&b->resource);
                    if (status < 0)
                        return remember_error(state, status);
                    status = dayan_resource_alloc(
                        &b->resource, 1U, (uint32_t)local_users);
                    if (status < 0)
                        return remember_error(state, status);
                    b->generation = generation;

                    status = wait_for_shared_generation(shared_b,
                                                        generation);
                    if (status < 0)
                        return remember_error(state, status);
                    const uint64_t remote_b =
                        __atomic_load_n(
                            &state->cluster->uram_base[source_lane],
                            __ATOMIC_ACQUIRE) +
                        Layout::b_base((int)(generation & 1U));

                    /* Preserve the source buffer's complete physical
                     * swizzled image while reading it through tensor xbar. */
                    load_matrix<Constants::kLsuB>(
                        Constants::kTileK * sizeof(uint16_t),
                        Constants::kTileN,
                        Constants::kTileK * sizeof(uint16_t),
                        Constants::kTileN, b->uram, remote_b);
                    dayan_wait_lsu();

                    status = dayan_shared_resource_consume(
                        &shared_b->resource);
                    if (status < 0)
                        return remember_error(state, status);
                    status = dayan_resource_produce(&b->resource);
                    if (status < 0)
                        return remember_error(state, status);
                }
                MATMUL_4C_TRACE("[4c] lane=%u gen=%llu B-published\n",
                                (unsigned)state->lane,
                                (unsigned long long)generation);
                dayan_yield();
            }
        }
    }
    MATMUL_4C_TRACE("[4c] lane=%u B-producer-done\n",
                    (unsigned)state->lane);
    return 0;
}

static inline int config_valid(const MatmulBf16Config *config)
{
    const uint64_t int_limit = 0x7ffff000ULL;

    if (!config || !config->m || !config->n || !config->k ||
        config->m > int_limit || config->n > int_limit ||
        config->k > int_limit || (config->n % 16U) != 0 ||
        (config->k % Constants::kTileK) != 0 ||
        config->a_stride < config->k * sizeof(uint16_t) ||
        config->bt_stride < config->k * sizeof(uint16_t) ||
        config->c_stride < config->n * sizeof(float) ||
        (config->a_stride & 63U) || (config->bt_stride & 63U) ||
        (config->c_stride & 63U) ||
        config->a_stride > 0x7fffffffffffffffULL ||
        config->bt_stride > 0x7fffffffffffffffULL ||
        config->c_stride > 0x7fffffffffffffffULL)
        return 0;
    return 1;
}

static int initialize_cluster_shared(ClusterShared *cluster, uint32_t lane,
                                     uint64_t uram)
{
    int status = 0;

    if (lane == 0U) {
        cluster->status = 0;
        cluster->status_lock = 0;
        for (int row = 0; row < Constants::kClusterRows; ++row) {
            for (int slice = 0; slice < MATMUL_BF16_4C_SLICES; ++slice) {
                SharedBuffer *buffer = &cluster->a[row][slice];

                status = dayan_shared_resource_init(&buffer->resource);
                if (status < 0)
                    break;
                buffer->generation = ~0ULL;
            }
            if (status < 0)
                break;
        }
        for (int column = 0;
             status == 0 && column < Constants::kClusterColumns; ++column) {
            for (int buffer_index = 0; buffer_index < 2; ++buffer_index) {
                SharedBuffer *buffer = &cluster->b[column][buffer_index];

                status = dayan_shared_resource_init(&buffer->resource);
                if (status < 0)
                    break;
                buffer->generation = ~0ULL;
            }
        }
        if (status < 0)
            cluster->status = status;
    }
    __atomic_store_n(&cluster->uram_base[lane], uram, __ATOMIC_RELEASE);
    status = dayan_cluster_barrier(Constants::kClusterCores,
                                   &cluster->barrier);
    if (status < 0)
        return status;
    return __atomic_load_n(&cluster->status, __ATOMIC_ACQUIRE);
}

} // namespace matmul_bf16_4c_detail

#endif /* __DAYAN_DEVICE__ */

__kernel__ void matmul_bf16_4c(const MatmulBf16Config *config,
                               const uint16_t *a, const uint16_t *bt,
                               float *c)
{
    using namespace matmul_bf16_4c_detail;

    const uint32_t lane = dayan_task_id(DAYAN_TASK_X);
    MATMUL_4C_TRACE(
        "[4c] lane=%u kernel-enter config=%p valid=%d a=%p bt=%p c=%p "
        "mode=%u dim=%u,%u,%u shared=%p\n",
        (unsigned)lane, (const void *)config, config_valid(config),
        (const void *)a, (const void *)bt, (void *)c,
        (unsigned)dayan_task_mode(),
        (unsigned)dayan_task_dim(DAYAN_TASK_X),
        (unsigned)dayan_task_dim(DAYAN_TASK_Y),
        (unsigned)dayan_task_dim(DAYAN_TASK_Z), shared);
#if MATMUL_BF16_4C_TRACE
    const int debug_base = 200 + (int)(lane < 4U ? lane : 4U) * 16;
    if (!config_valid(config))
        dayan_proc_exit(-(debug_base + 1));
    if (!a)
        dayan_proc_exit(-(debug_base + 2));
    if (!bt)
        dayan_proc_exit(-(debug_base + 3));
    if (!c)
        dayan_proc_exit(-(debug_base + 4));
    if (dayan_task_mode() != 4U)
        dayan_proc_exit(-(debug_base + 5));
    if (dayan_task_dim(DAYAN_TASK_X) != 4U)
        dayan_proc_exit(-(debug_base + 6));
    if (dayan_task_dim(DAYAN_TASK_Y) != 1U)
        dayan_proc_exit(-(debug_base + 7));
    if (dayan_task_dim(DAYAN_TASK_Z) != 1U)
        dayan_proc_exit(-(debug_base + 8));
    if (lane >= 4U)
        dayan_proc_exit(-(debug_base + 9));
    if (!shared)
        dayan_proc_exit(-(debug_base + 10));
#else
    if (!config_valid(config) || !a || !bt || !c ||
        dayan_task_mode() != 4U ||
        dayan_task_dim(DAYAN_TASK_X) != 4U ||
        dayan_task_dim(DAYAN_TASK_Y) != 1U ||
        dayan_task_dim(DAYAN_TASK_Z) != 1U || lane >= 4U || !shared)
        dayan_proc_exit(-22);
#endif

    void *uram = dayan_uram_base();
    MATMUL_4C_TRACE("[4c] lane=%u uram=%p size=%llu need=%llu\n",
                    (unsigned)lane, uram,
                    (unsigned long long)dayan_uram_size(),
                    (unsigned long long)Layout::total_bytes());
#if MATMUL_BF16_4C_TRACE
    if (!uram)
        dayan_proc_exit(-(debug_base + 11));
    if (dayan_uram_size() < Layout::total_bytes())
        dayan_proc_exit(-(debug_base + 12));
#else
    if (!uram || dayan_uram_size() < Layout::total_bytes())
        dayan_proc_exit(-12);
#endif

    const uint64_t physical_uram = (uint64_t)(uintptr_t)uram;
    const uint64_t expected_window =
        Constants::kUramWindowBase +
        (uint64_t)lane * Constants::kUramWindowStride;
#if MATMUL_BF16_4C_TRACE
    if (physical_uram < expected_window)
        dayan_proc_exit(-(debug_base + 13));
    if (physical_uram - expected_window >
        Constants::kUramWindowStride - Layout::total_bytes())
        dayan_proc_exit(-(debug_base + 14));
#else
    if (physical_uram < expected_window ||
        physical_uram - expected_window >
            Constants::kUramWindowStride - Layout::total_bytes())
        dayan_proc_exit(-22);
#endif

    ClusterShared *cluster = (ClusterShared *)shared;
    int status = initialize_cluster_shared(cluster, lane, physical_uram);
    if (status < 0)
        dayan_proc_exit(status);
    MATMUL_4C_TRACE("[4c] lane=%u init-barrier-done uram=0x%llx\n",
                    (unsigned)lane, (unsigned long long)physical_uram);

    MatmulState state;
    state.m = (int)config->m;
    state.n = (int)config->n;
    state.k = (int)config->k;
    state.count_m =
        (state.m + Constants::kClusterRows * Constants::kTileM - 1) /
        (Constants::kClusterRows * Constants::kTileM);
    state.count_n =
        (state.n + Constants::kClusterColumns * Constants::kTileN - 1) /
        (Constants::kClusterColumns * Constants::kTileN);
    state.count_k = (state.k + Constants::kTileK - 1) /
                    Constants::kTileK;
    state.lane = lane;
    state.row_lane = lane >> 1;
    state.column_lane = lane & 1U;
    state.a_base = (uint64_t)(uintptr_t)a;
    state.bt_base = (uint64_t)(uintptr_t)bt;
    state.c_base = (uint64_t)(uintptr_t)c;
    state.a_stride = config->a_stride;
    state.bt_stride = config->bt_stride;
    state.c_stride = config->c_stride;
    state.uram_base = physical_uram &
                      (Constants::kUramWindowStride - 1U);
    state.cluster = cluster;
    state.status = 0;
    state.start = 0;
    state.abort = 0;

    for (int index = 0; index < 2; ++index) {
        status = dayan_resource_init(&state.b[index].resource);
        if (status < 0)
            dayan_proc_exit(status);
        state.b[index].generation = ~0ULL;
        state.b[index].uram = state.uram_base + Layout::b_base(index);
    }

    status = dayan_create_proc("consumer-0", consumer_0, &state);
#if MATMUL_BF16_4C_SLICES == 3
    if (status >= 0)
        status = dayan_create_proc("consumer-1", consumer_1, &state);
    if (status >= 0)
        status = dayan_create_proc("consumer-2", consumer_2, &state);
#endif
    if (status < 0) {
        state.status = status;
        state.abort = 1;
        state.start = 1;
        (void)dayan_join_all();
        dayan_proc_exit(status);
    }

    state.start = 1;
    MATMUL_4C_TRACE("[4c] lane=%u children-started\n", (unsigned)lane);
    status = produce_b(&state);
    if (status < 0)
        state.status = status;

    status = dayan_join_all();
    MATMUL_4C_TRACE("[4c] lane=%u join-done status=%d\n",
                    (unsigned)lane, status);
    if (status < 0 && state.status == 0)
        state.status = status;
    if (state.status < 0)
        remember_error(&state, state.status);

    status = dayan_cluster_barrier(Constants::kClusterCores,
                                   &cluster->barrier);
    MATMUL_4C_TRACE("[4c] lane=%u final-barrier-done status=%d\n",
                    (unsigned)lane, status);
    if (status < 0 && state.status == 0)
        state.status = status;
    const int cluster_status =
        __atomic_load_n(&cluster->status, __ATOMIC_ACQUIRE);
    if (state.status < 0)
        dayan_proc_exit(state.status);
    if (cluster_status < 0)
        dayan_proc_exit(cluster_status);
}

DAYAN_KERNEL_DESC(matmul_bf16_4c, MATMUL_BF16_4C_SLICES,
                  MATMUL_BF16_4C_ATOMIC_BYTES,
                  MATMUL_BF16_4C_URAM_BYTES);

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
    std::printf("  defaults: M=2048 N=2048 K=2048\n");
    std::printf("  N must be a multiple of 16; K must be a multiple of %u.\n",
                (unsigned)MATMUL_BF16_4C_TILE_K);
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
            const int code = (int)((column * 7U + inner * 5U) % 7U) - 3;

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
    uint64_t m = 2048;
    uint64_t n = 2048;
    uint64_t k = 2048;
    size_t a_elements = 0;
    size_t bt_elements = 0;
    size_t c_elements = 0;
    size_t a_bytes = 0;
    size_t bt_bytes = 0;
    size_t c_bytes = 0;
    const uint32_t timeout_ms = 1800000;

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
        k > UINT64_C(0x7ffff000) || (n % 16U) != 0 ||
        (k % MATMUL_BF16_4C_TILE_K) != 0) {
        std::printf("unsupported shape: require M>0, N%%16==0, "
                    "K%%%u==0 and 32-bit dimensions\n",
                    (unsigned)MATMUL_BF16_4C_TILE_K);
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
            (m + 2U * MATMUL_BF16_4C_TILE_M - 1U) /
            (2U * MATMUL_BF16_4C_TILE_M);
        const uint64_t cluster_tiles_n =
            (n + 2U * MATMUL_BF16_4C_TILE_N - 1U) /
            (2U * MATMUL_BF16_4C_TILE_N);

        npu_kernel_launch_attrs_init(&launch_attrs, context, stream,
                                     "matmul-bf16-4c");
        launch_attrs.config.task_dim = NPUTaskDim3{4, 1, 1};
        launch_attrs.config.task_mode = NPUTaskMode::kFourCore;
        launch_attrs.timeout_ms = timeout_ms;
        std::printf("[host] launch matmul_bf16_4c: A[%llu,%llu] x "
                    "BT[%llu,%llu] -> C[%llu,%llu], tile=%ux%ux%u, "
                    "slices=%u, cluster-grid=%llux%llu\n",
                    (unsigned long long)m, (unsigned long long)k,
                    (unsigned long long)n, (unsigned long long)k,
                    (unsigned long long)m, (unsigned long long)n,
                    (unsigned)MATMUL_BF16_4C_TILE_M,
                    (unsigned)MATMUL_BF16_4C_TILE_N,
                    (unsigned)MATMUL_BF16_4C_TILE_K,
                    (unsigned)MATMUL_BF16_4C_SLICES,
                    (unsigned long long)cluster_tiles_m,
                    (unsigned long long)cluster_tiles_n);
        status = matmul_bf16_4c(config_buffer, a_buffer, bt_buffer,
                                c_buffer, &launch_attrs);
    }
    if (status == 0)
        status = download(stream, c.data(), c_buffer, c_bytes, timeout_ms);
    if (status == 0)
        status = validate_output(c, reference, (size_t)n);
    if (status < 0)
        std::printf("matmul_bf16_4c demo failed: %d\n", status);

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
