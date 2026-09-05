#include <dayan_kernel.h>
#include <stdint.h>

#ifndef MATMUL_BF16_1C_SLICES
#define MATMUL_BF16_1C_SLICES 3
#endif

#ifndef MATMUL_BF16_1C_URAM_BYTES
#define MATMUL_BF16_1C_URAM_BYTES (1024U * 1024U)
#endif

/*
 * B is supplied in transposed storage: A[M][K] * BT[N][K] -> C[M][N].
 * Strides are measured in bytes.
 */
struct MatmulBf16Config {
    uint64_t m;
    uint64_t n;
    uint64_t k;
    uint64_t a_stride;
    uint64_t bt_stride;
    uint64_t c_stride;
};

enum : uint64_t {
    kMatmulBf16PerfMagic = 0x4d4d314350455246ULL,
};

struct MatmulBf16Perf {
    uint64_t magic;
    uint64_t m;
    uint64_t n;
    uint64_t k;
    uint64_t kernel_cycles;
    uint64_t pipeline_cycles;
};

enum : uint64_t {
    kMatmulBf16TileLong = 384,
    kMatmulBf16TileShort = 256,
    kMatmulBf16TileK = 256,
};

#if defined(__DAYAN_DEVICE__)

#include <dayan_intrinsics.h>
#include <dayan_sys.h>

namespace matmul_bf16_1c_detail {

struct Constants {
    enum : int {
        kLsuLocal = 0,
        kLsuShared = 0,
        kLsuCStore = 1,
        kTileK = kMatmulBf16TileK,
        kUramAlignment = 32,
        kUramWindowStride = 2 * 1024 * 1024,
    };
};

enum : uint64_t {
    kStride512Swizzle = MAT_BATCH_SWIZZLE(7, 2, 3),
    kStride1024Swizzle = MAT_BATCH_SWIZZLE(7, 3, 3),
};

static_assert(MATMUL_BF16_1C_SLICES == 1 ||
                  MATMUL_BF16_1C_SLICES == 3,
              "MATMUL_BF16_1C_SLICES must be 1 or 3");
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

static inline uint64_t read_cycle()
{
    uint64_t cycle;

    __asm__ volatile("csrr %0, mcycle" : "=r"(cycle) : : "memory");
    return cycle;
}

template <bool ProducerLoadsA, int SliceCount> struct UramLayout {
    enum : int {
        kTileM = ProducerLoadsA ? kMatmulBf16TileShort
                                : kMatmulBf16TileLong,
        kTileN = ProducerLoadsA ? kMatmulBf16TileLong
                                : kMatmulBf16TileShort,
    };

    static constexpr int slice_offset(int slice)
    {
        return (ProducerLoadsA ? kTileN : kTileM) *
               slice / SliceCount;
    }

    static constexpr int slice_size(int slice)
    {
        return slice_offset(slice + 1) - slice_offset(slice);
    }

    static constexpr int max_rows(int slice)
    {
        return ProducerLoadsA ? kTileM : slice_size(slice);
    }

    static constexpr int max_cols(int slice)
    {
        return ProducerLoadsA ? slice_size(slice) : kTileN;
    }

    static constexpr uint64_t local_bytes(int slice)
    {
        return align_up((uint64_t)slice_size(slice) * Constants::kTileK *
                            sizeof(uint16_t),
                        Constants::kUramAlignment);
    }

    static constexpr uint64_t shared_bytes()
    {
        return align_up((uint64_t)(ProducerLoadsA ? kTileM : kTileN) *
                            Constants::kTileK *
                            sizeof(uint16_t),
                        Constants::kUramAlignment);
    }

    static constexpr uint64_t c_bytes(int slice)
    {
        return align_up((uint64_t)max_rows(slice) * max_cols(slice) *
                            sizeof(float),
                        Constants::kUramAlignment);
    }

    static constexpr uint64_t local_base(int slice)
    {
        uint64_t base = 0;

        for (int index = 0; index < slice; ++index)
            base = align_up(base + local_bytes(index),
                            Constants::kUramAlignment);
        return base;
    }

    static constexpr uint64_t local_end()
    {
        return align_up(local_base(SliceCount - 1) +
                            local_bytes(SliceCount - 1),
                        Constants::kUramAlignment);
    }

    static constexpr uint64_t shared_base(int buffer)
    {
        return align_up(local_end() +
                            (uint64_t)buffer * shared_bytes(),
                        Constants::kUramAlignment);
    }

    static constexpr uint64_t c_start()
    {
        return align_up(shared_base(1) + shared_bytes(),
                        Constants::kUramAlignment);
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

using AProducerLayout = UramLayout<true, MATMUL_BF16_1C_SLICES>;
using BProducerLayout = UramLayout<false, MATMUL_BF16_1C_SLICES>;

static_assert(MATMUL_BF16_1C_SLICES != 3 ||
                  AProducerLayout::slice_size(0) == 128,
              "A-producer consumers must own 128 columns");
static_assert(MATMUL_BF16_1C_SLICES != 3 ||
                  BProducerLayout::slice_size(0) == 128,
              "B-producer consumers must own 128 rows");
static_assert(MATMUL_BF16_1C_SLICES != 3 ||
                  AProducerLayout::total_bytes() == 832U * 1024U,
              "A-producer three-slice layout must use 832 KiB");
static_assert(MATMUL_BF16_1C_SLICES != 3 ||
                  BProducerLayout::total_bytes() == 832U * 1024U,
              "B-producer three-slice layout must use 832 KiB");
static_assert(AProducerLayout::total_bytes() <= MATMUL_BF16_1C_URAM_BYTES,
              "A-producer matmul URAM layout is too large");
static_assert(BProducerLayout::total_bytes() <= MATMUL_BF16_1C_URAM_BYTES,
              "B-producer matmul URAM layout is too large");

struct SharedBuffer {
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
    uint64_t a_base;
    uint64_t bt_base;
    uint64_t c_base;
    uint64_t a_stride;
    uint64_t bt_stride;
    uint64_t c_stride;
    uint64_t uram_base;
    SharedBuffer shared[2];
    volatile int status;
    volatile uint32_t start;
    volatile uint32_t abort;
};

template <bool ProducerLoadsA>
static inline int rows_for_tile(const MatmulState *state, int tile_m)
{
    using Layout = UramLayout<ProducerLoadsA, MATMUL_BF16_1C_SLICES>;
    const int row_base = tile_m * Layout::kTileM;

    if (state->m <= row_base)
        return 0;
    return min_int(state->m - row_base, Layout::kTileM);
}

template <bool ProducerLoadsA>
static inline int cols_for_tile(const MatmulState *state, int tile_n)
{
    using Layout = UramLayout<ProducerLoadsA, MATMUL_BF16_1C_SLICES>;
    const int col_base = tile_n * Layout::kTileN;

    if (state->n <= col_base)
        return 0;
    return min_int(state->n - col_base, Layout::kTileN);
}

template <bool ProducerLoadsA>
static inline int rows_for_slice(const MatmulState *state, int tile_m,
                                 int slice)
{
    using Layout = UramLayout<ProducerLoadsA, MATMUL_BF16_1C_SLICES>;

    if (ProducerLoadsA)
        return rows_for_tile<ProducerLoadsA>(state, tile_m);
    const int row_base = tile_m * Layout::kTileM +
                         Layout::slice_offset(slice);

    if (state->m <= row_base)
        return 0;
    return min_int(state->m - row_base, Layout::slice_size(slice));
}

template <bool ProducerLoadsA>
static inline int cols_for_slice(const MatmulState *state, int tile_n,
                                 int slice)
{
    using Layout = UramLayout<ProducerLoadsA, MATMUL_BF16_1C_SLICES>;

    if (!ProducerLoadsA)
        return cols_for_tile<ProducerLoadsA>(state, tile_n);
    const int col_base = tile_n * Layout::kTileN +
                         Layout::slice_offset(slice);

    if (state->n <= col_base)
        return 0;
    return min_int(state->n - col_base, Layout::slice_size(slice));
}

static inline int k_for_tile(const MatmulState *state, int tile_k)
{
    const int k_base = tile_k * Constants::kTileK;

    if (state->k <= k_base)
        return 0;
    return min_int(state->k - k_base, Constants::kTileK);
}

static inline int remember_error(MatmulState *state, int status)
{
    if (status < 0 && state->status == 0)
        state->status = status;
    return status;
}

/* major is the contiguous byte count in a row; minor is the row count. */
template <long TcsrId>
static inline void load_matrix(uint64_t major_left, uint64_t minor_left,
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
        __tx_lsu_dma_d2u<TcsrId>((long)uram_base, (long)dram_base);
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
        __tx_lsu_dma_d2u<TcsrId>((long)uram_base, (long)dram_base);
        uram_base += handled_bytes;
        dram_base += handled_bytes;
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

static inline int wait_for_generation(SharedBuffer *buffer,
                                      uint64_t expected_generation)
{
    for (;;) {
        if (buffer->generation == expected_generation)
            return dayan_resource_wait_ready(buffer->resource);

        /*
         * A READY buffer with another generation still has slower users.
         * Wait until they release it, then let the producer publish the next
         * generation.  The producer writes generation immediately after
         * allocating the FREE resource, before issuing its DMA.
         */
        dayan_yield();
    }
}

template <bool ProducerLoadsA, long GroupBase, int Slice>
static void consume_slice(void *argument)
{
    using Layout = UramLayout<ProducerLoadsA, MATMUL_BF16_1C_SLICES>;

    MatmulState *state = (MatmulState *)argument;
    constexpr int slice_offset = Layout::slice_offset(Slice);
    constexpr int max_rows = Layout::max_rows(Slice);
    constexpr int max_cols = Layout::max_cols(Slice);
    constexpr int local_rows = Layout::slice_size(Slice);
    constexpr uint64_t c_row_bytes = max_cols * sizeof(float);
    constexpr uint64_t c_swizzle =
        c_row_bytes == 1024 ? kStride1024Swizzle : kStride512Swizzle;
    const uint64_t local_uram =
        state->uram_base + Layout::local_base(Slice);
    const uint64_t c_uram = state->uram_base + Layout::c_base(Slice);

    static_assert(c_row_bytes == 512 || c_row_bytes == 1024 ||
                      c_row_bytes == 1536,
                  "unsupported C row stride for swizzle");

    while (!state->start)
        dayan_yield();
    if (state->abort)
        return;

    long checksum = 0;
    set_lsu_matrix<GroupBase + Constants::kLsuLocal>(
        checksum, local_rows,
        Constants::kTileK * (int)sizeof(uint16_t),
        (long)(ProducerLoadsA ? state->bt_stride : state->a_stride),
        Constants::kTileK * (int)sizeof(uint16_t));
    set_lsu_matrix<GroupBase + Constants::kLsuCStore>(
        checksum, max_rows, max_cols * (int)sizeof(float),
        max_cols * (int)sizeof(float), (long)state->c_stride);
    __tx_lsu_tcsr_write<
        (GroupBase + Constants::kLsuLocal) * LSU_GROUPSZ +
        LSU_URAMWRITESWIZZLEMBS>(kStride512Swizzle);
    __tx_lsu_tcsr_write<
        (GroupBase + Constants::kLsuCStore) * LSU_GROUPSZ +
        LSU_URAMREADSWIZZLEMBS>(c_swizzle);

    __tx_mat_tcsr_write<GroupBase * MAT_GROUPSZ + MAT_M>(max_rows - 1);
    __tx_mat_tcsr_write<GroupBase * MAT_GROUPSZ + MAT_N>(max_cols - 1);
    __tx_mat_tcsr_write<GroupBase * MAT_GROUPSZ + MAT_K>(
        Constants::kTileK - 1);
    __tx_mat_tcsr_write<GroupBase * MAT_GROUPSZ + MAT_ASTRIDE>(
        Constants::kTileK * sizeof(uint16_t));
    __tx_mat_tcsr_write<GroupBase * MAT_GROUPSZ + MAT_BSTRIDE>(
        Constants::kTileK * sizeof(uint16_t));
    __tx_mat_tcsr_write<GroupBase * MAT_GROUPSZ + MAT_CSTRIDE>(
        max_cols * sizeof(float));
    __tx_mat_tcsr_write<GroupBase * MAT_GROUPSZ + MAT_ASWIZZLEMBS>(
        kStride512Swizzle);
    __tx_mat_tcsr_write<GroupBase * MAT_GROUPSZ + MAT_BSWIZZLEMBS>(
        kStride512Swizzle);
    __tx_mat_tcsr_write<GroupBase * MAT_GROUPSZ + MAT_CSWIZZLEMBS>(c_swizzle);
    __tx_mat_tcsr_write<GroupBase * MAT_GROUPSZ + MAT_PERSISTENTMODE>(0);
    __tx_mat_tcsr_write<GroupBase * MAT_GROUPSZ + MAT_DATATYPE>(
        BaseNPUInstBuilder::DT_BF16);

    for (int tile_m = 0; tile_m < state->count_m; ++tile_m) {
        const int real_rows =
            rows_for_slice<ProducerLoadsA>(state, tile_m, Slice);

        if (real_rows == 0)
            continue;
        for (int tile_n = 0; tile_n < state->count_n; ++tile_n) {
            const int real_cols =
                cols_for_slice<ProducerLoadsA>(state, tile_n, Slice);

            if (real_cols == 0)
                continue;

            for (int tile_k = 0; tile_k < state->count_k; ++tile_k) {
                const uint64_t generation =
                    ((uint64_t)tile_m * state->count_n + tile_n) *
                        state->count_k +
                    tile_k;
                SharedBuffer *shared = &state->shared[generation & 1U];
                const int real_k = k_for_tile(state, tile_k);
                const bool first_k = tile_k == 0;
                const bool last_k = tile_k + 1 == state->count_k;
                const uint64_t local_dram =
                    (ProducerLoadsA ? state->bt_base : state->a_base) +
                    (uint64_t)tile_k * Constants::kTileK *
                        sizeof(uint16_t) +
                    (ProducerLoadsA
                         ? ((uint64_t)tile_n * Layout::kTileN +
                            slice_offset) * state->bt_stride
                         : ((uint64_t)tile_m * Layout::kTileM +
                            slice_offset) * state->a_stride);

                load_matrix<GroupBase + Constants::kLsuLocal>(
                    (uint64_t)real_k * sizeof(uint16_t),
                    ProducerLoadsA ? real_cols : real_rows,
                    Constants::kTileK * sizeof(uint16_t), local_rows,
                    local_uram, local_dram);
                dayan_wait_lsu();

                int status = wait_for_generation(shared, generation);
                if (status < 0) {
                    remember_error(state, status);
                    return;
                }

                __tx_mat_tcsr_write<GroupBase * MAT_GROUPSZ + MAT_M>(
                    real_rows - 1);
                __tx_mat_tcsr_write<GroupBase * MAT_GROUPSZ + MAT_N>(
                    real_cols - 1);
                __tx_mat_tcsr_write<GroupBase * MAT_GROUPSZ + MAT_K>(
                    real_k - 1);
                if (ProducerLoadsA) {
                    if (first_k)
                        __tx_mat_mm<GroupBase>((long)c_uram,
                                               (long)shared->uram,
                                               (long)local_uram);
                    else
                        __tx_mat_mma<GroupBase>((long)c_uram,
                                                (long)shared->uram,
                                                (long)local_uram);
                } else {
                    if (first_k)
                        __tx_mat_mm<GroupBase>((long)c_uram,
                                               (long)local_uram,
                                               (long)shared->uram);
                    else
                        __tx_mat_mma<GroupBase>((long)c_uram,
                                                (long)local_uram,
                                                (long)shared->uram);
                }
                dayan_wait_mat();

                status = dayan_resource_consume_mat(
                    DAYAN_RESOURCE_MASK(shared->resource));
                if (status < 0) {
                    remember_error(state, status);
                    return;
                }

                if (last_k) {
                    const uint64_t c_dram =
                        state->c_base +
                        ((uint64_t)tile_n * Layout::kTileN +
                         (ProducerLoadsA ? slice_offset : 0)) *
                            sizeof(float) +
                        ((uint64_t)tile_m * Layout::kTileM +
                         (ProducerLoadsA ? 0 : slice_offset)) *
                            state->c_stride;

                    store_matrix<GroupBase + Constants::kLsuCStore>(
                        (uint64_t)real_cols * sizeof(float), real_rows,
                        max_cols * sizeof(float), max_rows,
                        c_uram, c_dram);
                    /* The next local-input wait also retires this C store. */
                } else {
                    dayan_yield();
                }
            }
        }
    }
    dayan_wait_lsu();
    dayan_wait_mat();
}

template <bool ProducerLoadsA>
static void producer_thread(void *argument)
{
    using Layout = UramLayout<ProducerLoadsA, MATMUL_BF16_1C_SLICES>;

    MatmulState *state = (MatmulState *)argument;

    while (!state->start)
        dayan_yield();
    if (state->abort)
        return;

    long checksum = 0;
    set_lsu_matrix<Constants::kLsuShared>(
        checksum,
        ProducerLoadsA ? Layout::kTileM : Layout::kTileN,
        Constants::kTileK * (int)sizeof(uint16_t),
        (long)(ProducerLoadsA ? state->a_stride : state->bt_stride),
        Constants::kTileK * (int)sizeof(uint16_t));
    __tx_lsu_tcsr_write<
        Constants::kLsuShared * LSU_GROUPSZ + LSU_URAMWRITESWIZZLEMBS>(
        kStride512Swizzle);

    for (int tile_m = 0; tile_m < state->count_m; ++tile_m) {
        for (int tile_n = 0; tile_n < state->count_n; ++tile_n) {
            int users = 0;

            for (int slice = 0; slice < MATMUL_BF16_1C_SLICES; ++slice) {
                if (rows_for_slice<ProducerLoadsA>(
                        state, tile_m, slice) > 0 &&
                    cols_for_slice<ProducerLoadsA>(
                        state, tile_n, slice) > 0)
                    ++users;
            }
            if (!users)
                continue;

            for (int tile_k = 0; tile_k < state->count_k; ++tile_k) {
                const uint64_t generation =
                    ((uint64_t)tile_m * state->count_n + tile_n) *
                        state->count_k +
                    tile_k;
                SharedBuffer *shared = &state->shared[generation & 1U];
                const int real_k = k_for_tile(state, tile_k);
                const uint64_t shared_dram =
                    (ProducerLoadsA ? state->a_base : state->bt_base) +
                    (uint64_t)tile_k * Constants::kTileK *
                        sizeof(uint16_t) +
                    (ProducerLoadsA
                         ? (uint64_t)tile_m * Layout::kTileM *
                               state->a_stride
                         : (uint64_t)tile_n * Layout::kTileN *
                               state->bt_stride);
                int status = dayan_resource_wait_free(shared->resource);

                if (status < 0) {
                    remember_error(state, status);
                    return;
                }
                shared->generation = generation;
                load_matrix<Constants::kLsuShared>(
                    (uint64_t)real_k * sizeof(uint16_t),
                    ProducerLoadsA
                        ? rows_for_tile<ProducerLoadsA>(state, tile_m)
                        : cols_for_tile<ProducerLoadsA>(state, tile_n),
                    Constants::kTileK * sizeof(uint16_t),
                    ProducerLoadsA ? Layout::kTileM : Layout::kTileN,
                    shared->uram, shared_dram);
                status = dayan_resource_produce_lsu(
                    shared->resource, (uint32_t)users, 0);
                if (status < 0) {
                    remember_error(state, status);
                    return;
                }
                dayan_yield();
            }
        }
    }
}

template <bool ProducerLoadsA>
static int run_pipeline(const MatmulBf16Config *config,
                        const uint16_t *a, const uint16_t *bt, float *c,
                        uint64_t uram_base, uint64_t *pipeline_begin)
{
    using Layout = UramLayout<ProducerLoadsA, MATMUL_BF16_1C_SLICES>;

    MatmulState state;
    state.m = (int)config->m;
    state.n = (int)config->n;
    state.k = (int)config->k;
    state.count_m = (state.m + Layout::kTileM - 1) / Layout::kTileM;
    state.count_n = (state.n + Layout::kTileN - 1) / Layout::kTileN;
    state.count_k = (state.k + Constants::kTileK - 1) /
                    Constants::kTileK;
    state.a_base = (uint64_t)(uintptr_t)a;
    state.bt_base = (uint64_t)(uintptr_t)bt;
    state.c_base = (uint64_t)(uintptr_t)c;
    state.a_stride = config->a_stride;
    state.bt_stride = config->bt_stride;
    state.c_stride = config->c_stride;
    state.uram_base = uram_base;
    state.status = 0;
    state.start = 0;
    state.abort = 0;

    for (int index = 0; index < 2; ++index) {
        state.shared[index].resource = DAYAN_LOCAL_RESOURCE(index);
        state.shared[index].generation = ~0ULL;
        state.shared[index].uram =
            state.uram_base + Layout::shared_base(index);
    }

    dayan_thread_resources_t resources;
    dayan_thread_resources_init(&resources);
    resources.local_resource_count = 2;
    resources.lsu_tcsr_count = 1;
    int status = dayan_create_thread("producer",
                                     producer_thread<ProducerLoadsA>, &state,
                                     &resources);
    resources.lsu_tcsr_count = 2;
    resources.mat_tcsr_count = 1;
    if (status >= 0)
        status = dayan_create_thread("consumer-0",
                                     consume_slice<ProducerLoadsA, 0, 0>,
                                     &state,
                                     &resources);
#if MATMUL_BF16_1C_SLICES == 3
    if (status >= 0)
        status = dayan_create_thread("consumer-1",
                                     consume_slice<ProducerLoadsA, 0, 1>,
                                     &state,
                                     &resources);
    if (status >= 0)
        status = dayan_create_thread("consumer-2",
                                     consume_slice<ProducerLoadsA, 0, 2>,
                                     &state,
                                     &resources);
#endif
    if (status < 0) {
        state.status = status;
        state.abort = 1;
        state.start = 1;
        (void)dayan_thread_join_all();
        return status;
    }

    *pipeline_begin = read_cycle();
    state.start = 1;
    status = dayan_thread_join_all();
    if (status < 0 && state.status == 0)
        state.status = status;
    return state.status;
}

static inline int config_valid(const MatmulBf16Config *config)
{
    const uint64_t int_limit = 0x7ffff000ULL;

    if (!config || !config->m || !config->n || !config->k ||
        config->m > int_limit || config->n > int_limit ||
        config->k > int_limit ||
        (config->n % 16U) != 0 ||
        config->a_stride < config->k * sizeof(uint16_t) ||
        config->bt_stride < config->k * sizeof(uint16_t) ||
        config->c_stride < config->n * sizeof(float) ||
        (config->c_stride & 63U) ||
        config->a_stride > 0x7fffffffffffffffULL ||
        config->bt_stride > 0x7fffffffffffffffULL ||
        config->c_stride > 0x7fffffffffffffffULL)
        return 0;
    return 1;
}

} // namespace matmul_bf16_1c_detail

#endif /* __DAYAN_DEVICE__ */

__kernel__ void matmul_bf16_1c(const MatmulBf16Config *config,
                               const uint16_t *a, const uint16_t *bt,
                               float *c, MatmulBf16Perf *perf)
{
    using namespace matmul_bf16_1c_detail;

    const uint64_t kernel_begin = read_cycle();

    if (!config_valid(config) || !a || !bt || !c || !perf ||
        dayan_task_mode() != 1U ||
        dayan_task_dim(DAYAN_TASK_X) != 1U ||
        dayan_task_dim(DAYAN_TASK_Y) != 1U ||
        dayan_task_dim(DAYAN_TASK_Z) != 1U) {
        dayan_thread_exit(-22);
    }

    void *uram = dayan_uram_base();
    const bool producer_loads_a = config->m < config->n;
    const uint64_t required_uram =
        producer_loads_a ? AProducerLayout::total_bytes()
                         : BProducerLayout::total_bytes();
    if (!uram || dayan_uram_size() < required_uram)
        dayan_thread_exit(-12);

    const uint64_t uram_base = (uint64_t)(uintptr_t)uram &
                               (Constants::kUramWindowStride - 1U);
    uint64_t pipeline_begin = 0;
    const int status = producer_loads_a
        ? run_pipeline<true>(config, a, bt, c, uram_base, &pipeline_begin)
        : run_pipeline<false>(config, a, bt, c, uram_base, &pipeline_begin);
    if (status < 0)
        dayan_thread_exit(status);

    const uint64_t pipeline_end = read_cycle();
    const uint64_t kernel_end = read_cycle();

    perf->magic = kMatmulBf16PerfMagic;
    perf->m = config->m;
    perf->n = config->n;
    perf->k = config->k;
    perf->kernel_cycles = kernel_end - kernel_begin;
    perf->pipeline_cycles = pipeline_end - pipeline_begin;
}

DAYAN_KERNEL_RESOURCE_DESC_EX(
    matmul_bf16_1c, 1U, MATMUL_BF16_1C_SLICES + 1,
    DAYAN_KERNEL_RESOURCE_LOCKED_TASK, 0, MATMUL_BF16_1C_URAM_BYTES,
    2 * MATMUL_BF16_1C_SLICES + 1, MATMUL_BF16_1C_SLICES, 0, 0, 2, 0);

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
    std::printf("  defaults: M=768 N=512 K=1536\n");
    std::printf("  N must be a multiple of 16.\n");
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
    if (rows > SIZE_MAX / columns)
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
            (std::fabs(reference[index]) > 1.0e-6f
                 ? std::fabs(reference[index])
                 : 1.0f);

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
    uint64_t m = 768;
    uint64_t n = 512;
    uint64_t k = 1536;
    size_t a_elements = 0;
    size_t bt_elements = 0;
    size_t c_elements = 0;
    size_t a_bytes = 0;
    size_t bt_bytes = 0;
    size_t c_bytes = 0;
    /* Full-size RTL simulation can take several wall minutes. */
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
    MatmulBf16Perf perf = {};
    npu_runtime_options_t runtime_options;
    npu_kernel_launch_attrs_t launch_attrs;
    npu_runtime_t *runtime = nullptr;
    npu_context_t *context = nullptr;
    npu_stream_t *stream = nullptr;
    npu_buffer_t *config_buffer = nullptr;
    npu_buffer_t *a_buffer = nullptr;
    npu_buffer_t *bt_buffer = nullptr;
    npu_buffer_t *c_buffer = nullptr;
    npu_buffer_t *perf_buffer = nullptr;
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
        status = create_buffer(context, sizeof(perf), &perf_buffer);
    if (status == 0)
        status = upload(stream, config_buffer, &config, sizeof(config),
                        timeout_ms);
    if (status == 0)
        status = upload(stream, a_buffer, a.data(), a_bytes, timeout_ms);
    if (status == 0)
        status = upload(stream, bt_buffer, bt.data(), bt_bytes, timeout_ms);
    if (status == 0)
        status = upload(stream, perf_buffer, &perf, sizeof(perf), timeout_ms);

    if (status == 0) {
        const uint64_t tile_m = m < n ? kMatmulBf16TileShort
                                      : kMatmulBf16TileLong;
        const uint64_t tile_n = m < n ? kMatmulBf16TileLong
                                      : kMatmulBf16TileShort;

        npu_kernel_launch_attrs_init(&launch_attrs, context, stream,
                                     "matmul-bf16-1c");
        launch_attrs.config.task_dim = NPUTaskDim3{1, 1, 1};
        launch_attrs.config.task_mode = NPUTaskMode::kSingleCore;
        launch_attrs.timeout_ms = timeout_ms;
        std::printf("[host] launch matmul_bf16_1c: A[%llu,%llu] x "
                    "BT[%llu,%llu] -> C[%llu,%llu], slices=%u, "
                    "tile=%llux%llux%llu, producer=%s\n",
                    (unsigned long long)m, (unsigned long long)k,
                    (unsigned long long)n, (unsigned long long)k,
                    (unsigned long long)m, (unsigned long long)n,
                    (unsigned)MATMUL_BF16_1C_SLICES,
                    (unsigned long long)tile_m,
                    (unsigned long long)tile_n,
                    (unsigned long long)kMatmulBf16TileK,
                    m < n ? "A" : "B");
        status = matmul_bf16_1c(config_buffer, a_buffer, bt_buffer,
                                c_buffer, perf_buffer, &launch_attrs);
    }
    if (status == 0)
        status = download(stream, &perf, perf_buffer, sizeof(perf),
                          timeout_ms);
    if (status == 0 &&
        (perf.magic != kMatmulBf16PerfMagic || perf.m != m ||
         perf.n != n || perf.k != k || !perf.kernel_cycles ||
         !perf.pipeline_cycles || perf.pipeline_cycles > perf.kernel_cycles))
        status = -1;
    if (status == 0) {
        const double flops = 2.0 * (double)m * (double)n * (double)k;

        std::printf("[host] perf: kernel_cycles=%llu "
                    "pipeline_cycles=%llu kernel_flops_per_cycle=%.3f "
                    "pipeline_flops_per_cycle=%.3f\n",
                    (unsigned long long)perf.kernel_cycles,
                    (unsigned long long)perf.pipeline_cycles,
                    flops / (double)perf.kernel_cycles,
                    flops / (double)perf.pipeline_cycles);
    }
    if (status == 0)
        status = download(stream, c.data(), c_buffer, c_bytes, timeout_ms);
    if (status == 0)
        status = validate_output(c, reference, (size_t)n);
    if (status < 0)
        std::printf("matmul_bf16_1c demo failed: %d\n", status);

    if (perf_buffer)
        preserve_error(npu_buffer_destroy(perf_buffer), &status);
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
