#include <dayan_kernel.h>
#include <stdint.h>

#ifndef MATMUL_BF16_4C_SLICES
#define MATMUL_BF16_4C_SLICES 4
#endif

#ifndef MATMUL_BF16_4C_URAM_BYTES
#define MATMUL_BF16_4C_URAM_BYTES (1536U * 1024U)
#endif

#ifndef MATMUL_BF16_4C_ATOMIC_BYTES
#define MATMUL_BF16_4C_ATOMIC_BYTES 2048U
#endif

#ifndef MATMUL_BF16_4C_D2A_MODEL
#define MATMUL_BF16_4C_D2A_MODEL 1
#endif

#ifndef MATMUL_BF16_4C_BROADCAST_MODEL
#define MATMUL_BF16_4C_BROADCAST_MODEL 1
#endif

#if MATMUL_BF16_4C_BROADCAST_MODEL
#define MATMUL_BF16_4C_WORKERS 3U
#define MATMUL_BF16_4C_KERNEL matmul_bf16_broadcast_4c
#define MATMUL_BF16_4C_KERNEL_LABEL "mm4c-broadcast"
#define MATMUL_BF16_4C_KERNEL_DISPLAY "matmul_bf16_broadcast_4c"
#elif MATMUL_BF16_4C_D2A_MODEL
#define MATMUL_BF16_4C_WORKERS 4U
#define MATMUL_BF16_4C_KERNEL matmul_bf16_4c_d2a_model
#define MATMUL_BF16_4C_KERNEL_LABEL "mm4c-d2a"
#define MATMUL_BF16_4C_KERNEL_DISPLAY "matmul_bf16_4c_d2a_model"
#else
#define MATMUL_BF16_4C_WORKERS 4U
#define MATMUL_BF16_4C_KERNEL matmul_bf16_4c
#define MATMUL_BF16_4C_KERNEL_LABEL "matmul-bf16-4c"
#define MATMUL_BF16_4C_KERNEL_DISPLAY "matmul_bf16_4c"
#endif

#ifndef MATMUL_BF16_4C_TILE_M
#define MATMUL_BF16_4C_TILE_M 512
#endif

#ifndef MATMUL_BF16_4C_TILE_N
#define MATMUL_BF16_4C_TILE_N 384
#endif

#ifndef MATMUL_BF16_4C_TILE_K
#define MATMUL_BF16_4C_TILE_K 256
#endif

/*
 * B is supplied in transposed storage: A[M][K] * BT[N][K] -> C[M][N].
 * Strides are measured in bytes.  The four cluster lanes form this grid:
 *
 *                 N-even tile       N-odd tile
 *   lower M half      core 00           core 01
 *   upper M half      core 10           core 11
 *
 * A is sourced from DRAM by cores 00/11 and shared horizontally.  B is
 * sourced from DRAM by cores 01/10 and shared vertically.  Every Core runs
 * one DRAM Loader, one remote Copy worker, and two ordered MAT/Store workers.
 * The DRAM and Copy streams advance monotonically; no worker claims work from
 * a shared iterator.  Cross-Core readiness is expressed only with hardware
 * Shared Named Resources, while all same-Core dependencies use Local Named
 * Resources.  The two MAT workers statically alternate linear jobs and pass
 * a resource token after ordered issue, so MAT instructions cannot overtake.
 *
 * MATMUL_BF16_4C_D2A_MODEL treats the source D2U as a future D2A while
 * retaining the Copy worker as a shared-to-local resource adapter.
 * MATMUL_BF16_4C_BROADCAST_MODEL removes that adapter: remote MAT workers
 * wait on and consume the source Loader's Shared Named Resource directly.
 * Both
 * models assume that the source D2U also fills remote URAM, so Host numerical
 * validation is intentionally disabled.
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

#include <dayan_intrinsics.h>
#include <dayan_sys.h>

namespace matmul_bf16_4c_detail {

struct Constants {
    enum : int {
        kDramLoad = 0,
        kRemoteCopy = 0,
        kLsuCStore = 0,
        kTileM = MATMUL_BF16_4C_TILE_M,
        kTileN = MATMUL_BF16_4C_TILE_N,
        kTileK = MATMUL_BF16_4C_TILE_K,
        kClusterRows = 2,
        kClusterColumns = 2,
        kClusterCores = 4,
        kABufferCount = 1,
        kUramAlignment = 32,
        kUramWindowBase = 0x00800000,
        kUramWindowStride = 2 * 1024 * 1024,
    };
};

static_assert(MATMUL_BF16_4C_SLICES == 1 ||
                  MATMUL_BF16_4C_SLICES == 3 ||
                  MATMUL_BF16_4C_SLICES == 4,
              "MATMUL_BF16_4C_SLICES must be 1, 3, or 4");
static_assert(Constants::kTileM == 512 && Constants::kTileN == 384,
              "4-Core matmul uses a 512x384 output tile per Core");
static_assert(Constants::kTileK == 256,
              "RTL matmul reaches peak throughput with tile K fixed at 256");
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
        return Constants::kTileM * slice / SliceCount;
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

    static constexpr uint64_t a_buffer_bytes()
    {
        uint64_t base = 0;

        for (int index = 0; index < SliceCount; ++index)
            base = align_up(base + a_bytes(index),
                            Constants::kUramAlignment);
        return base;
    }

    static constexpr uint64_t a_base(int slice, int buffer)
    {
        uint64_t base = (uint64_t)buffer * a_buffer_bytes();

        for (int index = 0; index < slice; ++index)
            base = align_up(base + a_bytes(index),
                            Constants::kUramAlignment);
        return base;
    }

    static constexpr uint64_t a_end()
    {
        return Constants::kABufferCount * a_buffer_bytes();
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

static_assert(MATMUL_BF16_4C_SLICES == 1 ||
                  (MATMUL_BF16_4C_SLICES == 3 &&
                   Layout::slice_rows(0) == 170 &&
                   Layout::slice_rows(1) == 171 &&
                   Layout::slice_rows(2) == 171) ||
                  (MATMUL_BF16_4C_SLICES == 4 &&
                   Layout::slice_rows(0) == 128 &&
                   Layout::slice_rows(1) == 128 &&
                   Layout::slice_rows(2) == 128 &&
                   Layout::slice_rows(3) == 128),
              "slice mode must cover the 512-row tile exactly");
static_assert(Layout::total_bytes() == 1441792U,
              "512x384x256 single-A/double-B layout must use 1.375 MiB");
static_assert(Layout::total_bytes() <= MATMUL_BF16_4C_URAM_BYTES,
              "output-stationary matmul URAM layout is too large");

struct ClusterShared {
    dayan_cluster_barrier_t barrier;
    int32_t init_status;
    uint32_t reserved;
    uint64_t uram_base[Constants::kClusterCores];
};

static_assert(sizeof(ClusterShared) <= MATMUL_BF16_4C_ATOMIC_BYTES,
              "cluster matmul shared state exceeds its atomic-cache request");

struct LocalBuffer {
    dayan_resource_t resource;
    uint64_t uram;
};

struct MatConfig {
    int rows;
    int columns;
    int store_rows;
    int store_columns;
};

struct MatmulState {
    int m;
    int n;
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
    LocalBuffer a[MATMUL_BF16_4C_SLICES][Constants::kABufferCount];
    LocalBuffer b[2];
    dayan_resource_t mat_turn[2];
    uint64_t generation_count;
    MatConfig mat_config[MATMUL_BF16_4C_SLICES];
};

struct MatJob {
    uint64_t generation;
    int tile_m;
    int tile_n;
    int tile_k;
    int slice;
};

struct MatWorkerArgs {
    MatmulState *state;
    uint32_t worker;
};

static inline dayan_resource_t shared_a_resource(uint32_t row, int slice,
                                                 int buffer)
{
    return DAYAN_SHARED_RESOURCE(
        (row * MATMUL_BF16_4C_SLICES + (uint32_t)slice) *
            Constants::kABufferCount +
        (uint32_t)buffer);
}

static inline dayan_resource_t shared_b_resource(uint32_t column, int buffer)
{
    return DAYAN_SHARED_RESOURCE(
        Constants::kClusterRows * MATMUL_BF16_4C_SLICES *
            Constants::kABufferCount +
        column * 2U + (uint32_t)buffer);
}

static inline uint32_t a_source_lane(uint32_t row_lane)
{
    return row_lane == 0U ? 0U : 3U;
}

#if !MATMUL_BF16_4C_D2A_MODEL || MATMUL_BF16_4C_BROADCAST_MODEL
static inline uint32_t b_source_lane(uint32_t column_lane)
{
    return column_lane == 0U ? 2U : 1U;
}
#endif

static inline int row_partition_begin(const MatmulState *state,
                                      uint32_t row_lane)
{
    return (int)((uint64_t)state->m * row_lane /
                 Constants::kClusterRows);
}

static inline int row_partition_end(const MatmulState *state,
                                    uint32_t row_lane)
{
    return (int)((uint64_t)state->m * (row_lane + 1U) /
                 Constants::kClusterRows);
}

static inline int rows_for_slice(const MatmulState *state, int tile_m,
                                 uint32_t row_lane, int slice)
{
    const int row_base = row_partition_begin(state, row_lane) +
                         tile_m * Constants::kTileM +
                         Layout::slice_offset(slice);
    const int row_end = row_partition_end(state, row_lane);

    if (row_end <= row_base)
        return 0;
    return min_int(row_end - row_base, Layout::slice_rows(slice));
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

static inline void decode_generation(const MatmulState *state,
                                     uint64_t generation, int *tile_m,
                                     int *tile_n, int *tile_k)
{
    const uint64_t outer = generation / (uint64_t)state->count_k;

    *tile_k = (int)(generation % (uint64_t)state->count_k);
    *tile_n = (int)(outer % (uint64_t)state->count_n);
    *tile_m = (int)(outer / (uint64_t)state->count_n);
}

static inline void require_status(int status)
{
    if (status < 0)
        dayan_thread_exit(status);
}

#if !MATMUL_BF16_4C_BROADCAST_MODEL
static inline bool generation_has_local_work(const MatmulState *state,
                                             int tile_m, int tile_n)
{
    return active_slices(state, tile_m, state->row_lane) > 0 &&
           cols_for_tile(state, tile_n, state->column_lane) > 0;
}
#endif

template <long LoadGroup>
static inline int configure_loader(const MatmulState *state, bool loads_a,
                                   bool from_dram)
{
    long checksum = 0;
    constexpr int row_bytes =
        Constants::kTileK * (int)sizeof(uint16_t);
    constexpr uint64_t swizzle = MAT_BATCH_SWIZZLE(5, 3, 5);
    const int rows = loads_a ? Layout::slice_rows(0) :
                               Constants::kTileN;
    const long input_stride = from_dram ?
        (loads_a ? (long)state->a_stride : (long)state->bt_stride) :
        row_bytes;

    set_lsu_matrix<LoadGroup>(checksum, rows, row_bytes, input_stride,
                              row_bytes);
    __tx_lsu_tcsr_write<
        LoadGroup * LSU_GROUPSZ + LSU_URAMREADSWIZZLEMBS>(0);
    __tx_lsu_tcsr_write<
        LoadGroup * LSU_GROUPSZ + LSU_URAMWRITESWIZZLEMBS>(
            from_dram ? swizzle : 0);
    return rows;
}

template <long LoadGroup>
static inline void configure_loader_rows(int rows, int *configured_rows)
{
    if (rows == *configured_rows)
        return;
    __tx_lsu_tcsr_write<LoadGroup * LSU_GROUPSZ + LSU_DIM1>(rows - 1);
    *configured_rows = rows;
}

static inline void begin_local_resource(dayan_resource_t resource)
{
    require_status(dayan_resource_wait_free(resource));
}

static inline void begin_shared_resource(dayan_resource_t resource)
{
    require_status(dayan_resource_wait_free(resource));
}

template <long LoadGroup>
static void dram_load_a(MatmulState *state, uint64_t generation,
                        int tile_m, int tile_n, int tile_k,
                        int *configured_rows)
{
    (void)generation;
    const int columns = cols_for_tile(
        state, tile_n, state->column_lane);
    const int peer_columns = cols_for_tile(
        state, tile_n, state->column_lane ^ 1U);
    constexpr int a_buffer_index = 0;
    for (int slice = 0; slice < MATMUL_BF16_4C_SLICES; ++slice) {
        const int real_rows = rows_for_slice(
            state, tile_m, state->row_lane, slice);
        const bool local_active = real_rows > 0 && columns > 0;
        const bool peer_active = real_rows > 0 && peer_columns > 0;
        LocalBuffer *local = &state->a[slice][a_buffer_index];
        const dayan_resource_t shared_resource = shared_a_resource(
            state->row_lane, slice, a_buffer_index);

        if (!local_active && !peer_active)
            continue;
        if (local_active)
            begin_local_resource(local->resource);
        if (peer_active)
            begin_shared_resource(shared_resource);
        configure_loader_rows<LoadGroup>(real_rows, configured_rows);

        const uint64_t global_row =
            (uint64_t)row_partition_begin(state, state->row_lane) +
            (uint64_t)tile_m * Constants::kTileM +
            (uint64_t)Layout::slice_offset(slice);
        const uint64_t a_dram =
            state->a_base + global_row * state->a_stride +
            (uint64_t)tile_k * Constants::kTileK * sizeof(uint16_t);

        __tx_lsu_dma_d2u<LoadGroup>((long)local->uram, (long)a_dram);
        dayan_wait_lsu();
        if (peer_active)
            require_status(dayan_resource_produce_lsu(
                shared_resource, 1U, 0));
        if (local_active)
            require_status(dayan_resource_produce_lsu(
                local->resource, 1U, 0));
    }
}

template <long LoadGroup>
static void dram_load_b(MatmulState *state, uint64_t generation,
                        int tile_m, int tile_n, int tile_k,
                        int *configured_rows)
{
    const int real_columns = cols_for_tile(
        state, tile_n, state->column_lane);
    const int local_users = active_slices(
        state, tile_m, state->row_lane);
    const int peer_users = active_slices(
        state, tile_m, state->row_lane ^ 1U);
    const int buffer_index = (int)(generation & 1U);
    LocalBuffer *local = &state->b[buffer_index];
    const dayan_resource_t shared_resource = shared_b_resource(
        state->column_lane, buffer_index);

    if (!real_columns || (!local_users && !peer_users))
        return;
    if (local_users)
        begin_local_resource(local->resource);
    if (peer_users)
        begin_shared_resource(shared_resource);
    configure_loader_rows<LoadGroup>(real_columns, configured_rows);

    const uint64_t global_column =
        (uint64_t)(tile_n * Constants::kClusterColumns +
                   (int)state->column_lane) * Constants::kTileN;
    const uint64_t bt_dram =
        state->bt_base + global_column * state->bt_stride +
        (uint64_t)tile_k * Constants::kTileK * sizeof(uint16_t);

    __tx_lsu_dma_d2u<LoadGroup>((long)local->uram, (long)bt_dram);
    dayan_wait_lsu();
    if (peer_users)
        require_status(dayan_resource_produce_lsu(
            shared_resource,
            MATMUL_BF16_4C_BROADCAST_MODEL ? (uint32_t)peer_users : 1U,
            0));
    if (local_users)
        require_status(dayan_resource_produce_lsu(
            local->resource, (uint32_t)local_users, 0));
}

static void dram_loader(void *argument)
{
    MatmulState *state = (MatmulState *)argument;
    const bool loads_a =
        state->lane == a_source_lane(state->row_lane);
    int configured_rows = configure_loader<Constants::kDramLoad>(
        state, loads_a, true);

    for (uint64_t generation = 0;
         generation < state->generation_count; ++generation) {
        int tile_m;
        int tile_n;
        int tile_k;

        decode_generation(state, generation, &tile_m, &tile_n, &tile_k);
        if (loads_a) {
            dram_load_a<Constants::kDramLoad>(
                state, generation, tile_m, tile_n, tile_k,
                &configured_rows);
        } else {
            dram_load_b<Constants::kDramLoad>(
                state, generation, tile_m, tile_n, tile_k,
                &configured_rows);
        }
    }
    dayan_wait_lsu();
}

#if !MATMUL_BF16_4C_BROADCAST_MODEL
template <long LoadGroup>
static void copy_a(MatmulState *state, uint64_t generation,
                   int tile_m, int tile_n, int *configured_rows,
                   uint64_t *phases)
{
    (void)generation;
#if !MATMUL_BF16_4C_D2A_MODEL
    const uint32_t source_lane = a_source_lane(state->row_lane);
#endif
    const int columns = cols_for_tile(
        state, tile_n, state->column_lane);
    constexpr int a_buffer_index = 0;

    for (int slice = 0; slice < MATMUL_BF16_4C_SLICES; ++slice) {
        const int real_rows = rows_for_slice(
            state, tile_m, state->row_lane, slice);

        if (!real_rows || !columns)
            continue;
        LocalBuffer *local = &state->a[slice][a_buffer_index];
        const dayan_resource_t shared_resource = shared_a_resource(
            state->row_lane, slice, a_buffer_index);

        require_status(dayan_resource_wait(
            DAYAN_RESOURCE_MASK(local->resource),
            DAYAN_RESOURCE_MASK(shared_resource), *phases, 0));
#if !MATMUL_BF16_4C_D2A_MODEL
        configure_loader_rows<LoadGroup>(real_rows, configured_rows);
        const uint64_t source_uram =
            state->cluster->uram_base[source_lane] +
            Layout::a_base(slice, a_buffer_index);

        __tx_lsu_dma_d2u<LoadGroup>((long)local->uram,
                                    (long)source_uram);
#else
        (void)configured_rows;
#endif
        require_status(dayan_resource_produce_lsu(
            local->resource, 1U,
            DAYAN_RESOURCE_MASK(shared_resource)));
        *phases ^= DAYAN_RESOURCE_MASK(shared_resource);
    }
}

template <long LoadGroup>
static void copy_b(MatmulState *state, uint64_t generation,
                   int tile_m, int tile_n, int *configured_rows,
                   uint64_t *phases)
{
#if !MATMUL_BF16_4C_D2A_MODEL
    const uint32_t source_lane = b_source_lane(state->column_lane);
#endif
    const int real_columns = cols_for_tile(
        state, tile_n, state->column_lane);
    const int local_users = active_slices(
        state, tile_m, state->row_lane);
    const int buffer_index = (int)(generation & 1U);

    if (!real_columns || !local_users)
        return;
    LocalBuffer *local = &state->b[buffer_index];
    const dayan_resource_t shared_resource = shared_b_resource(
        state->column_lane, buffer_index);

    require_status(dayan_resource_wait(
        DAYAN_RESOURCE_MASK(local->resource),
        DAYAN_RESOURCE_MASK(shared_resource), *phases, 0));
#if !MATMUL_BF16_4C_D2A_MODEL
    configure_loader_rows<LoadGroup>(real_columns, configured_rows);
    const uint64_t source_uram =
        state->cluster->uram_base[source_lane] +
        Layout::b_base(buffer_index);

    __tx_lsu_dma_d2u<LoadGroup>((long)local->uram, (long)source_uram);
#else
    (void)configured_rows;
#endif
    require_status(dayan_resource_produce_lsu(
        local->resource, (uint32_t)local_users,
        DAYAN_RESOURCE_MASK(shared_resource)));
    *phases ^= DAYAN_RESOURCE_MASK(shared_resource);
}

static void remote_copy(void *argument)
{
    MatmulState *state = (MatmulState *)argument;
    uint64_t phases = 0;
    const bool copies_a =
        state->lane != a_source_lane(state->row_lane);
#if MATMUL_BF16_4C_D2A_MODEL
    int configured_rows = 0;
#else
    int configured_rows = configure_loader<Constants::kRemoteCopy>(
        state, copies_a, false);
#endif

    for (uint64_t generation = 0;
         generation < state->generation_count; ++generation) {
        int tile_m;
        int tile_n;
        int tile_k;

        decode_generation(state, generation, &tile_m, &tile_n, &tile_k);
        (void)tile_k;
        if (!generation_has_local_work(state, tile_m, tile_n))
            continue;
        if (copies_a) {
            copy_a<Constants::kRemoteCopy>(
                state, generation, tile_m, tile_n, &configured_rows, &phases);
        } else {
            copy_b<Constants::kRemoteCopy>(
                state, generation, tile_m, tile_n, &configured_rows, &phases);
        }
    }
    dayan_wait_lsu();
}
#endif

template <long GroupBase, int Slice>
static inline void configure_mat_slice(MatmulState *state)
{
    long checksum = 0;
    constexpr int rows = Layout::slice_rows(Slice);
    constexpr uint64_t swizzle = MAT_BATCH_SWIZZLE(5, 3, 5);

    set_lsu_matrix<GroupBase + Constants::kLsuCStore>(
        checksum, rows, Constants::kTileN * (int)sizeof(float),
        Constants::kTileN * (int)sizeof(float), (long)state->c_stride);
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
    __tx_mat_tcsr_write<GroupBase * MAT_GROUPSZ + MAT_ASWIZZLEMBS>(
        swizzle);
    __tx_mat_tcsr_write<GroupBase * MAT_GROUPSZ + MAT_BSWIZZLEMBS>(
        swizzle);
    __tx_mat_tcsr_write<GroupBase * MAT_GROUPSZ + MAT_CSWIZZLEMBS>(
        swizzle);
    __tx_mat_tcsr_write<GroupBase * MAT_GROUPSZ + MAT_PERSISTENTMODE>(0);
    __tx_mat_tcsr_write<GroupBase * MAT_GROUPSZ + MAT_DATATYPE>(
        BaseNPUInstBuilder::DT_BF16);
    state->mat_config[Slice].rows = rows;
    state->mat_config[Slice].columns = Constants::kTileN;
    state->mat_config[Slice].store_rows = rows;
    state->mat_config[Slice].store_columns = Constants::kTileN;
}

static inline void pass_mat_turn(MatmulState *state, uint32_t worker,
                                 dayan_resource_mask_t consume_ready);

template <long GroupBase, int Slice>
static void execute_mat_job(MatmulState *state, const MatJob *job,
                            uint32_t worker)
{
    const int real_rows = rows_for_slice(
        state, job->tile_m, state->row_lane, Slice);
    const int real_columns = cols_for_tile(
        state, job->tile_n, state->column_lane);
    constexpr int a_buffer_index = 0;
    const int b_buffer_index = (int)(job->generation & 1U);
    LocalBuffer *a = &state->a[Slice][a_buffer_index];
    LocalBuffer *b = &state->b[b_buffer_index];
    const uint64_t c_uram = state->uram_base + Layout::c_base(Slice);
    MatConfig *config = &state->mat_config[Slice];

    if (real_rows != config->rows) {
        __tx_mat_tcsr_write<GroupBase * MAT_GROUPSZ + MAT_M>(
            real_rows - 1);
        config->rows = real_rows;
    }
    if (real_columns != config->columns) {
        __tx_mat_tcsr_write<GroupBase * MAT_GROUPSZ + MAT_N>(
            real_columns - 1);
        config->columns = real_columns;
    }
    if (job->tile_k == 0)
        __tx_mat_mm<GroupBase>((long)c_uram, (long)a->uram,
                               (long)b->uram);
    else
        __tx_mat_mma<GroupBase>((long)c_uram, (long)a->uram,
                                (long)b->uram);
    /*
     * The next worker may enqueue only after this command.  The MAT command
     * stream remains strictly ordered, while each worker waits for its own
     * completion token and at most two commands can be outstanding.
     */
    dayan_resource_mask_t consume_ready =
        DAYAN_RESOURCE_MASK(state->mat_turn[worker]);
#if MATMUL_BF16_4C_BROADCAST_MODEL
    consume_ready |= state->lane == a_source_lane(state->row_lane)
        ? DAYAN_RESOURCE_MASK(a->resource)
        : DAYAN_RESOURCE_MASK(shared_a_resource(
              state->row_lane, Slice, a_buffer_index));
    consume_ready |= state->lane == b_source_lane(state->column_lane)
        ? DAYAN_RESOURCE_MASK(b->resource)
        : DAYAN_RESOURCE_MASK(shared_b_resource(
              state->column_lane, b_buffer_index));
#else
    consume_ready |= DAYAN_RESOURCE_MASK(a->resource) |
                     DAYAN_RESOURCE_MASK(b->resource);
#endif
    pass_mat_turn(state, worker, consume_ready);
    dayan_wait_mat();

    if (job->tile_k + 1 == state->count_k) {
        if (real_rows != config->store_rows) {
            __tx_lsu_tcsr_write<
                (GroupBase + Constants::kLsuCStore) * LSU_GROUPSZ +
                LSU_DIM1>(real_rows - 1);
            config->store_rows = real_rows;
        }
        if (real_columns != config->store_columns) {
            __tx_lsu_tcsr_write<
                (GroupBase + Constants::kLsuCStore) * LSU_GROUPSZ +
                LSU_DIM0>((uint64_t)real_columns * sizeof(float) / 64U -
                         1U);
            config->store_columns = real_columns;
        }
        const uint64_t global_row =
            (uint64_t)row_partition_begin(state, state->row_lane) +
            (uint64_t)job->tile_m * Constants::kTileM +
            Layout::slice_offset(Slice);
        const uint64_t global_column =
            (uint64_t)(job->tile_n * Constants::kClusterColumns +
                       (int)state->column_lane) * Constants::kTileN;
        const uint64_t c_dram =
            state->c_base + global_row * state->c_stride +
            global_column * sizeof(float);

        __tx_lsu_dma_u2d<GroupBase + Constants::kLsuCStore>(
            (long)c_dram, (long)c_uram);
        dayan_wait_lsu();
    }
}

static dayan_resource_mask_t mat_inputs(const MatmulState *state,
                                        uint64_t generation, int slice)
{
    const int b_index = (int)(generation & 1U);
    dayan_resource_t a = state->a[slice][0].resource;
    dayan_resource_t b = state->b[b_index].resource;
#if MATMUL_BF16_4C_BROADCAST_MODEL
    if (state->lane != a_source_lane(state->row_lane))
        a = shared_a_resource(state->row_lane, slice, 0);
    if (state->lane != b_source_lane(state->column_lane))
        b = shared_b_resource(state->column_lane, b_index);
#endif
    return DAYAN_RESOURCE_MASK(a) | DAYAN_RESOURCE_MASK(b);
}

static inline bool prepare_mat_job(MatmulState *state, uint64_t sequence,
                                   MatJob *job, uint64_t *phase_generation,
                                   uint64_t *phases)
{
    job->generation = sequence / MATMUL_BF16_4C_SLICES;
    job->slice = (int)(sequence % MATMUL_BF16_4C_SLICES);
    // Both workers follow all resource lifecycles, including generations where
    // only the other worker has active slices. Each input advances once per use.
    for (; *phase_generation < job->generation; ++*phase_generation) {
        int tile_m, tile_n, tile_k;
        decode_generation(state, *phase_generation, &tile_m, &tile_n, &tile_k);
        dayan_resource_mask_t used = 0;
        if (cols_for_tile(state, tile_n, state->column_lane)) {
            for (int slice = 0; slice < MATMUL_BF16_4C_SLICES; ++slice) {
                if (rows_for_slice(state, tile_m, state->row_lane, slice))
                    used |= mat_inputs(state, *phase_generation, slice);
            }
        }
        *phases ^= used;
    }
    decode_generation(state, job->generation, &job->tile_m,
                      &job->tile_n, &job->tile_k);
    const int rows = rows_for_slice(
        state, job->tile_m, state->row_lane, job->slice);
    const int columns = cols_for_tile(
        state, job->tile_n, state->column_lane);

    if (!rows || !columns)
        return false;
    require_status(dayan_resource_wait(
        0, mat_inputs(state, job->generation, job->slice), *phases, 0));
    return true;
}

static inline void wait_mat_turn(MatmulState *state, uint32_t worker,
                                 uint32_t phase)
{
    require_status(dayan_resource_wait(
        DAYAN_RESOURCE_MASK(state->mat_turn[worker ^ 1U]),
        DAYAN_RESOURCE_MASK(state->mat_turn[worker]),
        phase ? DAYAN_RESOURCE_MASK(state->mat_turn[worker]) : 0, 0));
}

static inline void pass_mat_turn(MatmulState *state, uint32_t worker,
                                 dayan_resource_mask_t consume_ready)
{
    require_status(dayan_resource_produce_mat(
        state->mat_turn[worker ^ 1U], 1U, consume_ready));
}

static void mat_worker(void *argument)
{
    MatWorkerArgs *args = (MatWorkerArgs *)argument;
    MatmulState *state = args->state;
    const uint32_t worker = args->worker;
    const uint64_t job_count =
        state->generation_count * MATMUL_BF16_4C_SLICES;
    uint64_t phase_generation = 0;
    uint64_t phases = 0;

    for (uint64_t sequence = worker; sequence < job_count;
         sequence += 2U) {
        MatJob job;
        const bool active = prepare_mat_job(state, sequence, &job,
                                            &phase_generation, &phases);

        wait_mat_turn(state, worker, (sequence / 2U) & 1U);
        if (active) {
            switch (job.slice) {
            case 0:
                configure_mat_slice<0, 0>(state);
                execute_mat_job<0, 0>(state, &job, worker);
                break;
#if MATMUL_BF16_4C_SLICES >= 3
            case 1:
                configure_mat_slice<0, 1>(state);
                execute_mat_job<0, 1>(state, &job, worker);
                break;
            case 2:
                configure_mat_slice<0, 2>(state);
                execute_mat_job<0, 2>(state, &job, worker);
                break;
#endif
#if MATMUL_BF16_4C_SLICES == 4
            case 3:
                configure_mat_slice<0, 3>(state);
                execute_mat_job<0, 3>(state, &job, worker);
                break;
#endif
            default:
                dayan_thread_exit(-22);
            }
        } else {
            pass_mat_turn(
                state, worker,
                DAYAN_RESOURCE_MASK(state->mat_turn[worker]));
        }
    }
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
        cluster->init_status = 0;
        cluster->reserved = 0;
        cluster->init_status = status;
    }
    cluster->uram_base[lane] = uram;
    status = dayan_cluster_barrier(Constants::kClusterCores,
                                   &cluster->barrier);
    if (status < 0)
        return status;
    return cluster->init_status;
}

} // namespace matmul_bf16_4c_detail

#endif /* __DAYAN_DEVICE__ */

__kernel__ void MATMUL_BF16_4C_KERNEL(const MatmulBf16Config *config,
                                     const uint16_t *a,
                                     const uint16_t *bt, float *c)
{
    using namespace matmul_bf16_4c_detail;

    const uint32_t lane = dayan_task_id(DAYAN_TASK_X);

    if (!config_valid(config) || !a || !bt || !c ||
        dayan_task_mode() != 4U ||
        dayan_task_dim(DAYAN_TASK_X) != 4U ||
        dayan_task_dim(DAYAN_TASK_Y) != 1U ||
        dayan_task_dim(DAYAN_TASK_Z) != 1U || lane >= 4U || !shared)
        dayan_thread_exit(-22);

    void *uram = dayan_uram_base();
    if (!uram || dayan_uram_size() < Layout::total_bytes())
        dayan_thread_exit(-12);

    const uint64_t physical_uram = (uint64_t)(uintptr_t)uram;
    const uint64_t expected_window =
        Constants::kUramWindowBase +
        (uint64_t)lane * Constants::kUramWindowStride;
    if (physical_uram < expected_window ||
        physical_uram - expected_window >
            Constants::kUramWindowStride - Layout::total_bytes())
        dayan_thread_exit(-22);

    ClusterShared *cluster = (ClusterShared *)shared;
    int status = initialize_cluster_shared(cluster, lane, physical_uram);
    if (status < 0)
        dayan_thread_exit(status);

    MatmulState state;
    state.m = (int)config->m;
    state.n = (int)config->n;
    state.count_m =
        (state.m + Constants::kClusterRows * Constants::kTileM - 1) /
        (Constants::kClusterRows * Constants::kTileM);
    state.count_n =
        (state.n + Constants::kClusterColumns * Constants::kTileN - 1) /
        (Constants::kClusterColumns * Constants::kTileN);
    state.count_k = (int)(config->k / Constants::kTileK);
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
    state.generation_count =
        (uint64_t)state.count_m * (uint64_t)state.count_n *
        (uint64_t)state.count_k;

    for (int slice = 0; slice < MATMUL_BF16_4C_SLICES; ++slice) {
        state.mat_config[slice].rows = Layout::slice_rows(slice);
        state.mat_config[slice].columns = Constants::kTileN;
        state.mat_config[slice].store_rows = Layout::slice_rows(slice);
        state.mat_config[slice].store_columns = Constants::kTileN;
        for (int index = 0; index < Constants::kABufferCount;
             ++index) {
            LocalBuffer *buffer = &state.a[slice][index];

            buffer->resource = DAYAN_LOCAL_RESOURCE(
                slice * Constants::kABufferCount + index);
            buffer->uram =
                state.uram_base + Layout::a_base(slice, index);
        }
    }
    for (int index = 0; index < 2; ++index) {
        state.b[index].resource = DAYAN_LOCAL_RESOURCE(
            MATMUL_BF16_4C_SLICES * Constants::kABufferCount + index);
        state.b[index].uram = state.uram_base + Layout::b_base(index);
        state.mat_turn[index] = DAYAN_LOCAL_RESOURCE(
            MATMUL_BF16_4C_SLICES * Constants::kABufferCount + 2 +
            index);
    }
    status = dayan_resource_wait_free(state.mat_turn[0]);
    if (status >= 0)
        status = dayan_resource_produce_mat(state.mat_turn[0], 1U, 0);
    if (status < 0)
        dayan_thread_exit(status);

    MatWorkerArgs mat_args[2] = {
        {&state, 0U},
        {&state, 1U},
    };
    dayan_thread_resources_t resources;
    dayan_thread_resources_init(&resources);
    resources.local_resource_count =
        MATMUL_BF16_4C_SLICES * Constants::kABufferCount + 4;
    resources.shared_resource_count =
        Constants::kClusterRows * MATMUL_BF16_4C_SLICES *
            Constants::kABufferCount +
        Constants::kClusterColumns * 2;
    resources.lsu_tcsr_count = 1;
    status = dayan_create_thread("dram-load", dram_loader, &state,
                                 &resources);
#if !MATMUL_BF16_4C_BROADCAST_MODEL
    if (status >= 0)
        status = dayan_create_thread("remote-copy", remote_copy, &state,
                                     &resources);
#endif
    resources.mat_tcsr_count = 1;
    if (status >= 0)
        status = dayan_create_thread("mat-store-0", mat_worker,
                                     &mat_args[0], &resources);
    if (status >= 0)
        status = dayan_create_thread("mat-store-1", mat_worker,
                                     &mat_args[1], &resources);
    if (status < 0)
        dayan_thread_exit(status);

    status = dayan_thread_join_all();
    if (status < 0)
        dayan_thread_exit(status);
    status = dayan_resource_wait_ready(state.mat_turn[0],
        (state.generation_count * MATMUL_BF16_4C_SLICES / 2U) & 1U);
    if (status >= 0)
        status = dayan_resource_consume_lsu(
            DAYAN_RESOURCE_MASK(state.mat_turn[0]));
    if (status < 0)
        dayan_thread_exit(status);
    dayan_wait_lsu();
}

#if MATMUL_BF16_4C_BROADCAST_MODEL
DAYAN_KERNEL_NAMED_RESOURCE_DESC(
    matmul_bf16_broadcast_4c, MATMUL_BF16_4C_WORKERS,
    MATMUL_BF16_4C_ATOMIC_BYTES, MATMUL_BF16_4C_URAM_BYTES,
    4, 2, 0, 0,
    MATMUL_BF16_4C_SLICES *
            matmul_bf16_4c_detail::Constants::kABufferCount +
        4,
    matmul_bf16_4c_detail::Constants::kClusterRows *
            MATMUL_BF16_4C_SLICES *
            matmul_bf16_4c_detail::Constants::kABufferCount +
        matmul_bf16_4c_detail::Constants::kClusterColumns * 2);
#elif MATMUL_BF16_4C_D2A_MODEL
DAYAN_KERNEL_NAMED_RESOURCE_DESC(
    matmul_bf16_4c_d2a_model, MATMUL_BF16_4C_WORKERS,
    MATMUL_BF16_4C_ATOMIC_BYTES, MATMUL_BF16_4C_URAM_BYTES,
    4, 2, 0, 0,
    MATMUL_BF16_4C_SLICES *
            matmul_bf16_4c_detail::Constants::kABufferCount +
        4,
    matmul_bf16_4c_detail::Constants::kClusterRows *
            MATMUL_BF16_4C_SLICES *
            matmul_bf16_4c_detail::Constants::kABufferCount +
        matmul_bf16_4c_detail::Constants::kClusterColumns * 2);
#else
DAYAN_KERNEL_NAMED_RESOURCE_DESC(
    matmul_bf16_4c, MATMUL_BF16_4C_WORKERS,
    MATMUL_BF16_4C_ATOMIC_BYTES, MATMUL_BF16_4C_URAM_BYTES,
    4, 2, 0, 0,
    MATMUL_BF16_4C_SLICES *
            matmul_bf16_4c_detail::Constants::kABufferCount +
        4,
    matmul_bf16_4c_detail::Constants::kClusterRows *
            MATMUL_BF16_4C_SLICES *
            matmul_bf16_4c_detail::Constants::kABufferCount +
        matmul_bf16_4c_detail::Constants::kClusterColumns * 2);
#endif

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
    std::printf("  defaults: M=1024 N=1536 K=1536\n");
    std::printf("  N must be a multiple of 16; K must be a multiple of %u.\n",
                (unsigned)MATMUL_BF16_4C_TILE_K);
    std::printf("  B is generated/stored as BT[N][K], and C is FP32.\n");
#if MATMUL_BF16_4C_D2A_MODEL
#if MATMUL_BF16_4C_BROADCAST_MODEL
    std::printf("  Broadcast timing model: one source D2U fills local+remote "
                "URAM; remote MAT waits on Shared Named Resource directly; "
                "output is not validated.\n");
#else
    std::printf("  D2A timing model: one source D2U represents local+remote "
                "broadcast; output is not validated.\n");
#endif
#endif
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

#if !MATMUL_BF16_4C_D2A_MODEL
__host__ static float bf16_to_float(uint16_t value)
{
    uint32_t bits = (uint32_t)value << 16;
    float result;

    std::memcpy(&result, &bits, sizeof(result));
    return result;
}
#endif

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

#if !MATMUL_BF16_4C_D2A_MODEL
__host__ static int download(npu_stream_t *stream, void *destination,
                             npu_buffer_t *buffer, size_t size,
                             uint32_t timeout_ms)
{
    npu_event_t *event = nullptr;
    int status = npu_memcpy_d2h_async(stream, destination, buffer, 0, size,
                                      &event);

    return status < 0 ? status : wait_and_destroy(event, timeout_ms);
}
#endif

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

#if !MATMUL_BF16_4C_D2A_MODEL
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
#endif

__host__ int main(int argc, char **argv)
{
    uint64_t m = 1024;
    uint64_t n = 1536;
    uint64_t k = 1536;
    size_t a_elements = 0;
    size_t bt_elements = 0;
    size_t c_elements = 0;
    size_t a_bytes = 0;
    size_t bt_bytes = 0;
    size_t c_bytes = 0;
    /* Large RTL-simulation cases can legitimately exceed 30 wall minutes. */
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
#if !MATMUL_BF16_4C_D2A_MODEL
    std::vector<float> c(c_elements, 0.0f);
    std::vector<float> reference(c_elements, 0.0f);
#endif
    initialize_inputs(&a, &bt, (size_t)m, (size_t)n, (size_t)k);
#if !MATMUL_BF16_4C_D2A_MODEL
    reference_matmul(a, bt, &reference, (size_t)m, (size_t)n, (size_t)k);
#endif

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
                                     MATMUL_BF16_4C_KERNEL_LABEL);
        launch_attrs.config.task_dim = NPUTaskDim3{4, 1, 1};
        launch_attrs.config.task_mode = NPUTaskMode::kFourCore;
        launch_attrs.timeout_ms = timeout_ms;
        std::printf("[host] launch %s: A[%llu,%llu] x "
                    "BT[%llu,%llu] -> C[%llu,%llu], tile=%ux%ux%u, "
                    "slices=%u, cluster-grid=%llux%llu\n",
                    MATMUL_BF16_4C_KERNEL_DISPLAY,
                    (unsigned long long)m, (unsigned long long)k,
                    (unsigned long long)n, (unsigned long long)k,
                    (unsigned long long)m, (unsigned long long)n,
                    (unsigned)MATMUL_BF16_4C_TILE_M,
                    (unsigned)MATMUL_BF16_4C_TILE_N,
                    (unsigned)MATMUL_BF16_4C_TILE_K,
                    (unsigned)MATMUL_BF16_4C_SLICES,
                    (unsigned long long)cluster_tiles_m,
                    (unsigned long long)cluster_tiles_n);
        status = MATMUL_BF16_4C_KERNEL(
            config_buffer, a_buffer, bt_buffer, c_buffer,
            &launch_attrs);
    }
#if MATMUL_BF16_4C_D2A_MODEL
    if (status == 0) {
#if MATMUL_BF16_4C_BROADCAST_MODEL
        std::printf("[host] PASS: direct broadcast timing model completed; "
                    "numerical validation intentionally skipped\n");
#else
        std::printf("[host] PASS: D2A timing model completed; "
                    "numerical validation intentionally skipped\n");
#endif
    }
#else
    if (status == 0)
        status = download(stream, c.data(), c_buffer, c_bytes, timeout_ms);
    if (status == 0)
        status = validate_output(c, reference, (size_t)n);
#endif
    if (status < 0)
        std::printf("%s demo failed: %d\n",
                    MATMUL_BF16_4C_KERNEL_DISPLAY, status);

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
