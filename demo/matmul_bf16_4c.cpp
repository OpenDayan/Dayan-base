#include <dayan_kernel.h>
#include <stdint.h>

/*
 * B is supplied in transposed storage: A[M][K] * BT[N][K] -> C[M][N].
 * Strides are measured in bytes.  The four cluster lanes form this grid:
 *
 *                 N-even tile       N-odd tile
 *   lower M half      core 00           core 01
 *   upper M half      core 10           core 11
 *
 * A is sourced from DRAM by cores 00/11 and shared horizontally.  B is
 * sourced from DRAM by cores 01/10 and shared vertically.  For M < N, each
 * Core double-buffers a full A tile and splits B/C into four N slices.  For
 * M >= N, each Core keeps the original four M-sliced A/C buffers and
 * double-buffers a full B tile.  Both paths are template instances of the
 * same pipeline.  Every Core runs one DRAM Loader, one remote Copy worker,
 * two ordered MAT workers, and one ordered C-Store worker.
 * The DRAM and Copy streams advance monotonically; no worker claims work from
 * a shared iterator.  Cross-Core readiness is expressed only with shared
 * resources in Atomic Cache, while all same-Core dependencies use local
 * resources.  The two MAT workers statically alternate linear jobs and pass
 * a resource token after ordered issue, so MAT instructions cannot overtake.
 * The C-Store worker consumes completed output slices in order.  Each C slice
 * is released back to MAT only after its U2D has completed.
 */
struct MatmulBf16Config {
    uint64_t m;
    uint64_t n;
    uint64_t k;
    uint64_t a_stride;
    uint64_t bt_stride;
    uint64_t c_stride;
};

struct MatmulBf16Shape {
    enum : int {
        kTileM = 512,
        kTileN = 384,
        kTileK = 256,
        kSlices = 4,
        kWorkers = 5,
        kAtomicBytes = 2048,
        kUramBytes = 1536 * 1024,
    };
};

#if defined(__DAYAN_DEVICE__)

#include <dayan_intrinsics.h>
#include <dayan_sys.h>

namespace matmul_bf16_4c_detail {

struct Constants {
    enum : int {
        kDramLoad = 0,
        kRemoteCopy = 1,
        kCStore = 3,
        kTileM = MatmulBf16Shape::kTileM,
        kTileN = MatmulBf16Shape::kTileN,
        kTileK = MatmulBf16Shape::kTileK,
        kSlices = MatmulBf16Shape::kSlices,
        kClusterRows = 2,
        kClusterColumns = 2,
        kClusterCores = 4,
        kCoherentCacheLine = 64,
        kUramAlignment = 32,
        kUramWindowBase = 0x00800000,
        kUramWindowStride = 2 * 1024 * 1024,
    };
};

static_assert(Constants::kTileM == 512 && Constants::kTileN == 384,
              "4-Core matmul uses a 512x384 output tile per Core");
static_assert(Constants::kTileK == 256,
              "RTL matmul reaches peak throughput with tile K fixed at 256");
static_assert(Constants::kTileM % Constants::kSlices == 0 &&
                  Constants::kTileN % Constants::kSlices == 0,
              "both output dimensions must split evenly across MAT groups");
static_assert(Constants::kClusterRows == Constants::kClusterColumns,
              "shared sliced/buffered resources assume a square Core grid");
static_assert((Constants::kUramWindowStride &
               (Constants::kUramWindowStride - 1)) == 0,
              "URAM window stride must be a power of two");

static inline int min_int(int left, int right)
{
    return left < right ? left : right;
}

template <bool ProducerLoadsA> struct UramLayout {
    static constexpr int slice_offset(int slice)
    {
        return (ProducerLoadsA ? Constants::kTileN : Constants::kTileM) *
               slice / Constants::kSlices;
    }

    static constexpr int slice_size(int slice)
    {
        return slice_offset(slice + 1) - slice_offset(slice);
    }

    static constexpr int max_rows(int slice)
    {
        return ProducerLoadsA ? Constants::kTileM : slice_size(slice);
    }

    static constexpr int max_cols(int slice)
    {
        return ProducerLoadsA ? slice_size(slice) : Constants::kTileN;
    }

    static constexpr int a_buffer_count()
    {
        return ProducerLoadsA ? 2 : Constants::kSlices;
    }

    static constexpr int b_buffer_count()
    {
        return ProducerLoadsA ? Constants::kSlices : 2;
    }

    static constexpr uint64_t a_bytes()
    {
        return (uint64_t)(ProducerLoadsA ? Constants::kTileM
                                         : slice_size(0)) *
               Constants::kTileK * sizeof(uint16_t);
    }

    static constexpr uint64_t b_bytes()
    {
        return (uint64_t)(ProducerLoadsA ? slice_size(0)
                                         : Constants::kTileN) *
               Constants::kTileK * sizeof(uint16_t);
    }

    static constexpr uint64_t c_bytes()
    {
        return (uint64_t)max_rows(0) * max_cols(0) * sizeof(float);
    }

    static constexpr uint64_t a_base(int buffer)
    {
        return (uint64_t)buffer * a_bytes();
    }

    static constexpr uint64_t b_start()
    {
        return (uint64_t)a_buffer_count() * a_bytes();
    }

    static constexpr uint64_t b_base(int buffer)
    {
        return b_start() + (uint64_t)buffer * b_bytes();
    }

    static constexpr uint64_t c_start()
    {
        return b_start() + (uint64_t)b_buffer_count() * b_bytes();
    }

    static constexpr uint64_t c_base(int slice)
    {
        return c_start() + (uint64_t)slice * c_bytes();
    }

    static constexpr uint64_t total_bytes()
    {
        return c_start() + Constants::kSlices * c_bytes();
    }
};

using AProducerLayout = UramLayout<true>;
using BProducerLayout = UramLayout<false>;

static_assert(BProducerLayout::total_bytes() == 1441792U,
              "512x384x256 single-A/double-B layout must use 1.375 MiB");
static_assert(AProducerLayout::total_bytes() == 1507328U,
              "512x384x256 double-A/sliced-B layout size changed");
static_assert(AProducerLayout::total_bytes() <= MatmulBf16Shape::kUramBytes,
              "A-producer matmul URAM layout is too large");
static_assert(BProducerLayout::total_bytes() <= MatmulBf16Shape::kUramBytes,
              "B-producer matmul URAM layout is too large");

struct alignas(Constants::kCoherentCacheLine) SharedBuffer {
    dayan_shared_resource_t resource;
};

static_assert(sizeof(SharedBuffer) == Constants::kCoherentCacheLine,
              "each shared resource must occupy one coherent cache line");

struct ClusterShared {
    dayan_cluster_barrier_t barrier;
    int32_t init_status;
    uint64_t uram_base[Constants::kClusterCores];
    SharedBuffer sliced[Constants::kClusterRows][Constants::kSlices];
    SharedBuffer buffered[Constants::kClusterRows][2];
};

static_assert(sizeof(ClusterShared) <= MatmulBf16Shape::kAtomicBytes,
              "cluster matmul shared state exceeds its atomic-cache request");

static inline ClusterShared &cluster_shared()
{
    return *static_cast<ClusterShared *>(shared);
}

template <bool ProducerLoadsA>
static inline SharedBuffer *shared_a(uint32_t row_lane, int index)
{
    return ProducerLoadsA ? &cluster_shared().buffered[row_lane][index]
                          : &cluster_shared().sliced[row_lane][index];
}

template <bool ProducerLoadsA>
static inline SharedBuffer *shared_b(uint32_t column_lane, int index)
{
    return ProducerLoadsA ? &cluster_shared().sliced[column_lane][index]
                          : &cluster_shared().buffered[column_lane][index];
}

struct LocalBuffer {
    dayan_resource_t resource;
    uint64_t uram;
};

struct TileConfig {
    int rows;
    int columns;
};

struct LsuConfig {
    int dim0;
    int dim1;
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
    LocalBuffer a[Constants::kSlices];
    LocalBuffer b[Constants::kSlices];
    dayan_resource_t load_issued[2];
    dayan_resource_t mat_turn[2];
    dayan_resource_t c[Constants::kSlices];
    uint64_t generation_count;
    TileConfig mat_config[Constants::kSlices];
    LsuConfig store_config;
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

static inline uint32_t a_source_lane(uint32_t row_lane)
{
    return row_lane == 0U ? 0U : 3U;
}

static inline uint32_t b_source_lane(uint32_t column_lane)
{
    return column_lane == 0U ? 2U : 1U;
}

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

static inline int rows_for_tile(const MatmulState *state, int tile_m,
                                uint32_t row_lane)
{
    const int row_base = row_partition_begin(state, row_lane) +
                         tile_m * Constants::kTileM;
    const int row_end = row_partition_end(state, row_lane);

    if (row_end <= row_base)
        return 0;
    return min_int(row_end - row_base, Constants::kTileM);
}

template <bool ProducerLoadsA>
static inline int rows_for_slice(const MatmulState *state, int tile_m,
                                 uint32_t row_lane, int slice)
{
    using Layout = UramLayout<ProducerLoadsA>;

    if (ProducerLoadsA)
        return rows_for_tile(state, tile_m, row_lane);
    const int row_base = row_partition_begin(state, row_lane) +
                         tile_m * Constants::kTileM +
                         Layout::slice_offset(slice);
    const int row_end = row_partition_end(state, row_lane);

    if (row_end <= row_base)
        return 0;
    return min_int(row_end - row_base, Layout::slice_size(slice));
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

template <bool ProducerLoadsA>
static inline int cols_for_slice(const MatmulState *state, int tile_n,
                                 uint32_t column_lane, int slice)
{
    using Layout = UramLayout<ProducerLoadsA>;

    if (!ProducerLoadsA)
        return cols_for_tile(state, tile_n, column_lane);
    const int global_tile_n = tile_n * Constants::kClusterColumns +
                              (int)column_lane;
    const int column_base = global_tile_n * Constants::kTileN +
                            Layout::slice_offset(slice);

    if (state->n <= column_base)
        return 0;
    return min_int(state->n - column_base, Layout::slice_size(slice));
}

template <bool ProducerLoadsA>
static inline int active_slices(const MatmulState *state, int tile_m,
                                int tile_n, uint32_t row_lane,
                                uint32_t column_lane)
{
    int count = 0;

    for (int slice = 0; slice < Constants::kSlices; ++slice) {
        if (rows_for_slice<ProducerLoadsA>(state, tile_m, row_lane,
                                           slice) > 0 &&
            cols_for_slice<ProducerLoadsA>(state, tile_n, column_lane,
                                           slice) > 0)
            ++count;
    }
    return count;
}

static inline int k_for_tile(const MatmulState *state, int tile_k)
{
    const int k_base = tile_k * Constants::kTileK;

    if (state->k <= k_base)
        return 0;
    return min_int(state->k - k_base, Constants::kTileK);
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
        dayan_proc_exit(status);
}

template <bool ProducerLoadsA>
static inline bool generation_has_local_work(const MatmulState *state,
                                             int tile_m, int tile_n)
{
    return active_slices<ProducerLoadsA>(
               state, tile_m, tile_n, state->row_lane,
               state->column_lane) > 0;
}

template <bool ProducerLoadsA, long LoadGroup>
static inline LsuConfig configure_loader(const MatmulState *state,
                                         bool loads_a, bool from_dram)
{
    using Layout = UramLayout<ProducerLoadsA>;

    long checksum = 0;
    constexpr int row_bytes =
        Constants::kTileK * (int)sizeof(uint16_t);
    constexpr uint64_t swizzle = MAT_BATCH_SWIZZLE(5, 3, 5);
    const int rows = loads_a ?
        (ProducerLoadsA ? Constants::kTileM : Layout::slice_size(0)) :
        (ProducerLoadsA ? Layout::slice_size(0) : Constants::kTileN);
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
    return {
        row_bytes / 64 - 1,
        rows - 1,
    };
}

template <long Tcsr>
static inline void configure_lsu_tcsr(int value, int *configured_value)
{
    if (value == *configured_value)
        return;
    __tx_lsu_tcsr_write<Tcsr>(value);
    *configured_value = value;
}

static inline void begin_resource(dayan_resource_t *resource,
                                  uint32_t consumers)
{
    require_status(dayan_resource_wait_free(resource));
    require_status(dayan_resource_alloc(resource, 1U, consumers));
}

static inline void begin_resource(dayan_shared_resource_t *resource,
                                  uint32_t consumers)
{
    require_status(dayan_shared_resource_wait_free(resource));
    require_status(dayan_shared_resource_alloc(resource, 1U, consumers));
}

template <long LoadGroup>
static inline void issue_dram_load(LocalBuffer *local,
                                   SharedBuffer *remote,
                                   uint32_t local_consumers,
                                   bool remote_active, uint64_t dram,
                                   uint64_t major_bytes,
                                   dayan_resource_t *load_issued,
                                   int *configured_dim0)
{
    if (local_consumers)
        begin_resource(&local->resource, local_consumers);
    if (remote_active)
        begin_resource(&remote->resource, 1U);

    uint64_t uram = local->uram;
    bool first_command = true;

    while (major_bytes) {
        uint64_t handled_bytes = major_bytes & ~0x3fULL;
        const uint64_t remaining_bytes = major_bytes & 0x3fULL;
        int dim0;
        uint64_t request_size;

        if (handled_bytes == 0) {
            handled_bytes = remaining_bytes;
            dim0 = 0;
            request_size = remaining_bytes;
        } else {
            dim0 = (int)(handled_bytes / 64U - 1U);
            request_size = 64U;
        }
        configure_lsu_tcsr<LoadGroup * LSU_GROUPSZ + LSU_DIM0>(
            dim0, configured_dim0);
        __tx_lsu_tcsr_write<
            LoadGroup * LSU_GROUPSZ + LSU_REQUESTREADSIZE>(
                request_size - 1U);
        __tx_lsu_dma_d2u<LoadGroup>((long)uram, (long)dram);
        if (first_command && load_issued)
            require_status(dayan_resource_produce(load_issued));
        first_command = false;
        uram += handled_bytes;
        dram += handled_bytes;
        major_bytes -= handled_bytes;
    }
    dayan_wait_lsu();

    if (remote_active)
        require_status(dayan_shared_resource_produce(&remote->resource));
    if (local_consumers)
        require_status(dayan_resource_produce(&local->resource));
}

template <long LoadGroup>
static inline void issue_remote_copy(LocalBuffer *local,
                                     SharedBuffer *remote,
                                     uint32_t local_consumers, int rows,
                                     uint64_t source_uram,
                                     int *configured_dim1)
{
    begin_resource(&local->resource, local_consumers);
    require_status(dayan_shared_resource_wait_ready(&remote->resource));
    configure_lsu_tcsr<LoadGroup * LSU_GROUPSZ + LSU_DIM1>(
        rows - 1, configured_dim1);
    __tx_lsu_dma_d2u<LoadGroup>((long)local->uram, (long)source_uram);
    dayan_wait_lsu();
    require_status(dayan_shared_resource_consume(&remote->resource));
    require_status(dayan_resource_produce(&local->resource));
}

template <bool ProducerLoadsA, long LoadGroup>
static void dram_load_a(MatmulState *state, uint64_t generation,
                        int tile_m, int tile_n, int tile_k,
                        LsuConfig *configured)
{
    using Layout = UramLayout<ProducerLoadsA>;

    const bool signal_copy =
        generation_has_local_work<ProducerLoadsA>(state, tile_m, tile_n);
    const int real_k = k_for_tile(state, tile_k);
    bool issued = false;

    if (signal_copy)
        begin_resource(&state->load_issued[generation & 1U], 1U);
    if (ProducerLoadsA) {
        const int real_rows = rows_for_tile(
            state, tile_m, state->row_lane);
        const int local_users = active_slices<ProducerLoadsA>(
            state, tile_m, tile_n, state->row_lane,
            state->column_lane);
        const int peer_users = active_slices<ProducerLoadsA>(
            state, tile_m, tile_n, state->row_lane,
            state->column_lane ^ 1U);

        if (!real_rows || (!local_users && !peer_users))
            return;
        const int buffer_index = (int)(generation & 1U);
        LocalBuffer *local = &state->a[buffer_index];
        SharedBuffer *shared_buffer = shared_a<ProducerLoadsA>(
            state->row_lane, buffer_index);

        configure_lsu_tcsr<LoadGroup * LSU_GROUPSZ + LSU_DIM1>(
            real_rows - 1, &configured->dim1);
        const uint64_t global_row =
            (uint64_t)row_partition_begin(state, state->row_lane) +
            (uint64_t)tile_m * Constants::kTileM;
        const uint64_t a_dram =
            state->a_base + global_row * state->a_stride +
            (uint64_t)tile_k * Constants::kTileK * sizeof(uint16_t);

        issue_dram_load<LoadGroup>(
            local, shared_buffer, (uint32_t)local_users, peer_users > 0,
            a_dram, (uint64_t)real_k * sizeof(uint16_t),
            signal_copy ? &state->load_issued[generation & 1U] : nullptr,
            &configured->dim0);
        return;
    }

    const int columns = cols_for_tile(
        state, tile_n, state->column_lane);
    const int peer_columns = cols_for_tile(
        state, tile_n, state->column_lane ^ 1U);
    for (int slice = 0; slice < Constants::kSlices; ++slice) {
        const int real_rows = rows_for_slice<ProducerLoadsA>(
            state, tile_m, state->row_lane, slice);
        const bool local_active = real_rows > 0 && columns > 0;
        const bool peer_active = real_rows > 0 && peer_columns > 0;
        LocalBuffer *local = &state->a[slice];
        SharedBuffer *shared_buffer = shared_a<ProducerLoadsA>(
            state->row_lane, slice);

        if (!local_active && !peer_active)
            continue;
        configure_lsu_tcsr<LoadGroup * LSU_GROUPSZ + LSU_DIM1>(
            real_rows - 1, &configured->dim1);

        const uint64_t global_row =
            (uint64_t)row_partition_begin(state, state->row_lane) +
            (uint64_t)tile_m * Constants::kTileM +
            (uint64_t)Layout::slice_offset(slice);
        const uint64_t a_dram =
            state->a_base + global_row * state->a_stride +
            (uint64_t)tile_k * Constants::kTileK * sizeof(uint16_t);
        dayan_resource_t *load_issued =
            !issued && signal_copy ?
                &state->load_issued[generation & 1U] : nullptr;

        issue_dram_load<LoadGroup>(
            local, shared_buffer, local_active ? 1U : 0U, peer_active,
            a_dram, (uint64_t)real_k * sizeof(uint16_t), load_issued,
            &configured->dim0);
        if (load_issued)
            issued = true;
    }
}

template <bool ProducerLoadsA, long LoadGroup>
static void dram_load_b(MatmulState *state, uint64_t generation,
                        int tile_m, int tile_n, int tile_k,
                        LsuConfig *configured)
{
    using Layout = UramLayout<ProducerLoadsA>;

    const bool signal_copy =
        generation_has_local_work<ProducerLoadsA>(state, tile_m, tile_n);
    const int real_k = k_for_tile(state, tile_k);
    bool issued = false;

    if (signal_copy)
        begin_resource(&state->load_issued[generation & 1U], 1U);
    if (ProducerLoadsA) {
        const int local_rows = rows_for_tile(
            state, tile_m, state->row_lane);
        const int peer_rows = rows_for_tile(
            state, tile_m, state->row_lane ^ 1U);

        for (int slice = 0; slice < Constants::kSlices; ++slice) {
            const int real_columns = cols_for_slice<ProducerLoadsA>(
                state, tile_n, state->column_lane, slice);
            const bool local_active = real_columns > 0 && local_rows > 0;
            const bool peer_active = real_columns > 0 && peer_rows > 0;
            LocalBuffer *local = &state->b[slice];
            SharedBuffer *shared_buffer = shared_b<ProducerLoadsA>(
                state->column_lane, slice);

            if (!local_active && !peer_active)
                continue;
            configure_lsu_tcsr<LoadGroup * LSU_GROUPSZ + LSU_DIM1>(
                real_columns - 1, &configured->dim1);
            const uint64_t global_column =
                (uint64_t)(tile_n * Constants::kClusterColumns +
                           (int)state->column_lane) * Constants::kTileN +
                (uint64_t)Layout::slice_offset(slice);
            const uint64_t bt_dram =
                state->bt_base + global_column * state->bt_stride +
                (uint64_t)tile_k * Constants::kTileK * sizeof(uint16_t);
            dayan_resource_t *load_issued =
                !issued && signal_copy ?
                    &state->load_issued[generation & 1U] : nullptr;

            issue_dram_load<LoadGroup>(
                local, shared_buffer, local_active ? 1U : 0U,
                peer_active, bt_dram,
                (uint64_t)real_k * sizeof(uint16_t), load_issued,
                &configured->dim0);
            if (load_issued)
                issued = true;
        }
        return;
    }

    const int real_columns = cols_for_tile(
        state, tile_n, state->column_lane);
    const int local_users = active_slices<ProducerLoadsA>(
        state, tile_m, tile_n, state->row_lane, state->column_lane);
    const int peer_users = active_slices<ProducerLoadsA>(
        state, tile_m, tile_n, state->row_lane ^ 1U,
        state->column_lane);
    const int buffer_index = (int)(generation & 1U);
    LocalBuffer *local = &state->b[buffer_index];
    SharedBuffer *shared_buffer = shared_b<ProducerLoadsA>(
        state->column_lane, buffer_index);

    if (!real_columns || (!local_users && !peer_users))
        return;
    configure_lsu_tcsr<LoadGroup * LSU_GROUPSZ + LSU_DIM1>(
        real_columns - 1, &configured->dim1);

    const uint64_t global_column =
        (uint64_t)(tile_n * Constants::kClusterColumns +
                   (int)state->column_lane) * Constants::kTileN;
    const uint64_t bt_dram =
        state->bt_base + global_column * state->bt_stride +
        (uint64_t)tile_k * Constants::kTileK * sizeof(uint16_t);

    issue_dram_load<LoadGroup>(
        local, shared_buffer, (uint32_t)local_users, peer_users > 0,
        bt_dram, (uint64_t)real_k * sizeof(uint16_t),
        signal_copy ? &state->load_issued[generation & 1U] : nullptr,
        &configured->dim0);
}

template <bool ProducerLoadsA>
static void dram_loader(void *argument)
{
    MatmulState *state = (MatmulState *)argument;
    const bool loads_a =
        state->lane == a_source_lane(state->row_lane);
    LsuConfig configured =
        configure_loader<ProducerLoadsA, Constants::kDramLoad>(
            state, loads_a, true);

    for (uint64_t generation = 0;
         generation < state->generation_count; ++generation) {
        int tile_m;
        int tile_n;
        int tile_k;

        decode_generation(state, generation, &tile_m, &tile_n, &tile_k);
        if (loads_a) {
            dram_load_a<ProducerLoadsA, Constants::kDramLoad>(
                state, generation, tile_m, tile_n, tile_k,
                &configured);
        } else {
            dram_load_b<ProducerLoadsA, Constants::kDramLoad>(
                state, generation, tile_m, tile_n, tile_k,
                &configured);
        }
    }
}

template <bool ProducerLoadsA, long LoadGroup>
static void copy_a(MatmulState *state, uint64_t generation,
                   int tile_m, int tile_n,
                   int *configured_dim1)
{
    using Layout = UramLayout<ProducerLoadsA>;

    const uint32_t source_lane = a_source_lane(state->row_lane);
    if (ProducerLoadsA) {
        const int real_rows = rows_for_tile(
            state, tile_m, state->row_lane);
        const int local_users = active_slices<ProducerLoadsA>(
            state, tile_m, tile_n, state->row_lane,
            state->column_lane);

        if (!real_rows || !local_users)
            return;
        const int buffer_index = (int)(generation & 1U);
        const uint64_t source_uram =
            cluster_shared().uram_base[source_lane] +
            Layout::a_base(buffer_index);

        issue_remote_copy<LoadGroup>(
            &state->a[buffer_index],
            shared_a<ProducerLoadsA>(state->row_lane, buffer_index),
            (uint32_t)local_users, real_rows, source_uram,
            configured_dim1);
        return;
    }

    const int columns = cols_for_tile(state, tile_n, state->column_lane);

    for (int slice = 0; slice < Constants::kSlices; ++slice) {
        const int real_rows = rows_for_slice<ProducerLoadsA>(
            state, tile_m, state->row_lane, slice);

        if (!real_rows || !columns)
            continue;
        LocalBuffer *local = &state->a[slice];
        SharedBuffer *shared_buffer = shared_a<ProducerLoadsA>(
            state->row_lane, slice);

        const uint64_t source_uram =
            cluster_shared().uram_base[source_lane] +
            Layout::a_base(slice);

        issue_remote_copy<LoadGroup>(
            local, shared_buffer, 1U, real_rows, source_uram,
            configured_dim1);
    }
}

template <bool ProducerLoadsA, long LoadGroup>
static void copy_b(MatmulState *state, uint64_t generation,
                   int tile_m, int tile_n, int *configured_dim1)
{
    using Layout = UramLayout<ProducerLoadsA>;

    const uint32_t source_lane = b_source_lane(state->column_lane);
    if (ProducerLoadsA) {
        const int real_rows = rows_for_tile(
            state, tile_m, state->row_lane);

        for (int slice = 0; slice < Constants::kSlices; ++slice) {
            const int real_columns = cols_for_slice<ProducerLoadsA>(
                state, tile_n, state->column_lane, slice);

            if (!real_rows || !real_columns)
                continue;
            const uint64_t source_uram =
                cluster_shared().uram_base[source_lane] +
                Layout::b_base(slice);

            issue_remote_copy<LoadGroup>(
                &state->b[slice],
                shared_b<ProducerLoadsA>(state->column_lane, slice),
                1U, real_columns, source_uram, configured_dim1);
        }
        return;
    }

    const int real_columns = cols_for_tile(
        state, tile_n, state->column_lane);
    const int local_users = active_slices<ProducerLoadsA>(
        state, tile_m, tile_n, state->row_lane, state->column_lane);
    const int buffer_index = (int)(generation & 1U);

    if (!real_columns || !local_users)
        return;
    LocalBuffer *local = &state->b[buffer_index];
    SharedBuffer *shared_buffer = shared_b<ProducerLoadsA>(
        state->column_lane, buffer_index);

    const uint64_t source_uram =
        cluster_shared().uram_base[source_lane] +
        Layout::b_base(buffer_index);

    issue_remote_copy<LoadGroup>(
        local, shared_buffer, (uint32_t)local_users, real_columns,
        source_uram, configured_dim1);
}

template <bool ProducerLoadsA>
static void remote_copy(void *argument)
{
    MatmulState *state = (MatmulState *)argument;
    const bool copies_a =
        state->lane != a_source_lane(state->row_lane);
    LsuConfig configured =
        configure_loader<ProducerLoadsA, Constants::kRemoteCopy>(
            state, copies_a, false);

    for (uint64_t generation = 0;
         generation < state->generation_count; ++generation) {
        int tile_m;
        int tile_n;
        int tile_k;

        decode_generation(state, generation, &tile_m, &tile_n, &tile_k);
        (void)tile_k;
        if (!generation_has_local_work<ProducerLoadsA>(
                state, tile_m, tile_n))
            continue;
        dayan_resource_t *load_issued =
            &state->load_issued[generation & 1U];

        require_status(dayan_resource_wait_ready(load_issued));
        require_status(dayan_resource_consume(load_issued));
        if (copies_a) {
            copy_a<ProducerLoadsA, Constants::kRemoteCopy>(
                state, generation, tile_m, tile_n, &configured.dim1);
        } else {
            copy_b<ProducerLoadsA, Constants::kRemoteCopy>(
                state, generation, tile_m, tile_n, &configured.dim1);
        }
    }
}

template <bool ProducerLoadsA, long GroupBase, int Slice>
static inline void configure_mat_group()
{
    using Layout = UramLayout<ProducerLoadsA>;

    constexpr int rows = Layout::max_rows(Slice);
    constexpr int columns = Layout::max_cols(Slice);
    constexpr uint64_t swizzle = MAT_BATCH_SWIZZLE(5, 3, 5);

    __tx_mat_tcsr_write<GroupBase * MAT_GROUPSZ + MAT_M>(rows - 1);
    __tx_mat_tcsr_write<GroupBase * MAT_GROUPSZ + MAT_N>(columns - 1);
    __tx_mat_tcsr_write<GroupBase * MAT_GROUPSZ + MAT_K>(
        Constants::kTileK - 1);
    __tx_mat_tcsr_write<GroupBase * MAT_GROUPSZ + MAT_ASTRIDE>(
        Constants::kTileK * sizeof(uint16_t));
    __tx_mat_tcsr_write<GroupBase * MAT_GROUPSZ + MAT_BSTRIDE>(
        Constants::kTileK * sizeof(uint16_t));
    __tx_mat_tcsr_write<GroupBase * MAT_GROUPSZ + MAT_CSTRIDE>(
        columns * sizeof(float));
    __tx_mat_tcsr_write<GroupBase * MAT_GROUPSZ + MAT_ASWIZZLEMBS>(
        swizzle);
    __tx_mat_tcsr_write<GroupBase * MAT_GROUPSZ + MAT_BSWIZZLEMBS>(
        swizzle);
    __tx_mat_tcsr_write<GroupBase * MAT_GROUPSZ + MAT_CSWIZZLEMBS>(
        swizzle);
    __tx_mat_tcsr_write<GroupBase * MAT_GROUPSZ + MAT_PERSISTENTMODE>(0);
    __tx_mat_tcsr_write<GroupBase * MAT_GROUPSZ + MAT_DATATYPE>(
        BaseNPUInstBuilder::DT_BF16);
}

template <bool ProducerLoadsA, long StoreGroup>
static inline LsuConfig configure_c_store(const MatmulState *state)
{
    using Layout = UramLayout<ProducerLoadsA>;

    constexpr int rows = Layout::max_rows(0);
    constexpr int columns = Layout::max_cols(0);
    long checksum = 0;
    constexpr uint64_t swizzle = MAT_BATCH_SWIZZLE(5, 3, 5);

    set_lsu_matrix<StoreGroup>(
        checksum, rows, columns * (int)sizeof(float),
        columns * (int)sizeof(float), (long)state->c_stride);
    __tx_lsu_tcsr_write<
        StoreGroup * LSU_GROUPSZ + LSU_URAMREADSWIZZLEMBS>(swizzle);
    return {
        columns * (int)sizeof(float) / 64 - 1,
        rows - 1,
    };
}

static inline void pass_mat_turn(MatmulState *state, uint32_t worker)
{
    dayan_resource_t *turn = &state->mat_turn[worker ^ 1U];

    require_status(dayan_resource_wait_free(turn));
    require_status(dayan_resource_alloc(turn, 0U, 1U));
}

template <bool ProducerLoadsA, long GroupBase, int Slice>
static void execute_mat_job(MatmulState *state, const MatJob &job,
                            uint32_t worker)
{
    using Layout = UramLayout<ProducerLoadsA>;

    const int real_rows = rows_for_slice<ProducerLoadsA>(
        state, job.tile_m, state->row_lane, Slice);
    const int real_columns = cols_for_slice<ProducerLoadsA>(
        state, job.tile_n, state->column_lane, Slice);
    const int a_buffer_index =
        ProducerLoadsA ? (int)(job.generation & 1U) : Slice;
    const int b_buffer_index =
        ProducerLoadsA ? Slice : (int)(job.generation & 1U);
    LocalBuffer *a = &state->a[a_buffer_index];
    LocalBuffer *b = &state->b[b_buffer_index];
    const uint64_t c_uram = state->uram_base + Layout::c_base(Slice);
    TileConfig *config = &state->mat_config[Slice];

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
    __tx_mat_tcsr_write<GroupBase * MAT_GROUPSZ + MAT_K>(
        k_for_tile(state, job.tile_k) - 1);
    if (job.tile_k == 0)
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
    pass_mat_turn(state, worker);
    dayan_wait_mat();

    if (job.tile_k + 1 == state->count_k)
        require_status(dayan_resource_produce(&state->c[Slice]));

    require_status(dayan_resource_consume(&a->resource));
    require_status(dayan_resource_consume(&b->resource));
}

template <bool ProducerLoadsA>
static void mat_worker(void *argument)
{
    MatWorkerArgs *args = (MatWorkerArgs *)argument;
    MatmulState *state = args->state;
    const uint32_t worker = args->worker;
    const uint64_t job_count =
        state->generation_count * Constants::kSlices;

    for (uint64_t sequence = worker; sequence < job_count;
         sequence += 2U) {
        MatJob job;
        job.generation = sequence / Constants::kSlices;
        job.slice = (int)(sequence % Constants::kSlices);
        decode_generation(state, job.generation, &job.tile_m,
                          &job.tile_n, &job.tile_k);
        const bool active =
            rows_for_slice<ProducerLoadsA>(
                state, job.tile_m, state->row_lane, job.slice) > 0 &&
            cols_for_slice<ProducerLoadsA>(
                state, job.tile_n, state->column_lane, job.slice) > 0;

        if (active) {
            const int a_buffer = ProducerLoadsA ?
                (int)(job.generation & 1U) : job.slice;
            const int b_buffer = ProducerLoadsA ?
                job.slice : (int)(job.generation & 1U);

            require_status(dayan_resource_wait_ready(
                &state->a[a_buffer].resource));
            require_status(dayan_resource_wait_ready(
                &state->b[b_buffer].resource));
            if (job.tile_k == 0) {
                dayan_resource_t *output = &state->c[job.slice];

                require_status(dayan_resource_wait_free(output));
                require_status(dayan_resource_alloc(output, 1U, 1U));
            }
        }

        dayan_resource_t *turn = &state->mat_turn[worker];

        require_status(dayan_resource_wait_ready(turn));
        require_status(dayan_resource_consume(turn));
        if (active) {
            switch (job.slice) {
            case 0:
                execute_mat_job<ProducerLoadsA, 0, 0>(
                    state, job, worker);
                break;
            case 1:
                execute_mat_job<ProducerLoadsA, 4, 1>(
                    state, job, worker);
                break;
            case 2:
                execute_mat_job<ProducerLoadsA, 2, 2>(
                    state, job, worker);
                break;
            case 3:
                execute_mat_job<ProducerLoadsA, 1, 3>(
                    state, job, worker);
                break;
            default:
                dayan_proc_exit(-22);
            }
        } else {
            pass_mat_turn(state, worker);
        }
    }
}

template <bool ProducerLoadsA>
static void issue_c_store(MatmulState *state, int tile_m, int tile_n,
                          int slice)
{
    using Layout = UramLayout<ProducerLoadsA>;

    const int real_rows = rows_for_slice<ProducerLoadsA>(
        state, tile_m, state->row_lane, slice);
    const int real_columns = cols_for_slice<ProducerLoadsA>(
        state, tile_n, state->column_lane, slice);
    require_status(dayan_resource_wait_ready(&state->c[slice]));
    configure_lsu_tcsr<Constants::kCStore * LSU_GROUPSZ + LSU_DIM1>(
        real_rows - 1, &state->store_config.dim1);
    configure_lsu_tcsr<Constants::kCStore * LSU_GROUPSZ + LSU_DIM0>(
        (int)((uint64_t)real_columns * sizeof(float) / 64U - 1U),
        &state->store_config.dim0);
    const uint64_t global_row =
        (uint64_t)row_partition_begin(state, state->row_lane) +
        (uint64_t)tile_m * Constants::kTileM +
        (ProducerLoadsA ? 0U : (uint64_t)Layout::slice_offset(slice));
    const uint64_t global_column =
        (uint64_t)(tile_n * Constants::kClusterColumns +
                   (int)state->column_lane) * Constants::kTileN +
        (ProducerLoadsA ? (uint64_t)Layout::slice_offset(slice) : 0U);
    const uint64_t c_uram = state->uram_base + Layout::c_base(slice);
    const uint64_t c_dram =
        state->c_base + global_row * state->c_stride +
        global_column * sizeof(float);

    __tx_lsu_dma_u2d<Constants::kCStore>((long)c_dram, (long)c_uram);
}

static inline void complete_c_store_batch(MatmulState *state,
                                          uint32_t active_mask)
{
    dayan_wait_lsu();
    for (int slice = 0; slice < Constants::kSlices; ++slice) {
        if (active_mask & (1U << slice))
            require_status(dayan_resource_consume(&state->c[slice]));
    }
}

template <bool ProducerLoadsA>
static void c_store_worker(void *argument)
{
    MatmulState *state = (MatmulState *)argument;
    const uint64_t output_count =
        (uint64_t)state->count_m * (uint64_t)state->count_n;

    for (uint64_t output = 0; output < output_count; ++output) {
        const int tile_m = (int)(output / (uint64_t)state->count_n);
        const int tile_n = (int)(output % (uint64_t)state->count_n);
        uint32_t active_mask = 0;
        int issued = 0;

        for (int slice = 0; slice < Constants::kSlices; ++slice) {
            const int rows = rows_for_slice<ProducerLoadsA>(
                state, tile_m, state->row_lane, slice);
            const int columns = cols_for_slice<ProducerLoadsA>(
                state, tile_n, state->column_lane, slice);

            if (!rows || !columns)
                continue;
            active_mask |= 1U << slice;
            issue_c_store<ProducerLoadsA>(state, tile_m, tile_n, slice);
            if (++issued == 2) {
                complete_c_store_batch(state, active_mask);
                active_mask = 0;
                issued = 0;
            }
        }
        if (active_mask)
            complete_c_store_batch(state, active_mask);
    }
}

static inline int config_valid(const MatmulBf16Config *config)
{
    const uint64_t int_limit = 0x7ffff000ULL;

    if (!config || !config->m || !config->n || !config->k ||
        config->m > int_limit || config->n > int_limit ||
        config->k > int_limit || (config->n % 16U) != 0 ||
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

static inline int initialize_shared_buffers(SharedBuffer *buffers, int count)
{
    for (int index = 0; index < count; ++index) {
        const int status = dayan_shared_resource_init(
            &buffers[index].resource);

        if (status < 0)
            return status;
    }
    return 0;
}

static int initialize_cluster_shared(ClusterShared &cluster, uint32_t lane,
                                     uint64_t uram)
{
    int status = 0;

    if (lane == 0U) {
        cluster.init_status = 0;
        status = initialize_shared_buffers(
            &cluster.sliced[0][0],
            Constants::kClusterRows * Constants::kSlices);
        if (status == 0)
            status = initialize_shared_buffers(
                &cluster.buffered[0][0], Constants::kClusterRows * 2);
        cluster.init_status = status;
    }
    cluster.uram_base[lane] = uram;
    status = dayan_cluster_barrier(Constants::kClusterCores,
                                   &cluster.barrier);
    if (status < 0)
        return status;
    return cluster.init_status;
}

template <bool ProducerLoadsA>
static int run_pipeline(const MatmulBf16Config *config,
                        const uint16_t *a, const uint16_t *bt, float *c,
                        uint32_t lane, uint64_t physical_uram)
{
    using Layout = UramLayout<ProducerLoadsA>;

    int status = initialize_cluster_shared(
        cluster_shared(), lane, physical_uram);
    if (status < 0)
        return status;

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
    state.generation_count =
        (uint64_t)state.count_m * (uint64_t)state.count_n *
        (uint64_t)state.count_k;

    for (int slice = 0; slice < Constants::kSlices; ++slice) {
        state.mat_config[slice].rows = Layout::max_rows(slice);
        state.mat_config[slice].columns = Layout::max_cols(slice);
        status = dayan_resource_init(&state.c[slice]);
        if (status < 0)
            return status;
    }
    for (int index = 0; index < Layout::a_buffer_count(); ++index) {
        status = dayan_resource_init(&state.a[index].resource);
        if (status < 0)
            return status;
        state.a[index].uram = state.uram_base + Layout::a_base(index);
    }
    for (int index = 0; index < Layout::b_buffer_count(); ++index) {
        status = dayan_resource_init(&state.b[index].resource);
        if (status < 0)
            return status;
        state.b[index].uram = state.uram_base + Layout::b_base(index);
    }
    for (int index = 0; index < 2; ++index) {
        status = dayan_resource_init(&state.load_issued[index]);
        if (status < 0)
            return status;
        status = dayan_resource_init(&state.mat_turn[index]);
        if (status < 0)
            return status;
    }
    status = dayan_resource_alloc(&state.mat_turn[0], 0U, 1U);
    if (status < 0)
        return status;

    configure_mat_group<ProducerLoadsA, 0, 0>();
    configure_mat_group<ProducerLoadsA, 4, 1>();
    configure_mat_group<ProducerLoadsA, 2, 2>();
    configure_mat_group<ProducerLoadsA, 1, 3>();
    state.store_config =
        configure_c_store<ProducerLoadsA, Constants::kCStore>(&state);
    MatWorkerArgs mat_args[2] = {
        {&state, 0U},
        {&state, 1U},
    };
    status = dayan_create_proc("dram-load", dram_loader<ProducerLoadsA>,
                               &state);
    if (status >= 0)
        status = dayan_create_proc("remote-copy",
                                   remote_copy<ProducerLoadsA>, &state);
    if (status >= 0)
        status = dayan_create_proc("mat-0", mat_worker<ProducerLoadsA>,
                                   &mat_args[0]);
    if (status >= 0)
        status = dayan_create_proc("mat-1", mat_worker<ProducerLoadsA>,
                                   &mat_args[1]);
    if (status >= 0)
        status = dayan_create_proc("c-store",
                                   c_store_worker<ProducerLoadsA>, &state);
    if (status < 0)
        return status;

    return dayan_join_all();
}

} // namespace matmul_bf16_4c_detail

#endif /* __DAYAN_DEVICE__ */

__kernel__ void matmul_bf16_4c(const MatmulBf16Config *config,
                              const uint16_t *a, const uint16_t *bt,
                              float *c)
{
    using namespace matmul_bf16_4c_detail;

    const uint32_t lane = dayan_task_id(DAYAN_TASK_X);

    if (!config_valid(config) || !a || !bt || !c ||
        dayan_task_mode() != 4U ||
        dayan_task_dim(DAYAN_TASK_X) != 4U ||
        dayan_task_dim(DAYAN_TASK_Y) != 1U ||
        dayan_task_dim(DAYAN_TASK_Z) != 1U || lane >= 4U || !shared)
        dayan_proc_exit(-22);

    void *uram = dayan_uram_base();
    const bool producer_loads_a = config->m < config->n;
    const uint64_t required_uram =
        producer_loads_a ? AProducerLayout::total_bytes()
                         : BProducerLayout::total_bytes();
    if (!uram || dayan_uram_size() < required_uram)
        dayan_proc_exit(-12);

    const uint64_t physical_uram = (uint64_t)(uintptr_t)uram;
    const uint64_t expected_window =
        Constants::kUramWindowBase +
        (uint64_t)lane * Constants::kUramWindowStride;
    if (physical_uram < expected_window ||
        physical_uram - expected_window >
            Constants::kUramWindowStride - required_uram)
        dayan_proc_exit(-22);

    const int status = producer_loads_a
        ? run_pipeline<true>(config, a, bt, c, lane, physical_uram)
        : run_pipeline<false>(config, a, bt, c, lane, physical_uram);
    if (status < 0)
        dayan_proc_exit(status);
}

DAYAN_KERNEL_DESC(matmul_bf16_4c, MatmulBf16Shape::kWorkers,
                  MatmulBf16Shape::kAtomicBytes,
                  MatmulBf16Shape::kUramBytes);

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
            (m + 2U * MatmulBf16Shape::kTileM - 1U) /
            (2U * MatmulBf16Shape::kTileM);
        const uint64_t cluster_tiles_n =
            (n + 2U * MatmulBf16Shape::kTileN - 1U) /
            (2U * MatmulBf16Shape::kTileN);

        npu_kernel_launch_attrs_init(&launch_attrs, context, stream,
                                     "matmul-bf16-4c");
        launch_attrs.config.task_dim = NPUTaskDim3{4, 1, 1};
        launch_attrs.config.task_mode = NPUTaskMode::kFourCore;
        launch_attrs.timeout_ms = timeout_ms;
        std::printf("[host] launch matmul_bf16_4c: A[%llu,%llu] x "
                    "BT[%llu,%llu] -> C[%llu,%llu], tile=%ux%ux%u, "
                    "slices=%u, cluster-grid=%llux%llu, producer=%s\n",
                    (unsigned long long)m, (unsigned long long)k,
                    (unsigned long long)n, (unsigned long long)k,
                    (unsigned long long)m, (unsigned long long)n,
                    (unsigned)MatmulBf16Shape::kTileM,
                    (unsigned)MatmulBf16Shape::kTileN,
                    (unsigned)MatmulBf16Shape::kTileK,
                    (unsigned)MatmulBf16Shape::kSlices,
                    (unsigned long long)cluster_tiles_m,
                    (unsigned long long)cluster_tiles_n,
                    m < n ? "A" : "B");
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
