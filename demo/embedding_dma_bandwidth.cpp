#include <dayan_kernel.h>
#include <stdint.h>

enum : uint64_t {
    kDefaultEntryBytes = 128,
    kDefaultWorkingSetBytes = 10880 * kDefaultEntryBytes,
    kMaxEntryBytes = 4096,
    kUramBytes = 768 * 1024,
    kWorkerUramBytes = kUramBytes / 2,
    kDramTileEntries = kWorkerUramBytes / sizeof(uint64_t),
    kGatherDram = 0,
    kScatterDram = 1,
    kGatherUram = 2,
    kScatterUram = 3,
    kCheckUram = 4, // Explicit, untimed D2U verification with U2D readback.
    kResultMagic = 0x454d42444d413031ULL,
};

struct EmbeddingDmaResult {
    uint64_t magic;
    uint64_t mode;
    uint64_t entries;
    uint64_t iterations;
    uint64_t cycles;
    uint64_t worker_entries[2];
    uint64_t entry_bytes;
};

static_assert(sizeof(EmbeddingDmaResult) == 64, "result ABI");

#if defined(__DAYAN_DEVICE__)
#include <dayan_intrinsics.h>
#include <dayan_sys.h>

namespace embedding_dma_detail {

// Group 1 moves only pointers. The final short DMA never accesses beyond
// count * 8 bytes in DRAM; its URAM side has a full 64-byte slot available.
static void load_pointers(long uram, long dram, uint64_t count)
{
    const uint64_t full_bytes = count * sizeof(uint64_t) & ~63ULL;
    const uint64_t tail_bytes = count * sizeof(uint64_t) & 63ULL;
    __tx_lsu_tcsr_write<LSU_GROUPSZ + LSU_REQUESTREADSIZE>(63);
    if (full_bytes) {
        __tx_lsu_tcsr_write<LSU_GROUPSZ + LSU_DIM0>(full_bytes / 64 - 1);
        __tx_lsu_dma_d2u<1>(uram, dram);
    }
    if (tail_bytes) {
        __tx_lsu_tcsr_write<LSU_GROUPSZ + LSU_DIM0>(0);
        __tx_lsu_tcsr_write<LSU_GROUPSZ + LSU_REQUESTREADSIZE>(tail_bytes - 1);
        __tx_lsu_dma_d2u<1>(uram + full_bytes, dram + full_bytes);
    }
    dayan_wait_lsu();
}

struct Worker {
    const uint64_t *pointers;
    uint8_t *packed;
    uint64_t count;
    uint64_t iterations;
    uint64_t mode;
    uint64_t entry_bytes;
    uint64_t lane;
    uint64_t begin;
    uint64_t end;
    uint64_t completed;
};

static void dma_worker(void *argument)
{
    Worker &w = *(Worker *)argument;
    const long table = (long)dayan_uram_base() + w.lane * kWorkerUramBytes;
    const uint64_t entry_bytes = w.entry_bytes;
    const uint64_t uram_tile_entries = kWorkerUramBytes / (entry_bytes + sizeof(uint64_t));
    const uint64_t pointer_bytes = (uram_tile_entries * sizeof(uint64_t) + 63) & ~63ULL;
    const long data = table + pointer_bytes;
    const uint64_t split = (w.count + 1) / 2;
    const uint64_t first_entry = w.lane == 0 ? 0 : split;
    const uint64_t last_entry = w.lane == 0 ? split : w.count;
    const uint64_t iterations = w.iterations;
    const uint64_t mode = w.mode;
    const uint64_t tile_entries = mode >= kGatherUram ? uram_tile_entries : kDramTileEntries;
    const uint64_t *pointers = w.pointers;
    uint8_t *packed_base = w.packed;
    long checksum = 0;
    set_lsu_matrix<0>(checksum, 1, entry_bytes, entry_bytes, entry_bytes);
    __tx_lsu_tcsr_write<LSU_TABLESTRIDE>(entry_bytes);
    set_lsu_matrix<1>(checksum, 1, 64, 64, 64);

    // Split the list evenly so BOTH workers run even when one large pointer
    // buffer can hold the entire input. Each worker owns its half of packed.
    // They share no ready flags, Resources, or producer/consumer handshake.
    w.begin = dayan_get_time();
    uint64_t completed = 0;
    for (uint64_t repeat = 0; repeat < iterations && first_entry < last_entry; ++repeat) {
        for (uint64_t first = first_entry; first < last_entry; first += tile_entries) {
            const uint64_t remaining = last_entry - first;
            const uint64_t count = remaining < tile_entries ? remaining : tile_entries;
            load_pointers(table, (long)(pointers + first), count);
            __tx_lsu_tcsr_write<LSU_TABLESIZE>(count - 1);
            const long packed = (long)packed_base + first * entry_bytes;
            if (mode == kGatherDram)
                __tx_lsu_dma_d2d_g<0>(packed, table);
            else if (mode == kScatterDram)
                __tx_lsu_dma_d2d_s<0>(table, packed);
            else if (mode == kScatterUram)
                __tx_lsu_dma_u2d_s<0>(table, data);
            else
                __tx_lsu_dma_d2u_g<0>(data, table);
            // The LSU still consumes this pointer table until completion.
            // Never overwrite it with the next tile while DMA is in flight.
            dayan_wait_lsu();
            if (mode == kCheckUram) {
                // Diagnostic launch only: never used by timed D2U measurements.
                __tx_lsu_tcsr_write<LSU_GROUPSZ + LSU_REQUESTREADSIZE>(63);
                __tx_lsu_tcsr_write<LSU_GROUPSZ + LSU_DIM0>(count * entry_bytes / 64 - 1);
                __tx_lsu_dma_u2d<1>(packed, data);
                dayan_wait_lsu();
            }
            completed += count;
        }
    }
    w.end = dayan_get_time();
    w.completed = completed;
}

} // namespace embedding_dma_detail
#endif

// Device pointers are absolute 64-bit D-side addresses, not Host pointers or
// row IDs. Each points at a readable/writable entry_bytes-byte region. Keep all
// referenced buffers alive in the same Context until the launch completes.
// 0: D2D gather; 1: D2D scatter; 2: D2U gather and discard; 3: U2D scatter.
// packed is ignored for mode 2 (the Host wrapper may pass the result Buffer).
// Mode 3 preloads one reusable source tile per worker BEFORE starting either
// worker. Subsequent tiles replay it, with no payload DRAM reads during timing.
__kernel__ void embedding_dma_bandwidth(const uint64_t *pointers, uint64_t count,
                                        uint8_t *packed, uint64_t mode,
                                        uint64_t iterations, uint64_t entry_bytes,
                                        EmbeddingDmaResult *result)
{
    using namespace embedding_dma_detail;
    if (!result || !pointers || !packed || mode > kCheckUram || !iterations ||
        (entry_bytes != 64 && entry_bytes != 128 && entry_bytes != 512 &&
         entry_bytes != 2048 && entry_bytes != 4096) ||
        count > UINT64_MAX / entry_bytes / iterations ||
        ((uintptr_t)pointers & 7U) || ((uintptr_t)packed & 63U) ||
        dayan_task_mode() != 1U || dayan_task_dim(DAYAN_TASK_X) != 1U ||
        dayan_task_dim(DAYAN_TASK_Y) != 1U || dayan_task_dim(DAYAN_TASK_Z) != 1U ||
        !dayan_uram_base() || dayan_uram_size() < kUramBytes)
        dayan_thread_exit(-22);

    if (mode == kScatterUram) {
        const uint64_t uram_tile_entries = kWorkerUramBytes / (entry_bytes + sizeof(uint64_t));
        const uint64_t pointer_bytes = (uram_tile_entries * sizeof(uint64_t) + 63) & ~63ULL;
        const uint64_t split = (count + 1) / 2;
        const long uram = (long)dayan_uram_base();
        for (unsigned lane = 0; lane < 2; ++lane) {
            const uint64_t first = lane == 0 ? 0 : split;
            const uint64_t length = lane == 0 ? split : count - split;
            const uint64_t n = length < uram_tile_entries ? length : uram_tile_entries;
            if (!n) continue;
            const uint64_t bytes = n * entry_bytes;
            long checksum = 0;
            set_lsu_matrix<0>(checksum, 1, bytes, bytes, bytes);
            __tx_lsu_dma_d2u<0>(uram + lane * kWorkerUramBytes + pointer_bytes,
                                 (long)(packed + first * entry_bytes));
        }
        // Finish BOTH preloads before creating workers, so neither worker's
        // timed interval contains the other worker's source initialization.
        dayan_wait_lsu();
    }

    Worker workers[2] = {
        {pointers, packed, count, iterations, mode, entry_bytes, 0, 0, 0, 0},
        {pointers, packed, count, iterations, mode, entry_bytes, 1, 0, 0, 0},
    };
    dayan_thread_resources_t resources;
    dayan_thread_resources_init(&resources);
    resources.lsu_tcsr_count = 2;
    int ids[2];
    ids[0] = dayan_create_thread("emb-dma-0", dma_worker, &workers[0], &resources);
    if (ids[0] < 0)
        dayan_thread_exit(ids[0]);
    ids[1] = dayan_create_thread("emb-dma-1", dma_worker, &workers[1], &resources);
    if (ids[1] < 0)
        dayan_thread_exit(ids[1]);
    for (unsigned lane = 0; lane < 2; ++lane) {
        int64_t child_status = 0;
        const int status = dayan_thread_join((uint32_t)ids[lane], &child_status);
        if (status < 0 || child_status < 0)
            dayan_thread_exit(status < 0 ? status : child_status);
    }
    uint64_t begin = UINT64_MAX;
    uint64_t end = 0;
    for (unsigned lane = 0; lane < 2; ++lane) {
        if (workers[lane].completed) {
            if (workers[lane].begin < begin) begin = workers[lane].begin;
            if (workers[lane].end > end) end = workers[lane].end;
        }
    }
    const EmbeddingDmaResult output = {
        kResultMagic, mode, count, iterations, count ? end - begin : 0,
        {workers[0].completed, workers[1].completed}, entry_bytes,
    };
    *result = output;
    dayan_cache_flush(result, sizeof(*result));
}

DAYAN_KERNEL_TASK_DESC(embedding_dma_bandwidth, 1, 2, 0, kUramBytes, 4, 0, 0, 0);

#if !defined(__DAYAN_DEVICE__)
#include "uram_config.h"
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <unordered_set>
#include <vector>

extern "C" __host__ char *getenv(const char *name);

__host__ static int wait_event(npu_event_t *event)
{
    npu_event_info_t info = {};
    const int status = npu_event_wait(event, 3600000, &info);
    const int destroyed = npu_event_destroy(event);
    if (status < 0) return status;
    if (info.status < std::numeric_limits<int>::min() ||
        info.status > std::numeric_limits<int>::max()) return -EIO;
    return info.status < 0 ? (int)info.status : destroyed;
}

__host__ static int transfer(npu_stream_t *stream, npu_buffer_t *buffer,
                            void *data, size_t bytes, bool to_device,
                            uint64_t offset = 0)
{
    npu_event_t *event = nullptr;
    const int status = to_device
        ? npu_memcpy_h2d_async(stream, buffer, offset, data, bytes, &event)
        : npu_memcpy_d2h_async(stream, data, buffer, offset, bytes, &event);
    return status < 0 ? status : wait_event(event);
}

__host__ static void preserve_error(int candidate, int *status)
{
    if (*status == 0 && candidate < 0) *status = candidate;
}

static constexpr double kBytesPerMB = 1024.0 * 1024.0; // Report convention requested by the user.
// Keep the physical 51.2 GB/s baseline fixed while changing display units.
static constexpr double kDramBandwidthBytesPerSecond = 51200000000.0;
static constexpr double kDramBandwidthMBs = kDramBandwidthBytesPerSecond / kBytesPerMB;
static const char *const kModeNames[] = {"D2D Gather", "D2D Scatter", "D2U Gather", "U2D Scatter"};
static constexpr uint64_t kModeOrder[] = {kGatherUram, kScatterUram, kGatherDram, kScatterDram};

static constexpr uint64_t kEntrySizes[] = {64, 128, 512, 2048, 4096};
static const char *const kEntryNames[] = {"64B", "128B", "512B", "2KB", "4KB"};

struct BenchmarkCase {
    const char *name;
    uint64_t entry_bytes;
    uint64_t count;
    EmbeddingDmaResult measurements[4];
    bool passed[4];
};

__host__ static void print_report(FILE *out, const std::vector<BenchmarkCase> &cases,
                                  uint64_t table_bytes, uint64_t working_set_bytes,
                                  uint64_t iterations, uint64_t clock_hz, uint64_t seed)
{
    std::fprintf(out, "# Embedding DMA bandwidth\n\n");
    std::fprintf(out, "Table: %llu B; target payload/pass: %llu B; iterations: %llu; seed: %llu; "
                     "one core, two independent DMA workers, 768 KiB URAM.\n\n",
                 (unsigned long long)table_bytes, (unsigned long long)working_set_bytes,
                 (unsigned long long)iterations, (unsigned long long)seed);
    std::fprintf(out, "Simulation: DRAMSim3; configuration: "
                     "`/workspace/npu-cluster-sim/configs/lpddr5-6400-2ch-32g.ini` (2ch, 32GB).\n\n");
    std::fprintf(out, "Memory baseline: dual-channel LPDDR5-6400, **%.3f MB/s** "
                     "(51,200,000,000 B/s). **1 MB = 1024 KB = 1024 * 1024 B**.\n\n", kDramBandwidthMBs);
    if (clock_hz)
        std::fprintf(out, "Reporting frequency: %llu Hz (%.6g GHz). "
                         "MB/s = B/cycle * Hz / 1048576.\n\n",
                     (unsigned long long)clock_hz, (double)clock_hz / 1e9);
    else
        std::fprintf(out, "CLOCK_HZ=0: MB/s and bandwidth utilization are omitted.\n\n");
    std::fprintf(out, "Cycles cover both workers, including pointer DMA, data DMA, waits and scheduling. "
                     "Setup, warmup, U2D source preload, result writes and verification are excluded.\n\n");
    std::fprintf(out, "## Workload sizes\n\n"
                     "Entries = target payload bytes / granularity, rounded to the nearest whole entry "
                     "(half rounds up). Payload excludes pointers; one shared table is reused across all sizes.\n\n"
                     "| Granularity | Entries/pass | Payload/pass B | Pointers/pass B | URAM entries/worker/tile |\n"
                     "| --- | ---: | ---: | ---: | ---: |\n");
    for (const BenchmarkCase &test : cases)
        std::fprintf(out, "| %s | %llu | %llu | %llu | %llu |\n", test.name,
                     (unsigned long long)test.count,
                     (unsigned long long)(test.count * test.entry_bytes),
                     (unsigned long long)(test.count * sizeof(uint64_t)),
                     (unsigned long long)(kWorkerUramBytes / (test.entry_bytes + sizeof(uint64_t))));
    std::fprintf(out, "\n## Pointer generation\n\n"
                     "For each granularity G, the Host seeds `std::mt19937_64` with `SEED=%llu`, "
                     "then uses `std::uniform_int_distribution<uint64_t>` over all complete rows in "
                     "`[0, floor(TABLE_BYTES / G) - 1]`. Each pointer is `table_device_base + row_id * G`, "
                     "aligned to G bytes. An `unordered_set` rejects duplicate row IDs until the required "
                     "entry count is reached. This samples uniformly without replacement and preserves "
                     "acceptance order, without sorting, address grouping or a fixed address stride.\n\n"
                     "The pointer array itself is contiguous and fetched in DMA batches; its target "
                     "addresses are random. Each size uses its own sample and reuses that sequence across "
                     "all four modes, warmup and timed iterations. The same seed and standard-library "
                     "implementation reproduce each sample; change `SEED` for another sample.\n\n",
                 (unsigned long long)seed);
    std::fprintf(out, "## Data Traffic\n\n"
                     "Payload = entries * granularity * iterations. D2U reads data, U2D writes data, "
                     "and D2D reads and writes each payload byte. Total Read = data reads + pointer reads; "
                     "Total Write = data writes. **Total Utilization = (Total Read + Total Write) / %.3f * 100%%**. "
                     "**Data Utilization = (data read MB/s + data write MB/s) / %.3f * 100%%**, "
                     "excluding pointers; D2D counts both data directions.\n\n",
                 kDramBandwidthMBs, kDramBandwidthMBs);
    std::fprintf(out, "| Granularity | Test | Cycles | Payload B/cycle |");
    if (clock_hz) std::fprintf(out, " Payload MB/s | Total Read MB/s | Total Write MB/s | Total Utilization | Data Utilization |");
    std::fprintf(out, "\n| --- | --- | ---: | ---: |");
    if (clock_hz) std::fprintf(out, " ---: | ---: | ---: | ---: | ---: |");
    std::fprintf(out, "\n");
    for (const BenchmarkCase &test : cases) {
        for (const uint64_t operation : kModeOrder) {
            if (!test.passed[operation]) continue;
            const EmbeddingDmaResult &result = test.measurements[operation];
            const double payload = (double)(result.entries * result.entry_bytes * result.iterations) / (double)result.cycles;
            std::fprintf(out, "| %s | %s | %llu | %.6f |", test.name, kModeNames[operation],
                         (unsigned long long)result.cycles, payload);
            if (clock_hz) {
                const double mbps = payload * (double)clock_hz / kBytesPerMB;
                const double read = operation == kScatterUram ? 0.0 : mbps;
                const double write = operation == kGatherUram ? 0.0 : mbps;
                const double pointer = mbps * sizeof(uint64_t) / result.entry_bytes;
                std::fprintf(out, " %.3f | %.3f | %.3f | %.6f%% | %.6f%% |", mbps, read + pointer, write,
                             (read + pointer + write) / kDramBandwidthMBs * 100.0,
                             (read + write) / kDramBandwidthMBs * 100.0);
            }
            std::fprintf(out, "\n");
        }
    }
    std::fprintf(out, "\n## Pointer Traffic\n\n"
                     "Each mode reads entries * 8 * iterations bytes of pointers via DMA; no pointer writes. "
                     "**Pointer Utilization = pointer MB/s / %.3f * 100%%**. "
                     "Included in Total Read and Total Utilization; excluded from Payload and Data Utilization.\n\n",
                 kDramBandwidthMBs);
    std::fprintf(out, "| Granularity | Test | Pointer bytes | Pointer B/cycle |");
    if (clock_hz) std::fprintf(out, " Pointer MB/s | Pointer Utilization |");
    std::fprintf(out, "\n| --- | --- | ---: | ---: |");
    if (clock_hz) std::fprintf(out, " ---: | ---: |");
    std::fprintf(out, "\n");
    for (const BenchmarkCase &test : cases) {
        for (const uint64_t operation : kModeOrder) {
            if (!test.passed[operation]) continue;
            const EmbeddingDmaResult &result = test.measurements[operation];
            const uint64_t bytes = result.entries * sizeof(uint64_t) * result.iterations;
            const double rate = (double)bytes / (double)result.cycles;
            std::fprintf(out, "| %s | %s | %llu | %.6f |", test.name, kModeNames[operation], (unsigned long long)bytes, rate);
            if (clock_hz) {
                const double mbps = rate * (double)clock_hz / kBytesPerMB;
                std::fprintf(out, " %.3f | %.6f%% |", mbps, mbps / kDramBandwidthMBs * 100.0);
            }
            std::fprintf(out, "\n");
        }
    }
    std::fprintf(out, "\n");
}

__host__ static bool parse_u64(const char *text, uint64_t *value)
{
    if (!text || text[0] < '0' || text[0] > '9') return false;
    char *end = nullptr;
    errno = 0;
    *value = std::strtoull(text, &end, 10);
    return errno == 0 && end != text && *end == '\0';
}

__host__ static void usage(const char *program)
{
    std::printf("usage: %s [MODE [TABLE_BYTES [COUNT128 [ITERATIONS [CLOCK_HZ [SEED]]]]]] "
                "[--granularity all|64|128|512|2048|4096] [--working-set BYTES] [--verify-uram] [--report FILE]\n", program);
    std::printf("defaults: all 20000000000 10880 10 1000000000 1 --granularity all\n"
                "  default sweep: 64B, 128B, 512B, 2KB, 4KB; four modes per size\n"
                "  MODE: all, gather-uram, scatter-uram, gather-dram, scatter-dram, both\n"
                "  aliases: uram=gather-uram, read=gather-dram, write=scatter-dram\n"
                "  both retains the two D2D modes for each selected size\n"
                "  target working set = COUNT128 * 128 B (default 1392640 B); round entries to nearest\n"
                "  --working-set BYTES overrides COUNT128; --granularity also accepts 64B/128B/512B/2KB/4KB\n"
                "  TABLE_BYTES must be a multiple of 64; target must fit table and be >= largest selected size\n"
                "  1 <= entries per size <= 1048576; 2 independent DMA threads; 384 KiB URAM per worker\n"
                "  Markdown uses B/cycle and MB/s, normalized to 1 GHz by default\n"
                "  1 MB = 1024 * 1024 B; memory baseline = %.3f MB/s\n"
                "  Total Read/Write include pointers; Data Utilization excludes pointers\n"
                "  CLOCK_HZ=0 omits MB/s conversion and utilization\n"
                "  --report FILE: Markdown path (default: embedding_dma_bandwidth.md)\n"
                "  --verify-uram adds separate, untimed D2U/U2D payload verification\n", kDramBandwidthMBs);
}

__host__ int main(int argc, char **argv)
{
    if (argc == 2 && std::strcmp(argv[1], "--help") == 0) {
        usage(argv[0]);
        return 0;
    }
    bool verify_uram = false, valid = true;
    const char *report_path = "embedding_dma_bandwidth.md";
    const char *granularity = "all";
    const char *working_set_option = nullptr;
    const char *positional[6] = {};
    unsigned positional_count = 0;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--verify-uram"))
            verify_uram = true;
        else if (!std::strcmp(argv[i], "--report") || !std::strcmp(argv[i], "--granularity") ||
                 !std::strcmp(argv[i], "--working-set")) {
            if (i + 1 == argc || !argv[i + 1][0]) { valid = false; break; }
            const char *option = argv[i];
            const char *value = argv[++i];
            if (!std::strcmp(option, "--report")) report_path = value;
            else if (!std::strcmp(option, "--granularity")) granularity = value;
            else working_set_option = value;
        } else if (positional_count < 6)
            positional[positional_count++] = argv[i];
        else
            valid = false;
    }
    const char *mode = positional_count ? positional[0] : "all";
    bool run_modes[4] = {};
    if (!std::strcmp(mode, "all"))
        for (auto &run : run_modes) run = true;
    else if (!std::strcmp(mode, "both"))
        run_modes[kGatherDram] = run_modes[kScatterDram] = true;
    else if (!std::strcmp(mode, "gather-dram") || !std::strcmp(mode, "read"))
        run_modes[kGatherDram] = true;
    else if (!std::strcmp(mode, "scatter-dram") || !std::strcmp(mode, "write"))
        run_modes[kScatterDram] = true;
    else if (!std::strcmp(mode, "gather-uram") || !std::strcmp(mode, "uram"))
        run_modes[kGatherUram] = true;
    else if (!std::strcmp(mode, "scatter-uram"))
        run_modes[kScatterUram] = true;
    uint64_t options[5] = {20000000000ULL, kDefaultWorkingSetBytes / kDefaultEntryBytes, 10, 1000000000, 1};
    valid = valid && (run_modes[0] || run_modes[1] || run_modes[2] || run_modes[3]);
    for (unsigned i = 1; valid && i < positional_count; ++i)
        valid = parse_u64(positional[i], &options[i - 1]);
    const uint64_t table_bytes = options[0];
    const uint64_t iterations = options[2], clock_hz = options[3], seed = options[4];
    uint64_t working_set_bytes = 0;
    if (working_set_option)
        valid = valid && parse_u64(working_set_option, &working_set_bytes);
    else if (options[1] <= UINT64_MAX / kDefaultEntryBytes)
        working_set_bytes = options[1] * kDefaultEntryBytes;
    else
        valid = false;
    uint64_t selected_size = 0;
    if (std::strcmp(granularity, "all")) {
        for (unsigned i = 0; i < 5; ++i)
            if (!std::strcmp(granularity, kEntryNames[i])) selected_size = kEntrySizes[i];
        if (!selected_size) valid = valid && parse_u64(granularity, &selected_size);
        bool supported = false;
        for (const uint64_t bytes : kEntrySizes) supported |= selected_size == bytes;
        valid = valid && supported;
    }
    if (!valid || table_bytes < 64 || table_bytes % 64 || !working_set_bytes ||
        working_set_bytes > table_bytes || !iterations) {
        usage(argv[0]);
        return 2;
    }
    std::vector<BenchmarkCase> cases;
    uint64_t max_count = 0, max_payload_bytes = 0;
    for (unsigned i = 0; i < 5; ++i) {
        const uint64_t entry_bytes = kEntrySizes[i];
        if (selected_size && selected_size != entry_bytes) continue;
        const uint64_t count = working_set_bytes / entry_bytes +
                               (working_set_bytes % entry_bytes >= entry_bytes / 2);
        if (working_set_bytes < entry_bytes || !count || count > 1048576 ||
            count > table_bytes / entry_bytes || iterations > UINT64_MAX / (count * entry_bytes)) {
            std::fprintf(stderr, "[host] invalid working set/count for granularity %s\n", kEntryNames[i]);
            usage(argv[0]);
            return 2;
        }
        BenchmarkCase test = {};
        test.name = kEntryNames[i];
        test.entry_bytes = entry_bytes;
        test.count = count;
        cases.push_back(test);
        if (count > max_count) max_count = count;
        if (count * entry_bytes > max_payload_bytes) max_payload_bytes = count * entry_bytes;
    }
    const bool need_packed = run_modes[kScatterDram] || run_modes[kScatterUram];
    const bool need_readback = run_modes[kGatherDram] || need_packed ||
                               (run_modes[kGatherUram] && verify_uram);

    npu_runtime_options_t runtime_options;
    npu_runtime_options_init(&runtime_options);
    runtime_options.default_timeout_ms = 3600000;
    const char *socket = getenv("DAYAN_SERVER_SOCKET");
    if (socket && *socket) runtime_options.server_socket = socket;
    npu_runtime_t *runtime = nullptr;
    npu_context_t *context = nullptr;
    npu_stream_t *stream = nullptr;
    enum { Table, Pointers, Packed, Readback, Results, BufferCount };
    npu_buffer_t *buffers[BufferCount] = {};
    const uint64_t sizes[BufferCount] = {
        table_bytes, max_count * 8, need_packed ? max_payload_bytes : 0,
        need_readback ? max_payload_bytes : 0,
        sizeof(EmbeddingDmaResult),
    };
    const char *buffer_names[BufferCount] = {
        "embedding table", "pointers", "packed input", "readback", "results",
    };
    const char *stage = "open Runtime";
    uint64_t capacity = 0;
    int status = npu_runtime_open(&runtime, &runtime_options);
    if (status == 0) {
        stage = "query URAM capacity";
        status = demo_uram_capacity(runtime, kUramBytes, &capacity);
    }
    if (status == 0) {
        stage = "create Context";
        status = npu_context_create(runtime, nullptr, &context);
    }
    if (status == 0) {
        stage = "create Stream";
        status = npu_stream_create(context, nullptr, &stream);
    }
    if (status == 0) {
        std::printf("[host] table=%llu B (%.3f GiB), auxiliary buffers=%llu B before alignment\n",
                    (unsigned long long)table_bytes, (double)table_bytes / 1073741824.0,
                    (unsigned long long)(sizes[Pointers] + sizes[Packed] + sizes[Readback] + sizes[Results]));
        std::fflush(stdout);
    }
    for (unsigned i = 0; status == 0 && i < BufferCount; ++i) {
        if (!sizes[i]) continue;
        stage = buffer_names[i];
        npu_buffer_desc_t desc;
        npu_buffer_desc_init(&desc, sizes[i]);
        if (i == Table) desc.alignment = kMaxEntryBytes;
        status = npu_buffer_create(context, &desc, &buffers[i]);
        if (status < 0) {
            std::fprintf(stderr, "[host] allocation failed: %s, requested=%llu B, status=%d\n",
                         buffer_names[i], (unsigned long long)sizes[i], status);
            if (status == -ENOMEM)
                std::fprintf(stderr,
                    "[host] This allocation uses device DRAM, not URAM. The XDMA Server defaults to 4 GiB.\n"
                    "[host] For the default 20 GB table, use --normal-memory-size-mib 24576 only if the backend\n"
                    "[host] provides that memory; cmodel also needs NPU_CMODEL_MEMORY_MIB=24576.\n"
                    "[host] For a smaller functional test: %s all 1048576 259 2\n", argv[0]);
        }
    }
    uint64_t table_address = 0;
    if (status == 0) {
        stage = "query table Device address";
        status = npu_buffer_get_device_address(buffers[Table], &table_address);
        if (status == -ENOSYS)
            std::fprintf(stderr, "[host] Server lacks the Device address query; rebuild and restart dayan_server.\n");
        if (status == 0 && (table_address & (kMaxEntryBytes - 1))) status = -EIO;
        if (status == 0 && table_address > UINT64_MAX - (table_bytes - 1))
            status = -EOVERFLOW;
    }
    // Reuse the same large table and auxiliary allocations for every size.
    bool any_passed = false;
    for (BenchmarkCase &test : cases) {
        if (status < 0) break;
        const uint64_t entry_bytes = test.entry_bytes, count = test.count;
        const uint64_t payload_bytes = count * entry_bytes;
        const uint64_t words_per_entry = entry_bytes / sizeof(uint32_t);
        const uint64_t uram_tile_entries = kWorkerUramBytes / (entry_bytes + sizeof(uint64_t));
        const uint64_t uram_pointer_bytes = (uram_tile_entries * sizeof(uint64_t) + 63) & ~63ULL;
        const uint64_t uram_data_bytes = uram_tile_entries * entry_bytes;
        // Uniform unique rows over the entire table, retaining acceptance order.
        std::mt19937_64 random(seed);
        std::uniform_int_distribution<uint64_t> row(0, table_bytes / entry_bytes - 1);
        std::unordered_set<uint64_t> selected;
        std::vector<uint64_t> offsets;
        selected.reserve(count);
        offsets.reserve(count);
        while (offsets.size() < count) {
            const uint64_t offset = row(random) * entry_bytes;
            if (selected.insert(offset).second) offsets.push_back(offset);
        }
        std::vector<uint32_t> expected(payload_bytes / sizeof(uint32_t));
        std::vector<uint32_t> observed(need_readback ? expected.size() : 0);
        for (uint64_t i = 0; i < count; ++i)
            for (uint64_t word = 0; word < words_per_entry; ++word)
                expected[i * words_per_entry + word] = 0x3f800000U |
                    (uint32_t)((offsets[i] / entry_bytes * 131 + word * 8191 + seed) & 0x7fffffU);
        // All input data and absolute pointers originate on the Host. Upload only
        // selected rows, keeping the Host footprint proportional to COUNT.
        std::vector<uint64_t> pointers(count);
        for (uint64_t i = 0; status == 0 && i < count; ++i) {
            pointers[i] = table_address + offsets[i];
            stage = "upload table row";
            status = transfer(stream, buffers[Table], expected.data() + i * words_per_entry,
                               entry_bytes, true, offsets[i]);
            if (status < 0)
                std::fprintf(stderr, "[host] table upload failed at entry=%llu offset=%llu\n",
                             (unsigned long long)i, (unsigned long long)offsets[i]);
        }
        if (status == 0) {
            stage = "upload Host-generated Device pointers";
            status = transfer(stream, buffers[Pointers], pointers.data(), count * 8, true);
        }
        npu_kernel_launch_attrs_t attrs;
        npu_kernel_launch_attrs_init(&attrs, context, stream, "embedding-dma");
        attrs.config.task_dim = NPUTaskDim3{1, 1, 1};
        attrs.config.task_mode = NPUTaskMode::kSingleCore;
        attrs.timeout_ms = 3600000;
        if (status == 0) {
            std::printf("[host] granularity=%s, table=%llu B, unique entries=%llu, payload/pass=%llu B, "
                        "iterations=%llu, seed=%llu\n",
                        test.name, (unsigned long long)table_bytes, (unsigned long long)count,
                        (unsigned long long)(payload_bytes), (unsigned long long)iterations,
                        (unsigned long long)seed);
            std::printf("[host] single core; 2 independent DMA workers; %llu KiB URAM/worker\n",
                        (unsigned long long)(kWorkerUramBytes / 1024));
            std::printf("[host] URAM modes: %llu B pointers + %llu B payload per worker, %llu entries/batch\n",
                        (unsigned long long)uram_pointer_bytes, (unsigned long long)uram_data_bytes,
                        (unsigned long long)uram_tile_entries);
            std::printf("[host] D2D modes: %llu B pointers per worker, %llu entries/batch\n",
                        (unsigned long long)kWorkerUramBytes, (unsigned long long)kDramTileEntries);
            std::fflush(stdout);
        }
        for (const uint64_t operation : kModeOrder) {
            if (status < 0) break;
            if (!run_modes[operation]) continue;
            const bool scatter = operation == kScatterDram || operation == kScatterUram;
            const bool check_payload = operation != kGatherUram || verify_uram;
            const uint64_t tile_entries = operation >= kGatherUram ? uram_tile_entries : kDramTileEntries;
            const uint64_t split = (count + 1) / 2;
            const uint64_t batches = (split + tile_entries - 1) / tile_entries +
                                    (count / 2 + tile_entries - 1) / tile_entries;
            std::printf("[host] %s: entries/worker=%llu/%llu, data DMA batches/pass=%llu\n",
                        kModeNames[operation], (unsigned long long)split,
                        (unsigned long long)(count / 2), (unsigned long long)batches);
            std::fflush(stdout);
            if (operation == kScatterUram) {
                // Each worker replays its Host-prepared source tile from URAM.
                // Match that repetition in the full reference, including tails.
                for (uint64_t i = 0; i < count; ++i) {
                    const uint64_t lane = i < split ? 0 : 1;
                    const uint64_t slot = (i - (lane ? split : 0)) % uram_tile_entries;
                    for (uint64_t word = 0; word < words_per_entry; ++word)
                        expected[i * words_per_entry + word] = 0x40000000U |
                            (uint32_t)((slot * 131 + word * 8191 + lane * 104729 + seed) & 0x7fffffU);
                }
            } else if (operation == kScatterDram) {
                // Different from setup, so a skipped scatter cannot pass validation.
                for (auto &word : expected) word ^= 0x00400000U;
            }
            if (scatter) {
                stage = "upload scatter data";
                status = transfer(stream, buffers[Packed], expected.data(), payload_bytes, true);
            }
            // D2U discards payload and needs no packed allocation; its ignored
            // argument still requires a valid Buffer handle for the Runtime ABI.
            npu_buffer_t *packed = operation == kGatherUram ? buffers[Results] :
                                  scatter ? buffers[Packed] : buffers[Readback];
            if (status == 0) {
                stage = "warmup DMA";
                status = embedding_dma_bandwidth(buffers[Pointers], count, packed,
                                                   operation, 1, entry_bytes, buffers[Results], &attrs);
            }
            if (status == 0) {
                stage = "timed DMA";
                status = embedding_dma_bandwidth(buffers[Pointers], count, packed,
                                                   operation, iterations, entry_bytes, buffers[Results], &attrs);
            }
            EmbeddingDmaResult result = {};
            if (status == 0) {
                stage = "read and validate timing results";
                status = transfer(stream, buffers[Results], &result, sizeof(result), false);
            }
            const uint64_t lane0 = (count + 1) / 2 * iterations;
            const uint64_t lane1 = count / 2 * iterations;
            if (status == 0 && (result.magic != kResultMagic || result.mode != operation ||
                result.entries != count || result.entry_bytes != entry_bytes || result.iterations != iterations || !result.cycles ||
                result.worker_entries[0] != lane0 || result.worker_entries[1] != lane1)) status = -EIO;
            if (status == 0 && scatter) {
                stage = "gather scatter results for verification";
                status = embedding_dma_bandwidth(buffers[Pointers], count, buffers[Readback], kGatherDram,
                                                   1, entry_bytes, buffers[Results], &attrs);
            } else if (status == 0 && operation == kGatherUram && verify_uram) {
                stage = "verify D2U payload with separate untimed U2D readback";
                status = embedding_dma_bandwidth(buffers[Pointers], count, buffers[Readback], kCheckUram,
                                                   1, entry_bytes, buffers[Results], &attrs);
            }
            if (status == 0 && check_payload) {
                stage = "read and validate payload";
                status = transfer(stream, buffers[Readback], observed.data(), payload_bytes, false);
            }
            for (size_t i = 0; status == 0 && check_payload && i < expected.size(); ++i) {
                if (observed[i] != expected[i]) {
                    std::fprintf(stderr, "[host] mismatch entry=%zu word=%zu got=%08x expected=%08x\n",
                                 i / words_per_entry, i % words_per_entry, observed[i], expected[i]);
                    status = -EIO;
                }
            }
            // Independent Host reads also check that pointers address the intended
            // table offsets, rather than only checking a scatter/gather round trip.
            for (uint64_t sample = 0; status == 0 && check_payload && sample < 17; ++sample) {
                stage = "read and validate table samples";
                const uint64_t index = sample * (count - 1) / 16;
                uint32_t actual[kMaxEntryBytes / sizeof(uint32_t)];
                status = transfer(stream, buffers[Table], actual, entry_bytes, false, offsets[index]);
                if (status == 0 && std::memcmp(actual, expected.data() + index * words_per_entry, entry_bytes)) {
                    std::fprintf(stderr, "[host] table mismatch at offset=%llu\n",
                                 (unsigned long long)offsets[index]);
                    status = -EIO;
                }
            }
            if (status == 0) {
                test.measurements[operation] = result;
                test.passed[operation] = true;
                any_passed = true;
                if (check_payload)
                    std::printf("[host] PASS: %s, all %llu x %llu FP32 bit patterns and table samples match%s\n",
                                kModeNames[operation], (unsigned long long)count, (unsigned long long)words_per_entry,
                                operation == kGatherUram ? " (separate untimed readback)" : "");
                else
                    std::printf("[host] PASS: %s DMA completion/counts; payload discarded without readback\n",
                                kModeNames[operation]);
            } else {
                std::fprintf(stderr, "[host] failed mode: %s\n", kModeNames[operation]);
            }
        }
        if (status < 0)
            std::fprintf(stderr, "[host] failed granularity: %s\n", test.name);
    }
    if (any_passed)
        print_report(stdout, cases, table_bytes, working_set_bytes, iterations, clock_hz, seed);
    if (status == 0) stage = "release Runtime resources";
    for (int i = BufferCount - 1; i >= 0; --i)
        if (buffers[i]) preserve_error(npu_buffer_destroy(buffers[i]), &status);
    if (stream) preserve_error(npu_stream_destroy(stream), &status);
    if (context) preserve_error(npu_context_destroy(context), &status);
    if (runtime) preserve_error(npu_runtime_close(runtime), &status);
    if (status == 0) {
        stage = "write Markdown report";
        FILE *report = std::fopen(report_path, "w");
        if (!report) {
            status = -errno;
        } else {
            print_report(report, cases, table_bytes, working_set_bytes, iterations, clock_hz, seed);
            if (std::ferror(report)) status = -EIO;
            if (std::fclose(report) != 0) preserve_error(-EIO, &status);
        }
        if (status == 0) std::printf("[host] Markdown report: %s\n", report_path);
        else std::fprintf(stderr, "[host] cannot write report: %s\n", report_path);
    }
    if (status < 0)
        std::fprintf(stderr, "[host] FAIL: embedding DMA at %s: status=%d\n", stage, status);
    return status < 0 ? 1 : 0;
}
#endif
