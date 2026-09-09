#include <dayan_kernel.h>
#include <stdint.h>

enum : uint64_t {
    kModeD2U = 0,
    kModeRemoteU2D = 1,
    kResultMagic = 0x4c535542574d3031ULL,
    kSampleCount = 5,
    kUramBytes = 1024U * 1024U,
    kSourceBytes = kUramBytes,
    kVerificationBytes = 64U * 1024U,
};

struct LsuBandwidthSample {
    uint64_t bytes;
    uint64_t cycles;
};

struct LsuBandwidthResults {
    uint64_t magic;
    uint64_t mode;
    uint64_t sample_count;
    uint64_t reserved;
    LsuBandwidthSample samples[kSampleCount];
};

#if defined(__DAYAN_DEVICE__)

#include <dayan_intrinsics.h>
#include <dayan_sys.h>

namespace lsu_bandwidth_detail {

enum : uint64_t {
    kClusterCores = 4,
    kUramWindowBase = 0x00800000,
    kUramWindowStride = 2U * 1024U * 1024U,
};

struct alignas(64) ClusterShared {
    dayan_cluster_barrier_t barrier;
    uint64_t uram_base[kClusterCores];
};

static_assert(sizeof(ClusterShared) <= 1024,
              "bandwidth shared state exceeds its allocation");
static_assert((kUramWindowStride & (kUramWindowStride - 1U)) == 0,
              "URAM window stride must be a power of two");

template <long Group>
static inline void configure_contiguous(uint64_t bytes)
{
    long checksum = 0;

    set_lsu_matrix<Group>(checksum, 1, (long)bytes, (long)bytes,
                          (long)bytes);
}

static inline void issue_d2u(uint64_t destination, uint64_t source)
{
    __tx_lsu_dma_d2u<0>((long)destination, (long)source);
}

static inline void issue_u2d(uint64_t destination, uint64_t source)
{
    __tx_lsu_dma_u2d<0>((long)destination, (long)source);
}

static void measure_d2u(const uint8_t *source, uint64_t local_uram,
                        LsuBandwidthResults *results)
{
    static constexpr uint64_t sizes[kSampleCount] = {
        4U * 1024U,
        16U * 1024U,
        64U * 1024U,
        256U * 1024U,
        1024U * 1024U,
    };
    const uint64_t source_base = (uint64_t)(uintptr_t)source;

    for (uint64_t sample = 0; sample < kSampleCount; ++sample) {
        const uint64_t bytes = sizes[sample];

        configure_contiguous<0>(bytes);
        uint64_t begin = dayan_get_time();
        issue_d2u(local_uram, source_base);
        dayan_wait_lsu();
        uint64_t end = dayan_get_time();

        results->samples[sample].bytes = bytes;
        results->samples[sample].cycles = end - begin;
    }
}

static void measure_remote_u2d(uint64_t local_uram, uint64_t remote_uram,
                               LsuBandwidthResults *results)
{
    static constexpr uint64_t sizes[kSampleCount] = {
        4U * 1024U,
        16U * 1024U,
        64U * 1024U,
        256U * 1024U,
        1024U * 1024U,
    };

    for (uint64_t sample = 0; sample < kSampleCount; ++sample) {
        const uint64_t bytes = sizes[sample];

        configure_contiguous<0>(bytes);
        uint64_t begin = dayan_get_time();
        issue_u2d(remote_uram, local_uram);
        dayan_wait_lsu();
        uint64_t end = dayan_get_time();

        results->samples[sample].bytes = bytes;
        results->samples[sample].cycles = end - begin;
    }
}

} // namespace lsu_bandwidth_detail

#endif /* __DAYAN_DEVICE__ */

__kernel__ void lsu_bandwidth(uint64_t mode, const uint8_t *source,
                              uint8_t *verification,
                              LsuBandwidthResults *results)
{
    using namespace lsu_bandwidth_detail;

    const uint32_t lane = dayan_task_id(DAYAN_TASK_X);

    if ((mode != kModeD2U && mode != kModeRemoteU2D) || !source ||
        !verification || !results || !shared || lane >= kClusterCores ||
        dayan_task_mode() != 4U ||
        dayan_task_dim(DAYAN_TASK_X) != kClusterCores ||
        dayan_task_dim(DAYAN_TASK_Y) != 1U ||
        dayan_task_dim(DAYAN_TASK_Z) != 1U)
        dayan_thread_exit(-22);

    void *uram = dayan_uram_base();
    if (!uram || dayan_uram_size() < kUramBytes)
        dayan_thread_exit(-12);

    const uint64_t physical_uram = (uint64_t)(uintptr_t)uram;
    const uint64_t expected_window =
        kUramWindowBase + (uint64_t)lane * kUramWindowStride;
    if (physical_uram < expected_window ||
        physical_uram - expected_window > kUramWindowStride - kUramBytes)
        dayan_thread_exit(-22);

    ClusterShared *cluster = (ClusterShared *)shared;
    cluster->uram_base[lane] = physical_uram;
    int status = dayan_cluster_barrier(kClusterCores, &cluster->barrier);
    if (status < 0)
        dayan_thread_exit(status);

    const uint64_t local_uram =
        physical_uram & (kUramWindowStride - 1U);

    if (lane == 0U) {
        results->magic = kResultMagic;
        results->mode = mode;
        results->sample_count = kSampleCount;
        results->reserved = 0;

        if (mode == kModeRemoteU2D) {
            /* Prepare the U2D source before starting the timed region. */
            configure_contiguous<0>(kUramBytes);
            issue_d2u(local_uram, (uint64_t)(uintptr_t)source);
            dayan_wait_lsu();
            measure_remote_u2d(local_uram, cluster->uram_base[1], results);
        } else {
            measure_d2u(source, local_uram, results);
        }
    }

    status = dayan_cluster_barrier(kClusterCores, &cluster->barrier);
    if (status < 0)
        dayan_thread_exit(status);

    /* Copy the final URAM image out after timing so the Host can validate it. */
    if ((mode == kModeD2U && lane == 0U) ||
        (mode == kModeRemoteU2D && lane == 1U)) {
        configure_contiguous<0>(kVerificationBytes);
        issue_u2d((uint64_t)(uintptr_t)verification, local_uram);
        dayan_wait_lsu();
    }

    status = dayan_cluster_barrier(kClusterCores, &cluster->barrier);
    if (status < 0)
        dayan_thread_exit(status);
}

DAYAN_KERNEL_DESC(lsu_bandwidth, 0, 1024, kUramBytes, 1, 0, 0, 0);

#if !defined(__DAYAN_DEVICE__)

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

extern "C" __host__ char *getenv(const char *name);

__host__ static void preserve_error(int candidate, int *status)
{
    if (*status == 0 && candidate < 0)
        *status = candidate;
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

__host__ static void initialize_source(std::vector<uint8_t> *source)
{
    for (size_t index = 0; index < source->size(); ++index)
        (*source)[index] = (uint8_t)((index * 131U + index / 4096U * 17U +
                                     29U) & 0xffU);
}

__host__ static int print_and_validate_results(
    uint64_t mode, const LsuBandwidthResults &results,
    const std::vector<uint8_t> &source,
    const std::vector<uint8_t> &verification)
{
    const char *name = mode == kModeD2U ? "D2U" : "remote-U2D";
    const size_t expected_offset = 0;

    if (results.magic != kResultMagic || results.mode != mode ||
        results.sample_count != kSampleCount) {
        std::printf("[host] FAIL: %s result header is invalid\n", name);
        return -1;
    }

    double peak = 0.0;
    uint64_t peak_bytes = 0;
    for (uint64_t index = 0; index < kSampleCount; ++index) {
        const LsuBandwidthSample &sample = results.samples[index];
        if (!sample.bytes || !sample.cycles) {
            std::printf("[host] FAIL: %s sample %llu is incomplete\n", name,
                        (unsigned long long)index);
            return -1;
        }
        const double bandwidth =
            (double)sample.bytes / (double)sample.cycles;

        if (bandwidth > peak) {
            peak = bandwidth;
            peak_bytes = sample.bytes;
        }
        std::printf("[host] %-10s size=%7llu B cycles=%9llu "
                    "bandwidth=%7.3f B/c\n",
                    name, (unsigned long long)sample.bytes,
                    (unsigned long long)sample.cycles, bandwidth);
    }

    if (std::memcmp(verification.data(), source.data() + expected_offset,
                    kVerificationBytes) != 0) {
        size_t mismatch = 0;
        while (mismatch < kVerificationBytes &&
               verification[mismatch] == source[expected_offset + mismatch])
            ++mismatch;
        std::printf("[host] FAIL: %s data mismatch at byte %zu\n", name,
                    mismatch);
        return -1;
    }

    std::printf("[host] %s peak=%.3f B/cycle at %llu B "
                "(%.3f GB/s at 1 GHz)\n",
                name, peak, (unsigned long long)peak_bytes, peak);
    std::printf("[host] PASS: %s bandwidth and %llu-byte data check\n",
                name, (unsigned long long)kVerificationBytes);
    return 0;
}

__host__ static int run_mode(
    uint64_t mode, npu_context_t *context, npu_stream_t *stream,
    npu_buffer_t *source_buffer, npu_buffer_t *verification_buffer,
    npu_buffer_t *results_buffer, const std::vector<uint8_t> &source,
    uint32_t timeout_ms)
{
    std::vector<uint8_t> verification(kVerificationBytes, 0);
    LsuBandwidthResults results = {};
    npu_kernel_launch_attrs_t attrs;
    int status = upload(stream, verification_buffer, verification.data(),
                        verification.size(), timeout_ms);

    if (status == 0)
        status = upload(stream, results_buffer, &results, sizeof(results),
                        timeout_ms);
    if (status == 0) {
        npu_kernel_launch_attrs_init(
            &attrs, context, stream,
            mode == kModeD2U ? "lsu-d2u" : "lsu-u2d-rmt");
        attrs.config.task_dim = NPUTaskDim3{4, 1, 1};
        attrs.config.task_mode = NPUTaskMode::kFourCore;
        attrs.timeout_ms = timeout_ms;
        std::printf("[host] launch isolated %s benchmark: one issuer, "
                    "one immediate-wait DMA per size, 4 KiB..1 MiB\n",
                    mode == kModeD2U ? "D2U" : "remote-U2D");
        status = lsu_bandwidth(mode, source_buffer, verification_buffer,
                               results_buffer, &attrs);
    }
    if (status == 0)
        status = download(stream, &results, results_buffer, sizeof(results),
                          timeout_ms);
    if (status == 0)
        status = download(stream, verification.data(), verification_buffer,
                          verification.size(), timeout_ms);
    if (status == 0)
        status = print_and_validate_results(mode, results, source,
                                            verification);
    return status;
}

__host__ static void usage(const char *program)
{
    std::printf("usage: %s [d2u|u2d]\n", program);
    std::printf("  with no argument, run both isolated bandwidth tests\n");
}

__host__ int main(int argc, char **argv)
{
    bool run_d2u = true;
    bool run_u2d = true;

    if (argc == 2 && std::strcmp(argv[1], "--help") == 0) {
        usage(argv[0]);
        return 0;
    }
    if (argc == 2 && std::strcmp(argv[1], "d2u") == 0) {
        run_u2d = false;
    } else if (argc == 2 && std::strcmp(argv[1], "u2d") == 0) {
        run_d2u = false;
    } else if (argc != 1) {
        usage(argv[0]);
        return 2;
    }

    constexpr uint32_t timeout_ms = 60U * 60U * 1000U;
    std::vector<uint8_t> source(kSourceBytes);
    initialize_source(&source);

    npu_runtime_options_t runtime_options;
    npu_runtime_t *runtime = nullptr;
    npu_context_t *context = nullptr;
    npu_stream_t *stream = nullptr;
    npu_buffer_t *source_buffer = nullptr;
    npu_buffer_t *verification_buffer = nullptr;
    npu_buffer_t *results_buffer = nullptr;
    int status = 0;
    const char *socket_path;

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
        status = create_buffer(context, source.size(), &source_buffer);
    if (status == 0)
        status = create_buffer(context, kVerificationBytes,
                               &verification_buffer);
    if (status == 0)
        status = create_buffer(context, sizeof(LsuBandwidthResults),
                               &results_buffer);
    if (status == 0)
        status = upload(stream, source_buffer, source.data(), source.size(),
                        timeout_ms);
    if (status == 0 && run_d2u)
        status = run_mode(kModeD2U, context, stream, source_buffer,
                          verification_buffer, results_buffer, source,
                          timeout_ms);
    if (status == 0 && run_u2d)
        status = run_mode(kModeRemoteU2D, context, stream, source_buffer,
                          verification_buffer, results_buffer, source,
                          timeout_ms);
    if (status < 0)
        std::printf("lsu_bandwidth demo failed: %d\n", status);

    if (results_buffer)
        preserve_error(npu_buffer_destroy(results_buffer), &status);
    if (verification_buffer)
        preserve_error(npu_buffer_destroy(verification_buffer), &status);
    if (source_buffer)
        preserve_error(npu_buffer_destroy(source_buffer), &status);
    if (stream)
        preserve_error(npu_stream_destroy(stream), &status);
    if (context)
        preserve_error(npu_context_destroy(context), &status);
    if (runtime)
        preserve_error(npu_runtime_close(runtime), &status);
    return status < 0 ? 1 : 0;
}

#endif /* !__DAYAN_DEVICE__ */
