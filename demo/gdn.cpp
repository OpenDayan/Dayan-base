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
#ifndef GDN_SOFTWARE_PIPELINE
#define GDN_SOFTWARE_PIPELINE 0
#endif
#ifndef GDN_VEC_TGMT
#define GDN_VEC_TGMT 1
#endif
#include "gdn_chunk.h"

#if GDN_SOFTWARE_PIPELINE && GDN_VEC_TGMT
#define GDN_KERNEL gdn_sw_vec_tgmt
#elif GDN_SOFTWARE_PIPELINE
#define GDN_KERNEL gdn_sw
#else
#define GDN_KERNEL gdn_tgmt
#endif

__kernel__ void GDN_KERNEL(const DayanGatedDeltaNetConfig *config,
                          const float *q, const float *k, const float *v,
                          const float *g, const float *beta,
                          const float *state, float *output)
{
    dayan_cache_invalidate(config, sizeof(*config));
    const bool four_core = dayan_task_mode() == 4;
    auto *barrier = (dayan_cluster_barrier_t *)shared;
    if (four_core) gdn_detail::gdn_check(dayan_cluster_barrier(4, barrier));
    const long status = gdn_detail::run_chunked(config, q, k, v, g, beta, state, output);
    if (four_core) gdn_detail::gdn_check(dayan_cluster_barrier(4, barrier));
    gdn_detail::gdn_check(status);
}

#if GDN_SOFTWARE_PIPELINE && GDN_VEC_TGMT
DAYAN_KERNEL_TASK_DESC(gdn_sw_vec_tgmt, 1, 0, 64, kGdnUramBytes, 1, 1, 1, 0);
#elif GDN_SOFTWARE_PIPELINE
DAYAN_KERNEL_TASK_DESC(gdn_sw, 1, 0, 64, kGdnUramBytes, 1, 1, 1, 0);
#else
DAYAN_KERNEL_NAMED_RESOURCE_TASK_DESC(gdn_tgmt, 1, 2, 64, kGdnUramBytes, 3, 2, 3, 0, 4, 0);
#endif

#if !defined(__DAYAN_DEVICE__)
#include "uram_config.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

extern "C" __host__ char *getenv(const char *name);
constexpr size_t kHead = 128;
constexpr uint32_t kTimeoutMs = 7200000;
constexpr const char *kName = !GDN_SOFTWARE_PIPELINE ? "gdn-tgmt" :
                             GDN_VEC_TGMT ? "gdn-sw-vec-tgmt" : "gdn-sw";

__host__ static void usage(const char *program)
{
    std::printf("usage: %s [tokens heads sequences [cores [tasks]]]\n", program);
    std::printf("  defaults: 1024 1 1; head_dim=128, chunk=64, scalar gate, one final state\n");
    std::printf("  cores=1|4, tasks=groups (default 1); barrier-only independent head/sequence chains\n");
    std::printf("  %s; GDN_OUTPUT optionally saves FP32 output and state\n",
                !GDN_SOFTWARE_PIPELINE ? "TGMT loader + prepare + root; VEC TGMT" :
                GDN_VEC_TGMT ? "single Proc software pipeline; VEC TGMT" :
                               "single Proc software pipeline; one VEC hardware task at a time");
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

__host__ static void initialize(std::vector<float> data[7], size_t rows)
{
    std::mt19937 generator(20261028U);
    std::uniform_real_distribution<float> qk(-1.0f, 1.0f), value(-0.25f, 0.25f);
    std::uniform_real_distribution<float> gate(-2.0f, -0.05f), beta(0.1f, 0.9f), state(-0.02f, 0.02f);
    for (int tensor = 0; tensor < 2; ++tensor) {
        for (float &x : data[tensor]) x = qk(generator);
        for (size_t row = 0; row < rows; ++row) {
            float sum = 0;
            for (size_t d = 0; d < kHead; ++d) {
                const float x = data[tensor][row * kHead + d];
                sum += x * x;
            }
            const float scale = 1.0f / std::sqrt(sum + 1.0e-6f);
            for (size_t d = 0; d < kHead; ++d) data[tensor][row * kHead + d] *= scale;
        }
    }
    for (float &x : data[2]) x = value(generator);
    for (float &x : data[3]) x = gate(generator);
    for (float &x : data[4]) x = beta(generator);
    for (float &x : data[5]) x = state(generator);
    std::fill(data[6].begin(), data[6].end(), std::numeric_limits<float>::quiet_NaN());
}

__host__ static void reference(const std::vector<float> data[7], std::vector<float> &expected,
                              size_t tokens, size_t heads, size_t sequences)
{
    const size_t attn = tokens * heads * sequences * kHead;
    const float scale = 1.0f / std::sqrt(float(kHead));
    std::copy(data[5].begin(), data[5].end(), expected.begin() + attn);
    for (size_t sequence = 0; sequence < sequences; ++sequence) {
        for (size_t head = 0; head < heads; ++head) {
            float *s = expected.data() + attn + (sequence * heads + head) * kHead * kHead;
            for (size_t token = 0; token < tokens; ++token) {
                const size_t row = (sequence * tokens + token) * heads + head;
                const float *q = data[0].data() + row * kHead;
                const float *k = data[1].data() + row * kHead;
                const float *v = data[2].data() + row * kHead;
                const float decay = std::exp(data[3][row]);
                for (size_t j = 0; j < kHead * kHead; ++j) s[j] *= decay;
                for (size_t j = 0; j < kHead; ++j) {
                    float dot = 0;
                    for (size_t i = 0; i < kHead; ++i) dot += s[j * kHead + i] * k[i];
                    const float delta = (v[j] - dot) * data[4][row];
                    for (size_t i = 0; i < kHead; ++i) s[j * kHead + i] += delta * k[i];
                    dot = 0;
                    for (size_t i = 0; i < kHead; ++i) dot += s[j * kHead + i] * q[i];
                    expected[row * kHead + j] = dot * scale;
                }
            }
        }
    }
}

__host__ static int validate(const std::vector<float> &actual, const std::vector<float> &expected, size_t attn)
{
    int status = 0;
    for (int part = 0; part < 2; ++part) {
        const size_t start = part ? attn : 0;
        const size_t end = part ? actual.size() : attn;
        const float atol = part ? 2.0e-4f : 2.0e-5f;
        constexpr float rtol = 3.0e-3f;
        float max_abs = 0, max_rel = 0;
        double squared_error = 0, squared_reference = 0;
        size_t failures = 0;
        for (size_t i = start; i < end; ++i) {
            const float error = std::abs(actual[i] - expected[i]);
            const float rel = error / std::max(std::abs(expected[i]), 1.0e-6f);
            if (!std::isfinite(actual[i]) || error > atol + rtol * std::abs(expected[i])) {
                if (failures < 4) std::printf("[host] mismatch %s[%zu]: got=%g expected=%g\n",
                                            part ? "state" : "output", i - start, actual[i], expected[i]);
                ++failures;
            }
            max_abs = std::max(max_abs, error);
            max_rel = std::max(max_rel, rel);
            squared_error += double(error) * error;
            squared_reference += double(expected[i]) * expected[i];
        }
        const double l2 = std::sqrt(squared_error / std::max(squared_reference, 1.0e-30));
        std::printf("[host] %s elements=%zu max_abs=%g max_rel=%g relative_l2=%g failures=%zu\n",
                    part ? "state" : "output", end - start, max_abs, max_rel, l2, failures);
        if (failures || !std::isfinite(l2) || l2 > rtol) status = -1;
    }
    std::printf("[host] %s: all %zu output/state elements checked against FP32 recurrence\n",
                status ? "FAIL" : "PASS", actual.size());
    return status;
}

__host__ int main(int argc, char **argv)
{
    uint64_t tokens = 1024, heads = 1, sequences = 1, cores = 1, tasks = 1;
    if (argc == 2 && std::strcmp(argv[1], "--help") == 0) { usage(argv[0]); return 0; }
    if ((argc != 1 && (argc < 4 || argc > 6)) ||
        (argc >= 4 && (parse_dimension(argv[1], &tokens) || parse_dimension(argv[2], &heads) ||
                       parse_dimension(argv[3], &sequences))) ||
        (argc >= 5 && parse_dimension(argv[4], &cores)) ||
        (argc >= 6 && parse_dimension(argv[5], &tasks)) ||
        (cores != 1 && cores != 4) || tasks > UINT32_MAX / cores) { usage(argv[0]); return 2; }
    if (tokens < 16 || tokens > 16384 || heads > 64 || sequences > 16 ||
        tokens * heads * sequences * kHead > 16U * 1024U * 1024U) {
        std::fprintf(stderr, "unsupported shape: 16<=tokens<=16384, heads<=64, sequences<=16, tensors<=64MiB\n");
        return 2;
    }
    const size_t rows = tokens * heads * sequences;
    const size_t attn = rows * kHead;
    const size_t states = heads * sequences * kHead * kHead;
    std::vector<float> data[7];
    for (int i = 0; i < 3; ++i) data[i].resize(attn);
    data[3].resize(rows);
    data[4].resize(rows);
    data[5].resize(states);
    data[6].resize(attn + states);
    std::vector<float> expected(attn + states);
    initialize(data, rows);
    reference(data, expected, tokens, heads, sequences);

    DayanGatedDeltaNetConfig config = {};
    config.head_dim = kHead;
    config.heads = config.q_heads = config.k_heads = heads;
    config.tokens = tokens;
    config.sequences = config.q_sequences = config.k_sequences = sequences;
    config.snapshots = config.gate_width = 1;
    const float scale = 1.0f / std::sqrt(float(kHead));
    std::memcpy(&config.scale_bits, &scale, sizeof(scale));
    for (uint64_t *nb : {config.q_nb, config.k_nb, config.v_nb, config.g_nb, config.beta_nb}) {
        const uint64_t width = (nb == config.g_nb || nb == config.beta_nb) ? 1 : kHead;
        nb[0] = sizeof(float);
        nb[1] = width * sizeof(float);
        nb[2] = heads * nb[1];
        nb[3] = tokens * nb[2];
    }

    npu_runtime_options_t options;
    npu_runtime_options_init(&options);
    options.default_timeout_ms = kTimeoutMs;
    const char *socket = getenv("DAYAN_SERVER_SOCKET");
    if (socket && *socket) options.server_socket = socket;
    npu_runtime_t *runtime = nullptr;
    npu_context_t *context = nullptr;
    npu_stream_t *stream = nullptr;
    npu_buffer_t *buffers[8] = {};
    uint64_t capacity = 0;
    int status = npu_runtime_open(&runtime, &options);
    if (status == 0) status = demo_uram_capacity(runtime, kGdnUramBytes, &capacity);
    if (status == 0) status = npu_context_create(runtime, nullptr, &context);
    if (status == 0) status = npu_stream_create(context, nullptr, &stream);
    if (status == 0) status = create_buffer(context, sizeof(config), &buffers[0]);
    if (status == 0) status = upload(stream, buffers[0], &config, sizeof(config), kTimeoutMs);
    for (int i = 0; i < 7 && status == 0; ++i) {
        status = create_buffer(context, data[i].size() * sizeof(float), &buffers[i + 1]);
        if (status == 0) status = upload(stream, buffers[i + 1], data[i].data(), data[i].size() * sizeof(float), kTimeoutMs);
    }
    if (status == 0) {
        npu_kernel_launch_attrs_t attrs;
        npu_kernel_launch_attrs_init(&attrs, context, stream, kName);
        attrs.config.task_dim = cores == 4 ? NPUTaskDim3{4, (uint32_t)tasks, 1}
                                          : NPUTaskDim3{(uint32_t)tasks, 1, 1};
        attrs.config.task_mode = cores == 4 ? NPUTaskMode::kFourCore : NPUTaskMode::kSingleCore;
        std::printf("[host] cores=%llu groups=%llu, independent chains, inter-core barrier only\n",
                    (unsigned long long)cores, (unsigned long long)tasks);
        attrs.timeout_ms = kTimeoutMs;
        std::printf("[host] launch %s: tokens=%llu heads=%llu sequences=%llu head_dim=128 chunk=64 chunks=%llu seed=20261028\n",
                    kName, (unsigned long long) tokens, (unsigned long long) heads,
                    (unsigned long long) sequences, (unsigned long long) ((tokens + 63) / 64 * heads * sequences));
        std::fflush(stdout);
        status = GDN_KERNEL(buffers[0], buffers[1], buffers[2], buffers[3], buffers[4], buffers[5], buffers[6], buffers[7], &attrs);
    }
    if (status == 0) status = download(stream, data[6].data(), buffers[7], data[6].size() * sizeof(float), kTimeoutMs);
    const char *dump = getenv("GDN_OUTPUT");
    if (status == 0 && dump && *dump) {
        FILE *file = std::fopen(dump, "wb");
        if (!file) status = -1;
        else {
            const size_t written = std::fwrite(data[6].data(), sizeof(float), data[6].size(), file);
            const int closed = std::fclose(file);
            if (written != data[6].size() || closed) status = -1;
        }
    }
    if (status == 0) status = validate(data[6], expected, attn);
    for (int i = 7; i >= 0; --i) if (buffers[i]) preserve_error(npu_buffer_destroy(buffers[i]), &status);
    if (stream) preserve_error(npu_stream_destroy(stream), &status);
    if (context) preserve_error(npu_context_destroy(context), &status);
    if (runtime) preserve_error(npu_runtime_close(runtime), &status);
    if (status) std::fprintf(stderr, "gdn demo failed: %d\n", status);
    return status ? 1 : 0;
}
#endif
