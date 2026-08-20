#include <dayan_kernel.h>

#if defined(__DAYAN_DEVICE__)
#include <dayan_intrinsics.h>
#include <dayan_sys.h>
#endif

#include <stdint.h>

enum : uint64_t {
    kAddElements = 64,
    kAddBytes = kAddElements * sizeof(float),
    kAddUramBytes = 1024,
};

enum VecTcsr : long {
    VEC_CONTROL = 0,
    VEC_SYNC = 1,
    VEC_DIM0 = 2,
    VEC_DIM1 = 3,
    VEC_DIM2 = 4,
    VEC_DIM3 = 5,
    VEC_DIM4 = 6,
    VEC_DIM5 = 7,
    VEC_IMPL_DEF_HINT = 8,
    VEC_V0_BASE = 37,
    VEC_V0_DIM0_STRIDE = 38,
    VEC_V0_DIM1_STRIDE = 39,
    VEC_V0_DIM2_STRIDE = 40,
    VEC_V0_DIM3_STRIDE = 41,
    VEC_V0_DIM4_STRIDE = 42,
    VEC_V0_DIM5_STRIDE = 43,
};

__kernel__ void add_fp32_vec(const float *a, const float *b, float *c)
{
    const uint64_t uram = (uint64_t)(uintptr_t)dayan_uram_base();
    const uint64_t a_uram = uram;
    const uint64_t b_uram = uram + kAddBytes;
    const uint64_t c_uram = uram + 2U * kAddBytes;

    /* LSU group 0: one contiguous 256-byte row. */
    __tx_lsu_tcsr_write<LSU_BLOCK>(0);
    __tx_lsu_tcsr_write<LSU_DIM0>(kAddBytes / 64U - 1U);
    __tx_lsu_tcsr_write<LSU_DIM1>(0);
    __tx_lsu_tcsr_write<LSU_DIM2>(0);
    __tx_lsu_tcsr_write<LSU_INBLOCKSTRIDE>(0);
    __tx_lsu_tcsr_write<LSU_INDIM0STRIDE>(64);
    __tx_lsu_tcsr_write<LSU_INDIM1STRIDE>(kAddBytes);
    __tx_lsu_tcsr_write<LSU_INDIM2STRIDE>(0);
    __tx_lsu_tcsr_write<LSU_OUTBLOCKSTRIDE>(0);
    __tx_lsu_tcsr_write<LSU_OUTDIM0STRIDE>(64);
    __tx_lsu_tcsr_write<LSU_OUTDIM1STRIDE>(kAddBytes);
    __tx_lsu_tcsr_write<LSU_OUTDIM2STRIDE>(0);
    __tx_lsu_tcsr_write<LSU_BLOCKTRANSPOSE>(6);
    __tx_lsu_tcsr_write<LSU_BLOCKPADDING>(0);
    __tx_lsu_tcsr_write<LSU_REQUESTREADSIZE>(63);
    __tx_lsu_tcsr_write<LSU_PADDINGVALUE>(0);
    __tx_lsu_tcsr_write<LSU_URAMWRITESWIZZLEMBS>(0);
    __tx_lsu_tcsr_write<LSU_URAMREADSWIZZLEMBS>(0);
    __tx_lsu_tcsr_write<LSU_REQUESTWRITESIZE>(63);
    __tx_lsu_tcsr_write<LSU_TABLESIZE>(0);
    __tx_lsu_tcsr_write<LSU_TABLESTRIDE>(64);

    /* VEC: one thread, one 64-lane FP32 vector, no address offset. */
    __tx_vec_tcsr_write<VEC_SYNC>(0);
    __tx_vec_tcsr_write<VEC_DIM0>(0);
    __tx_vec_tcsr_write<VEC_DIM1>(0);
    __tx_vec_tcsr_write<VEC_DIM2>(0);
    __tx_vec_tcsr_write<VEC_DIM3>(0);
    __tx_vec_tcsr_write<VEC_DIM4>(0);
    __tx_vec_tcsr_write<VEC_DIM5>(0);
    __tx_vec_tcsr_write<VEC_IMPL_DEF_HINT>(0);
    __tx_vec_tcsr_write<VEC_V0_BASE>(0);
    __tx_vec_tcsr_write<VEC_V0_DIM0_STRIDE>(0);
    __tx_vec_tcsr_write<VEC_V0_DIM1_STRIDE>(0);
    __tx_vec_tcsr_write<VEC_V0_DIM2_STRIDE>(0);
    __tx_vec_tcsr_write<VEC_V0_DIM3_STRIDE>(0);
    __tx_vec_tcsr_write<VEC_V0_DIM4_STRIDE>(0);
    __tx_vec_tcsr_write<VEC_V0_DIM5_STRIDE>(0);

    /* Load A and B from Device memory to URAM. */
    __tx_lsu_dma_d2u<0>((long)a_uram, (long)a);
    __tx_lsu_dma_d2u<0>((long)b_uram, (long)b);
    dayan_wait_lsu();

    /* Calc: load two Vec registers, add, then write the result to URAM. */
    __tx_vec_lsu_u2v<0, 0>((long)a_uram);
    __tx_vec_lsu_u2v<1, 0>((long)b_uram);
    __tx_vec_vv_fp32_add<0, 16, 0, 0>();
    __tx_vec_lsu_v2u<0, 0>((long)c_uram);

    /* Five end fields all equal 4: execute the four queued Vec instructions. */
    __tx_vec_tcsr_write<VEC_CONTROL>(0x0404040404L);
    dayan_wait_vec();

    /* Store C from URAM to Device memory. */
    __tx_lsu_dma_u2d<0>((long)c, (long)c_uram);
    dayan_wait_lsu();
}

DAYAN_KERNEL_LOCKED_TASK_DESC(add_fp32_vec, 0, 0, kAddUramBytes);

#if !defined(__DAYAN_DEVICE__)

#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>

extern "C" __host__ char *getenv(const char *name);

enum : uint32_t {
    kTimeoutMs = 300000,
};

__host__ static int wait_and_destroy(npu_event_t *event)
{
    npu_event_info_t info;
    int status;
    int destroy_status;

    if (!event)
        return -1;
    status = npu_event_wait(event, kTimeoutMs, &info);
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

__host__ static void preserve_error(int candidate, int *status)
{
    if (*status == 0 && candidate < 0)
        *status = candidate;
}

__host__ int main(int argc, char **argv)
{
    float a[kAddElements];
    float b[kAddElements];
    float c[kAddElements] = {};
    npu_runtime_options_t options;
    npu_runtime_t *runtime = nullptr;
    npu_context_t *context = nullptr;
    npu_stream_t *stream = nullptr;
    npu_buffer_t *a_buffer = nullptr;
    npu_buffer_t *b_buffer = nullptr;
    npu_buffer_t *c_buffer = nullptr;
    npu_event_t *event = nullptr;
    const char *socket_path;
    int status = 0;

    if (argc == 2 && std::strcmp(argv[1], "--help") == 0) {
        std::printf("usage: %s\n", argv[0]);
        return 0;
    }
    if (argc != 1)
        return 2;

    for (uint32_t i = 0; i < kAddElements; ++i) {
        a[i] = (float)i * 0.25f - 8.0f;
        b[i] = 3.5f - (float)i * 0.125f;
    }

    npu_runtime_options_init(&options);
    socket_path = getenv("DAYAN_SERVER_SOCKET");
    if (socket_path && *socket_path)
        options.server_socket = socket_path;
    status = npu_runtime_open(&runtime, &options);
    if (status == 0)
        status = npu_context_create(runtime, nullptr, &context);
    if (status == 0)
        status = npu_stream_create(context, nullptr, &stream);
    if (status == 0) {
        npu_buffer_desc_t desc;
        npu_buffer_desc_init(&desc, kAddBytes);
        desc.alignment = 64;
        status = npu_buffer_create(context, &desc, &a_buffer);
        if (status == 0)
            status = npu_buffer_create(context, &desc, &b_buffer);
        if (status == 0)
            status = npu_buffer_create(context, &desc, &c_buffer);
    }
    if (status == 0)
        status = npu_memcpy_h2d_async(stream, a_buffer, 0, a, sizeof(a),
                                      &event);
    if (status == 0) {
        status = wait_and_destroy(event);
        event = nullptr;
    }
    if (status == 0)
        status = npu_memcpy_h2d_async(stream, b_buffer, 0, b, sizeof(b),
                                      &event);
    if (status == 0) {
        status = wait_and_destroy(event);
        event = nullptr;
    }
    if (status == 0) {
        npu_kernel_launch_attrs_t attrs;
        npu_kernel_launch_attrs_init(&attrs, context, stream,
                                     "add-fp32-vec");
        attrs.config.task_dim = NPUTaskDim3{1, 1, 1};
        attrs.config.task_mode = NPUTaskMode::kSingleCore;
        attrs.timeout_ms = kTimeoutMs;
        status = add_fp32_vec(a_buffer, b_buffer, c_buffer, &attrs);
    }
    if (status == 0)
        status = npu_memcpy_d2h_async(stream, c, c_buffer, 0, sizeof(c),
                                      &event);
    if (status == 0) {
        status = wait_and_destroy(event);
        event = nullptr;
    }

    if (status == 0) {
        float max_absolute_error = 0.0f;
        float max_relative_error = 0.0f;
        uint32_t mismatches = 0;

        for (uint32_t i = 0; i < kAddElements; ++i) {
            const float expected = a[i] + b[i];
            const float error = std::fabs(c[i] - expected);
            const float relative_error =
                error / std::fmax(std::fabs(expected), 1.0e-6f);

            if (error > max_absolute_error)
                max_absolute_error = error;
            if (relative_error > max_relative_error)
                max_relative_error = relative_error;
            if (!std::isfinite(c[i]) || error > 1.0e-6f) {
                if (mismatches < 4U)
                    std::printf("mismatch[%u]: actual=%g expected=%g\n",
                                i, c[i], expected);
                ++mismatches;
            }
        }
        std::printf("[host] elements=%llu mismatches=%u max_abs_error=%g "
                    "max_rel_error=%g\n",
                    (unsigned long long)kAddElements, mismatches,
                    max_absolute_error, max_relative_error);
        if (mismatches != 0U)
            status = -1;
    }

    if (status == 0)
        std::printf("[host] PASS: FP32 Add with VEC\n");
    else
        std::printf("[host] FAIL: FP32 Add with VEC, status=%d\n", status);

    if (event)
        preserve_error(npu_event_destroy(event), &status);
    if (c_buffer)
        preserve_error(npu_buffer_destroy(c_buffer), &status);
    if (b_buffer)
        preserve_error(npu_buffer_destroy(b_buffer), &status);
    if (a_buffer)
        preserve_error(npu_buffer_destroy(a_buffer), &status);
    if (stream)
        preserve_error(npu_stream_destroy(stream), &status);
    if (context)
        preserve_error(npu_context_destroy(context), &status);
    if (runtime)
        preserve_error(npu_runtime_close(runtime), &status);
    return status == 0 ? 0 : 1;
}

#endif
