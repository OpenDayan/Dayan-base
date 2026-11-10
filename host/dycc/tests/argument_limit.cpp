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
#if defined(__DAYAN_DEVICE__)
#include <dayan_sys.h>
#endif

__kernel__ void argument_limit(
    uint64_t a0, uint64_t a1, uint64_t a2, uint64_t a3,
    uint64_t a4, uint64_t a5, uint64_t a6, uint64_t a7,
    uint64_t a8, uint64_t a9, uint64_t a10, uint64_t a11,
    uint64_t a12, uint64_t a13, uint64_t a14, uint64_t a15,
    uint64_t a16, uint64_t a17, uint64_t a18, uint64_t a19,
    uint64_t a20, uint64_t a21, uint64_t a22, uint64_t a23,
    uint64_t a24, uint64_t a25, uint64_t a26, uint64_t a27,
    uint64_t a28, uint64_t a29, uint64_t a30, uint64_t *out
#if defined(TEST_33_ARGS)
    , uint64_t extra
#endif
)
{
    out += dayan_task_id(DAYAN_TASK_X) * 31;
    out[0] = a0;
    out[1] = a1;
    out[2] = a2;
    out[3] = a3;
    out[4] = a4;
    out[5] = a5;
    out[6] = a6;
    out[7] = a7;
    out[8] = a8;
    out[9] = a9;
    out[10] = a10;
    out[11] = a11;
    out[12] = a12;
    out[13] = a13;
    out[14] = a14;
    out[15] = a15;
    out[16] = a16;
    out[17] = a17;
    out[18] = a18;
    out[19] = a19;
    out[20] = a20;
    out[21] = a21;
    out[22] = a22;
    out[23] = a23;
    out[24] = a24;
    out[25] = a25;
    out[26] = a26;
    out[27] = a27;
    out[28] = a28;
    out[29] = a29;
    out[30] = a30;
}
DAYAN_KERNEL_DESC(argument_limit, 0, 0, 0, 0, 0, 0, 0);

#if !defined(__DAYAN_DEVICE__) && defined(TEST_CMODEL)
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
extern "C" __host__ char *getenv(const char *name);

__host__ int main(void)
{
    npu_runtime_options_t options;
    npu_runtime_t *runtime;
    npu_context_t *context;
    npu_stream_t *stream;
    npu_buffer_t *buffer;
    npu_buffer_desc_t desc;
    npu_kernel_launch_attrs_t attrs;
    npu_runtime_options_init(&options);
    if (const char *socket = getenv("DAYAN_SERVER_SOCKET"))
        options.server_socket = socket;
    assert(npu_runtime_open(&runtime, &options) == 0);
    assert(npu_context_create(runtime, nullptr, &context) == 0);
    assert(npu_stream_create(context, nullptr, &stream) == 0);
    npu_buffer_desc_init(&desc, 1024);
    desc.alignment = 64;
    assert(npu_buffer_create(context, &desc, &buffer) == 0);
    npu_kernel_launch_attrs_init(&attrs, context, stream, "args32");
    attrs.config.task_dim = NPUTaskDim3{4, 1, 1};
    attrs.config.task_mode = NPUTaskMode::kFourCore;
    attrs.timeout_ms = 30000;
    for (uint64_t round = 0; round < 3; ++round) {
        uint64_t base = 0xfedcba9876543200ULL + round * 32;
        assert(argument_limit(
            base + 0, base + 1, base + 2, base + 3, base + 4, base + 5, base + 6, base + 7,
            base + 8, base + 9, base + 10, base + 11, base + 12, base + 13, base + 14, base + 15,
            base + 16, base + 17, base + 18, base + 19, base + 20, base + 21, base + 22, base + 23,
            base + 24, base + 25, base + 26, base + 27, base + 28, base + 29, base + 30, buffer, &attrs) == 0);
        uint64_t result[4 * 31];
        npu_event_t *event;
        npu_event_info_t info;
        assert(npu_memcpy_d2h_async(stream, result, buffer, 0,
                                    sizeof(result), &event) == 0);
        assert(npu_event_wait(event, 30000, &info) == 0 && info.status == 0);
        assert(npu_event_destroy(event) == 0);
        for (unsigned i = 0; i < 4 * 31; ++i)
            assert(result[i] == base + i % 31);
    }
    assert(npu_buffer_destroy(buffer) == 0);
    assert(npu_stream_destroy(stream) == 0);
    assert(npu_context_destroy(context) == 0);
    assert(npu_runtime_close(runtime) == 0);
    puts("PASS: 32 arguments on four Harts, three launches");
    return 0;
}
#endif
