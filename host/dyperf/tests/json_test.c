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

#define main dyperf_command_main
#include "../dyperf.c"
#undef main

static uint64_t record(uint64_t counter, uint8_t slot, uint8_t state)
{
    return (counter & NPU_PROFILE_COUNTER_MASK) |
           ((uint64_t)slot << NPU_PROFILE_SLOT_SHIFT) |
           ((uint64_t)state << NPU_PROFILE_STATE_SHIFT);
}

static size_t append_define(unsigned char *data, size_t offset,
                            uint64_t counter, uint8_t slot,
                            uint8_t root, const char *name)
{
    uint64_t value = record(counter, slot, NPU_PROFILE_RECORD_SLOT_DEFINE | root);

    memcpy(data + offset, &value, sizeof(value));
    memset(data + offset + sizeof(value), 0, NPU_PROC_NAME_SIZE);
    memcpy(data + offset + sizeof(value), name, strlen(name));
    return offset + NPU_PROFILE_SLOT_DEFINE_SIZE;
}

static size_t append_state(unsigned char *data, size_t offset,
                           uint64_t counter, uint8_t slot, uint8_t state)
{
    uint64_t value = record(counter, slot, state);

    memcpy(data + offset, &value, sizeof(value));
    return offset + NPU_PROFILE_RECORD_SIZE;
}

static int write_chunked(struct trace_writer *writer, uint32_t cluster,
                         uint32_t hart, const unsigned char *data, size_t size)
{
    static const uint8_t chunks[] = { 1, 7, 5, 11, 8, 3, 17, 2, 23 };
    size_t offset = 0;
    size_t chunk_index = 0;

    while (offset < size) {
        npu_profile_batch_info_t batch;
        size_t chunk = chunks[chunk_index++ % sizeof(chunks)];
        int status;

        if (chunk > size - offset)
            chunk = size - offset;
        batch = (npu_profile_batch_info_t){
            cluster, hart, (uint32_t)chunk, 0,
        };
        status = trace_write_batch(writer, &batch, data + offset);
        if (status < 0)
            return status;
        offset += chunk;
    }
    return 0;
}

static int rejects_record(const unsigned char *data, uint32_t size)
{
    npu_profile_batch_info_t batch = { 0, 0, size, 0 };
    struct trace_writer writer;
    int status;

    memset(&writer, 0, sizeof(writer));
    writer.file = tmpfile();
    writer.first = 1;
    writer.clock_hz = DYPERF_DEFAULT_CLOCK_HZ;
    if (!writer.file)
        return 0;
    status = trace_write_batch(&writer, &batch, data);
    (void)trace_writer_close(&writer);
    return status == -EPROTO;
}

static int write_rounds(struct trace_writer *writer, uint32_t cluster)
{
    static const struct {
        uint8_t slot, root;
        const char *name;
    } definitions[] = {
        {1, 1, "z-root"}, {8, 8, "a-root"}, {9, 8, "post"},
        {5, 1, "load-b"}, {2, 1, "load-a"}, {3, 1, "load-a"},
        {2, 1, "load-a"},
    };
    unsigned char data[256];
    size_t size = 0;
    int status;

    size = append_define(data, size, 1, 17, 0, "rpc-receive");
    size = append_define(data, size, 2, 19, 0, "profile-flush");
    status = write_chunked(writer, cluster, 1, data, size);
    if (status < 0)
        return status;
    /* Interleaved families, noncontiguous slots, child reuse and >256 rounds. */
    for (unsigned round = 0; round < 300; ++round) {
        size = 0;
        for (size_t i = 0; i < sizeof(definitions) / sizeof(definitions[0]); ++i) {
            uint64_t cycle = 100 + round * 100 + i;

            size = append_define(data, size, cycle, definitions[i].slot,
                                  definitions[i].root, definitions[i].name);
            size = append_state(data, size, cycle, definitions[i].slot,
                                 NPU_PROFILE_PROC_READY);
        }
        status = write_chunked(writer, cluster, 1, data, size);
        if (status < 0)
            return status;
    }
    return 0;
}

int main(int argc, char **argv)
{
    unsigned char hart3[128];
    unsigned char hart0[40];
    npu_profile_batch_info_t batch;
    struct trace_writer writer;
    char *command[] = {
        "/bin/sh", "-c",
        "test \"$DAYAN_SERVER_SOCKET\" = /tmp/dyperf-test.sock; exit 23",
        0,
    };
    size_t hart3_size = 0;
    size_t hart0_size = 0;
    unsigned char malformed[NPU_PROFILE_SLOT_DEFINE_SIZE] = { 0 };
    uint64_t malformed_header;
    pid_t command_pid;
    int command_status;
    int status;

    if (strcmp(trace_state_name(NPU_PROFILE_PROC_POLLING), "POLLING") ||
        strcmp(trace_state_name(NPU_PROFILE_PROC_WAIT_LSU), "WAIT_LSU") ||
        strcmp(trace_state_name(NPU_PROFILE_PROC_WAIT_MAT), "WAIT_MAT") ||
        strcmp(trace_state_name(NPU_PROFILE_PROC_WAIT_VEC), "WAIT_VEC") ||
        strcmp(trace_state_name(NPU_PROFILE_PROC_WAIT_MACRO), "WAIT_MACRO") ||
        strcmp(trace_state_name(NPU_PROFILE_PROC_YIELD), "YIELD") ||
        strcmp(trace_state_name(NPU_PROFILE_PROC_WAIT_JOIN), "WAIT_JOIN") ||
        strcmp(trace_state_name(NPU_PROFILE_PROC_WAIT_RESOURCE),
               "WAIT_RESOURCE"))
        return 1;

    if (argc != 2)
        return 2;
    hart3_size = append_define(hart3, hart3_size,
                               UINT64_C(0xfffffffffff0), 7, 7, "worker");
    hart3_size = append_state(hart3, hart3_size,
                              UINT64_C(0xfffffffffff0), 7,
                              NPU_PROFILE_PROC_WAIT_EVENT);
    hart3_size = append_state(hart3, hart3_size, UINT64_C(0x10), 7,
                              NPU_PROFILE_PROC_READY);
    hart3_size = append_define(hart3, hart3_size, UINT64_C(0x20), 9, 7,
                               "late\"child");
    hart3_size = append_state(hart3, hart3_size, UINT64_C(0x20), 9,
                              NPU_PROFILE_PROC_WAIT_START);
    hart3_size = append_state(hart3, hart3_size, UINT64_C(0x28), 7,
                              NPU_PROFILE_PROC_RUNNING);
    hart3_size = append_define(hart3, hart3_size, UINT64_C(0x30), 7, 7,
                               "worker-2");
    hart3_size = append_state(hart3, hart3_size, UINT64_C(0x30), 7,
                              NPU_PROFILE_PROC_WAIT_START);
    hart3_size = append_state(hart3, hart3_size, UINT64_C(0x38), 7,
                              NPU_PROFILE_PROC_RUNNING);

    /* Legacy definitions with no family tag remain readable. */
    hart0_size = append_define(hart0, hart0_size, 100, 1, 0, "zeta");
    hart0_size = append_state(hart0, hart0_size, 100, 1,
                              NPU_PROFILE_PROC_RUNNING);
    hart0_size = append_state(hart0, hart0_size, 120, 1,
                              NPU_PROFILE_PROC_WAIT_EVENT);

    status = trace_writer_open(&writer, argv[1]);
    if (status < 0)
        return 1;
    /* Exercise header, name payload, and following state split points. */
    status = write_chunked(&writer, 0, 3, hart3, hart3_size);
    if (status == 0) {
        batch = (npu_profile_batch_info_t){
            0, 0, (uint32_t)hart0_size, 0,
        };
        status = trace_write_batch(&writer, &batch, hart0);
    }
    if (status == 0)
        status = write_rounds(&writer, 0);
    if (status == 0)
        status = write_rounds(&writer, 2);
    if (trace_writer_close(&writer) < 0)
        status = -EIO;
    if (command_start(command, "/tmp/dyperf-test.sock", &command_pid) < 0 ||
        command_wait(command_pid, 0, &command_status) < 0 ||
        command_status != 23)
        status = -EIO;
    malformed_header = record(1, 1, 0x91);
    memcpy(malformed, &malformed_header, sizeof(malformed_header));
    if (!rejects_record(malformed, NPU_PROFILE_RECORD_SIZE))
        status = -EIO;
    malformed_header = record(1, 1, NPU_PROFILE_PROC_READY);
    memcpy(malformed, &malformed_header, sizeof(malformed_header));
    if (!rejects_record(malformed, NPU_PROFILE_RECORD_SIZE))
        status = -EIO;
    malformed_header = record(1, NPU_PROFILE_SLOT_COUNT,
                              NPU_PROFILE_RECORD_SLOT_DEFINE);
    memcpy(malformed, &malformed_header, sizeof(malformed_header));
    memcpy(malformed + NPU_PROFILE_RECORD_SIZE, "invalid", 8);
    if (!rejects_record(malformed, NPU_PROFILE_SLOT_DEFINE_SIZE))
        status = -EIO;
    /* Reject a child whose root has not been defined, and a system family. */
    append_define(malformed, 0, 1, 2, 1, "child");
    if (!rejects_record(malformed, sizeof(malformed)))
        status = -EIO;
    append_define(malformed, 0, 1, 17, 1, "system");
    if (!rejects_record(malformed, sizeof(malformed)))
        status = -EIO;
    return status < 0;
}
