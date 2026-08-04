#define main dyperf_command_main
#include "../dyperf.c"
#undef main

int main(int argc, char **argv)
{
    const npu_profile_event_t events[] = {
        { UINT32_C(0xfffffff0), 7, NPU_PROFILE_PROC_WAIT_EVENT, "worker" },
        { UINT32_C(0x00000010), 7, NPU_PROFILE_PROC_READY, "worker" },
        { UINT32_C(0x00000020), 9, NPU_PROFILE_PROC_WAIT_START,
          "late\"child" },
    };
    const npu_profile_batch_info_t batch = {
        0, 3, 3, 0,
    };
    const npu_profile_event_t hart0_events[] = {
        { UINT32_C(100), 1, NPU_PROFILE_PROC_RUNNING, "zeta" },
        { UINT32_C(120), 1, NPU_PROFILE_PROC_WAIT_EVENT, "zeta" },
    };
    const npu_profile_batch_info_t hart0_batch = {
        0, 0, 2, 0,
    };
    struct trace_writer writer;
    char *command[] = {
        "/bin/sh", "-c",
        "test \"$DAYAN_SERVER_SOCKET\" = /tmp/dyperf-test.sock; exit 23",
        0,
    };
    pid_t command_pid;
    int command_status;
    int status;

    if (argc != 2)
        return 2;
    status = trace_writer_open(&writer, argv[1]);
    if (status < 0)
        return 1;
    status = trace_write_batch(&writer, &batch, events);
    if (status == 0)
        status = trace_write_batch(&writer, &hart0_batch, hart0_events);
    if (trace_writer_close(&writer) < 0)
        status = -EIO;
    if (command_start(command, "/tmp/dyperf-test.sock", &command_pid) < 0 ||
        command_wait(command_pid, 0, &command_status) < 0 ||
        command_status != 23)
        status = -EIO;
    return status < 0;
}
