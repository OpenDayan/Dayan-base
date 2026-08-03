#define main dyperf_command_main
#include "../dyperf.c"
#undef main

int main(int argc, char **argv)
{
    const npu_profile_event_t events[] = {
        { UINT32_C(0xfffffff0), 7, NPU_PROFILE_PROC_WAIT_EVENT },
        { UINT32_C(0x00000010), 7, NPU_PROFILE_PROC_READY },
    };
    const npu_profile_batch_info_t batch = {
        2, 3, 2, 0,
    };
    struct trace_writer writer;
    int status;

    if (argc != 2)
        return 2;
    status = trace_writer_open(&writer, argv[1]);
    if (status < 0)
        return 1;
    status = trace_write_batch(&writer, &batch, events);
    if (trace_writer_close(&writer) < 0)
        status = -EIO;
    return status < 0;
}
