#define main dyperf_command_main
#include "../dyperf.c"
#undef main

#include <assert.h>

int main(int argc, char **argv)
{
    struct trace_writer writer;
    struct trace_target anchor = { 0 };
    npu_pcu_core_sample_t previous = { .cycle = 1000 };
    npu_pcu_core_sample_t sample;
    double values[DYPERF_PCU_METRIC_COUNT];
    double real;
    unsigned int i;

    assert(argc == 2);
    assert(parse_real("0.1s", 1, &real) == 0 && real == 0.1);
    assert(parse_real("0", 1, &real) == 0 && real == 0);
    assert(parse_real("nan", 1, &real) == -EINVAL);
    assert(parse_real("inf", 1, &real) == -EINVAL);
    assert(parse_real("-0.1", 1, &real) == -EINVAL);
    assert(parse_real("1ms", 1, &real) == -EINVAL);
    assert(parse_real("1s", 0, &real) == -EINVAL);
    anchor.have_pcu = 1;
    anchor.pcu_first_cycle = (UINT64_C(2) << 48) + 1000;
    assert(extend_counter(&anchor, 980) == (UINT64_C(2) << 48) + 980);
    memset(&anchor, 0, sizeof(anchor));
    anchor.have_pcu = 1;
    anchor.pcu_first_cycle = (UINT64_C(2) << 48) + 10;
    assert(extend_counter(&anchor, NPU_PROFILE_COUNTER_MASK - 5) ==
           (UINT64_C(2) << 48) - 6);
    assert(extend_counter(&anchor, 20) == (UINT64_C(2) << 48) + 20);

    for (i = 0; i < NPU_PCU_COUNTER_COUNT; ++i)
        previous.counters[i] = 10000 + i;
    sample = previous;
    sample.cycle = 2000;
    for (i = 0; i < NPU_PCU_URAM_BANK_COUNT; ++i) {
        sample.counters[NPU_PCU_URAM_READ(i)] += i < 16 ? 1000 : 0;
        sample.counters[NPU_PCU_URAM_WRITE(i)] += 250;
    }
    sample.counters[NPU_PCU_MAT_EXEC] += 750;
    sample.counters[NPU_PCU_LSU_READ] += 125;
    sample.counters[NPU_PCU_LSU_WRITE] += 250;
    sample.counters[NPU_PCU_VEC_ISSUE(0)] += 500;
    sample.counters[NPU_PCU_VEC_ISSUE(1)] += 250;
    sample.counters[NPU_PCU_SCALAR_COMMIT] += 1500;
    pcu_metrics(&previous, &sample, 500000000.0, 64, values);
    assert(values[0] == 50 && values[1] == 2500 && values[2] == 50);
    assert(values[3] == 25 && values[4] == 0 && values[5] == 0);
    assert(values[6] == 75 && values[7] == 4 && values[8] == 8);
    assert(values[9] == 0.5 && values[10] == 0.25 && values[11] == 0.75);
    assert(values[12] == 1.5);

    assert(trace_writer_open(&writer, argv[1]) == 0);
    writer.pcu_enabled = 1;
    writer.clock_hz = 500000000.0;
    assert(trace_pcu_sample(&writer, 2, 1, &previous) == 0);
    assert(trace_pcu_sample(&writer, 2, 1, &sample) == 0);
    assert(trace_pcu_sample(&writer, 2, 1, &sample) == 0);
    sample.counters[0]++;
    assert(trace_pcu_sample(&writer, 2, 1, &sample) == -EPROTO);
    sample.counters[0]--;
    sample.cycle = 4000;
    assert(trace_pcu_sample(&writer, 2, 1, &sample) == 0);
    sample.cycle = 3000;
    assert(trace_pcu_sample(&writer, 2, 1, &sample) == -ERANGE);
    assert(trace_writer_close(&writer) == 0);

    sample.cycle = previous.cycle + 1000;
    previous.counters[NPU_PCU_SCALAR_COMMIT] = UINT64_MAX - 100;
    sample.counters[NPU_PCU_SCALAR_COMMIT] = 49;
    pcu_metrics(&previous, &sample, 1e9, 64, values);
    assert(fabs(values[12] - 0.15) < 1e-12);
    return 0;
}
