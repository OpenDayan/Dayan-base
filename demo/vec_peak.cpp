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

/*
 * VEC throughput microbenchmark: how close can a kernel get to the unit's
 * designed peak of 128 FLOP/cycle (two 64-lane ALU instructions per cycle)?
 *
 * Every configuration is a register-only FlexVec block: a seed prologue, a
 * loop body of ALU instructions replayed by the hardware loops in each forked
 * task, and one store per task. It runs twice with different outer loop
 * counts; the Host reports the difference, so block emission, VFork, the
 * prologue, ramp-up/drain and the unit wait cancel and only the steady state
 * remains. Operand values are chosen so every task ends on a known constant,
 * which the kernel checks to catch a silently dropped block.
 */

enum VecPeakMode : uint64_t {
    kModeAdd = 0,       /* x = x[d] + 0 */
    kModeMul,           /* x = x[d] * 1 */
    kModeSfu,           /* x = recip(x[d]) */
    kModeAddSfu,        /* add, sfu alternating */
    kModeLoadAdd,       /* u2v, add alternating */
    kModeLoad3Add,      /* u2v, add, add, add */
    kModeLoad,          /* u2v only */
    kModeMul2,          /* x*x, alternating destination queue */
    kModeFlat,          /* add; no hardware loop, one VFork per body */
    kModeAddMul,        /* add, mul alternating */
    kModeAddMul2,       /* add -> queue 0, mul -> queue 1 */
    kModeMulAdd,        /* mul, then an add that consumes it */
    kModeLoadAddMul,    /* u2v, add, mul rotating */
    kModeAddAddMulMul,  /* add, add, mul, mul */
    kModeAddMul1R,      /* add, mul; each reads one register twice */
    kModeAddMulOwn,     /* add on queue 0 only, mul on queue 1 only */
    kModeAddMulConst,   /* add, mul from two fixed constants */
    kModeAddMulConst1,  /* add, mul from one fixed constant */
    kModeAddMul2R,      /* add, mul; each reads two recent registers */
    kModeCount,
};

enum : uint64_t {
    kResultMagic = 0x5645435045414b31ULL,
    kLanes = 64,
    kVectorBytes = kLanes * sizeof(float),
    kMaxTasks = 64,
    kSlotBytes = kMaxTasks * kVectorBytes,
    kSourceBytes = 2U * kSlotBytes,
    kUramBytes = 2U * kSlotBytes,
    kSeedInstructions = 30,
    kMaxBody = 255 - kSeedInstructions - 1,
    kMaxInner = 256,
    kMaxConfigs = 96,
    kMaxCores = 4,
    kPeakFlopPerCycle = 128,
    kOneBits = 0x3f800000,
    kPoisonBits = 0xdeadbeef,
};

struct VecPeakConfig {
    uint64_t mode;
    uint64_t tasks;         /* VEC_DIM2: forked FlexVec tasks */
    uint64_t distance;      /* dependency distance in pushes: 1,2,4,8,15 */
    uint64_t body;          /* loop-body instructions */
    uint64_t inner;         /* VEC_DIM0 count */
    uint64_t outer_short;   /* VEC_DIM1 count; flat: VFork count */
    uint64_t outer_long;
    uint64_t reserved;
};

struct VecPeakSample {
    uint64_t short_cycles;
    uint64_t long_cycles;
    uint64_t bad_slots;
    uint64_t reserved;
};

struct alignas(64) VecPeakResults {
    uint64_t magic;
    uint64_t lane;
    uint64_t config_count;
    uint64_t begin;
    uint64_t end;
    uint64_t reserved[3];
    VecPeakSample samples[kMaxConfigs];
};

static_assert(kUramBytes == 32U * 1024U,
              "vec_peak URAM must be one 32 KiB allocation unit");
static_assert(sizeof(VecPeakResults) % 64U == 0,
              "per-Core results must not share a cache line");

#if defined(__DAYAN_DEVICE__)

#include <dayan_intrinsics.h>
#include <dayan_sys.h>

namespace vec_peak_detail {

/* Vec register operand: queue select in bit 4, age in the low four bits. */
enum : long {
    kQueue0 = 0,
    kQueue1 = 16,
    kSfuRecip = 5,
    kVecChannelBytes = 32,
};

static inline bool zero_chain(uint64_t mode)
{
    return mode == kModeAddMul1R || mode == kModeAddMulOwn ||
           mode == kModeAddMul2R;
}

static inline uint32_t expected_bits(uint64_t mode)
{
    return zero_chain(mode) || mode == kModeAddMulConst1 ? 0U : kOneBits;
}

static inline void configure_lsu(uint64_t bytes)
{
    long checksum = 0;

    set_lsu_matrix<0>(checksum, 1, (long)bytes, (long)bytes, (long)bytes);
}

/* One 256-byte slot per task: only DIM2 advances the address. */
template <long Base> static inline void configure_vec_iterator()
{
    __tx_vec_tcsr_write<Base + 0>(0);
    __tx_vec_tcsr_write<Base + 1>(0);
    __tx_vec_tcsr_write<Base + 2>(0);
    __tx_vec_tcsr_write<Base + 3>(kVectorBytes);
    __tx_vec_tcsr_write<Base + 4>(0);
    __tx_vec_tcsr_write<Base + 5>(0);
    __tx_vec_tcsr_write<Base + 6>(0);
}

static inline void configure_vec(uint64_t outer, uint64_t inner,
                                 uint64_t tasks)
{
    __tx_vec_tcsr_write<VEC_DIM0>((long)inner - 1);
    __tx_vec_tcsr_write<VEC_DIM1>((long)outer - 1);
    __tx_vec_tcsr_write<VEC_DIM2>((long)tasks - 1);
    __tx_vec_tcsr_write<VEC_DIM3>(0);
    __tx_vec_tcsr_write<VEC_DIM4>(0);
    __tx_vec_tcsr_write<VEC_DIM5>(0);
    __tx_vec_tcsr_write<VEC_IMPL_DEF_HINT>(0);
    configure_vec_iterator<VEC_V0_BASE>();
    configure_vec_iterator<VEC_V1_BASE>();
    __tx_vec_tcsr_write<VEC_V0_URAM_CHANNEL_STRIDE>(kVecChannelBytes);
    __tx_vec_tcsr_write<VEC_V1_URAM_CHANNEL_STRIDE>(kVecChannelBytes);
}

static inline long vec_queue_free()
{
    long free_slots;

    __tx_vec_tcsr_read<VEC_CONTROL>(free_slots);
    return free_slots;
}

static inline void wait_vec_queue(long instructions)
{
    while (vec_queue_free() < instructions)
        dayan_yield();
}

/* The five 8-bit segment ends of a fixed-control-flow VFork. */
static inline void fork_vec(long prologue0, long prologue1, long loop,
                            long epilogue1, long epilogue0)
{
    __tx_vec_tcsr_write<VEC_CONTROL>(prologue0 | (prologue1 << 8) |
                                     (loop << 16) | (epilogue1 << 24) |
                                     (epilogue0 << 32));
}

/*
 * kSeedInstructions broadcasts fill both queues so every age the body reads
 * is defined on the first pass. Queue 1 normally ends {.., 1.0, 0.0}: age 0
 * is the add identity and age 1 the mul identity.
 */
static inline void emit_seed(uint64_t mode)
{
    const long queue0 = zero_chain(mode) ? 0 : (long)kOneBits;

    if (mode == kModeAddMul2) {
        for (uint32_t i = 0; i < 15; ++i)
            __tx_vec_lsu_g2v_broadcast32<1, 0>(0);
    } else if (mode == kModeAddMulOwn) {
        for (uint32_t i = 0; i < 15; ++i)
            __tx_vec_lsu_g2v_broadcast32<1, 0>((long)kOneBits);
    } else {
        for (uint32_t i = 0; i < 14; ++i)
            __tx_vec_lsu_g2v_broadcast32<1, 0>((long)kOneBits);
        __tx_vec_lsu_g2v_broadcast32<1, 0>(0);
    }
    for (uint32_t i = 0; i < 15; ++i)
        __tx_vec_lsu_g2v_broadcast32<0, 0>(queue0);
}

/*
 * Exactly `body` instructions. `Age` names the value Distance pushes back in
 * queue 0. vv_fp32_{add,mul}<Vb, Vrs1, Vrs2, Rm> computes Vrs1 OP Vrs2.
 */
template <long Distance>
static inline void emit_body(uint64_t mode, uint64_t body, uint64_t ones)
{
    constexpr long Age = Distance - 1;
    constexpr long Age2 = Age / 2;

    static_assert(Distance >= 1 && Distance <= 15,
                  "age 15 aliases the register being written");
    for (uint64_t i = 0; i < body; ++i) {
        const bool odd = (i & 1U) != 0;

        switch (mode) {
        case kModeMul:
            __tx_vec_vv_fp32_mul<0, kQueue0 + Age, kQueue1 + 1, 0>();
            break;
        case kModeSfu:
            __tx_vec_v_fp32_sfu<0, kQueue0 + Age, kSfuRecip>();
            break;
        case kModeAddSfu:
            if (odd)
                __tx_vec_v_fp32_sfu<0, kQueue0 + Age, kSfuRecip>();
            else
                __tx_vec_vv_fp32_add<0, kQueue0 + Age, kQueue1, 0>();
            break;
        case kModeLoadAdd:
            if (odd)
                __tx_vec_vv_fp32_add<0, kQueue0 + Age, kQueue1, 0>();
            else
                __tx_vec_lsu_u2v<0, 0>((long)ones);
            break;
        case kModeLoad3Add:
            if (i & 3U)
                __tx_vec_vv_fp32_add<0, kQueue0 + Age, kQueue1, 0>();
            else
                __tx_vec_lsu_u2v<0, 0>((long)ones);
            break;
        case kModeLoad:
            __tx_vec_lsu_u2v<0, 0>((long)ones);
            break;
        case kModeMul2:
        case kModeAddMulOwn:
            if (odd)
                __tx_vec_vv_fp32_mul<1, kQueue1 + Age, kQueue1 + Age, 0>();
            else if (mode == kModeMul2)
                __tx_vec_vv_fp32_mul<0, kQueue0 + Age, kQueue0 + Age, 0>();
            else
                __tx_vec_vv_fp32_add<0, kQueue0 + Age, kQueue0 + Age, 0>();
            break;
        case kModeAddMul:
            if (odd)
                __tx_vec_vv_fp32_mul<0, kQueue0 + Age, kQueue1 + 1, 0>();
            else
                __tx_vec_vv_fp32_add<0, kQueue0 + Age, kQueue1, 0>();
            break;
        case kModeAddMul2:
            if (odd)
                __tx_vec_vv_fp32_mul<1, kQueue1 + Age, kQueue1 + Age, 0>();
            else
                __tx_vec_vv_fp32_add<0, kQueue0 + Age, kQueue1, 0>();
            break;
        case kModeMulAdd:
            if (odd)
                __tx_vec_vv_fp32_add<0, kQueue0, kQueue1, 0>();
            else
                __tx_vec_vv_fp32_mul<0, kQueue0 + Age, kQueue1 + 1, 0>();
            break;
        case kModeLoadAddMul:
            if (i % 3U == 0)
                __tx_vec_lsu_u2v<0, 0>((long)ones);
            else if (i % 3U == 1)
                __tx_vec_vv_fp32_add<0, kQueue0 + Age, kQueue1, 0>();
            else
                __tx_vec_vv_fp32_mul<0, kQueue0 + Age, kQueue1 + 1, 0>();
            break;
        case kModeAddAddMulMul:
            if (i & 2U)
                __tx_vec_vv_fp32_mul<0, kQueue0 + Age, kQueue1 + 1, 0>();
            else
                __tx_vec_vv_fp32_add<0, kQueue0 + Age, kQueue1, 0>();
            break;
        case kModeAddMul1R:
            if (odd)
                __tx_vec_vv_fp32_mul<0, kQueue0 + Age, kQueue0 + Age, 0>();
            else
                __tx_vec_vv_fp32_add<0, kQueue0 + Age, kQueue0 + Age, 0>();
            break;
        case kModeAddMulConst:
            if (odd)
                __tx_vec_vv_fp32_mul<0, kQueue1 + 1, kQueue1 + 1, 0>();
            else
                __tx_vec_vv_fp32_add<0, kQueue1, kQueue1, 0>();
            break;
        case kModeAddMulConst1:
            if (odd)
                __tx_vec_vv_fp32_mul<0, kQueue1, kQueue1, 0>();
            else
                __tx_vec_vv_fp32_add<0, kQueue1, kQueue1, 0>();
            break;
        case kModeAddMul2R:
            if (odd)
                __tx_vec_vv_fp32_mul<0, kQueue0 + Age, kQueue0 + Age2, 0>();
            else
                __tx_vec_vv_fp32_add<0, kQueue0 + Age, kQueue0 + Age2, 0>();
            break;
        default: /* kModeAdd, kModeFlat */
            __tx_vec_vv_fp32_add<0, kQueue0 + Age, kQueue1, 0>();
            break;
        }
    }
}

/* One timed run; returns its cycles and counts the wrong task slots. */
template <long Distance>
static uint64_t measure(const VecPeakConfig &config, uint64_t outer,
                        uint64_t ones, uint64_t slots, uint64_t poison,
                        uint64_t *bad_slots)
{
    const long body = (long)config.body;
    const long instructions = kSeedInstructions + body + 1;

    /* Poison the result slots so a dropped block cannot pass. */
    configure_lsu(config.tasks * kVectorBytes);
    __tx_lsu_dma_d2u<0>((long)slots, (long)poison);
    dayan_wait_lsu();

    uint64_t begin;
    if (config.mode == kModeFlat) {
        configure_vec(1, 1, config.tasks);
        begin = dayan_get_time();
        for (uint64_t block = 0; block < outer; ++block) {
            wait_vec_queue(instructions);
            emit_seed(config.mode);
            emit_body<Distance>(config.mode, config.body, ones);
            __tx_vec_lsu_v2u<kQueue0, 1>((long)slots);
            fork_vec(instructions, instructions, instructions, instructions,
                     instructions);
        }
    } else {
        configure_vec(outer, config.inner, config.tasks);
        begin = dayan_get_time();
        wait_vec_queue(instructions);
        emit_seed(config.mode);
        emit_body<Distance>(config.mode, config.body, ones);
        __tx_vec_lsu_v2u<kQueue0, 1>((long)slots);
        fork_vec(kSeedInstructions, kSeedInstructions,
                 kSeedInstructions + body, kSeedInstructions + body,
                 instructions);
    }
    dayan_wait_vec();
    const uint64_t end = dayan_get_time();

    const uint32_t expected = expected_bits(config.mode);
    for (uint64_t task = 0; task < config.tasks; ++task) {
        const volatile uint32_t *slot = (const volatile uint32_t *)
            (uintptr_t)(slots + task * kVectorBytes);

        if (slot[0] != expected || slot[kLanes - 1] != expected)
            ++*bad_slots;
    }
    return end - begin;
}

template <long Distance>
static void run_config(const VecPeakConfig &config, uint64_t ones,
                       uint64_t slots, uint64_t poison, VecPeakSample *sample)
{
    uint64_t bad_slots = 0;

    sample->short_cycles = measure<Distance>(
        config, config.outer_short, ones, slots, poison, &bad_slots);
    sample->long_cycles = measure<Distance>(
        config, config.outer_long, ones, slots, poison, &bad_slots);
    sample->bad_slots = bad_slots;
    sample->reserved = 0;
}

static bool valid_config(const VecPeakConfig &config)
{
    const uint64_t d = config.distance;

    return config.mode < kModeCount && config.tasks >= 1 &&
           config.tasks <= kMaxTasks && config.body >= 1 &&
           config.body <= kMaxBody && config.inner >= 1 &&
           config.inner <= kMaxInner && config.outer_short >= 1 &&
           config.outer_long > config.outer_short &&
           (d == 1 || d == 2 || d == 4 || d == 8 || d == 15);
}

static void dispatch_config(const VecPeakConfig &config, uint64_t ones,
                            uint64_t slots, uint64_t poison,
                            VecPeakSample *sample)
{
    switch (config.distance) {
    case 1:
        run_config<1>(config, ones, slots, poison, sample);
        break;
    case 2:
        run_config<2>(config, ones, slots, poison, sample);
        break;
    case 4:
        run_config<4>(config, ones, slots, poison, sample);
        break;
    case 8:
        run_config<8>(config, ones, slots, poison, sample);
        break;
    default:
        run_config<15>(config, ones, slots, poison, sample);
        break;
    }
}

} // namespace vec_peak_detail

#endif /* __DAYAN_DEVICE__ */

/*
 * source holds kSlotBytes of 1.0f followed by kSlotBytes of poison. Task X
 * runs the whole table on its own Core and fills results[X].
 */
__kernel__ void vec_peak(const VecPeakConfig *configs, uint64_t config_count,
                         const uint32_t *source, VecPeakResults *results)
{
    using namespace vec_peak_detail;

    const uint32_t lane = dayan_task_id(DAYAN_TASK_X);
    const uint32_t lanes = dayan_task_dim(DAYAN_TASK_X);

    if (!configs || !source || !results || !config_count ||
        config_count > kMaxConfigs || lane >= kMaxCores ||
        (lanes != 1U && lanes != kMaxCores) ||
        dayan_task_dim(DAYAN_TASK_Y) != 1U ||
        dayan_task_dim(DAYAN_TASK_Z) != 1U)
        dayan_thread_exit(-22);
    for (uint64_t index = 0; index < config_count; ++index) {
        if (!valid_config(configs[index]))
            dayan_thread_exit(-22);
    }

    void *uram = dayan_uram_base();
    if (!uram || dayan_uram_size() < kUramBytes)
        dayan_thread_exit(-12);

    const uint64_t ones = (uint64_t)(uintptr_t)uram;
    const uint64_t slots = ones + kSlotBytes;
    const uint64_t poison = (uint64_t)(uintptr_t)source + kSlotBytes;
    const long queue_free = vec_queue_free();
    VecPeakResults *result = &results[lane];
    VecPeakConfig warmup = {};
    uint64_t warmup_bad = 0;

    configure_lsu(kSlotBytes);
    __tx_lsu_dma_d2u<0>((long)ones, (long)source);
    dayan_wait_lsu();

    /* One small untimed block absorbs the first VFork's cold-start cost. */
    warmup.mode = kModeAdd;
    warmup.tasks = 16;
    warmup.body = 32;
    warmup.inner = 1;
    (void)measure<15>(warmup, 1, ones, slots, poison, &warmup_bad);

    result->begin = dayan_get_time();
    for (uint64_t index = 0; index < config_count; ++index)
        dispatch_config(configs[index], ones, slots, poison,
                        &result->samples[index]);
    result->end = dayan_get_time();
    result->magic = kResultMagic;
    result->lane = lane;
    result->config_count = config_count;
    dayan_cache_flush(result, sizeof(*result));

    /*
     * The Vec instruction queue outlives this Proc: an instruction left
     * unforked would run at the head of the next kernel's first VFork.
     */
    if (vec_queue_free() != queue_free)
        dayan_thread_exit(-5);
}

/*
 * One Task per Core, no child Thread; one LSU and one VEC TCSR group. Not a
 * locked Task, because the same kernel also launches in four-Core mode: run
 * it on an otherwise idle Server.
 */
DAYAN_KERNEL_TASK_DESC(vec_peak, 1, 0, 0, kUramBytes, 1, 0, 1, 0);

#if !defined(__DAYAN_DEVICE__)

#include "uram_config.h"

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <vector>

extern "C" __host__ char *getenv(const char *name);

struct ModeName {
    uint64_t mode;
    const char *name;
    const char *what;
};

static const ModeName kModeNames[] = {
    {kModeAdd, "add", "x = x[d] + 0"},
    {kModeMul, "mul", "x = x[d] * 1"},
    {kModeSfu, "sfu", "x = recip(x[d])"},
    {kModeAddSfu, "add-sfu", "add, sfu alternating"},
    {kModeLoadAdd, "ld-add", "u2v, add alternating"},
    {kModeLoad3Add, "ld-3add", "u2v, add, add, add"},
    {kModeLoad, "ld", "u2v only (no ALU work)"},
    {kModeMul2, "mul2", "x*x, alternating destination queue"},
    {kModeFlat, "flat", "add without hardware loops, one VFork per body"},
    {kModeAddMul, "add-mul", "add, mul alternating"},
    {kModeAddMul2, "add-mul2", "add -> queue 0, mul -> queue 1"},
    {kModeMulAdd, "mul-add", "mul, then an add that consumes it"},
    {kModeLoadAddMul, "ld-add-mul", "u2v, add, mul rotating"},
    {kModeAddAddMulMul, "aamm", "add, add, mul, mul"},
    {kModeAddMul1R, "am-1r", "add, mul; x OP x, one register read twice"},
    {kModeAddMulOwn, "am-own", "add on queue 0 only, mul on queue 1 only"},
    {kModeAddMulConst, "am-k", "add, mul from two fixed constants"},
    {kModeAddMulConst1, "am-k1", "add, mul from one fixed constant"},
    {kModeAddMul2R, "am-2r", "add, mul; two different recent registers"},
};

static_assert(sizeof(kModeNames) / sizeof(kModeNames[0]) == kModeCount,
              "every mode needs a name");

__host__ static const char *mode_name(uint64_t mode)
{
    return mode < kModeCount ? kModeNames[mode].name : "?";
}

/* ALU instructions in one body; loads are data movement, not FLOPs. */
__host__ static uint64_t alu_per_body(uint64_t mode, uint64_t body)
{
    switch (mode) {
    case kModeLoadAdd:
        return body / 2;
    case kModeLoad3Add:
        return body - (body + 3) / 4;
    case kModeLoadAddMul:
        return body - (body + 2) / 3;
    case kModeLoad:
        return 0;
    default:
        return body;
    }
}

/* Body executions summed over the tasks of one run. */
__host__ static uint64_t bodies(const VecPeakConfig &config, uint64_t outer)
{
    return config.mode == kModeFlat ? config.tasks * outer :
                                      config.tasks * config.inner * outer;
}

/* Size the loop so the long run adds about `work` body instructions. */
__host__ static VecPeakConfig make_config(uint64_t mode, uint64_t tasks,
                                          uint64_t distance, uint64_t body,
                                          uint64_t work)
{
    VecPeakConfig config = {};
    const uint64_t per_pass = tasks * body;
    uint64_t passes = (work + per_pass - 1) / per_pass;

    if (passes < 1)
        passes = 1;
    config.mode = mode;
    config.tasks = tasks;
    config.distance = distance;
    config.body = body;
    if (mode == kModeFlat) {
        config.inner = 1;
        config.outer_short = 2;
        config.outer_long = 2 + passes;
    } else {
        config.inner = passes > kMaxInner ? (uint64_t)kMaxInner : passes;
        config.outer_short = 1;
        config.outer_long = 2;
    }
    return config;
}

struct SweepBuilder {
    std::vector<VecPeakConfig> *configs;
    uint64_t work;
};

__host__ static void add(const SweepBuilder &sweep, uint64_t mode,
                         uint64_t tasks, uint64_t distance,
                         uint64_t body = 192)
{
    sweep.configs->push_back(
        make_config(mode, tasks, distance, body, sweep.work));
}

__host__ static bool append_sweep(const char *name, uint64_t work,
                                  std::vector<VecPeakConfig> *configs)
{
    const SweepBuilder sweep = {configs, work};

    if (std::strcmp(name, "summary") == 0) {
        for (uint64_t tasks : {1, 8, 16})
            add(sweep, kModeAdd, tasks, 15);
        add(sweep, kModeMul, 16, 15);
        add(sweep, kModeSfu, 16, 15);
        add(sweep, kModeLoadAdd, 16, 15);
        add(sweep, kModeAddMul, 16, 15);
        add(sweep, kModeAddMul2R, 16, 8);
        add(sweep, kModeAddMul1R, 16, 15);
        add(sweep, kModeFlat, 16, 15, kMaxBody);
    } else if (std::strcmp(name, "threads") == 0) {
        for (uint64_t tasks : {1, 2, 4, 8, 16, 32, 64})
            add(sweep, kModeAdd, tasks, 15);
    } else if (std::strcmp(name, "depend") == 0) {
        for (uint64_t distance : {1, 2, 4, 8, 15})
            add(sweep, kModeAdd, 1, distance);
        for (uint64_t tasks : {8, 16}) {
            add(sweep, kModeAdd, tasks, 1);
            add(sweep, kModeAdd, tasks, 4);
        }
        add(sweep, kModeSfu, 1, 1);
        add(sweep, kModeSfu, 1, 15);
    } else if (std::strcmp(name, "ops") == 0) {
        for (uint64_t tasks : {16, 64}) {
            for (uint64_t mode : {kModeAdd, kModeMul, kModeSfu, kModeAddSfu,
                                  kModeLoadAdd, kModeLoad3Add, kModeLoad})
                add(sweep, mode, tasks, 15);
        }
        add(sweep, kModeMul2, 16, 8);
    } else if (std::strcmp(name, "body") == 0) {
        for (uint64_t body : {8, 32, 192})
            add(sweep, kModeAdd, 16, 8, body);
    } else if (std::strcmp(name, "flat") == 0) {
        for (uint64_t tasks : {1, 16, 64})
            add(sweep, kModeFlat, tasks, 15, kMaxBody);
    } else if (std::strcmp(name, "mix") == 0) {
        add(sweep, kModeAdd, 16, 15);
        add(sweep, kModeMul, 16, 15);
        for (uint64_t tasks : {1, 8, 16, 32, 64})
            add(sweep, kModeAddMul, tasks, 15);
        add(sweep, kModeAddMul, 16, 1);
        for (uint64_t tasks : {16, 64}) {
            add(sweep, kModeAddMul2, tasks, 8);
            add(sweep, kModeMulAdd, tasks, 15);
            add(sweep, kModeLoadAddMul, tasks, 15);
        }
    } else if (std::strcmp(name, "reads") == 0) {
        add(sweep, kModeAddMul, 16, 15);
        add(sweep, kModeAddAddMulMul, 16, 15);
        for (uint64_t distance : {1, 4, 8, 15})
            add(sweep, kModeAddMul1R, 16, distance);
        add(sweep, kModeAddMul1R, 32, 15);
        for (uint64_t distance : {2, 8, 15})
            add(sweep, kModeAddMul2R, 16, distance);
        add(sweep, kModeAddMul2R, 32, 15);
        add(sweep, kModeAddMulOwn, 16, 8);
        add(sweep, kModeAddMulConst, 16, 15);
        add(sweep, kModeAddMulConst1, 16, 15);
    } else if (std::strcmp(name, "all") == 0) {
        for (const char *each : {"threads", "depend", "ops", "body", "flat",
                                 "mix", "reads"})
            append_sweep(each, work, configs);
    } else {
        return false;
    }
    return true;
}

__host__ static int parse_u64(const char *text, uint64_t *value)
{
    char *end;

    if (!text || text[0] < '0' || text[0] > '9')
        return -1;
    errno = 0;
    *value = std::strtoull(text, &end, 10);
    return errno || *end ? -1 : 0;
}

/* MODE[:TASKS[:DISTANCE[:BODY]]], e.g. am-1r:16:15. */
__host__ static bool append_spec(const char *spec, uint64_t work,
                                 std::vector<VecPeakConfig> *configs)
{
    char text[64];
    uint64_t fields[3] = {16, 15, 0};
    uint64_t mode = kModeCount;
    uint32_t count = 0;

    if (std::strlen(spec) >= sizeof(text))
        return false;
    std::strcpy(text, spec);
    char *cursor = std::strchr(text, ':');
    if (cursor)
        *cursor++ = '\0';
    for (const ModeName &entry : kModeNames) {
        if (std::strcmp(text, entry.name) == 0)
            mode = entry.mode;
    }
    if (mode == kModeCount)
        return false;
    while (cursor) {
        char *next = std::strchr(cursor, ':');

        if (next)
            *next++ = '\0';
        if (count == 3 || parse_u64(cursor, &fields[count++]) < 0)
            return false;
        cursor = next;
    }

    const uint64_t tasks = fields[0];
    const uint64_t distance = fields[1];
    const uint64_t body =
        fields[2] ? fields[2] : mode == kModeFlat ? (uint64_t)kMaxBody : 192;

    if (!tasks || tasks > kMaxTasks || !body || body > kMaxBody ||
        (distance != 1 && distance != 2 && distance != 4 && distance != 8 &&
         distance != 15))
        return false;
    configs->push_back(make_config(mode, tasks, distance, body, work));
    return true;
}

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
    desc.alignment = 64;
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

struct Rate {
    uint64_t alu;       /* ALU instructions added by the long run */
    double cycles;      /* cycles added by the long run */
    double flop;        /* FLOP/cycle; 0 when the timing is unusable */
};

__host__ static Rate steady_rate(const VecPeakConfig &config,
                                 const VecPeakSample &sample)
{
    Rate rate = {};

    rate.alu = bodies(config, config.outer_long - config.outer_short) *
               alu_per_body(config.mode, config.body);
    rate.cycles = (double)sample.long_cycles - (double)sample.short_cycles;
    if (rate.cycles > 0)
        rate.flop = (double)rate.alu * kLanes / rate.cycles;
    return rate;
}

__host__ static int print_and_validate_results(
    const std::vector<VecPeakConfig> &configs,
    const std::vector<VecPeakResults> &results)
{
    const uint64_t cores = results.size();
    int status = 0;

    for (uint64_t core = 0; core < cores; ++core) {
        const VecPeakResults &result = results[core];

        if (result.magic != kResultMagic || result.lane != core ||
            result.config_count != configs.size()) {
            std::printf("[host] FAIL: core %llu result header is invalid\n",
                        (unsigned long long)core);
            return -1;
        }
        std::printf("[host] core=%llu sweep cycles %llu..%llu (%llu)\n",
                    (unsigned long long)core,
                    (unsigned long long)result.begin,
                    (unsigned long long)result.end,
                    (unsigned long long)(result.end - result.begin));
    }

    for (size_t index = 0; index < configs.size(); ++index) {
        const VecPeakConfig &config = configs[index];
        double cluster_alu = 0.0;
        double slowest = 0.0;
        bool timed = true;

        for (uint64_t core = 0; core < cores; ++core) {
            const VecPeakSample &sample = results[core].samples[index];
            const Rate rate = steady_rate(config, sample);

            std::printf("[host] core=%llu %-10s tasks=%2llu dist=%2llu "
                        "body=%3llu short=%9llu long=%9llu ",
                        (unsigned long long)core, mode_name(config.mode),
                        (unsigned long long)config.tasks,
                        (unsigned long long)config.distance,
                        (unsigned long long)config.body,
                        (unsigned long long)sample.short_cycles,
                        (unsigned long long)sample.long_cycles);
            if (rate.cycles > 0)
                std::printf("alu/cycle=%5.3f flop/cycle=%6.1f peak=%5.1f%% ",
                            rate.flop / kLanes, rate.flop,
                            100.0 * rate.flop / kPeakFlopPerCycle);
            else
                std::printf("timing unusable (long <= short) ");
            std::printf("%s\n", sample.bad_slots ? "BAD" : "ok");
            if (sample.bad_slots) {
                std::printf("[host] FAIL: core %llu %s left %llu task "
                            "slot(s) without the expected value\n",
                            (unsigned long long)core, mode_name(config.mode),
                            (unsigned long long)sample.bad_slots);
                status = -1;
            }
            if (rate.cycles <= 0)
                timed = false;
            cluster_alu += (double)rate.alu;
            if (rate.cycles > slowest)
                slowest = rate.cycles;
        }
        if (cores > 1 && timed) {
            const double flop = cluster_alu * kLanes / slowest;

            std::printf("[host] cluster %-10s tasks=%2llu dist=%2llu "
                        "flop/cycle=%6.1f peak=%5.1f%% of %llu\n",
                        mode_name(config.mode),
                        (unsigned long long)config.tasks,
                        (unsigned long long)config.distance, flop,
                        100.0 * flop / (double)(cores * kPeakFlopPerCycle),
                        (unsigned long long)(cores * kPeakFlopPerCycle));
        }
    }
    return status;
}

__host__ static void usage(const char *program)
{
    std::printf("usage: %s [--cores 1|4] [--work N] [SWEEP|CONFIG]...\n",
                program);
    std::printf("  --cores 4  one sweep on each Core of a four-Core Cluster\n");
    std::printf("  --work N   body instructions added by each long run "
                "(default 16384; try 1024 on cmodel)\n");
    std::printf("SWEEP (default: summary):\n");
    std::printf("  summary  headline of every result below\n");
    std::printf("  threads  add, 1..64 tasks: does the task grid fill it?\n");
    std::printf("  depend   add/sfu vs dependency distance: latency\n");
    std::printf("  ops      add, mul, sfu, loads at 16 and 64 tasks\n");
    std::printf("  body     add with an 8/32/192-instruction loop body\n");
    std::printf("  flat     no hardware loops, one VFork per body\n");
    std::printf("  mix      add and mul interleaved: do they co-issue?\n");
    std::printf("  reads    add+mul operand patterns: register-file limits\n");
    std::printf("  all      every sweep except summary\n");
    std::printf("CONFIG: MODE[:TASKS[:DISTANCE[:BODY]]], default 16:15:192;\n");
    std::printf("  TASKS 1..64, DISTANCE 1|2|4|8|15, BODY 1..%llu. MODE:\n",
                (unsigned long long)kMaxBody);
    for (const ModeName &entry : kModeNames)
        std::printf("  %-10s %s\n", entry.name, entry.what);
    std::printf("Designed peak: %llu FLOP/cycle per Core. Cycle figures are "
                "only meaningful on RTL or hardware.\n",
                (unsigned long long)kPeakFlopPerCycle);
}

__host__ int main(int argc, char **argv)
{
    std::vector<VecPeakConfig> configs;
    uint64_t cores = 1;
    uint64_t work = 16384;
    bool named = false;

    if (argc == 2 && std::strcmp(argv[1], "--help") == 0) {
        usage(argv[0]);
        return 0;
    }
    /* Options first: every sweep is sized with the final --work value. */
    for (int index = 1; index < argc; ++index) {
        uint64_t *value = nullptr;

        if (std::strcmp(argv[index], "--cores") == 0)
            value = &cores;
        else if (std::strcmp(argv[index], "--work") == 0)
            value = &work;
        else
            continue;
        if (index + 1 >= argc || parse_u64(argv[++index], value) < 0) {
            usage(argv[0]);
            return 2;
        }
    }
    for (int index = 1; index < argc; ++index) {
        if (std::strcmp(argv[index], "--cores") == 0 ||
            std::strcmp(argv[index], "--work") == 0) {
            ++index;
            continue;
        }
        named = true;
        if (!append_sweep(argv[index], work, &configs) &&
            !append_spec(argv[index], work, &configs)) {
            usage(argv[0]);
            return 2;
        }
    }
    if (!named)
        append_sweep("summary", work, &configs);
    if ((cores != 1 && cores != kMaxCores) || !work || configs.empty() ||
        configs.size() > kMaxConfigs) {
        usage(argv[0]);
        return 2;
    }

    /* An RTL sweep runs for minutes to hours; do not time it out. */
    constexpr uint32_t timeout_ms = 24U * 60U * 60U * 1000U;
    std::vector<uint32_t> source(kSourceBytes / sizeof(uint32_t));
    std::vector<VecPeakResults> results(cores);

    for (size_t index = 0; index < source.size(); ++index)
        source[index] = index < source.size() / 2 ? (uint32_t)kOneBits :
                                                    (uint32_t)kPoisonBits;

    npu_runtime_options_t runtime_options;
    npu_runtime_topology_t topology = {};
    npu_runtime_t *runtime = nullptr;
    npu_context_t *context = nullptr;
    npu_stream_t *stream = nullptr;
    npu_buffer_t *config_buffer = nullptr;
    npu_buffer_t *source_buffer = nullptr;
    npu_buffer_t *results_buffer = nullptr;
    const size_t config_bytes = configs.size() * sizeof(configs[0]);
    const size_t result_bytes = results.size() * sizeof(results[0]);
    const char *socket_path;
    int status = 0;

    npu_runtime_options_init(&runtime_options);
    runtime_options.default_timeout_ms = timeout_ms;
    socket_path = getenv("DAYAN_SERVER_SOCKET");
    if (socket_path && *socket_path)
        runtime_options.server_socket = socket_path;

    uint64_t uram_capacity = 0;
    status = npu_runtime_open(&runtime, &runtime_options);
    if (status == 0)
        status = demo_uram_capacity(runtime, kUramBytes, &uram_capacity);
    if (status == 0)
        status = npu_runtime_get_topology(runtime, &topology);
    if (status == 0 && cores == kMaxCores &&
        topology.enabled_core_mask != 0xf) {
        std::fprintf(stderr, "vec_peak --cores 4 requires a Server with "
                             "four enabled Cores\n");
        status = -EINVAL;
    }
    if (status == 0)
        status = npu_context_create(runtime, nullptr, &context);
    if (status == 0)
        status = npu_stream_create(context, nullptr, &stream);
    if (status == 0)
        status = create_buffer(context, config_bytes, &config_buffer);
    if (status == 0)
        status = create_buffer(context, kSourceBytes, &source_buffer);
    if (status == 0)
        status = create_buffer(context, result_bytes, &results_buffer);
    if (status == 0)
        status = upload(stream, config_buffer, configs.data(), config_bytes,
                        timeout_ms);
    if (status == 0)
        status = upload(stream, source_buffer, source.data(), kSourceBytes,
                        timeout_ms);
    if (status == 0)
        status = upload(stream, results_buffer, results.data(), result_bytes,
                        timeout_ms);
    if (status == 0) {
        npu_kernel_launch_attrs_t attrs;

        npu_kernel_launch_attrs_init(&attrs, context, stream, "vec-peak");
        attrs.config.task_dim = NPUTaskDim3{(uint32_t)cores, 1, 1};
        attrs.config.task_mode = cores == kMaxCores ?
            NPUTaskMode::kFourCore : NPUTaskMode::kSingleCore;
        attrs.timeout_ms = timeout_ms;
        std::printf("[host] launch vec_peak: %zu config(s) on %llu Core(s), "
                    "%llu body instructions per long run, peak %llu "
                    "FLOP/cycle per Core\n",
                    configs.size(), (unsigned long long)cores,
                    (unsigned long long)work,
                    (unsigned long long)kPeakFlopPerCycle);
        std::fflush(stdout);
        status = vec_peak(config_buffer, configs.size(), source_buffer,
                          results_buffer, &attrs);
    }
    if (status == 0)
        status = download(stream, results.data(), results_buffer,
                          result_bytes, timeout_ms);
    if (status == 0)
        status = print_and_validate_results(configs, results);

    if (status == 0)
        std::printf("[host] PASS: VEC throughput, %zu config(s) on %llu "
                    "Core(s), every task slot checked\n",
                    configs.size(), (unsigned long long)cores);
    else
        std::printf("[host] FAIL: VEC throughput, status=%d\n", status);

    if (results_buffer)
        preserve_error(npu_buffer_destroy(results_buffer), &status);
    if (source_buffer)
        preserve_error(npu_buffer_destroy(source_buffer), &status);
    if (config_buffer)
        preserve_error(npu_buffer_destroy(config_buffer), &status);
    if (stream)
        preserve_error(npu_stream_destroy(stream), &status);
    if (context)
        preserve_error(npu_context_destroy(context), &status);
    if (runtime)
        preserve_error(npu_runtime_close(runtime), &status);
    return status == 0 ? 0 : 1;
}

#endif /* !__DAYAN_DEVICE__ */
