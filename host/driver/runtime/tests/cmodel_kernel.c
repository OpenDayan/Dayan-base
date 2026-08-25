#include <stdint.h>

#include <dayan_sys.h>

#define TEST_FAILURE (~(uint64_t)0)
#define TEST_CHILD_THREADS 15U
#define LOCK_TEST_CHILD_THREADS 2U
#define LOCK_TEST_RESULT 0x1234567800003000ULL
#define NORMAL_TEST_RESULT 0x1234567800004000ULL

struct lock_test_results {
    volatile uint64_t release;
    uint64_t release_padding[7];
    uint64_t root_begin;
    uint64_t child_begin[LOCK_TEST_CHILD_THREADS];
    uint64_t child_end[LOCK_TEST_CHILD_THREADS];
    volatile uint64_t started_mask;
    uint64_t family_end;
    uint64_t normal_begin;
};

struct lock_child_work {
    struct lock_test_results *results;
    uint32_t index;
};

struct child_work {
    uint64_t *output;
    uint64_t *uram;
    uint64_t *uram_tail;
    uint64_t value;
};

static uint64_t read_mcycle(void)
{
    uint64_t cycle;

    __asm__ volatile("csrr %0, mcycle" : "=r"(cycle) : : "memory");
    return cycle;
}

static void locked_child(void *argument)
{
    struct lock_child_work *work = argument;
    struct lock_test_results *results = work->results;
    uint64_t bit = 1ULL << work->index;

    results->child_begin[work->index] = read_mcycle();
    results->started_mask |= bit;
    dayan_cache_flush((unsigned char *)results + 64U, 64U);
    for (;;) {
        dayan_cache_invalidate((const void *)&results->release,
                               sizeof(results->release));
        if (results->release)
            break;
        dayan_yield();
    }
    results->child_end[work->index] = read_mcycle();
    dayan_cache_flush((unsigned char *)results + 64U, 64U);
    dayan_thread_exit((int64_t)(LOCK_TEST_RESULT + work->index));
}

static void child_write(void *argument)
{
    struct child_work *work = argument;

    /* Ensure the root can return before its child has completed. */
    dayan_yield();
    work->uram[0] = work->value;
    *work->uram_tail = work->value ^ 0x55aa55aa55aa55aaULL;
    __asm__ volatile("fence rw, rw" ::: "memory");
    *work->output =
        work->uram[0] == work->value &&
        *work->uram_tail ==
            (work->value ^ 0x55aa55aa55aa55aaULL) ?
        work->value : TEST_FAILURE;
    dayan_thread_exit((int64_t)work->value);
}

static void nested_write(void *argument)
{
    struct child_work *work = argument;
    int64_t result = -1;
    int child = dayan_create_thread("child-write", child_write, work, 0);

    if (child < 0 || dayan_thread_join((uint32_t)child, &result) < 0 ||
        result != (int64_t)work->value)
        dayan_thread_exit(-1);
    dayan_thread_exit(result);
}

void grid_kernel(uint64_t *output, uint64_t base)
{
    uint32_t x = dayan_task_id(DAYAN_TASK_X);
    uint32_t y = dayan_task_id(DAYAN_TASK_Y);
    uint32_t z = dayan_task_id(DAYAN_TASK_Z);
    uint32_t dim_x = dayan_task_dim(DAYAN_TASK_X);
    uint32_t dim_y = dayan_task_dim(DAYAN_TASK_Y);
    uint64_t linear = x + (uint64_t)dim_x * (y + (uint64_t)dim_y * z);
    uint64_t expected = base + (uint64_t)dayan_task_mode() * 100U + linear;
    uint64_t *uram = dayan_uram_base();
    struct child_work *work;
    int children[TEST_CHILD_THREADS];
    int64_t child_result;

    if (!uram || dayan_uram_size() != 0x180000U ||
        ((uintptr_t)uram & (DAYAN_URAM_ALLOCATION_GRANULE - 1U))) {
        output[linear] = TEST_FAILURE;
        return;
    }
    if (dayan_task_mode() == 4U) {
        dayan_cluster_barrier_t *barrier =
            (dayan_cluster_barrier_t *)shared;
        volatile uint64_t *published =
            (volatile uint64_t *)((unsigned char *)shared +
                                  sizeof(*barrier));

        if (x == 0U)
            *published = base;
        if (dayan_cluster_barrier(4, barrier) < 0 || *published != base) {
            output[linear] = TEST_FAILURE;
            return;
        }
    }
    work = (struct child_work *)((unsigned char *)uram + 64U);
    work->output = &output[linear];
    work->uram = uram;
    work->uram_tail = (uint64_t *)((unsigned char *)uram +
                                  dayan_uram_size() - sizeof(uint64_t));
    work->value = expected;
    if (dayan_create_thread("", child_write, work, 0) >= 0 ||
        dayan_create_thread("child-name-is-too-long", child_write, work, 0) >= 0) {
        output[linear] = TEST_FAILURE;
        return;
    }
    for (uint32_t child = 0; child < TEST_CHILD_THREADS; ++child) {
        children[child] = dayan_create_thread("child-write", child_write,
                                              work, 0);
        if (children[child] < 0) {
            output[linear] = TEST_FAILURE;
            return;
        }
    }
    if (dayan_create_thread("child-write", child_write, work, 0) >= 0)
        output[linear] = TEST_FAILURE;
    child_result = -1;
    if (dayan_thread_join((uint32_t)children[0], &child_result) < 0 ||
        child_result != (int64_t)expected ||
        dayan_thread_join((uint32_t)children[1], 0) < 0 ||
        dayan_create_thread("nested-write", nested_write, work, 0) < 0 ||
        dayan_thread_join_all() < 0)
        output[linear] = TEST_FAILURE;
    dayan_thread_exit((int64_t)expected);
}

void kill_kernel(uint64_t *output, uint64_t value)
{
    *output = value;
    for (;;)
        dayan_yield();
}

void locked_kernel(struct lock_test_results *results)
{
    struct lock_child_work work[LOCK_TEST_CHILD_THREADS];
    int children[LOCK_TEST_CHILD_THREADS];
    uint32_t index;

    dayan_cache_invalidate(results, sizeof(*results));
    results->root_begin = read_mcycle();
    for (index = 0; index < LOCK_TEST_CHILD_THREADS; ++index) {
        work[index].results = results;
        work[index].index = index;
        children[index] = dayan_create_thread("lock-child", locked_child,
                                              &work[index], 0);
        if (children[index] < 0)
            dayan_thread_exit(children[index]);
    }
    for (index = 0; index < LOCK_TEST_CHILD_THREADS; ++index) {
        int64_t result = -1;

        if (dayan_thread_join((uint32_t)children[index], &result) < 0 ||
            result != (int64_t)(LOCK_TEST_RESULT + index))
            dayan_thread_exit(-1);
    }
    results->family_end = read_mcycle();
    dayan_cache_flush((unsigned char *)results + 64U, 64U);
    dayan_thread_exit((int64_t)LOCK_TEST_RESULT);
}

void normal_kernel(struct lock_test_results *results)
{
    results->normal_begin = read_mcycle();
    dayan_cache_flush((unsigned char *)results + 64U, 64U);
    dayan_thread_exit((int64_t)NORMAL_TEST_RESULT);
}

void sequence_kernel(uint64_t *output, uint64_t value)
{
    *output = value;
    dayan_thread_exit((int64_t)value);
}
