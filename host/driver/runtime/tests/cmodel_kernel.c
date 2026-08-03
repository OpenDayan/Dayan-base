#include <stdint.h>

#include <dayan_sys.h>

#define TEST_FAILURE (~(uint64_t)0)
#define TEST_CHILD_PROCS 15U

struct child_work {
    uint64_t *output;
    uint64_t *uram;
    uint64_t *uram_tail;
    uint64_t value;
};

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
    dayan_proc_exit((int64_t)work->value);
}

static void nested_write(void *argument)
{
    struct child_work *work = argument;
    int64_t result = -1;
    int child = dayan_create_proc("child-write", child_write, work);

    if (child < 0 || dayan_join((uint32_t)child, &result) < 0 ||
        result != (int64_t)work->value)
        dayan_proc_exit(-1);
    dayan_proc_exit(result);
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
    int children[TEST_CHILD_PROCS];
    int64_t child_result;

    if (!uram || dayan_uram_size() != 0x180000U ||
        ((uintptr_t)uram & 1023U)) {
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
    if (dayan_create_proc("", child_write, work) >= 0 ||
        dayan_create_proc("child-name-is-too-long", child_write, work) >= 0) {
        output[linear] = TEST_FAILURE;
        return;
    }
    for (uint32_t child = 0; child < TEST_CHILD_PROCS; ++child) {
        children[child] = dayan_create_proc("child-write", child_write,
                                            work);
        if (children[child] < 0) {
            output[linear] = TEST_FAILURE;
            return;
        }
    }
    if (dayan_create_proc("child-write", child_write, work) >= 0)
        output[linear] = TEST_FAILURE;
    child_result = -1;
    if (dayan_join((uint32_t)children[0], &child_result) < 0 ||
        child_result != (int64_t)expected ||
        dayan_join((uint32_t)children[1], 0) < 0 ||
        dayan_create_proc("nested-write", nested_write, work) < 0 ||
        dayan_join_all() < 0)
        output[linear] = TEST_FAILURE;
    dayan_proc_exit((int64_t)expected);
}

void kill_kernel(uint64_t *output, uint64_t value)
{
    *output = value;
    for (;;)
        dayan_yield();
}
