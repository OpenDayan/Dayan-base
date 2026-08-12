#include <limits.h>
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <dayan_sys.h>

#ifdef __cplusplus
#include <dayan_intrinsics.h>
#endif

_Static_assert(DAYAN_PROC_NAME_SIZE == 16U, "public proc name ABI");

static void compile_child(void *argument)
{
    (void)argument;
}

#ifdef __cplusplus
static void compile_intrinsics(long destination, long left, long right)
{
    long checksum;
    long value;

    set_lsu_matrix<0>(checksum, 1, 64, 64, 64);
    __tx_lsu_tcsr_write<LSU_DIM0>(0);
    __tx_lsu_tcsr_read<LSU_DIM0>(value);
    __tx_lsu_dma_u2u<0>(destination, left);
    __tx_lsu_dma_u2d<0>(destination, left);
    __tx_lsu_dma_d2u<0>(destination, left);
    __tx_lsu_dma_d2d<0>(destination, left);
    __tx_mat_tcsr_write<MAT_DATATYPE>(BaseNPUInstBuilder::DT_BF16);
    __tx_mat_tcsr_read<MAT_DATATYPE>(value);
    __tx_mat_mm<0>(destination, left, right);
    __tx_mat_mma<0>(destination, left, right);
    __tx_mat_conv2d<0>(destination, left, right);
    __tx_mat_conv2da<0>(destination, left, right);
    __tx_mat_setptr<0>(destination, left, right);
    (void)checksum;
    (void)value;
}
#endif

int operator_api_compile(dayan_resource_t *resource,
                         char *buffer, size_t size)
{
    dayan_shared_resource_t *shared_resource =
        (dayan_shared_resource_t *)shared;
    dayan_cluster_barrier_t *barrier =
        (dayan_cluster_barrier_t *)shared;
    void *memory = malloc(64);
    void *uram;
    int64_t child_result = 0;
    int child;
    int status;

    memset(memory, 0, 64);
#ifdef __cplusplus
    compile_intrinsics((long)buffer, (long)memory, (long)resource);
#endif
    status = dayan_resource_init(resource);
    status |= dayan_resource_alloc(resource, 1, 1);
    status |= dayan_resource_produce(resource);
    status |= dayan_resource_wait_ready(resource);
    status |= dayan_resource_consume(resource);
    status |= dayan_resource_wait_free(resource);
    status |= dayan_shared_resource_init(shared_resource);
    status |= dayan_shared_resource_alloc(shared_resource, 1, 1);
    status |= dayan_shared_resource_produce(shared_resource);
    status |= dayan_shared_resource_wait_ready(shared_resource);
    status |= dayan_shared_resource_consume(shared_resource);
    status |= dayan_shared_resource_wait_free(shared_resource);
    child = dayan_create_proc("compile-child", compile_child, memory);
    if (child > 0)
        status |= dayan_join((uint32_t)child, &child_result);
    else
        status |= child;
    status |= dayan_join_all();
    uram = dayan_uram_base();
    if (uram && !dayan_uram_size())
        status = -1;
    status |= dayan_cluster_barrier(4, barrier);
    dayan_yield();
    dayan_wait_dma();
    dayan_wait_lsu();
    dayan_wait_mat();
    dayan_wait_vec();
    dayan_wait_macro();
    (void)printf("status=%d", status);
    (void)fflush(stdout);
    free(memory);

    if (size == SIZE_MAX)
        dayan_proc_exit(child_result);

    return snprintf(buffer, size, "%d %x", status,
                    (unsigned)fabsf((float)CHAR_BIT));
}
