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

_Static_assert(DAYAN_THREAD_NAME_SIZE == 16U, "public Thread name ABI");

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
    __tx_vec_tcsr_write<0>(0);
    __tx_vec_tcsr_read<0>(value);
    __tx_vec_sb_bar(value);
    __tx_vec_lsu_u2v<0, 0>(left);
    __tx_vec_lsu_u2v_broadcast32<1, 1>(right);
    __tx_vec_lsu_v2u<0, 0>(destination);
    __tx_vec_lsu_v2u_mask<1, 1, 0>(destination);
    __tx_vec_shuffle_xor<0, 0>(1);
    __tx_vec_shuffle_ror<0, 0>(1);
    __tx_vec_shuffle_broadcast<1, 1>(1);
    __tx_vec_shuffle_compress<1, 1>(1);
    __tx_vec_vv_fp32_add<0, 0, 0, 0>();
    __tx_vec_vv_fp32_sub<0, 0, 0, 0>();
    __tx_vec_vv_fp32_mul<0, 0, 0, 0>();
    __tx_vec_vv_mux2<0, 0, 0, 0>();
    __tx_vec_vv_int32_cmp<0, 0, 0, 0>();
    __tx_vec_vv_fp32_cmp<0, 0, 0, 0>();
    __tx_vec_v_fp32_sfu<0, 0, 0>();
    __tx_vec_cvt_fp32_int32<0, 0, 0>();
    __tx_vec_cvt_s32_fp32<0, 0, 0>();
    __tx_vec_cvt_fp32_fp16<0, 0, 0>();
    __tx_vec_cvt_fp32_bf16<0, 0, 0>();
    __tx_vec_cvt_fp16_fp32<0, 0, 0>();
    __tx_vec_cvt_bf16_fp32<0, 0, 0>();
    __tx_vec_cvt_mxfp8_bf16<0, 0, 0>();
    __tx_vec_cvt_mxfp4_bf16<0, 0, 0>();
    __tx_vec_s_int32_add<0, 0, 0>();
    __tx_vec_s_bitwise<0, 0, 0, 0>();
    __tx_macro_tcsr_write<0>(0);
    __tx_macro_elemwise_add_fp32<0, 0>(destination, left, right);
    __tx_macro_elemwise_sub_fp32<0, 0>(destination, left, right);
    __tx_macro_elemwise_mul_fp32<0, 0>(destination, left, right);
    __tx_macro_elemwise_max_fp32<0, 0>(destination, left, right);
    __tx_macro_elemwise_min_fp32<0, 0>(destination, left, right);
    __tx_macro_reduce_max_fp32<0, 0>(destination, left, right);
    __tx_macro_reduce_add_fp32<0>(destination, left);
    __tx_macro_sfu_log2_fp32<0>(destination, left);
    __tx_macro_sfu_exp2_fp32<0>(destination, left);
    __tx_macro_sfu_sin_fp32<0>(destination, left);
    __tx_macro_sfu_cos_fp32<0>(destination, left);
    __tx_macro_sfu_rsqrt_fp32<0>(destination, left);
    __tx_macro_sfu_recip_fp32<0>(destination, left);
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
    child = dayan_create_thread("compile-child", compile_child, memory, 0);
    if (child > 0)
        status |= dayan_thread_join((uint32_t)child, &child_result);
    else
        status |= child;
    status |= dayan_thread_join_all();
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
    dayan_cache_invalidate(memory, 64);
    dayan_cache_flush(memory, 64);
    (void)printf("status=%d", status);
    (void)fflush(stdout);
    free(memory);

    if (size == SIZE_MAX)
        dayan_thread_exit(child_result);

    return snprintf(buffer, size, "%d %x", status,
                    (unsigned)fabsf((float)CHAR_BIT));
}
