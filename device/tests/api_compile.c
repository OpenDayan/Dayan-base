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
static_assert(MAT_DATATYPE_PAIR(BaseNPUInstBuilder::DT_MATIN_TF32, BaseNPUInstBuilder::DT_MATIN_TF32) == 0x8410, "TF32 pair encoding");
static_assert(MAT_DATATYPE_PAIR(BaseNPUInstBuilder::DT_MATIN_BF16, BaseNPUInstBuilder::DT_MATIN_FP16) == 0x8452, "mixed 16-bit encoding");
static_assert(MAT_DATATYPE_PAIR(BaseNPUInstBuilder::DT_MATIN_E4M3 | BaseNPUInstBuilder::DT_MATIN_MX_BIT, BaseNPUInstBuilder::DT_MATIN_S8) == 0x8224, "A-only MX encoding");
static_assert(MAT_DATATYPE_PAIR(BaseNPUInstBuilder::DT_MATIN_S8, BaseNPUInstBuilder::DT_MATIN_E4M3 | BaseNPUInstBuilder::DT_MATIN_MX_BIT) == 0x8908, "B-only MX encoding");

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
    __tx_mat_tcsr_write<MAT_DATATYPE>(MAT_DATATYPE_PAIR(BaseNPUInstBuilder::DT_MATIN_TF32, BaseNPUInstBuilder::DT_MATIN_TF32));
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
    __tx_vec_vv_fp32_max<0, 0, 0, 0>();
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

int operator_api_compile(dayan_resource_t resource,
                         char *buffer, size_t size)
{
    dayan_resource_t next = DAYAN_LOCAL_RESOURCE(1);
    dayan_resource_t shared_resource = DAYAN_SHARED_RESOURCE(0);
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
    status = dayan_resource_wait_free(resource);
    status |= dayan_set_proc_name("compile-proc");
    status |= dayan_resource_produce_lsu(resource, 1, 0);
    status |= dayan_resource_wait_ready(resource, 0);
    status |= dayan_resource_wait(DAYAN_RESOURCE_MASK(next),
                                  DAYAN_RESOURCE_MASK(resource), 0, 0);
    status |= dayan_resource_produce_mat(
        next, 1, DAYAN_RESOURCE_MASK(resource));
    status |= dayan_resource_wait_free(shared_resource);
    status |= dayan_resource_produce_vec(shared_resource, 1, 0);
    status |= dayan_resource_wait_ready(shared_resource, 0);
    status |= dayan_resource_produce_macro(
        resource, 1, DAYAN_RESOURCE_MASK(shared_resource));
    status |= dayan_resource_consume_lsu(DAYAN_RESOURCE_MASK(resource));
    status |= dayan_resource_consume_mat(DAYAN_RESOURCE_MASK(resource));
    status |= dayan_resource_consume_vec(DAYAN_RESOURCE_MASK(resource));
    status |= dayan_resource_consume_macro(DAYAN_RESOURCE_MASK(resource));
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
    (void)dayan_get_time();
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
