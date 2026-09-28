#include <dayan_kernel.h>

#if defined(__DAYAN_DEVICE__)
#include <dayan_intrinsics.h>
#include <dayan_sys.h>
#endif

#include <stddef.h>
#include <stdint.h>

#define SILU_TILE_ELEMS 1024U
#define SILU_TILE_BYTES 4096U
#define SILU_LANE_BYTES (3U*SILU_TILE_BYTES)
#define DAYAN_VECTOR_DESC(name) DAYAN_KERNEL_TASK_DESC(name,4,1,0,32768,2,0,0,2)

#if defined(__DAYAN_DEVICE__)
static void invalid_arguments(void){dayan_thread_exit(-22);}
static inline void check_result(int result){
    if(result<0)dayan_thread_exit(result);
}
static inline uint32_t f32_bits(float v){uint32_t b;__builtin_memcpy(&b,&v,4);return b;}
static inline void configure_contiguous_lsu(uint64_t bytes){
    __tx_lsu_tcsr_write<LSU_BLOCK>(0);
    __tx_lsu_tcsr_write<LSU_DIM0>((bytes+63)/64-1);
    __tx_lsu_tcsr_write<LSU_DIM1>(0);
    __tx_lsu_tcsr_write<LSU_DIM2>(0);
    __tx_lsu_tcsr_write<LSU_INBLOCKSTRIDE>(0);
    __tx_lsu_tcsr_write<LSU_INDIM0STRIDE>(64);
    __tx_lsu_tcsr_write<LSU_INDIM1STRIDE>(bytes);
    __tx_lsu_tcsr_write<LSU_INDIM2STRIDE>(0);
    __tx_lsu_tcsr_write<LSU_OUTBLOCKSTRIDE>(0);
    __tx_lsu_tcsr_write<LSU_OUTDIM0STRIDE>(64);
    __tx_lsu_tcsr_write<LSU_OUTDIM1STRIDE>(bytes);
    __tx_lsu_tcsr_write<LSU_OUTDIM2STRIDE>(0);
    __tx_lsu_tcsr_write<LSU_BLOCKTRANSPOSE>(6);
    __tx_lsu_tcsr_write<LSU_BLOCKPADDING>(0);
    __tx_lsu_tcsr_write<LSU_REQUESTREADSIZE>(63);
    __tx_lsu_tcsr_write<LSU_PADDINGVALUE>(0);
    __tx_lsu_tcsr_write<LSU_URAMWRITESWIZZLEMBS>(0);
    __tx_lsu_tcsr_write<LSU_URAMREADSWIZZLEMBS>(0);
    __tx_lsu_tcsr_write<LSU_REQUESTWRITESIZE>(63);
    __tx_lsu_tcsr_write<LSU_TABLESIZE>(0);
    __tx_lsu_tcsr_write<LSU_TABLESTRIDE>(64);
}
static inline void configure_macro_1d(uint32_t n){
    __tx_macro_tcsr_write<MACRO_DIM0>(n-1);
    __tx_macro_tcsr_write<MACRO_DIM1>(0);
    __tx_macro_tcsr_write<MACRO_DIM2>(0);
    __tx_macro_tcsr_write<MACRO_DIM3>(0);
    __tx_macro_tcsr_write<MACRO_SRC0DIM1STRIDE>(0);
    __tx_macro_tcsr_write<MACRO_SRC0DIM2STRIDE>(0);
    __tx_macro_tcsr_write<MACRO_SRC0DIM3STRIDE>(0);
    __tx_macro_tcsr_write<MACRO_SRC1DIM1STRIDE>(0);
    __tx_macro_tcsr_write<MACRO_SRC1DIM2STRIDE>(0);
    __tx_macro_tcsr_write<MACRO_SRC1DIM3STRIDE>(0);
    __tx_macro_tcsr_write<MACRO_DSTDIM1STRIDE>(0);
    __tx_macro_tcsr_write<MACRO_DSTDIM2STRIDE>(0);
    __tx_macro_tcsr_write<MACRO_DSTDIM3STRIDE>(0);
    __tx_macro_tcsr_write<MACRO_SRC0SWIZZLEMBS>(0);
    __tx_macro_tcsr_write<MACRO_SRC1SWIZZLEMBS>(0);
    __tx_macro_tcsr_write<MACRO_DSTSWIZZLEMBS>(0);
}
static inline uint64_t vector_resources(uint64_t n){
    const uint64_t u=(uint64_t)(uintptr_t)dayan_uram_base();
    if(!u||!n||(n&1023)||dayan_uram_size()<32768)invalid_arguments();
    configure_contiguous_lsu(SILU_TILE_BYTES);
    configure_macro_1d(SILU_TILE_ELEMS);
    return u;
}
template<long G,typename T,typename U>static inline void macro_iv_add_fp32(T r,float v,U s){
    __tx_macro_elemwise_add_fp32<G,2>(r,(long)f32_bits(v),(long)s);
}
template<long G,typename T,typename U>static inline void macro_iv_mul_fp32(T r,float v,U s){
    __tx_macro_elemwise_mul_fp32<G,2>(r,(long)f32_bits(v),(long)s);
}

struct State {
    const float *input;
    float *output;
    uint64_t blocks;
    uint64_t uram;
};

struct LoadArgs {
    State *state;
    int lane;
};

static void silu_lane(void *argument){
    const LoadArgs *args=(LoadArgs *)argument;
    const State *s=args->state;
    const uint64_t input_uram=s->uram+(uint64_t)args->lane*SILU_LANE_BYTES;
    const uint64_t middle_uram=input_uram+SILU_TILE_BYTES;
    const uint64_t output_uram=middle_uram+SILU_TILE_BYTES;
    configure_contiguous_lsu(SILU_TILE_BYTES);
    configure_macro_1d(SILU_TILE_ELEMS);
    for(uint64_t q=args->lane;q<s->blocks;q+=2){
        const uint64_t offset=q*SILU_TILE_ELEMS;
        __tx_lsu_dma_d2u<0>((long)input_uram,(long)(s->input+offset));
        dayan_wait_lsu();
        macro_iv_mul_fp32<0>(middle_uram,-1.4426950408889634f,input_uram);
        __tx_macro_sfu_exp2_fp32<0>(output_uram,middle_uram);
        macro_iv_add_fp32<0>(middle_uram,1.0f,output_uram);
        __tx_macro_sfu_recip_fp32<0>(output_uram,middle_uram);
        __tx_macro_elemwise_mul_fp32<0,0>(middle_uram,input_uram,output_uram);
        dayan_wait_macro();
        __tx_lsu_dma_u2d<0>((long)(s->output+offset),(long)middle_uram);
        dayan_wait_lsu();
    }
}
#endif

__kernel__ void rvnpu_silu_f32(const float *x,float *y,uint32_t d0,uint32_t d1,uint32_t d2,uint32_t d3) {
#if defined(__DAYAN_DEVICE__)
    if(dayan_task_mode()!=1||dayan_task_dim(DAYAN_TASK_X)!=1U||
        dayan_task_dim(DAYAN_TASK_Y)!=1U||dayan_task_dim(DAYAN_TASK_Z)!=1U)
        invalid_arguments();
    const uint64_t n=(uint64_t)d0*d1*d2*d3;
    const uint64_t uram=vector_resources(n);
    const uint64_t blocks=n/SILU_TILE_ELEMS;
    dayan_cache_invalidate(x,n*sizeof(float));
    dayan_cache_invalidate(y,n*sizeof(float));
    State s={x,y,blocks,uram};
    LoadArgs lanes[2]={{&s,0},{&s,1}};
    dayan_thread_resources_t resources={};
    resources.lsu_tcsr_count=1;
    resources.macro_tcsr_count=1;
    check_result(dayan_create_thread("silu",silu_lane,&lanes[1],&resources));
    silu_lane(&lanes[0]);
    check_result(dayan_thread_join_all());
#endif
}
DAYAN_VECTOR_DESC(rvnpu_silu_f32);
