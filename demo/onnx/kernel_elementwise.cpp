#include <dayan_kernel.h>

#if defined(__DAYAN_DEVICE__)
#include <dayan_intrinsics.h>
#include <dayan_sys.h>
#endif

#include <stddef.h>
#include <stdint.h>

#define DAYAN_SCALAR_DESC(name) DAYAN_KERNEL_TASK_DESC(name,4,0,0,0,0,0,0,0)
#define DAYAN_VECTOR_DESC(name) DAYAN_KERNEL_TASK_DESC(name,4,0,0,32768,1,0,0,1)
#define DAYAN_VECTOR_PIPELINE_DESC(name) DAYAN_KERNEL_TASK_DESC(name,4,1,0,32768,2,0,0,2)

#define ELEMENTWISE_TILE_ELEMS 1024U
#define ELEMENTWISE_TILE_BYTES 4096U
#define ELEMENTWISE_LANE_BYTES (3U*ELEMENTWISE_TILE_BYTES)

#if defined(__DAYAN_DEVICE__)
static void invalid_arguments(void){dayan_thread_exit(-22);}
static inline void check_result(int result){
    if(result<0)dayan_thread_exit(result);
}
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
    configure_contiguous_lsu(ELEMENTWISE_TILE_BYTES);
    configure_macro_1d(ELEMENTWISE_TILE_ELEMS);
    return u;
}

enum class ElementwiseOp { Add, Mul, Sub };

template<ElementwiseOp Op>static inline void elementwise_inst(uint64_t dst,uint64_t lhs,uint64_t rhs){
    if(Op==ElementwiseOp::Add)__tx_macro_elemwise_add_fp32<0,0>((long)dst,(long)lhs,(long)rhs);
    else if(Op==ElementwiseOp::Mul)__tx_macro_elemwise_mul_fp32<0,0>((long)dst,(long)lhs,(long)rhs);
    else __tx_macro_elemwise_sub_fp32<0,0>((long)dst,(long)lhs,(long)rhs);
}

struct State {
    const float *a;
    const float *b;
    float *c;
    uint64_t blocks;
    uint64_t uram;
};

struct LoadArgs {
    State *state;
    int lane;
};

template<ElementwiseOp Op>static void elementwise_lane(void *argument){
    const LoadArgs *args=(LoadArgs *)argument;
    const State *s=args->state;
    const uint64_t a_uram=s->uram+(uint64_t)args->lane*ELEMENTWISE_LANE_BYTES;
    const uint64_t b_uram=a_uram+ELEMENTWISE_TILE_BYTES;
    const uint64_t c_uram=b_uram+ELEMENTWISE_TILE_BYTES;
    configure_contiguous_lsu(ELEMENTWISE_TILE_BYTES);
    configure_macro_1d(ELEMENTWISE_TILE_ELEMS);
    for(uint64_t q=args->lane;q<s->blocks;q+=2){
        const uint64_t offset=q*ELEMENTWISE_TILE_ELEMS;
        __tx_lsu_dma_d2u<0>((long)a_uram,(long)(s->a+offset));
        __tx_lsu_dma_d2u<0>((long)b_uram,(long)(s->b+offset));
        dayan_wait_lsu();
        elementwise_inst<Op>(c_uram,a_uram,b_uram);
        dayan_wait_macro();
        __tx_lsu_dma_u2d<0>((long)(s->c+offset),(long)c_uram);
        dayan_wait_lsu();
    }
}

template<ElementwiseOp Op>static void elementwise_entry(const float *a,const float *b,float *c,uint64_t n){
    if(!a||!b||!c||dayan_task_mode()!=1||dayan_task_dim(DAYAN_TASK_X)!=1U||
        dayan_task_dim(DAYAN_TASK_Y)!=1U||dayan_task_dim(DAYAN_TASK_Z)!=1U)
        invalid_arguments();
    const uint64_t uram=vector_resources(n);
    const uint64_t blocks=n/ELEMENTWISE_TILE_ELEMS;
    State s={a,b,c,blocks,uram};
    LoadArgs lanes[2]={{&s,0},{&s,1}};
    dayan_thread_resources_t resources={};
    resources.lsu_tcsr_count=1;
    resources.macro_tcsr_count=1;
    check_result(dayan_create_thread("elementwise",elementwise_lane<Op>,&lanes[1],&resources));
    elementwise_lane<Op>(&lanes[0]);
    check_result(dayan_thread_join_all());
}
#endif


__kernel__ void rvnpu_add_fp32(const float *a, const float *b, float *c,
    uint64_t n) {
#if defined(__DAYAN_DEVICE__)
    elementwise_entry<ElementwiseOp::Add>(a, b, c, n);
#endif
}

DAYAN_VECTOR_PIPELINE_DESC(rvnpu_add_fp32);


__kernel__ void rvnpu_mul_fp32(const float *a, const float *b, float *c, uint64_t n) {
#if defined(__DAYAN_DEVICE__)
    elementwise_entry<ElementwiseOp::Mul>(a, b, c, n);
#endif
}

DAYAN_VECTOR_PIPELINE_DESC(rvnpu_mul_fp32);


__kernel__ void rvnpu_sub_fp32(const float *a, const float *b, float *c, uint64_t n) {
#if defined(__DAYAN_DEVICE__)
    elementwise_entry<ElementwiseOp::Sub>(a, b, c, n);
#endif
}

DAYAN_VECTOR_PIPELINE_DESC(rvnpu_sub_fp32);


__kernel__ void rvnpu_div_fp32(const float *a, const float *b, float *c, uint64_t n) {
#if defined(__DAYAN_DEVICE__)
    const uint64_t uram=vector_resources(n);
    const uint64_t blocks=n/ELEMENTWISE_TILE_ELEMS;
    dayan_cache_invalidate(a,n*sizeof(float));
    dayan_cache_invalidate(b,n*sizeof(float));
    dayan_cache_invalidate(c,n*sizeof(float));
    for(uint64_t q=0;q<blocks;++q){
        const uint64_t au=uram;
        const uint64_t bu=uram+ELEMENTWISE_TILE_BYTES;
        const uint64_t reciprocal=uram+2U*ELEMENTWISE_TILE_BYTES;
        const uint64_t cu=uram+3U*ELEMENTWISE_TILE_BYTES;
        __tx_lsu_dma_d2u<0>((long)au,(long)(a+q*1024U));
        dayan_wait_lsu();
        __tx_lsu_dma_d2u<0>((long)bu,(long)(b+q*1024U));
        dayan_wait_lsu();
        __tx_macro_sfu_recip_fp32<0>(reciprocal,bu);
        __tx_macro_elemwise_mul_fp32<0,0>(cu,au,reciprocal);
        dayan_wait_macro();
        __tx_lsu_dma_u2d<0>((long)(c+q*1024U),(long)cu);
        dayan_wait_lsu();
    }
#endif
}
DAYAN_VECTOR_DESC(rvnpu_div_fp32);


__kernel__ void rvnpu_binary_broadcast_fp32(const float *a,const float *b,float *c,
    uint32_t a0,uint32_t a1,uint32_t a2,uint32_t a3,
    uint32_t b0,uint32_t b1,uint32_t b2,uint32_t b3,
    uint32_t op) {
#if defined(__DAYAN_DEVICE__)
    const uint64_t total=(uint64_t)a0*a1*a2*a3;
    if (!a||!b||!c||!total||op>3) invalid_arguments();
    dayan_cache_invalidate(a,total*sizeof(float));
    dayan_cache_invalidate(b,(uint64_t)b0*b1*b2*b3*sizeof(float));
    dayan_cache_invalidate(c,total*sizeof(float));
    const uint32_t ad[4]={a0,a1,a2,a3};
    const uint32_t bd[4]={b0,b1,b2,b3};
    uint64_t a_stride[4]={1U,1U,1U,1U};
    uint64_t b_stride[4]={1U,1U,1U,1U};
    for (int d=1;d<4;++d) {
        a_stride[d]=a_stride[d-1]*(uint64_t)ad[d-1];
        b_stride[d]=b_stride[d-1]*(uint64_t)bd[d-1];
    }
    for (uint64_t i=0;i<total;++i) {
        uint64_t r=i, ai=0, bi=0;
        for (int d=0;d<4;++d) {
            const uint32_t x=(uint32_t)(r%ad[d]);
            r/=ad[d];
            ai+=(uint64_t)x*a_stride[d];
            if (bd[d]!=1U) bi+=(uint64_t)x*b_stride[d];
        }
        float v = op==0?a[ai]+b[bi]:op==1?a[ai]*b[bi]:op==2?a[ai]-b[bi]:a[ai]/b[bi];
        c[i]=v;
    }
#endif
}

DAYAN_SCALAR_DESC(rvnpu_binary_broadcast_fp32);
