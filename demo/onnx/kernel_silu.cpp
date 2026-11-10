#include <dayan_kernel.h>

#if defined(__DAYAN_DEVICE__)
#include <dayan_intrinsics.h>
#include <dayan_sys.h>
#endif

#include <stddef.h>
#include <stdint.h>

#define SILU_TILE_ELEMS 1024U
#define SILU_URAM_BYTES 196608U
#define SILU_MACRO_BATCH_ELEMS 4096U
#define SILU_MACRO_SCRATCH_BYTES (2U * 16384U)
#define SILU_MAX_CHUNK_ELEMS 16384U
/* DIM0 counts above 95 hang the unit on streaming loops (RTL-verified). */
#define SILU_VEC_BLOCK_VECTORS 96U
#define DAYAN_VECTOR_DESC(name) DAYAN_KERNEL_TASK_DESC(name,4,0,0,SILU_URAM_BYTES,2,0,1,1)

#if defined(__DAYAN_DEVICE__)
static void invalid_arguments(void){dayan_thread_exit(-22);}
static inline uint32_t f32_bits(float v){uint32_t b;__builtin_memcpy(&b,&v,4);return b;}
template<long G>static inline void configure_contiguous_lsu(uint64_t bytes){
    __tx_lsu_tcsr_write<G*LSU_GROUPSZ+LSU_BLOCK>(0);
    __tx_lsu_tcsr_write<G*LSU_GROUPSZ+LSU_DIM0>((bytes+63)/64-1);
    __tx_lsu_tcsr_write<G*LSU_GROUPSZ+LSU_DIM1>(0);
    __tx_lsu_tcsr_write<G*LSU_GROUPSZ+LSU_DIM2>(0);
    __tx_lsu_tcsr_write<G*LSU_GROUPSZ+LSU_INBLOCKSTRIDE>(0);
    __tx_lsu_tcsr_write<G*LSU_GROUPSZ+LSU_INDIM0STRIDE>(64);
    __tx_lsu_tcsr_write<G*LSU_GROUPSZ+LSU_INDIM1STRIDE>(bytes);
    __tx_lsu_tcsr_write<G*LSU_GROUPSZ+LSU_INDIM2STRIDE>(0);
    __tx_lsu_tcsr_write<G*LSU_GROUPSZ+LSU_OUTBLOCKSTRIDE>(0);
    __tx_lsu_tcsr_write<G*LSU_GROUPSZ+LSU_OUTDIM0STRIDE>(64);
    __tx_lsu_tcsr_write<G*LSU_GROUPSZ+LSU_OUTDIM1STRIDE>(bytes);
    __tx_lsu_tcsr_write<G*LSU_GROUPSZ+LSU_OUTDIM2STRIDE>(0);
    __tx_lsu_tcsr_write<G*LSU_GROUPSZ+LSU_BLOCKTRANSPOSE>(6);
    __tx_lsu_tcsr_write<G*LSU_GROUPSZ+LSU_BLOCKPADDING>(0);
    __tx_lsu_tcsr_write<G*LSU_GROUPSZ+LSU_REQUESTREADSIZE>(63);
    __tx_lsu_tcsr_write<G*LSU_GROUPSZ+LSU_PADDINGVALUE>(0);
    __tx_lsu_tcsr_write<G*LSU_GROUPSZ+LSU_URAMWRITESWIZZLEMBS>(0);
    __tx_lsu_tcsr_write<G*LSU_GROUPSZ+LSU_URAMREADSWIZZLEMBS>(0);
    __tx_lsu_tcsr_write<G*LSU_GROUPSZ+LSU_REQUESTWRITESIZE>(63);
    __tx_lsu_tcsr_write<G*LSU_GROUPSZ+LSU_TABLESIZE>(0);
    __tx_lsu_tcsr_write<G*LSU_GROUPSZ+LSU_TABLESTRIDE>(64);
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
template<long G,typename T,typename U>static inline void macro_iv_mul_fp32(T r,float v,U s){
    __tx_macro_elemwise_mul_fp32<G,2>(r,(long)f32_bits(v),(long)s);
}
template<long G,typename T,typename U>static inline void macro_iv_add_fp32(T r,float v,U s){
    __tx_macro_elemwise_add_fp32<G,2>(r,(long)f32_bits(v),(long)s);
}

// MACRO half: one batch (n <= 4096) through the shared scratch pair; the
// final multiply lands in `dst`, a contiguous slice of the staging output.
static void macro_half(uint64_t x,uint64_t dst,uint64_t m,uint64_t o,uint64_t n){
    configure_macro_1d(n);
    macro_iv_mul_fp32<0>(m,-1.4426950408889634f,x);
    __tx_macro_sfu_exp2_fp32<0>(o,m);
    macro_iv_add_fp32<0>(m,1.0f,o);
    __tx_macro_sfu_recip_fp32<0>(o,m);
    __tx_macro_elemwise_mul_fp32<0,0>(dst,x,o);
}

static inline long vec_queue_free(){
    long free_slots;
    __tx_vec_tcsr_read<VEC_CONTROL>(free_slots);
    return free_slots;
}
static inline void wait_vec_queue(long instructions){
    while(vec_queue_free()<instructions)dayan_yield();
}
template<long Base>static inline void configure_vec_iterator(){
    __tx_vec_tcsr_write<Base>(0);
    __tx_vec_tcsr_write<Base+1>(256);
    __tx_vec_tcsr_write<Base+2>(0);
    __tx_vec_tcsr_write<Base+3>(0);
    __tx_vec_tcsr_write<Base+4>(0);
    __tx_vec_tcsr_write<Base+5>(0);
}
static inline void configure_vec_block(uint64_t vectors){
    __tx_vec_tcsr_write<VEC_DIM0>((long)vectors-1);
    __tx_vec_tcsr_write<VEC_DIM1>(0);
    __tx_vec_tcsr_write<VEC_DIM2>(0);
    __tx_vec_tcsr_write<VEC_DIM3>(0);
    __tx_vec_tcsr_write<VEC_DIM4>(0);
    __tx_vec_tcsr_write<VEC_DIM5>(0);
    __tx_vec_tcsr_write<VEC_IMPL_DEF_HINT>(0);
    configure_vec_iterator<VEC_V0_BASE>();
    configure_vec_iterator<VEC_V1_BASE>();
    __tx_vec_tcsr_write<VEC_V0_URAM_CHANNEL_STRIDE>(32);
    __tx_vec_tcsr_write<VEC_V1_URAM_CHANNEL_STRIDE>(32);
}
static inline void fork_vec(long p0,long p1,long loop,long e1,long e0){
    __tx_vec_tcsr_write<VEC_CONTROL>(p0|(p1<<8)|(loop<<16)|(e1<<24)|(e0<<32));
}

// One streaming silu block of at most 96 vectors (larger DIM0 counts hang
// the unit); constants re-seeded per block, returns without waiting.
static uint64_t vec_block_issue(uint64_t x,uint64_t y,uint64_t vectors){
    const uint64_t block=vectors>SILU_VEC_BLOCK_VECTORS?
        SILU_VEC_BLOCK_VECTORS:vectors;
    configure_vec_block(block);
    wait_vec_queue(9);
    __tx_vec_lsu_g2v_broadcast32<1,0>((long)f32_bits(1.0f));
    __tx_vec_lsu_g2v_broadcast32<1,0>((long)f32_bits(-1.4426950408889634f));
    __tx_vec_lsu_u2v<0,0>((long)x);
    __tx_vec_vv_fp32_mul<0,0,16,0>();
    __tx_vec_v_fp32_sfu<0,0,1>();
    __tx_vec_vv_fp32_add<0,0,17,0>();
    __tx_vec_v_fp32_sfu<0,0,5>();
    __tx_vec_vv_fp32_mul<0,4,0,0>();
    __tx_vec_lsu_v2u<0,1>((long)y);
    fork_vec(2,2,9,9,9);
    return block;
}

// The SFU load of each half goes to its own unit and the halves execute
// concurrently (RTL-verified). MACRO batches write their final multiply
// straight into a contiguous staging copy of the output, so one u2d drains
// everything; TCSR is written exactly once because rewriting MACRO TCSR
// while VEC runs corrupts the chain. Sizes whose double-buffered staging
// does not fit fall back to in-place MACRO results plus a separate VEC
// staging area (two stores).
static void silu_split(const float *input,float *output,uint64_t n,uint64_t uram){
    const uint64_t x=uram;
    /* Balance the halves: MACRO runs ~2.43 c/elem, VEC ~1.59 plus ~1.1k
     * per block, so equal element counts leave MACRO idle-waiting. */
    const uint64_t half=n/2U;
    const uint64_t n_mac=half<SILU_MACRO_BATCH_ELEMS
        ?((100U*n+69000U)/251U+32U)&~63U
        :half/SILU_MACRO_BATCH_ELEMS*SILU_MACRO_BATCH_ELEMS;
    const uint64_t n_vec=n-n_mac;
    const uint64_t nvt=n_vec/64U;
        const uint64_t vec_x=x+n_mac*4U;
    const uint64_t staged=n*8U+2U*16384U<=SILU_URAM_BYTES;
    const uint64_t y_out=staged?x+n*4U:x;      /* MACRO destinations */
    const uint64_t y_vec=staged?y_out+n_mac*4U:x+n*4U;
    const uint64_t m=staged?y_vec+n_vec*4U:y_vec+n_vec*4U;
    const uint64_t o=m+16384U;
    configure_contiguous_lsu<0>(n*4U);
    __tx_lsu_dma_d2u<0>((long)x,(long)input);
    dayan_wait_lsu();
    uint64_t done=vec_block_issue(vec_x,y_vec,nvt);
    if(n_mac<=SILU_MACRO_BATCH_ELEMS){
        macro_half(x,y_out,m,o,n_mac);
        while(done<nvt){
            dayan_wait_vec();
            done+=vec_block_issue(vec_x+done*256U,y_vec+done*256U,nvt-done);
        }
        dayan_wait_vec();
        dayan_wait_macro();
    }else{
        uint64_t off=0;
        configure_macro_1d(SILU_MACRO_BATCH_ELEMS);
        while(off<n_mac*4U){
            macro_iv_mul_fp32<0>(m,-1.4426950408889634f,x+off);
            __tx_macro_sfu_exp2_fp32<0>(o,m);
            macro_iv_add_fp32<0>(m,1.0f,o);
            __tx_macro_sfu_recip_fp32<0>(o,m);
            __tx_macro_elemwise_mul_fp32<0,0>(y_out+off,x+off,o);
            off+=16384U;
            if(done<nvt&&off<n_mac*4U){
                dayan_wait_vec();
                done+=vec_block_issue(vec_x+done*256U,y_vec+done*256U,
                                      nvt-done);
            }
            dayan_wait_macro();
        }
        while(done<nvt){
            dayan_wait_vec();
            done+=vec_block_issue(vec_x+done*256U,y_vec+done*256U,nvt-done);
        }
        dayan_wait_vec();
    }
    if(staged){
        configure_contiguous_lsu<1>(n*4U);
        __tx_lsu_dma_u2d<1>((long)output,(long)y_out);
        dayan_wait_lsu();
    }else{
        configure_contiguous_lsu<1>(n_mac*4U);
        __tx_lsu_dma_u2d<1>((long)output,(long)y_out);
        dayan_wait_lsu();
        configure_contiguous_lsu<1>(n_vec*4U);
        __tx_lsu_dma_u2d<1>((long)(output+n_mac),(long)y_vec);
        dayan_wait_lsu();
    }
}

#endif

__kernel__ void rvnpu_silu_f32(const float *x,float *y,uint32_t d0,uint32_t d1,uint32_t d2,uint32_t d3) {
#if defined(__DAYAN_DEVICE__)
    const uint32_t tasks=dayan_task_dim(DAYAN_TASK_X);
    if(!((dayan_task_mode()==1&&tasks==1U)||(dayan_task_mode()==4&&tasks==4U))||
        dayan_task_dim(DAYAN_TASK_Y)!=1U||dayan_task_dim(DAYAN_TASK_Z)!=1U)
        invalid_arguments();
    const uint64_t n=(uint64_t)d0*d1*d2*d3;
    const uint64_t uram=(uint64_t)(uintptr_t)dayan_uram_base();
    if(!uram||!n||(n&1023)||dayan_uram_size()<SILU_URAM_BYTES)
        invalid_arguments();
    // Whole 1024-element tiles per core: no partial DMA or vector is shared.
    const uint64_t blocks=n/SILU_TILE_ELEMS;
    const uint64_t first=blocks*dayan_task_id(DAYAN_TASK_X)/tasks;
    const uint64_t last=blocks*(dayan_task_id(DAYAN_TASK_X)+1U)/tasks;
    if(first==last)return;
    uint64_t offset = first * SILU_TILE_ELEMS;
    uint64_t remaining = (last - first) * SILU_TILE_ELEMS;
    // Process each core's complete-tile range in URAM-sized chunks. The old
    // whole-range call returned -22 for large YOLO feature maps because its
    // input/output staging was sized for at most one local URAM window.
    while (remaining != 0) {
        const uint64_t chunk = remaining < SILU_MAX_CHUNK_ELEMS
            ? remaining : SILU_MAX_CHUNK_ELEMS;
        silu_split(x + offset, y + offset, chunk, uram);
        offset += chunk;
        remaining -= chunk;
    }
#endif
}
DAYAN_VECTOR_DESC(rvnpu_silu_f32);
