#include <dayan_kernel.h>

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#if defined(__DAYAN_DEVICE__)
#include <dayan_intrinsics.h>
#include <dayan_sys.h>
#endif

// 配置驱动 conv 探针:wi/hi/ci/co/wk/hk/pad 全部完整值语义,stride_ci 独立可配。
__kernel__ void conv2d_cfg(const uint16_t *in_ptr, const uint16_t *k_ptr,
                           uint32_t *out_ptr, uint32_t ci, uint32_t co,
                           uint32_t wk, uint32_t hk, uint32_t pad,
                           uint32_t wi, uint32_t hi, uint32_t stride_ci)
{
#if defined(__DAYAN_DEVICE__)
    const uint64_t uram = (uint64_t)(uintptr_t)dayan_uram_base();
    const uint64_t i_u = uram;
    const uint64_t k_u = uram + 4096;
    const uint64_t o_u = uram + 8192;
    const uint64_t m_u = uram + 12288;
    volatile uint32_t *m_pat = (volatile uint32_t *)(uintptr_t)m_u;
    const uint32_t wo = wi - wk + 2 * pad + 1;
    const uint32_t ib = wi * hi * stride_ci * 2;
    const uint32_t kb = hk * wk * stride_ci * co * 2;

    __tx_lsu_tcsr_write<LSU_BLOCK>(0);
    __tx_lsu_tcsr_write<LSU_DIM1>(0);
    __tx_lsu_tcsr_write<LSU_DIM2>(0);
    __tx_lsu_tcsr_write<LSU_INBLOCKSTRIDE>(64);
    __tx_lsu_tcsr_write<LSU_INDIM0STRIDE>(64);
    __tx_lsu_tcsr_write<LSU_INDIM1STRIDE>(64);
    __tx_lsu_tcsr_write<LSU_OUTBLOCKSTRIDE>(64);
    __tx_lsu_tcsr_write<LSU_OUTDIM0STRIDE>(64);
    __tx_lsu_tcsr_write<LSU_OUTDIM1STRIDE>(64);
    __tx_lsu_tcsr_write<LSU_BLOCKTRANSPOSE>(6);
    __tx_lsu_tcsr_write<LSU_BLOCKPADDING>(0);
    __tx_lsu_tcsr_write<LSU_REQUESTREADSIZE>(63);
    __tx_lsu_tcsr_write<LSU_REQUESTWRITESIZE>(63);
    __tx_lsu_tcsr_write<LSU_PADDINGVALUE>(0);
    __tx_lsu_tcsr_write<LSU_URAMWRITESWIZZLEMBS>(0);
    __tx_lsu_tcsr_write<LSU_URAMREADSWIZZLEMBS>(0);
    __tx_lsu_tcsr_write<LSU_TABLESIZE>(0);
    __tx_lsu_tcsr_write<LSU_TABLESTRIDE>(64);

    volatile uint32_t *o_pat = (volatile uint32_t *)(uintptr_t)o_u;
    for (uint32_t i = 0; i < 256; ++i)
        o_pat[i] = 0xCAFEBABEu;

    const uint32_t ib_f = ib & ~63u, ib_t = ib & 63u;
    if (ib_f) {
        __tx_lsu_tcsr_write<LSU_DIM0>(ib_f / 64 - 1);
        __tx_lsu_dma_d2u<0>(i_u, (long)in_ptr);
        dayan_wait_lsu();
    }
    if (ib_t) {
        __tx_lsu_tcsr_write<LSU_DIM0>(0);
        __tx_lsu_dma_d2u<0>(i_u + ib_f, (long)in_ptr + ib_f);
        dayan_wait_lsu();
    }
    const uint32_t kb_f = kb & ~63u, kb_t = kb & 63u;
    if (kb_f) {
        __tx_lsu_tcsr_write<LSU_DIM0>(kb_f / 64 - 1);
        __tx_lsu_dma_d2u<0>(k_u, (long)k_ptr);
        dayan_wait_lsu();
    }
    if (kb_t) {
        __tx_lsu_tcsr_write<LSU_DIM0>(0);
        __tx_lsu_dma_d2u<0>(k_u + kb_f, (long)k_ptr + kb_f);
        dayan_wait_lsu();
    }
    m_pat[0] = 0xE0000001u;
    __tx_lsu_tcsr_write<LSU_DIM0>(0);
    __tx_lsu_dma_u2d<0>((long)(out_ptr + 256), (long)m_u);
    dayan_wait_lsu();

    __tx_mat_tcsr_write<CONV_CI>(ci);
    __tx_mat_tcsr_write<CONV_WO>(wo);
    __tx_mat_tcsr_write<CONV_CO>(co);
    __tx_mat_tcsr_write<CONV_WISTRIDE>(stride_ci * 2);
    __tx_mat_tcsr_write<CONV_COSTRIDE>(stride_ci * 2);
    __tx_mat_tcsr_write<CONV_WOSTRIDE>(co * 4);
    __tx_mat_tcsr_write<CONV_WI>(wi);
    __tx_mat_tcsr_write<CONV_HI>(hi);
    __tx_mat_tcsr_write<CONV_WK>(wk);
    __tx_mat_tcsr_write<CONV_HK>(hk);
    __tx_mat_tcsr_write<CONV_HO>(wo);
    __tx_mat_tcsr_write<CONV_HISTRIDE>(wi * stride_ci * 2);
    __tx_mat_tcsr_write<CONV_WKSTRIDE>(stride_ci * 2);
    __tx_mat_tcsr_write<CONV_HKSTRIDE>(wk * stride_ci * 2);
    __tx_mat_tcsr_write<CONV_HOSTRIDE>(wo * co * 4);
    __tx_mat_tcsr_write<CONV_ISWIZZLEMBS>(0);
    __tx_mat_tcsr_write<CONV_KSWIZZLEMBS>(0);
    __tx_mat_tcsr_write<CONV_OSWIZZLEMBS>(0);
    __tx_mat_tcsr_write<CONV_DATATYPE>(0x02);
    __tx_mat_tcsr_write<CONV_PADL>(pad);
    __tx_mat_tcsr_write<CONV_PADU>(pad);
    __tx_mat_tcsr_write<CONV_DILATIONW>(1);
    __tx_mat_tcsr_write<CONV_DILATIONH>(1);
    __tx_mat_tcsr_write<CONV_STRIDEW>(1);
    __tx_mat_tcsr_write<CONV_STRIDEH>(1);
    __tx_mat_tcsr_write<CONV_STRIDEDILATIONW>(0);
    __tx_mat_tcsr_write<CONV_STRIDEDILATIONH>(0);
    m_pat[0] = 0xE0000002u;
    __tx_lsu_tcsr_write<LSU_DIM0>(0);
    __tx_lsu_dma_u2d<0>((long)(out_ptr + 256), (long)m_u);
    dayan_wait_lsu();

    __tx_mat_conv2d<0>((long)o_u, (long)i_u, (long)k_u);
    dayan_wait_mat();
    m_pat[0] = 0xE0000003u;
    __tx_lsu_tcsr_write<LSU_DIM0>(0);
    __tx_lsu_dma_u2d<0>((long)(out_ptr + 256), (long)m_u);
    dayan_wait_lsu();

    __tx_lsu_tcsr_write<LSU_DIM0>(15);
    __tx_lsu_dma_u2d<0>((long)out_ptr, (long)o_u);
    dayan_wait_lsu();
    m_pat[0] = 0xE0000004u;
    __tx_lsu_tcsr_write<LSU_DIM0>(0);
    __tx_lsu_dma_u2d<0>((long)(out_ptr + 256), (long)m_u);
    dayan_wait_lsu();
#endif
}
DAYAN_KERNEL_DESC(conv2d_cfg, 0, 0, 32768, 1, 1, 0, 0);

#if !defined(__DAYAN_DEVICE__)
#include "npu_runtime.h"
#include <vector>

__host__ static int wait_event(npu_event_t *event, uint32_t timeout_ms)
{
    int status = npu_event_wait(event, timeout_ms, nullptr);
    npu_event_destroy(event);
    return status;
}

struct Case {
    const char *name;
    uint32_t wk, hk, pad;  // 几何:I 4x4 Ci16 Co16 固定
    int imode;             // 0 全0, 1 全1, 2 单热(y,x,c), 3 双通道
    int kmode;             // 0 全0, 1 全1, 2 单热(r,s,c,k), 3 双热(c0,c1同k)
    uint32_t iy, ix, ic;   // imode=2 热点
    uint32_t kr, ks, kc, kk;  // kmode=2 热点
    uint32_t ic2, kc2;        // imode=3/kmode=3 第二点
    // 期望:pos(y,x)*16+co -> 值,由 host 参考模型算出
    std::vector<std::pair<uint32_t, float>> expect;
};

__host__ static void build_expect(Case &cs)
{
    const uint32_t wo = 4 - cs.wk + 2 * cs.pad + 1;
    // 参考模型:直接按 spec 伪代码算,零填充
    for (uint32_t h = 0; h < wo; ++h)
        for (uint32_t w = 0; w < wo; ++w)
            for (uint32_t k = 0; k < 16; ++k) {
                float sum = 0;
                for (uint32_t r = 0; r < cs.hk; ++r)
                    for (uint32_t s = 0; s < cs.wk; ++s) {
                        const long y = (long)h - cs.pad + r;
                        const long x = (long)w - cs.pad + s;
                        if (y < 0 || y >= 4 || x < 0 || x >= 4) continue;
                        for (uint32_t c = 0; c < 16; ++c) {
                            float iv = 0, kv = 0;
                            if (cs.imode == 1) iv = 1;
                            else if (cs.imode == 2)
                                iv = (y == cs.iy && x == cs.ix && c == cs.ic) ? 1 : 0;
                            else if (cs.imode == 3)
                                iv = (c == cs.ic || c == cs.ic2) ? 1 : 0;
                            if (cs.kmode == 1) kv = 1;
                            else if (cs.kmode == 2)
                                kv = (r == cs.kr && s == cs.ks && c == cs.kc &&
                                      k == cs.kk) ? 1 : 0;
                            else if (cs.kmode == 3)
                                kv = ((c == cs.kc || c == cs.kc2) && k == cs.kk) ? 1 : 0;
                            sum += iv * kv;
                        }
                    }
                if (sum != 0)
                    cs.expect.push_back({(h * wo + w) * 16 + k, sum});
            }
}

__host__ static int run_case(npu_context_t *context, npu_stream_t *stream,
                             npu_buffer_t *bi, npu_buffer_t *bk,
                             npu_buffer_t *bo, Case &cs)
{
    const uint32_t wo = 4 - cs.wk + 2 * cs.pad + 1;
    const size_t ib = 4 * 4 * 16 * 2, kb = cs.hk * cs.wk * 16 * 16 * 2,
                 ob = 1088;
    npu_event_t *e = nullptr;
    std::vector<uint16_t> iv(ib / 2, 0), kv(kb / 2, 0);
    if (cs.imode == 1)
        std::fill(iv.begin(), iv.end(), 0x3F80);
    else if (cs.imode == 2)
        iv[(cs.iy * 4 + cs.ix) * 16 + cs.ic] = 0x3F80;
    else if (cs.imode == 3) {
        for (uint32_t p = 0; p < 16; ++p) {
            iv[p * 16 + cs.ic] = 0x3F80;
            iv[p * 16 + cs.ic2] = 0x3F80;
        }
    }
    if (cs.kmode == 1)
        std::fill(kv.begin(), kv.end(), 0x3F80);
    else if (cs.kmode == 2)
        kv[((cs.kr * cs.wk + cs.ks) * 16 + cs.kk) * 16 + cs.kc] = 0x3F80;
    else if (cs.kmode == 3) {
        const uint32_t base = cs.kk * 16;
        kv[base + cs.kc] = 0x3F80;
        kv[base + cs.kc2] = 0x3F80;
    }
    std::vector<uint32_t> ov(ob / 4, 0);
    int status = npu_memcpy_h2d_async(stream, bi, 0, iv.data(), ib, &e);
    if (status >= 0) status = wait_event(e, 60000);
    if (status < 0) {
        printf("%s: h2d i fail %d\n", cs.name, status);
        return status;
    }
    status = npu_memcpy_h2d_async(stream, bk, 0, kv.data(), kb, &e);
    if (status >= 0) status = wait_event(e, 60000);
    if (status < 0) {
        printf("%s: h2d k fail %d\n", cs.name, status);
        return status;
    }

    npu_kernel_launch_attrs_t attrs;
    npu_kernel_launch_attrs_init(&attrs, context, stream, "conv2d_cfg");
    attrs.config.task_dim = NPUTaskDim3{1, 1, 1};
    attrs.config.task_mode = NPUTaskMode::kSingleCore;
    attrs.timeout_ms = 480000;
    status = conv2d_cfg(bi, bk, bo, 16, 16, cs.wk, cs.hk, cs.pad, 4, 4, 16,
                        &attrs);
    if (status < 0) {
        printf("%s: LAUNCH FAIL %d (kernel wedged)\n", cs.name, status);
        return status;
    }
    status = npu_memcpy_d2h_async(stream, ov.data(), bo, 0, ob, &e);
    if (status >= 0) status = wait_event(e, 60000);
    if (status < 0) {
        printf("%s: d2h fail %d\n", cs.name, status);
        return status;
    }
    build_expect(cs);
    // 对照:期望非零位置上的实测值 + 期望零位置上的非零污染 + 漏写
    int miss = 0, extra = 0, valbad = 0, notw = 0, exact = 0;
    std::vector<uint8_t> is_exp(256, 0);
    for (auto &pr : cs.expect)
        is_exp[pr.first] = 1;
    printf("%s (wo=%u exp-nonzero=%zu):\n", cs.name, wo, cs.expect.size());
    for (auto &pr : cs.expect) {
        const uint32_t v = ov[pr.first];
        const uint32_t pos = pr.first;
        if (v == 0xCAFEBABEu) {
            if (miss < 4)
                printf("  MISS y%u,x%u,c%u (not written)\n", pos / 16 / wo,
                       (pos / 16) % wo, pos % 16);
            ++miss;
        } else {
            float got = *(float *)&v;
            if (got == pr.second) ++exact;
            else {
                if (valbad < 6)
                    printf("  VAL y%u,x%u,c%u got=%.1f exp=%.1f (x%.2f)\n",
                           pos / 16 / wo, (pos / 16) % wo, pos % 16, got,
                           pr.second, got / pr.second);
                ++valbad;
            }
        }
    }
    for (uint32_t i = 0; i < 256; ++i) {
        if (is_exp[i]) continue;
        const uint32_t v = ov[i];
        if (v == 0xCAFEBABEu) ++notw;
        else if (v != 0) {
            if (extra < 4)
                printf("  EXTRA y%u,x%u,c%u =%.1f (should be 0)\n", i / 16 / wo,
                       (i / 16) % wo, i % 16, *(float *)&v);
            ++extra;
        }
    }
    printf("  => exact=%d valbad=%d miss=%d extra=%d unwritten-out=%d\n",
           exact, valbad, miss, extra, notw);
    return 0;
}

__host__ int main()
{
    npu_runtime_options_t opts;
    npu_runtime_t *runtime = nullptr;
    npu_context_t *context = nullptr;
    npu_stream_t *stream = nullptr;
    npu_buffer_t *bi = nullptr, *bk = nullptr, *bo = nullptr;
    int status;

    npu_runtime_options_init(&opts);
    if ((status = npu_runtime_open(&runtime, &opts)) < 0) goto fail;
    if ((status = npu_context_create(runtime, nullptr, &context)) < 0) goto fail;
    if ((status = npu_stream_create(context, nullptr, &stream)) < 0) goto fail;

    npu_buffer_desc_t desc;
    npu_buffer_desc_init(&desc, 4 * 4 * 16 * 2);
    if ((status = npu_buffer_create(context, &desc, &bi)) < 0) goto fail;
    npu_buffer_desc_init(&desc, 3 * 3 * 16 * 16 * 2);
    if ((status = npu_buffer_create(context, &desc, &bk)) < 0) goto fail;
    npu_buffer_desc_init(&desc, 1088);
    if ((status = npu_buffer_create(context, &desc, &bo)) < 0) goto fail;

    {
        // 15 scalars: wk hk pad imode kmode iy ix ic kr ks kc kk ic2 kc2
        Case cases[] = {
            {"A1_ihot(0,0,c0)_k1x1", 1, 1, 0, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, {}},
            // {"A2_ihot(2,3,c5)_k1x1", 1, 1, 0, 2, 1, 2, 3, 5, 0, 0, 0, 0, 0, 0, {}},
            // {"B1_khot(c0,k0)_i0", 1, 1, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, {}},
            // {"B2_khot(c8,k7)_i0", 1, 1, 0, 0, 2, 0, 0, 0, 0, 0, 8, 7, 0, 0, {}},
            // {"B3_khot(c15,k15)_i0", 1, 1, 0, 0, 2, 0, 0, 0, 0, 0, 15, 15, 0, 0, {}},
            // {"C1_2ch_2k_i2ch", 1, 1, 0, 3, 3, 0, 0, 0, 0, 0, 0, 0, 8, 8, {}},
            // {"A3_ihot(1,1,c9)_k3x3p0", 3, 3, 0, 2, 1, 1, 1, 9, 0, 0, 0, 0, 0, 0, {}},
            // {"A4_ihot(0,0,c0)_k3x3p0", 3, 3, 0, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, {}},
            // {"B4_khot(r2s2,c0,k0)_3x3", 3, 3, 0, 1, 2, 0, 0, 0, 2, 2, 0, 0, 0, 0, {}},
            // {"D1_ihot0_khot(c0k0)_1x1", 1, 1, 0, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, {}},
            // {"E1_ihot(0,0,c0)_k3x3p1", 3, 3, 1, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, {}},
        };
        for (Case &cs : cases) {
            if (run_case(context, stream, bi, bk, bo, cs) < 0) {
                printf("case %s wedged env; stopping\n", cs.name);
                break;
            }
        }
        return 0;
    }
fail:
    printf("FAIL status=%d\n", status);
    return 1;
}
#endif
