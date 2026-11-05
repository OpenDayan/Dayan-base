# VEC 测试用例指令依赖序列

> 自动提取自各测试 main.cpp 的发射序列（load/store/g2s 同在一个指令块内，共同决定队列年龄）。
> 选择子 vrs（5 位）= bit4 选循环队列 bank + 低 4 位 age（0 = 该 bank 最近一次写入）；vb = 写回 bank；sel = 迭代器组号。
> 整块无中间等待，一次 CONTROL 启动（start_vec_block(N)，N = 块内指令数）。

## 总表

| 测试 | tag | 指令数 | 序列摘要 |
| --- | --- | ---: | --- |
| [vec_bank1_shuf_alu_dep_test](#vec_bank1_shuf_alu_dep_test) | vdep02 | 6 | lsu.u2v → shuffle.ror → shuffle.xor → lsu.u2v … |
| [vec_bf16_pack_roundtrip_test](#vec_bf16_pack_roundtrip_test) | vdep03 | 6 | lsu.u2v → cvt.bf16.fp32 → shuffle.compress → shuffle.compress … |
| [vec_bf16_widen_add_dep_test](#vec_bf16_widen_add_dep_test) | vdep04 | 6 | lsu.u2v → lsu.u2v → shuffle.compress → cvt.fp32.bf16 … |
| [vec_cmp_mux_add_dep_test](#vec_cmp_mux_add_dep_test) | vdep05 | 9 | lsu.u2v → lsu.u2v → lsu.u2v → lsu.u2v … |
| [vec_cvt_bf16_compress_dep_test](#vec_cvt_bf16_compress_dep_test) | vdep06 | 4 | lsu.u2v → cvt.bf16.fp32 → shuffle.compress → lsu.v2u |
| [vec_cvt_bf16_compress_or_dep_test](#vec_cvt_bf16_compress_or_dep_test) | vdep07 | 6 | lsu.u2v → lsu.u2v → cvt.bf16.fp32 → shuffle.compress … |
| [vec_cvt_bf16_stride_dep_test](#vec_cvt_bf16_stride_dep_test) | vdep08 | 4 | lsu.u2v → cvt.bf16.fp32 → shuffle.compress → lsu.v2u |
| [vec_cvt_i32_fp32_dep_test](#vec_cvt_i32_fp32_dep_test) | vdep09 | 6 | lsu.u2v → lsu.u2v → cvt.s32.fp32 → vv.int32.add … |
| [vec_decompress_bf16_widen_dep_test](#vec_decompress_bf16_widen_dep_test) | vdep10 | 4 | lsu.u2v → shuffle.compress → cvt.fp32.bf16 → lsu.v2u |
| [vec_dyn_compress_bf16_dep_test](#vec_dyn_compress_bf16_dep_test) | vdep11 | 16 | lsu.u2v → cvt.bf16.fp32 → shuffle.compress → lsu.v2u … |
| [vec_exp2_fp32_mul_dep_test](#vec_exp2_fp32_mul_dep_test) | vdep12 | 5 | lsu.u2v → v.float32.sfu(exp2) → lsu.u2v → vv.fp32.mul … |
| [vec_exp2_recip_fp32_mul_dep_test](#vec_exp2_recip_fp32_mul_dep_test) | vdep13 | 6 | lsu.u2v → v.float32.sfu(exp2) → v.float32.sfu(recip) → lsu.u2v … |
| [vec_fp16_pack_roundtrip_test](#vec_fp16_pack_roundtrip_test) | vdep14 | 6 | lsu.u2v → cvt.fp16.fp32 → shuffle.compress → shuffle.compress … |
| [vec_fp32_add_max_mul_chain_test](#vec_fp32_add_max_mul_chain_test) | vdep15 | 8 | lsu.u2v → lsu.u2v → lsu.u2v → lsu.u2v … |
| [vec_i32_add_xor_rol_dep_test](#vec_i32_add_xor_rol_dep_test) | vdep01 | 8 | lsu.u2v → lsu.u2v → lsu.u2v → lsu.u2v … |
| [vec_i32_mulu24_maxu_and_chain_test](#vec_i32_mulu24_maxu_and_chain_test) | vdep16 | 8 | lsu.u2v → lsu.u2v → lsu.u2v → lsu.u2v … |
| [vec_icmp_iadd_test](#vec_icmp_iadd_test) | vdep17 | 9 | lsu.u2v → lsu.u2v → lsu.u2v → lsu.u2v … |
| [vec_pack_xor_ror_dep_test](#vec_pack_xor_ror_dep_test) | vdep18 | 7 | lsu.u2v → cvt.bf16.fp32 → shuffle.compress → lsu.u2v … |
| [vec_pred_bit_test](#vec_pred_bit_test) | vdep19 | 13 | lsu.u2v → lsu.u2v → lsu.u2v → lsu.u2v … |
| [vec_pred_shuf_test](#vec_pred_shuf_test) | vdep20 | 10 | lsu.u2v → lsu.u2v → lsu.u2v → lsu.u2v … |
| [vec_recip_exp2_fp32_add_dep_test](#vec_recip_exp2_fp32_add_dep_test) | vdep21 | 6 | lsu.u2v → v.float32.sfu(recip) → v.float32.sfu(exp2) → lsu.u2v … |
| [vec_recip_fp32_mul_dep_test](#vec_recip_fp32_mul_dep_test) | vdep22 | 5 | lsu.u2v → v.float32.sfu(recip) → lsu.u2v → vv.fp32.mul … |
| [vec_shuffle_broadcast_mul_dep_test](#vec_shuffle_broadcast_mul_dep_test) | vdep23 | 5 | lsu.u2v → shuffle.broadcast → lsu.u2v → vv.fp32.mul … |
| [vec_shuffle_ror_xor_sub_dep_test](#vec_shuffle_ror_xor_sub_dep_test) | vdep24 | 6 | lsu.u2v → shuffle.ror → shuffle.xor → lsu.u2v … |
| [vec_shuffle_xor_add_dep_test](#vec_shuffle_xor_add_dep_test) | vdep25 | 5 | lsu.u2v → shuffle.xor → lsu.u2v → vv.fp32.add … |
| [vec_shuffle_xor_dyn_add_dep_test](#vec_shuffle_xor_dyn_add_dep_test) | vdep26 | 6 | lsu.u2v → lsu.g2s → shuffle.xor.dyn → lsu.u2v … |
| [vec_vv_fp32_add_mul_sub_chain_test](#vec_vv_fp32_add_mul_sub_chain_test) | vdep27 | 8 | lsu.u2v → lsu.u2v → lsu.u2v → lsu.u2v … |

## 逐条序列

<a id="vec_bank1_shuf_alu_dep_test"></a>
### vec_bank1_shuf_alu_dep_test（vdep02，6 条）
```
 1. tx.vec.lsu.u2v            vb=1 sel=0
 2. tx.vec.shuffle.ror        vrs1=16 vb=1 ctrl=3
 3. tx.vec.shuffle.xor        vrs1=16 vb=1 ctrl=1
 4. tx.vec.lsu.u2v            vb=0 sel=0
 5. tx.vec.vv.fp32.sub        vrs1=16 vrs2=0 vb=1 rm=0
 6. tx.vec.lsu.v2u            vrs1=16 sel=0
```

<a id="vec_bf16_pack_roundtrip_test"></a>
### vec_bf16_pack_roundtrip_test（vdep03，6 条）
```
 1. tx.vec.lsu.u2v            vb=0 sel=0
 2. tx.vec.cvt.bf16.fp32      vrs1=0 vb=0 rm=0
 3. tx.vec.shuffle.compress   vrs1=0 vb=0 ctrl=0
 4. tx.vec.shuffle.compress   vrs1=0 vb=0 ctrl=32
 5. tx.vec.cvt.fp32.bf16      vrs1=0 vb=0 rm=0
 6. tx.vec.lsu.v2u            vrs1=0 sel=0
```

<a id="vec_bf16_widen_add_dep_test"></a>
### vec_bf16_widen_add_dep_test（vdep04，6 条）
```
 1. tx.vec.lsu.u2v            vb=1 sel=0
 2. tx.vec.lsu.u2v            vb=0 sel=0
 3. tx.vec.shuffle.compress   vrs1=0 vb=0 ctrl=32
 4. tx.vec.cvt.fp32.bf16      vrs1=0 vb=0 rm=0
 5. tx.vec.vv.fp32.add        vrs1=0 vrs2=16 vb=0 rm=0
 6. tx.vec.lsu.v2u            vrs1=0 sel=0
```

<a id="vec_cmp_mux_add_dep_test"></a>
### vec_cmp_mux_add_dep_test（vdep05，9 条）
```
 1. tx.vec.lsu.u2v            vb=0 sel=0
 2. tx.vec.lsu.u2v            vb=1 sel=0
 3. tx.vec.lsu.u2v            vb=0 sel=0
 4. tx.vec.lsu.u2v            vb=1 sel=0
 5. tx.vec.lsu.u2v            vb=1 sel=0
 6. tx.vec.vv.fp32.cmp(>)     lhs=1 rhs=18 →sd=2
 7. tx.vec.vv.mux2            vrs1(0侧)=0 vrs2(1侧)=17 s2=2 vb=0
 8. tx.vec.vv.fp32.add        vrs1=0 vrs2=16 vb=0 rm=0
 9. tx.vec.lsu.v2u            vrs1=0 sel=0
```

<a id="vec_cvt_bf16_compress_dep_test"></a>
### vec_cvt_bf16_compress_dep_test（vdep06，4 条）
```
 1. tx.vec.lsu.u2v            vb=0 sel=0
 2. tx.vec.cvt.bf16.fp32      vrs1=0 vb=0 rm=0
 3. tx.vec.shuffle.compress   vrs1=0 vb=0 ctrl=0
 4. tx.vec.lsu.v2u            vrs1=0 sel=0
```

<a id="vec_cvt_bf16_compress_or_dep_test"></a>
### vec_cvt_bf16_compress_or_dep_test（vdep07，6 条）
```
 1. tx.vec.lsu.u2v            vb=0 sel=0
 2. tx.vec.lsu.u2v            vb=1 sel=0
 3. tx.vec.cvt.bf16.fp32      vrs1=0 vb=0 rm=0
 4. tx.vec.shuffle.compress   vrs1=0 vb=0 ctrl=0
 5. tx.vec.vv.int32.bitwise(LUT=0x0e)vrs1=0 vrs2=16 vb=0
 6. tx.vec.lsu.v2u            vrs1=0 sel=0
```

<a id="vec_cvt_bf16_stride_dep_test"></a>
### vec_cvt_bf16_stride_dep_test（vdep08，4 条）
```
 1. tx.vec.lsu.u2v            vb=0 sel=0
 2. tx.vec.cvt.bf16.fp32      vrs1=0 vb=0 rm=0
 3. tx.vec.shuffle.compress   vrs1=0 vb=0 ctrl=0
 4. tx.vec.lsu.v2u            vrs1=0 sel=0
```

<a id="vec_cvt_i32_fp32_dep_test"></a>
### vec_cvt_i32_fp32_dep_test（vdep09，6 条）
```
 1. tx.vec.lsu.u2v            vb=0 sel=0
 2. tx.vec.lsu.u2v            vb=1 sel=0
 3. tx.vec.cvt.s32.fp32       vrs1=0 vb=0 rm=0
 4. tx.vec.vv.int32.add       vrs1=0 vrs2=16 vb=0
 5. tx.vec.cvt.fp32.int32     vrs1=0 vb=0 rm=0
 6. tx.vec.lsu.v2u            vrs1=0 sel=0
```

<a id="vec_decompress_bf16_widen_dep_test"></a>
### vec_decompress_bf16_widen_dep_test（vdep10，4 条）
```
 1. tx.vec.lsu.u2v            vb=0 sel=0
 2. tx.vec.shuffle.compress   vrs1=0 vb=0 ctrl=32
 3. tx.vec.cvt.fp32.bf16      vrs1=0 vb=0 rm=0
 4. tx.vec.lsu.v2u            vrs1=0 sel=0
```

<a id="vec_dyn_compress_bf16_dep_test"></a>
### vec_dyn_compress_bf16_dep_test（vdep11，16 条）
```
 1. tx.vec.lsu.u2v            vb=0 sel=0
 2. tx.vec.cvt.bf16.fp32      vrs1=0 vb=0 rm=0
 3. tx.vec.shuffle.compress   vrs1=0 vb=0 ctrl=compress_ctrl
 4. tx.vec.lsu.v2u            vrs1=0 sel=0
 5. tx.vec.lsu.u2v            vb=0 sel=0
 6. tx.vec.cvt.bf16.fp32      vrs1=0 vb=0 rm=0
 7. tx.vec.lsu.g2s            →sd=1 val=compress_ctrl
 8. tx.vec.shuffle.compress.dynvrs1=0 vb=0 s1=1
 9. tx.vec.lsu.v2u            vrs1=0 sel=0
10. tx.vec.shuffle.compress   vrs1=0 vb=0 ctrl=decompress_ctrl
11. tx.vec.lsu.v2u            vrs1=0 sel=0
12. tx.vec.lsu.g2s            →sd=2 val=decompress_ctrl
13. tx.vec.shuffle.compress.dynvrs1=1 vb=0 s1=2
14. tx.vec.lsu.v2u            vrs1=0 sel=0
15. tx.vec.cvt.fp32.bf16      vrs1=0 vb=0 rm=0
16. tx.vec.lsu.v2u            vrs1=0 sel=0
```

<a id="vec_exp2_fp32_mul_dep_test"></a>
### vec_exp2_fp32_mul_dep_test（vdep12，5 条）
```
 1. tx.vec.lsu.u2v            vb=0 sel=0
 2. tx.vec.v.float32.sfu(exp2)vrs1=0 vb=0
 3. tx.vec.lsu.u2v            vb=1 sel=0
 4. tx.vec.vv.fp32.mul        vrs1=0 vrs2=16 vb=0 rm=0
 5. tx.vec.lsu.v2u            vrs1=0 sel=0
```

<a id="vec_exp2_recip_fp32_mul_dep_test"></a>
### vec_exp2_recip_fp32_mul_dep_test（vdep13，6 条）
```
 1. tx.vec.lsu.u2v            vb=0 sel=0
 2. tx.vec.v.float32.sfu(exp2)vrs1=0 vb=0
 3. tx.vec.v.float32.sfu(recip)vrs1=0 vb=0
 4. tx.vec.lsu.u2v            vb=1 sel=0
 5. tx.vec.vv.fp32.mul        vrs1=0 vrs2=16 vb=0 rm=0
 6. tx.vec.lsu.v2u            vrs1=0 sel=0
```

<a id="vec_fp16_pack_roundtrip_test"></a>
### vec_fp16_pack_roundtrip_test（vdep14，6 条）
```
 1. tx.vec.lsu.u2v            vb=0 sel=0
 2. tx.vec.cvt.fp16.fp32      vrs1=0 vb=0 rm=0
 3. tx.vec.shuffle.compress   vrs1=0 vb=0 ctrl=0
 4. tx.vec.shuffle.compress   vrs1=0 vb=0 ctrl=32
 5. tx.vec.cvt.fp32.fp16      vrs1=0 vb=0 rm=0
 6. tx.vec.lsu.v2u            vrs1=0 sel=0
```

<a id="vec_fp32_add_max_mul_chain_test"></a>
### vec_fp32_add_max_mul_chain_test（vdep15，8 条）
```
 1. tx.vec.lsu.u2v            vb=0 sel=0
 2. tx.vec.lsu.u2v            vb=1 sel=0
 3. tx.vec.lsu.u2v            vb=1 sel=0
 4. tx.vec.lsu.u2v            vb=1 sel=0
 5. tx.vec.vv.fp32.add        vrs1=0 vrs2=18 vb=0 rm=0
 6. tx.vec.vv.fp32.max        vrs1=0 vrs2=17 vb=0 rm=0
 7. tx.vec.vv.fp32.mul        vrs1=0 vrs2=16 vb=0 rm=0
 8. tx.vec.lsu.v2u            vrs1=0 sel=0
```

<a id="vec_i32_add_xor_rol_dep_test"></a>
### vec_i32_add_xor_rol_dep_test（vdep01，8 条）
```
 1. tx.vec.lsu.u2v            vb=0 sel=0
 2. tx.vec.lsu.u2v            vb=1 sel=0
 3. tx.vec.lsu.u2v            vb=1 sel=0
 4. tx.vec.lsu.u2v            vb=1 sel=0
 5. tx.vec.vv.int32.add       vrs1=0 vrs2=18 vb=0
 6. tx.vec.vv.int32.bitwise(LUT=0x06)vrs1=0 vrs2=17 vb=0
 7. tx.vec.vv.int32.rol       vrs1=0 vrs2=16 vb=0
 8. tx.vec.lsu.v2u            vrs1=0 sel=0
```

<a id="vec_i32_mulu24_maxu_and_chain_test"></a>
### vec_i32_mulu24_maxu_and_chain_test（vdep16，8 条）
```
 1. tx.vec.lsu.u2v            vb=0 sel=0
 2. tx.vec.lsu.u2v            vb=1 sel=0
 3. tx.vec.lsu.u2v            vb=1 sel=0
 4. tx.vec.lsu.u2v            vb=1 sel=0
 5. tx.vec.vv.int32.mulu24    vrs1=0 vrs2=18 vb=0
 6. tx.vec.vv.int32.maxu      vrs1=0 vrs2=17 vb=0
 7. tx.vec.vv.int32.bitwise(LUT=0x08)vrs1=0 vrs2=16 vb=0
 8. tx.vec.lsu.v2u            vrs1=0 sel=0
```

<a id="vec_icmp_iadd_test"></a>
### vec_icmp_iadd_test（vdep17，9 条）
```
 1. tx.vec.lsu.u2v            vb=0 sel=0
 2. tx.vec.lsu.u2v            vb=1 sel=0
 3. tx.vec.lsu.u2v            vb=0 sel=0
 4. tx.vec.lsu.u2v            vb=1 sel=0
 5. tx.vec.lsu.u2v            vb=1 sel=0
 6. tx.vec.vv.int32.cmp(<)    lhs=1 rhs=18 →sd=2
 7. tx.vec.vv.mux2            vrs1(0侧)=0 vrs2(1侧)=17 s2=2 vb=0
 8. tx.vec.vv.int32.add       vrs1=0 vrs2=16 vb=0
 9. tx.vec.lsu.v2u            vrs1=0 sel=0
```

<a id="vec_pack_xor_ror_dep_test"></a>
### vec_pack_xor_ror_dep_test（vdep18，7 条）
```
 1. tx.vec.lsu.u2v            vb=0 sel=0
 2. tx.vec.cvt.bf16.fp32      vrs1=0 vb=0 rm=0
 3. tx.vec.shuffle.compress   vrs1=0 vb=0 ctrl=0
 4. tx.vec.lsu.u2v            vb=1 sel=0
 5. tx.vec.vv.int32.bitwise(LUT=0x06)vrs1=0 vrs2=16 vb=0
 6. tx.vec.shuffle.ror        vrs1=0 vb=0 ctrl=3
 7. tx.vec.lsu.v2u            vrs1=0 sel=0
```

<a id="vec_pred_bit_test"></a>
### vec_pred_bit_test（vdep19，13 条）
```
 1. tx.vec.lsu.u2v            vb=0 sel=0
 2. tx.vec.lsu.u2v            vb=1 sel=0
 3. tx.vec.lsu.u2v            vb=0 sel=0
 4. tx.vec.lsu.u2v            vb=1 sel=0
 5. tx.vec.lsu.u2v            vb=0 sel=0
 6. tx.vec.lsu.u2v            vb=1 sel=0
 7. tx.vec.lsu.u2v            vb=1 sel=0
 8. tx.vec.vv.fp32.cmp(>)     lhs=2 rhs=19 →sd=2
 9. tx.vec.vv.fp32.cmp(<)     lhs=1 rhs=18 →sd=3
10. tx.vec.s.bitwise(LUT=0x0e)s1=2 s2=3 →sd=4
11. tx.vec.vv.mux2            vrs1(0侧)=0 vrs2(1侧)=17 s2=4 vb=0
12. tx.vec.vv.fp32.add        vrs1=0 vrs2=16 vb=0 rm=0
13. tx.vec.lsu.v2u            vrs1=0 sel=0
```

<a id="vec_pred_shuf_test"></a>
### vec_pred_shuf_test（vdep20，10 条）
```
 1. tx.vec.lsu.u2v            vb=0 sel=0
 2. tx.vec.lsu.u2v            vb=1 sel=0
 3. tx.vec.lsu.u2v            vb=0 sel=0
 4. tx.vec.lsu.u2v            vb=0 sel=0
 5. tx.vec.lsu.u2v            vb=1 sel=0
 6. tx.vec.vv.fp32.cmp(>)     lhs=2 rhs=17 →sd=2
 7. tx.vec.vv.mux2            vrs1(0侧)=1 vrs2(1侧)=0 s2=2 vb=0
 8. tx.vec.shuffle.xor        vrs1=0 vb=0 ctrl=1
 9. tx.vec.vv.fp32.mul        vrs1=0 vrs2=16 vb=0 rm=0
10. tx.vec.lsu.v2u            vrs1=0 sel=0
```

<a id="vec_recip_exp2_fp32_add_dep_test"></a>
### vec_recip_exp2_fp32_add_dep_test（vdep21，6 条）
```
 1. tx.vec.lsu.u2v            vb=0 sel=0
 2. tx.vec.v.float32.sfu(recip)vrs1=0 vb=0
 3. tx.vec.v.float32.sfu(exp2)vrs1=0 vb=0
 4. tx.vec.lsu.u2v            vb=1 sel=0
 5. tx.vec.vv.fp32.add        vrs1=0 vrs2=16 vb=0 rm=0
 6. tx.vec.lsu.v2u            vrs1=0 sel=0
```

<a id="vec_recip_fp32_mul_dep_test"></a>
### vec_recip_fp32_mul_dep_test（vdep22，5 条）
```
 1. tx.vec.lsu.u2v            vb=0 sel=0
 2. tx.vec.v.float32.sfu(recip)vrs1=0 vb=0
 3. tx.vec.lsu.u2v            vb=1 sel=0
 4. tx.vec.vv.fp32.mul        vrs1=0 vrs2=16 vb=0 rm=0
 5. tx.vec.lsu.v2u            vrs1=0 sel=0
```

<a id="vec_shuffle_broadcast_mul_dep_test"></a>
### vec_shuffle_broadcast_mul_dep_test（vdep23，5 条）
```
 1. tx.vec.lsu.u2v            vb=0 sel=0
 2. tx.vec.shuffle.broadcast  vrs1=0 vb=0 ctrl=13
 3. tx.vec.lsu.u2v            vb=1 sel=0
 4. tx.vec.vv.fp32.mul        vrs1=0 vrs2=16 vb=0 rm=0
 5. tx.vec.lsu.v2u            vrs1=0 sel=0
```

<a id="vec_shuffle_ror_xor_sub_dep_test"></a>
### vec_shuffle_ror_xor_sub_dep_test（vdep24，6 条）
```
 1. tx.vec.lsu.u2v            vb=0 sel=0
 2. tx.vec.shuffle.ror        vrs1=0 vb=0 ctrl=3
 3. tx.vec.shuffle.xor        vrs1=0 vb=0 ctrl=1
 4. tx.vec.lsu.u2v            vb=1 sel=0
 5. tx.vec.vv.fp32.sub        vrs1=0 vrs2=16 vb=0 rm=0
 6. tx.vec.lsu.v2u            vrs1=0 sel=0
```

<a id="vec_shuffle_xor_add_dep_test"></a>
### vec_shuffle_xor_add_dep_test（vdep25，5 条）
```
 1. tx.vec.lsu.u2v            vb=0 sel=0
 2. tx.vec.shuffle.xor        vrs1=0 vb=0 ctrl=1
 3. tx.vec.lsu.u2v            vb=1 sel=0
 4. tx.vec.vv.fp32.add        vrs1=0 vrs2=16 vb=0 rm=0
 5. tx.vec.lsu.v2u            vrs1=0 sel=0
```

<a id="vec_shuffle_xor_dyn_add_dep_test"></a>
### vec_shuffle_xor_dyn_add_dep_test（vdep26，6 条）
```
 1. tx.vec.lsu.u2v            vb=0 sel=0
 2. tx.vec.lsu.g2s            →sd=1 val=3
 3. tx.vec.shuffle.xor.dyn    vrs1=0 vb=0 s1=1
 4. tx.vec.lsu.u2v            vb=1 sel=0
 5. tx.vec.vv.fp32.add        vrs1=0 vrs2=16 vb=0 rm=0
 6. tx.vec.lsu.v2u            vrs1=0 sel=0
```

<a id="vec_vv_fp32_add_mul_sub_chain_test"></a>
### vec_vv_fp32_add_mul_sub_chain_test（vdep27，8 条）
```
 1. tx.vec.lsu.u2v            vb=0 sel=0
 2. tx.vec.lsu.u2v            vb=1 sel=0
 3. tx.vec.lsu.u2v            vb=1 sel=0
 4. tx.vec.lsu.u2v            vb=1 sel=0
 5. tx.vec.vv.fp32.add        vrs1=0 vrs2=18 vb=0 rm=0
 6. tx.vec.vv.fp32.mul        vrs1=0 vrs2=17 vb=0 rm=0
 7. tx.vec.vv.fp32.sub        vrs1=0 vrs2=16 vb=0 rm=0
 8. tx.vec.lsu.v2u            vrs1=0 sel=0
```

---

## 紧凑序列（单行版）

> 每个测试的完整依赖链单行速览（与逐条版一一对应）。

**vec_bank1_shuf_alu_dep_test（vdep02，6 条）**

（lsu.u2v vb=1 sel=0）+（shuffle.ror vrs1=16 vb=1 ctrl=3）+（shuffle.xor vrs1=16 vb=1 ctrl=1）+（lsu.u2v vb=0 sel=0）+（vv.fp32.sub vrs1=16 vrs2=0 vb=1 rm=0）+（lsu.v2u vrs1=16 sel=0）

**vec_bf16_pack_roundtrip_test（vdep03，6 条）**

（lsu.u2v vb=0 sel=0）+（cvt.bf16.fp32 vrs1=0 vb=0 rm=0）+（shuffle.compress vrs1=0 vb=0 ctrl=0）+（shuffle.compress vrs1=0 vb=0 ctrl=32）+（cvt.fp32.bf16 vrs1=0 vb=0 rm=0）+（lsu.v2u vrs1=0 sel=0）

**vec_bf16_widen_add_dep_test（vdep04，6 条）**

（lsu.u2v vb=1 sel=0）+（lsu.u2v vb=0 sel=0）+（shuffle.compress vrs1=0 vb=0 ctrl=32）+（cvt.fp32.bf16 vrs1=0 vb=0 rm=0）+（vv.fp32.add vrs1=0 vrs2=16 vb=0 rm=0）+（lsu.v2u vrs1=0 sel=0）

**vec_cmp_mux_add_dep_test（vdep05，9 条）**

（lsu.u2v vb=0 sel=0）+（lsu.u2v vb=1 sel=0）+（lsu.u2v vb=0 sel=0）+（lsu.u2v vb=1 sel=0）+（lsu.u2v vb=1 sel=0）+（vv.fp32.cmp(>) lhs=1 rhs=18 →sd=2）+（vv.mux2 vrs1(0侧)=0 vrs2(1侧)=17 s2=2 vb=0）+（vv.fp32.add vrs1=0 vrs2=16 vb=0 rm=0）+（lsu.v2u vrs1=0 sel=0）

**vec_cvt_bf16_compress_dep_test（vdep06，4 条）**

（lsu.u2v vb=0 sel=0）+（cvt.bf16.fp32 vrs1=0 vb=0 rm=0）+（shuffle.compress vrs1=0 vb=0 ctrl=0）+（lsu.v2u vrs1=0 sel=0）

**vec_cvt_bf16_compress_or_dep_test（vdep07，6 条）**

（lsu.u2v vb=0 sel=0）+（lsu.u2v vb=1 sel=0）+（cvt.bf16.fp32 vrs1=0 vb=0 rm=0）+（shuffle.compress vrs1=0 vb=0 ctrl=0）+（vv.int32.bitwise(LUT=0x0e) vrs1=0 vrs2=16 vb=0）+（lsu.v2u vrs1=0 sel=0）

**vec_cvt_bf16_stride_dep_test（vdep08，4 条）**

（lsu.u2v vb=0 sel=0）+（cvt.bf16.fp32 vrs1=0 vb=0 rm=0）+（shuffle.compress vrs1=0 vb=0 ctrl=0）+（lsu.v2u vrs1=0 sel=0）

**vec_cvt_i32_fp32_dep_test（vdep09，6 条）**

（lsu.u2v vb=0 sel=0）+（lsu.u2v vb=1 sel=0）+（cvt.s32.fp32 vrs1=0 vb=0 rm=0）+（vv.int32.add vrs1=0 vrs2=16 vb=0）+（cvt.fp32.int32 vrs1=0 vb=0 rm=0）+（lsu.v2u vrs1=0 sel=0）

**vec_decompress_bf16_widen_dep_test（vdep10，4 条）**

（lsu.u2v vb=0 sel=0）+（shuffle.compress vrs1=0 vb=0 ctrl=32）+（cvt.fp32.bf16 vrs1=0 vb=0 rm=0）+（lsu.v2u vrs1=0 sel=0）

**vec_dyn_compress_bf16_dep_test（vdep11，16 条）**

（lsu.u2v vb=0 sel=0）+（cvt.bf16.fp32 vrs1=0 vb=0 rm=0）+（shuffle.compress vrs1=0 vb=0 ctrl=compress_ctrl）+（lsu.v2u vrs1=0 sel=0）+（lsu.u2v vb=0 sel=0）+（cvt.bf16.fp32 vrs1=0 vb=0 rm=0）+（lsu.g2s →sd=1 val=compress_ctrl）+（shuffle.compress.dyn vrs1=0 vb=0 s1=1）+（lsu.v2u vrs1=0 sel=0）+（shuffle.compress vrs1=0 vb=0 ctrl=decompress_ctrl）+（lsu.v2u vrs1=0 sel=0）+（lsu.g2s →sd=2 val=decompress_ctrl）+（shuffle.compress.dyn vrs1=1 vb=0 s1=2）+（lsu.v2u vrs1=0 sel=0）+（cvt.fp32.bf16 vrs1=0 vb=0 rm=0）+（lsu.v2u vrs1=0 sel=0）

**vec_exp2_fp32_mul_dep_test（vdep12，5 条）**

（lsu.u2v vb=0 sel=0）+（v.float32.sfu(exp2) vrs1=0 vb=0）+（lsu.u2v vb=1 sel=0）+（vv.fp32.mul vrs1=0 vrs2=16 vb=0 rm=0）+（lsu.v2u vrs1=0 sel=0）

**vec_exp2_recip_fp32_mul_dep_test（vdep13，6 条）**

（lsu.u2v vb=0 sel=0）+（v.float32.sfu(exp2) vrs1=0 vb=0）+（v.float32.sfu(recip) vrs1=0 vb=0）+（lsu.u2v vb=1 sel=0）+（vv.fp32.mul vrs1=0 vrs2=16 vb=0 rm=0）+（lsu.v2u vrs1=0 sel=0）

**vec_fp16_pack_roundtrip_test（vdep14，6 条）**

（lsu.u2v vb=0 sel=0）+（cvt.fp16.fp32 vrs1=0 vb=0 rm=0）+（shuffle.compress vrs1=0 vb=0 ctrl=0）+（shuffle.compress vrs1=0 vb=0 ctrl=32）+（cvt.fp32.fp16 vrs1=0 vb=0 rm=0）+（lsu.v2u vrs1=0 sel=0）

**vec_fp32_add_max_mul_chain_test（vdep15，8 条）**

（lsu.u2v vb=0 sel=0）+（lsu.u2v vb=1 sel=0）+（lsu.u2v vb=1 sel=0）+（lsu.u2v vb=1 sel=0）+（vv.fp32.add vrs1=0 vrs2=18 vb=0 rm=0）+（vv.fp32.max vrs1=0 vrs2=17 vb=0 rm=0）+（vv.fp32.mul vrs1=0 vrs2=16 vb=0 rm=0）+（lsu.v2u vrs1=0 sel=0）

**vec_i32_add_xor_rol_dep_test（vdep01，8 条）**

（lsu.u2v vb=0 sel=0）+（lsu.u2v vb=1 sel=0）+（lsu.u2v vb=1 sel=0）+（lsu.u2v vb=1 sel=0）+（vv.int32.add vrs1=0 vrs2=18 vb=0）+（vv.int32.bitwise(LUT=0x06) vrs1=0 vrs2=17 vb=0）+（vv.int32.rol vrs1=0 vrs2=16 vb=0）+（lsu.v2u vrs1=0 sel=0）

**vec_i32_mulu24_maxu_and_chain_test（vdep16，8 条）**

（lsu.u2v vb=0 sel=0）+（lsu.u2v vb=1 sel=0）+（lsu.u2v vb=1 sel=0）+（lsu.u2v vb=1 sel=0）+（vv.int32.mulu24 vrs1=0 vrs2=18 vb=0）+（vv.int32.maxu vrs1=0 vrs2=17 vb=0）+（vv.int32.bitwise(LUT=0x08) vrs1=0 vrs2=16 vb=0）+（lsu.v2u vrs1=0 sel=0）

**vec_icmp_iadd_test（vdep17，9 条）**

（lsu.u2v vb=0 sel=0）+（lsu.u2v vb=1 sel=0）+（lsu.u2v vb=0 sel=0）+（lsu.u2v vb=1 sel=0）+（lsu.u2v vb=1 sel=0）+（vv.int32.cmp(<) lhs=1 rhs=18 →sd=2）+（vv.mux2 vrs1(0侧)=0 vrs2(1侧)=17 s2=2 vb=0）+（vv.int32.add vrs1=0 vrs2=16 vb=0）+（lsu.v2u vrs1=0 sel=0）

**vec_pack_xor_ror_dep_test（vdep18，7 条）**

（lsu.u2v vb=0 sel=0）+（cvt.bf16.fp32 vrs1=0 vb=0 rm=0）+（shuffle.compress vrs1=0 vb=0 ctrl=0）+（lsu.u2v vb=1 sel=0）+（vv.int32.bitwise(LUT=0x06) vrs1=0 vrs2=16 vb=0）+（shuffle.ror vrs1=0 vb=0 ctrl=3）+（lsu.v2u vrs1=0 sel=0）

**vec_pred_bit_test（vdep19，13 条）**

（lsu.u2v vb=0 sel=0）+（lsu.u2v vb=1 sel=0）+（lsu.u2v vb=0 sel=0）+（lsu.u2v vb=1 sel=0）+（lsu.u2v vb=0 sel=0）+（lsu.u2v vb=1 sel=0）+（lsu.u2v vb=1 sel=0）+（vv.fp32.cmp(>) lhs=2 rhs=19 →sd=2）+（vv.fp32.cmp(<) lhs=1 rhs=18 →sd=3）+（s.bitwise(LUT=0x0e) s1=2 s2=3 →sd=4）+（vv.mux2 vrs1(0侧)=0 vrs2(1侧)=17 s2=4 vb=0）+（vv.fp32.add vrs1=0 vrs2=16 vb=0 rm=0）+（lsu.v2u vrs1=0 sel=0）

**vec_pred_shuf_test（vdep20，10 条）**

（lsu.u2v vb=0 sel=0）+（lsu.u2v vb=1 sel=0）+（lsu.u2v vb=0 sel=0）+（lsu.u2v vb=0 sel=0）+（lsu.u2v vb=1 sel=0）+（vv.fp32.cmp(>) lhs=2 rhs=17 →sd=2）+（vv.mux2 vrs1(0侧)=1 vrs2(1侧)=0 s2=2 vb=0）+（shuffle.xor vrs1=0 vb=0 ctrl=1）+（vv.fp32.mul vrs1=0 vrs2=16 vb=0 rm=0）+（lsu.v2u vrs1=0 sel=0）

**vec_recip_exp2_fp32_add_dep_test（vdep21，6 条）**

（lsu.u2v vb=0 sel=0）+（v.float32.sfu(recip) vrs1=0 vb=0）+（v.float32.sfu(exp2) vrs1=0 vb=0）+（lsu.u2v vb=1 sel=0）+（vv.fp32.add vrs1=0 vrs2=16 vb=0 rm=0）+（lsu.v2u vrs1=0 sel=0）

**vec_recip_fp32_mul_dep_test（vdep22，5 条）**

（lsu.u2v vb=0 sel=0）+（v.float32.sfu(recip) vrs1=0 vb=0）+（lsu.u2v vb=1 sel=0）+（vv.fp32.mul vrs1=0 vrs2=16 vb=0 rm=0）+（lsu.v2u vrs1=0 sel=0）

**vec_shuffle_broadcast_mul_dep_test（vdep23，5 条）**

（lsu.u2v vb=0 sel=0）+（shuffle.broadcast vrs1=0 vb=0 ctrl=13）+（lsu.u2v vb=1 sel=0）+（vv.fp32.mul vrs1=0 vrs2=16 vb=0 rm=0）+（lsu.v2u vrs1=0 sel=0）

**vec_shuffle_ror_xor_sub_dep_test（vdep24，6 条）**

（lsu.u2v vb=0 sel=0）+（shuffle.ror vrs1=0 vb=0 ctrl=3）+（shuffle.xor vrs1=0 vb=0 ctrl=1）+（lsu.u2v vb=1 sel=0）+（vv.fp32.sub vrs1=0 vrs2=16 vb=0 rm=0）+（lsu.v2u vrs1=0 sel=0）

**vec_shuffle_xor_add_dep_test（vdep25，5 条）**

（lsu.u2v vb=0 sel=0）+（shuffle.xor vrs1=0 vb=0 ctrl=1）+（lsu.u2v vb=1 sel=0）+（vv.fp32.add vrs1=0 vrs2=16 vb=0 rm=0）+（lsu.v2u vrs1=0 sel=0）

**vec_shuffle_xor_dyn_add_dep_test（vdep26，6 条）**

（lsu.u2v vb=0 sel=0）+（lsu.g2s →sd=1 val=3）+（shuffle.xor.dyn vrs1=0 vb=0 s1=1）+（lsu.u2v vb=1 sel=0）+（vv.fp32.add vrs1=0 vrs2=16 vb=0 rm=0）+（lsu.v2u vrs1=0 sel=0）

**vec_vv_fp32_add_mul_sub_chain_test（vdep27，8 条）**

（lsu.u2v vb=0 sel=0）+（lsu.u2v vb=1 sel=0）+（lsu.u2v vb=1 sel=0）+（lsu.u2v vb=1 sel=0）+（vv.fp32.add vrs1=0 vrs2=18 vb=0 rm=0）+（vv.fp32.mul vrs1=0 vrs2=17 vb=0 rm=0）+（vv.fp32.sub vrs1=0 vrs2=16 vb=0 rm=0）+（lsu.v2u vrs1=0 sel=0）
