# mat_setptr_offsets_test

从 `dayan-samples/Instructions/MAT/mat_setptr_offsets.cpp` 迁移的裸机版本，发射走 `instr_enc`
（txenc mat_enc/mat_setptr_enc）+`instr_enc/tx_test.h`（txtst）骨架。

## 覆盖意图

tx.mat.setptr 非零偏移探针：三个寄存器参数用有效 URAM 基址上的偏移（rs1+64、rs2+128、哑 rd 侧 +32）。
setptr 的 rd 字段固定 00000（不写寄存器），rs1/rs2 为 A/B 侧额外矩阵指针；
S8 不消费 scale 数据，MM 结果仅作解码/调度哨兵，不声明 MX scale 数据语义。

输出交由 meta.json 定义的 ref/dut 对拍，设备侧不做尾部自判断。

## 构建运行

```sh
make -C dayan-base/device_tests MAT/mat_setptr_offsets_test
cd sim && python3 tools/qemu_rvtng_dayan_emu_runner.py ../dayan-base/device_tests/MAT/mat_setptr_offsets_test
```
