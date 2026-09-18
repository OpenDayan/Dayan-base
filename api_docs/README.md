# Dayan 对外 API 参考手册

该目录按 HAL、Service、Runtime、RPC 和 dapi 五层记录当前对外接口，主体内容采用公开
头文件中的声明。为指导新平台接入，HAL 章节还描述了 `hal_internal.h` 和
`dataplane_internal.h` 中两组内部 Ops 的实现契约；这些 Ops 不属于应用程序 API。其它
模块私有结构、静态实现函数和仅供测试使用的接口不在范围内。

## 构建

需要 XeLaTeX、latexmk、`pdfinfo`、`pdffonts` 和 `pdftotext`：

```sh
make -C api_docs
make -C api_docs check
```

稳定输出为 `api_docs/build/dayan_api_reference.pdf`。`make check` 会检查字体嵌入、五层
章节是否存在，使用 `symbols/*.symbols` 与 `device/dayan-api.symbols` 检查函数覆盖率，
并将公开头文件中的版本常量、枚举和类型与对应章节逐项对照。

## API 边界

- HAL：`host/driver/hal/include/{npu_hal_config,hal,dataplane}.h`；
- Service：`host/driver/server/{service,server}.h`；
- Runtime：`host/include/{npu_runtime,npu_package,dayan_kernel,dayan_runtime}.h`，以及
  `host/dycc/include/dayan_compiler.h` 的源码标注；
- RPC：`device/include/rpc.h` 中的命令、共享描述符和模块接口；
- dapi：`device/include/api/*.h` 与 `device/dayan-api.symbols`。

新增或删除公开接口时，应同步更新对应章节和 symbol manifest，再执行 `make check`。

HAL 章节另有内部 Xilinx DMA 配置说明：`driver` 区分 XDMA/QDMA，`profile` 保留
simulator/fpga，completion 可选 IRQ/polling。相关配置头文件位于
`host/driver/hal/backend/xilinx_dma/`，不是应用 Runtime 的公开接口。
