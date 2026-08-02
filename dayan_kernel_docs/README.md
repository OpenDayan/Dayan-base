# Dayan Kernel 编程与 ABI 手册

正文入口为 `dayan_kernel_manual.tex`，章节按以下顺序组织：

1. 软件栈架构；
2. 工具链与构建环境；
3. 算子编程模型；
4. Dayan API 与标准库；
5. Dayan OS 内部机制；
6. Host RPC 服务与传输实现；
7. RISC-V ABI 和交付检查附录。

## 构建

需要 XeLaTeX、latexmk、`pdfinfo`、`pdffonts` 和 `pdftotext`：

```sh
make
make check
```

稳定输出为 `build/dayan_kernel_manual.pdf`。`make check` 还会检查字体嵌入、章节关键
文本，并确认 `../dayan-api.symbols` 中的每个公开符号都在 API 或标准库章节出现。

## 内容来源

- `include/api/*.h` 和 `dayan-api.symbols` 定义 Device 公开接口；
- `host/dycc/`、`host/include/dayan_kernel.h` 定义源码切分和 Host wrapper；
- `host/driver/` 定义 Host HAL、Service、Runtime、加载、内存和 RPC 行为；
- `include/{context,rpc,thread,trap}.h` 中的静态断言约束内部 ABI；
- `linker.ld`、顶层 `Makefile` 和硬件寄存器文档约束地址与镜像布局。

接口或布局变化后，先更新实现和测试，再修改对应章节并运行 `make check`。
