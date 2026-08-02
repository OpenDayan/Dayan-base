# Dayan Kernel 编程与 ABI 手册

本目录使用 LaTeX 维护 Dayan Runtime 面向算子 kernel 作者、Runtime 开发者和
Host 加载器开发者的系统手册。正文入口为 `dayan_kernel_manual.tex`，章节位于
`chapters/`，版式参考 `~/release/docs` 中的 Dayan NPU 集成手册。

## 构建

需要 XeLaTeX、latexmk、`pdfinfo`、`pdffonts` 和 `pdftotext`。系统应安装
`Noto Sans CJK SC` 与 `DejaVu Sans Mono`；常见 Linux 发行版的 Noto CJK 字体包
即可提供前者。

```sh
make
make check
```

稳定输出为 `build/dayan_kernel_manual.pdf`，同时保留带 Git 版本号的 PDF。
`make check` 会检查 PDF、字体嵌入、关键 ABI 文本，并确认 `../dayan-api.symbols`
中的每个公开符号都在 API 章节中出现。

## 内容维护约定

- `include/api/*.h` 是算子源码接口的权威定义。
- `dayan-api.symbols` 是运行时加载器允许解析的权威符号白名单。
- `include/context.h`、`include/rpc.h`、`include/trap.h`、`linker.ld` 和顶层
  `Makefile` 是 ABI 与镜像布局的权威定义。
- `~/release/docs/chapters/07_register_list.tex` 是硬件寄存器地址的权威定义。

接口或布局变化时，应先修改代码中的静态断言和测试，再同步修改本手册并执行
`make check`。
