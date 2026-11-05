# VEC vv int32 add single test

这是从 `dayan-samples/Instructions/VEC/vec_vv_int32_add_single.cpp` 迁移到
`dayan-base/device_tests` 的裸机版本，不修改原有的
`vec_check_int_add_test`。

## 覆盖目标

- 64 lanes、`int32`；
- single-core、single-task；
- 输入语义与原样例一致：
  ```text
  a[i] = 7 * i - 200
  b[i] = 3 * i - 70
  ```
- 执行顺序为 `VEC read A -> VEC read B -> int32 add -> VEC write`；
- 输出与 `a[i] + b[i]` 逐 lane exact compare；
- 检查输入/输出 guard 区域，发现越界写时记录 diagnostic。

## 与原 Runtime 样例的差异

原样例使用 Host Runtime 创建 buffer 并调用 `__tx_vec_vv_int32_add`。本版本遵循
`dayan-base/device_tests` 的裸机测试协议：

- 通过 `DAYAN_TEST` 注册入口；
- 使用固定 DDR/URAM 地址；
- 使用 `meta.json` 的 output/diagnostic compare；
- 使用裸机 raw emitter `0x8400820b` 发射 VEC add；
- 失败时写入 diagnostic 并调用 `dayan_thread_exit(-22)`。

## 构建

```sh
make -C device_tests VEC/vec_vv_int32_add_single_test
```

也可以构建全部 device tests：

```sh
make device-tests
```

产物位于：

```text
build/device_tests/VEC/vec_vv_int32_add_single_test/
```
