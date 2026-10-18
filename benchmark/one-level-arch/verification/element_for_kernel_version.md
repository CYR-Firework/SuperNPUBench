# Element-for kernel 交付版本

核心示例是 [`element_atomic_topk.hpp`](../kernels/basic_op/topk/element_atomic_topk.hpp)。
它处理 U32 承载的 16-bit key，保持 Tile 加载、元素计算、Tile 写回的源码顺序。
6 个 `#pragma pto element for` 区域分别承担四个工作区清零和两轮 radix histogram。
元素循环内的移位、按位与、条件和原子加使用普通 C++，没有手写 barrier。

```cpp
TLOAD(keys, input + begin, valid);
auto parts = TPARTVIEW<32>(keys, valid);
// 取出当前 logical part；布局由公共 API 保留。
auto view = parts.part(part);
TMOV(values, view);
const auto &key = TPARTELEMENT(values);
auto &old_value = TPARTELEMENT(old);
#pragma pto element for
for (unsigned element = 0; element < 32; ++element) {
  if (element < valid_elements)
    old_value[element] = __atomic_fetch_add(
        &high_hist[key[element] >> 8], 1u, __ATOMIC_RELAXED);
  else
    old_value[element] = 0;
}
TSTORE(workspace.old_high + begin, old, parts, part);
```

代码片段展示一个 part 的计算；完整初始化、分区和两轮选择见头文件。
cutoff 扫描和紧凑输出包含跨元素依赖，仍按普通顺序 C++ 执行。
输入无需 padding；原子 old-value 工作区按 128 元素补齐，写回按逻辑输入顺序。
重复 key 按 TopK 配额保留，输出集合无序。

QLI 的 [`qli_element.hpp`](../kernels/solution/qli/qli_element.hpp) 及
simple/dynamic kernel 还覆盖反复的 Tile reduction → element update → Tile reduction，
包括 outlined Local Tile helper、运行期边界和相同 workspace 重复调用。

## 构建与运行

使用彼此匹配的开发版 LLVM、TileOP API 和 gfrun。完整运行库提供 musl sysroot；
`COMPILER_DIR` 可以指向独立编译器 build 的 bin 目录。每个运行目录必须是新目录。

```bash
export COMPILER_DIR=/path/to/frozen-compiler/bin
export COMPILER_SOURCE_ROOT=/path/to/compiler-source
export LINX_RUNTIME_ROOT=/path/to/linx_blockisa_llvm_musl
export SYSROOT="$LINX_RUNTIME_ROOT/sysroot"
export RESOURCE_DIR=/path/to/private-clang-resource
export API_INCLUDE=/path/to/Linx-TileOP-API/include
export SSM=/path/to/SuperScalarModel
export MODEL_SOURCE_ROOT="$SSM"
export GFRUN=/path/to/frozen/gfrun
export MODELS=gfrun

ARTIFACT_DIR=/path/to/new/topk bash run_element_atomic_topk.sh
ARTIFACT_DIR=/path/to/new/suite bash run_tile_element_suite.sh
```

上述命令在本文件目录执行。primary runner 在运行目录创建私有 resource overlay，
suite runner 将选定 API 安装到 `RESOURCE_DIR`，因此后者应是本次验证专用目录。
二者都保留原有 120 秒 gfrun 超时，验证 ELF 指令、独立 golden 和运行前后的输入哈希。
失败或缺失输出不能以 finisher 文本代替正确性证明。

[solution 回归入口](run_solution_gfrun_regressions.py) 另外覆盖 LLVM #129 的 21 个
原始配置，使用隔离对象目录、逐例状态，以及已有的数值 oracle。
QLI 和原始 low-precision matmul 的运行入口见
[`pto_element_migration.md`](pto_element_migration.md)。

## 验收范围

当前验证批次的精确版本与结果写入 `pto_element_migration.json` 的最新 checkpoint。
最终验收需要包含 default suite 64 项、原 777 元素 TopK、四个真实四 PE wrapper、
17 个 TopK 边界配置，以及 QLI、predicate 和原始 matmul 的相应数值用例。
本轮只验收 gfrun。gfsim/TimingSim 另行验证；CUBE_M16/A16 完整 element 迁移及
剩余 QSMLA 配置仍保留在原始 21 应用清单中，不由 TopK 示例替代。
