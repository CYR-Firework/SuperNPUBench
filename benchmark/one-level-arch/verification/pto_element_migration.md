# PTO element for 全量迁移与验证


2026-10-12 用户更新：本次所有类型、原始应用配置和四个多线程 wrapper 均以
pto-spec + public API + compiler + gfrun 为验收范围。执行时显式使用
`MODELS=gfrun`；gfsim 留到后续独立 PR，不阻塞本轮 functional 闭环。
原始规模、分配、别名语义、独立 golden 和不可变 provenance 要求保留。
历史双模型证据按原输入保留，不改写为新的执行结果。

目标是把活跃的稀疏、histogram、gather、scatter 和不规则访问 workload 写成
`#pragma pto element for` 标记的普通 C++ 元素循环。正式 TileOp API 负责
存储和 view，编译器沿 Clang CFG、LLVM SSA 和 edge mask 处理控制流与中间值。
程序员通过 `ElementTile`、`TPARTVIEW`、`TPARTELEMENT` 交替使用 TileOp 和
元素表达式，无需写物理 layout、lane 或私有生产头文件。

[JSON 清单](pto_element_migration.json) 仍以 21 条原应用路径和 4 个多线程
wrapper 为分母。`verified_profile` 只表示其证据所列源码和配置通过；未覆盖的
原 dtype、shape、算法路径和 wrapper 不能由 foundation 用例推断完成。
`verified_functional_profile` 记录本轮 pto-spec/compiler/API/gfrun 范围；
它不表示已取得 gfsim timing 证据。现有 FP32 functional 记录保持原证据。

## 已验证的范围

- 原 `element_atomic_topk` 的 777 元素输入、37 个输出、两级 256-bin
  histogram 和 coherence `[0, 9, 10, 0]` 使用 generic CFG 编译并在双模型
  通过原独立 golden。另有三份 boundary shard 和 unsharded 的 17 次调用，
  保留同一共享 workspace、空输入、K 边界和重复调用行为。
- 原 `histogram_tile_element` 和 `selected_radix_tile_element` 以原 263 元素
  输入、两个完整 Tile 和 7 元素 tail 通过 generic 编译和双模型。各 bin 的
  atomic old values 按合法无序执行检查，golden 未改变。
- 12-case suite 另覆盖 U32/S32 表达式、嵌套 CFG/continue、普通 GM 数组、
  Tile→element→Tile typed views、indexed gather 和 paired atomic CFG。
  paired atomic 用例允许 element 之间无序，但独立 DFS oracle 强制每个
  element 的第一次原子操作早于第二次，并检查两次 old value 的使用。
- 地址路径保留 S32/U32 元素索引，TLEA 扩展并缩放为 S64/U64 字节偏移。
  LLVM GEP 的 i64 表达不要求额外 64-bit 转换；相同已定型物理源共享 TLEA。
  signedness-sensitive 运算及必要的实体 descriptor retag 保留。

当前冻结证据为 `tile_element_suite/20261010-generic-atomic-integrated` 和
`element_atomic_topk/20261010-element-atomic-topk-integrated`。每例重新编译同一 ELF，
在 gfrun/gfsim 独立导出并比较主存，检查 optimized IR、object、A3 和队列守恒。
精确 clean heads、ELF/hash 和逐项结果记录在 artifact provenance 与 JSON；
库存更新本身是随后独立的文档提交，不修改已验证的 kernel、compiler 或 model。
早期 P1a/P1b foundation 证据保留为历史记录，不代替本次 generic 验证。

## FP32 functional 闭环

原 `elementwise_if_for` 的三个512-byte零缓冲区、128个FP32元素和完整
TileOp→element if→TileOp kernel，以及独立强化用例，均已通过最终
compiler/正式API构建、IR/原生指令检查和 gfrun 精确 raw-bit golden。
强化用例包含两个分支、signed-zero、相消和64-byte输出 guards；输入保持不变。
冻结证据为 `20261012-fp32-functional-original-final` 和
`20261012-fp32-functional-strong-final`。执行输入全部 clean：
compiler `a616844`、benchmark `d7e0f63`、gfrun `965d5fd2`、API `b95ac2ef`。

FP32 本轮验收按用户要求为 pto-spec + compiler + gfrun；runner 显式
`MODELS=gfrun`，选择记录在 immutable provenance。gfsim 的物理字节
definedness、masked TNEG 和 raw-part TMOV 在独立 timing PR 中继续，
functional PASS 不代表 gfsim parity。其余应用/wrapper 仍按原范围跟踪。

## FP32 masked memory functional 闭环

原 `elementwise_masked_gather_scatter` 保留五个128-element数组、原始U8 mask、
原始输出和未使用的零 `gathered_storage`。公开API加载S32索引与U32 mask
transport，通过32-element parts 执行普通C++ masked gather、raw `TMOV`、
普通C++ masked scatter。没有私有生产头文件或kernel识别器。

独立强化用例保持128-element容量，覆盖signed负索引、inactive危险索引、
非0/1字节mask、重复地址相同payload、正负zero/inf/qNaN/sNaN以及subnormal。
输出guard的实际ELF地址严格相邻。两次clean-input运行均PASS+FROZEN：
`20261012-fp32-memory-original-final` 和 `20261012-fp32-memory-strong-final`。
执行输入为compiler `c1116751a03`、benchmark `bfa549d`、API `b95ac2ef`、
gfrun `965d5fd2`。独立审计验证54项SHA256、全部5/7段原始内存以及输入不变。
编译器36项定向测试与12项oracle/破坏性canary测试通过。

此项仅新增一个FP32 functional profile：现有3个历史双模型profile、2个FP32
functional profile；其余16条原应用及4个wrapper仍未闭环。API后续修复和
整数B.ASSEMBLE工作使用独立提交/证据，不混入上述冻结版本。

## 当前缺口

- 其余16条原应用路径和4个多线程 wrapper 仍待完整迁移与验证；
  已完成 functional 验证的 FP32 应用仅剩独立 timing 跟进。
  特别是原 gather 的 rank-2 FP32 行索引/广播和四个配置，不能由 U32
  indexed-gather foundation 替代。concat 的 S32/FP16、512 元素及完整数据容量
  也仍须保留，不能缩小成 32 元素测试来宣称完成。
- 当前已验证的32-element profile包括S32/U32和FP32。普通CFG的
  if/continue/PHI、i32/f32 indexed gather、独立unit-stride scatter，以及
  明确unordered/source契约下的单一隔离indexed scatter已有编译支持。
  System/Monotonic i32 atomic add保持既有约束。FP32 memory现已取得原应用
  functional证据；这不代表任意别名或多条无序写流都可接受。
  varying内层循环、原配置所需的FP16/更宽carrier、其他原子操作/ordering、
  未证明隔离的indexed写入和O0 typed spill仍未闭合。
- API 提供某 dtype 的存储/transport，不等于该 dtype 的 marked element loop
  已可执行。QLI 私有 histogram 汇编、稀疏 MLA 的 spill/标量索引阶段、动态
  哈希探测都需要各自的完整算法迁移，不能用新的小型 histogram 替代。

## 必须取得的证据

1. 前端实际解析 canonical pragma；不支持的语义应诊断，不得忽略标记或
   无声退化成普通标量循环。
2. 检查 CFG/SSA、predicate、typed byte offset 和物理指令，保留源码的
   element 内顺序、原始输入、输出、副作用、guards 和资源容量。
3. 每个 kernel 和适用配置由指定 compiler 和正式安装 API 重新编译；
   所有本轮配置显式选择 gfrun 并记录 provenance，对照独立 golden；
   后续 timing PR 再使用同一 ELF 追加 gfsim 验证。
4. 后续 timing PR 要求 gfsim A3 为零、实际队列守恒；不通过资源/超时扩大、提前释放、替代
   reference-core 输出、finisher override 或改 golden 隐藏错误。
5. 分别记录普通/原子 memory、dtype/shape 和多线程覆盖。超时、unsupported、
   断言或缺少结果都不能计为通过。

原始 opcode conformance 用例保留直接测试目的。历史 two-level、retired
mnemonic 和缺失头文件的旧 driver 单独登记，不隐式恢复已退休 ISA 语义。

## 关联跟踪

- [NPUbench #202](https://github.com/PTO-ISA/SuperNPUBench/pull/202)
- [LLVM #118](https://github.com/LinxISA/llvm-project/pull/118)
- [TileOp API #249](https://github.com/LinxISA/Linx-TileOP-API/pull/249)
- [FP32 gfrun #949](https://github.com/LinxISA/SuperScalarModel/pull/949)
- [模型 #925](https://github.com/LinxISA/SuperScalarModel/pull/925)
- [规格 #369](https://github.com/PTO-ISA/pto-spec/pull/369)
- [集成 issue #370](https://github.com/PTO-ISA/pto-spec/issues/370)

整个目标尚未完成；上述通过结果只关闭对应的精确验证 profile。
