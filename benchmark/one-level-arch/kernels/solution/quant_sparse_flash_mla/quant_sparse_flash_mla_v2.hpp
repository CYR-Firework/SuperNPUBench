#ifndef QUANT_SPARSE_FLASH_MLA_V2_HPP
#define QUANT_SPARSE_FLASH_MLA_V2_HPP

// =============================================================================
// quant_sparse_flash_mla_v2.hpp — 统一五模式四 PE TADD QSMLA kernel 的
// 动态 shape 变体。
//
// 与 quant_sparse_flash_mla.hpp（静态版本）的差异：
//   1. 所有 Vec 引擎 Tile 使用 DYNAMIC ValidRow/ValidCol。运行时通过
//      Tile(VR, VC) 构造函数按 KV 块设定有效区域，因此 B.DIM 编码的是
//      寄存器形式（"B.DIM %[reg], 0"）而非立即数形式
//      （"B.DIM zero, %c[imm]"）。
//   2. Vec tile 类型保留 DYNAMIC ValidRow/ValidCol（寄存器形式 B.DIM），
//      但 tW / tMask / tPShard 的运行时 valid 固定为全宽 kTk：CUBE 载体
//      上 TLOAD_CUBE 不编码 lb2 且链上生产者的物理 col 均由 valid 派生，
//      与行归约（TROWMAX / TROWSUM）lb2 的静态 Cols 在部分块时不一致
//      （模型 Block.cpp 行归约描述符契约拒绝）。软件 mask（-1e30）保证
//      全宽处理数值逐位等价；待上游补齐 CUBE 物理列编码后恢复动态
//      ValidCol。
//   3. Shared tile（协作 TMATMUL 操作数）与 Cube 累加器 tile 保持静态
//      ——PTO v0.58 要求协作矩阵操作使用编译期有效形状（TMATMUL
//      static_assert）。
//   4. gmGatherKV / 迭代器不变（它们消费物理形状）。
//   5. Vec tile 使用 M32 CELL 格式（BLayout::CubeM32，PTO-ISA #291）：
//      物理 carrier 为 32 行 CELL 列阵（128B/CELL），elementwise /
//      TCVT / TLOAD / TSTORE 经 B.DATR CUBE_M32 与 ND2M32 / M322ND
//      自动编码。行归约遵循 PTO #311：目的地是保持源物理列跨度的
//      宽载体（tileMaxWide / tileSumWide，valid [kPeRows, 1]），随后用
//      TREDUCEPREFIXVIEW 借出首个 128B CELL 作为紧凑行值视图参与
//      后续算术（TMAX / TADD 等带 reduction-prefix 重载）。行状态
//      tile（m / l）的有效行恒为 kPeRows，故为静态 valid。
//
// 文件结构（自上而下）：
//   Section 1 — QsmlaV2Tiles：共享类型目录（tile / GM tensor /
//               迭代器别名与维度常量）。
//   Section 2 — KV 块 staging 辅助函数（标量 GM -> buffer 拷贝）。
//   Section 3 — span 解码：work item -> (ori 窗口/索引, cmp 数量)。
//   Section 4 — QsmlaV2PassEnv：所有 visitor 共享的 per-PE kernel
//               上下文，外加 (m, l, O) 状态生命周期辅助函数。
//   Section 5 — qsmla_v2_compute_score_tile：经协作 TMATMUL 计算的
//               Q @ K^T（pass1 / pass2 共用）。
//   Section 6 — qsmla_v2_visit_source_pass1 / pass2：对单个 KV 源的
//               两趟 online-softmax。
//   Section 7 — qsmla_v2_store_output：HIF8 反缩放 + 类型转换 + GM
//               写回。
//   Section 8 — kernel 入口：仅做 work 循环编排。
//
// 重构约束：tile 对象永远在消费它的函数内部声明（tile 跨函数边界传递
// 会让 clang 后端 spill 出 raw tile store——参见模型的 RawTileSourceFits
// 断言）。跨函数传递的状态仅限 QsmlaV2PassEnv，其成员均为标量状态包装
// （指针 / 迭代器 / 浮点数）。
// =============================================================================

#include <type_traits>
#include <common/pto_tileop.hpp>
// 方案 3（2026-09-24）：不再 include test/common/template_asm.h——其
// MGATHER/MSCATTER 旧封装（BSTART.TMA 4/5 助记符已废弃）与 TileOP 新版
// （BSTART.TLSU MGATHER）同名且重载决议更特化，会拦截调用并产生
// "Match Instruction Error"。v2 不使用该头的任何其他符号。
#include "qsmla_config.hpp"
#include "qsmla_mode.hpp"

using namespace pto;

// =============================================================================
// Section 1 — Tile 类型目录
//
// 协作 TMATMUL 操作数与 Cube 累加器为 STATIC（PTO v0.58："Matrix dynamic
// valid shapes are not supported"）；所有 Vec 引擎 tile 为 DYNAMIC
// （运行时 ValidRow/ValidCol）。
// =============================================================================
template <typename qdtype, typename kvdtype, typename odttype,
          typename Config>
struct QsmlaV2Tiles {
    static constexpr int kPeNum = 4;
    static constexpr int kGroupM = Config::TileM;
    static constexpr int kTk = Config::TileK;
    static constexpr int kTd = Config::TileD;
    static constexpr int kD = Config::D;
    static constexpr int kDb = Config::D / kTd;
    static constexpr int kPeRows = kGroupM / kPeNum;
    static constexpr float kHif8ProbabilityScale = 16.0f;
    static constexpr bool kUseHif8Probability =
        std::is_same_v<qdtype, __hif8>;

    // ---- 协作 TMATMUL 操作数与累加器 ----
    // B（K/V）保持 Shared（SharedMatrixRight，四 PE 共享一次加载）；
    // A（Q/P）改为 per-PE 本地 CUBE_M32 Left 分片 [kPeRows, ·]——
    // CubeOutputLayout 跟随本地 A 的布局，使累加器得以使用
    // CubeAccumulatorM32（纯 Shared A/B 时模型按 localM=16 强制派生
    // CUBE_M16，与 M32 载体的 TSTORE_CUBE M322ND 冲突）。Local-A /
    // Shared-B 协作形式要求显式传入 core-total group_M（TMATMUL 五参
    // 重载，LB0 编码 group_M 而非分片行数）。
    using tileKMatrix = SharedMatrixRight<kvdtype, kTk, kTd>;
    using tileVMatrix = SharedMatrixRight<kvdtype, kTk, kTd>;
    using tileKShared = SharedTile<tileKMatrix>;
    using tileVShared = SharedTile<tileVMatrix>;
    using tileQLocal = CubeTileM32<qdtype, kPeRows, kTd>;   // per-PE Q 分片
    using tilePLocal = CubeTileM32<qdtype, kPeRows, kTk>;   // per-PE P 分片
    using tileScoreCube = CubeAccumulatorM32<float, kPeRows, kTk>;
    using tilePVCube = CubeAccumulatorM32<float, kPeRows, kTd>;

    // ---- 动态 Vec tile（M32 CELL 格式）----
    // 物理形状仍为编译期 [kPeRows, kTk] 的 M32 CELL 列阵（存储行高恒为
    // 32，128B/CELL），但有效区域在运行时设定。动态维度的 B.DIM 使用
    // 寄存器形式（"B.DIM %[reg], 0"）。
    // Step C/D（2026-09-23，模式门控）：优化模式（csa/swa/hca/
    // ori_cmp）tileW 族静态化（TCVT CUBE_M16/M32 要求 src/dst valid
    // shape 一致）+ O/PV 半驻留载体（Vec location + CubeM32 布局静态
    // valid）。ORI_SPARSE 模式 gfsim 在静态化 tileW 下死锁（TMA
    // l1d_refill_unmatched，gfrun 正常——时序模型缺口），整体保持
    // 原始 DYNAMIC 形态。
#if defined(QSMLA_USE_CSA_TADD_4PE) || \
    defined(QSMLA_USE_TADD_4PE) || \
    defined(QSMLA_USE_HCA_TADD_4PE) || \
    defined(QSMLA_USE_ORI_CMP_SPARSE_TADD_4PE)
    using tileW = VecTileM32<float, kPeRows, kTk>;
    using tileMask = tileW;
    using tilePShard = VecTileM32<qdtype, kPeRows, kTk>;
    using tileOCube = VecTileM32<float, kPeRows, kTd>;
    using tileOCastS = VecTileM32<odttype, kPeRows, kTd>;
#define QSMLA_TW_DECL_W typename Tiles::tileW tW;
#define QSMLA_TW_DECL_M typename Tiles::tileMask tMask;
#define QSMLA_TFINAL_DECL \
    typename Tiles::tileOCube tFinalO; \
    typename Tiles::tileOCastS tOCast;
#else
    using tileW =
        VecTileM32<float, kPeRows, kTk, DYNAMIC, DYNAMIC>;
    using tileMask = tileW;
    using tilePShard =
        VecTileM32<qdtype, kPeRows, kTk, DYNAMIC, DYNAMIC>;
#define QSMLA_TW_DECL_W typename Tiles::tileW tW(kPeRows, kTk);
#define QSMLA_TW_DECL_M typename Tiles::tileMask tMask(kPeRows, kTk);
#define QSMLA_TFINAL_DECL \
    typename Tiles::tileO tFinalO(kPeRows, kTd); \
    typename Tiles::tileOCast tOCast(kPeRows, kTd);
#endif
    using tileO =
        VecTileM32<float, kPeRows, kTd, DYNAMIC, DYNAMIC>;
    using tileOCast =
        VecTileM32<odttype, kPeRows, kTd, DYNAMIC, DYNAMIC>;
    // 行状态（m / l）：有效形状恒为 [kPeRows, 1]，静态声明——
    // TREDUCEPREFIXVIEW 的 SubTile 与宽载体 ValidRow 必须一致，且
    // view 的 GetValidRow() 返回编译期常量，DYNAMIC 会得到 -1。
    using tileMax = VecTileM32<float, kPeRows, 1, kPeRows, 1>;
    using tileSum = tileMax;
    // PTO #311 行归约宽载体：物理保持源的列跨度 [32, kTk]，有效区域
    // [kPeRows, 1]；归约结果落在首个 CELL，由 TREDUCEPREFIXVIEW 借出。
    using tileMaxWide = VecTileM32<float, kPeRows, kTk, kPeRows, 1>;
    using tileSumWide = tileMaxWide;

    // ---- GM tensor 与迭代器（物理形状）----
    using gmQ = global_tensor<qdtype, RowMajor<kGroupM, Config::D>>;
    using gmGatherKV = global_tensor<kvdtype, RowMajor<kTk, Config::D>>;
    using gmO = global_tensor<odttype, RowMajor<kGroupM, Config::D>>;
    // Q 的 per-PE 分片迭代器：[kGroupM, D] 按 [kPeRows, kTd] 分块，
    // (pe_id, dd) 即本 PE 的 Q 行分片。
    using itQLocal = global_iterator<gmQ, tileQLocal>;
    using itK = global_iterator<gmGatherKV, tileKMatrix>;
    using itV = global_iterator<gmGatherKV, tileVMatrix>;
    using itO = global_iterator<gmO, tileOCast>;

    using gmScoreScratch = global_tensor<float, RowMajor<kPeRows, kTk>>;
    using gmProbScratch = global_tensor<qdtype, RowMajor<kGroupM, kTk>>;
    using gmPVScratch = global_tensor<float, RowMajor<kPeRows, Config::D>>;
    using gmRowState = global_tensor<float, RowMajor<kPeRows, 1>>;
    using itProbShard = global_iterator<gmProbScratch, tilePShard>;
    // P 分片的本地读回迭代器：prob_scratch [kGroupM, kTk] 按
    // [kPeRows, kTk] 分块，(pe_id, 0) 即本 PE 写入的 P 分片。
    using itPLocal = global_iterator<gmProbScratch, tilePLocal>;
    using itPVScratch = global_iterator<gmPVScratch, tileO>;
    using itRowState = global_iterator<gmRowState, tileMax>;
    using gmMask = global_tensor<float, RowMajor<kPeRows, kTk>>;
    using itMask = global_iterator<gmMask, tileMask>;

    // ---- 方案 3（2026-09-24）：MGATHER 引擎化 indexed gather ----
    // 替代 staging 的标量逐 word 拷贝（45K cycles/段的纯标量独占）。
    // IndexTile[i][j] = selected[i]*kD + j（元素下标，MGATHER 契约），
    // 构造链：TCOLEXPAND(iota[1,kD] 广播) + TROWEXPANDADD(行首[kTk,1])
    // + MGATHER([kTk,kD] HIF8) + TSTORE(落 staging GM 视图)。
    // 标量残留：每块 ≤kTk 次行首下标写 + kernel 级一次 iota 构造。
    // 注意：gather 链全部使用 RowMajor/NORM 布局（Tile 默认布局）——
    // TCOLEXPAND 的广播源要求物理 one-row（VecTileM32 的 M32 载体物理
    // 行高恒 32，[1,N] 逻辑行映射到 32 物理行会被 gfrun 的 COPY
    // expansion 广播形状断言拒绝）；MGATHER dst 只做 staging 中转，
    // 走 NORM 选择器（#207 的 CUBE_M32 是给直连 CUBE 链准备的），
    // 之后 TSTORE 以 ND 行主序落盘，与 itK 的 RowMajor 读取匹配。
    // 分片化（2026-09-28）：IndexTile 按 kGatherTile=32 列切片。两个目的：
    // ① TCOLEXPAND 的 one-row 广播源受 gfsim 的 CELL 粒度约束
    //    （broadcastElements <= 128B/elementBytes = 32@FP32）——[1,kD] 整行
    //    广播触发 "unary EXPAND" 断言；[1,32] 切片恰好压线。
    // ② MGATHER 的 IndexTile 读回通路（RWDB→TileReg bridge→CellReg）在
    //    单指令 64KB/512 beats 大批量读下停滞（sent236/recv184 悬空）；
    //    [32,32] S32 切片 = 4KB/32 beats，回到该通路被验证过的量级。
    // FP32 做扩展算术（gfsim 的 TROWEXPAND 只建模 float 路径），TCVT 落
    // S32 下标（值域均为 FP32 精确整数，RNE 无损）。
    static constexpr int kGatherTile = 32;
    static constexpr int kGatherParts = kD / kGatherTile;
    using tileGatherIota = Tile<Location::Vec, float, 1, kGatherTile>;
    using tileGatherRow = Tile<Location::Vec, float, kTk, 1>;
    using tileGatherIdxF = Tile<Location::Vec, float, kTk, kGatherTile>;
    using tileGatherIdx = Tile<Location::Vec, int32_t, kTk, kGatherTile>;
    using tileGatherDst = Tile<Location::Vec, kvdtype, kTk, kGatherTile>;
    using gmGatherIota = global_tensor<float, RowMajor<1, kGatherTile>>;
    using gmGatherRow = global_tensor<float, RowMajor<kTk, 1>>;
    // staging 落盘：按 [kTk, kGatherTile] 分片的迭代器（gmGatherKV 的
    // 列分块，片 p 即 (0, p)——与 itK 的 (0, dd) 迭代同构）。
    using itGatherStage = global_iterator<gmGatherKV, tileGatherDst>;
};

// =============================================================================
// Section 2 — KV 块 staging 辅助函数
//
// 一个 KV 块要么直接从 GM 读取（连续、满 kTk 行），要么 staging 到
// PE 私有的 kv_tile_buf（索引 gather / 尾块）。软件 mask 用 -inf 标记
// 补齐行；v2 额外通过动态 ValidCol 把真实几何交给硬件。
// =============================================================================
template <typename KVType>
struct QsmlaV2KVBlock {
    KVType* tile_ptr;  // 直接 GM 指针或 staging 后的 buffer
    int valid_rows;    // 本块内逻辑 KV token 数（<= kTk）
};

// ---- 性能优化（2026-09-23，借鉴 ops-transformer arch35 的零填充省略）：
// 尾块只拷贝 valid_rows 行，无效行不再清零——残留行（上一块的 staging
// 数据）产生的垃圾 score 会被软件 mask（-1e30）压制为精确 0（exp 下溢
// → P=0 → 对 l 与 PV 均无贡献），数值逐位等价。----
template <typename Env, typename KVType, int kTk, int kD>
static __attribute__((always_inline)) inline void qsmla_v2_stage_source_tile(
    Env& env, KVType* kv_tile_buf, KVType* source, const int* selected,
    int range_begin, int logical_begin, int valid_rows)
{
    if constexpr (std::is_same_v<KVType, __hif8>) {
        // ---- 方案 3：MGATHER 引擎化（HIF8 路径）----
        // 标量只写行首元素下标（≤kTk 次，非 kTk*kD/4 次 word 拷贝）；
        // 布局/搬运全部交给 tile 链。尾块行首填 0（gather 到池首行，
        // 无效行由软件 mask 压制，数值等价）。
        using Tiles = typename Env::Tiles;
        for (int row = 0; row < kTk; ++row) {
            env.gather_row_buf[row] = row < valid_rows
                ? static_cast<float>(selected[logical_begin + row] * kD)
                : 0.0f;
        }
        typename Tiles::gmGatherRow gRow(env.gather_row_buf);
        typename Tiles::tileGatherRow tRow;
        TLOAD(tRow, gRow);
        typename Tiles::gmGatherKV gPool(source);
        typename Tiles::itGatherStage itStage(kv_tile_buf);
        // 片间依赖链（S32 TADD 读上一片的 index tile）：强制 16 片
        // MGATHER 在 TLSU 流内串行发射——四线程的同 instId 实例并发
        // 挤入 InstBuffer（16 深）时存在竞争丢失（B89 案例：stid=0
        // 请求未入 buffer、3 入者只激活 1 个），串行化后任一时刻只有
        // 一个 instId 的 4 实例在 SL2，规避该缺口。链结果不被消费。
        typename Tiles::tileGatherIdx tChain;
        TCI(tChain, 0);
        for (int part = 0; part < Tiles::kGatherParts; ++part) {
            // iota 片天然携带列基址（iota_buf[p*32+j] = p*32+j）
            typename Tiles::gmGatherIota gIota(
                env.gather_iota_buf + part * Tiles::kGatherTile);
            typename Tiles::tileGatherIota tIota;
            TLOAD(tIota, gIota);
            typename Tiles::tileGatherIdxF tIdxF0;
            TCOLEXPAND(tIdxF0, tIota);
            typename Tiles::tileGatherIdxF tIdxF;
            TROWEXPANDADD(tIdxF, tIdxF0, tRow);
            typename Tiles::tileGatherIdx tIdx;
            TCVT(tIdx, tIdxF);
            TADD(tChain, tChain, tIdx);  // 片间串行链
            typename Tiles::tileGatherDst tDst;
            MGATHER(tDst, gPool, tIdx);
            auto gStagePart = itStage(0, part);
            TSTORE(gStagePart, tDst);
        }
        return;
    }
    // FP16 路径：保留标量 staging（HIF8 是主优化目标）。
    for (int row = 0; row < kTk; ++row) {
        const int source_row = row < valid_rows
            ? (selected == nullptr
                   ? range_begin + logical_begin + row
                   : selected[logical_begin + row])
            : 0;
        if constexpr (std::is_same_v<KVType, __hif8>) {
            auto* destination = reinterpret_cast<uint32_t*>(
                kv_tile_buf + row * kD);
            auto* source_words = reinterpret_cast<const uint32_t*>(
                source + source_row * kD);
            for (int word = 0; word < kD / 4; ++word) {
                destination[word] =
                    row < valid_rows ? source_words[word] : 0U;
            }
        } else {
            for (int dim = 0; dim < kD; ++dim) {
                kv_tile_buf[row * kD + dim] =
                    row < valid_rows
                        ? source[source_row * kD + dim]
                        : static_cast<KVType>(0.0f);
            }
        }
    }
}

// ---- 性能优化（2026-09-23）：mask 增量构造——满块（valid_rows==kTk）
// mask 全 0（mask_buf 由 work 级清零维护），直接返回；尾块只写无效列
// 段 [valid_rows, kTk)。标量写从 kPeRows×kTk/块 降到接近 0。----
template <int kPeRows, int kTk>
static __attribute__((always_inline)) inline void qsmla_v2_build_source_mask(
    float* mask_buf, int valid_rows)
{
#if !defined(QSMLA_USE_TADD_4PE)
    // 满块零构造：TLOAD 读 zero_mask_buf（kernel 级一次清零）。
    // SWA（TADD_4PE）除外——其 mask 优化路径触发 gfsim BFU
    // GetLocalPipeID 断言（fbid→local pipe 映射缺口，形态无关），
    // 回退原全量构造。
    if (valid_rows == kTk) {
        return;
    }
#endif
    for (int row = 0; row < kPeRows; ++row) {
        for (int column = 0; column < kTk; ++column) {
            mask_buf[row * kTk + column] =
                column < valid_rows ? 0.0f : -1.0e30f;
        }
    }
}

template <typename Env, typename KVType, int kTk, int kD, int kPeRows>
static __attribute__((always_inline)) inline QsmlaV2KVBlock<KVType> qsmla_v2_prepare_kv_block(
    Env& env, KVType* kv_tile_buf, float* mask_buf,
    KVType* source, const int* selected,
    int range_begin, int logical_begin, int logical_count,
    int allow_direct)
{
    const int remaining = logical_count - logical_begin;
    const int valid_rows = remaining < kTk ? remaining : kTk;
    const bool direct_contiguous =
        qsmla_use_direct_contiguous_tile(
            allow_direct, selected != nullptr, valid_rows, kTk);
    KVType* tile_ptr = kv_tile_buf;
    if (direct_contiguous) {
        tile_ptr = source + (range_begin + logical_begin) * kD;
    } else {
        qsmla_v2_stage_source_tile<Env, KVType, kTk, kD>(
            env, kv_tile_buf, source, selected, range_begin,
            logical_begin, valid_rows);
    }
    qsmla_v2_build_source_mask<kPeRows, kTk>(mask_buf, valid_rows);
    return QsmlaV2KVBlock<KVType>{tile_ptr, valid_rows};
}

// =============================================================================
// Section 3 — span 解码（work item -> KV 源 span）
// =============================================================================

// ORI span：SWA/HCA/CSA 为 [begin, begin+count) 窗口；ORI_SPARSE /
// ORI_CMP_SPARSE 为收集后的 top-k 索引列表。
struct QsmlaV2OriSpan {
    int begin;  // 首个窗口 token（索引模式为 0）
    int count;  // 待访问的逻辑 KV token 数
};

template <typename ModeConfig, typename Config>
static __attribute__((always_inline)) inline QsmlaV2OriSpan qsmla_v2_decode_ori_span(
    const QsmlaWorkItem& work, int ori_win_left, int ori_win_right,
    const int* ori_topk_length, const int* ori_sparse_indices,
    int* ori_selected)
{
    QsmlaV2OriSpan span{0, 0};
    if constexpr (ModeConfig::Mode == QsmlaMode::SWA ||
                  ModeConfig::Mode == QsmlaMode::HCA ||
                  ModeConfig::Mode == QsmlaMode::CSA) {
        const QsmlaSwaRange range = qsmla_swa_range(
            ModeConfig::OriS2, Config::S1, work.q_token,
            ori_win_left, ori_win_right);
        span.begin = range.begin;
        span.count = range.end - range.begin;
    } else if constexpr (ModeConfig::HasIndexedOri) {
        const std::size_t list_offset =
            ((static_cast<std::size_t>(work.batch) * Config::S1
              + work.q_token) * Config::N2 + work.kv_head)
            * ModeConfig::OriTopK;
        const std::size_t length_offset =
            (static_cast<std::size_t>(work.batch) * Config::S1
             + work.q_token) * Config::N2 + work.kv_head;
        int candidate_count = ori_topk_length[length_offset];
        candidate_count = qsmla_sparse_clamp(
            candidate_count, 0, ModeConfig::OriTopK);
        span.count = qsmla_sparse_collect_indices(
            ori_selected, ModeConfig::OriTopK,
            ori_sparse_indices + list_offset, candidate_count,
            ModeConfig::OriS2,
            qsmla_sparse_ori_valid_end(
                ModeConfig::OriS2, Config::S1, work.q_token));
    }
    return span;
}

// CMP span：HCA 为稠密前缀；CSA / ORI_CMP 为收集后的 top-k。
template <typename ModeConfig, typename Config>
static __attribute__((always_inline)) inline int qsmla_v2_decode_cmp_span(
    const QsmlaWorkItem& work, int cmp_ratio,
    const int* cmp_topk_length, const int* cmp_sparse_indices,
    int* cmp_selected)
{
    int count = 0;
    if constexpr (ModeConfig::HasCmp) {
        const int cmp_valid_end = qsmla_csa_cmp_valid_end(
            ModeConfig::CmpS2, Config::S1,
            work.q_token, cmp_ratio);
        if constexpr (ModeConfig::Mode == QsmlaMode::HCA) {
            count = cmp_valid_end;
        } else {
            const std::size_t list_offset =
                ((static_cast<std::size_t>(work.batch) * Config::S1
                  + work.q_token) * Config::N2 + work.kv_head)
                * ModeConfig::CmpTopK;
            const std::size_t length_offset =
                (static_cast<std::size_t>(work.batch) * Config::S1
                 + work.q_token) * Config::N2 + work.kv_head;
            int candidate_count = ModeConfig::CmpTopK;
            if (cmp_topk_length != nullptr) {
                candidate_count = cmp_topk_length[length_offset];
            }
            candidate_count = qsmla_sparse_clamp(
                candidate_count, 0, ModeConfig::CmpTopK);
            count = qsmla_sparse_collect_indices(
                cmp_selected, ModeConfig::CmpTopK,
                cmp_sparse_indices + list_offset, candidate_count,
                ModeConfig::CmpS2, cmp_valid_end);
        }
    }
    return count;
}

// =============================================================================
// Section 4 — per-PE kernel 上下文与 (m, l, O) 状态生命周期
//
// QsmlaV2PassEnv 打包 KV 块 visitor 所需的、跨 work item 不变的全部
// 状态：score / 行状态 / P staging / PV scratch 区域的 GM 视图与迭代器、
// kernel 级标量、以及 PE 私有 staging buffer。所有成员均为标量状态包装
// ——不含任何 tile 对象。
// =============================================================================
template <typename qdtype, typename kvdtype, typename odttype,
          typename Config>
struct QsmlaV2PassEnv {
    using Tiles = QsmlaV2Tiles<qdtype, kvdtype, odttype, Config>;

    // GM 视图 / 迭代器（轻量指针包装）。
    typename Tiles::gmScoreScratch gScore;   // PE 私有 score staging
    typename Tiles::itRowState gIterMax;     // (m) 行状态
    typename Tiles::itRowState gIterSum;     // (l) 行状态
    typename Tiles::itProbShard gIterProb;   // P staging 的 PE 分片
    typename Tiles::itPLocal gIterPLocal;     // 本 PE 的 P 分片读回视图
    typename Tiles::itPVScratch gIterPV;     // PE 私有 O 累加器

    // kernel 级不变量。
    float softmax_scale;
    float q_descale;
    int pe_id;

    // PE 私有标量 scratch buffer。
    float* mask_buf;
    float* zero_mask_buf;  // 满块 mask 源（恒全 0，kernel 级一次清零）
    float* gather_iota_buf;  // iota [kD] FP32（kernel 级一次构造）
    float* gather_row_buf;   // 行首元素下标 [kTk] FP32（每块 ≤kTk 次标量写）
    kvdtype* kv_tile_buf;
};

// 复位 online-softmax 行状态：m = -inf，l = 0。
template <typename Env>
static __attribute__((always_inline)) inline void qsmla_v2_init_row_state(Env& env)
{
    using Tiles = typename Env::Tiles;
    typename Tiles::tileMax tMax;
    typename Tiles::tileSum tSum;
    TEXPANDS(tMax, -1e30f);
    TEXPANDS(tSum, 0.0f);
    auto gMaxState = env.gIterMax(0, 0);
    auto gSumState = env.gIterSum(0, 0);
    TSTORE(gMaxState, tMax);
    TSTORE(gSumState, tSum);
}

// pass1 之后：把 l 替换为 1/l，供 pass2 原地缩放 P。
template <typename Env>
static __attribute__((always_inline)) inline void qsmla_v2_recip_row_state(Env& env)
{
    using Tiles = typename Env::Tiles;
    auto gSumState = env.gIterSum(0, 0);
    typename Tiles::tileSum tFinalSum;
    TLOAD(tFinalSum, gSumState);
    typename Tiles::tileSum tInvSum;
    TRECIP(tInvSum, tFinalSum);
    TSTORE(gSumState, tInvSum);
}

// pass2 之前：清零 PE 私有 O 累加器。
template <typename Env>
static __attribute__((always_inline)) inline void qsmla_v2_reset_o_state(Env& env)
{
    using Tiles = typename Env::Tiles;
    constexpr int kPeRows = Tiles::kPeRows;
    constexpr int kTd = Tiles::kTd;
    constexpr int kDb = Tiles::kDb;
#pragma clang loop unroll(disable)
    for (int out_dd = 0; out_dd < kDb; ++out_dd) {
        typename Tiles::tileO tZeroO(kPeRows, kTd);
        TEXPANDS(tZeroO, 0.0f);
        auto gOState = env.gIterPV(0, out_dd);
        TSTORE(gOState, tZeroO);
    }
}

// =============================================================================
// Section 5 — Score tile（Q @ K^T）
//
// 本 PE 的 Q 分片（本地 CUBE_M32 Left，[kPeRows, kTd]）与当前 KV 块的
// Shared K 送入 Local-A/Shared-B 协作 TMATMUL（显式 group_M = kGroupM），
// 把 kDb 个 D 切片累加进一个 M32 Cube score tile，再落到 PE 的 score
// scratch。pass1 与 pass2 共用（两趟方案会重算同一条 QK^T 链）。
// =============================================================================
template <typename Env, typename QIter, typename KIter>
static __attribute__((always_inline)) inline void qsmla_v2_compute_score_tile(
    Env& env, QIter q_iter, KIter k_iter)
{
    using Tiles = typename Env::Tiles;
    constexpr int kDb = Tiles::kDb;
    constexpr int kGroupM = Tiles::kGroupM;
    typename Tiles::tileScoreCube tScoreCube;
#pragma clang loop unroll(full)
    for (int dd = 0; dd < kDb; ++dd) {
        typename Tiles::tileQLocal tQLocal;
        typename Tiles::tileKShared tKShared;
        auto gQ = q_iter(env.pe_id, dd);
        auto gK = k_iter(0, dd);
        TLOAD_CUBE(tQLocal, gQ);
        TLOAD<typename Tiles::tileKMatrix, 1>(tKShared, gK);
        if (dd == 0) {
            TMATMUL(tScoreCube, tQLocal, tKShared,
                    fixp::keep_acc(), kGroupM);
        } else {
            TMATMUL_ACC(tScoreCube, tScoreCube, tQLocal, tKShared,
                        fixp::keep_acc(), kGroupM);
        }
    }

    TSTORE_CUBE(env.gScore, tScoreCube);
}

// =============================================================================
// Section 6 — 单 KV 源的逐块 visitor
//
// 两趟 pass 均按 kTk 宽的块遍历一个 KV 源（先 ORI 窗口/列表，模式带
// CMP 时再走 CMP）。score tile 每块重算；pass1 维护 (m, l)；pass2 消费
// 最终的 (m, 1/l) 构造量化 P 分片并把 PV 累加进 O。
//
// 注意："score tile 缩放 + 加 mask" 序列无法提取成辅助函数，因为
// tileW 必须保持函数局部（见文件头约束）；两个 pass 中有意保留重复。
// =============================================================================
template <typename Env, typename QIter, typename KVType>
static __attribute__((always_inline)) inline void qsmla_v2_visit_source_pass1(
    Env& env, QIter q_iter,
    KVType* source, const int* selected,
    int range_begin, int logical_count, int allow_direct,
    float kv_descale)
{
    using Tiles = typename Env::Tiles;
    constexpr int kTk = Tiles::kTk;
    constexpr int kD = Tiles::kD;
    constexpr int kPeRows = Tiles::kPeRows;
    const float score_scale =
        env.softmax_scale * env.q_descale * kv_descale;
    const int block_count = (logical_count + kTk - 1) / kTk;
    auto gMaxState = env.gIterMax(0, 0);
    auto gSumState = env.gIterSum(0, 0);
    for (int block = 0; block < block_count; ++block) {
        const int logical_begin = block * kTk;
        const QsmlaV2KVBlock<KVType> kv = qsmla_v2_prepare_kv_block<Env, KVType, kTk, kD, kPeRows>(
            env, env.kv_tile_buf, env.mask_buf, source, selected,
            range_begin, logical_begin, logical_count, allow_direct);
        typename Tiles::itK gIterK(kv.tile_ptr);

        typename Tiles::tileMax tMax;
        typename Tiles::tileSum tSum;
        TLOAD(tMax, gMaxState);
        TLOAD(tSum, gSumState);

        qsmla_v2_compute_score_tile(env, q_iter, gIterK);

        // M32 载体按全宽 valid 运行（kTk）：TLOAD_CUBE 不编码 lb2，CUBE
        // 链上所有生产者的物理 col 都从 valid 区域派生，而 TROWMAX /
        // TROWSUM 的 lb2 用 tile 类型的静态 Cols——部分块时两者不一致
        // 会被模型的行归约描述符契约拒绝（Block.cpp:2536）。软件 mask
        //（-1e30）保证全宽处理数值逐位等价：exp(-1e30 - m) 下溢为精确
        // 0，行 max 由有效列主导。动态 ValidCol 待上游补齐 CUBE 加载
        // 的物理列编码后恢复。
        QSMLA_TW_DECL_W
        TLOAD(tW, env.gScore);
        TMULS(tW, tW, score_scale);
#if defined(QSMLA_USE_TADD_4PE)
        typename Tiles::itMask gIterMask(env.mask_buf);
#else
        float* maskSrc = env.mask_buf;
        if (kv.valid_rows == kTk) {
            maskSrc = env.zero_mask_buf;
        }
        typename Tiles::itMask gIterMask(maskSrc);
#endif
        QSMLA_TW_DECL_M
        auto gMask = gIterMask(0, 0);
        TLOAD(tMask, gMask);
        TADD(tW, tW, tMask);

        // PTO #311：行归约写入保持源物理列跨度的宽载体，再用
        // TREDUCEPREFIXVIEW 借出首个 CELL 作为紧凑行值参与算术。
        typename Tiles::tileMaxWide tLocalMaxWide;
        TROWMAX(tLocalMaxWide, tW);
        auto tLocalMax =
            TREDUCEPREFIXVIEW<typename Tiles::tileMax>(tLocalMaxWide);
        typename Tiles::tileMax tNewMax;
        TMAX(tNewMax, tMax, tLocalMax);
        typename Tiles::tileMax tScale;
        TSUB(tScale, tMax, tNewMax);
        TEXP(tScale, tScale);
        typename Tiles::tileSum tScaledOldSum;
        TMUL(tScaledOldSum, tSum, tScale);
        TROWEXPANDSUB(tW, tW, tNewMax);
        TEXP(tW, tW);
        typename Tiles::tileSumWide tLocalSumWide;
        TROWSUM(tLocalSumWide, tW);
        auto tLocalSum =
            TREDUCEPREFIXVIEW<typename Tiles::tileSum>(tLocalSumWide);
        TADD(tSum, tScaledOldSum, tLocalSum);
        tMax = tNewMax;
        TSTORE(gMaxState, tMax);
        TSTORE(gSumState, tSum);
    }
}

// ---- Step C/D 模式门控（2026-09-23）：P 直连 + PV fixpipe 半驻留。
// csa/swa 已验证（gfrun 逐位一致 + gfsim 通过）；ORI_SPARSE 的 gfsim
// 在此组合下死锁（TMA l1d_refill_unmatched → Deadlock，gfrun 正常），
// 时序模型缺口待上游排查，该模式预处理器层回退基线路径。----
#if defined(QSMLA_USE_CSA_TADD_4PE) || \
    defined(QSMLA_USE_TADD_4PE) || \
    defined(QSMLA_USE_HCA_TADD_4PE) || \
    defined(QSMLA_USE_ORI_CMP_SPARSE_TADD_4PE)
#define QSMLA_PV_PATH(TW, KV_DESCALE, GITV) \
    typename Tiles::tilePLocal tPLocal; \
    TCVT(tPLocal, (TW)); \
    TSTORE(gMaxState, tMax); \
    TSTORE(gSumState, tInvSum); \
    _Pragma("clang loop unroll(disable)") \
    for (int out_dd = 0; out_dd < kDb; ++out_dd) { \
        auto gOState = env.gIterPV(0, out_dd); \
        typename Tiles::tileOCube tO; \
        TLOAD(tO, gOState); \
        typename Tiles::tileVShared tVShared; \
        auto gV = (GITV)(0, out_dd); \
        TLOAD<typename Tiles::tileVMatrix, 1>(tVShared, gV); \
        typename Tiles::tileOCube tPV; \
        TMATMUL(tPV, tPLocal, tVShared, \
                fixp::convert<FixpPreQuantMode::None>().transpose_b(), \
                kGroupM); \
        TMULS(tPV, tPV, (KV_DESCALE)); \
        TADD(tO, tO, tPV); \
        TSTORE(gOState, tO); \
    }
#else
#define QSMLA_PV_PATH(TW, KV_DESCALE, GITV) \
    typename Tiles::tilePShard tPShard(kPeRows, kTk); \
    TCVT(tPShard, (TW)); \
    auto gProbShard = env.gIterProb(env.pe_id, 0); \
    TSTORE(gProbShard, tPShard); \
    TSTORE(gMaxState, tMax); \
    TSTORE(gSumState, tInvSum); \
    _Pragma("clang loop unroll(disable)") \
    for (int out_dd = 0; out_dd < kDb; ++out_dd) { \
        auto gOState = env.gIterPV(0, out_dd); \
        typename Tiles::tileO tO(kPeRows, kTd); \
        TLOAD(tO, gOState); \
        typename Tiles::tilePLocal tPLocal; \
        auto gPShard = env.gIterPLocal(env.pe_id, 0); \
        TLOAD_CUBE(tPLocal, gPShard); \
        typename Tiles::tileVShared tVShared; \
        auto gV = (GITV)(0, out_dd); \
        TLOAD<typename Tiles::tileVMatrix, 1>(tVShared, gV); \
        typename Tiles::tilePVCube tPVCube; \
        TMATMUL(tPVCube, tPLocal, tVShared, \
                fixp::keep_acc().transpose_b(), kGroupM); \
        TSTORE_CUBE(gOState, tPVCube); \
        typename Tiles::tileO tPV(kPeRows, kTd); \
        TLOAD(tPV, gOState); \
        TMULS(tPV, tPV, (KV_DESCALE)); \
        TADD(tO, tO, tPV); \
        TSTORE(gOState, tO); \
    }
#endif

template <typename Env, typename QIter, typename KVType>
static __attribute__((always_inline)) inline void qsmla_v2_visit_source_pass2(
    Env& env, QIter q_iter,
    KVType* source, const int* selected,
    int range_begin, int logical_count, int allow_direct,
    float kv_descale)
{
    using Tiles = typename Env::Tiles;
    constexpr int kTk = Tiles::kTk;
    constexpr int kD = Tiles::kD;
    constexpr int kTd = Tiles::kTd;
    constexpr int kPeRows = Tiles::kPeRows;
    constexpr int kDb = Tiles::kDb;
    constexpr int kGroupM = Tiles::kGroupM;
    const float score_scale =
        env.softmax_scale * env.q_descale * kv_descale;
    const int block_count = (logical_count + kTk - 1) / kTk;
    auto gMaxState = env.gIterMax(0, 0);
    auto gSumState = env.gIterSum(0, 0);
    for (int block = 0; block < block_count; ++block) {
        const int logical_begin = block * kTk;
        const QsmlaV2KVBlock<KVType> kv = qsmla_v2_prepare_kv_block<Env, KVType, kTk, kD, kPeRows>(
            env, env.kv_tile_buf, env.mask_buf, source, selected,
            range_begin, logical_begin, logical_count, allow_direct);
        typename Tiles::itK gIterK(kv.tile_ptr);
        typename Tiles::itV gIterV(kv.tile_ptr);

        typename Tiles::tileMax tMax;
        typename Tiles::tileSum tInvSum;
        TLOAD(tMax, gMaxState);
        TLOAD(tInvSum, gSumState);

        qsmla_v2_compute_score_tile(env, q_iter, gIterK);

        // 全宽 tileW（见 pass1 注释：CUBE 部分有效列与行归约 lb2 契约
        // 不兼容，mask 保证等价）。
        QSMLA_TW_DECL_W
        TLOAD(tW, env.gScore);
        TMULS(tW, tW, score_scale);
#if defined(QSMLA_USE_TADD_4PE)
        typename Tiles::itMask gIterMask(env.mask_buf);
#else
        float* maskSrc = env.mask_buf;
        if (kv.valid_rows == kTk) {
            maskSrc = env.zero_mask_buf;
        }
        typename Tiles::itMask gIterMask(maskSrc);
#endif
        QSMLA_TW_DECL_M
        auto gMask = gIterMask(0, 0);
        TLOAD(tMask, gMask);
        TADD(tW, tW, tMask);
        TROWEXPANDSUB(tW, tW, tMax);
        TEXP(tW, tW);
        TROWEXPANDMUL(tW, tW, tInvSum);
        if constexpr (Tiles::kUseHif8Probability) {
            TMULS(tW, tW, Tiles::kHif8ProbabilityScale);
        }

        QSMLA_PV_PATH(tW, kv_descale, gIterV)
    }
}

// =============================================================================
// Section 7 — 输出收尾：撤销 HIF8 概率缩放、转换到输出 dtype、把该 PE
// 的 O 行写回 GM。
// =============================================================================
template <typename Env, typename OutType>
static __attribute__((always_inline)) inline void qsmla_v2_store_output(
    Env& env, OutType* out_ptr,
    std::size_t work_out_offset)
{
    using Tiles = typename Env::Tiles;
    constexpr int kD = Tiles::kD;
    constexpr int kTd = Tiles::kTd;
    constexpr int kPeRows = Tiles::kPeRows;
    constexpr int kDb = Tiles::kDb;
#pragma clang loop unroll(disable)
    for (int out_dd = 0; out_dd < kDb; ++out_dd) {
        auto gOState = env.gIterPV(0, out_dd);
        QSMLA_TFINAL_DECL
        TLOAD(tFinalO, gOState);
        if constexpr (Tiles::kUseHif8Probability) {
            TMULS(tFinalO, tFinalO,
                  1.0f / Tiles::kHif8ProbabilityScale);
        }
        TCVT(tOCast, tFinalO);
        typename Tiles::itO gIterO(out_ptr + work_out_offset
                                   + env.pe_id * kPeRows * kD);
        auto gO = gIterO(0, out_dd);
        TSTORE(gO, tOCast);
    }
}

// =============================================================================
// Section 8 — kernel 入口：仅做 work 循环编排。
//
// 每个 work item：解码 ORI/CMP span → 跑 pass1（online softmax m, l）→
// 把 l 翻转为 1/l、清零 O → 跑 pass2（P 量化 + PV 累加）→ 写回输出。
// 所有重活都在 Section 2-7。
// =============================================================================
template <typename qdtype, typename kvdtype, typename odttype,
          typename ModeConfig>
void quant_sparse_flash_mla_tadd_4pe_bsnd_pto_v2(
    odttype* out_ptr,
    qdtype* q_ptr,
    kvdtype* ori_kv_ptr,
    kvdtype* cmp_kv_ptr,
    const int* ori_sparse_indices,
    const int* cmp_sparse_indices,
    const int* ori_topk_length,
    const int* cmp_topk_length,
    float softmax_scale,
    float q_descale,
    float ori_kv_descale,
    float cmp_kv_descale,
    int cmp_ratio,
    int ori_win_left,
    int ori_win_right,
    float* score_scratch,
    qdtype* prob_scratch,
    float* pv_scratch)
{
    using Config = typename ModeConfig::Base;
    using Tiles = QsmlaV2Tiles<qdtype, kvdtype, odttype, Config>;
    using Env = QsmlaV2PassEnv<qdtype, kvdtype, odttype, Config>;
    constexpr int kGroupM = Tiles::kGroupM;
    constexpr int kPeRows = Tiles::kPeRows;
    constexpr int kOriIndexStorage =
        ModeConfig::OriTopK > 0 ? ModeConfig::OriTopK : 1;
    constexpr int kCmpIndexStorage =
        ModeConfig::CmpTopK > 0 ? ModeConfig::CmpTopK : 1;

    static_assert(Config::N2 == 1,
                  "sparse four-PE BSND requires contiguous N2=1 KV");
    static_assert(Config::GSliceMax == 64 && Config::G % 64 == 0,
                  "sparse four-PE requires complete 64-head G slices");
    static_assert(kGroupM == 64 && Tiles::kTk == 32,
                  "sparse v1 fixes TileM=64 and TileK=32");
    static_assert(Config::D % Tiles::kTd == 0,
                  "sparse four-PE D-tail support is deferred");
    static_assert(!std::is_same_v<kvdtype, __hif8> || Config::D % 4 == 0,
                  "HIF8 gather uses four-byte carriers");

    const int pe_id = static_cast<int>(get_thread_idx());

    constexpr int kMaskElements = kPeRows * Tiles::kTk;
    float mask_buf[kMaskElements];
    float zero_mask_buf[kMaskElements];
    for (int i = 0; i < kMaskElements; ++i) {
        zero_mask_buf[i] = 0.0f;
    }
    // 方案 3：gather 辅助数组（iota 一次构造；行首下标每块重写）。
    float gather_iota_buf[Config::D];
    for (int j = 0; j < Config::D; ++j) {
        gather_iota_buf[j] = static_cast<float>(j);
    }
    float gather_row_buf[Tiles::kTk];
    kvdtype kv_tile_buf[Tiles::kTk * Config::D];
    int ori_selected[kOriIndexStorage];
    int cmp_selected[kCmpIndexStorage];

    // PE 私有 scratch 布局（共享 GM，见 harness 约定）：
    //   score_scratch: [PE, kPeRows*kTk] score staging + (m, l) 行状态
    //   prob_scratch : [PE, kPeRows*kTk] P 分片，再按全宽读回
    //   pv_scratch   : [PE, kPeRows*D]   O 累加器
    float* pe_score_scratch =
        score_scratch + pe_id * kPeRows * Tiles::kTk;
    Env env{
        typename Tiles::gmScoreScratch(pe_score_scratch),
        typename Tiles::itRowState(pe_score_scratch),
        typename Tiles::itRowState(pe_score_scratch + kPeRows),
        typename Tiles::itProbShard(prob_scratch),
        typename Tiles::itPLocal(prob_scratch),
        typename Tiles::itPVScratch(
            pv_scratch +
            qsmla_full_o_scratch_pe_offset(pe_id, kPeRows, Config::D)),
        softmax_scale,
        q_descale,
        pe_id,
        mask_buf,
        zero_mask_buf,
        gather_iota_buf,
        gather_row_buf,
        kv_tile_buf,
    };

    for (int work_id = 0; work_id < Config::WorkCount; ++work_id) {
        const QsmlaWorkItem work = Config::decode_work(work_id);
        qdtype* work_q = q_ptr + Config::q_work_offset(work);
        const std::size_t work_out_offset = Config::out_work_offset(work);
        typename Tiles::itQLocal q_iter(work_q);

        kvdtype* work_ori = ori_kv_ptr +
            ((static_cast<std::size_t>(work.batch) * ModeConfig::OriS2)
             * Config::N2 + work.kv_head) * Config::D;
        kvdtype* work_cmp = nullptr;
        if constexpr (ModeConfig::HasCmp) {
            work_cmp = cmp_kv_ptr +
                ((static_cast<std::size_t>(work.batch) * ModeConfig::CmpS2)
                 * Config::N2 + work.kv_head) * Config::D;
        }

        const QsmlaV2OriSpan ori_span = qsmla_v2_decode_ori_span<ModeConfig, Config>(
            work, ori_win_left, ori_win_right,
            ori_topk_length, ori_sparse_indices, ori_selected);
        const int cmp_count = qsmla_v2_decode_cmp_span<ModeConfig, Config>(
            work, cmp_ratio,
            cmp_topk_length, cmp_sparse_indices, cmp_selected);

        // Pass 1：两个源上的 online softmax 统计量 (m, l)。
        qsmla_v2_init_row_state(env);
        qsmla_v2_visit_source_pass1(
            env, q_iter, work_ori,
            ModeConfig::HasIndexedOri ? ori_selected : nullptr,
            ori_span.begin, ori_span.count, true, ori_kv_descale);
        if constexpr (ModeConfig::HasCmp) {
            qsmla_v2_visit_source_pass1(
                env, q_iter, work_cmp,
                ModeConfig::HasIndexedCmp ? cmp_selected : nullptr,
                0, cmp_count, false, cmp_kv_descale);
        }

        // Pass 2：量化 P + PV 累加。
        qsmla_v2_recip_row_state(env);
        qsmla_v2_reset_o_state(env);
        qsmla_v2_visit_source_pass2(
            env, q_iter, work_ori,
            ModeConfig::HasIndexedOri ? ori_selected : nullptr,
            ori_span.begin, ori_span.count, true, ori_kv_descale);
        if constexpr (ModeConfig::HasCmp) {
            qsmla_v2_visit_source_pass2(
                env, q_iter, work_cmp,
                ModeConfig::HasIndexedCmp ? cmp_selected : nullptr,
                0, cmp_count, false, cmp_kv_descale);
        }

        qsmla_v2_store_output(env, out_ptr, work_out_offset);
    }
}

#endif // QUANT_SPARSE_FLASH_MLA_V2_HPP
