#pragma once

/**
 * @file transpose_2d_v2.hpp
 * @brief 2D 转置新接口草案（不依赖已退役的 TTRANS）
 *
 * 背景：`TTRANS` 在 PTO-ISA 0.58.5 已退役（`jcore/template_asm.hpp:15236`
 * static_assert），现有 `tile_transpose_2d` 依赖它，需替换。
 *
 * 本接口改用 **MGATHER 索引搬运**（`jcore/template_asm.hpp:885`，TLSU 引擎，
 * "PTO 0.58.7: offset lanes are explicit byte displacements from base"）：
 * 为每个输出元素算出其源元素在输入中的 **字节偏移**，一条 MGATHER 并发取回，
 * 再 TSTORE 写出。这正是 `detail/transpose.hpp::tile_transpose_nd` 在通用 N 维
 * 转置里用的机制，本文件把它特化到 2D（[Rows,Cols] → [Cols,Rows]）。
 *
 * 已评估的其它 ISA/API 路径（不采用的原因）：
 *   - `TGATHER`（`jcore/TGather.hpp`，TEPL 111）：语义是**按行选择**取数
 *     （`dst[j,:] = src[indices[j], :]`），只能整行搬，无法表达转置的逐元素
 *     列→行重排，故不用于转置。
 *   - `TLOAD/TSTORE` 布局搬运（`TLOAD2_ND2NZ/ND2ZN/DN2ZN`、`TSTORE2_DN2DN` 等，
 *     `jcore/template_asm.hpp:529+`）：面向 ND↔NZ / DN↔ZN 的布局转换，用 DN
 *     （列主序）视图理论上可表达转置，但需配套 DN 全局张量/分块形式，契约更复杂，
 *     列为后续可选（见文末 TODO）。
 *
 * 数据流：GM → UB(Vec tile) → GM（MGATHER 直接 GM→UB，无 TLOAD）。
 *
 * 命名：`transpose_2d_v2`（多 PE 包装）+ `transpose_2d_v2_tile`（分块实现）。
 *
 * 状态：**草案，未接入测试、未编译验证**。需以 numpy `x.T` golden bit-exact
 * （atol=rtol=0）验证后再评估性能。
 */

#include <common/pto_tileop.hpp>
#include "basic_op/utils/spmd_partition.hpp"

#include <cstdint>

namespace supernpu::tile_isa {

namespace detail {
// 编译期判断与 log2（仅用于常量除数的快速路径）
constexpr bool IsPow2(int v) { return v > 0 && (v & (v - 1)) == 0; }
constexpr int Log2Const(int v) {
    int r = 0;
    while (v > 1) {
        v >>= 1;
        ++r;
    }
    return r;
}
}  // namespace detail

// 常量除数求商/余：dst_i = lin / Div，dst_j = lin % Div。
// 优化：除数为 2 的幂时，用右移 + 掩码（`TSHRS`/`TANDS`）替代 `TDIVS`（硬件除法慢）。
// 否则退化为 `TDIVS + TMULS + TSUB`；如需进一步加速，非 2 幂情形可改用
// “乘倒数再截断为整数”，但浮点倒数对一般除数不保证精确，需评估误差（见文末 TODO）。
template <int Div, typename OffTile>
inline void DivModByConst(OffTile &dst_i, OffTile &dst_j, OffTile &lin) {
    if constexpr (detail::IsPow2(Div)) {
        TSHRS(dst_i, lin, static_cast<std::uint32_t>(detail::Log2Const(Div)));
        TANDS(dst_j, lin, static_cast<std::uint32_t>(Div - 1));
    } else {
        OffTile tmp;
        TDIVS(dst_i, lin, static_cast<std::uint32_t>(Div));
        TMULS(tmp, dst_i, static_cast<std::uint32_t>(Div));
        TSUB(dst_j, lin, tmp);
    }
}

/**
 * @brief 单执行单元 2D 转置（MGATHER 索引搬运，无 TTRANS）。
 *
 * 语义：output[Cols, Rows] = transpose(input[Rows, Cols])，均为行主序。
 * 输出按 `TileElements` 的平坦区间切 tile，每个输出元素 o 的映射：
 *   i = o / Rows        （输出行 = 输入列）
 *   j = o % Rows        （输出列 = 输入行）
 *   源元素 (j, i) 的输入平坦下标 = j * Cols + i
 *   字节偏移 = (j * Cols + i) * sizeof(DType)
 *
 * @tparam DType         数据类型
 * @tparam Rows          输入行数
 * @tparam Cols          输入列数
 * @tparam TileElements  每个输出 tile 的元素数（默认 512，须 32B 对齐）
 * @param input        输入指针（Rows × Cols，行主序）
 * @param output       输出指针（Cols × Rows，行主序）
 * @param tile_start   本执行单元处理的起始输出 tile 序号
 * @param tile_stride  输出 tile 调度步长（多 PE 用）
 * @param handle_tail  是否负责唯一的尾部 tile
 */
template <typename DType, int Rows, int Cols, int TileElements = 512>
void transpose_2d_v2_tile(DType *input, DType *output, int tile_start = 0,
                          int tile_stride = 1, bool handle_tail = true) {
    // ========== 编译期安全检查 ==========
    static_assert(Rows > 0 && Cols > 0, "矩阵维度必须为正");
    static_assert(TileElements > 0, "TileElements 必须为正");
    static_assert(TileElements * static_cast<int>(sizeof(DType)) % 32 == 0,
                  "Tile 行必须 32 字节对齐");
    static_assert(static_cast<unsigned long long>(Rows) * Cols * sizeof(DType) <=
                      0x100000000ULL,
                  "MGATHER byte offset 使用 uint32，不能溢出");

    constexpr int kNumel = Rows * Cols;
    constexpr int kFullTiles = kNumel / TileElements;
    constexpr int kTailElements = kNumel % TileElements;

    // ========== 类型定义 ==========
    // 输入/输出展平为一维（转置不改变存储总量，只改逻辑索引）
    using InputGlobal = pto::global_tensor<DType, pto::RowMajor<1, kNumel>>;
    using OutputGlobal = pto::global_tensor<DType, pto::RowMajor<1, kNumel>>;
    using DataTile = pto::Tile<pto::Location::Vec, DType, 1, TileElements,
                               pto::BLayout::RowMajor>;
    using OffsetTile = pto::Tile<pto::Location::Vec, std::uint32_t, 1, TileElements,
                                 pto::BLayout::RowMajor>;
    using OutputIterator = global_iterator<OutputGlobal, DataTile>;

    InputGlobal input_global(input);
    OutputIterator output_iter(output);

    // ========== 完整 tile ==========
    for (int t = tile_start; t < kFullTiles; t += tile_stride) {
        auto output_global = output_iter(0, t);
        DataTile dst;
        OffsetTile lin, i_coord, j_coord, idx, off;

        std::uint32_t base = static_cast<std::uint32_t>(t) * TileElements;
        TCI(lin, base);                          // o = [base, base+TileElements)
        DivModByConst<Rows>(i_coord, j_coord, lin);  // i = o / Rows, j = o % Rows
        TMULS(idx, j_coord, static_cast<std::uint32_t>(Cols));  // j * Cols
        TADD(idx, idx, i_coord);                                // j * Cols + i
        TMULS(off, idx, static_cast<std::uint32_t>(sizeof(DType)));  // byte offset

        MGATHER(dst, input_global, off);  // 按 byte offset 从 GM 取数到 UB
        TSTORE(output_global, dst);       // 写回输出 GM
    }

    // ========== 尾部 tile ==========
    if constexpr (kTailElements != 0) {
        if (handle_tail) {
            using TailDataTile =
                pto::Tile<pto::Location::Vec, DType, 1, TileElements,
                          pto::BLayout::RowMajor, 1, kTailElements>;
            using TailOffsetTile =
                pto::Tile<pto::Location::Vec, std::uint32_t, 1, TileElements,
                          pto::BLayout::RowMajor, 1, kTailElements>;

            auto output_global = output_iter(0, kFullTiles);
            TailDataTile dst;
            TailOffsetTile lin, i_coord, j_coord, idx, off;

            std::uint32_t base = static_cast<std::uint32_t>(kFullTiles) * TileElements;
            TCI(lin, base);
            DivModByConst<Rows>(i_coord, j_coord, lin);
            TMULS(idx, j_coord, static_cast<std::uint32_t>(Cols));
            TADD(idx, idx, i_coord);
            TMULS(off, idx, static_cast<std::uint32_t>(sizeof(DType)));

            MGATHER(dst, input_global, off);
            TSTORE(output_global, dst);
        }
    }
}

}  // namespace supernpu::tile_isa

namespace supernpu::multi_thread {

/**
 * @brief 多 PE 2D 转置（新接口，MGATHER 路径）。
 *
 * 输出平坦 tile 按 round-robin 分给各 PE（PE `tid` 从第 `tid` 个 tile 起、
 * 步长 `PeCount`），各 PE 写互不相交的输出区间；尾部 tile 由 PE0 负责。
 * 调用方需在所有 PE 上调用本接口。
 */
template <typename DType, int Rows, int Columns, int TileElements = 512,
          std::uint32_t PeCount = kDefaultPeCount>
void transpose_2d_v2(DType *input, DType *output) {
    static_assert(PeCount > 0);
    const int tid = static_cast<int>(get_thread_idx());
    supernpu::tile_isa::transpose_2d_v2_tile<DType, Rows, Columns, TileElements>(
        input, output, tid, static_cast<int>(PeCount), /*handle_tail=*/tid == 0);
}

}  // namespace supernpu::multi_thread

// ============================================================================
// TODO / 待确认（延续 transpose_working.md）
//
// 1. 数值验证：接入 `test/kernel/transpose/` 后，须以 numpy `x.T` golden
//    bit-exact（atol=rtol=0）通过。注意 MGATHER 索引单位在不同 PTO 版本存在
//    byte-offset / element 两种语义（见 `kernels/basic_op/README.md:258`、
//    `test/solution/common/README.md:35`）；本文件按 0.58.7 的“byte displacement”
//    语义编写，换版本时首先核对这一点。
// 2. 性能：MGATHER 为逐元素随机 GM 访问，可能不如（已退役的）块转置；后续可选：
//    (a) TLOAD/TSTORE 布局搬运（`TLOAD2_DN2ZN`/`TSTORE2_DN2DN` 等，用列主序视图表达转置）；
//    (b) 若目标后端支持 `__vec__`（当前 `utils/layout_transform.hpp` 注明
//        “SIMT constructs not supported under __linx”），可写块向量转置 kernel。
//    两者均需进一步查证契约与可用性，不在本草案内。
// 3. UB 预算与 TileElements 取值（默认 512）需按目标平台实测调整。
// ============================================================================
