#pragma once

/**
 * @file transpose_2d_v2_1.hpp
 * @brief 2D 转置 v2.1：coalesced TLOAD + 散列写（MSCATTER）
 *
 * 动机（见 `SuperScalar/transpose_working.md §8.11`）：v2 用 MGATHER 逐元素
 * **随机读**（stride 访问），实测 5.34 元素/周期，比连续 TLOAD（9.19 元素/周期）
 * 慢 ~1.72×。本版把"随机"从读侧挪到写侧：**连续 TLOAD 装载输入块**，再用
 * `MSCATTER`（TLSU，按 byte offset 散列写 GM）写到转置后的位置。
 *
 * 说明：本 ISA 无 UB 内置换指令（`TTRANS` 已退役、`__vec__` 在 `__linx` 下不可用），
 * 故"UB 内局部置换"以"写侧散列"等价实现——两个方向必有一侧 strided。
 *
 * 与 v2 对称：
 *   v2  : 输出平坦切块 → 算源 offset → MGATHER(随机读) → TSTORE(连续写)
 *   v2.1: 输入平坦切块 → 算目的 offset → TLOAD(连续读) → MSCATTER(随机写)
 *
 * 语义：output[Cols, Rows] = transpose(input[Rows, Cols])，均行主序。
 * 数据流：GM --TLOAD--> UB --MSCATTER--> GM。
 *
 * 状态：草案，未编译/未验证（待接入测试）。
 */

#include <common/pto_tileop.hpp>
#include "basic_op/utils/spmd_partition.hpp"

#include <cstdint>

namespace supernpu::tile_isa {
namespace v21_detail {

constexpr bool IsPow2(int v) { return v > 0 && (v & (v - 1)) == 0; }
constexpr int Log2Const(int v) {
    int r = 0;
    while (v > 1) {
        v >>= 1;
        ++r;
    }
    return r;
}

// dst_q = lin / Div, dst_r = lin % Div（Div 为编译期常量；2 的幂走移位/掩码）
template <int Div, typename OffTile>
inline void DivModByConst(OffTile &dst_q, OffTile &dst_r, OffTile &lin) {
    if constexpr (IsPow2(Div)) {
        TSHRS(dst_q, lin, static_cast<std::uint32_t>(Log2Const(Div)));
        TANDS(dst_r, lin, static_cast<std::uint32_t>(Div - 1));
    } else {
        OffTile tmp;
        TDIVS(dst_q, lin, static_cast<std::uint32_t>(Div));
        TMULS(tmp, dst_q, static_cast<std::uint32_t>(Div));
        TSUB(dst_r, lin, tmp);
    }
}

}  // namespace v21_detail

/**
 * @brief 单执行单元 2D 转置 v2.1（TLOAD + MSCATTER）。
 * 输入按 TileElements 的平坦区间切块，每块 TLOAD 连续读入，再按转置关系散列写到输出。
 * 输入平坦下标 o：r = o/Cols，c = o%Cols；输出平坦下标 = c*Rows + r。
 */
template <typename DType, int Rows, int Cols, int TileElements = 512>
void transpose_2d_v2_1_tile(DType *input, DType *output, int tile_start = 0,
                            int tile_stride = 1, bool handle_tail = true) {
    static_assert(Rows > 0 && Cols > 0, "矩阵维度必须为正");
    static_assert(TileElements > 0, "TileElements 必须为正");
    static_assert(TileElements * static_cast<int>(sizeof(DType)) % 32 == 0,
                  "Tile 行必须 32 字节对齐");
    static_assert(static_cast<unsigned long long>(Rows) * Cols * sizeof(DType) <=
                      0x100000000ULL,
                  "MSCATTER byte offset 使用 uint32，不能溢出");

    constexpr int kNumel = Rows * Cols;
    constexpr int kFullTiles = kNumel / TileElements;
    constexpr int kTailElements = kNumel % TileElements;

    using InputGlobal = pto::global_tensor<DType, pto::RowMajor<1, kNumel>>;
    using OutputGlobal = pto::global_tensor<DType, pto::RowMajor<1, kNumel>>;
    using DataTile = pto::Tile<pto::Location::Vec, DType, 1, TileElements,
                               pto::BLayout::RowMajor>;
    using OffsetTile = pto::Tile<pto::Location::Vec, std::uint32_t, 1, TileElements,
                                 pto::BLayout::RowMajor>;
    using InputIterator = global_iterator<InputGlobal, DataTile>;

    OutputGlobal output_global(output);  // MSCATTER 基址
    InputIterator input_iter(input);

    // ========== 完整 tile ==========
    for (int t = tile_start; t < kFullTiles; t += tile_stride) {
        auto input_global = input_iter(0, t);
        DataTile src;
        OffsetTile lin, r_coord, c_coord, idx, off;

        std::uint32_t base = static_cast<std::uint32_t>(t) * TileElements;
        TLOAD(src, input_global);  // 连续读
        TCI(lin, base);            // o = 输入平坦下标
        v21_detail::DivModByConst<Cols>(r_coord, c_coord, lin);  // r=o/Cols, c=o%Cols
        TMULS(idx, c_coord, static_cast<std::uint32_t>(Rows));   // c*Rows
        TADD(idx, idx, r_coord);                                 // c*Rows + r
        TMULS(off, idx, static_cast<std::uint32_t>(sizeof(DType)));
        MSCATTER(output_global, src, off);  // 散列写
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

            auto input_global = input_iter(0, kFullTiles);
            TailDataTile src;
            TailOffsetTile lin, r_coord, c_coord, idx, off;

            std::uint32_t base = static_cast<std::uint32_t>(kFullTiles) * TileElements;
            TLOAD(src, input_global);
            TCI(lin, base);
            v21_detail::DivModByConst<Cols>(r_coord, c_coord, lin);
            TMULS(idx, c_coord, static_cast<std::uint32_t>(Rows));
            TADD(idx, idx, r_coord);
            TMULS(off, idx, static_cast<std::uint32_t>(sizeof(DType)));
            MSCATTER(output_global, src, off);
        }
    }
}

}  // namespace supernpu::tile_isa

namespace supernpu::multi_thread {

template <typename DType, int Rows, int Columns, int TileElements = 512,
          std::uint32_t PeCount = kDefaultPeCount>
void transpose_2d_v2_1(DType *input, DType *output) {
    static_assert(PeCount > 0);
    const int tid = static_cast<int>(get_thread_idx());
    supernpu::tile_isa::transpose_2d_v2_1_tile<DType, Rows, Columns, TileElements>(
        input, output, tid, static_cast<int>(PeCount), /*handle_tail=*/tid == 0);
}

}  // namespace supernpu::multi_thread
