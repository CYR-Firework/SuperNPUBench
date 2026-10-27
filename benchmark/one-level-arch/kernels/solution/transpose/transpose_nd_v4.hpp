#pragma once

/**
 * @file transpose_nd_v4.hpp
 * @brief 固定 perm 特化转置 v4（对标 ops-math `conversion/transpose_v2`）
 *
 * 参考 `SuperScalar/ops-math-transpose-reference.md`：ops-math 的 `transpose_v2` 只支持
 * **三种固定 perm**：
 *   - `{0,2,1}`   （Transpose021）
 *   - `{1,0,2}`   （Transpose102）
 *   - `{0,2,1,3}` （Transpose0213 = BNSD→BSND）
 * 其余 perm 报 `Unsupported perm.`（`transpose_v2/op_host/transpose_v2_tiling.cpp:63`）。
 *
 * 本版在 PTO ISA 上用 **MGATHER** 实现这三种固定 perm——统一为"**前缀 + 交换两轴 + 后缀**"
 * 的形式：`[Prefix, X, Y, Suffix] → [Prefix, Y, X, Suffix]`。
 *
 * | Kind | 语义 | Prefix | X | Y | Suffix |
 * |---|---|---|---|---|---|
 * | `Per021`  | `[A,B,C]→[A,C,B]`     | A | B | C | 1 |
 * | `Per102`  | `[A,B,C]→[B,A,C]`     | 1 | A | B | C |
 * | `Per0213` | `[A,B,C,D]→[A,C,B,D]` | A | B | C | D |
 *
 * 偏移（输出平坦 o，输出 shape `[Prefix,Y,X,Suffix]`）：
 *   p = o/(Y*X*Sfx)；rem = o%(Y*X*Sfx)；y = rem/(X*Sfx)；x = (rem%(X*Sfx))/Sfx；s = rem%Sfx
 *   源平坦 = ((p*X + x)*Y + y)*Sfx + s
 *
 * 数据流：GM --MGATHER--> UB --TSTORE--> GM。状态：草案，未编译验证（待测试）。
 */

#include <common/pto_tileop.hpp>
#include "basic_op/utils/spmd_partition.hpp"

#include <array>
#include <cstdint>

namespace supernpu::tile_isa {

/** transpose_v2 支持的三种固定 perm */
enum class TransposeV2Perm { Per021, Per102, Per0213 };

namespace nd_v4_detail {

constexpr bool IsPow2(int v) { return v > 0 && (v & (v - 1)) == 0; }
constexpr int Log2Const(int v) {
    int r = 0;
    while (v > 1) {
        v >>= 1;
        ++r;
    }
    return r;
}
template <int Div, typename OffTile>
inline void DivModByConst(OffTile &q, OffTile &r, OffTile &lin) {
    if constexpr (IsPow2(Div)) {
        TSHRS(q, lin, static_cast<std::uint32_t>(Log2Const(Div)));
        TANDS(r, lin, static_cast<std::uint32_t>(Div - 1));
    } else {
        OffTile tmp;
        TDIVS(q, lin, static_cast<std::uint32_t>(Div));
        TMULS(tmp, q, static_cast<std::uint32_t>(Div));
        TSUB(r, lin, tmp);
    }
}

// 返回 {Prefix, X, Y, Suffix}
template <TransposeV2Perm Perm, int D0, int D1, int D2, int D3>
constexpr std::array<int, 4> V4Shape() {
    if constexpr (Perm == TransposeV2Perm::Per021) {
        return {D0, D1, D2, 1};
    } else if constexpr (Perm == TransposeV2Perm::Per102) {
        return {1, D0, D1, D2};
    } else {
        return {D0, D1, D2, D3};
    }
}

}  // namespace nd_v4_detail

/**
 * @brief 单执行单元固定 perm 特化转置（v4，MGATHER）。
 *
 * @tparam Perm 固定 perm 种类（Per021 / Per102 / Per0213）
 * @tparam DType 数据类型
 * @tparam D0,D1,D2,D3 维度尺寸（含义随 Perm 不同，见文件头表格；未用的填 1）
 * @tparam TileElements 每输出 tile 元素数（默认 512，须 32B 对齐）
 */
template <TransposeV2Perm Perm, typename DType, int D0, int D1, int D2, int D3 = 1,
          int TileElements = 512>
void transpose_nd_v4_tile(DType *input, DType *output, int tile_start = 0,
                          int tile_stride = 1, bool handle_tail = true) {
    constexpr auto kShape = nd_v4_detail::V4Shape<Perm, D0, D1, D2, D3>();
    constexpr int kPrefix = kShape[0];
    constexpr int kX = kShape[1];
    constexpr int kY = kShape[2];
    constexpr int kSfx = kShape[3];
    constexpr int kNumel = kPrefix * kX * kY * kSfx;

    static_assert(kPrefix > 0 && kX > 0 && kY > 0 && kSfx > 0, "维度必须为正");
    static_assert(TileElements > 0 && TileElements * static_cast<int>(sizeof(DType)) % 32 == 0,
                  "Tile 行必须 32 字节对齐");
    static_assert(static_cast<unsigned long long>(kNumel) * sizeof(DType) <= 0x100000000ULL,
                  "byte offset 使用 uint32，不能溢出");

    constexpr int kFullTiles = kNumel / TileElements;
    constexpr int kTailElements = kNumel % TileElements;
    constexpr int kYXS = kY * kX * kSfx;
    constexpr int kXS = kX * kSfx;

    using InputGlobal = pto::global_tensor<DType, pto::RowMajor<1, kNumel>>;
    using OutputGlobal = pto::global_tensor<DType, pto::RowMajor<1, kNumel>>;
    using DataTile = pto::Tile<pto::Location::Vec, DType, 1, TileElements, pto::BLayout::RowMajor>;
    using OffsetTile = pto::Tile<pto::Location::Vec, std::uint32_t, 1, TileElements,
                                 pto::BLayout::RowMajor>;
    using OutputIterator = global_iterator<OutputGlobal, DataTile>;

    InputGlobal input_global(input);
    OutputIterator output_iter(output);

    auto compute_off = [&](OffsetTile &lin, OffsetTile &p_c, OffsetTile &rem0, OffsetTile &y_c,
                           OffsetTile &rem1, OffsetTile &x_c, OffsetTile &s_c, OffsetTile &t1,
                           OffsetTile &t2, OffsetTile &off, std::uint32_t base) {
        TCI(lin, base);
        nd_v4_detail::DivModByConst<kYXS>(p_c, rem0, lin);   // p = o/(Y*X*S), rem0 = o%(Y*X*S)
        nd_v4_detail::DivModByConst<kXS>(y_c, rem1, rem0);   // y = rem0/(X*S), rem1 = rem0%(X*S)
        nd_v4_detail::DivModByConst<kSfx>(x_c, s_c, rem1);   // x = rem1/S, s = rem1%S
        // 源平坦 = ((p*X + x)*Y + y)*Sfx + s
        TMULS(t1, p_c, static_cast<std::uint32_t>(kX));      // p*X
        TADD(t1, t1, x_c);                                   // p*X + x
        TMULS(t1, t1, static_cast<std::uint32_t>(kY));       // *Y
        TADD(t1, t1, y_c);                                   // + y
        TMULS(t1, t1, static_cast<std::uint32_t>(kSfx));     // *Sfx
        TADD(t2, t1, s_c);                                   // + s
        TMULS(off, t2, static_cast<std::uint32_t>(sizeof(DType)));
    };

    for (int t = tile_start; t < kFullTiles; t += tile_stride) {
        auto output_global = output_iter(0, t);
        DataTile dst;
        OffsetTile lin, p_c, rem0, y_c, rem1, x_c, s_c, t1, t2, off;
        compute_off(lin, p_c, rem0, y_c, rem1, x_c, s_c, t1, t2, off,
                    static_cast<std::uint32_t>(t) * TileElements);
        MGATHER(dst, input_global, off);
        TSTORE(output_global, dst);
    }

    if constexpr (kTailElements != 0) {
        if (handle_tail) {
            using TailDataTile = pto::Tile<pto::Location::Vec, DType, 1, TileElements,
                                           pto::BLayout::RowMajor, 1, kTailElements>;
            using TailOffsetTile = pto::Tile<pto::Location::Vec, std::uint32_t, 1, TileElements,
                                             pto::BLayout::RowMajor, 1, kTailElements>;
            auto output_global = output_iter(0, kFullTiles);
            TailDataTile dst;
            TailOffsetTile lin, p_c, rem0, y_c, rem1, x_c, s_c, t1, t2, off;
            compute_off(lin, p_c, rem0, y_c, rem1, x_c, s_c, t1, t2, off,
                        static_cast<std::uint32_t>(kFullTiles) * TileElements);
            MGATHER(dst, input_global, off);
            TSTORE(output_global, dst);
        }
    }
}

}  // namespace supernpu::tile_isa

namespace supernpu::multi_thread {

template <supernpu::tile_isa::TransposeV2Perm Perm, typename DType, int D0, int D1, int D2,
          int D3 = 1, int TileElements = 512, std::uint32_t PeCount = kDefaultPeCount>
void transpose_nd_v4(DType *input, DType *output) {
    static_assert(PeCount > 0);
    const int tid = static_cast<int>(get_thread_idx());
    supernpu::tile_isa::transpose_nd_v4_tile<Perm, DType, D0, D1, D2, D3, TileElements>(
        input, output, tid, static_cast<int>(PeCount), /*handle_tail=*/tid == 0);
}

}  // namespace supernpu::multi_thread
