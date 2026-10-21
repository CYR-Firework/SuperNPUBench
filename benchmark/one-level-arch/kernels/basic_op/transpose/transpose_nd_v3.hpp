#pragma once

/**
 * @file transpose_nd_v3.hpp
 * @brief 通用 N 维 permute 转置 v3（对标 ops-math `conversion/transpose` 通用算子）
 *
 * 参考 `SuperScalar/ops-math-transpose-reference.md`：ops-math 的 `transpose` 是
 * **通用维度置换算子**（任意轴顺序、≤8 维，对应 aclnnPermute）。本版在 PTO ISA 上
 * 用 **MGATHER 索引搬运** 实现"任意 perm"的通用转置。
 *
 * 与 v4 的差异：
 *   - **v3 支持任意 perm**（输出第 i 轴 = 输入第 Perm[i] 轴），Rank 1..8；对标 `transpose`；
 *   - v4 只支持固定 perm（对标 `transpose_v2`：`{0,2,1}`/`{1,0,2}`/`{0,2,1,3}`）。
 *
 * 算法（与 detail/transpose.hpp 的 nd 路径同构，把"交换两轴"推广为"任意置换"）：
 *   对每个输出平坦下标 o：
 *     对 input_dim 从内到外：
 *       output_dim = inverse_perm[input_dim]
 *       out_stride = ∏ out_shape[output_dim+1..Rank-1]
 *       coord = (o / out_stride) % out_shape[output_dim]   // 该输出的坐标 = 对应输入的坐标
 *       offset += coord * in_stride * sizeof(DType)
 *       in_stride *= in_shape[input_dim]
 *   再 MGATHER / TSTORE。
 *
 * 数据流：GM --MGATHER--> UB --TSTORE--> GM。
 *
 * 状态：草案，未编译/未验证（待接入测试）。
 */

#include <common/pto_tileop.hpp>
#include "basic_op/utils/spmd_partition.hpp"

#include <array>
#include <cstdint>

namespace supernpu::tile_isa {
namespace nd_v3_detail {

// 由正 perm 关系（输出轴 i = 输入轴 Perm[i]）构造逆映射：inv[input_dim] = output_dim
template <int Rank, int... Perm>
constexpr std::array<int, Rank> MakeInversePerm() {
    int p[Rank > 0 ? Rank : 1] = {Perm...};
    std::array<int, Rank> inv{};
    for (int i = 0; i < Rank; ++i) {
        inv[p[i]] = i;
    }
    return inv;
}

// 单点逆映射：返回 output_dim（满足 Perm[output_dim] == input_dim）
template <int Rank, int... Perm>
constexpr int InvAt(int inputDim) {
    const int p[Rank > 0 ? Rank : 1] = {Perm...};
    for (int i = 0; i < Rank; ++i) {
        if (p[i] == inputDim) return i;
    }
    return 0;
}

// 正向 perm 数组：perm[i] = Perm[i]
template <int Rank, int... Perm>
constexpr std::array<int, Rank> MakePerm() {
    return {Perm...};
}

}  // namespace nd_v3_detail

/**
 * @brief 单执行单元通用 permute 转置（v3，MGATHER）。
 *
 * @tparam DType        数据类型
 * @tparam Rank         维度数（1..8）
 * @tparam Elements     总元素数（编译期）
 * @tparam TileElements 每个输出 tile 的元素数（默认 512，须 32B 对齐）
 * @tparam Perm         置换：输出轴 i 对应输入轴 Perm[i]（长度 = Rank）
 * @param input     输入指针（按 in_shape 行主序展平）
 * @param output    输出指针（按 out_shape 行主序展平）
 * @param in_shape  输入各维尺寸（长度 Rank）
 * @param out_shape 输出各维尺寸（out_shape[i] = in_shape[Perm[i]]，长度 Rank）
 */
template <typename DType, int Rank, int Elements, int TileElements = 512, int... Perm>
void transpose_nd_v3_tile(DType *input, DType *output, const std::uint32_t *in_shape,
                          const std::uint32_t *out_shape, int tile_start = 0,
                          int tile_stride = 1, bool handle_tail = true) {
    static_assert(sizeof...(Perm) == Rank, "Perm 长度必须等于 Rank");
    static_assert(Rank >= 1 && Rank <= 8, "Rank 支持 1..8");
    static_assert(TileElements > 0 && TileElements * static_cast<int>(sizeof(DType)) % 32 == 0,
                  "Tile 行必须 32 字节对齐");
    static_assert(static_cast<unsigned long long>(Elements) * sizeof(DType) <= 0x100000000ULL,
                  "MGATHER byte offset 使用 uint32，不能溢出");

    constexpr auto kInv = nd_v3_detail::MakeInversePerm<Rank, Perm...>();  // kInv[input_dim]=output_dim
    static_assert([=] {
        const int p[Rank > 0 ? Rank : 1] = {Perm...};
        for (int i = 0; i < Rank; ++i) {
            if (kInv[p[i]] != i) return false;
        }
        return true;
    }(), "kInv 与 Perm 不一致（逆映射错误）");
    constexpr int kFullTiles = Elements / TileElements;
    constexpr int kTailElements = Elements % TileElements;

    using InputGlobal = pto::global_tensor<DType, pto::RowMajor<1, Elements>>;
    using OutputGlobal = pto::global_tensor<DType, pto::RowMajor<1, Elements>>;
    using DataTile = pto::Tile<pto::Location::Vec, DType, 1, TileElements, pto::BLayout::RowMajor>;
    using OffsetTile = pto::Tile<pto::Location::Vec, std::uint32_t, 1, TileElements,
                                 pto::BLayout::RowMajor>;
    using OutputIterator = global_iterator<OutputGlobal, DataTile>;

    InputGlobal input_global(input);
    OutputIterator output_iter(output);

    for (int t = tile_start; t < kFullTiles; t += tile_stride) {
        auto output_global = output_iter(0, t);
        DataTile dst;
        OffsetTile lin, quotient, cycle, cycle_base, coordinate, contribution, off_tile;

        std::uint32_t base = static_cast<std::uint32_t>(t) * TileElements;
        TCI(lin, base);  // 输出平坦下标 o
        TEXPANDS(off_tile, static_cast<std::uint32_t>(0));
        std::uint32_t istr[Rank];
        {
            std::uint32_t acc = 1;
            for (int k = Rank - 1; k >= 0; --k) {
                istr[k] = acc;
                acc *= in_shape[k];
            }
        }
        #pragma clang loop unroll(full)
        for (int od = 0; od < Rank; ++od) {
            std::uint32_t out_stride = 1;
            #pragma clang loop unroll(full)
            for (int dim = od + 1; dim < Rank; ++dim) {
                out_stride *= out_shape[dim];
            }
            if (out_stride == 1) {
                TCVT(quotient, lin);
            } else {
                TDIVS(quotient, lin, out_stride);
            }
            TDIVS(cycle, quotient, out_shape[od]);
            TMULS(cycle_base, cycle, out_shape[od]);
            TSUB(coordinate, quotient, cycle_base);
            TMULS(contribution, coordinate,
                  istr[nd_v3_detail::MakePerm<Rank, Perm...>()[od]] *
                      static_cast<std::uint32_t>(sizeof(DType)));
            TADD(off_tile, off_tile, contribution);
        }
#ifdef V3_DUMP_OFF
        {
            using UGlobal = pto::global_tensor<std::uint32_t, pto::RowMajor<1, Elements>>;
            using UIter = global_iterator<UGlobal, OffsetTile>;
            UIter uit(reinterpret_cast<std::uint32_t *>(output));
            auto go = uit(0, t);
            TSTORE(go, off_tile);
        }
#else
        MGATHER(dst, input_global, off_tile);
        TSTORE(output_global, dst);
#endif
    }

    if constexpr (kTailElements != 0) {
        if (handle_tail) {
            using TailDataTile = pto::Tile<pto::Location::Vec, DType, 1, TileElements,
                                           pto::BLayout::RowMajor, 1, kTailElements>;
            using TailOffsetTile = pto::Tile<pto::Location::Vec, std::uint32_t, 1, TileElements,
                                             pto::BLayout::RowMajor, 1, kTailElements>;
            auto output_global = output_iter(0, kFullTiles);
            TailDataTile dst;
            TailOffsetTile lin, quotient, cycle, cycle_base, coordinate, contribution, off_tile;
            std::uint32_t base = static_cast<std::uint32_t>(kFullTiles) * TileElements;
            TCI(lin, base);
            TEXPANDS(off_tile, static_cast<std::uint32_t>(0));
            std::uint32_t istr[Rank];
            {
                std::uint32_t acc = 1;
                for (int k = Rank - 1; k >= 0; --k) {
                    istr[k] = acc;
                    acc *= in_shape[k];
                }
            }
            #pragma clang loop unroll(full)
            for (int od = 0; od < Rank; ++od) {
                std::uint32_t out_stride = 1;
                #pragma clang loop unroll(full)
                for (int dim = od + 1; dim < Rank; ++dim) {
                    out_stride *= out_shape[dim];
                }
                if (out_stride == 1) {
                    TCVT(quotient, lin);
                } else {
                    TDIVS(quotient, lin, out_stride);
                }
                TDIVS(cycle, quotient, out_shape[od]);
                TMULS(cycle_base, cycle, out_shape[od]);
                TSUB(coordinate, quotient, cycle_base);
                TMULS(contribution, coordinate,
                      istr[nd_v3_detail::MakePerm<Rank, Perm...>()[od]] *
                          static_cast<std::uint32_t>(sizeof(DType)));
                TADD(off_tile, off_tile, contribution);
            }
            MGATHER(dst, input_global, off_tile);
            TSTORE(output_global, dst);
        }
    }
}

}  // namespace supernpu::tile_isa

namespace supernpu::multi_thread {

/**
 * @brief 多 PE 通用 permute 转置（v3）。
 * 输出平坦 tile 按 round-robin 分给各 PE；尾部 tile 由 PE0 负责。
 * 注意：PeCount 需显式指定（在 Perm 之前），如 `transpose_nd_v3<T,4,N,512,4, 0,2,1,3>`。
 */
template <typename DType, int Rank, int Elements, int TileElements = 512,
          std::uint32_t PeCount = kDefaultPeCount, int... Perm>
void transpose_nd_v3(DType *input, DType *output, const std::uint32_t *in_shape,
                     const std::uint32_t *out_shape) {
    static_assert(PeCount > 0);
    const int tid = static_cast<int>(get_thread_idx());
    supernpu::tile_isa::transpose_nd_v3_tile<DType, Rank, Elements, TileElements, Perm...>(
        input, output, in_shape, out_shape, tid, static_cast<int>(PeCount), /*handle_tail=*/tid == 0);
}

}  // namespace supernpu::multi_thread
