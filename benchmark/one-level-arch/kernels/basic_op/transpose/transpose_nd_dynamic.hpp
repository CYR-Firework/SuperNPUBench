#pragma once

/**
 * @file transpose_nd_dynamic.hpp
 * @brief 动态 shape 通用 permute 转置（运行时 rank / shape / perm）
 *
 * 对标 ops-math `conversion/transpose` 的"动态 shape"形态：**编译期只固定 tile 容量**，
 * `rank / in_shape / out_shape / perm / numel` **全部运行时决定**（host 侧运行时算，传参进 kernel）。
 *
 * 与 v3/v3.1 的差异：
 *   - v3.1：rank/shape/perm 全**编译期**常量（Spec 模板包）；
 *   - v3  ：rank/perm/Elements 编译期，shape 运行时**值**；
 *   - **本版：rank/shape/perm/numel 全运行时**（CapElements 只是容量上界）。
 *
 * 算法：按输出平坦 tile 分块；每元素用**运行时 rank 循环**做坐标分解求输入 byte offset；
 * 尾块用 `TMIN` 把线性索引 clamp 到 `numel-1`（避免越界读），输出需**补齐到 tile 整数倍**。
 *
 * 数据流：GM --MGATHER--> UB --TSTORE--> GM。
 */

#include <common/pto_tileop.hpp>
#include "basic_op/utils/spmd_partition.hpp"

#include <cstdint>

namespace supernpu::tile_isa {

/**
 * @brief 动态 shape 通用 permute 转置（单执行单元）。
 *
 * @tparam DType       数据类型
 * @tparam CapElements 容量上界（编译期；≥ 补齐后的元素数）
 * @tparam TileElements 每 tile 元素数（编译期，须 32B 对齐）
 * @tparam PeCount     PE 数
 * @param input     输入指针（容量 CapElements）
 * @param output    输出指针（**必须补齐到 ceil(numel/TileElements)*TileElements**）
 * @param rank      维数（运行时）
 * @param in_shape  输入各维尺寸（运行时，长度 rank）
 * @param out_shape 输出各维尺寸（运行时，out_shape[i]=in_shape[perm[i]]）
 * @param perm      置换（运行时，输出轴 i = 输入轴 perm[i]）
 * @param numel     总元素数（运行时）
 */
template <typename DType, int CapElements, int TileElements = 512,
          std::uint32_t PeCount = supernpu::multi_thread::kDefaultPeCount>
void transpose_nd_dynamic_tile(DType *input, DType *output, int rank,
                               const std::uint32_t *in_shape, const std::uint32_t *out_shape,
                               const std::uint32_t *inv_perm, std::uint32_t numel,
                               int pe_id) {
    static_assert(CapElements > 0 && TileElements > 0 &&
                      TileElements * static_cast<int>(sizeof(DType)) % 32 == 0,
                  "Tile 行必须 32 字节对齐");
    static_assert(static_cast<unsigned long long>(CapElements) * sizeof(DType) <= 0x100000000ULL,
                  "offset 使用 uint32");

    using InputGlobal = pto::global_tensor<DType, pto::RowMajor<1, CapElements>>;
    using OutputGlobal = pto::global_tensor<DType, pto::RowMajor<1, CapElements>>;
    using DataTile = pto::Tile<pto::Location::Vec, DType, 1, TileElements, pto::BLayout::RowMajor>;
    using OffsetTile = pto::Tile<pto::Location::Vec, std::uint32_t, 1, TileElements,
                                 pto::BLayout::RowMajor>;
    using OutputIterator = global_iterator<OutputGlobal, DataTile>;

    InputGlobal input_global(input);
    OutputIterator output_iter(output);

    const std::uint32_t kTiles =
        (numel + static_cast<std::uint32_t>(TileElements) - 1) /
        static_cast<std::uint32_t>(TileElements);

    for (std::uint32_t t = static_cast<std::uint32_t>(pe_id); t < kTiles;
         t += static_cast<std::uint32_t>(PeCount)) {
        DataTile dst;
        OffsetTile lin, lim, quotient, cycle, cycle_base, coordinate, contribution, off_tile;

        TCI(lin, t * static_cast<std::uint32_t>(TileElements));  // 输出平坦下标 o
        TEXPANDS(lim, numel - 1u);
        TMIN(lin, lin, lim);  // clamp 到 [0, numel-1]，避免尾块越界读
        TEXPANDS(off_tile, static_cast<std::uint32_t>(0));

        std::uint32_t in_stride = 1;
        for (int input_dim = rank - 1; input_dim >= 0; --input_dim) {  // 运行时 rank 循环
            const int output_dim = static_cast<int>(inv_perm[input_dim]);
            std::uint32_t out_stride = 1;
            for (int dim = output_dim + 1; dim < rank; ++dim) {
                out_stride *= out_shape[dim];
            }
            if (out_stride == 1) {
                TCVT(quotient, lin);
            } else {
                TDIVS(quotient, lin, out_stride);
            }
            TDIVS(cycle, quotient, out_shape[output_dim]);
            TMULS(cycle_base, cycle, out_shape[output_dim]);
            TSUB(coordinate, quotient, cycle_base);
            TMULS(contribution, coordinate,
                  in_stride * static_cast<std::uint32_t>(sizeof(DType)));
            TADD(off_tile, off_tile, contribution);
            in_stride *= in_shape[input_dim];
        }

        auto out_global = output_iter(0, static_cast<int>(t));
#ifdef DYN_DUMP_OFF
        {
            using UGlobal = pto::global_tensor<std::uint32_t, pto::RowMajor<1, CapElements>>;
            using UIter = global_iterator<UGlobal, OffsetTile>;
            UIter uit(reinterpret_cast<std::uint32_t *>(output));
            auto go = uit(0, static_cast<int>(t));
            TSTORE(go, off_tile);
        }
#else
        MGATHER(dst, input_global, off_tile);
        TSTORE(out_global, dst);
#endif
    }
}

/**
 * @brief 动态 shape 通用 permute 转置（多 PE 包装）。
 * 输出 tile 按 round-robin 分给各 PE。
 * @param perm 运行时置换（输出轴 i = 输入轴 perm[i]）。
 */
template <typename DType, int CapElements, int TileElements = 512,
          std::uint32_t PeCount = supernpu::multi_thread::kDefaultPeCount>
void transpose_nd_dynamic(DType *input, DType *output, int rank,
                          const std::uint32_t *in_shape, const std::uint32_t *out_shape,
                          const std::uint32_t *perm, std::uint32_t numel) {
    static_assert(PeCount > 0);
    std::uint32_t inv_perm[8] = {0, 1, 2, 3, 4, 5, 6, 7};
    for (int i = 0; i < rank; ++i) {
        inv_perm[perm[i]] = static_cast<std::uint32_t>(i);  // inv_perm[input_dim]=output_dim
    }
    const int tid = static_cast<int>(get_thread_idx());
    transpose_nd_dynamic_tile<DType, CapElements, TileElements, PeCount>(
        input, output, rank, in_shape, out_shape, inv_perm, numel, tid);
}

}  // namespace supernpu::tile_isa
