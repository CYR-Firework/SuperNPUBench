#include <cstdint>

#include "benchmark.h"
#include "fileop.h"
#include "multi_thread_res_check.h"

#include <common/pto_tileop.hpp>
#include "basic_op/utils/spmd_partition.hpp"

// 纯拷贝 kernel（TLOAD 连续块 + TSTORE），用于隔离测量 TLOAD 搬运效率，
// 与 transpose(MGATHER) 在相同元素数/tile 数下对照。
#ifndef COPY_NUMEL
#define COPY_NUMEL 262144  // 512*512
#endif
#ifndef COPY_TILE_ELEMENTS
#define COPY_TILE_ELEMENTS 512
#endif

namespace {

constexpr int kNumel = COPY_NUMEL;
constexpr int kTE = COPY_TILE_ELEMENTS;
alignas(4096) std::int32_t input[kNumel];
alignas(4096) std::int32_t output[kNumel];
#ifdef RES_CHECK
MultiThreadResCheckSync res_check_sync{};
#endif

template <typename DType, int N, int TE, int PeCount>
void copy_flat(DType *in, DType *out) {
    using InputGlobal = pto::global_tensor<DType, pto::RowMajor<1, N>>;
    using OutputGlobal = pto::global_tensor<DType, pto::RowMajor<1, N>>;
    using DataTile = pto::Tile<pto::Location::Vec, DType, 1, TE, pto::BLayout::RowMajor>;
    using InputIterator = global_iterator<InputGlobal, DataTile>;
    using OutputIterator = global_iterator<OutputGlobal, DataTile>;

    InputIterator in_iter(in);
    OutputIterator out_iter(out);
    constexpr int kTiles = N / TE;
    const int tid = static_cast<int>(get_thread_idx());
    for (int t = tid; t < kTiles; t += PeCount) {
        auto in_global = in_iter(0, t);
        auto out_global = out_iter(0, t);
        DataTile dst;
        TLOAD(dst, in_global);
        TSTORE(out_global, dst);
    }
}

}  // namespace

int main() {
    const std::uint32_t tid = get_thread_idx();
#ifdef RES_CHECK
    if (tid == 0) {
        readBinaryFile(CHK_DIR "/input.bin", reinterpret_cast<uint8_t *>(input),
                       sizeof(input));
    }
    res_check_publish_inputs(res_check_sync, tid);
#endif
    BENCHSTART;
    copy_flat<std::int32_t, kNumel, kTE, 4>(input, output);
    BENCHEND;
#ifdef RES_CHECK
    res_check_wait_for_all(res_check_sync, tid);
    if (tid == 0) {
        writeBinaryFile(CHK_DIR "/output.bin",
                        reinterpret_cast<uint8_t *>(output), sizeof(output));
    }
#endif
    return 0;
}
