#include <cstdint>

#include "benchmark.h"
#include "fileop.h"
#include "multi_thread_res_check.h"
#include "basic_op/transpose/transpose_nd_v3.hpp"

// 通用 permute 转置测试（v3）：
//   PERM_CASE=0: 4D [1,16,936,128], perm=(0,2,1,3) → [1,936,16,128]（BNSD→BSND）
//   PERM_CASE=1: 3D [16,32,64],    perm=(2,0,1)   → [64,16,32]
#ifndef PERM_CASE
#define PERM_CASE 0
#endif

#if PERM_CASE == 1
constexpr int kRank = 3;
constexpr std::uint32_t kInShape[3] = {16, 32, 64};
#else
constexpr int kRank = 4;
constexpr std::uint32_t kInShape[4] = {1, 16, 936, 128};
#endif

namespace {
constexpr int kElements = [] {
    int n = 1;
    for (int i = 0; i < kRank; ++i) n *= static_cast<int>(kInShape[i]);
    return n;
}();
std::uint32_t g_out_shape[kRank];
std::uint32_t g_in_shape[kRank];
alignas(4096) std::int32_t input[kElements];
alignas(4096) std::int32_t output[kElements];
#ifdef RES_CHECK
MultiThreadResCheckSync res_check_sync{};
#endif
}  // namespace

int main() {
    const std::uint32_t tid = get_thread_idx();
    for (int i = 0; i < kRank; ++i) g_in_shape[i] = kInShape[i];
#if PERM_CASE == 1
    constexpr int kPerm[kRank] = {2, 0, 1};
#else
    constexpr int kPerm[kRank] = {0, 2, 1, 3};
#endif
    for (int i = 0; i < kRank; ++i) g_out_shape[i] = kInShape[kPerm[i]];

#ifdef RES_CHECK
    if (tid == 0) {
        readBinaryFile(CHK_DIR "/input.bin", reinterpret_cast<uint8_t *>(input),
                       sizeof(input));
    }
    res_check_publish_inputs(res_check_sync, tid);
#endif
    BENCHSTART;
#if PERM_CASE == 1
    supernpu::multi_thread::transpose_nd_v3<
        std::int32_t, 3, kElements, 512, 4, 2, 0, 1>(input, output, g_in_shape, g_out_shape);
#else
    supernpu::multi_thread::transpose_nd_v3<
        std::int32_t, 4, kElements, 512, 4, 0, 2, 1, 3>(input, output, g_in_shape, g_out_shape);
#endif
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
