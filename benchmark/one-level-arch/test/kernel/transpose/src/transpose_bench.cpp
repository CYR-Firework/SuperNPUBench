#include <cstdint>

#include "benchmark.h"
#include "fileop.h"
#include "multi_thread_res_check.h"
#include "basic_op/transpose/transpose_nd_v3.hpp"

// 真实 benchmark case（来自 tmp/transpose_case.txt），用 v3（通用 permute）测试。
//   0  b32  int8     (1024,256,17)     perm [1,0,2]
//   1  b36  int8     (512,64,128,17)   perm [2,1,0,3]     （buffer 太大，模拟器跑不了）
//   2  b08  float16  (94440,32)        perm [1,0]
//   3  b035 float32  (64,8,8,16,256)   perm [0,1,4,3,2]   （buffer 太大，模拟器跑不了）
//   4  b36' 缩小变体 int8 (64,64,32,17)    perm [2,1,0,3]   （验证同 perm/rank）
//   5  b035'缩小变体 fp32 (8,8,8,16,32)    perm [0,1,4,3,2] （验证同 perm/rank）
#ifndef BENCH_CASE
#define BENCH_CASE 0
#endif

#if BENCH_CASE == 0
using T = std::int8_t;
constexpr int kRank = 3;
constexpr int kElements = 1024 * 256 * 17;
constexpr std::uint32_t kInShape[3] = {1024, 256, 17};
constexpr int kPerm[3] = {1, 0, 2};
#elif BENCH_CASE == 1
using T = std::int8_t;
constexpr int kRank = 4;
constexpr int kElements = 512 * 64 * 128 * 17;
constexpr std::uint32_t kInShape[4] = {512, 64, 128, 17};
constexpr int kPerm[4] = {2, 1, 0, 3};
#elif BENCH_CASE == 2
using T = std::int16_t;  // float16 等宽视图（转置与 dtype 无关）
constexpr int kRank = 2;
constexpr int kElements = 94440 * 32;
constexpr std::uint32_t kInShape[2] = {94440, 32};
constexpr int kPerm[2] = {1, 0};
#elif BENCH_CASE == 3
using T = float;
constexpr int kRank = 5;
constexpr int kElements = 64 * 8 * 8 * 16 * 256;
constexpr std::uint32_t kInShape[5] = {64, 8, 8, 16, 256};
constexpr int kPerm[5] = {0, 1, 4, 3, 2};
#elif BENCH_CASE == 4
using T = std::int8_t;
constexpr int kRank = 4;
constexpr int kElements = 64 * 64 * 32 * 17;
constexpr std::uint32_t kInShape[4] = {64, 64, 32, 17};
constexpr int kPerm[4] = {2, 1, 0, 3};
#elif BENCH_CASE == 5
using T = float;
constexpr int kRank = 5;
constexpr int kElements = 8 * 8 * 8 * 16 * 32;
constexpr std::uint32_t kInShape[5] = {8, 8, 8, 16, 32};
constexpr int kPerm[5] = {0, 1, 4, 3, 2};
#elif BENCH_CASE == 6
using T = std::int32_t;
constexpr int kRank = 4;
constexpr int kElements = 8 * 8 * 4 * 5;
constexpr std::uint32_t kInShape[4] = {8, 8, 4, 5};
constexpr int kPerm[4] = {2, 1, 0, 3};
#elif BENCH_CASE == 7
using T = std::int32_t;
constexpr int kRank = 4;
constexpr int kElements = 8 * 8 * 4 * 5;
constexpr std::uint32_t kInShape[4] = {8, 8, 4, 5};
constexpr int kPerm[4] = {0, 2, 1, 3};
#elif BENCH_CASE == 8
using T = std::int32_t;
constexpr int kRank = 4;
constexpr int kElements = 8 * 8 * 4 * 8;
constexpr std::uint32_t kInShape[4] = {8, 8, 4, 8};
constexpr int kPerm[4] = {0, 2, 1, 3};
#else
using T = std::int32_t;
constexpr int kRank = 4;
constexpr int kElements = 8 * 8 * 8 * 8;
constexpr std::uint32_t kInShape[4] = {8, 8, 8, 8};
constexpr int kPerm[4] = {0, 2, 1, 3};
#endif

namespace {
alignas(4096) T input[kElements];
alignas(4096) T output[kElements];
#ifdef RES_CHECK
MultiThreadResCheckSync res_check_sync{};
#endif
}  // namespace

int main() {
    const std::uint32_t tid = get_thread_idx();
    std::uint32_t in_shape[kRank];
    std::uint32_t out_shape[kRank];
    for (int i = 0; i < kRank; ++i) {
        in_shape[i] = kInShape[i];
        out_shape[i] = kInShape[kPerm[i]];
    }
#ifdef RES_CHECK
    if (tid == 0) {
        readBinaryFile(CHK_DIR "/input.bin", reinterpret_cast<uint8_t *>(input),
                       sizeof(input));
    }
    res_check_publish_inputs(res_check_sync, tid);
#endif
    BENCHSTART;
#if BENCH_CASE == 0
    supernpu::multi_thread::transpose_nd_v3<T, 3, kElements, 512, 4, 1, 0, 2>(
        input, output, in_shape, out_shape);
#elif BENCH_CASE == 1
    supernpu::multi_thread::transpose_nd_v3<T, 4, kElements, 512, 4, 2, 1, 0, 3>(
        input, output, in_shape, out_shape);
#elif BENCH_CASE == 2
    supernpu::multi_thread::transpose_nd_v3<T, 2, kElements, 512, 4, 1, 0>(
        input, output, in_shape, out_shape);
#elif BENCH_CASE == 3
    supernpu::multi_thread::transpose_nd_v3<T, 5, kElements, 512, 4, 0, 1, 4, 3, 2>(
        input, output, in_shape, out_shape);
#elif BENCH_CASE == 4
    supernpu::multi_thread::transpose_nd_v3<T, 4, kElements, 512, 4, 2, 1, 0, 3>(
        input, output, in_shape, out_shape);
#elif BENCH_CASE == 5
    supernpu::multi_thread::transpose_nd_v3<T, 5, kElements, 512, 4, 0, 1, 4, 3, 2>(
        input, output, in_shape, out_shape);
#elif BENCH_CASE == 6
    supernpu::multi_thread::transpose_nd_v3<T, 4, kElements, 512, 4, 2, 1, 0, 3>(
        input, output, in_shape, out_shape);
#elif BENCH_CASE == 7
    supernpu::multi_thread::transpose_nd_v3<T, 4, kElements, 512, 4, 0, 2, 1, 3>(
        input, output, in_shape, out_shape);
#elif BENCH_CASE == 8
    supernpu::multi_thread::transpose_nd_v3<T, 4, kElements, 512, 4, 0, 2, 1, 3>(
        input, output, in_shape, out_shape);
#else
    supernpu::multi_thread::transpose_nd_v3<T, 4, kElements, 512, 4, 0, 2, 1, 3>(
        input, output, in_shape, out_shape);
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
