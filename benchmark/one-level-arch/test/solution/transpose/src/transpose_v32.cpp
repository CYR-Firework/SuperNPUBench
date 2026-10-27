#include <cstdint>

#include "benchmark.h"
#include "fileop.h"
#include "multi_thread_res_check.h"
#include "solution/transpose/transpose_nd_v3_2.hpp"

// v3.2 测试（动态 shape + 场景分派）：shape/perm 均以**运行时数组**传入，
// 覆盖三类场景与常见 rank/perm。
//   CASE=0 1D [4096]        perm(0)              -> TENSOR_MOVE
//   CASE=1 2D [64,64]       perm(1,0)            -> GATHER
//   CASE=2 3D [64,64,256]   perm(1,0,2)          -> N_LAST
//   CASE=3 3D [32,64,512]   perm(0,2,1)          -> GATHER
//   CASE=4 3D [16,32,64]    perm(2,0,1)          -> GATHER
//   CASE=5 4D [8,8,8,8]     perm(0,2,1,3)        -> GATHER
//   CASE=6 4D [1,16,936,128] perm(0,2,1,3)       -> GATHER
//   CASE=7 5D [8,8,8,16,32] perm(0,1,4,3,2)      -> GATHER
#ifndef V32_CASE
#define V32_CASE 0
#endif

#if V32_CASE == 1
constexpr int kRank = 2;
constexpr std::uint32_t kShape[2] = {64, 64};
constexpr std::uint32_t kPerm[2] = {1, 0};
constexpr int kCap = 64 * 64;
#elif V32_CASE == 2
constexpr int kRank = 3;
constexpr std::uint32_t kShape[3] = {64, 64, 256};
constexpr std::uint32_t kPerm[3] = {1, 0, 2};
constexpr int kCap = 64 * 64 * 256;
#elif V32_CASE == 3
constexpr int kRank = 3;
constexpr std::uint32_t kShape[3] = {32, 64, 512};
constexpr std::uint32_t kPerm[3] = {0, 2, 1};
constexpr int kCap = 32 * 64 * 512;
#elif V32_CASE == 4
constexpr int kRank = 3;
constexpr std::uint32_t kShape[3] = {16, 32, 64};
constexpr std::uint32_t kPerm[3] = {2, 0, 1};
constexpr int kCap = 16 * 32 * 64;
#elif V32_CASE == 5
constexpr int kRank = 4;
constexpr std::uint32_t kShape[4] = {8, 8, 8, 8};
constexpr std::uint32_t kPerm[4] = {0, 2, 1, 3};
constexpr int kCap = 8 * 8 * 8 * 8;
#elif V32_CASE == 6
constexpr int kRank = 4;
constexpr std::uint32_t kShape[4] = {1, 16, 936, 128};
constexpr std::uint32_t kPerm[4] = {0, 2, 1, 3};
constexpr int kCap = 1 * 16 * 936 * 128;
#elif V32_CASE == 7
constexpr int kRank = 5;
constexpr std::uint32_t kShape[5] = {8, 8, 8, 16, 32};
constexpr std::uint32_t kPerm[5] = {0, 1, 4, 3, 2};
constexpr int kCap = 8 * 8 * 8 * 16 * 32;
#else
constexpr int kRank = 1;
constexpr std::uint32_t kShape[1] = {4096};
constexpr std::uint32_t kPerm[1] = {0};
constexpr int kCap = 4096;
#endif

namespace {
alignas(4096) std::int32_t input[kCap];
alignas(4096) std::int32_t output[kCap];
#ifdef RES_CHECK
MultiThreadResCheckSync res_check_sync{};
#endif
}  // namespace

int main() {
    const std::uint32_t tid = get_thread_idx();
    // 运行时 shape/perm（模拟动态传入）
    std::uint32_t rt_in[kRank];
    std::uint32_t rt_out[kRank];
    std::uint32_t rt_perm[kRank];
    for (int i = 0; i < kRank; ++i) {
        rt_in[i] = kShape[i];
        rt_perm[i] = kPerm[i];
        rt_out[i] = kShape[kPerm[i]];
    }
    // 自初始化：所有 PE 写入相同数据（确定性），供 host 侧 golden 比对
    for (int i = 0; i < kCap; ++i) input[i] = i;
    for (int i = 0; i < kCap; ++i) output[i] = -1;
#ifdef RES_CHECK
    res_check_publish_inputs(res_check_sync, tid);
#endif
    BENCHSTART;
    supernpu::multi_thread::transpose_nd_v3_2<std::int32_t, kCap, 512, 4>(
        input, output, kRank, rt_in, rt_out, rt_perm, static_cast<std::uint32_t>(kCap));
    BENCHEND;
#ifdef RES_CHECK
    res_check_wait_for_all(res_check_sync, tid);
#endif
    if (tid == 0) {
        writeBinaryFile(
            "/mnt/workspace/gitCode/ChenYiran/SuperScalar/SuperNPUBench/benchmark/one-level-arch/compare/solution_transpose_transpose_v32_PE4/output.bin",
            reinterpret_cast<std::uint8_t *>(output),
            sizeof(std::int32_t) * static_cast<std::size_t>(kCap));
    }
    return 0;
}
