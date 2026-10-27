#include <cstdint>

#include "benchmark.h"
#include "fileop.h"
#include "multi_thread_res_check.h"
#include "solution/transpose/transpose_nd_dynamic.hpp"

// 参数化动态 shape 测试（rank/shape/perm 均可 -D 覆盖），用于定位编译器 bug 的触发边界。
#ifndef DYN_RANK
#define DYN_RANK 4
#endif
#ifndef D0
#define D0 8
#endif
#ifndef D1
#define D1 8
#endif
#ifndef D2
#define D2 8
#endif
#ifndef D3
#define D3 8
#endif
#ifndef P0
#define P0 0
#endif
#ifndef P1
#define P1 2
#endif
#ifndef P2
#define P2 1
#endif
#ifndef P3
#define P3 3
#endif

constexpr int kRank = DYN_RANK;
constexpr std::uint32_t kDimsAll[4] = {D0, D1, D2, D3};
constexpr std::uint32_t kPermAll[4] = {P0, P1, P2, P3};

namespace {
constexpr std::uint32_t kInShape[kRank] = {kDimsAll[0],
#if DYN_RANK > 1
                                           kDimsAll[1],
#endif
#if DYN_RANK > 2
                                           kDimsAll[2],
#endif
#if DYN_RANK > 3
                                           kDimsAll[3],
#endif
};
constexpr std::uint32_t kPermU[kRank] = {kPermAll[0],
#if DYN_RANK > 1
                                         kPermAll[1],
#endif
#if DYN_RANK > 2
                                         kPermAll[2],
#endif
#if DYN_RANK > 3
                                         kPermAll[3],
#endif
};
constexpr std::uint32_t kOutShape[kRank] = {kInShape[kPermU[0]],
#if DYN_RANK > 1
                                            kInShape[kPermU[1]],
#endif
#if DYN_RANK > 2
                                            kInShape[kPermU[2]],
#endif
#if DYN_RANK > 3
                                            kInShape[kPermU[3]],
#endif
};
int NumelInit() {
    int n = 1;
    for (int i = 0; i < kRank; ++i) n *= static_cast<int>(kInShape[i]);
    return n;
}
constexpr int kNumel = 1 * ([] {
    int n = 1;
    for (int i = 0; i < kRank; ++i) n *= static_cast<int>(kInShape[i]);
    return n;
}());
constexpr int kTE = 512;
constexpr int kCap = ((kNumel + kTE - 1) / kTE) * kTE;

alignas(4096) std::int32_t input[kCap];
alignas(4096) std::int32_t output[kCap];
#ifdef RES_CHECK
MultiThreadResCheckSync res_check_sync{};
#endif
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
    // 运行时拷贝（模拟"动态传入"）：如果这步被误编译，kernel 会读到错的数组
    std::uint32_t rt_in[kRank], rt_out[kRank], rt_perm[kRank];
    for (int i = 0; i < kRank; ++i) {
        rt_in[i] = kInShape[i];
        rt_out[i] = kOutShape[i];
        rt_perm[i] = kPermU[i];
    }
#ifdef RES_CHECK
    if (tid == 0) {
        std::uint32_t dbg[28];
        dbg[0] = kRank;
        for (int i = 0; i < kRank; ++i) dbg[1 + i] = kInShape[i];        // constexpr
        for (int i = 0; i < kRank; ++i) dbg[5 + i] = rt_in[i];           // loop copy
        for (int i = 0; i < kRank; ++i) dbg[9 + i] = kOutShape[i];       // constexpr
        for (int i = 0; i < kRank; ++i) dbg[13 + i] = rt_out[i];         // loop copy
        for (int i = 0; i < kRank; ++i) dbg[17 + i] = kPermU[i];         // constexpr
        for (int i = 0; i < kRank; ++i) dbg[21 + i] = rt_perm[i];        // loop copy
        writeBinaryFile(CHK_DIR "/rt_arrays.bin", reinterpret_cast<uint8_t *>(dbg), sizeof(dbg));
    }
#endif
    BENCHSTART;
    supernpu::tile_isa::transpose_nd_dynamic<std::int32_t, kCap, kTE, 4>(
        input, output, kRank, rt_in, rt_out, rt_perm,
        static_cast<std::uint32_t>(kNumel));
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
