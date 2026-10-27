#include <cstdint>

#include "benchmark.h"
#include "fileop.h"
#include "multi_thread_res_check.h"
#ifdef TRANSPOSE_USE_V21
#include "solution/transpose/transpose_2d_v2_1.hpp"
#else
#include "solution/transpose/transpose_2d_v2.hpp"
#endif

namespace {
// 有尾块用例：[64 x 63] int32，共 64*63 = 4032 = 7*512 + 448 → 尾块 448 元素，
// 用于覆盖 transpose_2d_v2 的 handle_tail 分支（[64x64] 无尾块）。
constexpr int kRows = 64;
constexpr int kCols = 63;
alignas(4096) std::int32_t input[kRows * kCols];
alignas(4096) std::int32_t output[kRows * kCols];
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
    BENCHSTART;
#ifdef TRANSPOSE_USE_V21
    supernpu::multi_thread::transpose_2d_v2_1<
        std::int32_t, kRows, kCols, 512>(input, output);
#else
    supernpu::multi_thread::transpose_2d_v2<
        std::int32_t, kRows, kCols, 512>(input, output);
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
