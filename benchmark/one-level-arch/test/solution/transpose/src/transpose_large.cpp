#include <cstdint>

#include "benchmark.h"
#include "fileop.h"
#include "multi_thread_res_check.h"
#ifdef TRANSPOSE_USE_V21
#include "solution/transpose/transpose_2d_v2_1.hpp"
#else
#include "solution/transpose/transpose_2d_v2.hpp"
#endif

// 大 case 2D 转置，用于检验"放大 tile 是否更优"。三个维度均可用 -D 覆盖。
#ifndef TRANSPOSE_ROWS
#define TRANSPOSE_ROWS 512
#endif
#ifndef TRANSPOSE_COLS
#define TRANSPOSE_COLS 512
#endif
#ifndef TRANSPOSE_TILE_ELEMENTS
#define TRANSPOSE_TILE_ELEMENTS 512
#endif

namespace {
alignas(4096) std::int32_t input[TRANSPOSE_ROWS * TRANSPOSE_COLS];
alignas(4096) std::int32_t output[TRANSPOSE_ROWS * TRANSPOSE_COLS];
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
        std::int32_t, TRANSPOSE_ROWS, TRANSPOSE_COLS,
        TRANSPOSE_TILE_ELEMENTS>(input, output);
#else
    supernpu::multi_thread::transpose_2d_v2<
        std::int32_t, TRANSPOSE_ROWS, TRANSPOSE_COLS,
        TRANSPOSE_TILE_ELEMENTS>(input, output);
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
