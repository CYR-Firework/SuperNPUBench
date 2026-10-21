#include <cstdint>

#include "benchmark.h"
#include "fileop.h"

// 最小复现（可运行版 / SuperNPUBench 测试框架）
//   期望：uniform_out = {8,8,8,8}；nonuniform_out = {1,2,3,4}
//   实测（Linx）：uniform_out = {8,16,8,16}；nonuniform_out = {1,2,3,4}（正确）
#define CDIR \
    "/mnt/workspace/gitCode/ChenYiran/SuperScalar/SuperNPUBench/benchmark/one-level-arch/compare/kernel_transpose_transpose_minrepro_PE4/"

int main() {
    constexpr std::uint32_t U[4] = {8, 8, 8, 8};     // uniform, nonzero  → 触发 bug
    constexpr std::uint32_t N[4] = {1, 2, 3, 4};     // non-uniform       → 对照组（正确）

    writeBinaryFile(CDIR "uniform_out.bin", reinterpret_cast<const uint8_t *>(U), 16);
    writeBinaryFile(CDIR "nonuniform_out.bin", reinterpret_cast<const uint8_t *>(N), 16);
    return 0;
}
