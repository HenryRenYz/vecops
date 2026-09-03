//
// Copyright (c) vecops contributors.
//

/**
 * Focused benchmark for compile-time C^T = B^T*A^T orientation selection.
 *
 * Every case is registered as an adjacent swap_on/swap_off pair.  Constant
 * extents intentionally exercise the same compile-time packing/orientation
 * decisions used by attention kernels without adding variants to MatmulBench.
 */

#include "vecops/Features.h"

#include <benchmark/benchmark.h>

#include <cerrno>
#include <cmath>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#if defined(ARCH_X86_FAMILY)
#include <sys/syscall.h>
#include <unistd.h>
#endif

#include "MatmulBenchCommon.h"
#include "vecops/matmul/Atom.h"

namespace vecops::bench::matmul {

#if defined(ARCH_X86_FAMILY)
using PrimaryAtom = ::vecops::matmul::AMX_BF16F32;
#else
using PrimaryAtom = ::vecops::matmul::SME_BF16F32;
#endif

template <typename Atom, bool EnableSwapAB>
using OrientationConfig = ops::MatmulConfig<
    Atom, ::vecops::matmul::family_selection::Automatic,
    kernel::matmul_policy::Automatic,
    ::vecops::matmul::GenericTiledTuning<>,
    platform::SystemCacheInfoProvider, EnableSwapAB>;

template <bool Transposed, typename T, nint_t Rows, nint_t Columns>
auto matrix(T* data) {
  if constexpr (Transposed) {
    return make_tensor(
        data, make_layout(
                  make_shape(cint<Rows>, cint<Columns>),
                  make_strides(cint<1>, cint<Rows>)));
  } else {
    return make_tensor(
        data, make_layout(
                  make_shape(cint<Rows>, cint<Columns>),
                  make_strides(cint<Columns>, cint<1>)));
  }
}

template <typename Tensor>
void fill_matrix(Tensor& tensor, nint_t rows, nint_t columns, int seed) {
  using T = typename Tensor::ElementType;
  for (nint_t row = 0; row < rows; ++row) {
    for (nint_t column = 0; column < columns; ++column) {
      const int value = static_cast<int>(
          (row * 13 + column * 7 + seed) % 29) - 14;
      tensor(row, column) = static_cast<T>(
          static_cast<float>(value) / 31.0f);
    }
  }
}

template <typename Atom, typename MemoryA, typename MemoryB,
          bool AT, bool BT, bool CT, bool EnableSwapAB,
          nint_t M, nint_t N, nint_t K>
void run_orientation(benchmark::State& state) {
  using TA = typename Atom::TA;
  using TB = typename Atom::TB;
  using Acc = typename Atom::TAcc;
  std::vector<MemoryA> a(static_cast<std::size_t>(M * K));
  std::vector<MemoryB> b(static_cast<std::size_t>(K * N));
  std::vector<Acc> c(static_cast<std::size_t>(M * N));
  auto a_tensor = matrix<AT, MemoryA, M, K>(a.data());
  auto conventional_b = matrix<BT, MemoryB, K, N>(b.data());
  auto b_tensor = transpose_view<0, 1>(conventional_b);
  auto c_tensor = matrix<CT, Acc, M, N>(c.data());
  fill_matrix(a_tensor, M, K, 3);
  fill_matrix(conventional_b, K, N, 11);

  auto operation = ops::matmul(OrientationConfig<Atom, EnableSwapAB>{});
  kernel::Workspace storage(operation.required_workspace(
      cint<M>, cint<N>, cint<K>, a_tensor, b_tensor, c_tensor));
  auto workspace = storage.view();
  ExecutionSession execution{workspace};
  operation(
      execution, cint<M>, cint<N>, cint<K>,
      a_tensor, b_tensor, c_tensor);

  constexpr nint_t SampleRow = M / 2;
  constexpr nint_t SampleColumn = N / 2;
  Acc expected{};
  for (nint_t kk = 0; kk < K; ++kk) {
    expected += static_cast<Acc>(static_cast<TA>(
                    a_tensor(SampleRow, kk))) *
        static_cast<Acc>(static_cast<TB>(
            conventional_b(kk, SampleColumn)));
  }
  const double error = std::abs(
      static_cast<double>(c_tensor(SampleRow, SampleColumn)) -
      static_cast<double>(expected));
  if (error > 5.0e-3 * std::max(1.0, std::abs(static_cast<double>(expected)))) {
    state.SkipWithError("orientation result verification failed");
    return;
  }

  for (auto _ : state) {
    benchmark::DoNotOptimize(a.data());
    operation(
        execution, cint<M>, cint<N>, cint<K>,
        a_tensor, b_tensor, c_tensor);
    benchmark::DoNotOptimize(c.data());
    benchmark::ClobberMemory();
  }
  state.counters["OP/s"] = benchmark::Counter(
      2.0 * static_cast<double>(M) * static_cast<double>(N) *
          static_cast<double>(K),
      benchmark::Counter::kIsIterationInvariantRate);
  state.counters["workspace"] = benchmark::Counter(
      static_cast<double>(operation.required_workspace(
          cint<M>, cint<N>, cint<K>, a_tensor, b_tensor, c_tensor)));
}

template <bool AT, bool BT, bool CT, nint_t M, nint_t N, nint_t K>
void register_pair(const char* label) {
  const std::string stem = std::string("orientation/") + label +
      "/m" + std::to_string(M) + "/n" + std::to_string(N) +
      "/k" + std::to_string(K);
  benchmark::RegisterBenchmark(
      (stem + "/swap_on").c_str(),
      &run_orientation<
          PrimaryAtom, typename PrimaryAtom::TA, typename PrimaryAtom::TB,
          AT, BT, CT, true, M, N, K>);
  benchmark::RegisterBenchmark(
      (stem + "/swap_off").c_str(),
      &run_orientation<
          PrimaryAtom, typename PrimaryAtom::TA, typename PrimaryAtom::TB,
          AT, BT, CT, false, M, N, K>);
}

template <nint_t M, nint_t N, nint_t K>
void register_conversion_pair(const char* label) {
  const std::string stem = std::string("orientation/") + label +
      "/m" + std::to_string(M) + "/n" + std::to_string(N) +
      "/k" + std::to_string(K);
  benchmark::RegisterBenchmark(
      (stem + "/swap_on").c_str(),
      &run_orientation<
          PrimaryAtom, float32_t, float32_t,
          true, true, true, true, M, N, K>);
  benchmark::RegisterBenchmark(
      (stem + "/swap_off").c_str(),
      &run_orientation<
          PrimaryAtom, float32_t, float32_t,
          true, true, true, false, M, N, K>);
}

void register_orientation_benchmarks() {
  register_pair<false, false, false, 64, 128, 256>("case1_rrr");
  register_pair<true,  false, false, 64, 128, 256>("case2_trr");
  register_pair<false, true,  false, 64, 128, 256>("case3_rtr");
  register_pair<false, false, true,  64, 128, 256>("case4_rrt");
  register_pair<true,  true,  false, 64, 128, 256>("case5_ttr");
  register_pair<true,  false, true,  64, 128, 256>("case6_trt");
  register_pair<false, true,  true,  64, 128, 256>("case7_rtt");
  register_pair<true,  true,  true,  64, 128, 256>("case8_ttt");

  // Attention-oriented probes: exact 2x2 tiles, K=16, and N=4/8.
  register_pair<true, true, true, 32, 32, 64>("case8_2x2_tiles");
  register_pair<true, true, true, 32, 32, 16>("case8_k16");
  register_pair<false, false, true, 128, 4, 64>("case4_n4");
  register_pair<false, false, true, 128, 8, 64>("case4_n8");
  register_conversion_pair<64, 128, 256>("case8_fp32_to_bf16");

  // AMX's case-5 gate deliberately spans the current long-K boundary.
  register_pair<true, true, false, 64, 128, 16>("case5_gate");
  register_pair<true, true, false, 64, 128, 32>("case5_gate");
  register_pair<true, true, false, 64, 128, 64>("case5_gate");
  register_pair<true, true, false, 64, 128, 128>("case5_gate");
}

bool enable_orientation_benchmark() {
#if defined(ARCH_X86_FAMILY)
  constexpr long ArchRequestXcompPerm = 0x1023;
  constexpr long XfeatureTileData = 18;
  return syscall(
      SYS_arch_prctl, ArchRequestXcompPerm, XfeatureTileData) == 0;
#else
  return true;
#endif
}

} // namespace vecops::bench::matmul

int main(int argc, char** argv) {
  using namespace vecops::bench::matmul;
  if (!enable_orientation_benchmark()) {
    std::cerr << "Unable to initialize matmul orientation benchmark: "
              << std::strerror(errno) << '\n';
    return 1;
  }
  register_orientation_benchmarks();
  return run_benchmarks(argc, argv, "matmul_orientation");
}
