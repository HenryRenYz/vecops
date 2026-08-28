//
// Copyright (c) vecops contributors.
//

#ifndef VECOPS_BENCHMARKS_OPS_MATMUL_BENCH_COMMON_H
#define VECOPS_BENCHMARKS_OPS_MATMUL_BENCH_COMMON_H

#include <benchmark/benchmark.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <type_traits>
#include <vector>

#include "BenchmarkUtils.h"
#include "vecops/gemm/Packing.h"
#include "vecops/kernel/Workspace.h"
#include "vecops/ops/Matmul.h"
#include "vecops/ops/MatmulPack.h"

#ifndef VECOPS_BENCH_ARCH_CODE
#define VECOPS_BENCH_ARCH_CODE "unknown"
#endif

#ifndef VECOPS_SOURCE_DIR
#define VECOPS_SOURCE_DIR "."
#endif

namespace vecops::bench::matmul {

using namespace ::vecops::meta;
using namespace ::vecops::tensor;

enum class InputMode : uint32_t {
  Raw = 1u << 0,
  PackedA = 1u << 1,
  PackedB = 1u << 2,
  PackedAB = 1u << 3,
};

inline constexpr uint32_t mode_bit(InputMode mode) {
  return static_cast<uint32_t>(mode);
}

inline constexpr uint32_t kRawAndPackedAB =
    mode_bit(InputMode::Raw) | mode_bit(InputMode::PackedAB);
inline constexpr uint32_t kAllInputModes =
    mode_bit(InputMode::Raw) | mode_bit(InputMode::PackedA) |
    mode_bit(InputMode::PackedB) | mode_bit(InputMode::PackedAB);
inline constexpr uint32_t kWeightReuseModes =
    mode_bit(InputMode::Raw) | mode_bit(InputMode::PackedB) |
    mode_bit(InputMode::PackedAB);

struct MatmulCase {
  const char* group;
  const char* name;
  nint_t m;
  nint_t n;
  nint_t k;
  uint32_t modes;
};

template <typename Atom>
struct AtomName;

template <typename Atom>
const char* atom_name() {
  return AtomName<Atom>::value;
}

template <typename T>
const char* dtype_name() {
  if constexpr (std::same_as<T, bfloat16_t>) return "bf16";
  if constexpr (std::same_as<T, float16_t>) return "fp16";
  if constexpr (std::same_as<T, float32_t>) return "fp32";
  if constexpr (std::same_as<T, float64_t>) return "fp64";
  if constexpr (std::same_as<T, int8_t>) return "int8";
  if constexpr (std::same_as<T, uint8_t>) return "uint8";
  if constexpr (std::same_as<T, int32_t>) return "int32";
  return "unknown";
}

template <InputMode Mode>
const char* input_mode_name() {
  if constexpr (Mode == InputMode::Raw) return "raw";
  if constexpr (Mode == InputMode::PackedA) return "packed_a";
  if constexpr (Mode == InputMode::PackedB) return "packed_b";
  return "packed_ab";
}

template <typename T>
void fill_input(std::vector<T>& input, int seed) {
  for (std::size_t i = 0; i < input.size(); ++i) {
    if constexpr (std::same_as<T, uint8_t>) {
      input[i] = static_cast<T>((i * 5 + static_cast<std::size_t>(seed)) % 7);
    } else if constexpr (std::same_as<T, int8_t>) {
      input[i] = static_cast<T>(
          static_cast<int>((i * 7 + static_cast<std::size_t>(seed)) % 11) - 5);
    } else {
      const float value = static_cast<float>(
          static_cast<int>((i * 13 + static_cast<std::size_t>(seed)) % 29) -
          14) / 31.0f;
      input[i] = static_cast<T>(value);
    }
  }
}

template <typename Atom>
bool verify_samples(
    const std::vector<typename Atom::TA>& a,
    const std::vector<typename Atom::TB>& b,
    const std::vector<typename Atom::TAcc>& c,
    nint_t m, nint_t n, nint_t k) {
  using Acc = typename Atom::TAcc;
  const std::array<std::array<nint_t, 2>, 5> samples{{
      {{0, 0}},
      {{m - 1, n - 1}},
      {{m / 2, n / 2}},
      {{0, n - 1}},
      {{m - 1, 0}},
  }};
  for (const auto& sample : samples) {
    const nint_t row = sample[0];
    const nint_t col = sample[1];
    Acc expected{};
    for (nint_t kk = 0; kk < k; ++kk) {
      expected += static_cast<Acc>(a[static_cast<std::size_t>(row * k + kk)]) *
                  static_cast<Acc>(b[static_cast<std::size_t>(col * k + kk)]);
    }
    const Acc actual = c[static_cast<std::size_t>(row * n + col)];
    if constexpr (std::is_floating_point_v<Acc>) {
      constexpr bool low_precision_input =
          sizeof(typename Atom::TA) < sizeof(Acc) ||
          sizeof(typename Atom::TB) < sizeof(Acc);
      const double tolerance = low_precision_input ? 5.0e-3 : 2.0e-4;
      const double scale = std::max(
          {1.0, std::abs(static_cast<double>(expected)),
           std::abs(static_cast<double>(actual))});
      if (std::abs(static_cast<double>(actual) -
                   static_cast<double>(expected)) > tolerance * scale) {
        return false;
      }
    } else if (actual != expected) {
      return false;
    }
  }
  return true;
}

template <typename Atom, InputMode Mode,
          typename TilePolicy = kernel::matmul_policy::Automatic,
          typename ATensor, typename BTensor,
          typename CTensor, typename M, typename N, typename K>
void run_operation(
    benchmark::State& state, const MatmulCase& test_case,
    M m, N n, K k,
    const ATensor& a_input, const BTensor& b_input, const CTensor& c_output,
    const std::vector<typename Atom::TA>& a,
    const std::vector<typename Atom::TB>& b,
    std::vector<typename Atom::TAcc>& c,
    nint_t packed_a_elements, nint_t packed_b_elements) {
  auto operation = ops::make_matmul<Atom, TilePolicy>(
      m, n, k, a_input, b_input, c_output);
  kernel::Workspace operation_storage(operation.required_workspace());
  auto operation_workspace = operation_storage.view();
  ExecutionSession execution{operation_workspace};

  operation(execution);
  if (!verify_samples<Atom>(
          a, b, c, test_case.m, test_case.n, test_case.k)) {
    state.SkipWithError("matmul result verification failed");
    return;
  }

  for (auto _ : state) {
    benchmark::DoNotOptimize(a.data());
    benchmark::DoNotOptimize(b.data());
    operation(execution);
    benchmark::DoNotOptimize(c.data());
    benchmark::ClobberMemory();
  }

  const int64_t mnk = static_cast<int64_t>(test_case.m) * test_case.n *
                      test_case.k;
  const int64_t logical_flops = 2 * mnk;
  const int64_t logical_bytes =
      static_cast<int64_t>(test_case.m) * test_case.k *
          sizeof(typename Atom::TA) +
      static_cast<int64_t>(test_case.n) * test_case.k *
          sizeof(typename Atom::TB) +
      static_cast<int64_t>(test_case.m) * test_case.n *
          sizeof(typename Atom::TAcc);
  const nint_t active_packed_a =
      Mode == InputMode::PackedA || Mode == InputMode::PackedAB
          ? packed_a_elements
          : 0;
  const nint_t active_packed_b =
      Mode == InputMode::PackedB || Mode == InputMode::PackedAB
          ? packed_b_elements
          : 0;
  const nint_t logical_packed_elements =
      (Mode == InputMode::PackedA || Mode == InputMode::PackedAB
           ? test_case.m * test_case.k
           : 0) +
      (Mode == InputMode::PackedB || Mode == InputMode::PackedAB
           ? test_case.n * test_case.k
           : 0);

  state.SetItemsProcessed(state.iterations() * mnk);
  state.SetBytesProcessed(state.iterations() * logical_bytes);
  state.counters["FLOP/s"] = benchmark::Counter(
      static_cast<double>(logical_flops),
      benchmark::Counter::kIsIterationInvariantRate);
  state.counters["m"] = benchmark::Counter(double(test_case.m));
  state.counters["n"] = benchmark::Counter(double(test_case.n));
  state.counters["k"] = benchmark::Counter(double(test_case.k));
  state.counters["workspace_bytes"] =
      benchmark::Counter(double(operation.required_workspace()));
  state.counters["packed_elements"] = benchmark::Counter(
      double(active_packed_a + active_packed_b));
  state.counters["packing_ratio"] = benchmark::Counter(
      logical_packed_elements == 0
          ? 1.0
          : double(active_packed_a + active_packed_b) /
                double(logical_packed_elements));
}

template <typename Atom, InputMode Mode,
          typename TilePolicy = kernel::matmul_policy::Automatic,
          typename M, typename N, typename K>
void run_case_with_extents(
    benchmark::State& state, const MatmulCase& test_case,
    M m, N n, K k) {
  using TA = typename Atom::TA;
  using TB = typename Atom::TB;
  using Acc = typename Atom::TAcc;

  std::vector<TA> a(
      static_cast<std::size_t>(test_case.m * test_case.k));
  std::vector<TB> b(
      static_cast<std::size_t>(test_case.n * test_case.k));
  std::vector<Acc> c(
      static_cast<std::size_t>(test_case.m * test_case.n), Acc{});
  fill_input(a, 3);
  fill_input(b, 11);

  auto a_layout = make_layout(make_shape(m, k));
  auto b_layout = make_layout(make_shape(n, k));
  auto c_layout = make_layout(make_shape(m, n));
  auto a_tensor = make_tensor(a.data(), a_layout);
  auto b_tensor = make_tensor(b.data(), b_layout);
  auto c_tensor = make_tensor(c.data(), c_layout);

  auto packed_a_layout =
      ops::matmul_packed_layout<Atom, gemm::Operand::A>(a_layout);
  auto packed_b_layout =
      ops::matmul_packed_layout<Atom, gemm::Operand::B>(b_layout);
  const nint_t packed_a_elements = numel(packed_a_layout);
  const nint_t packed_b_elements = numel(packed_b_layout);
  const nint_t packed_a_bytes = packed_a_elements * nint_t{sizeof(TA)};
  const nint_t packed_b_bytes = packed_b_elements * nint_t{sizeof(TB)};
  kernel::Workspace packed_a_storage(packed_a_bytes + 64);
  kernel::Workspace packed_b_storage(packed_b_bytes + 64);
  auto packed_a_workspace = packed_a_storage.view();
  auto packed_b_workspace = packed_b_storage.view();
  auto* packed_a = static_cast<TA*>(
      packed_a_workspace.allocate(packed_a_bytes, 64));
  auto* packed_b = static_cast<TB*>(
      packed_b_workspace.allocate(packed_b_bytes, 64));
  auto packed_a_tensor = make_tensor(packed_a, packed_a_layout);
  auto packed_b_tensor = make_tensor(packed_b, packed_b_layout);

  ExecutionSession pack_execution{};
  if constexpr (Mode == InputMode::PackedA || Mode == InputMode::PackedAB) {
    ops::matmul_pack<Atom, gemm::Operand::A>(
        pack_execution, a_tensor, packed_a_tensor);
  }
  if constexpr (Mode == InputMode::PackedB || Mode == InputMode::PackedAB) {
    ops::matmul_pack<Atom, gemm::Operand::B>(
        pack_execution, b_tensor, packed_b_tensor);
  }

  if constexpr (Mode == InputMode::Raw) {
    run_operation<Atom, Mode, TilePolicy>(
        state, test_case, m, n, k,
        a_tensor, b_tensor, c_tensor, a, b, c,
        packed_a_elements, packed_b_elements);
  } else if constexpr (Mode == InputMode::PackedA) {
    run_operation<Atom, Mode, TilePolicy>(
        state, test_case, m, n, k,
        packed_a_tensor, b_tensor, c_tensor, a, b, c,
        packed_a_elements, packed_b_elements);
  } else if constexpr (Mode == InputMode::PackedB) {
    run_operation<Atom, Mode, TilePolicy>(
        state, test_case, m, n, k,
        a_tensor, packed_b_tensor, c_tensor, a, b, c,
        packed_a_elements, packed_b_elements);
  } else {
    run_operation<Atom, Mode, TilePolicy>(
        state, test_case, m, n, k,
        packed_a_tensor, packed_b_tensor, c_tensor,
        a, b, c, packed_a_elements, packed_b_elements);
  }
}

template <typename Atom, InputMode Mode,
          typename TilePolicy = kernel::matmul_policy::Automatic>
void run_case(benchmark::State& state, const MatmulCase& test_case) {
  run_case_with_extents<Atom, Mode, TilePolicy>(
      state, test_case,
      Any{test_case.m}, Any{test_case.n}, Any{test_case.k});
}

template <typename Atom, InputMode Mode,
          nint_t M, nint_t N, nint_t K,
          typename TilePolicy = kernel::matmul_policy::Automatic>
void run_fixed_case(benchmark::State& state, const MatmulCase& test_case) {
  run_case_with_extents<Atom, Mode, TilePolicy>(
      state, test_case, cint<M>, cint<N>, cint<K>);
}

template <typename Atom, InputMode Mode>
std::string benchmark_name(const MatmulCase& test_case) {
  using TA = typename Atom::TA;
  using TB = typename Atom::TB;
  using Acc = typename Atom::TAcc;
  return "Matmul/" + std::string(test_case.group) +
         "/case:" + test_case.name +
         "/shape:" + std::to_string(test_case.m) + "x" +
             std::to_string(test_case.n) + "x" +
             std::to_string(test_case.k) +
         "/input:" + input_mode_name<Mode>() +
         "/dtype:" + dtype_name<TA>() + "x" + dtype_name<TB>() +
             "_acc_" + dtype_name<Acc>() +
         "/atom:" + atom_name<Atom>() +
         "/arch:" + VECOPS_BENCH_ARCH_CODE;
}

template <typename Atom, InputMode Mode>
void register_mode(const MatmulCase& test_case) {
  if ((test_case.modes & mode_bit(Mode)) == 0) return;
  const auto name = benchmark_name<Atom, Mode>(test_case);
  benchmark::RegisterBenchmark(
      name.c_str(),
      [test_case](benchmark::State& state) {
        run_case<Atom, Mode>(state, test_case);
      })
      ->Unit(benchmark::kMicrosecond)
      ->MinTime(0.02)
      ->Repetitions(3)
      ->ReportAggregatesOnly(true);
}

template <typename Atom, InputMode Mode,
          nint_t M, nint_t N, nint_t K>
void register_fixed_mode(const char* group, const char* name) {
  constexpr uint32_t ModeMask = mode_bit(Mode);
  const MatmulCase test_case{group, name, M, N, K, ModeMask};
  const auto full_name = benchmark_name<Atom, Mode>(test_case);
  benchmark::RegisterBenchmark(
      full_name.c_str(),
      [test_case](benchmark::State& state) {
        run_fixed_case<Atom, Mode, M, N, K>(state, test_case);
      })
      ->Unit(benchmark::kMicrosecond)
      ->MinTime(0.02)
      ->Repetitions(3)
      ->ReportAggregatesOnly(true);
}

template <typename Atom, InputMode Mode, typename TilePolicy,
          nint_t M, nint_t N, nint_t K>
void register_fixed_policy_mode(
    const char* group, const char* name, const char* policy_name) {
  constexpr uint32_t ModeMask = mode_bit(Mode);
  const MatmulCase test_case{group, name, M, N, K, ModeMask};
  const auto full_name = benchmark_name<Atom, Mode>(test_case) +
      "/policy:" + policy_name;
  benchmark::RegisterBenchmark(
      full_name.c_str(),
      [test_case](benchmark::State& state) {
        run_fixed_case<Atom, Mode, M, N, K, TilePolicy>(state, test_case);
      })
      ->Unit(benchmark::kMicrosecond)
      ->MinTime(0.02)
      ->Repetitions(3)
      ->ReportAggregatesOnly(true);
}

template <typename Atom>
void register_case(const MatmulCase& test_case) {
  register_mode<Atom, InputMode::Raw>(test_case);
  register_mode<Atom, InputMode::PackedA>(test_case);
  register_mode<Atom, InputMode::PackedB>(test_case);
  register_mode<Atom, InputMode::PackedAB>(test_case);
}

template <typename Atom>
void register_cases(const std::vector<MatmulCase>& cases) {
  for (const auto& test_case : cases) register_case<Atom>(test_case);
}

// Mixed packing is covered by the primary Atom. Dtype probes instantiate only
// the two end points to keep this template-heavy benchmark reasonably small.
template <typename Atom>
void register_probe_cases(const std::vector<MatmulCase>& cases) {
  for (const auto& test_case : cases) {
    register_mode<Atom, InputMode::Raw>(test_case);
    register_mode<Atom, InputMode::PackedAB>(test_case);
  }
}

inline std::vector<MatmulCase> representative_cases(
    nint_t tile, nint_t k_step, int max_strip) {
  std::vector<MatmulCase> cases{
      {"microkernel", "acc_1x1", tile, tile, 4 * k_step,
       mode_bit(InputMode::PackedAB)},
      {"microkernel", "acc_1x2", tile, 2 * tile, 4 * k_step,
       mode_bit(InputMode::PackedAB)},
      {"microkernel", "acc_1x3", tile, 3 * tile, 4 * k_step,
       mode_bit(InputMode::PackedAB)},
      {"microkernel", "acc_2x1", 2 * tile, tile, 4 * k_step,
       mode_bit(InputMode::PackedAB)},
      {"microkernel", "acc_3x1", 3 * tile, tile, 4 * k_step,
       mode_bit(InputMode::PackedAB)},
      {"microkernel", "acc_2x2", 2 * tile, 2 * tile, 4 * k_step,
       mode_bit(InputMode::PackedAB)},

      {"tail", "m_tail", 2 * tile + 1, 2 * tile, 8 * k_step,
       kRawAndPackedAB},
      {"tail", "n_tail", 2 * tile, 2 * tile + 1, 8 * k_step,
       kRawAndPackedAB},
      {"tail", "mn_tail", 2 * tile + 3, 3 * tile + 5, 8 * k_step,
       kRawAndPackedAB},
      {"tail", "k_tail", 2 * tile, 2 * tile, 8 * k_step + 1,
       kRawAndPackedAB},
      {"tail", "mnk_tail", 2 * tile + 3, 3 * tile + 5,
       8 * k_step + 1, kAllInputModes},

      {"representative", "decode_projection", 1, 1152, 896,
       kWeightReuseModes},
      {"representative", "batch_projection", 128, 1024, 768,
       kWeightReuseModes},
      {"representative", "square", 256, 256, 256,
       kWeightReuseModes},
      {"representative", "ragged_projection", 127, 1025, 769,
       kWeightReuseModes},
  };
  static constexpr const char* horizontal_names[] = {
      "", "", "", "", "acc_1x4", "acc_1x5", "acc_1x6"};
  static constexpr const char* vertical_names[] = {
      "", "", "", "", "acc_4x1", "acc_5x1", "acc_6x1"};
  for (int strip = 4; strip <= max_strip; ++strip) {
    cases.insert(cases.begin() + 6, {
        {"microkernel", horizontal_names[strip],
         tile, strip * tile, 4 * k_step,
         mode_bit(InputMode::PackedAB)},
        {"microkernel", vertical_names[strip],
         strip * tile, tile, 4 * k_step,
         mode_bit(InputMode::PackedAB)},
    });
  }
  return cases;
}

inline std::vector<MatmulCase> dtype_probe_cases(
    nint_t tile, nint_t k_step) {
  return {{"dtype_coverage", "tail_probe", tile + 3, tile + 5,
           8 * k_step + 1, kRawAndPackedAB}};
}

inline int run_benchmarks(
    int argc, char** argv, const char* result_stem) {
  const auto default_csv = vecops::bench::default_result_path(
      VECOPS_SOURCE_DIR, result_stem, VECOPS_BENCH_ARCH_CODE, "csv");
  auto injected = vecops::bench::default_google_benchmark_output_args(
      argc, argv, default_csv, "csv");
  std::vector<char*> args;
  args.reserve(static_cast<std::size_t>(argc) + injected.size());
  for (int i = 0; i < argc; ++i) args.push_back(argv[i]);
  for (auto& arg : injected) args.push_back(arg.data());
  int bench_argc = static_cast<int>(args.size());
  benchmark::Initialize(&bench_argc, args.data());
  if (benchmark::ReportUnrecognizedArguments(bench_argc, args.data())) {
    return 1;
  }
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  vecops::bench::print_default_output_path(argc, argv, default_csv);
  return 0;
}

} // namespace vecops::bench::matmul

#endif // VECOPS_BENCHMARKS_OPS_MATMUL_BENCH_COMMON_H
