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
#include <utility>
#include <vector>

#include "BenchmarkUtils.h"
#include "vecops/matmul/Packing.h"
#include "vecops/kernel/Workspace.h"
#include "vecops/matmul/Matmul.h"
#include "vecops/matmul/MatmulPack.h"

#ifndef VECOPS_BENCH_ARCH_CODE
#define VECOPS_BENCH_ARCH_CODE "unknown"
#endif

#ifndef VECOPS_SOURCE_DIR
#define VECOPS_SOURCE_DIR "."
#endif

namespace vecops::bench::matmul {

#if !defined(VECOPS_MATMUL_CATALOG_CASE_SHARDS)
#define VECOPS_MATMUL_CATALOG_CASE_SHARDS 1
#endif

inline constexpr int MatmulCatalogCaseShardCount =
    VECOPS_MATMUL_CATALOG_CASE_SHARDS;

using namespace ::vecops::meta;
using namespace ::vecops::tensor;

enum class InputMode : uint32_t {
  Raw = 1u << 0,
  PackedA = 1u << 1,
  PackedB = 1u << 2,
  PackedAB = 1u << 3,
  OnlinePackedA = 1u << 4,
  OnlinePackedB = 1u << 5,
  OnlinePackedAB = 1u << 6,
};

enum class ExtentMode : uint8_t {
  Dynamic,
  Const,
};

template <ExtentMode Mode>
constexpr const char* extent_mode_name() {
  if constexpr (Mode == ExtentMode::Dynamic) return "Dynamic";
  return "Const";
}

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
  if constexpr (Mode == InputMode::PackedAB) return "packed_ab";
  if constexpr (Mode == InputMode::OnlinePackedA) return "online_packed_a";
  if constexpr (Mode == InputMode::OnlinePackedB) return "online_packed_b";
  return "online_packed_ab";
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
std::string benchmark_name(
    const MatmulCase& test_case,
    const char* extent = extent_mode_name<ExtentMode::Dynamic>()) {
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
         "/extent:" + extent +
         "/arch:" + VECOPS_BENCH_ARCH_CODE;
}

template <typename Atom, InputMode Mode, ExtentMode Extents,
          nint_t M, nint_t N, nint_t K,
          typename TilePolicy = kernel::matmul_policy::Automatic>
void register_extent_mode(const char* group, const char* name) {
  constexpr uint32_t ModeMask = mode_bit(Mode);
  const MatmulCase test_case{group, name, M, N, K, ModeMask};
  const auto full_name = benchmark_name<Atom, Mode>(
      test_case, extent_mode_name<Extents>());
  auto* registered = benchmark::RegisterBenchmark(
      full_name.c_str(),
      [test_case](benchmark::State& state) {
        if constexpr (Extents == ExtentMode::Dynamic) {
          run_case<Atom, Mode, TilePolicy>(state, test_case);
        } else {
          run_fixed_case<Atom, Mode, M, N, K, TilePolicy>(state, test_case);
        }
      })
      ->Unit(benchmark::kMicrosecond);
  ::vecops::bench::configure_registered_benchmark(
      registered, 0.02, 3)->ReportAggregatesOnly(true);
}

template <typename Atom, InputMode Mode,
          nint_t M, nint_t N, nint_t K,
          typename TilePolicy = kernel::matmul_policy::Automatic>
void register_extent_pair_mode(const char* group, const char* name) {
#if defined(VECOPS_SPLIT_EXTENT_SHARDS) && \
    defined(VECOPS_TARGET_SHARD_ACTIVE)
  if constexpr ((VECOPS_TARGET_SHARD_INDEX % 2) == 0) {
    register_extent_mode<
        Atom, Mode, ExtentMode::Dynamic, M, N, K, TilePolicy>(group, name);
  } else {
    register_extent_mode<
        Atom, Mode, ExtentMode::Const, M, N, K, TilePolicy>(group, name);
  }
#else
  register_extent_mode<
      Atom, Mode, ExtentMode::Dynamic, M, N, K, TilePolicy>(group, name);
  register_extent_mode<
      Atom, Mode, ExtentMode::Const, M, N, K, TilePolicy>(group, name);
#endif
}

template <typename Atom, nint_t M, nint_t N, nint_t K,
          uint32_t Modes, int CaseShard = 0, int ShardTag = 0>
void register_extent_pair(const char* group, const char* name) {
#if defined(VECOPS_TARGET_SHARD_ACTIVE)
  static_assert(CaseShard >= 0);
  constexpr bool ActiveCaseShard =
      (ShardTag % MatmulCatalogCaseShardCount) ==
      (CaseShard % MatmulCatalogCaseShardCount);
#else
  constexpr bool ActiveCaseShard = true;
#endif
  if constexpr (ActiveCaseShard) {
    if constexpr ((Modes & mode_bit(InputMode::Raw)) != 0) {
      register_extent_pair_mode<Atom, InputMode::Raw, M, N, K>(group, name);
    }
    if constexpr ((Modes & mode_bit(InputMode::PackedA)) != 0) {
      register_extent_pair_mode<Atom, InputMode::PackedA, M, N, K>(group, name);
    }
    if constexpr ((Modes & mode_bit(InputMode::PackedB)) != 0) {
      register_extent_pair_mode<Atom, InputMode::PackedB, M, N, K>(group, name);
    }
    if constexpr ((Modes & mode_bit(InputMode::PackedAB)) != 0) {
      register_extent_pair_mode<Atom, InputMode::PackedAB, M, N, K>(
          group, name);
    }
  }
}

template <typename Atom, InputMode Mode>
void register_mode(const MatmulCase& test_case) {
  if ((test_case.modes & mode_bit(Mode)) == 0) return;
  const auto name = benchmark_name<Atom, Mode>(test_case);
  auto* registered = benchmark::RegisterBenchmark(
      name.c_str(),
      [test_case](benchmark::State& state) {
        run_case<Atom, Mode>(state, test_case);
      })
      ->Unit(benchmark::kMicrosecond);
  ::vecops::bench::configure_registered_benchmark(
      registered, 0.02, 3)->ReportAggregatesOnly(true);
}

template <typename Atom, InputMode Mode,
          nint_t M, nint_t N, nint_t K>
void register_fixed_mode(const char* group, const char* name) {
  constexpr uint32_t ModeMask = mode_bit(Mode);
  const MatmulCase test_case{group, name, M, N, K, ModeMask};
  const auto full_name = benchmark_name<Atom, Mode>(test_case);
  auto* registered = benchmark::RegisterBenchmark(
      full_name.c_str(),
      [test_case](benchmark::State& state) {
        run_fixed_case<Atom, Mode, M, N, K>(state, test_case);
      })
      ->Unit(benchmark::kMicrosecond);
  ::vecops::bench::configure_registered_benchmark(
      registered, 0.02, 3)->ReportAggregatesOnly(true);
}

template <typename Atom, InputMode Mode, typename TilePolicy,
          nint_t M, nint_t N, nint_t K, ExtentMode Extents>
void register_policy_extent_mode(
    const char* group, const char* name, const char* policy_name) {
  constexpr uint32_t ModeMask = mode_bit(Mode);
  const MatmulCase test_case{group, name, M, N, K, ModeMask};
  const auto full_name = benchmark_name<Atom, Mode>(
      test_case, extent_mode_name<Extents>()) + "/policy:" + policy_name;
  auto* registered = benchmark::RegisterBenchmark(
      full_name.c_str(),
      [test_case](benchmark::State& state) {
        if constexpr (Extents == ExtentMode::Dynamic) {
          run_case<Atom, Mode, TilePolicy>(state, test_case);
        } else {
          run_fixed_case<Atom, Mode, M, N, K, TilePolicy>(state, test_case);
        }
      })
      ->Unit(benchmark::kMicrosecond);
  ::vecops::bench::configure_registered_benchmark(
      registered, 0.02, 3)->ReportAggregatesOnly(true);
}

template <typename Atom, InputMode Mode, typename TilePolicy,
          nint_t M, nint_t N, nint_t K>
void register_policy_extent_pair(
    const char* group, const char* name, const char* policy_name) {
#if defined(VECOPS_SPLIT_EXTENT_SHARDS) && \
    defined(VECOPS_TARGET_SHARD_ACTIVE)
  if constexpr ((VECOPS_TARGET_SHARD_INDEX % 2) == 0) {
    register_policy_extent_mode<
        Atom, Mode, TilePolicy, M, N, K, ExtentMode::Dynamic>(
            group, name, policy_name);
  } else {
    register_policy_extent_mode<
        Atom, Mode, TilePolicy, M, N, K, ExtentMode::Const>(
            group, name, policy_name);
  }
#else
  register_policy_extent_mode<
      Atom, Mode, TilePolicy, M, N, K, ExtentMode::Dynamic>(
          group, name, policy_name);
  register_policy_extent_mode<
      Atom, Mode, TilePolicy, M, N, K, ExtentMode::Const>(
          group, name, policy_name);
#endif
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

// Dtype probes cover both mixed-packing directions as well as the two end
// points.  Packing formats and mixed raw/packed kernel plans are Atom-specific;
// validating only the primary Atom can hide dtype-specific regressions.
template <typename Atom>
void register_probe_cases(const std::vector<MatmulCase>& cases) {
  for (const auto& test_case : cases) {
    register_mode<Atom, InputMode::Raw>(test_case);
    register_mode<Atom, InputMode::PackedA>(test_case);
    register_mode<Atom, InputMode::PackedB>(test_case);
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
       kAllInputModes},
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
  return {
      {"dtype_coverage", "tail_probe", tile + 3, tile + 5,
       8 * k_step + 1, kAllInputModes},
      {"dtype_coverage", "decode_probe", 1, 1024, 1024,
       kAllInputModes},
      {"dtype_coverage", "batch_probe", 64, 256, 256,
       kAllInputModes},
  };
}

/**
 * Runtime-shape coverage shared by the primary AMX/SME atom.
 *
 * Keep these separate from representative_cases(): the latter is a compact
 * microkernel/tail regression set, while this list deliberately spans model
 * serving, convolution lowering, skinny-K and strongly rectangular GEMMs.
 * All dimensions are runtime values, so expanding the list adds benchmark
 * registrations without multiplying kernel template instantiations.
 */
inline std::vector<MatmulCase> workload_shape_cases(
    nint_t tile, nint_t k_step) {
  return {
      // Boundary probes around the native spatial and K tiles.
      {"shape_coverage", "below_tile", tile - 1, tile - 1,
       4 * k_step - 1, kRawAndPackedAB},
      {"shape_coverage", "one_by_one", 1, 1, 8 * k_step + 1,
       kRawAndPackedAB},
      {"shape_coverage", "single_row_wide", 1, 8 * tile + 3,
       16 * k_step + 1, kAllInputModes},
      {"shape_coverage", "single_col_tall", 8 * tile + 3, 1,
       16 * k_step + 1, kAllInputModes},
      {"shape_coverage", "skinny_k", 16 * tile, 16 * tile,
       2 * k_step, kWeightReuseModes},
      {"shape_coverage", "long_k", 2 * tile + 1, 2 * tile + 3,
       256 * k_step + 1, kWeightReuseModes},

      // Transformer inference: token/decode, small batches and prefill.
      {"transformer", "decode_qkv_4k", 1, 4096, 4096,
       kWeightReuseModes},
      {"transformer", "decode_qkv_batch4", 4, 4096, 4096,
       kWeightReuseModes},
      {"transformer", "prefill_qkv_4k", 128, 4096, 4096,
       kWeightReuseModes},
      {"transformer", "decode_mlp_up", 1, 11008, 4096,
       kWeightReuseModes},
      {"transformer", "decode_mlp_down", 1, 4096, 11008,
       kWeightReuseModes},
      {"transformer", "prefill_mlp_up", 128, 11008, 4096,
       kWeightReuseModes},
      {"transformer", "attention_score", 128, 128, 128,
       kWeightReuseModes},

      // Common GEMM shapes outside transformer projections.
      {"workload", "conv_1x1_lowering", 3136, 64, 576,
       kWeightReuseModes},
      {"workload", "batched_square_k4k", 256, 256, 4096,
       kWeightReuseModes},
      {"workload", "rank_reduction", 1024, 64, 4096,
       kWeightReuseModes},
      {"workload", "rank_expansion", 64, 1024, 4096,
       kWeightReuseModes},
  };
}

/**
 * Runtime grid for shape-aware dispatch and vector-fallback crossovers.
 *
 * x86 already dispatches selected raw tiny/GEMV problems away from AMX.  SME
 * currently always enters its MOPA traversal, so keep an architecture-common
 * grid that makes the two backends directly comparable and provides controls
 * for future SVE-vs-SME selection.  Values are absolute rather than k_step
 * scaled so both machines benchmark the same mathematical shapes.
 */
inline std::vector<MatmulCase> dispatch_shape_cases(nint_t tile) {
  return {
      {"dispatch_tiny", "m1_n8_k64", 1, 8, 64, kRawAndPackedAB},
      {"dispatch_tiny", "m8_n1_k64", 8, 1, 64, kRawAndPackedAB},
      {"dispatch_tiny", "m2_n4_k65", 2, 4, 65, kRawAndPackedAB},
      {"dispatch_tiny", "m4_n2_k65", 4, 2, 65, kRawAndPackedAB},
      {"dispatch_tiny", "m4_n4_k256", 4, 4, 256, kRawAndPackedAB},
      {"dispatch_tiny", "m8_n8_k257", 8, 8, 257, kRawAndPackedAB},
      {"dispatch_tiny", "m16_n16_k1024", 16, 16, 1024,
       kRawAndPackedAB},

      {"dispatch_gemv_row", "n16_k65", 1, 16, 65, kWeightReuseModes},
      {"dispatch_gemv_row", "n32_k256", 1, 32, 256,
       kWeightReuseModes},
      {"dispatch_gemv_row", "n64_k257", 1, 64, 257,
       kWeightReuseModes},
      {"dispatch_gemv_row", "n128_k1024", 1, 128, 1024,
       kWeightReuseModes},
      {"dispatch_gemv_row", "n256_k1025", 1, 256, 1025,
       kWeightReuseModes},
      {"dispatch_gemv_row", "n512_k4096", 1, 512, 4096,
       kWeightReuseModes},
      {"dispatch_gemv_row", "n1024_k4097", 1, 1024, 4097,
       kWeightReuseModes},

      {"dispatch_gemv_col", "m16_k65", 16, 1, 65, kAllInputModes},
      {"dispatch_gemv_col", "m32_k256", 32, 1, 256, kAllInputModes},
      {"dispatch_gemv_col", "m64_k257", 64, 1, 257, kAllInputModes},
      {"dispatch_gemv_col", "m128_k1024", 128, 1, 1024,
       kAllInputModes},
      {"dispatch_gemv_col", "m256_k1025", 256, 1, 1025,
       kAllInputModes},
      {"dispatch_gemv_col", "m512_k4096", 512, 1, 4096,
       kAllInputModes},
      {"dispatch_gemv_col", "m1024_k4097", 1024, 1, 4097,
       kAllInputModes},

      {"dispatch_tail", "short_m_long_k", tile + 1, 2 * tile + 1,
       1024, kAllInputModes},
      {"dispatch_tail", "short_n_long_k", 2 * tile + 1, tile + 1,
       1025, kAllInputModes},
  };
}

/** Compact raw-only SVE/SME crossover coverage for secondary floating atoms. */
inline std::vector<MatmulCase> floating_skinny_probe_cases() {
  constexpr uint32_t Raw = mode_bit(InputMode::Raw);
  return {
      {"dispatch_float_skinny", "m1_n8_k64", 1, 8, 64, Raw},
      {"dispatch_float_skinny", "m8_n1_k64", 8, 1, 64, Raw},
      {"dispatch_float_skinny", "m1_n16_k65", 1, 16, 65, Raw},
      {"dispatch_float_skinny", "m16_n1_k65", 16, 1, 65, Raw},
      {"dispatch_float_skinny", "m1_n64_k257", 1, 64, 257, Raw},
      {"dispatch_float_skinny", "m64_n1_k257", 64, 1, 257, Raw},
  };
}

/** Compact raw-only SVE dot/SME crossover coverage for integer atoms. */
inline std::vector<MatmulCase> integer_skinny_probe_cases() {
  constexpr uint32_t Raw = mode_bit(InputMode::Raw);
  return {
      {"dispatch_integer_skinny", "m1_n8_k64", 1, 8, 64, Raw},
      {"dispatch_integer_skinny", "m8_n1_k64", 8, 1, 64, Raw},
      {"dispatch_integer_skinny", "m1_n16_k65", 1, 16, 65, Raw},
      {"dispatch_integer_skinny", "m16_n1_k65", 16, 1, 65, Raw},
      {"dispatch_integer_skinny", "m1_n64_k257", 1, 64, 257, Raw},
      {"dispatch_integer_skinny", "m64_n1_k257", 64, 1, 257, Raw},
  };
}

/** PackedAB tiny shapes used to gate ordinary-SVE MMLA against SME MOPA. */
inline std::vector<MatmulCase> packed_mmla_probe_cases() {
  constexpr uint32_t PackedAB = mode_bit(InputMode::PackedAB);
  return {
      {"dispatch_packed_mmla", "m2_n2_k65", 2, 2, 65, PackedAB},
      {"dispatch_packed_mmla", "m2_n2_k257", 2, 2, 257, PackedAB},
      {"dispatch_packed_mmla", "m2_n2_k513", 2, 2, 513, PackedAB},
      {"dispatch_packed_mmla", "m2_n2_k1025", 2, 2, 1025, PackedAB},
      {"dispatch_packed_mmla", "m2_n4_k65", 2, 4, 65, PackedAB},
      {"dispatch_packed_mmla", "m2_n4_k257", 2, 4, 257, PackedAB},
      {"dispatch_packed_mmla", "m2_n4_k513", 2, 4, 513, PackedAB},
      {"dispatch_packed_mmla", "m2_n4_k1025", 2, 4, 1025, PackedAB},
      {"dispatch_packed_mmla", "m4_n2_k65", 4, 2, 65, PackedAB},
      {"dispatch_packed_mmla", "m4_n2_k257", 4, 2, 257, PackedAB},
      {"dispatch_packed_mmla", "m4_n2_k513", 4, 2, 513, PackedAB},
      {"dispatch_packed_mmla", "m4_n2_k1025", 4, 2, 1025, PackedAB},
      {"dispatch_packed_mmla", "m2_n8_k65", 2, 8, 65, PackedAB},
      {"dispatch_packed_mmla", "m2_n8_k257", 2, 8, 257, PackedAB},
      {"dispatch_packed_mmla", "m2_n8_k513", 2, 8, 513, PackedAB},
      {"dispatch_packed_mmla", "m2_n8_k1025", 2, 8, 1025, PackedAB},
      {"dispatch_packed_mmla", "m8_n2_k65", 8, 2, 65, PackedAB},
      {"dispatch_packed_mmla", "m8_n2_k257", 8, 2, 257, PackedAB},
      {"dispatch_packed_mmla", "m8_n2_k513", 8, 2, 513, PackedAB},
      {"dispatch_packed_mmla", "m8_n2_k1025", 8, 2, 1025, PackedAB},
      {"dispatch_packed_mmla", "m4_n4_k65", 4, 4, 65, PackedAB},
      {"dispatch_packed_mmla", "m4_n4_k257", 4, 4, 257, PackedAB},
      {"dispatch_packed_mmla", "m4_n4_k513", 4, 4, 513, PackedAB},
      {"dispatch_packed_mmla", "m4_n4_k1025", 4, 4, 1025, PackedAB},
  };
}

// Compile-time counterparts of the runtime catalogs above.  Keeping the
// dimensions as non-type template arguments is what lets the Const variant
// expose the complete shape to the matmul metaprogram, while the paired
// Dynamic registration still constructs Any extents from the same values.
template <typename Atom, nint_t Tile, nint_t KStep, int MaxStrip,
          int ShardTag = 0>
void register_representative_extent_pairs() {
  register_extent_pair<Atom, Tile, Tile, 4 * KStep,
                       mode_bit(InputMode::PackedAB), 0, ShardTag>(
      "microkernel", "acc_1x1");
  register_extent_pair<Atom, Tile, 2 * Tile, 4 * KStep,
                       mode_bit(InputMode::PackedAB), 1, ShardTag>(
      "microkernel", "acc_1x2");
  register_extent_pair<Atom, Tile, 3 * Tile, 4 * KStep,
                       mode_bit(InputMode::PackedAB), 2, ShardTag>(
      "microkernel", "acc_1x3");
  register_extent_pair<Atom, 2 * Tile, Tile, 4 * KStep,
                       mode_bit(InputMode::PackedAB), 3, ShardTag>(
      "microkernel", "acc_2x1");
  register_extent_pair<Atom, 3 * Tile, Tile, 4 * KStep,
                       mode_bit(InputMode::PackedAB), 4, ShardTag>(
      "microkernel", "acc_3x1");
  register_extent_pair<Atom, 2 * Tile, 2 * Tile, 4 * KStep,
                       mode_bit(InputMode::PackedAB), 5, ShardTag>(
      "microkernel", "acc_2x2");
  if constexpr (MaxStrip >= 4) {
    register_extent_pair<Atom, Tile, 4 * Tile, 4 * KStep,
                         mode_bit(InputMode::PackedAB), 6, ShardTag>(
        "microkernel", "acc_1x4");
    register_extent_pair<Atom, 4 * Tile, Tile, 4 * KStep,
                         mode_bit(InputMode::PackedAB), 7, ShardTag>(
        "microkernel", "acc_4x1");
  }
  if constexpr (MaxStrip >= 5) {
    register_extent_pair<Atom, Tile, 5 * Tile, 4 * KStep,
                         mode_bit(InputMode::PackedAB), 8, ShardTag>(
        "microkernel", "acc_1x5");
    register_extent_pair<Atom, 5 * Tile, Tile, 4 * KStep,
                         mode_bit(InputMode::PackedAB), 9, ShardTag>(
        "microkernel", "acc_5x1");
  }
  if constexpr (MaxStrip >= 6) {
    register_extent_pair<Atom, Tile, 6 * Tile, 4 * KStep,
                         mode_bit(InputMode::PackedAB), 10, ShardTag>(
        "microkernel", "acc_1x6");
    register_extent_pair<Atom, 6 * Tile, Tile, 4 * KStep,
                         mode_bit(InputMode::PackedAB), 11, ShardTag>(
        "microkernel", "acc_6x1");
  }
  register_extent_pair<Atom, 2 * Tile + 1, 2 * Tile, 8 * KStep,
                       kRawAndPackedAB, 12, ShardTag>("tail", "m_tail");
  register_extent_pair<Atom, 2 * Tile, 2 * Tile + 1, 8 * KStep,
                       kRawAndPackedAB, 13, ShardTag>("tail", "n_tail");
  register_extent_pair<Atom, 2 * Tile + 3, 3 * Tile + 5, 8 * KStep,
                       kRawAndPackedAB, 14, ShardTag>("tail", "mn_tail");
  register_extent_pair<Atom, 2 * Tile, 2 * Tile, 8 * KStep + 1,
                       kRawAndPackedAB, 15, ShardTag>("tail", "k_tail");
  register_extent_pair<Atom, 2 * Tile + 3, 3 * Tile + 5,
                       8 * KStep + 1, kAllInputModes, 16, ShardTag>(
      "tail", "mnk_tail");
  register_extent_pair<Atom, 1, 1152, 896, kWeightReuseModes, 17, ShardTag>(
      "representative", "decode_projection");
  register_extent_pair<Atom, 128, 1024, 768, kWeightReuseModes, 18,
                       ShardTag>(
      "representative", "batch_projection");
  register_extent_pair<Atom, 256, 256, 256, kAllInputModes, 19, ShardTag>(
      "representative", "square");
  register_extent_pair<Atom, 127, 1025, 769, kWeightReuseModes, 20,
                       ShardTag>(
      "representative", "ragged_projection");
}

template <typename Atom, nint_t Tile, nint_t KStep, int CaseBase = 0,
          int ShardTag = 0>
void register_dtype_probe_extent_pairs() {
  register_extent_pair<Atom, Tile + 3, Tile + 5, 8 * KStep + 1,
                       kAllInputModes, CaseBase + 0, ShardTag>(
      "dtype_coverage", "tail_probe");
  register_extent_pair<Atom, 1, 1024, 1024, kAllInputModes, CaseBase + 1,
                       ShardTag>(
      "dtype_coverage", "decode_probe");
  register_extent_pair<Atom, 64, 256, 256, kAllInputModes, CaseBase + 2,
                       ShardTag>(
      "dtype_coverage", "batch_probe");
}

template <typename Atom, nint_t Tile, nint_t KStep, int ShardTag = 0>
void register_workload_extent_pairs() {
  register_extent_pair<Atom, Tile - 1, Tile - 1, 4 * KStep - 1,
                       kRawAndPackedAB, 0, ShardTag>(
      "shape_coverage", "below_tile");
  register_extent_pair<Atom, 1, 1, 8 * KStep + 1, kRawAndPackedAB, 1,
                       ShardTag>(
      "shape_coverage", "one_by_one");
  register_extent_pair<Atom, 1, 8 * Tile + 3, 16 * KStep + 1,
                       kAllInputModes, 2, ShardTag>(
      "shape_coverage", "single_row_wide");
  register_extent_pair<Atom, 8 * Tile + 3, 1, 16 * KStep + 1,
                       kAllInputModes, 3, ShardTag>(
      "shape_coverage", "single_col_tall");
  register_extent_pair<Atom, 16 * Tile, 16 * Tile, 2 * KStep,
                       kWeightReuseModes, 4, ShardTag>(
      "shape_coverage", "skinny_k");
  register_extent_pair<Atom, 2 * Tile + 1, 2 * Tile + 3,
                       256 * KStep + 1, kWeightReuseModes, 5, ShardTag>(
      "shape_coverage", "long_k");
  register_extent_pair<Atom, 1, 4096, 4096, kWeightReuseModes, 6, ShardTag>(
      "transformer", "decode_qkv_4k");
  register_extent_pair<Atom, 4, 4096, 4096, kWeightReuseModes, 7, ShardTag>(
      "transformer", "decode_qkv_batch4");
  register_extent_pair<Atom, 128, 4096, 4096, kWeightReuseModes, 8,
                       ShardTag>(
      "transformer", "prefill_qkv_4k");
  register_extent_pair<Atom, 1, 11008, 4096, kWeightReuseModes, 9,
                       ShardTag>(
      "transformer", "decode_mlp_up");
  register_extent_pair<Atom, 1, 4096, 11008, kWeightReuseModes, 10,
                       ShardTag>(
      "transformer", "decode_mlp_down");
  register_extent_pair<Atom, 128, 11008, 4096, kWeightReuseModes, 11,
                       ShardTag>(
      "transformer", "prefill_mlp_up");
  register_extent_pair<Atom, 128, 128, 128, kWeightReuseModes, 12,
                       ShardTag>(
      "transformer", "attention_score");
  register_extent_pair<Atom, 3136, 64, 576, kWeightReuseModes, 13,
                       ShardTag>(
      "workload", "conv_1x1_lowering");
  register_extent_pair<Atom, 256, 256, 4096, kWeightReuseModes, 14,
                       ShardTag>(
      "workload", "batched_square_k4k");
  register_extent_pair<Atom, 1024, 64, 4096, kWeightReuseModes, 15,
                       ShardTag>(
      "workload", "rank_reduction");
  register_extent_pair<Atom, 64, 1024, 4096, kWeightReuseModes, 16,
                       ShardTag>(
      "workload", "rank_expansion");
}

template <typename Atom, nint_t Tile, int ShardTag = 0>
void register_dispatch_extent_pairs() {
  register_extent_pair<Atom, 1, 8, 64, kRawAndPackedAB, 0, ShardTag>(
      "dispatch_tiny", "m1_n8_k64");
  register_extent_pair<Atom, 8, 1, 64, kRawAndPackedAB, 1, ShardTag>(
      "dispatch_tiny", "m8_n1_k64");
  register_extent_pair<Atom, 2, 4, 65, kRawAndPackedAB, 2, ShardTag>(
      "dispatch_tiny", "m2_n4_k65");
  register_extent_pair<Atom, 4, 2, 65, kRawAndPackedAB, 3, ShardTag>(
      "dispatch_tiny", "m4_n2_k65");
  register_extent_pair<Atom, 4, 4, 256, kRawAndPackedAB, 4, ShardTag>(
      "dispatch_tiny", "m4_n4_k256");
  register_extent_pair<Atom, 8, 8, 257, kRawAndPackedAB, 5, ShardTag>(
      "dispatch_tiny", "m8_n8_k257");
  register_extent_pair<Atom, 16, 16, 1024, kRawAndPackedAB, 6, ShardTag>(
      "dispatch_tiny", "m16_n16_k1024");
#define VECOPS_REGISTER_GEMV_PAIR(SIDE, LABEL, M, N, K, MODES, CASE_INDEX) \
  register_extent_pair<Atom, M, N, K, MODES, CASE_INDEX, ShardTag>( \
      SIDE, LABEL)
  VECOPS_REGISTER_GEMV_PAIR(
      "dispatch_gemv_row", "n16_k65", 1, 16, 65, kWeightReuseModes, 7);
  VECOPS_REGISTER_GEMV_PAIR(
      "dispatch_gemv_row", "n32_k256", 1, 32, 256, kWeightReuseModes, 8);
  VECOPS_REGISTER_GEMV_PAIR(
      "dispatch_gemv_row", "n64_k257", 1, 64, 257, kWeightReuseModes, 9);
  VECOPS_REGISTER_GEMV_PAIR(
      "dispatch_gemv_row", "n128_k1024", 1, 128, 1024,
      kWeightReuseModes, 10);
  VECOPS_REGISTER_GEMV_PAIR(
      "dispatch_gemv_row", "n256_k1025", 1, 256, 1025,
      kWeightReuseModes, 11);
  VECOPS_REGISTER_GEMV_PAIR(
      "dispatch_gemv_row", "n512_k4096", 1, 512, 4096,
      kWeightReuseModes, 12);
  VECOPS_REGISTER_GEMV_PAIR(
      "dispatch_gemv_row", "n1024_k4097", 1, 1024, 4097,
      kWeightReuseModes, 13);
  VECOPS_REGISTER_GEMV_PAIR(
      "dispatch_gemv_col", "m16_k65", 16, 1, 65, kAllInputModes, 14);
  VECOPS_REGISTER_GEMV_PAIR(
      "dispatch_gemv_col", "m32_k256", 32, 1, 256, kAllInputModes, 15);
  VECOPS_REGISTER_GEMV_PAIR(
      "dispatch_gemv_col", "m64_k257", 64, 1, 257, kAllInputModes, 16);
  VECOPS_REGISTER_GEMV_PAIR(
      "dispatch_gemv_col", "m128_k1024", 128, 1, 1024,
      kAllInputModes, 17);
  VECOPS_REGISTER_GEMV_PAIR(
      "dispatch_gemv_col", "m256_k1025", 256, 1, 1025,
      kAllInputModes, 18);
  VECOPS_REGISTER_GEMV_PAIR(
      "dispatch_gemv_col", "m512_k4096", 512, 1, 4096,
      kAllInputModes, 19);
  VECOPS_REGISTER_GEMV_PAIR(
      "dispatch_gemv_col", "m1024_k4097", 1024, 1, 4097,
      kAllInputModes, 20);
#undef VECOPS_REGISTER_GEMV_PAIR
  register_extent_pair<Atom, Tile + 1, 2 * Tile + 1, 1024,
                       kAllInputModes, 21, ShardTag>(
      "dispatch_tail", "short_m_long_k");
  register_extent_pair<Atom, 2 * Tile + 1, Tile + 1, 1025,
                       kAllInputModes, 22, ShardTag>(
      "dispatch_tail", "short_n_long_k");
}

template <typename Atom, uint32_t Modes = mode_bit(InputMode::Raw),
          int CaseBase = 0, int ShardTag = 0>
void register_skinny_extent_pairs(const char* group) {
#define VECOPS_REGISTER_SKINNY_PAIR(LABEL, M, N, K, CASE_OFFSET) \
  register_extent_pair<Atom, M, N, K, Modes, CaseBase + CASE_OFFSET, \
                       ShardTag>( \
      group, LABEL)
  VECOPS_REGISTER_SKINNY_PAIR("m1_n8_k64", 1, 8, 64, 0);
  VECOPS_REGISTER_SKINNY_PAIR("m8_n1_k64", 8, 1, 64, 1);
  VECOPS_REGISTER_SKINNY_PAIR("m1_n16_k65", 1, 16, 65, 2);
  VECOPS_REGISTER_SKINNY_PAIR("m16_n1_k65", 16, 1, 65, 3);
  VECOPS_REGISTER_SKINNY_PAIR("m1_n64_k257", 1, 64, 257, 4);
  VECOPS_REGISTER_SKINNY_PAIR("m64_n1_k257", 64, 1, 257, 5);
#undef VECOPS_REGISTER_SKINNY_PAIR
}

template <typename Atom, int CaseBase = 0, int ShardTag = 0>
void register_extended_integer_skinny_extent_pairs() {
  register_skinny_extent_pairs<
      Atom, mode_bit(InputMode::Raw), CaseBase, ShardTag>(
      "dispatch_integer_skinny");
  register_extent_pair<Atom, 1, 8, 4097, mode_bit(InputMode::Raw),
                       CaseBase + 6, ShardTag>(
      "dispatch_integer_skinny", "m1_n8_k4097");
  register_extent_pair<Atom, 8, 1, 4097, mode_bit(InputMode::Raw),
                       CaseBase + 7, ShardTag>(
      "dispatch_integer_skinny", "m8_n1_k4097");
}

template <typename Atom, int CaseBase = 0, int ShardTag = 0>
void register_packed_mmla_extent_pairs() {
#define VECOPS_REGISTER_MMLA_PAIR(M, N, K, CASE_OFFSET) \
  register_extent_pair<Atom, M, N, K, mode_bit(InputMode::PackedAB), \
                       CaseBase + CASE_OFFSET, ShardTag>( \
      "dispatch_packed_mmla", "m" #M "_n" #N "_k" #K)
  VECOPS_REGISTER_MMLA_PAIR(2, 2, 65, 0);
  VECOPS_REGISTER_MMLA_PAIR(2, 2, 257, 1);
  VECOPS_REGISTER_MMLA_PAIR(2, 2, 513, 2);
  VECOPS_REGISTER_MMLA_PAIR(2, 2, 1025, 3);
  VECOPS_REGISTER_MMLA_PAIR(2, 4, 65, 4);
  VECOPS_REGISTER_MMLA_PAIR(2, 4, 257, 5);
  VECOPS_REGISTER_MMLA_PAIR(2, 4, 513, 6);
  VECOPS_REGISTER_MMLA_PAIR(2, 4, 1025, 7);
  VECOPS_REGISTER_MMLA_PAIR(4, 2, 65, 8);
  VECOPS_REGISTER_MMLA_PAIR(4, 2, 257, 9);
  VECOPS_REGISTER_MMLA_PAIR(4, 2, 513, 10);
  VECOPS_REGISTER_MMLA_PAIR(4, 2, 1025, 11);
  VECOPS_REGISTER_MMLA_PAIR(2, 8, 65, 12);
  VECOPS_REGISTER_MMLA_PAIR(2, 8, 257, 13);
  VECOPS_REGISTER_MMLA_PAIR(2, 8, 513, 14);
  VECOPS_REGISTER_MMLA_PAIR(2, 8, 1025, 15);
  VECOPS_REGISTER_MMLA_PAIR(8, 2, 65, 16);
  VECOPS_REGISTER_MMLA_PAIR(8, 2, 257, 17);
  VECOPS_REGISTER_MMLA_PAIR(8, 2, 513, 18);
  VECOPS_REGISTER_MMLA_PAIR(8, 2, 1025, 19);
  VECOPS_REGISTER_MMLA_PAIR(4, 4, 65, 20);
  VECOPS_REGISTER_MMLA_PAIR(4, 4, 257, 21);
  VECOPS_REGISTER_MMLA_PAIR(4, 4, 513, 22);
  VECOPS_REGISTER_MMLA_PAIR(4, 4, 1025, 23);
#undef VECOPS_REGISTER_MMLA_PAIR
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
