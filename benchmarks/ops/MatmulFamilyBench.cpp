// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
//
//

/**
 * Focused benchmark coverage for explicitly selected matmul kernel families.
 *
 * This target is intentionally separate from MatmulBench: adding family
 * policy instantiations to the production catalog would rebuild every ARM
 * shard and obscure its long-lived code-size baseline.
 */

#include "vecops/platform/Features.h"

#define VECOPS_MATMUL_CATALOG_CASE_SHARDS 1
#include "MatmulBenchCommon.h"

#include <cerrno>
#include <cstring>
#include <iostream>

#if defined(ARCH_X86_FAMILY)
#include <sys/syscall.h>
#include <unistd.h>
#endif

#include "vecops/matmul/Atom.h"
#include "vecops/matmul/Quantization.h"

namespace vecops::bench::matmul {

#if defined(ARCH_X86_FAMILY)

template <>
struct AtomName<::vecops::matmul::AMX_BF16F32> {
  static constexpr const char* value = "AMX_BF16F32";
};

template <typename A, typename B>
using FamilyI8 = ::vecops::matmul::AMX_I8I32<A, B>;

#else

template <>
struct AtomName<::vecops::matmul::SME_BF16F32> {
  static constexpr const char* value = "SME_BF16F32";
};

template <>
struct AtomName<::vecops::matmul::SME_F16F32> {
  static constexpr const char* value = "SME_F16F32";
};

template <>
struct AtomName<::vecops::matmul::SME_F32F32> {
  static constexpr const char* value = "SME_F32F32";
};

#if defined(HAS_SME_F64F64)
template <>
struct AtomName<::vecops::matmul::SME_F64F64> {
  static constexpr const char* value = "SME_F64F64";
};
#endif

template <typename A, typename B>
using FamilyI8 = ::vecops::matmul::SME_I8I32<A, B>;

#endif

using FamilyI8S8S8 = FamilyI8<int8_t, int8_t>;
using FamilyI8S8U8 = FamilyI8<int8_t, uint8_t>;
using FamilyI8U8S8 = FamilyI8<uint8_t, int8_t>;
using FamilyI8U8U8 = FamilyI8<uint8_t, uint8_t>;

template <>
struct AtomName<FamilyI8S8S8> {
  static constexpr const char* value = "I8I32_s8s8";
};
template <>
struct AtomName<FamilyI8S8U8> {
  static constexpr const char* value = "I8I32_s8u8";
};
template <>
struct AtomName<FamilyI8U8S8> {
  static constexpr const char* value = "I8I32_u8s8";
};
template <>
struct AtomName<FamilyI8U8U8> {
  static constexpr const char* value = "I8I32_u8u8";
};

template <typename Atom, InputMode Mode, typename Family,
          nint_t M, nint_t N, nint_t K>
void register_required(const char* group, const char* name) {
  register_required_family_extent_pair<
      Atom, Mode, Family, M, N, K>(group, name);
}

#if !defined(ARCH_X86_FAMILY)
template <typename FamilySelection,
          typename MExtent, typename NExtent, typename KExtent>
void run_runtime_quant(
    benchmark::State& state,
    nint_t logical_m, nint_t logical_n, nint_t logical_k,
    MExtent m, NExtent n, KExtent k) {
  using Atom = ::vecops::matmul::SME_I8I32<uint8_t, int8_t>;
  using Acc = int32_t;
  std::vector<float32_t> a(
      static_cast<std::size_t>(logical_m * logical_k));
  std::vector<int8_t> b(
      static_cast<std::size_t>(logical_n * logical_k));
  std::vector<Acc> correction(static_cast<std::size_t>(logical_n));
  std::vector<float32_t> multipliers(static_cast<std::size_t>(logical_m));
  std::vector<float32_t> row_scales(static_cast<std::size_t>(logical_m));
  std::vector<float32_t> column_scales(static_cast<std::size_t>(logical_n));
  std::vector<float32_t> c(
      static_cast<std::size_t>(logical_m * logical_n));
  int32_t zero_point = 7;
  for (nint_t row = 0; row < logical_m; ++row) {
    const float32_t multiplier = static_cast<float32_t>(4 << (row % 2));
    multipliers[static_cast<std::size_t>(row)] = multiplier;
    row_scales[static_cast<std::size_t>(row)] = 1.0f / multiplier;
    for (nint_t kk = 0; kk < logical_k; ++kk) {
      const int value = static_cast<int>((row * logical_k + kk) % 7) - 3;
      a[static_cast<std::size_t>(row * logical_k + kk)] =
          static_cast<float32_t>(value) / multiplier;
    }
  }
  fill_input(b, 11);
  for (nint_t col = 0; col < logical_n; ++col) {
    Acc sum{};
    for (nint_t kk = 0; kk < logical_k; ++kk)
      sum += b[static_cast<std::size_t>(col * logical_k + kk)];
    correction[static_cast<std::size_t>(col)] = -zero_point * sum;
    column_scales[static_cast<std::size_t>(col)] =
        0.03125f + static_cast<float32_t>(col % 5) * 0.0078125f;
  }

  auto a_layout = make_layout(make_shape(m, k));
  auto b_layout = make_layout(make_shape(n, k));
  auto c_layout = make_layout(make_shape(m, n));
  auto raw_b = make_tensor(b.data(), b_layout);
  auto packed_layout = ::vecops::matmul::packed_layout<
      Atom, ::vecops::matmul::Operand::B>(b_layout);
  const nint_t packed_bytes = numel(packed_layout) * nint_t{sizeof(int8_t)};
  kernel::Workspace packed_storage(packed_bytes + 64);
  auto packed_workspace = packed_storage.view();
  auto* packed_data = static_cast<int8_t*>(
      packed_workspace.allocate(packed_bytes, 64));
  auto packed_b = make_tensor(packed_data, packed_layout);
  ExecutionSession pack_execution{};
  ops::matmul_pack(ops::MatmulPackConfig<
      Atom, ::vecops::matmul::Operand::B>{})(
          pack_execution, raw_b, packed_b);

  auto quantize = ::vecops::matmul::
      make_runtime_per_row_asymmetric_quantize_transform({
          multipliers.data(), &zero_point, 0});
  auto dequantize = ::vecops::matmul::
      make_runtime_per_row_column_dequantize_transform({
          row_scales.data(), column_scales.data(), 0});
  auto correction_layout = make_layout(
      make_shape(m, n), make_strides(cint<0>, cint<1>));
  auto a_input = tensor::input<uint8_t>(
      make_tensor(a.data(), a_layout), quantize);
  auto b_input = tensor::input<int8_t>(packed_b);
  auto c_input = tensor::input<Acc>(
      make_tensor(correction.data(), correction_layout));
  auto c_output = tensor::output<Acc>(
      make_tensor(c.data(), c_layout), dequantize);
  using Config = ops::MatmulConfig<Atom, FamilySelection>;
  auto operation = ops::matmul(Config{});
  kernel::Workspace operation_storage(operation.required_workspace(
      m, n, k, a_input, b_input, c_input, c_output));
  auto operation_workspace = operation_storage.view();
  ExecutionSession execution{operation_workspace};

  operation(execution, m, n, k, a_input, b_input, c_input, c_output);
  for (nint_t row = 0; row < logical_m; ++row) {
    for (nint_t col = 0; col < logical_n; ++col) {
      Acc expected = correction[static_cast<std::size_t>(col)];
      for (nint_t kk = 0; kk < logical_k; ++kk) {
        const auto qa = static_cast<uint8_t>(
            a[static_cast<std::size_t>(row * logical_k + kk)] *
                multipliers[static_cast<std::size_t>(row)] +
            zero_point);
        expected += static_cast<Acc>(qa) *
            static_cast<Acc>(b[static_cast<std::size_t>(
                col * logical_k + kk)]);
      }
      const float32_t reference = static_cast<float32_t>(expected) *
          row_scales[static_cast<std::size_t>(row)] *
          column_scales[static_cast<std::size_t>(col)];
      const float32_t actual = c[static_cast<std::size_t>(
          row * logical_n + col)];
      const float32_t tolerance = 1.0e-5f *
          std::max({1.0f, std::abs(reference), std::abs(actual)});
      if (std::abs(reference - actual) > tolerance) {
        state.SkipWithError("runtime-quant result verification failed");
        return;
      }
    }
  }

  for (auto _ : state) {
    benchmark::DoNotOptimize(a.data());
    operation(execution, m, n, k, a_input, b_input, c_input, c_output);
    benchmark::DoNotOptimize(c.data());
    benchmark::ClobberMemory();
  }
  const int64_t operations = 2 * static_cast<int64_t>(logical_m) *
      logical_n * logical_k;
  state.counters["OP/s"] = benchmark::Counter(
      static_cast<double>(operations),
      benchmark::Counter::kIsIterationInvariantRate);
  state.counters["m"] = benchmark::Counter(double(logical_m));
  state.counters["n"] = benchmark::Counter(double(logical_n));
  state.counters["k"] = benchmark::Counter(double(logical_k));
}

template <typename FamilySelection,
          ExtentMode Extents, nint_t M, nint_t N, nint_t K>
void register_runtime_quant_extent(const char* case_name) {
  const std::string name =
      "MatmulFamily/family_runtime_quant/case:" +
      std::string(case_name) + "/shape:" + std::to_string(M) + "x" +
      std::to_string(N) + "x" + std::to_string(K) +
      "/input:packed_b/extent:" + extent_mode_name<Extents>() +
      "/family:" + family_selection_name<FamilySelection>() +
      "/arch:" + VECOPS_BENCH_ARCH_CODE;
  auto* registered = benchmark::RegisterBenchmark(
      name.c_str(), [](benchmark::State& state) {
        if constexpr (Extents == ExtentMode::Const) {
          run_runtime_quant<FamilySelection>(
              state, M, N, K, cint<M>, cint<N>, cint<K>);
        } else {
          run_runtime_quant<FamilySelection>(
              state, M, N, K, Any{M}, Any{N}, Any{K});
        }
      })->Unit(benchmark::kMicrosecond);
  ::vecops::bench::configure_registered_benchmark(
      registered, 0.02, 3)->ReportAggregatesOnly(true);
}

template <typename FamilySelection, nint_t M, nint_t N, nint_t K>
void register_runtime_quant_pair(const char* case_name) {
  register_runtime_quant_extent<
      FamilySelection, ExtentMode::Dynamic, M, N, K>(case_name);
  register_runtime_quant_extent<
      FamilySelection, ExtentMode::Const, M, N, K>(case_name);
}
#endif

void register_family_benchmarks() {
  using Small = ::vecops::matmul::kernel_family::SmallVector;
#if defined(ARCH_X86_FAMILY)
  using Residual = ::vecops::matmul::kernel_family::ResidualSplit;
  register_extent_pair_mode<
      ::vecops::matmul::AMX_BF16F32, InputMode::Raw,
      1, 1, 256>("historical_auto", "bf16_m1_n1_k256");
  register_extent_pair_mode<
      ::vecops::matmul::AMX_BF16F32, InputMode::Raw,
      1, 16, 1024>("historical_auto", "bf16_m1_n16_k1024");
  register_extent_pair_mode<
      ::vecops::matmul::AMX_BF16F32, InputMode::Raw,
      1, 64, 1024>("historical_auto", "bf16_m1_n64_k1024");
  register_extent_pair_mode<
      FamilyI8S8U8, InputMode::Raw,
      1, 1, 256>("historical_auto", "s8u8_m1_n1_k256");
  register_extent_pair_mode<
      FamilyI8S8U8, InputMode::Raw,
      1, 16, 1024>("historical_auto", "s8u8_m1_n16_k1024");
  register_extent_pair_mode<
      FamilyI8S8U8, InputMode::Raw,
      1, 64, 1024>("historical_auto", "s8u8_m1_n64_k1024");
  register_extent_pair_mode<
      FamilyI8S8U8, InputMode::Raw,
      8, 8, 128>("historical_auto", "s8u8_m8_n8_k128");
  register_required<
      ::vecops::matmul::AMX_BF16F32, InputMode::Raw,
      Small, 1, 1, 256>("family_small_vector", "bf16_m1_n1_k256");
  register_required<
      ::vecops::matmul::AMX_BF16F32, InputMode::Raw,
      Small, 1, 16, 1024>("family_small_vector", "bf16_m1_n16_k1024");
  register_required<
      ::vecops::matmul::AMX_BF16F32, InputMode::Raw,
      Small, 1, 64, 1024>("family_small_vector", "bf16_m1_n64_k1024");
  register_required<
      FamilyI8S8U8, InputMode::Raw,
      Small, 1, 1, 256>("family_small_vector", "s8u8_m1_n1_k256");
  register_required<
      FamilyI8S8U8, InputMode::Raw,
      Small, 1, 16, 1024>("family_small_vector", "s8u8_m1_n16_k1024");
  register_required<
      FamilyI8S8U8, InputMode::Raw,
      Small, 1, 64, 1024>("family_small_vector", "s8u8_m1_n64_k1024");
  register_required<
      FamilyI8S8U8, InputMode::Raw,
      Small, 8, 8, 128>("family_small_vector", "s8u8_m8_n8_k128");
  register_required<
      ::vecops::matmul::AMX_BF16F32, InputMode::PackedAB,
      Residual, 17, 33, 1024>(
          "family_residual_split", "selected_profit_shape");
  register_required<
      ::vecops::matmul::AMX_BF16F32, InputMode::PackedAB,
      Residual, 35, 53, 257>(
          "family_residual_split", "generalized_support_shape");
#else
  using PackedDot = ::vecops::matmul::kernel_family::PackedDot;
  using Automatic = ::vecops::matmul::family_selection::Automatic;
  using RequireRuntimeQuant = ::vecops::matmul::family_selection::Require<
      ::vecops::matmul::kernel_family::RuntimeQuantInt8>;
  register_extent_pair_mode<
      ::vecops::matmul::SME_BF16F32, InputMode::Raw,
      1, 8, 4097>("historical_auto", "bf16_m1_n8_k4097");
  register_extent_pair_mode<
      FamilyI8S8S8, InputMode::Raw,
      64, 256, 256>("historical_auto", "s8s8_m64_n256_k256");
  register_required<
      ::vecops::matmul::SME_BF16F32, InputMode::Raw,
      Small, 1, 8, 4097>("family_small_vector", "bf16_m1_n8_k4097");
  register_required<
      ::vecops::matmul::SME_F16F32, InputMode::Raw,
      Small, 1, 16, 65>("family_small_vector", "fp16_m1_n16_k65");
  register_required<
      ::vecops::matmul::SME_F32F32, InputMode::Raw,
      Small, 1, 64, 257>("family_small_vector", "fp32_m1_n64_k257");
#if defined(HAS_SME_F64F64)
  register_required<
      ::vecops::matmul::SME_F64F64, InputMode::Raw,
      Small, 1, 64, 257>("family_small_vector", "fp64_m1_n64_k257");
#endif
  register_required<
      FamilyI8S8S8, InputMode::Raw,
      Small, 1, 64, 257>("family_small_vector", "s8s8_m1_n64_k257");
  register_required<
      FamilyI8U8U8, InputMode::Raw,
      Small, 64, 1, 257>("family_small_vector", "u8u8_m64_n1_k257");
  register_required<
      FamilyI8S8U8, InputMode::Raw,
      Small, 1, 64, 257>("family_small_vector", "s8u8_m1_n64_k257");
  register_required<
      FamilyI8U8S8, InputMode::Raw,
      Small, 64, 1, 257>("family_small_vector", "u8s8_m64_n1_k257");

  register_runtime_quant_pair<
      Automatic, 1, 256, 256>("decode_m1_n256_k256_auto");
  register_runtime_quant_pair<
      Automatic, 64, 256, 256>("medium_m64_n256_k256_auto");
  register_runtime_quant_pair<
      RequireRuntimeQuant, 1, 256, 256>("decode_m1_n256_k256");
  register_runtime_quant_pair<
      RequireRuntimeQuant, 5, 19, 67>("generalized_m5_n19_k67");
  register_runtime_quant_pair<
      RequireRuntimeQuant, 64, 256, 256>("medium_m64_n256_k256");

  register_required<
      ::vecops::matmul::SME_BF16F32, InputMode::PackedAB,
      PackedDot, 2, 8, 1025>("family_packed_dot", "bf16_primary_long_k");
  register_required<
      ::vecops::matmul::SME_BF16F32, InputMode::PackedAB,
      PackedDot, 4, 4, 257>("family_packed_dot", "bf16_primary_square");
  register_required<
      ::vecops::matmul::SME_BF16F32, InputMode::PackedAB,
      PackedDot, 2, 4, 1025>("family_packed_dot", "bf16_tiny_wide");
  register_required<
      ::vecops::matmul::SME_BF16F32, InputMode::PackedAB,
      PackedDot, 4, 2, 257>("family_packed_dot", "bf16_tiny_tall");
  register_required<
      FamilyI8S8S8, InputMode::PackedAB,
      PackedDot, 2, 8, 513>("family_packed_dot", "s8s8_primary");
  register_required<
      FamilyI8S8S8, InputMode::PackedAB,
      PackedDot, 2, 4, 513>("family_packed_dot", "s8s8_tiny");
  register_required<
      FamilyI8U8U8, InputMode::PackedAB,
      PackedDot, 2, 8, 513>("family_packed_dot", "u8u8_primary");
  register_required<
      FamilyI8U8U8, InputMode::PackedAB,
      PackedDot, 4, 2, 257>("family_packed_dot", "u8u8_tiny");
#endif
}

bool enable_family_benchmark() {
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
  if (!enable_family_benchmark()) {
    std::cerr << "Unable to initialize matmul family benchmark: "
              << std::strerror(errno) << '\n';
    return 1;
  }
  register_family_benchmarks();
  return run_benchmarks(argc, argv, "matmul_family");
}
