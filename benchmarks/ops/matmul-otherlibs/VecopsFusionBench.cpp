// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
#include "ProviderCommon.h"

#include <cerrno>
#include <cstring>
#include <iostream>
#include <string_view>

#if defined(ARCH_X86_FAMILY)
#include <sys/syscall.h>
#include <unistd.h>
#endif

#include "vecops/kernel/Workspace.h"
#include "vecops/matmul/Atom.h"
#include "vecops/ops/Matmul.h"
#include "vecops/tensor/DataAccess.h"
#include "vecops/tensor/Tensor.h"
#include "vecops/vec/Vec.h"

#ifndef VECOPS_FUSION_OPERATION
#error "VECOPS_FUSION_OPERATION must select bias(0), bias_relu(1), or bias_silu(2)"
#endif

namespace vecops::bench::matmul_otherlibs {

#if defined(ARCH_X86_FAMILY)
using NativeAtom = ::vecops::matmul::AMX_BF16F32;
#else
using NativeAtom = ::vecops::matmul::SME_BF16F32;
#endif

#if VECOPS_FUSION_OPERATION == 0
inline constexpr Operation FusionOperation = Operation::Bias;
#elif VECOPS_FUSION_OPERATION == 1
inline constexpr Operation FusionOperation = Operation::BiasRelu;
#elif VECOPS_FUSION_OPERATION == 2
inline constexpr Operation FusionOperation = Operation::BiasSilu;
#else
#error "invalid VECOPS_FUSION_OPERATION"
#endif

enum class FusionExtent : std::uint8_t { TokenDynamic, AllConst };

constexpr const char* fusion_extent_name(FusionExtent extent) {
  return extent == FusionExtent::TokenDynamic ? "TokenDynamic" : "AllConst";
}

#if defined(__GNUC__) || defined(__clang__)
__attribute__((noinline))
#endif
nint_t opaque_fusion_extent(nint_t value) {
  benchmark::DoNotOptimize(value);
  return value;
}

template <Operation Op, typename Tensor>
auto make_fused_output(const Tensor& tensor) {
  using namespace ::vecops;
  if constexpr (Op == Operation::Bias) {
    return tensor::output<float>(tensor);
  } else if constexpr (Op == Operation::BiasRelu) {
    auto relu = tensor::make_elementwise_vec_transform<float, float>(
        [](auto tag, auto value) VECOPS_KERNEL_LAMBDA {
          return vec::max(tag, value, vec::zeros(tag));
        });
    return tensor::output<float>(tensor, relu);
  } else {
    auto silu = tensor::make_elementwise_vec_transform<float, float>(
        [](auto tag, auto value) VECOPS_KERNEL_LAMBDA {
          const auto one = vec::fill(tag, 1.0f);
          const auto sigmoid = vec::div(
              tag, one,
              vec::add(tag, one, vec::exp(tag, vec::neg(tag, value))));
          return vec::mul(tag, value, sigmoid);
        });
    return tensor::output<float>(tensor, silu);
  }
}

template <Operation Op, typename MExtent, typename NExtent, typename KExtent>
void run_fusion_with_extents(
    benchmark::State& state, const Case& c,
    MExtent internal_m, NExtent internal_n, KExtent k) {
  using namespace ::vecops::meta;
  using namespace ::vecops::tensor;

  Buffers buffers(c);
  buffers.prepare_output(Op);
  auto a_layout = make_layout(make_shape(internal_m, k));
  auto b_layout = make_layout(make_shape(internal_n, k));
  auto c_layout = make_layout(make_shape(internal_m, internal_n));
  auto bias_layout = make_layout(
      make_shape(internal_m, internal_n),
      make_strides(cint<0>, cint<1>));
  auto a = make_tensor(buffers.x.data(), a_layout);
  auto b = make_tensor(buffers.weight.data(), b_layout);
  auto out = make_tensor(buffers.output.data(), c_layout);
  auto bias = input<float>(make_tensor(buffers.bias.data(), bias_layout));
  auto c_output = make_fused_output<Op>(out);
  auto operation = ::vecops::ops::matmul(
      ::vecops::ops::MatmulConfig<NativeAtom>{});
  ::vecops::kernel::Workspace storage(operation.required_workspace(
      internal_m, internal_n, k, a, b, bias, c_output));
  auto workspace = storage.view();

  operation(workspace, internal_m, internal_n, k, a, b, bias, c_output);
  std::string error;
  if (!verify_samples(c, Op, buffers, &error)) {
    state.SkipWithError(error);
    return;
  }
  buffers.prepare_output(Op);
  for (auto _ : state) {
    benchmark::DoNotOptimize(buffers.x.data());
    operation(workspace, internal_m, internal_n, k, a, b, bias, c_output);
    benchmark::DoNotOptimize(buffers.output.data());
    benchmark::ClobberMemory();
  }
  set_counters(state, c, false, true, false);
}

template <FusionExtent Extent, nint_t M, nint_t N, nint_t K>
void run_fusion(benchmark::State& state, const Case& c) {
  using namespace ::vecops::meta;
  if constexpr (Extent == FusionExtent::AllConst) {
    run_fusion_with_extents<FusionOperation>(
        state, c, cint<N>, cint<M>, cint<K>);
  } else {
    run_fusion_with_extents<FusionOperation>(
        state, c, Any{opaque_fusion_extent(c.n)}, cint<M>, cint<K>);
  }
}

template <FusionExtent Extent, nint_t M, nint_t N, nint_t K>
void register_fusion_extent(const Case& c) {
  if ((c.operations & op_bit(FusionOperation)) == 0) return;
  const auto name = benchmark_name(
      "vecops", c, FusionOperation, fusion_extent_name(Extent),
      "raw_e2e", "native_fused");
  auto* registered = benchmark::RegisterBenchmark(
      name.c_str(), [c](benchmark::State& state) {
        run_fusion<Extent, M, N, K>(state, c);
      });
  registered->Unit(benchmark::kMicrosecond);
  configure_comparison_benchmark(registered, 0.1, 7);
}

template <nint_t M, nint_t N, nint_t K>
void register_fusion_case(const Case& c) {
  register_fusion_extent<FusionExtent::TokenDynamic, M, N, K>(c);
  register_fusion_extent<FusionExtent::AllConst, M, N, K>(c);
}

void register_fusion_cases() {
  register_fusion_case<4096, 1, 4096>(CoreCases[5]);
  register_fusion_case<4096, 4, 4096>(CoreCases[6]);
  register_fusion_case<4096, 128, 4096>(CoreCases[7]);
  register_fusion_case<11008, 1, 4096>(CoreCases[8]);
  register_fusion_case<11008, 128, 4096>(CoreCases[9]);
  register_fusion_case<4096, 1, 11008>(CoreCases[10]);
  register_fusion_case<4096, 128, 11008>(CoreCases[11]);
  register_fusion_case<128, 16384, 128>(AF3Cases[0]);
  register_fusion_case<128, 65536, 128>(AF3Cases[1]);
  register_fusion_case<128, 196608, 128>(AF3Cases[2]);
  register_fusion_case<128, 65536, 128>(AF3Cases[3]);
}

bool enable_fusion_native_atom() {
#if defined(ARCH_X86_FAMILY)
  constexpr long ArchRequestXcompPerm = 0x1023;
  constexpr long XfeatureTileData = 18;
  return syscall(SYS_arch_prctl, ArchRequestXcompPerm, XfeatureTileData) == 0;
#else
  return true;
#endif
}

} // namespace vecops::bench::matmul_otherlibs

int main(int argc, char** argv) {
  using namespace vecops::bench::matmul_otherlibs;
  if (!enable_fusion_native_atom()) {
    std::cerr << "Native matrix state unavailable: " << std::strerror(errno)
              << '\n';
    return 1;
  }
  register_fusion_cases();
#if VECOPS_FUSION_OPERATION == 0
  constexpr std::string_view Stem = "matmul_otherlibs_vecops_bias";
#elif VECOPS_FUSION_OPERATION == 1
  constexpr std::string_view Stem = "matmul_otherlibs_vecops_bias_relu";
#else
  constexpr std::string_view Stem = "matmul_otherlibs_vecops_bias_silu";
#endif
  return run_registered_benchmarks(argc, argv, Stem);
}
