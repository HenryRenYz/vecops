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

#ifndef VECOPS_CASE_GROUP
#error "invalid VECOPS_CASE_GROUP"
#endif

namespace vecops::bench::matmul_otherlibs {

#if defined(ARCH_X86_FAMILY)
using NativeAtom = ::vecops::matmul::AMX_BF16F32;
#else
using NativeAtom = ::vecops::matmul::SME_BF16F32;
#endif

enum class ExtentFlavor : std::uint8_t {
  AllDynamic,
  TokenDynamic,
  AllConst,
};

constexpr const char* extent_name(ExtentFlavor flavor) {
  switch (flavor) {
    case ExtentFlavor::AllDynamic: return "AllDynamic";
    case ExtentFlavor::TokenDynamic: return "TokenDynamic";
    case ExtentFlavor::AllConst: return "AllConst";
  }
  return "unknown";
}

#if defined(__GNUC__) || defined(__clang__)
__attribute__((noinline))
#endif
nint_t opaque_runtime_extent(nint_t value) {
  benchmark::DoNotOptimize(value);
  return value;
}

template <Operation Op, typename MExtent, typename NExtent, typename KExtent,
          typename BatchExtent>
void run_vecops_with_extents(
    benchmark::State& state, const Case& c,
    MExtent internal_m, NExtent internal_n, KExtent k,
    BatchExtent batch) {
  using namespace ::vecops::meta;
  using namespace ::vecops::tensor;

  Buffers buffers(c);
  buffers.prepare_output(Op);

  auto operation = ::vecops::ops::matmul(
      ::vecops::ops::MatmulConfig<NativeAtom>{});

  const auto run_rank2 = [&](auto& state_or_null) {
    auto a_layout = make_layout(make_shape(internal_m, k));
    auto b_layout = make_layout(make_shape(internal_n, k));
    auto c_layout = make_layout(make_shape(internal_m, internal_n));
    auto a = make_tensor(buffers.x.data(), a_layout);
    auto b = make_tensor(buffers.weight.data(), b_layout);
    auto out = make_tensor(buffers.output.data(), c_layout);

    if constexpr (Op == Operation::Gemm) {
      ::vecops::kernel::Workspace storage(
          operation.required_workspace(internal_m, internal_n, k, a, b, out));
      auto workspace = storage.view();
      if constexpr (std::same_as<std::remove_cvref_t<decltype(state_or_null)>,
                                 std::nullptr_t>) {
        operation(workspace, internal_m, internal_n, k, a, b, out);
      } else {
        for (auto _ : state_or_null) {
          benchmark::DoNotOptimize(buffers.x.data());
          operation(workspace, internal_m, internal_n, k, a, b, out);
          benchmark::DoNotOptimize(buffers.output.data());
          benchmark::ClobberMemory();
        }
      }
    } else if constexpr (Op == Operation::GemmAdd) {
      // Keep the comparison contract identical to BLAS beta=1: C is both the
      // accumulator input and the destination.  A separate-C-input variant is
      // useful as a diagnostic, but it has a different memory footprint and
      // must not be mixed into the cross-library result table.
      auto c_input = input<float>(out);
      auto c_output = output<float>(out);
      ::vecops::kernel::Workspace storage(operation.required_workspace(
          internal_m, internal_n, k, a, b, c_input, c_output));
      auto workspace = storage.view();
      if constexpr (std::same_as<std::remove_cvref_t<decltype(state_or_null)>,
                                 std::nullptr_t>) {
        operation(
            workspace, internal_m, internal_n, k, a, b, c_input, c_output);
      } else {
        for (auto _ : state_or_null) {
          state_or_null.PauseTiming();
          buffers.prepare_output(Op);
          state_or_null.ResumeTiming();
          benchmark::DoNotOptimize(buffers.x.data());
          operation(
              workspace, internal_m, internal_n, k, a, b, c_input, c_output);
          benchmark::DoNotOptimize(buffers.output.data());
          benchmark::ClobberMemory();
        }
      }
    }
  };

  const auto run_rank3 = [&](auto& state_or_null) {
    auto a_layout = make_layout(make_shape(batch, internal_m, k));
    auto b_layout = make_layout(make_shape(batch, internal_n, k));
    auto c_layout = make_layout(make_shape(batch, internal_m, internal_n));
    auto a = make_tensor(buffers.x.data(), a_layout);
    auto b = make_tensor(buffers.weight.data(), b_layout);
    auto out = make_tensor(buffers.output.data(), c_layout);

    if constexpr (Op == Operation::Gemm) {
      ::vecops::kernel::Workspace storage(
          operation.required_workspace(internal_m, internal_n, k, a, b, out));
      auto workspace = storage.view();
      if constexpr (std::same_as<std::remove_cvref_t<decltype(state_or_null)>,
                                 std::nullptr_t>) {
        operation(workspace, internal_m, internal_n, k, a, b, out);
      } else {
        for (auto _ : state_or_null) {
          benchmark::DoNotOptimize(buffers.x.data());
          operation(workspace, internal_m, internal_n, k, a, b, out);
          benchmark::DoNotOptimize(buffers.output.data());
          benchmark::ClobberMemory();
        }
      }
    } else {
      auto c_input = input<float>(out);
      auto c_output = output<float>(out);
      ::vecops::kernel::Workspace storage(operation.required_workspace(
          internal_m, internal_n, k, a, b, c_input, c_output));
      auto workspace = storage.view();
      if constexpr (std::same_as<std::remove_cvref_t<decltype(state_or_null)>,
                                 std::nullptr_t>) {
        operation(
            workspace, internal_m, internal_n, k, a, b, c_input, c_output);
      } else {
        for (auto _ : state_or_null) {
          state_or_null.PauseTiming();
          buffers.prepare_output(Op);
          state_or_null.ResumeTiming();
          benchmark::DoNotOptimize(buffers.x.data());
          operation(
              workspace, internal_m, internal_n, k, a, b, c_input, c_output);
          benchmark::DoNotOptimize(buffers.output.data());
          benchmark::ClobberMemory();
        }
      }
    }
  };

  std::nullptr_t no_state = nullptr;
  if constexpr (BatchExtent::is_const && BatchExtent::value == 1) {
    run_rank2(no_state);
  } else {
    run_rank3(no_state);
  }
  std::string error;
  if (!verify_samples(c, Op, buffers, &error)) {
    state.SkipWithError(error);
    return;
  }
  buffers.prepare_output(Op);
  if constexpr (BatchExtent::is_const && BatchExtent::value == 1) {
    run_rank2(state);
  } else {
    run_rank3(state);
  }
  set_counters(state, c, c.batch > 1, true, false);
}

template <Operation Op, ExtentFlavor Flavor,
          nint_t M, nint_t N, nint_t K, nint_t Batch>
void run_vecops(benchmark::State& state, const Case& c) {
  using namespace ::vecops::meta;
  if constexpr (Flavor == ExtentFlavor::AllConst) {
    run_vecops_with_extents<Op>(
        state, c, cint<N>, cint<M>, cint<K>, cint<Batch>);
  } else if constexpr (Flavor == ExtentFlavor::TokenDynamic) {
    run_vecops_with_extents<Op>(
        state, c, Any{opaque_runtime_extent(c.n)}, cint<M>, cint<K>,
        cint<Batch>);
  } else {
    run_vecops_with_extents<Op>(
        state, c,
        Any{opaque_runtime_extent(c.n)}, Any{opaque_runtime_extent(c.m)},
        Any{opaque_runtime_extent(c.k)}, cint<Batch>);
  }
}

template <Operation Op, ExtentFlavor Flavor,
          nint_t M, nint_t N, nint_t K, nint_t Batch>
void register_vecops_operation(const Case& c) {
  const auto name = benchmark_name(
      "vecops", c, Op, extent_name(Flavor), "raw_e2e", "native");
  auto* registered = benchmark::RegisterBenchmark(
      name.c_str(), [c](benchmark::State& state) {
        run_vecops<Op, Flavor, M, N, K, Batch>(state, c);
      });
  registered->Unit(benchmark::kMicrosecond);
  configure_comparison_benchmark(registered, 0.1, 7);
}

template <ExtentFlavor Flavor,
          nint_t M, nint_t N, nint_t K, nint_t Batch>
void register_vecops_extent(const Case& c) {
  register_vecops_operation<Operation::Gemm, Flavor, M, N, K, Batch>(c);
  register_vecops_operation<Operation::GemmAdd, Flavor, M, N, K, Batch>(c);
}

template <nint_t M, nint_t N, nint_t K, nint_t Batch = 1>
void register_vecops_case(const Case& c) {
  if (c.model == Model::General) {
    register_vecops_extent<ExtentFlavor::AllDynamic, M, N, K, Batch>(c);
  } else {
    register_vecops_extent<ExtentFlavor::TokenDynamic, M, N, K, Batch>(c);
  }
  register_vecops_extent<ExtentFlavor::AllConst, M, N, K, Batch>(c);
}

void register_vecops_cases() {
#if VECOPS_CASE_GROUP == 0
  register_vecops_case<256, 256, 256>(CoreCases[0]);
  register_vecops_case<1025, 127, 769>(CoreCases[1]);
  register_vecops_case<64, 3136, 576>(CoreCases[2]);
  register_vecops_case<64, 1024, 4096>(CoreCases[3]);
  register_vecops_case<1024, 64, 4096>(CoreCases[4]);
#elif VECOPS_CASE_GROUP == 1
  register_vecops_case<4096, 1, 4096>(CoreCases[5]);
  register_vecops_case<4096, 4, 4096>(CoreCases[6]);
  register_vecops_case<4096, 128, 4096>(CoreCases[7]);
  register_vecops_case<11008, 1, 4096>(CoreCases[8]);
  register_vecops_case<11008, 128, 4096>(CoreCases[9]);
  register_vecops_case<4096, 1, 11008>(CoreCases[10]);
  register_vecops_case<4096, 128, 11008>(CoreCases[11]);
  register_vecops_case<128, 128, 128>(CoreCases[12]);
#elif VECOPS_CASE_GROUP == 2
  register_vecops_case<128, 16384, 128>(AF3Cases[0]);
  register_vecops_case<128, 65536, 128>(AF3Cases[1]);
  register_vecops_case<128, 196608, 128>(AF3Cases[2]);
  register_vecops_case<128, 65536, 128>(AF3Cases[3]);
  register_vecops_case<4, 16384, 128>(AF3Cases[4]);
  register_vecops_case<4, 262144, 128>(AF3Cases[5]);
  register_vecops_case<4, 2359296, 128>(AF3Cases[6]);
#elif VECOPS_CASE_GROUP == 3
  register_vecops_case<512, 512, 32, 4>(AF3Cases[8]);
  register_vecops_case<1536, 1536, 32, 4>(AF3Cases[9]);
  register_vecops_case<2048, 2048, 32, 4>(AF3Cases[10]);
#elif VECOPS_CASE_GROUP == 4
  register_vecops_case<128, 128, 128, 128>(AF3Cases[15]);
  register_vecops_case<512, 512, 512, 128>(AF3Cases[16]);
  register_vecops_case<1536, 1536, 1536, 1>(AF3Cases[17]);
#elif VECOPS_CASE_GROUP == 5
  // A08 instantiates the extra shallow-K FastPacked SME plan.  Keeping it in
  // its own TU bounds optimizer memory without changing the case catalog or
  // result schema.
  register_vecops_case<128, 128, 32, 4>(AF3Cases[7]);
#elif VECOPS_CASE_GROUP == 6
  register_vecops_case<32, 128, 128, 4>(AF3Cases[11]);
  register_vecops_case<32, 512, 512, 4>(AF3Cases[12]);
  register_vecops_case<32, 1536, 1536, 4>(AF3Cases[13]);
  register_vecops_case<32, 2048, 2048, 4>(AF3Cases[14]);
#else
#error "invalid VECOPS_CASE_GROUP"
#endif
}

bool enable_native_atom() {
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
  if (!enable_native_atom()) {
    std::cerr << "Unable to initialize native matrix state: "
              << std::strerror(errno) << '\n';
    return 1;
  }
  register_vecops_cases();
#if VECOPS_CASE_GROUP == 0
  constexpr std::string_view Stem = "matmul_otherlibs_vecops_general";
#elif VECOPS_CASE_GROUP == 1
  constexpr std::string_view Stem = "matmul_otherlibs_vecops_llm";
#elif VECOPS_CASE_GROUP == 2
  constexpr std::string_view Stem = "matmul_otherlibs_vecops_af3_projection";
#elif VECOPS_CASE_GROUP == 3
  constexpr std::string_view Stem = "matmul_otherlibs_vecops_af3_attention";
#else
  constexpr std::string_view Stem = "matmul_otherlibs_vecops_af3_triangle";
#endif
  return run_registered_benchmarks(argc, argv, Stem);
}
