#include "ProviderCommon.h"

#include <array>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <iostream>

#if defined(ARCH_X86_FAMILY)
#include <sys/syscall.h>
#include <unistd.h>
#endif

#include "vecops/execution/ExecutionSession.h"
#include "vecops/kernel/Workspace.h"
#include "vecops/matmul/Atom.h"
#include "vecops/matmul/Packing.h"
#include "vecops/ops/Matmul.h"
#include "vecops/ops/MatmulPack.h"
#include "vecops/tensor/Tensor.h"

namespace vecops::bench::matmul_otherlibs {

#if defined(ARCH_X86_FAMILY)
using ProbeAtom = ::vecops::matmul::AMX_BF16F32;
#else
using ProbeAtom = ::vecops::matmul::SME_BF16F32;
#endif

enum class ProbePhase : std::uint8_t {
  PackA,
  PackB,
  PreparedBExecute,
  PreparedABCompute,
};

enum class ProbeExtent : std::uint8_t {
  AllDynamic,
  TokenDynamic,
  AllConst,
};

constexpr const char* extent_name(ProbeExtent extent) {
  switch (extent) {
    case ProbeExtent::AllDynamic: return "AllDynamic";
    case ProbeExtent::TokenDynamic: return "TokenDynamic";
    case ProbeExtent::AllConst: return "AllConst";
  }
  return "unknown";
}

constexpr const char* phase_name(ProbePhase phase) {
  switch (phase) {
    case ProbePhase::PackA: return "pack_a";
    case ProbePhase::PackB: return "pack_b";
    case ProbePhase::PreparedBExecute: return "prepared_b_execute";
    case ProbePhase::PreparedABCompute: return "prepared_ab_compute";
  }
  return "unknown";
}

template <ProbePhase Phase, meta::ValueType MExtent,
          meta::ValueType NExtent, meta::ValueType KExtent>
void run_phase_probe(
    benchmark::State& state, const Case& c,
    MExtent m, NExtent n, KExtent k) {
  using namespace ::vecops;
  using namespace ::vecops::meta;
  using namespace ::vecops::tensor;
  static_assert(Phase == ProbePhase::PackA || Phase == ProbePhase::PackB ||
                Phase == ProbePhase::PreparedBExecute ||
                Phase == ProbePhase::PreparedABCompute);
  if (c.batch != 1) {
    state.SkipWithError("phase probe currently requires a rank-two case");
    return;
  }

  Buffers buffers(c);
  buffers.prepare_output(Operation::Gemm);
  auto a = make_tensor(
      buffers.x.data(), make_layout(make_shape(m, k)));
  auto b = make_tensor(
      buffers.weight.data(), make_layout(make_shape(n, k)));
  auto out = make_tensor(
      buffers.output.data(), make_layout(make_shape(m, n)));

  const auto a_layout = matmul::packed_layout<
      ProbeAtom, matmul::Operand::A>(a.layout());
  const auto b_layout = matmul::packed_layout<
      ProbeAtom, matmul::Operand::B>(b.layout());
  using TA = typename ProbeAtom::TA;
  using TB = typename ProbeAtom::TB;
  const nint_t a_bytes = numel(a_layout) * static_cast<nint_t>(sizeof(TA));
  const nint_t b_bytes = numel(b_layout) * static_cast<nint_t>(sizeof(TB));
  kernel::Workspace packed_owner(a_bytes + b_bytes + 128);
  auto packed_workspace = packed_owner.view();
  auto* a_data = static_cast<TA*>(packed_workspace.allocate(a_bytes, 64));
  auto* b_data = static_cast<TB*>(packed_workspace.allocate(b_bytes, 64));
  auto packed_a = make_tensor(a_data, a_layout);
  auto packed_b = make_tensor(b_data, b_layout);
  ExecutionSession pack_execution{};
  const auto a_packer = ops::matmul_pack(
      ops::MatmulPackConfig<ProbeAtom, matmul::Operand::A>{});
  const auto b_packer = ops::matmul_pack(
      ops::MatmulPackConfig<ProbeAtom, matmul::Operand::B>{});

  const auto pack_a = [&] {
    a_packer(pack_execution, a, packed_a);
  };
  const auto pack_b = [&] {
    b_packer(pack_execution, b, packed_b);
  };
  pack_a();
  pack_b();

  auto operation = ops::matmul(ops::MatmulConfig<ProbeAtom>{});
  const auto verify_prepared_ab = [&] {
    kernel::Workspace storage(operation.required_workspace(
        m, n, k, packed_a, packed_b, out));
    auto workspace = storage.view();
    operation(workspace, m, n, k, packed_a, packed_b, out);
  };

  if constexpr (Phase == ProbePhase::PackA || Phase == ProbePhase::PackB) {
    verify_prepared_ab();
    std::string error;
    if (!verify_samples(c, Operation::Gemm, buffers, &error)) {
      state.SkipWithError(error);
      return;
    }
    for (auto _ : state) {
      benchmark::DoNotOptimize(buffers.x.data());
      benchmark::DoNotOptimize(buffers.weight.data());
      if constexpr (Phase == ProbePhase::PackA) pack_a();
      else pack_b();
      benchmark::ClobberMemory();
    }
  } else {
    const auto run_compute = [&](const auto& selected_a) {
      kernel::Workspace storage(operation.required_workspace(
          m, n, k, selected_a, packed_b, out));
      auto workspace = storage.view();
      operation(workspace, m, n, k, selected_a, packed_b, out);
      std::string error;
      if (!verify_samples(c, Operation::Gemm, buffers, &error)) {
        state.SkipWithError(error);
        return;
      }
      buffers.prepare_output(Operation::Gemm);
      for (auto _ : state) {
        operation(workspace, m, n, k, selected_a, packed_b, out);
        benchmark::DoNotOptimize(buffers.output.data());
        benchmark::ClobberMemory();
      }
    };
    if constexpr (Phase == ProbePhase::PreparedABCompute)
      run_compute(packed_a);
    else
      run_compute(a);
  }

  if (state.skipped()) return;
  const bool packs_a = Phase == ProbePhase::PackA;
  constexpr bool PreparedB =
      Phase == ProbePhase::PreparedBExecute ||
      Phase == ProbePhase::PreparedABCompute;
  constexpr bool PackingIncluded =
      Phase == ProbePhase::PackA || Phase == ProbePhase::PackB ||
      Phase == ProbePhase::PreparedBExecute;
  set_counters(state, c, false, PackingIncluded, PreparedB);
  state.counters["packed_bytes"] = benchmark::Counter(
      static_cast<double>(packs_a ? a_bytes : b_bytes));
}

template <ProbePhase Phase, ProbeExtent Extent,
          nint_t M, nint_t N, nint_t K>
void register_phase(const Case& c) {
  const auto name = benchmark_name(
      "vecops", c, Operation::Gemm, extent_name(Extent), phase_name(Phase),
      "diagnostic", "phase_probe");
  auto* registered = benchmark::RegisterBenchmark(
      name.c_str(), [c](benchmark::State& state) {
        using namespace ::vecops::meta;
        if constexpr (Extent == ProbeExtent::AllConst) {
          run_phase_probe<Phase>(state, c, cint<N>, cint<M>, cint<K>);
        } else if constexpr (Extent == ProbeExtent::TokenDynamic) {
          run_phase_probe<Phase>(
              state, c, Any{c.n}, cint<M>, cint<K>);
        } else {
          run_phase_probe<Phase>(
              state, c, Any{c.n}, Any{c.m}, Any{c.k});
        }
      });
  registered->Unit(benchmark::kMicrosecond);
  configure_comparison_benchmark(registered, 0.1, 7);
}

template <ProbeExtent Extent, nint_t M, nint_t N, nint_t K>
void register_extent(const Case& c) {
  register_phase<ProbePhase::PackA, Extent, M, N, K>(c);
  register_phase<ProbePhase::PackB, Extent, M, N, K>(c);
  register_phase<ProbePhase::PreparedBExecute, Extent, M, N, K>(c);
  register_phase<ProbePhase::PreparedABCompute, Extent, M, N, K>(c);
}

template <nint_t M, nint_t N, nint_t K>
void register_general_case(const Case& c) {
  register_extent<ProbeExtent::AllDynamic, M, N, K>(c);
  register_extent<ProbeExtent::AllConst, M, N, K>(c);
}

template <nint_t M, nint_t N, nint_t K>
void register_model_case(const Case& c) {
  register_extent<ProbeExtent::TokenDynamic, M, N, K>(c);
  register_extent<ProbeExtent::AllConst, M, N, K>(c);
}

void register_cases() {
#if defined(ARCH_X86_FAMILY)
  register_general_case<1024, 64, 4096>(CoreCases[4]);
  register_model_case<4096, 128, 4096>(CoreCases[7]);
  register_model_case<11008, 128, 4096>(CoreCases[9]);
  register_model_case<4096, 128, 11008>(CoreCases[11]);
  register_model_case<128, 16384, 128>(AF3Cases[0]);
  register_model_case<1536, 1536, 1536>(AF3Cases[17]);
#else
  register_general_case<64, 3136, 576>(CoreCases[2]);
  register_model_case<4096, 1, 4096>(CoreCases[5]);
  register_model_case<11008, 1, 4096>(CoreCases[8]);
  register_model_case<128, 16384, 128>(AF3Cases[0]);
#endif
}

bool enable_probe_atom() {
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
  if (!enable_probe_atom()) {
    std::cerr << "Native matrix state unavailable: " << std::strerror(errno)
              << '\n';
    return 1;
  }
  register_cases();
  return run_registered_benchmarks(
      argc, argv, "matmul_otherlibs_vecops_phase_probe");
}
