#include "ProviderCommon.h"

#include <array>
#include <cerrno>
#include <cstring>
#include <iostream>

#if defined(ARCH_X86_FAMILY)
#include <sys/syscall.h>
#include <unistd.h>
#endif

#include "vecops/kernel/Workspace.h"
#include "vecops/matmul/Atom.h"
#include "vecops/matmul/Config.h"
#include "vecops/matmul/Family.h"
#include "vecops/ops/Matmul.h"
#include "vecops/ops/MatmulPack.h"
#include "vecops/tensor/Tensor.h"

namespace vecops::bench::matmul_otherlibs {

#if defined(ARCH_X86_FAMILY)
using ProbeAtom = ::vecops::matmul::AMX_BF16F32;
#else
using ProbeAtom = ::vecops::matmul::SME_BF16F32;
#endif

struct TileProfile {
  nint_t mc;
  nint_t nc;
  nint_t kc;
};

inline constexpr std::array ProbeProfiles{
    TileProfile{128, 512, 128},
    TileProfile{128, 512, 256},
    TileProfile{128, 512, 512},
    TileProfile{128, 1024, 256},
    TileProfile{128, 1024, 512},
    TileProfile{128, 2048, 256},
    TileProfile{128, 512, 1024},
    TileProfile{128, 512, 2048},
    TileProfile{128, 512, 4096},
    TileProfile{128, 1024, 4096},
    TileProfile{128, 2048, 4096},
    TileProfile{128, 512, 8192},
    TileProfile{128, 512, 11008},
    TileProfile{64, 512, 256},
    TileProfile{256, 512, 256},
    TileProfile{384, 768, 256},
    TileProfile{672, 1360, 384},
};

template <bool PreparedB>
void run_tiling_probe(
    benchmark::State& state, const Case& c, TileProfile profile) {
  using namespace ::vecops;
  using Tiles = matmul::CacheTiling<meta::Any, meta::Any, meta::Any>;
  using Tuning = matmul::GenericTiledTuning<
      Tiles, matmul::loop_order::NKM>;
  using Config = ops::MatmulConfig<
      ProbeAtom,
      matmul::family_selection::Require<matmul::kernel_family::GenericTiled>,
      kernel::matmul_policy::Automatic, Tuning>;

  Buffers buffers(c);
  buffers.prepare_output(Operation::Gemm);
  meta::Any internal_m{c.n};
  meta::Any internal_n{c.m};
  meta::Any k{c.k};
  Config config{.generic_tiled = Tuning{.cache_tiling = Tiles{
      meta::Any{profile.mc}, meta::Any{profile.nc}, meta::Any{profile.kc}}}};
  auto operation = ops::matmul(config);

  const auto run = [&](const auto& a, const auto& b, const auto& out) {
    kernel::Workspace storage(operation.required_workspace(
        internal_m, internal_n, k, a, b, out));
    auto workspace = storage.view();
    operation(workspace, internal_m, internal_n, k, a, b, out);
    std::string error;
    if (!verify_samples(c, Operation::Gemm, buffers, &error)) {
      state.SkipWithError(error);
      return;
    }
    buffers.prepare_output(Operation::Gemm);
    for (auto _ : state) {
      operation(workspace, internal_m, internal_n, k, a, b, out);
      benchmark::DoNotOptimize(buffers.output.data());
      benchmark::ClobberMemory();
    }
  };
  if (c.batch == 1) {
    auto a = tensor::make_tensor(
        buffers.x.data(),
        tensor::make_layout(tensor::make_shape(internal_m, k)));
    auto b = tensor::make_tensor(
        buffers.weight.data(),
        tensor::make_layout(tensor::make_shape(internal_n, k)));
    auto out = tensor::make_tensor(
        buffers.output.data(),
        tensor::make_layout(tensor::make_shape(internal_m, internal_n)));
    if constexpr (PreparedB) {
      const auto packed_layout = matmul::packed_layout<
          ProbeAtom, matmul::Operand::B>(b.layout());
      using TB = typename ProbeAtom::TB;
      const nint_t packed_bytes =
          tensor::numel(packed_layout) * static_cast<nint_t>(sizeof(TB));
      kernel::Workspace packed_owner(packed_bytes + 63);
      auto packed_workspace = packed_owner.view();
      auto* packed_data = static_cast<TB*>(
          packed_workspace.allocate(packed_bytes, 64));
      auto packed_b = tensor::make_tensor(packed_data, packed_layout);
      execution::ExecutionSession pack_execution{};
      ops::matmul_pack(ops::MatmulPackConfig<
          ProbeAtom, matmul::Operand::B>{})(pack_execution, b, packed_b);
      run(a, packed_b, out);
    } else {
      run(a, b, out);
    }
  } else {
    if constexpr (PreparedB) {
      state.SkipWithError("prepared-B tiling probe currently requires rank two");
      return;
    }
    const meta::Any batch{c.batch};
    auto a = tensor::make_tensor(
        buffers.x.data(),
        tensor::make_layout(tensor::make_shape(batch, internal_m, k)));
    auto b = tensor::make_tensor(
        buffers.weight.data(),
        tensor::make_layout(tensor::make_shape(batch, internal_n, k)));
    auto out = tensor::make_tensor(
        buffers.output.data(),
        tensor::make_layout(
            tensor::make_shape(batch, internal_m, internal_n)));
    run(a, b, out);
  }
  if (state.skipped()) return;
  set_counters(state, c, c.batch > 1, !PreparedB, PreparedB);
  state.counters["configured_mc"] = benchmark::Counter(profile.mc);
  state.counters["configured_nc"] = benchmark::Counter(profile.nc);
  state.counters["configured_kc"] = benchmark::Counter(profile.kc);
}

template <bool PreparedB = false>
void register_probe_case(const Case& c) {
  for (const auto profile : ProbeProfiles) {
    const auto mc = std::to_string(profile.mc);
    const auto nc = std::to_string(profile.nc);
    const auto kc = std::to_string(profile.kc);
    const auto name = benchmark_name(
        "vecops", c, Operation::Gemm, "TilingProbe", "tiling_probe",
        "native", PreparedB ? "probe_prepared_b" : "probe",
        mc, nc, kc);
    auto* registered = benchmark::RegisterBenchmark(
        name.c_str(), [c, profile](benchmark::State& state) {
          run_tiling_probe<PreparedB>(state, c, profile);
        });
    registered->Unit(benchmark::kMicrosecond);
    configure_comparison_benchmark(registered, 0.05, 5);
  }
}

void register_probe_cases() {
  register_probe_case(CoreCases[3]);  // G04
  register_probe_case(CoreCases[4]);  // G05
  register_probe_case(CoreCases[7]);  // L03
  register_probe_case(CoreCases[9]);  // L05
  register_probe_case(CoreCases[11]); // L07
  register_probe_case<true>(CoreCases[7]);  // L03 prepared B
  register_probe_case<true>(CoreCases[9]);  // L05 prepared B
  register_probe_case<true>(CoreCases[11]); // L07 prepared B
  register_probe_case(AF3Cases[2]);   // A03
  register_probe_case(AF3Cases[6]);   // A07
  register_probe_case(AF3Cases[9]);   // A10
  register_probe_case(AF3Cases[17]);  // A18
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
  register_probe_cases();
  return run_registered_benchmarks(
      argc, argv, "matmul_otherlibs_vecops_tiling_probe");
}
