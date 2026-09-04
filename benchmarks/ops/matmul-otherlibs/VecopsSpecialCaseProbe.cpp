#include "ProviderCommon.h"

#include <cblas.h>

#include <cerrno>
#include <cstring>
#include <iostream>
#include <type_traits>

#include <sys/syscall.h>
#include <unistd.h>

#include "vecops/kernel/Workspace.h"
#include "vecops/matmul/Atom.h"
#include "vecops/ops/Matmul.h"
#include "vecops/tensor/Tensor.h"

namespace vecops::bench::matmul_otherlibs {

using NativeAtom = ::vecops::matmul::AMX_BF16F32;

template <Operation Op>
void run_default_vecops(benchmark::State& state, const Case& c) {
  using namespace ::vecops;
  Buffers buffers(c);
  buffers.prepare_output(Op);
  const meta::Any m{c.n};
  const meta::Any n{c.m};
  const meta::Any k{c.k};
  auto a = tensor::make_tensor(
      buffers.x.data(), tensor::make_layout(tensor::make_shape(m, k)));
  auto b = tensor::make_tensor(
      buffers.weight.data(), tensor::make_layout(tensor::make_shape(n, k)));
  auto out = tensor::make_tensor(
      buffers.output.data(), tensor::make_layout(tensor::make_shape(m, n)));
  auto operation = ops::matmul(ops::MatmulConfig<NativeAtom>{});
  auto prior_tensor = tensor::make_tensor(
      buffers.initial.data(), tensor::make_layout(tensor::make_shape(m, n)));
  auto prior = tensor::input<float>(prior_tensor);
  auto output = tensor::output<float>(out);
  const nint_t workspace_bytes = [&] {
    if constexpr (Op == Operation::Gemm)
      return operation.required_workspace(m, n, k, a, b, out);
    else
      return operation.required_workspace(m, n, k, a, b, prior, output);
  }();
  kernel::Workspace storage(workspace_bytes);
  auto workspace = storage.view();
  const auto run = [&] {
    if constexpr (Op == Operation::Gemm) {
      operation(workspace, m, n, k, a, b, out);
    } else {
      operation(workspace, m, n, k, a, b, prior, output);
    }
  };
  run();
  std::string error;
  if (!verify_samples(c, Op, buffers, &error)) {
    state.SkipWithError(error);
    return;
  }
  buffers.prepare_output(Op);
  for (auto _ : state) {
    run();
    benchmark::DoNotOptimize(buffers.output.data());
    benchmark::ClobberMemory();
  }
  set_counters(state, c, false, true, false);
}

template <Operation Op>
void run_small_n_vector(benchmark::State& state, const Case& c) {
  using namespace ::vecops::kernel::matmul_details::amx;
  Buffers buffers(c);
  buffers.prepare_output(Op);
  const auto run = [&] {
    small_vector_dense_matmul<Op == Operation::GemmAdd>(
        buffers.x.data(), c.k, buffers.weight.data(), c.k,
        buffers.output.data(), c.m, c.n, c.m, c.k,
        Op == Operation::GemmAdd ? buffers.initial.data() : nullptr, c.m);
  };
  run();
  std::string error;
  if (!verify_samples(c, Op, buffers, &error)) {
    state.SkipWithError(error);
    return;
  }
  buffers.prepare_output(Op);
  for (auto _ : state) {
    run();
    benchmark::DoNotOptimize(buffers.output.data());
    benchmark::ClobberMemory();
  }
  set_counters(state, c, false, false, false);
}

template <Operation Op>
void run_openblas_probe(benchmark::State& state, const Case& c) {
  static_assert(sizeof(bfloat16_t) == sizeof(bfloat16));
  Buffers buffers(c);
  buffers.prepare_output(Op);
  const auto run = [&] {
    cblas_sbgemm(
        CblasRowMajor, CblasNoTrans, CblasTrans,
        static_cast<blasint>(c.n), static_cast<blasint>(c.m),
        static_cast<blasint>(c.k), 1.0f,
        reinterpret_cast<const bfloat16*>(buffers.x.data()),
        static_cast<blasint>(c.k),
        reinterpret_cast<const bfloat16*>(buffers.weight.data()),
        static_cast<blasint>(c.k),
        Op == Operation::GemmAdd ? 1.0f : 0.0f,
        buffers.output.data(), static_cast<blasint>(c.m));
  };
  run();
  std::string error;
  if (!verify_samples(c, Op, buffers, &error)) {
    state.SkipWithError(error);
    return;
  }
  buffers.prepare_output(Op);
  for (auto _ : state) {
    run();
    benchmark::DoNotOptimize(buffers.output.data());
    benchmark::ClobberMemory();
  }
  set_counters(state, c, false, true, false);
}

template <Operation Op, typename Run>
void register_probe(
    std::string_view provider, const Case& c,
    std::string_view phase, std::string_view tuning, Run run) {
  const auto name = benchmark_name(
      provider, c, Op, "AllDynamic", phase, "diagnostic", tuning);
  auto* registered = benchmark::RegisterBenchmark(
      name.c_str(), [c, run](benchmark::State& state) { run(state, c); });
  registered->Unit(benchmark::kMicrosecond);
  configure_comparison_benchmark(registered, 0.1, 7);
}

template <Operation Op>
void register_ragged_case(const Case& c) {
  register_probe<Op>(
      "vecops", c, "ragged_sweep", "default",
      run_default_vecops<Op>);
  register_probe<Op>(
      "OpenBLAS", c, "ragged_sweep", "default",
      run_openblas_probe<Op>);
}

void register_cases() {
  for (const Case& c : {AF3Cases[4], AF3Cases[5], AF3Cases[6]}) {
    register_probe<Operation::Gemm>(
        "vecops", c, "special_leaf", "avx512_small_n",
        run_small_n_vector<Operation::Gemm>);
    register_probe<Operation::GemmAdd>(
        "vecops", c, "special_leaf", "avx512_small_n",
        run_small_n_vector<Operation::GemmAdd>);
  }

  constexpr std::array ragged_cases{
      Case{"R00_aligned", Model::General, "ragged_probe", 1024, 128, 768,
           1, 0, 0, 1, GemmOps},
      Case{"R01_m_tail", Model::General, "ragged_probe", 1025, 128, 768,
           1, 0, 0, 1, GemmOps},
      Case{"R02_n_tail", Model::General, "ragged_probe", 1024, 127, 768,
           1, 0, 0, 1, GemmOps},
      Case{"R03_k_tail", Model::General, "ragged_probe", 1024, 128, 769,
           1, 0, 0, 1, GemmOps},
      Case{"R04_all_tail", Model::General, "ragged_probe", 1025, 127, 769,
           1, 0, 0, 1, GemmOps},
  };
  for (const Case& c : ragged_cases) {
    register_ragged_case<Operation::Gemm>(c);
    register_ragged_case<Operation::GemmAdd>(c);
  }
}

bool enable_amx() {
  constexpr long ArchRequestXcompPerm = 0x1023;
  constexpr long XfeatureTileData = 18;
  return syscall(SYS_arch_prctl, ArchRequestXcompPerm, XfeatureTileData) == 0;
}

} // namespace vecops::bench::matmul_otherlibs

int main(int argc, char** argv) {
  using namespace vecops::bench::matmul_otherlibs;
  if (!enable_amx()) {
    std::cerr << "AMX tile state unavailable: " << std::strerror(errno) << '\n';
    return 1;
  }
  openblas_set_num_threads(1);
  register_cases();
  return run_registered_benchmarks(
      argc, argv, "matmul_otherlibs_vecops_special_case_probe");
}
