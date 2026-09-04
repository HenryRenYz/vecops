#include "ProviderCommon.h"

#include <array>
#include <cstdint>

#include "vecops/execution/ExecutionSession.h"
#include "vecops/execution/details/arm/Resources.h"
#include "vecops/kernel/Workspace.h"
#include "vecops/matmul/Atom.h"
#include "vecops/matmul/Packing.h"
#include "vecops/ops/Matmul.h"
#include "vecops/ops/MatmulPack.h"
#include "vecops/tensor/Tensor.h"

namespace vecops::bench::matmul_otherlibs {

using ProbeAtom = ::vecops::matmul::SME_BF16F32;

enum class BatchPhase : std::uint8_t {
  Rank3Raw,
  Rank2RawLoop,
  Rank2PreparedBLoop,
  Rank2PreparedABLoop,
  Rank3RawOuterRegion,
  Rank2RawLoopOuterRegion,
  Rank2PreparedBLoopOuterRegion,
  Rank2PreparedABLoopOuterRegion,
};

constexpr const char* phase_name(BatchPhase phase) {
  switch (phase) {
    case BatchPhase::Rank3Raw: return "rank3_raw";
    case BatchPhase::Rank2RawLoop: return "rank2_raw_loop";
    case BatchPhase::Rank2PreparedBLoop: return "rank2_prepared_b_loop";
    case BatchPhase::Rank2PreparedABLoop: return "rank2_prepared_ab_loop";
    case BatchPhase::Rank3RawOuterRegion:
      return "rank3_raw_outer_region";
    case BatchPhase::Rank2RawLoopOuterRegion:
      return "rank2_raw_loop_outer_region";
    case BatchPhase::Rank2PreparedBLoopOuterRegion:
      return "rank2_prepared_b_loop_outer_region";
    case BatchPhase::Rank2PreparedABLoopOuterRegion:
      return "rank2_prepared_ab_loop_outer_region";
  }
  return "unknown";
}

template <Operation Op, bool OuterRegion>
void run_rank3_raw(benchmark::State& state, const Case& c) {
  using namespace ::vecops;
  using namespace ::vecops::meta;
  using namespace ::vecops::tensor;
  Buffers buffers(c);
  buffers.prepare_output(Op);
  const Any m{c.n};
  constexpr auto n = cint<128>;
  constexpr auto k = cint<32>;
  constexpr auto batch = cint<4>;
  auto a = make_tensor(
      buffers.x.data(), make_layout(make_shape(batch, m, k)));
  auto b = make_tensor(
      buffers.weight.data(), make_layout(make_shape(batch, n, k)));
  auto out = make_tensor(
      buffers.output.data(), make_layout(make_shape(batch, m, n)));
  auto prior_tensor = make_tensor(
      buffers.initial.data(), make_layout(make_shape(batch, m, n)));
  auto prior = input<float>(prior_tensor);
  auto output = tensor::output<float>(out);
  auto operation = ops::matmul(ops::MatmulConfig<ProbeAtom>{});
  const nint_t workspace_bytes = [&] {
    if constexpr (Op == Operation::Gemm)
      return operation.required_workspace(m, n, k, a, b, out);
    else
      return operation.required_workspace(m, n, k, a, b, prior, output);
  }();
  kernel::Workspace storage(workspace_bytes);
  auto workspace = storage.view();
  ExecutionSession execution{workspace};
  const auto run_one = [&](auto& scope) VECOPS_INLINE_LAMBDA_NOEXCEPT {
    if constexpr (Op == Operation::Gemm)
      operation(scope, m, n, k, a, b, out);
    else
      operation(scope, m, n, k, a, b, prior, output);
  };
  const auto run = [&] {
    if constexpr (OuterRegion) {
      execution.with_resources(
          execution::details::arm::StreamingZARegion{},
          [&](auto& active) VECOPS_INLINE_LAMBDA_NOEXCEPT {
            run_one(active);
          });
    } else {
      run_one(execution);
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
  set_counters(state, c, true, true, false);
}

template <Operation Op, BatchPhase Phase>
void run_rank2_loop(benchmark::State& state, const Case& c) {
  using namespace ::vecops;
  using namespace ::vecops::meta;
  using namespace ::vecops::tensor;
  constexpr bool PackA =
      Phase == BatchPhase::Rank2PreparedABLoop ||
      Phase == BatchPhase::Rank2PreparedABLoopOuterRegion;
  constexpr bool PackB =
      Phase == BatchPhase::Rank2PreparedBLoop ||
      Phase == BatchPhase::Rank2PreparedBLoopOuterRegion || PackA;
  constexpr bool OuterRegion =
      Phase == BatchPhase::Rank2RawLoopOuterRegion ||
      Phase == BatchPhase::Rank2PreparedBLoopOuterRegion ||
      Phase == BatchPhase::Rank2PreparedABLoopOuterRegion;
  Buffers buffers(c);
  buffers.prepare_output(Op);
  const Any m{c.n};
  constexpr auto n = cint<128>;
  constexpr auto k = cint<32>;
  const auto a_layout = make_layout(make_shape(m, k));
  const auto b_layout = make_layout(make_shape(n, k));
  const auto c_layout = make_layout(make_shape(m, n));
  const auto packed_a_layout =
      matmul::packed_layout<ProbeAtom, matmul::Operand::A>(a_layout);
  const auto packed_b_layout =
      matmul::packed_layout<ProbeAtom, matmul::Operand::B>(b_layout);
  using TA = typename ProbeAtom::TA;
  using TB = typename ProbeAtom::TB;
  const nint_t a_bytes = numel(packed_a_layout) * sizeof(TA);
  const nint_t b_bytes = numel(packed_b_layout) * sizeof(TB);
  kernel::Workspace packed_owner(
      c.batch * (a_bytes + b_bytes + 128));
  auto packed_workspace = packed_owner.view();
  std::array<TA*, 4> packed_a_data{};
  std::array<TB*, 4> packed_b_data{};
  ExecutionSession pack_execution{};
  const auto a_packer = ops::matmul_pack(
      ops::MatmulPackConfig<ProbeAtom, matmul::Operand::A>{});
  const auto b_packer = ops::matmul_pack(
      ops::MatmulPackConfig<ProbeAtom, matmul::Operand::B>{});
  for (nint_t batch_index = 0; batch_index < c.batch; ++batch_index) {
    auto raw_a = make_tensor(
        buffers.x.data() + checked_elements(batch_index, c.n, c.k),
        a_layout);
    auto raw_b = make_tensor(
        buffers.weight.data() + checked_elements(batch_index, c.m, c.k),
        b_layout);
    if constexpr (PackA) {
      packed_a_data[static_cast<std::size_t>(batch_index)] =
          static_cast<TA*>(packed_workspace.allocate(a_bytes, 64));
      auto packed_a = make_tensor(
          packed_a_data[static_cast<std::size_t>(batch_index)],
          packed_a_layout);
      a_packer(pack_execution, raw_a, packed_a);
    }
    if constexpr (PackB) {
      packed_b_data[static_cast<std::size_t>(batch_index)] =
          static_cast<TB*>(packed_workspace.allocate(b_bytes, 64));
      auto packed_b = make_tensor(
          packed_b_data[static_cast<std::size_t>(batch_index)],
          packed_b_layout);
      b_packer(pack_execution, raw_b, packed_b);
    }
  }

  auto operation = ops::matmul(ops::MatmulConfig<ProbeAtom>{});
  auto raw_a0 = make_tensor(buffers.x.data(), a_layout);
  auto raw_b0 = make_tensor(buffers.weight.data(), b_layout);
  auto packed_a0 = make_tensor(packed_a_data[0], packed_a_layout);
  auto packed_b0 = make_tensor(packed_b_data[0], packed_b_layout);
  auto out0 = make_tensor(buffers.output.data(), c_layout);
  auto prior_tensor0 = make_tensor(buffers.initial.data(), c_layout);
  auto prior0 = input<float>(prior_tensor0);
  auto output0 = tensor::output<float>(out0);
  const auto& selected_a0 = [&]() -> const auto& {
    if constexpr (PackA) return packed_a0;
    else return raw_a0;
  }();
  const auto& selected_b0 = [&]() -> const auto& {
    if constexpr (PackB) return packed_b0;
    else return raw_b0;
  }();
  const nint_t workspace_bytes = [&] {
    if constexpr (Op == Operation::Gemm) {
      return operation.required_workspace(
          m, n, k, selected_a0, selected_b0, out0);
    } else {
      return operation.required_workspace(
          m, n, k, selected_a0, selected_b0, prior0, output0);
    }
  }();
  kernel::Workspace compute_owner(workspace_bytes);
  auto workspace = compute_owner.view();
  ExecutionSession execution{workspace};
  const auto run_loop = [&](auto& scope) VECOPS_INLINE_LAMBDA_NOEXCEPT {
    for (nint_t batch_index = 0; batch_index < c.batch; ++batch_index) {
      auto raw_a = make_tensor(
          buffers.x.data() + checked_elements(batch_index, c.n, c.k),
          a_layout);
      auto raw_b = make_tensor(
          buffers.weight.data() + checked_elements(batch_index, c.m, c.k),
          b_layout);
      auto packed_a = make_tensor(
          packed_a_data[static_cast<std::size_t>(batch_index)],
          packed_a_layout);
      auto packed_b = make_tensor(
          packed_b_data[static_cast<std::size_t>(batch_index)],
          packed_b_layout);
      auto out = make_tensor(
          buffers.output.data() + checked_elements(batch_index, c.n, c.m),
          c_layout);
      auto prior_tensor = make_tensor(
          buffers.initial.data() + checked_elements(batch_index, c.n, c.m),
          c_layout);
      auto prior = input<float>(prior_tensor);
      auto output = tensor::output<float>(out);
      const auto& selected_a = [&]() -> const auto& {
        if constexpr (PackA) return packed_a;
        else return raw_a;
      }();
      const auto& selected_b = [&]() -> const auto& {
        if constexpr (PackB) return packed_b;
        else return raw_b;
      }();
      if constexpr (Op == Operation::Gemm)
        operation(scope, m, n, k, selected_a, selected_b, out);
      else
        operation(
            scope, m, n, k, selected_a, selected_b, prior, output);
    }
  };
  const auto run = [&] {
    if constexpr (OuterRegion) {
      execution.with_resources(
          execution::details::arm::StreamingZARegion{},
          [&](auto& active) VECOPS_INLINE_LAMBDA_NOEXCEPT {
            run_loop(active);
          });
    } else {
      run_loop(execution);
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
  set_counters(state, c, false, !PackA, PackB);
}

template <Operation Op, BatchPhase Phase>
void register_phase(const Case& c) {
  const auto name = benchmark_name(
      "vecops", c, Op, "TokenDynamic", phase_name(Phase),
      "diagnostic", "batch_phase_probe");
  auto* registered = benchmark::RegisterBenchmark(
      name.c_str(), [c](benchmark::State& state) {
        if constexpr (Phase == BatchPhase::Rank3Raw ||
                      Phase == BatchPhase::Rank3RawOuterRegion)
          run_rank3_raw<
              Op, Phase == BatchPhase::Rank3RawOuterRegion>(state, c);
        else
          run_rank2_loop<Op, Phase>(state, c);
      });
  registered->Unit(benchmark::kMicrosecond);
  configure_comparison_benchmark(registered, 0.1, 7);
}

void register_cases() {
  const Case& c = AF3Cases[7];
  register_phase<Operation::Gemm, BatchPhase::Rank3Raw>(c);
  register_phase<Operation::Gemm, BatchPhase::Rank2RawLoop>(c);
  register_phase<Operation::Gemm, BatchPhase::Rank2PreparedBLoop>(c);
  register_phase<Operation::Gemm, BatchPhase::Rank2PreparedABLoop>(c);
  register_phase<Operation::Gemm, BatchPhase::Rank3RawOuterRegion>(c);
  register_phase<Operation::Gemm, BatchPhase::Rank2RawLoopOuterRegion>(c);
  register_phase<
      Operation::Gemm, BatchPhase::Rank2PreparedBLoopOuterRegion>(c);
  register_phase<
      Operation::Gemm, BatchPhase::Rank2PreparedABLoopOuterRegion>(c);
  register_phase<Operation::GemmAdd, BatchPhase::Rank3Raw>(c);
  register_phase<Operation::GemmAdd, BatchPhase::Rank2RawLoop>(c);
  register_phase<Operation::GemmAdd, BatchPhase::Rank2PreparedBLoop>(c);
  register_phase<Operation::GemmAdd, BatchPhase::Rank2PreparedABLoop>(c);
  register_phase<Operation::GemmAdd, BatchPhase::Rank3RawOuterRegion>(c);
  register_phase<Operation::GemmAdd, BatchPhase::Rank2RawLoopOuterRegion>(c);
  register_phase<
      Operation::GemmAdd, BatchPhase::Rank2PreparedBLoopOuterRegion>(c);
  register_phase<
      Operation::GemmAdd, BatchPhase::Rank2PreparedABLoopOuterRegion>(c);
}

} // namespace vecops::bench::matmul_otherlibs

int main(int argc, char** argv) {
  using namespace vecops::bench::matmul_otherlibs;
  register_cases();
  return run_registered_benchmarks(
      argc, argv, "matmul_otherlibs_vecops_batch_phase_probe");
}
