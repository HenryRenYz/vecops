//
// Copyright (c) vecops contributors.
//

#ifndef VECOPS_MATMUL_DETAILS_TILER_H
#define VECOPS_MATMUL_DETAILS_TILER_H

#include <type_traits>
#include <utility>

#include "vecops/matmul/Tiling.h"
#include "vecops/matmul/details/tiled/Accumulator.h"
#include "vecops/matmul/details/planning/FamilySelector.h"
#include "vecops/matmul/details/kernel/Kernel.h"
#include "vecops/matmul/details/tiled/LoopNest.h"
#include "vecops/matmul/details/tiled/OperandController.h"
#include "vecops/matmul/details/tiled/PolicyTraits.h"
#include "vecops/matmul/details/tiled/ProblemMapper.h"
#include "vecops/matmul/details/tiled/Scheduler.h"
#include "vecops/tensor/DataAccess.h"

namespace vecops::matmul::details {

template <tensor::InputSpecLike Spec,
          meta::ValueType M, meta::ValueType N>
VECOPS_INLINE auto narrow_c_input(
    const Spec& spec, nint_t m_origin, M m,
    nint_t n_origin, N n) {
  auto rows = tensor::narrow_view<0>(spec, m_origin, m);
  return tensor::narrow_view<1>(rows, n_origin, n);
}

template <tensor::OutputSpecLike Spec,
          meta::ValueType M, meta::ValueType N>
VECOPS_INLINE auto narrow_c_output(
    const Spec& spec, nint_t m_origin, M m,
    nint_t n_origin, N n) {
  auto rows = tensor::narrow_view<0>(spec, m_origin, m);
  return tensor::narrow_view<1>(rows, n_origin, n);
}

template <typename Config, typename Implementation,
          execution::ExecutionScope Scope,
          meta::ValueType M, meta::ValueType N, meta::ValueType K,
          tensor::InputSpecLike ASpec, tensor::InputSpecLike BSpec,
          tensor::InputSpecLike CInputSpec,
          tensor::OutputSpecLike COutputSpec>
VECOPS_INLINE void run_tiled_rank2(
    Scope& scope, const Config& config, M m, N n, K k,
    const ASpec& a, const BSpec& b,
    const CInputSpec& c_input, const COutputSpec& c_output) {
  using AtomT = typename Config::Atom;
  using Order = resolved_loop_order_t<Config>;
  using Tuning = typename Config::GenericTuning;
  using APacking = typename Tuning::APacking;
  using BPacking = typename Tuning::BPacking;
  using AController = OperandController<
      Operand::A, APacking, Order, AtomT>;
  using BController = OperandController<
      Operand::B, BPacking, Order, AtomT>;
  static_assert(ASpec::InputTensor::Ndim == 2 &&
                BSpec::InputTensor::Ndim == 2 &&
                CInputSpec::InputTensor::Ndim == 2 &&
                COutputSpec::OutputTensor::Ndim == 2,
                "generic matmul Tiler currently consumes rank-two leaves");

  const auto problem = ProblemMapper::map(m, n, k);

  auto tiling = resolve_cache_tiling(config);
  constexpr nint_t KR = std::remove_cvref_t<decltype(AtomT::K_R)>::value;
  VECOPS_ASSERT(static_cast<nint_t>(tiling.kc) % KR == 0,
                "KC must be a multiple of the Atom K step");
  auto& workspace = scope.workspace_view();
  const auto operation_mark = workspace.mark();
  void* scratch = nullptr;
  const nint_t scratch_bytes =
      kernel::matmul_implementation::scratch_bytes<Implementation>();
  if (scratch_bytes != 0) scratch = workspace.allocate(scratch_bytes, 64);

  const nint_t logical_m = static_cast<nint_t>(problem.m);
  const nint_t logical_n = static_cast<nint_t>(problem.n);
  const nint_t logical_k = static_cast<nint_t>(problem.k);
  constexpr bool GeneratesKLoop = LoopNest<
      Order, decltype(tiling)>::template generates_loop<Axis::K, M, N, K>;
  const bool split_k = GeneratesKLoop &&
      logical_k > static_cast<nint_t>(tiling.kc);
  using Acc = typename AtomT::TAcc;
  constexpr bool OutputAcc = uses_output_accumulator_v<Config, COutputSpec>;
  const nint_t acc_m = axis_precedes_k_v<Axis::M, Order>
      ? std::min(logical_m, static_cast<nint_t>(tiling.mc))
      : logical_m;
  const nint_t acc_n = axis_precedes_k_v<Axis::N, Order>
      ? std::min(logical_n, static_cast<nint_t>(tiling.nc))
      : logical_n;
  Acc* acc_data = nullptr;
  if (split_k && !OutputAcc) {
    acc_data = static_cast<Acc*>(workspace.allocate(
        acc_m * acc_n * static_cast<nint_t>(sizeof(Acc)), 64));
  }
  auto acc_tensor = [&]() {
    if constexpr (OutputAcc) {
      return output_acc_storage<Acc>(c_output, logical_m, logical_n);
    } else {
      return tensor::make_tensor(
          acc_data, tensor::make_layout(
              tensor::make_shape(meta::Any{acc_m}, meta::Any{acc_n})));
    }
  }();
  auto acc_input = tensor::input<Acc>(acc_tensor);
  auto acc_output = tensor::output<Acc>(acc_tensor);

  auto run_loop = [&](const auto& whole_a, const auto& whole_b) {
    LoopNest<Order, decltype(tiling)>::run_phased(
        m, n, k, tiling,
        [&](const auto& block, auto phase) {
        auto a_block = narrow_input<AtomT, Operand::A>(
            whole_a, block.m_origin, block.m,
            block.k_origin, block.k);
        auto b_block = narrow_input<AtomT, Operand::B>(
            whole_b, block.n_origin, block.n,
            block.k_origin, block.k);
        auto c_input_block = narrow_c_input(
            c_input, block.m_origin, block.m,
            block.n_origin, block.n);
        auto c_output_block = narrow_c_output(
            c_output, block.m_origin, block.m,
            block.n_origin, block.n);
        AController::with_panel(
            scope, a_block, [&](const auto& active_a) {
          BController::with_panel(
              scope, b_block, [&](const auto& active_b) {
            if constexpr (decltype(phase)::first_k &&
                          decltype(phase)::last_k) {
              Scheduler<
                  AtomT, typename Config::SchedulerPolicy, Implementation>::run(
                      scope, block.m, block.n, block.k,
                      active_a, active_b,
                      c_input_block, c_output_block, scratch);
              return;
            }
            const nint_t acc_m_origin =
                axis_precedes_k_v<Axis::M, Order> ? 0 : block.m_origin;
            const nint_t acc_n_origin =
                axis_precedes_k_v<Axis::N, Order> ? 0 : block.n_origin;
            auto acc_input_block = narrow_c_input(
                acc_input, acc_m_origin, block.m,
                acc_n_origin, block.n);
            auto acc_output_block = narrow_c_output(
                acc_output, acc_m_origin, block.m,
                acc_n_origin, block.n);
            if constexpr (decltype(phase)::first_k) {
              Scheduler<
                  AtomT, typename Config::SchedulerPolicy, Implementation>::run(
                      scope, block.m, block.n, block.k,
                      active_a, active_b,
                      c_input_block, acc_output_block, scratch);
            } else if constexpr (decltype(phase)::last_k) {
              Scheduler<
                  AtomT, typename Config::SchedulerPolicy, Implementation>::run(
                      scope, block.m, block.n, block.k,
                      active_a, active_b,
                      acc_input_block, c_output_block, scratch);
            } else {
              Scheduler<
                  AtomT, typename Config::SchedulerPolicy, Implementation>::run(
                      scope, block.m, block.n, block.k,
                      active_a, active_b,
                      acc_input_block, acc_output_block, scratch);
            }
          });
        });
      });
  };
  const nint_t l3_full_k_budget = std::max<nint_t>(
      config.cache_info_provider().l3_bytes / 2, 1);
  const bool allow_full_a =
      APacking::extent == PackingExtent::full_k ||
      logical_m * logical_k * static_cast<nint_t>(sizeof(typename AtomT::TA))
          <= l3_full_k_budget / 2;
  const bool allow_full_b =
      BPacking::extent == PackingExtent::full_k ||
      logical_n * logical_k * static_cast<nint_t>(sizeof(typename AtomT::TB))
          <= l3_full_k_budget / 2;
  AController::with_whole_operand(
      scope, a, allow_full_a, [&](const auto& whole_a) {
        BController::with_whole_operand(
            scope, b, allow_full_b, [&](const auto& whole_b) {
              run_loop(whole_a, whole_b);
            });
      });
  workspace.rewind(operation_mark);
}

template <typename Config, typename Implementation,
          meta::ValueType M, meta::ValueType N, meta::ValueType K,
          tensor::InputSpecLike ASpec, tensor::InputSpecLike BSpec,
          tensor::OutputSpecLike COutputSpec>
VECOPS_INLINE nint_t tiled_workspace_bytes(
    const Config& config, M m, N n, K k,
    const ASpec& a, const BSpec& b, const COutputSpec&) {
  using AtomT = typename Config::Atom;
  auto tiling = resolve_cache_tiling(config);
  nint_t bytes = kernel::matmul_implementation::scratch_bytes<Implementation>();
  const nint_t logical_m = static_cast<nint_t>(m);
  const nint_t logical_n = static_cast<nint_t>(n);
  const nint_t logical_k = static_cast<nint_t>(k);
  using Order = resolved_loop_order_t<Config>;
  constexpr bool GeneratesKLoop = LoopNest<
      Order, decltype(tiling)>::template generates_loop<Axis::K, M, N, K>;
  if (GeneratesKLoop && logical_k > static_cast<nint_t>(tiling.kc) &&
      !uses_output_accumulator_v<Config, COutputSpec>) {
    const nint_t acc_m = axis_precedes_k_v<Axis::M, Order>
        ? std::min(logical_m, static_cast<nint_t>(tiling.mc))
        : logical_m;
    const nint_t acc_n = axis_precedes_k_v<Axis::N, Order>
        ? std::min(logical_n, static_cast<nint_t>(tiling.nc))
        : logical_n;
    bytes += acc_m * acc_n *
        static_cast<nint_t>(sizeof(typename AtomT::TAcc)) + 63;
  }
  const nint_t panel_m = std::min(logical_m,
      static_cast<nint_t>(tiling.mc));
  const nint_t panel_n = std::min(logical_n,
      static_cast<nint_t>(tiling.nc));
  const nint_t panel_k = std::min(logical_k,
      static_cast<nint_t>(tiling.kc));
  using Tuning = typename Config::GenericTuning;
  if constexpr (should_pack_v<typename Tuning::APacking,
                              AtomT, Operand::A, ASpec>) {
    const nint_t packing_m = full_k_packing_v<
        typename Tuning::APacking, Operand::A, Order>
        ? logical_m : panel_m;
    const nint_t packing_k = full_k_packing_v<
        typename Tuning::APacking, Operand::A, Order>
        ? logical_k : panel_k;
    auto layout = packed_layout<AtomT, Operand::A>(tensor::make_layout(
        tensor::make_shape(meta::Any{packing_m}, meta::Any{packing_k})));
    bytes += tensor::numel(layout) *
        static_cast<nint_t>(sizeof(typename AtomT::TA)) + 63;
  }
  if constexpr (should_pack_v<typename Tuning::BPacking,
                              AtomT, Operand::B, BSpec>) {
    const nint_t packing_n = full_k_packing_v<
        typename Tuning::BPacking, Operand::B, Order>
        ? logical_n : panel_n;
    const nint_t packing_k = full_k_packing_v<
        typename Tuning::BPacking, Operand::B, Order>
        ? logical_k : panel_k;
    auto layout = packed_layout<AtomT, Operand::B>(tensor::make_layout(
        tensor::make_shape(meta::Any{packing_n}, meta::Any{packing_k})));
    bytes += tensor::numel(layout) *
        static_cast<nint_t>(sizeof(typename AtomT::TB)) + 63;
  }
  return bytes;
}

} // namespace vecops::matmul::details

#endif // VECOPS_MATMUL_DETAILS_TILER_H
