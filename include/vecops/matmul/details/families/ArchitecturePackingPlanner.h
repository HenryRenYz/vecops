//
// Copyright (c) vecops contributors.
//

#ifndef VECOPS_MATMUL_DETAILS_FAMILIES_ARCHITECTURE_PACKING_PLANNER_H
#define VECOPS_MATMUL_DETAILS_FAMILIES_ARCHITECTURE_PACKING_PLANNER_H

#include "vecops/execution/details/arm/Resources.h"
#include "vecops/matmul/Packing.h"

namespace vecops::ops::matmul_details {

/** Packed-layout, byte-accounting, and packing-region ownership. */
template <typename Invocation>
struct ArchitecturePackingPlanner {
  template <::vecops::matmul::Operand Side, typename Spec>
  VECOPS_ALWAYS_INLINE static auto packed_layout(const Spec& spec) {
    constexpr int Rank = Spec::InputTensor::Ndim;
    if constexpr (Rank == 2) {
      return matmul_packed_layout<typename Invocation::AtomType, Side>(
          spec.input_layout());
    } else {
      static_assert(Rank == 3);
      const auto& layout = spec.input_layout();
      return matmul_packed_layout<typename Invocation::AtomType, Side>(
          tensor::make_layout(tensor::make_shape(
              tensor::size_value<Rank - 2>(layout),
              tensor::size_value<Rank - 1>(layout))));
    }
  }

  template <::vecops::matmul::Operand Side, typename Spec>
  VECOPS_ALWAYS_INLINE static nint_t packed_bytes(const Spec& spec) {
    if constexpr (Invocation::template AutoPackOperand<Side, Spec>) {
      using Element = typename ::vecops::matmul::packing_t<
          typename Invocation::AtomType, Side>::Element;
      const auto layout = packed_layout<Side>(spec);
      return tensor::numel(layout) * static_cast<nint_t>(sizeof(Element)) + 63;
    } else {
      return 0;
    }
  }

  VECOPS_ALWAYS_INLINE static auto batch_rows_packed_a_layout(
      const Invocation& op) {
    static_assert(
        Invocation::BatchRowsPackACandidate ||
        Invocation::SMEBatchRowsFullPackCandidate);
    const auto batch = tensor::size_value<0>(op.c_output_.output_layout());
    const auto flat_m = batch * op.m_;
    const auto flat_layout = tensor::make_layout(
        tensor::make_shape(flat_m, op.k_));
    return matmul_packed_layout<
        typename Invocation::AtomType, ::vecops::matmul::Operand::A>(
            flat_layout);
  }

  VECOPS_ALWAYS_INLINE static nint_t batch_rows_packed_a_bytes(
      const Invocation& op) {
    if constexpr (
        !Invocation::BatchRowsPackACandidate &&
        !Invocation::SMEBatchRowsFullPackCandidate) {
      return 0;
    } else {
      const auto layout = batch_rows_packed_a_layout(op);
      return tensor::numel(layout) * static_cast<nint_t>(
          sizeof(typename Invocation::AtomType::TA)) + 63;
    }
  }

  template <execution::ExecutionScope Scope>
  VECOPS_ALWAYS_INLINE static void execute(
      Scope& scope, const Invocation& op) {
    if constexpr (Invocation::SingleStreamingAutoPackRegion) {
      scope.with_resources(
          execution::details::arm::StreamingZARegion{},
          [&](auto& active) VECOPS_INLINE_LAMBDA_NOEXCEPT {
            op.execute_auto_packed_in_scope(active);
          });
    } else {
      op.execute_auto_packed_in_scope(scope);
    }
  }
};

} // namespace vecops::ops::matmul_details

#endif // VECOPS_MATMUL_DETAILS_FAMILIES_ARCHITECTURE_PACKING_PLANNER_H
