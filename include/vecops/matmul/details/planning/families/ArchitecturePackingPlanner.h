//
// Copyright (c) vecops contributors.
//

#ifndef VECOPS_MATMUL_DETAILS_FAMILIES_ARCHITECTURE_PACKING_PLANNER_H
#define VECOPS_MATMUL_DETAILS_FAMILIES_ARCHITECTURE_PACKING_PLANNER_H

/**
 * @file vecops/matmul/details/planning/families/ArchitecturePackingPlanner.h
 * @brief Packed-layout construction, byte accounting, and packing-region
 *        ownership for online packing.
 *
 * Every byte count rounds up to a 64-byte boundary ("+ 63") to match the
 * aligned workspace.allocate() calls the execution paths make, so
 * required_workspace() can never under-report an allocation.  The execute()
 * entry decides whether the whole pack-and-multiply sequence runs inside a
 * single Streaming+ZA region.
 */

#include "vecops/execution/details/arm/Resources.h"
#include "vecops/matmul/Packing.h"

namespace vecops::matmul::details {

/** Packed-layout, byte-accounting, and packing-region ownership. */
template <typename Invocation>
struct ArchitecturePackingPlanner {
  /// Layout of the packed copy of one operand.  Rank-three inputs drop the
  /// batch dimension first: online packing reaches rank three only through
  /// a shared (broadcast) operand, which the rank gates in
  /// ArchitectureFamilyInvocation guarantee at every call site.
  template <::vecops::matmul::Operand Side, typename Spec>
  VECOPS_ALWAYS_INLINE static auto packed_layout(const Spec& spec) {
    constexpr int Rank = Spec::InputTensor::Ndim;
    if constexpr (Rank == 2) {
      return ::vecops::matmul::packed_layout<typename Invocation::AtomType, Side>(
          spec.input_layout());
    } else {
      static_assert(Rank == 3);
      const auto& layout = spec.input_layout();
      return ::vecops::matmul::packed_layout<typename Invocation::AtomType, Side>(
          tensor::make_layout(tensor::make_shape(
              tensor::size<Rank - 2>(layout),
              tensor::size<Rank - 1>(layout))));
    }
  }

  /// Bytes for the packed operand copy, 64-byte rounded, or 0 when this
  /// operand never packs.
  template <::vecops::matmul::Operand Side, typename Spec>
  VECOPS_ALWAYS_INLINE static nint_t packed_bytes(const Spec& spec) {
    if constexpr (Invocation::template AutoPackOperand<Side, Spec>) {
      using Element = typename ::vecops::matmul::packing_t<
          typename Invocation::AtomType, Side>::Element;
      const auto layout = packed_layout<Side>(spec);
      // Round up to the 64-byte allocation granularity used at run time.
      return tensor::numel(layout) * static_cast<nint_t>(sizeof(Element)) + 63;
    } else {
      return 0;
    }
  }

  /// Layout of the packed [batch*M, K] A consumed by the flattened
  /// batch-rows paths.
  VECOPS_ALWAYS_INLINE static auto batch_rows_packed_a_layout(
      const Invocation& op) {
    static_assert(
        Invocation::BatchRowsPackACandidate ||
        Invocation::SMEBatchRowsFullPackCandidate);
    const auto batch = tensor::size<0>(op.c_output_.output_layout());
    const auto flat_m = batch * op.m_;
    const auto flat_layout = tensor::make_layout(
        tensor::make_shape(flat_m, op.k_));
    return ::vecops::matmul::packed_layout<
        typename Invocation::AtomType, ::vecops::matmul::Operand::A>(
            flat_layout);
  }

  /// Bytes for that packed A, 64-byte rounded; 0 when both flatten-pack
  /// candidates are off.
  VECOPS_ALWAYS_INLINE static nint_t batch_rows_packed_a_bytes(
      const Invocation& op) {
    if constexpr (
        !Invocation::BatchRowsPackACandidate &&
        !Invocation::SMEBatchRowsFullPackCandidate) {
      return 0;
    } else {
      const auto layout = batch_rows_packed_a_layout(op);
      // Same 64-byte rounding contract as packed_bytes().
      return tensor::numel(layout) * static_cast<nint_t>(
          sizeof(typename Invocation::AtomType::TA)) + 63;
    }
  }

  /// Run the online-packed path, wrapping it in a single Streaming+ZA
  /// region when SingleStreamingAutoPackRegion holds: the pack kernels and
  /// the matrix multiply then share one streaming interval instead of
  /// crossing region boundaries between packing and computing.
  template <bool PackA, bool PackB, execution::ExecutionScope Scope>
  VECOPS_ALWAYS_INLINE static void execute(
      Scope& scope, const Invocation& op) {
    static_assert(PackA || PackB);
    if constexpr (Invocation::template SingleStreamingAutoPackRegion<
                      PackA, PackB>) {
      scope.with_resources(
          execution::details::arm::StreamingZARegion{},
          [&](auto& active) VECOPS_INLINE_LAMBDA_NOEXCEPT {
            op.template execute_auto_packed_in_scope<PackA, PackB>(active);
          });
    } else {
      op.template execute_auto_packed_in_scope<PackA, PackB>(scope);
    }
  }
};

} // namespace vecops::matmul::details

#endif // VECOPS_MATMUL_DETAILS_FAMILIES_ARCHITECTURE_PACKING_PLANNER_H
