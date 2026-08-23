//
// Copyright (c) vecops contributors.
//

#ifndef VECOPS_OPS_TRANSPOSE_H
#define VECOPS_OPS_TRANSPOSE_H

#include <type_traits>
#include <utility>

#include "vecops/execution/ExecutionSession.h"
#include "vecops/kernel/Transpose2D.h"
#include "vecops/ops/details/transpose/Selection.h"
#include "vecops/tensor/DataAccess.h"

/**
 * @file Transpose.h
 * @brief Prepared and one-shot rank-two transpose operators.
 *
 * Operand/layout types are normalized before execution, making SME eligibility
 * and resource requirements part of the operation type. Calling an operation
 * with an `ExecutionScope` opens only missing resources. Data conversion and
 * transforms remain DataAccess responsibilities around the selected kernel.
 */

namespace vecops::ops {

/**
 * @brief Prepared rank-two transpose operation with typed resource requirements.
 * @tparam Compute Logical compute element type used by DataAccess.
 * @tparam InputSpec Normalized readable operand specification.
 * @tparam OutputSpec Normalized writable operand specification.
 * @tparam Policy Transpose policy selected by the caller.
 * @tparam UseSME Compile-time implementation decision.
 *
 * Operand and layout types are part of the operator type, so hardware
 * requirements are known before entering an execution region. The object owns
 * Specs, not tensor storage; referenced tensor memory must outlive invocation.
 */
template <vec::Element Compute,
          typename InputSpec, typename OutputSpec, typename Policy,
          bool UseSME>
class Transpose {
public:
  using ComputeType = Compute;
  using Implementation = std::conditional_t<
      UseSME,
      kernel::transpose2d_implementation::SME,
      kernel::transpose2d_implementation::Vector>;
  /** Resources required by the selected Vector or SME implementation. */
  using ResourceRequirements = typename kernel::transpose2d_implementation::
      Traits<Implementation>::ResourceRequirements;

  /** Construct a prepared operation from normalized input/output Specs. */
  VECOPS_INLINE Transpose(
      InputSpec input, OutputSpec output, Policy policy = {})
      : input_(std::move(input)), output_(std::move(output)), policy_(policy) {
    static_assert(InputSpec::InputTensor::Ndim == 2,
                  "Transpose accepts rank-two input");
    static_assert(OutputSpec::OutputTensor::Ndim == 2,
                  "Transpose accepts rank-two output");
  }

  template <execution::ExecutionScope Scope>
  /**
   * @brief Execute under `scope`, entering this operation's missing resources.
   * @param scope Root session or enclosing active region.
   */
  VECOPS_INLINE void operator()(Scope& scope) const {
    scope.with_resources(
        *this, [&](auto& active) VECOPS_INLINE_LAMBDA { execute(active); });
  }

private:
  template <execution::ExecutionScope Scope>
  VECOPS_INLINE void execute(Scope& scope) const {
    using M = tensor::size_type_t<0, typename InputSpec::InputLayout>;
    using N = tensor::size_type_t<1, typename InputSpec::InputLayout>;
    const M m{tensor::get<0>(input_.input_layout().shape())};
    const N n{tensor::get<1>(input_.input_layout().shape())};
    VECOPS_ASSERT(
        static_cast<nint_t>(tensor::get<0>(
            output_.output_layout().shape())) == static_cast<nint_t>(n) &&
        static_cast<nint_t>(tensor::get<1>(
            output_.output_layout().shape())) == static_cast<nint_t>(m),
        "transpose output shape must be (N, M)");

    using InputPolicy = tensor::InputAccessPolicy<
        1, 1, tensor::AccessPlan::direct>;
    using OutputPolicy = tensor::OutputAccessPolicy<
        1, tensor::AccessPlan::direct>;
    kernel::with_operands(
        scope,
        tensor::operand(input_, InputPolicy{}),
        tensor::operand(output_, OutputPolicy{}),
        [&](auto& source, auto& destination) VECOPS_INLINE_LAMBDA {
          tensor::Coord<2> origin{};
          kernel::transpose2d_bound<0, 1, 0, 1>(
              scope, m, n, source, origin, destination, origin, policy_,
              Implementation{});
          destination.commit();
        });
  }

  InputSpec input_;
  OutputSpec output_;
  [[no_unique_address]] Policy policy_;
};

template <vec::Element Compute,
          tensor::InputOperand Input,
          tensor::OutputOperand Output,
          typename Policy = kernel::transpose2d_policy::Automatic>
/**
 * @brief Normalize operands and construct a typed transpose operation.
 * @param input Readable rank-two Tensor or InputSpec.
 * @param output Writable rank-two Tensor or OutputSpec shaped `(N,M)`.
 * @param policy Automatic or Gather policy.
 * @return Prepared `Transpose` whose type includes the implementation choice.
 *
 * SME is selected only when tensor and layout types prove every required
 * property. Runtime contiguity does not introduce an SME fallback branch.
 */
VECOPS_INLINE auto make_transpose(
    Input&& input, Output&& output, Policy policy = {}) {
  auto input_spec = tensor::as_input_spec<Compute>(std::forward<Input>(input));
  auto output_spec = tensor::as_output_spec<Compute>(
      std::forward<Output>(output));
  using InputSpec = std::remove_cvref_t<decltype(input_spec)>;
  using OutputSpec = std::remove_cvref_t<decltype(output_spec)>;
  constexpr bool UseSME = transpose_details::use_sme_v<
      Input, Output, InputSpec, OutputSpec, Policy>;
  return Transpose<Compute, InputSpec, OutputSpec, Policy, UseSME>{
      std::move(input_spec), std::move(output_spec), policy};
}

template <vec::Element Compute,
          execution::ExecutionScope Scope,
          tensor::InputOperand Input,
          tensor::OutputOperand Output,
          typename Policy = kernel::transpose2d_policy::Automatic>
/**
 * @brief Execute one transpose through an existing execution scope.
 * @param scope Root session or active enclosing region.
 * @param input Readable rank-two operand of shape `(M,N)`.
 * @param output Writable rank-two operand of shape `(N,M)`.
 * @param policy Automatic or forced Gather policy.
 */
VECOPS_INLINE void transpose(
    Scope& scope, Input&& input, Output&& output, Policy policy = {}) {
  auto operation = make_transpose<Compute>(
      std::forward<Input>(input), std::forward<Output>(output), policy);
  operation(scope);
}

template <vec::Element Compute,
          tensor::InputOperand Input,
          tensor::OutputOperand Output,
          typename Policy = kernel::transpose2d_policy::Automatic>
/**
 * @brief WorkspaceView compatibility overload for one-shot transpose.
 * @param workspace Caller-owned scratch storage used by DataAccess.
 * @param input Readable rank-two operand of shape `(M,N)`.
 * @param output Writable rank-two operand of shape `(N,M)`.
 * @param policy Automatic or forced Gather policy.
 *
 * A temporary `ExecutionSession` borrows the workspace and applies the same
 * compile-time resource protocol as the scope overload.
 */
VECOPS_INLINE void transpose(
    kernel::WorkspaceView& workspace,
    Input&& input, Output&& output, Policy policy = {}) {
  ExecutionSession execution{workspace};
  transpose<Compute>(
      execution, std::forward<Input>(input), std::forward<Output>(output),
      policy);
}

} // namespace vecops::ops

#endif // VECOPS_OPS_TRANSPOSE_H
