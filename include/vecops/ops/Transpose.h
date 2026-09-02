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
 * @brief Reusable Config-only rank-two transpose operator.
 *
 * Operand/layout types are normalized before execution, making SME eligibility
 * and resource requirements part of the operation type. Calling an operation
 * with an `ExecutionScope` opens only missing resources. Data conversion and
 * transforms remain DataAccess responsibilities around the selected kernel.
 */

namespace vecops::ops {

template <vec::Element ComputeT = float32_t,
          typename PolicyT = kernel::transpose2d_policy::Automatic>
struct TransposeConfig {
  using ComputeType = ComputeT;
  using Policy = PolicyT;

  [[no_unique_address]] PolicyT policy{};
};

namespace transpose_details {

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
class PreparedTranspose {
public:
  using ComputeType = Compute;
  using Implementation = std::conditional_t<
      UseSME,
      kernel::transpose2d_implementation::SME,
      kernel::transpose2d_implementation::Vector>;
  /** Resources required by the selected Vector or SME implementation. */
  using ResourceRequirements =
      kernel::transpose2d_implementation::resource_requirements_t<
          Implementation>;

  /** Construct a prepared operation from normalized input/output Specs. */
  VECOPS_INLINE PreparedTranspose(
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
        *this, [&](auto& active) VECOPS_INLINE_LAMBDA_NOEXCEPT {
          execute(active);
        });
  }

private:
  template <execution::ExecutionScope Scope>
  VECOPS_INLINE void execute(Scope& scope) const {
    using M = tensor::size_type_t<0, typename InputSpec::InputLayout>;
    using N = tensor::size_type_t<1, typename InputSpec::InputLayout>;
    const M m{tensor::size_value<0>(input_.input_layout())};
    const N n{tensor::size_value<1>(input_.input_layout())};
    VECOPS_ASSERT(
        static_cast<nint_t>(tensor::size_value<0>(
            output_.output_layout())) == static_cast<nint_t>(n) &&
        static_cast<nint_t>(tensor::size_value<1>(
            output_.output_layout())) == static_cast<nint_t>(m),
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

template <typename Config,
          tensor::InputOperand Input,
          tensor::OutputOperand Output>
VECOPS_INLINE auto prepare_transpose(
    const Config& config, Input&& input, Output&& output) {
  using Compute = typename Config::ComputeType;
  using Policy = typename Config::Policy;
  auto input_spec = tensor::as_input_spec<Compute>(std::forward<Input>(input));
  auto output_spec = tensor::as_output_spec<Compute>(
      std::forward<Output>(output));
  using InputSpec = std::remove_cvref_t<decltype(input_spec)>;
  using OutputSpec = std::remove_cvref_t<decltype(output_spec)>;
  constexpr bool UseSME = transpose_details::use_sme_v<
      Input, Output, InputSpec, OutputSpec, Policy>;
  return PreparedTranspose<Compute, InputSpec, OutputSpec, Policy, UseSME>{
      std::move(input_spec), std::move(output_spec), config.policy};
}

} // namespace transpose_details

/**
 * Reusable rank-two transpose operator. The object stores configuration only;
 * operand types select the prepared implementation independently per call.
 */
template <typename Config = TransposeConfig<>>
class Transpose {
public:
  const Config config;

  VECOPS_INLINE constexpr explicit Transpose(Config cfg = {})
      : config(std::move(cfg)) {}

  template <tensor::InputOperand Input, tensor::OutputOperand Output>
  VECOPS_INLINE nint_t required_workspace(
      const Input&, const Output&) const {
    return 0;
  }

  template <execution::ExecutionScope Scope,
            tensor::InputOperand Input, tensor::OutputOperand Output>
  VECOPS_INLINE void operator()(
      Scope& scope, Input&& input, Output&& output) const {
    auto prepared = transpose_details::prepare_transpose(
        config, std::forward<Input>(input), std::forward<Output>(output));
    prepared(scope);
  }

  template <tensor::InputOperand Input, tensor::OutputOperand Output>
  VECOPS_INLINE void operator()(
      kernel::WorkspaceView& workspace,
      Input&& input, Output&& output) const {
    ExecutionSession execution{workspace};
    (*this)(execution, std::forward<Input>(input),
            std::forward<Output>(output));
  }
};

template <typename Config = TransposeConfig<>>
VECOPS_INLINE constexpr auto transpose(Config config = {}) {
  return Transpose<Config>{std::move(config)};
}

} // namespace vecops::ops

#endif // VECOPS_OPS_TRANSPOSE_H
