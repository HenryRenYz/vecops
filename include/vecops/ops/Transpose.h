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
 * @file vecops/ops/Transpose.h
 * @brief Reusable Config-only rank-two transpose operator.
 *
 * Operand/layout types are normalized before execution, making SME eligibility
 * and resource requirements part of the operation type. Calling an operation
 * with an `ExecutionScope` opens only missing resources. Data conversion and
 * transforms remain DataAccess responsibilities around the selected kernel.
 *
 * The operator copies an `M×N` input into an `N×M` output (`out(j,i) =
 * in(i,j)`). Two granularity levels are exposed: the `Transpose` class is the
 * stateless entry point (normalize operands, select a backend, run), while
 * `transpose_details::prepare_transpose` returns a `PreparedTranspose` that
 * bakes operand types and the SME decision into one reusable value.
 *
 * ## Usage
 *
 * @code
 * #include "vecops/ops/Transpose.h"
 * using namespace vecops;
 *
 * auto op = ops::transpose();                    // float32, automatic policy
 * op(scope, input_tensor, output_tensor);        // out is (N, M)
 *
 * // Or prepare once, run many times with identical operand types:
 * auto prepared = ops::transpose_details::prepare_transpose(
 *     config, input_tensor, output_tensor);
 * prepared(scope);
 * @endcode
 *
 * ## Pitfalls
 *
 * - Only rank-two operands are accepted (static_assert); higher-rank
 *   transposes go through tensor/DataAccess materialization.
 * - The output shape must be `(N, M)` — dimension 0 holds the input's column
 *   count. A mismatch is caught by assertion at execution.
 * - The config object carries only compile-time values; each call re-selects
 *   the implementation from the operand types at hand, so one `Transpose`
 *   value can serve inputs with different element types or extents.
 * - Element-type conversion is performed through DataAccess (the
 *   `ComputeType` of the config), not by the transpose kernel itself.
 */

namespace vecops::ops {

/**
 * @brief Configuration bundle for the rank-two transpose operator.
 *
 * @tparam ComputeT Logical compute element type the transpose runs in;
 *                  memory-side conversions to/from this type are fused into
 *                  the DataAccess loads/stores around the kernel.
 * @tparam PolicyT  Backend policy; `Automatic` picks the register kernel
 *                  (or SME when eligible), `Gather` forces the generic
 *                  gather/store implementation.
 */
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
    const M m{tensor::size<0>(input_.input_layout())};
    const N n{tensor::size<1>(input_.input_layout())};
    VECOPS_ASSERT(
        static_cast<nint_t>(tensor::size<0>(
            output_.output_layout())) == static_cast<nint_t>(n) &&
        static_cast<nint_t>(tensor::size<1>(
            output_.output_layout())) == static_cast<nint_t>(m),
        "transpose output shape must be (N, M)");

    // Unit-stride vector lines along each operand's dim 1 (asserted by the
    // SME selection), one read pass, and a direct (non-materializing) access
    // plan on both sides: the transpose kernel consumes the operands as-is
    // and never needs a DataAccess staging buffer.
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
          // Axis roles <0, 1, 0, 1>: input dim 0 is the logical row axis
          // and dim 1 the column axis; output dim 0 receives the source
          // column index and dim 1 the source row index — the plain
          // out(j,i) = in(i,j) mapping for this rank-2 pair.
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

/**
 * @brief Normalize operands and bake the implementation decision into a
 * reusable `PreparedTranspose`.
 *
 * This is where SME eligibility is actually decided: `use_sme_v` inspects
 * the normalized specs (rank, policy, raw-tensor operands, no transforms,
 * unit-stride dim 1 on both sides) and the element/extent types. The
 * returned object carries that decision as a template parameter, so its
 * `ResourceRequirements` are known before any execution region is entered.
 *
 * @tparam Config  A `TransposeConfig` bundle.
 * @tparam Input   Readable rank-2 operand.
 * @tparam Output  Writable rank-2 operand.
 * @param config   Configuration supplying compute type and policy.
 * @param input    The M×N source operand.
 * @param output   The N×M destination operand.
 * @return A `PreparedTranspose` bound to the normalized operand specs.
 */
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
