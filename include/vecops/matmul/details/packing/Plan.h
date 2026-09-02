//
// Copyright (c) vecops contributors.
//

#ifndef VECOPS_MATMUL_DETAILS_PACK_PLAN_H
#define VECOPS_MATMUL_DETAILS_PACK_PLAN_H

#include <cstdint>
#include <type_traits>
#include <utility>

#include "vecops/Assertion.h"
#include "vecops/execution/ExecutionSession.h"
#include "vecops/matmul/Packing.h"
#include "vecops/matmul/details/packing/Kernel.h"
#include "vecops/tensor/DataAccess.h"
#include "vecops/util/Math.h"

namespace vecops::matmul::details {

template <::vecops::matmul::Atom Atom, ::vecops::matmul::Operand Side,
          typename InputSpec, typename OutputSpec>
using SelectedPackImplementation = std::conditional_t<
    kernel::matmul_pack_details::Backend<
        typename ::vecops::matmul::packing_t<Atom, Side>::FormatType,
        kernel::matmul_pack_implementation::SME>::template eligible<
            InputSpec, OutputSpec>,
    kernel::matmul_pack_implementation::SME,
    std::conditional_t<
        kernel::matmul_pack_details::Backend<
            typename ::vecops::matmul::packing_t<Atom, Side>::FormatType,
            kernel::matmul_pack_implementation::SMEFP32ToFP64>::
            template eligible<InputSpec, OutputSpec>,
        kernel::matmul_pack_implementation::SMEFP32ToFP64,
        std::conditional_t<
            kernel::matmul_pack_details::Backend<
                typename ::vecops::matmul::packing_t<Atom, Side>::FormatType,
                kernel::matmul_pack_implementation::SMEStagedTransform>::
                template eligible<InputSpec, OutputSpec>,
            kernel::matmul_pack_implementation::SMEStagedTransform,
            std::conditional_t<
                kernel::matmul_pack_details::Backend<
                    typename ::vecops::matmul::packing_t<Atom, Side>::FormatType,
                    kernel::matmul_pack_implementation::SMEStagedFP16ToFP32>::
                    template eligible<InputSpec, OutputSpec>,
                kernel::matmul_pack_implementation::SMEStagedFP16ToFP32,
                std::conditional_t<
                    kernel::matmul_pack_details::Backend<
                        typename ::vecops::matmul::packing_t<Atom, Side>::FormatType,
                        kernel::matmul_pack_implementation::SMEPostprocess>::
                        template eligible<InputSpec, OutputSpec>,
                    kernel::matmul_pack_implementation::SMEPostprocess,
                    kernel::matmul_pack_implementation::Vector>>>>>;

/** Stateless typed plan for one ordinary A/B packing call. */
template <::vecops::matmul::Atom Atom, ::vecops::matmul::Operand Side,
          typename InputSpec, typename OutputSpec>
class MatmulPackPlan {
  using Packing = ::vecops::matmul::packing_t<Atom, Side>;

public:
  using ComputeType = typename Packing::Element;
  using Implementation = SelectedPackImplementation<
      Atom, Side, InputSpec, OutputSpec>;
  using ResourceRequirements =
      kernel::matmul_pack_implementation::resource_requirements_t<
          Atom, Side, Implementation>;

  static consteval void validate_types() {
    static_assert(InputSpec::InputTensor::Ndim == 2,
                  "matmul pack input must be rank two");
    static_assert(
        std::same_as<typename OutputSpec::MemoryElement, ComputeType>,
        "packed output memory element must equal the Atom operand type");
    static_assert(
        std::same_as<typename OutputSpec::ComputeType, ComputeType>,
        "packed output compute element must equal the Atom operand type");
    static_assert(
        std::same_as<typename OutputSpec::TransformType, tensor::NoTransform>,
        "packed output does not accept an output transform");
  }

  VECOPS_INLINE nint_t required_workspace(
      const InputSpec& input, const OutputSpec& output) const {
    validate_types();
    validate(input, output);
    return 0;
  }

  template <execution::ExecutionScope Scope>
  VECOPS_INLINE void operator()(
      Scope& scope, const InputSpec& input,
      const OutputSpec& output) const {
    validate_types();
    validate(input, output);
    scope.with_resources(
        *this, [&](auto& active) VECOPS_INLINE_LAMBDA_NOEXCEPT {
          execute(active, input, output);
        });
  }

  VECOPS_INLINE void operator()(
      kernel::WorkspaceView& workspace, const InputSpec& input,
      const OutputSpec& output) const {
    ExecutionSession execution{workspace};
    (*this)(execution, input, output);
  }

private:
  VECOPS_INLINE static void validate(
      const InputSpec& input, const OutputSpec& output) {
    VECOPS_ASSERT(static_cast<nint_t>(tensor::size_value<0>(
                      input.input_layout())) >= 0 &&
                      static_cast<nint_t>(tensor::size_value<1>(
                          input.input_layout())) >= 0,
                  "matrix packing extents must be non-negative");
    VECOPS_ASSERT(
        (::vecops::matmul::is_corresponding_packed_layout<Atom, Side>(
            input.input_layout(), output.output_layout())),
        "packed output layout does not correspond to the input layout");
    if (tensor::numel(output.output_layout()) != 0) {
      VECOPS_ASSERT(is_aligned(64, output.tensor().data()),
                    "packed output base must be 64-byte aligned");
      VECOPS_ASSERT(
          reinterpret_cast<const void*>(input.tensor().data()) !=
              reinterpret_cast<const void*>(output.tensor().data()),
          "matrix packing does not support in-place output");
    }
  }

  template <execution::ExecutionScope Scope>
  VECOPS_ALWAYS_INLINE static void execute(
      Scope& scope, const InputSpec& input, const OutputSpec& output) {
    using InputPolicy = tensor::InputAccessPolicy<
        Packing::VectorAxis, 1, tensor::AccessPlan::direct>;
    using OutputPolicy = tensor::OutputAccessPolicy<
        OutputSpec::OutputTensor::Ndim - 1, tensor::AccessPlan::direct>;
    kernel::with_operands(
        scope,
        tensor::operand(input, InputPolicy{}),
        tensor::operand(output, OutputPolicy{}),
        [&](auto& source, auto& destination) VECOPS_INLINE_LAMBDA {
          kernel::matmul_pack_bound<Atom, Side>(
              scope, source, destination, Implementation{});
          destination.commit();
        });
  }

};

/**
 * Pack signed-byte B and generate the asymmetric-A column correction.
 *
 * The packed ABI is identical to MatmulPack.  The independent int32[N]
 * sidecar stores -a_zero_point * sum_k(B[n,k]) and can be broadcast directly
 * as the C input of an unsigned-A x signed-B matmul.  This is deliberately a
 * parallel operation instead of a nullable option on MatmulPack, so ordinary
 * packing keeps exactly the same template path and generated instructions.
 */
/** Stateless typed plan for B packing with an asymmetric-A correction. */
template <::vecops::matmul::Atom Atom,
          typename InputSpec, typename OutputSpec,
          typename CompensationOutputSpec>
class MatmulPackBCompensatedPlan {
  using Packing = ::vecops::matmul::packing_t<Atom, ::vecops::matmul::Operand::B>;

public:
  using ComputeType = typename Packing::Element;
  using Implementation = SelectedPackImplementation<
      Atom, ::vecops::matmul::Operand::B, InputSpec, OutputSpec>;
  using ResourceRequirements =
      kernel::matmul_pack_implementation::resource_requirements_t<
          Atom, ::vecops::matmul::Operand::B, Implementation>;

  static consteval void validate_types() {
    static_assert(std::same_as<typename Atom::TA, uint8_t> &&
                  std::same_as<typename Atom::TB, int8_t> &&
                  std::same_as<typename Atom::TAcc, int32_t>,
                  "column compensation requires a u8 x s8 -> i32 atom");
    static_assert(InputSpec::InputTensor::Ndim == 2,
                  "compensated B-pack input must be rank two");
    static_assert(CompensationOutputSpec::OutputTensor::Ndim == 1,
                  "B compensation output must be rank one");
    static_assert(std::same_as<ComputeType, int8_t>);
    static_assert(
        std::same_as<typename OutputSpec::MemoryElement, ComputeType> &&
        std::same_as<typename OutputSpec::ComputeType, ComputeType> &&
        std::same_as<typename OutputSpec::TransformType, tensor::NoTransform>,
        "packed output must be native signed byte with no transform");
    static_assert(
        std::same_as<
            typename CompensationOutputSpec::MemoryElement, int32_t> &&
        std::same_as<
            typename CompensationOutputSpec::ComputeType, int32_t> &&
        std::same_as<
            typename CompensationOutputSpec::TransformType,
            tensor::NoTransform>,
        "B compensation output must be native int32 with no transform");
  }

  VECOPS_INLINE nint_t required_workspace(
      const InputSpec& input, const OutputSpec& output,
      const CompensationOutputSpec& compensation) const {
    validate_types();
    validate(input, output, compensation);
    return 0;
  }

  template <execution::ExecutionScope Scope>
  VECOPS_INLINE void operator()(
      Scope& scope, const InputSpec& input, const OutputSpec& output,
      const CompensationOutputSpec& compensation,
      int32_t a_zero_point) const {
    validate_types();
    validate(input, output, compensation);
    scope.with_resources(
        *this, [&](auto& active) VECOPS_INLINE_LAMBDA_NOEXCEPT {
          execute(active, input, output, compensation, a_zero_point);
        });
  }

  VECOPS_INLINE void operator()(
      kernel::WorkspaceView& workspace, const InputSpec& input,
      const OutputSpec& output,
      const CompensationOutputSpec& compensation,
      int32_t a_zero_point) const {
    ExecutionSession execution{workspace};
    (*this)(execution, input, output, compensation, a_zero_point);
  }

private:
  VECOPS_INLINE static void validate(
      const InputSpec& input, const OutputSpec& output,
      const CompensationOutputSpec& compensation) {
    const auto& input_layout = input.input_layout();
    VECOPS_ASSERT(static_cast<nint_t>(
                      tensor::size_value<0>(input_layout)) >= 0 &&
                      static_cast<nint_t>(
                          tensor::size_value<1>(input_layout)) >= 0,
                  "matrix packing extents must be non-negative");
    VECOPS_ASSERT(
        (::vecops::matmul::is_corresponding_packed_layout<
            Atom, ::vecops::matmul::Operand::B>(input_layout, output.output_layout())),
        "packed B layout does not correspond to the input layout");
    VECOPS_ASSERT(
        static_cast<nint_t>(tensor::size_value<0>(
            compensation.output_layout())) ==
            static_cast<nint_t>(tensor::size_value<0>(input_layout)),
        "B compensation extent must equal N");
    VECOPS_ASSERT(static_cast<nint_t>(tensor::stride_value<0>(
                      compensation.output_layout())) == 1,
                  "B compensation output must be contiguous");
    const auto* input_data =
        reinterpret_cast<const void*>(input.tensor().data());
    auto* output_data = reinterpret_cast<void*>(output.tensor().data());
    auto* compensation_data =
        reinterpret_cast<void*>(compensation.tensor().data());
    if (tensor::numel(output.output_layout()) != 0) {
      VECOPS_ASSERT(is_aligned(64, output_data),
                    "packed output base must be 64-byte aligned");
      VECOPS_ASSERT(input_data != output_data,
                    "matrix packing does not support in-place output");
      VECOPS_ASSERT(output_data != compensation_data,
                    "packed output and compensation must not alias");
    }
    if (tensor::numel(compensation.output_layout()) != 0) {
      VECOPS_ASSERT(input_data != compensation_data,
                    "input and compensation must not alias");
    }
  }

  template <execution::ExecutionScope Scope>
  VECOPS_ALWAYS_INLINE static void execute(
      Scope& scope, const InputSpec& input, const OutputSpec& output,
      const CompensationOutputSpec& compensation, int32_t a_zero_point) {
    using InputPolicy = tensor::InputAccessPolicy<
        Packing::VectorAxis, 1, tensor::AccessPlan::direct>;
    using OutputPolicy = tensor::OutputAccessPolicy<
        OutputSpec::OutputTensor::Ndim - 1, tensor::AccessPlan::direct>;
    using CompensationPolicy =
        tensor::OutputAccessPolicy<0, tensor::AccessPlan::direct>;
    kernel::with_operands(
        scope,
        tensor::operand(input, InputPolicy{}),
        tensor::operand(output, OutputPolicy{}),
        tensor::operand(compensation, CompensationPolicy{}),
        [&](auto& source, auto& destination, auto& compensation)
            VECOPS_INLINE_LAMBDA {
          kernel::matmul_pack_b_compensated_bound<Atom>(
              scope, source, destination, compensation, a_zero_point,
              Implementation{});
          destination.commit();
          compensation.commit();
        });
  }

};

template <::vecops::matmul::Atom Atom, ::vecops::matmul::Operand Side,
          tensor::InputOperand Input, tensor::OutputOperand Output>
VECOPS_INLINE auto select_matmul_pack_plan(
    Input&&, Output&&) {
  using Element = typename ::vecops::matmul::packing_t<Atom, Side>::Element;
  using InputSpec = std::remove_cvref_t<decltype(
      tensor::as_input_spec<Element>(std::declval<Input>()))>;
  using OutputSpec = std::remove_cvref_t<decltype(
      tensor::as_output_spec<Element>(std::declval<Output>()))>;
  return MatmulPackPlan<
      Atom, Side, InputSpec, OutputSpec>{};
}

template <::vecops::matmul::Atom Atom, ::vecops::matmul::Operand Side,
          tensor::InputOperand Input, tensor::OutputOperand Output>
VECOPS_INLINE nint_t matmul_pack_workspace_bytes(
    Input&& input, Output&& output) {
  using Element = typename ::vecops::matmul::packing_t<Atom, Side>::Element;
  auto input_spec = tensor::as_input_spec<Element>(
      std::forward<Input>(input));
  auto output_spec = tensor::as_output_spec<Element>(
      std::forward<Output>(output));
  auto plan = select_matmul_pack_plan<Atom, Side>(input_spec, output_spec);
  return plan.required_workspace(input_spec, output_spec);
}

template <::vecops::matmul::Atom Atom, ::vecops::matmul::Operand Side,
          execution::ExecutionScope Scope,
          tensor::InputOperand Input, tensor::OutputOperand Output>
VECOPS_INLINE void run_matmul_pack(
    Scope& scope, Input&& input, Output&& output) {
  using Element = typename ::vecops::matmul::packing_t<Atom, Side>::Element;
  auto input_spec = tensor::as_input_spec<Element>(
      std::forward<Input>(input));
  auto output_spec = tensor::as_output_spec<Element>(
      std::forward<Output>(output));
  auto plan = select_matmul_pack_plan<Atom, Side>(input_spec, output_spec);
  plan(scope, input_spec, output_spec);
}

template <::vecops::matmul::Atom Atom,
          tensor::InputOperand Input,
          tensor::OutputOperand Output,
          tensor::OutputOperand CompensationOutput>
VECOPS_INLINE auto select_matmul_pack_b_compensated_plan(
    Input&&, Output&&, CompensationOutput&&) {
  using Element = typename ::vecops::matmul::packing_t<
      Atom, ::vecops::matmul::Operand::B>::Element;
  using InputSpec = std::remove_cvref_t<decltype(
      tensor::as_input_spec<Element>(std::declval<Input>()))>;
  using OutputSpec = std::remove_cvref_t<decltype(
      tensor::as_output_spec<Element>(std::declval<Output>()))>;
  using CompensationSpec = std::remove_cvref_t<decltype(
      tensor::as_output_spec<int32_t>(
          std::declval<CompensationOutput>()))>;
  return MatmulPackBCompensatedPlan<
      Atom, InputSpec, OutputSpec, CompensationSpec>{};
}

template <::vecops::matmul::Atom Atom,
          tensor::InputOperand Input,
          tensor::OutputOperand Output,
          tensor::OutputOperand CompensationOutput>
VECOPS_INLINE nint_t matmul_pack_b_compensated_workspace_bytes(
    Input&& input, Output&& output, CompensationOutput&& compensation) {
  using Element = typename ::vecops::matmul::packing_t<
      Atom, ::vecops::matmul::Operand::B>::Element;
  auto input_spec = tensor::as_input_spec<Element>(
      std::forward<Input>(input));
  auto output_spec = tensor::as_output_spec<Element>(
      std::forward<Output>(output));
  auto compensation_spec = tensor::as_output_spec<int32_t>(
      std::forward<CompensationOutput>(compensation));
  auto plan = select_matmul_pack_b_compensated_plan<Atom>(
      input_spec, output_spec, compensation_spec);
  return plan.required_workspace(
      input_spec, output_spec, compensation_spec);
}

template <::vecops::matmul::Atom Atom,
          execution::ExecutionScope Scope,
          tensor::InputOperand Input,
          tensor::OutputOperand Output,
          tensor::OutputOperand CompensationOutput>
VECOPS_INLINE void run_matmul_pack_b_compensated(
    Scope& scope, Input&& input, Output&& output,
    CompensationOutput&& compensation, int32_t a_zero_point) {
  using Element = typename ::vecops::matmul::packing_t<
      Atom, ::vecops::matmul::Operand::B>::Element;
  auto input_spec = tensor::as_input_spec<Element>(
      std::forward<Input>(input));
  auto output_spec = tensor::as_output_spec<Element>(
      std::forward<Output>(output));
  auto compensation_spec = tensor::as_output_spec<int32_t>(
      std::forward<CompensationOutput>(compensation));
  auto plan = select_matmul_pack_b_compensated_plan<Atom>(
      input_spec, output_spec, compensation_spec);
  plan(scope, input_spec, output_spec, compensation_spec, a_zero_point);
}

} // namespace vecops::matmul::details

#endif // VECOPS_MATMUL_DETAILS_PACK_PLAN_H
