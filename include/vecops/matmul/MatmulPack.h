//
// Copyright (c) vecops contributors.
//

#ifndef VECOPS_MATMUL_MATMUL_PACK_H
#define VECOPS_MATMUL_MATMUL_PACK_H

#include <cstdint>
#include <type_traits>
#include <utility>

#include "vecops/Assertion.h"
#include "vecops/execution/ExecutionSession.h"
#include "vecops/matmul/Packing.h"
#include "vecops/matmul/details/PackKernel.h"
#include "vecops/tensor/DataAccess.h"
#include "vecops/util/Math.h"

namespace vecops::ops {

namespace matmul_pack_details {

template <gemm::Atom Atom, gemm::Operand Side,
          typename InputSpec, typename OutputSpec>
using SelectedImplementation = std::conditional_t<
    kernel::matmul_pack_details::Backend<
        typename gemm::packing_t<Atom, Side>::FormatType,
        kernel::matmul_pack_implementation::SME>::template eligible<
            InputSpec, OutputSpec>,
    kernel::matmul_pack_implementation::SME,
    std::conditional_t<
        kernel::matmul_pack_details::Backend<
            typename gemm::packing_t<Atom, Side>::FormatType,
            kernel::matmul_pack_implementation::SMEFP32ToFP64>::
            template eligible<InputSpec, OutputSpec>,
        kernel::matmul_pack_implementation::SMEFP32ToFP64,
        std::conditional_t<
            kernel::matmul_pack_details::Backend<
                typename gemm::packing_t<Atom, Side>::FormatType,
                kernel::matmul_pack_implementation::SMEStagedTransform>::
                template eligible<InputSpec, OutputSpec>,
            kernel::matmul_pack_implementation::SMEStagedTransform,
            std::conditional_t<
                kernel::matmul_pack_details::Backend<
                    typename gemm::packing_t<Atom, Side>::FormatType,
                    kernel::matmul_pack_implementation::SMEStagedFP16ToFP32>::
                    template eligible<InputSpec, OutputSpec>,
                kernel::matmul_pack_implementation::SMEStagedFP16ToFP32,
                std::conditional_t<
                    kernel::matmul_pack_details::Backend<
                        typename gemm::packing_t<Atom, Side>::FormatType,
                        kernel::matmul_pack_implementation::SMEPostprocess>::
                        template eligible<InputSpec, OutputSpec>,
                    kernel::matmul_pack_implementation::SMEPostprocess,
                    kernel::matmul_pack_implementation::Vector>>>>>;

} // namespace matmul_pack_details

template <gemm::Atom Atom, gemm::Operand Side,
          tensor::LayoutLike InputLayout>
VECOPS_INLINE auto matmul_packed_layout(const InputLayout& input_layout) {
  return gemm::packed_layout<Atom, Side>(input_layout);
}

template <gemm::Atom Atom, gemm::Operand Side,
          typename InputSpec, typename OutputSpec>
class MatmulPack {
  using Packing = gemm::packing_t<Atom, Side>;

public:
  using ComputeType = typename Packing::Element;
  using Implementation = matmul_pack_details::SelectedImplementation<
      Atom, Side, InputSpec, OutputSpec>;
  using ResourceRequirements =
      kernel::matmul_pack_implementation::resource_requirements_t<
          Atom, Side, Implementation>;

  VECOPS_INLINE MatmulPack(InputSpec input, OutputSpec output)
      : input_(std::move(input)), output_(std::move(output)) {
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
    validate();
  }

  VECOPS_INLINE const auto& input_layout() const {
    return input_.input_layout();
  }

  VECOPS_INLINE const auto& output_layout() const {
    return output_.output_layout();
  }

  VECOPS_INLINE nint_t required_workspace() const { return 0; }

  template <execution::ExecutionScope Scope>
  VECOPS_INLINE void operator()(Scope& scope) const {
    scope.with_resources(
        *this, [&](auto& active) VECOPS_INLINE_LAMBDA_NOEXCEPT {
          execute(active);
        });
  }

  VECOPS_INLINE void operator()(kernel::WorkspaceView& workspace) const {
    ExecutionSession execution{workspace};
    (*this)(execution);
  }

private:
  VECOPS_INLINE void validate() const {
    VECOPS_ASSERT(static_cast<nint_t>(tensor::size_value<0>(
                      input_.input_layout())) >= 0 &&
                      static_cast<nint_t>(tensor::size_value<1>(
                          input_.input_layout())) >= 0,
                  "matrix packing extents must be non-negative");
    VECOPS_ASSERT(
        (gemm::is_corresponding_packed_layout<Atom, Side>(
            input_.input_layout(), output_.output_layout())),
        "packed output layout does not correspond to the input layout");
    if (tensor::numel(output_.output_layout()) != 0) {
      VECOPS_ASSERT(is_aligned(64, output_.tensor().data()),
                    "packed output base must be 64-byte aligned");
      VECOPS_ASSERT(
          reinterpret_cast<const void*>(input_.tensor().data()) !=
              reinterpret_cast<const void*>(output_.tensor().data()),
          "matrix packing does not support in-place output");
    }
  }

  template <execution::ExecutionScope Scope>
  VECOPS_ALWAYS_INLINE void execute(Scope& scope) const {
    validate();
    using InputPolicy = tensor::InputAccessPolicy<
        Packing::VectorAxis, 1, tensor::AccessPlan::direct>;
    using OutputPolicy = tensor::OutputAccessPolicy<
        OutputSpec::OutputTensor::Ndim - 1, tensor::AccessPlan::direct>;
    kernel::with_operands(
        scope,
        tensor::operand(input_, InputPolicy{}),
        tensor::operand(output_, OutputPolicy{}),
        [&](auto& source, auto& destination) VECOPS_INLINE_LAMBDA {
          kernel::matmul_pack_bound<Atom, Side>(
              scope, source, destination, Implementation{});
          destination.commit();
        });
  }

  InputSpec input_;
  OutputSpec output_;
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
template <gemm::Atom Atom,
          typename InputSpec, typename OutputSpec,
          typename CompensationOutputSpec>
class MatmulPackBCompensated {
  using Packing = gemm::packing_t<Atom, gemm::Operand::B>;

public:
  using ComputeType = typename Packing::Element;
  using Implementation = matmul_pack_details::SelectedImplementation<
      Atom, gemm::Operand::B, InputSpec, OutputSpec>;
  using ResourceRequirements =
      kernel::matmul_pack_implementation::resource_requirements_t<
          Atom, gemm::Operand::B, Implementation>;

  VECOPS_INLINE MatmulPackBCompensated(
      InputSpec input, OutputSpec output,
      CompensationOutputSpec compensation, int32_t a_zero_point)
      : input_(std::move(input)), output_(std::move(output)),
        compensation_(std::move(compensation)),
        a_zero_point_(a_zero_point) {
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
    validate();
  }

  VECOPS_INLINE nint_t required_workspace() const { return 0; }

  template <execution::ExecutionScope Scope>
  VECOPS_INLINE void operator()(Scope& scope) const {
    scope.with_resources(
        *this, [&](auto& active) VECOPS_INLINE_LAMBDA_NOEXCEPT {
          execute(active);
        });
  }

  VECOPS_INLINE void operator()(kernel::WorkspaceView& workspace) const {
    ExecutionSession execution{workspace};
    (*this)(execution);
  }

private:
  VECOPS_INLINE void validate() const {
    const auto& input_layout = input_.input_layout();
    VECOPS_ASSERT(static_cast<nint_t>(
                      tensor::size_value<0>(input_layout)) >= 0 &&
                      static_cast<nint_t>(
                          tensor::size_value<1>(input_layout)) >= 0,
                  "matrix packing extents must be non-negative");
    VECOPS_ASSERT(
        (gemm::is_corresponding_packed_layout<
            Atom, gemm::Operand::B>(input_layout, output_.output_layout())),
        "packed B layout does not correspond to the input layout");
    VECOPS_ASSERT(
        static_cast<nint_t>(tensor::size_value<0>(
            compensation_.output_layout())) ==
            static_cast<nint_t>(tensor::size_value<0>(input_layout)),
        "B compensation extent must equal N");
    VECOPS_ASSERT(static_cast<nint_t>(tensor::stride_value<0>(
                      compensation_.output_layout())) == 1,
                  "B compensation output must be contiguous");
    const auto* input_data =
        reinterpret_cast<const void*>(input_.tensor().data());
    auto* output_data = reinterpret_cast<void*>(output_.tensor().data());
    auto* compensation_data =
        reinterpret_cast<void*>(compensation_.tensor().data());
    if (tensor::numel(output_.output_layout()) != 0) {
      VECOPS_ASSERT(is_aligned(64, output_data),
                    "packed output base must be 64-byte aligned");
      VECOPS_ASSERT(input_data != output_data,
                    "matrix packing does not support in-place output");
      VECOPS_ASSERT(output_data != compensation_data,
                    "packed output and compensation must not alias");
    }
    if (tensor::numel(compensation_.output_layout()) != 0) {
      VECOPS_ASSERT(input_data != compensation_data,
                    "input and compensation must not alias");
    }
  }

  template <execution::ExecutionScope Scope>
  VECOPS_ALWAYS_INLINE void execute(Scope& scope) const {
    validate();
    using InputPolicy = tensor::InputAccessPolicy<
        Packing::VectorAxis, 1, tensor::AccessPlan::direct>;
    using OutputPolicy = tensor::OutputAccessPolicy<
        OutputSpec::OutputTensor::Ndim - 1, tensor::AccessPlan::direct>;
    using CompensationPolicy =
        tensor::OutputAccessPolicy<0, tensor::AccessPlan::direct>;
    kernel::with_operands(
        scope,
        tensor::operand(input_, InputPolicy{}),
        tensor::operand(output_, OutputPolicy{}),
        tensor::operand(compensation_, CompensationPolicy{}),
        [&](auto& source, auto& destination, auto& compensation)
            VECOPS_INLINE_LAMBDA {
          kernel::matmul_pack_b_compensated_bound<Atom>(
              scope, source, destination, compensation, a_zero_point_,
              Implementation{});
          destination.commit();
          compensation.commit();
        });
  }

  InputSpec input_;
  OutputSpec output_;
  CompensationOutputSpec compensation_;
  int32_t a_zero_point_;
};

template <gemm::Atom Atom, gemm::Operand Side,
          tensor::InputOperand Input, tensor::OutputOperand Output>
VECOPS_INLINE auto make_matmul_pack(Input&& input, Output&& output) {
  using Element = typename gemm::packing_t<Atom, Side>::Element;
  auto input_spec = tensor::as_input_spec<Element>(
      std::forward<Input>(input));
  auto output_spec = tensor::as_output_spec<Element>(
      std::forward<Output>(output));
  return MatmulPack<
      Atom, Side,
      std::remove_cvref_t<decltype(input_spec)>,
      std::remove_cvref_t<decltype(output_spec)>>{
          std::move(input_spec), std::move(output_spec)};
}

template <gemm::Atom Atom,
          tensor::InputOperand Input,
          tensor::OutputOperand Output,
          tensor::OutputOperand CompensationOutput>
VECOPS_INLINE auto make_matmul_pack_b_compensated(
    Input&& input, Output&& output, CompensationOutput&& compensation,
    int32_t a_zero_point) {
  using Element = typename gemm::packing_t<
      Atom, gemm::Operand::B>::Element;
  auto input_spec = tensor::as_input_spec<Element>(
      std::forward<Input>(input));
  auto output_spec = tensor::as_output_spec<Element>(
      std::forward<Output>(output));
  auto compensation_spec = tensor::as_output_spec<int32_t>(
      std::forward<CompensationOutput>(compensation));
  return MatmulPackBCompensated<
      Atom,
      std::remove_cvref_t<decltype(input_spec)>,
      std::remove_cvref_t<decltype(output_spec)>,
      std::remove_cvref_t<decltype(compensation_spec)>>{
          std::move(input_spec), std::move(output_spec),
          std::move(compensation_spec), a_zero_point};
}

template <gemm::Atom Atom, gemm::Operand Side,
          execution::ExecutionScope Scope,
          tensor::InputOperand Input, tensor::OutputOperand Output>
VECOPS_INLINE void matmul_pack(
    Scope& scope, Input&& input, Output&& output) {
  auto operation = make_matmul_pack<Atom, Side>(
      std::forward<Input>(input), std::forward<Output>(output));
  operation(scope);
}

template <gemm::Atom Atom, gemm::Operand Side,
          tensor::InputOperand Input, tensor::OutputOperand Output>
VECOPS_INLINE void matmul_pack(
    kernel::WorkspaceView& workspace, Input&& input, Output&& output) {
  ExecutionSession execution{workspace};
  matmul_pack<Atom, Side>(
      execution, std::forward<Input>(input), std::forward<Output>(output));
}

template <gemm::Atom Atom,
          execution::ExecutionScope Scope,
          tensor::InputOperand Input,
          tensor::OutputOperand Output,
          tensor::OutputOperand CompensationOutput>
VECOPS_INLINE void matmul_pack_b_compensated(
    Scope& scope, Input&& input, Output&& output,
    CompensationOutput&& compensation, int32_t a_zero_point) {
  auto operation = make_matmul_pack_b_compensated<Atom>(
      std::forward<Input>(input), std::forward<Output>(output),
      std::forward<CompensationOutput>(compensation), a_zero_point);
  operation(scope);
}

template <gemm::Atom Atom,
          tensor::InputOperand Input,
          tensor::OutputOperand Output,
          tensor::OutputOperand CompensationOutput>
VECOPS_INLINE void matmul_pack_b_compensated(
    kernel::WorkspaceView& workspace, Input&& input, Output&& output,
    CompensationOutput&& compensation, int32_t a_zero_point) {
  ExecutionSession execution{workspace};
  matmul_pack_b_compensated<Atom>(
      execution, std::forward<Input>(input), std::forward<Output>(output),
      std::forward<CompensationOutput>(compensation), a_zero_point);
}

} // namespace vecops::ops

#endif // VECOPS_MATMUL_MATMUL_PACK_H
