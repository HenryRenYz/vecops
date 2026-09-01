//
// Copyright (c) vecops contributors.
//

#ifndef VECOPS_MATMUL_DETAILS_PACK_OPERATION_H
#define VECOPS_MATMUL_DETAILS_PACK_OPERATION_H

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

template <::vecops::matmul::Atom AtomT, ::vecops::matmul::Operand SideV>
struct MatmulPackConfig {
  using Atom = AtomT;
  static constexpr ::vecops::matmul::Operand side = SideV;
};

template <::vecops::matmul::Atom AtomT>
struct MatmulPackBCompensatedConfig {
  using Atom = AtomT;
  int32_t a_zero_point = 0;
};

namespace matmul_pack_details {

template <::vecops::matmul::Atom Atom, ::vecops::matmul::Operand Side,
          typename InputSpec, typename OutputSpec>
using SelectedImplementation = std::conditional_t<
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

} // namespace matmul_pack_details

template <::vecops::matmul::Atom Atom, ::vecops::matmul::Operand Side,
          tensor::LayoutLike InputLayout>
VECOPS_INLINE auto matmul_packed_layout(const InputLayout& input_layout) {
  return ::vecops::matmul::packed_layout<Atom, Side>(input_layout);
}

template <::vecops::matmul::Atom Atom, ::vecops::matmul::Operand Side,
          typename InputSpec, typename OutputSpec>
class PreparedMatmulPack {
  using Packing = ::vecops::matmul::packing_t<Atom, Side>;

public:
  using ComputeType = typename Packing::Element;
  using Implementation = matmul_pack_details::SelectedImplementation<
      Atom, Side, InputSpec, OutputSpec>;
  using ResourceRequirements =
      kernel::matmul_pack_implementation::resource_requirements_t<
          Atom, Side, Implementation>;

  VECOPS_INLINE PreparedMatmulPack(InputSpec input, OutputSpec output)
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
        (::vecops::matmul::is_corresponding_packed_layout<Atom, Side>(
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
template <::vecops::matmul::Atom Atom,
          typename InputSpec, typename OutputSpec,
          typename CompensationOutputSpec>
class PreparedMatmulPackBCompensated {
  using Packing = ::vecops::matmul::packing_t<Atom, ::vecops::matmul::Operand::B>;

public:
  using ComputeType = typename Packing::Element;
  using Implementation = matmul_pack_details::SelectedImplementation<
      Atom, ::vecops::matmul::Operand::B, InputSpec, OutputSpec>;
  using ResourceRequirements =
      kernel::matmul_pack_implementation::resource_requirements_t<
          Atom, ::vecops::matmul::Operand::B, Implementation>;

  VECOPS_INLINE PreparedMatmulPackBCompensated(
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
        (::vecops::matmul::is_corresponding_packed_layout<
            Atom, ::vecops::matmul::Operand::B>(input_layout, output_.output_layout())),
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

namespace matmul_pack_details {

template <::vecops::matmul::Atom Atom, ::vecops::matmul::Operand Side,
          tensor::InputOperand Input, tensor::OutputOperand Output>
VECOPS_INLINE auto prepare_matmul_pack(Input&& input, Output&& output) {
  using Element = typename ::vecops::matmul::packing_t<Atom, Side>::Element;
  auto input_spec = tensor::as_input_spec<Element>(
      std::forward<Input>(input));
  auto output_spec = tensor::as_output_spec<Element>(
      std::forward<Output>(output));
  return PreparedMatmulPack<
      Atom, Side,
      std::remove_cvref_t<decltype(input_spec)>,
      std::remove_cvref_t<decltype(output_spec)>>{
          std::move(input_spec), std::move(output_spec)};
}

template <::vecops::matmul::Atom Atom, ::vecops::matmul::Operand Side,
          execution::ExecutionScope Scope,
          tensor::InputOperand Input, tensor::OutputOperand Output>
VECOPS_INLINE void run_matmul_pack(
    Scope& scope, Input&& input, Output&& output) {
  auto prepared = prepare_matmul_pack<Atom, Side>(
      std::forward<Input>(input), std::forward<Output>(output));
  prepared(scope);
}

template <::vecops::matmul::Atom Atom,
          tensor::InputOperand Input,
          tensor::OutputOperand Output,
          tensor::OutputOperand CompensationOutput>
VECOPS_INLINE auto prepare_matmul_pack_b_compensated(
    Input&& input, Output&& output, CompensationOutput&& compensation,
    int32_t a_zero_point) {
  using Element = typename ::vecops::matmul::packing_t<
      Atom, ::vecops::matmul::Operand::B>::Element;
  auto input_spec = tensor::as_input_spec<Element>(
      std::forward<Input>(input));
  auto output_spec = tensor::as_output_spec<Element>(
      std::forward<Output>(output));
  auto compensation_spec = tensor::as_output_spec<int32_t>(
      std::forward<CompensationOutput>(compensation));
  return PreparedMatmulPackBCompensated<
      Atom,
      std::remove_cvref_t<decltype(input_spec)>,
      std::remove_cvref_t<decltype(output_spec)>,
      std::remove_cvref_t<decltype(compensation_spec)>>{
          std::move(input_spec), std::move(output_spec),
          std::move(compensation_spec), a_zero_point};
}

template <::vecops::matmul::Atom Atom,
          execution::ExecutionScope Scope,
          tensor::InputOperand Input,
          tensor::OutputOperand Output,
          tensor::OutputOperand CompensationOutput>
VECOPS_INLINE void run_matmul_pack_b_compensated(
    Scope& scope, Input&& input, Output&& output,
    CompensationOutput&& compensation, int32_t a_zero_point) {
  auto prepared = prepare_matmul_pack_b_compensated<Atom>(
      std::forward<Input>(input), std::forward<Output>(output),
      std::forward<CompensationOutput>(compensation), a_zero_point);
  prepared(scope);
}

} // namespace matmul_pack_details

template <typename Config>
class MatmulPack {
public:
  const Config config;

  VECOPS_INLINE constexpr explicit MatmulPack(Config cfg = {})
      : config(std::move(cfg)) {}

  template <tensor::InputOperand Input, tensor::OutputOperand Output>
  VECOPS_INLINE nint_t required_workspace(const Input&, const Output&) const {
    return 0;
  }

  template <execution::ExecutionScope Scope,
            tensor::InputOperand Input, tensor::OutputOperand Output>
  VECOPS_INLINE void operator()(
      Scope& scope, Input&& input, Output&& output) const {
    auto prepared = matmul_pack_details::prepare_matmul_pack<
        typename Config::Atom, Config::side>(
            std::forward<Input>(input), std::forward<Output>(output));
    prepared(scope);
  }

  template <tensor::InputOperand Input, tensor::OutputOperand Output>
  VECOPS_INLINE void operator()(
      kernel::WorkspaceView& workspace, Input&& input, Output&& output) const {
    ExecutionSession execution{workspace};
    (*this)(execution, std::forward<Input>(input),
            std::forward<Output>(output));
  }
};

template <typename Config>
class MatmulPackBCompensated {
public:
  const Config config;

  VECOPS_INLINE constexpr explicit MatmulPackBCompensated(Config cfg = {})
      : config(std::move(cfg)) {}

  template <tensor::InputOperand Input, tensor::OutputOperand Output,
            tensor::OutputOperand CompensationOutput>
  VECOPS_INLINE nint_t required_workspace(
      const Input&, const Output&, const CompensationOutput&) const {
    return 0;
  }

  template <execution::ExecutionScope Scope,
            tensor::InputOperand Input, tensor::OutputOperand Output,
            tensor::OutputOperand CompensationOutput>
  VECOPS_INLINE void operator()(
      Scope& scope, Input&& input, Output&& output,
      CompensationOutput&& compensation) const {
    auto prepared = matmul_pack_details::prepare_matmul_pack_b_compensated<
        typename Config::Atom>(
            std::forward<Input>(input), std::forward<Output>(output),
            std::forward<CompensationOutput>(compensation),
            config.a_zero_point);
    prepared(scope);
  }

  template <tensor::InputOperand Input, tensor::OutputOperand Output,
            tensor::OutputOperand CompensationOutput>
  VECOPS_INLINE void operator()(
      kernel::WorkspaceView& workspace, Input&& input, Output&& output,
      CompensationOutput&& compensation) const {
    ExecutionSession execution{workspace};
    (*this)(execution, std::forward<Input>(input),
            std::forward<Output>(output),
            std::forward<CompensationOutput>(compensation));
  }
};

template <typename Config>
VECOPS_INLINE constexpr auto matmul_pack(Config config) {
  return MatmulPack<Config>{std::move(config)};
}

template <typename Config>
VECOPS_INLINE constexpr auto matmul_pack_b_compensated(Config config) {
  return MatmulPackBCompensated<Config>{std::move(config)};
}

} // namespace vecops::ops

#endif // VECOPS_MATMUL_DETAILS_PACK_OPERATION_H
