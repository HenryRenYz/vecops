//
// Copyright (c) vecops contributors.
//

#ifndef VECOPS_OPS_MATMUL_PACK_H
#define VECOPS_OPS_MATMUL_PACK_H

#include <type_traits>
#include <utility>

#include "vecops/Assertion.h"
#include "vecops/execution/ExecutionSession.h"
#include "vecops/gemm/Packing.h"
#include "vecops/kernel/MatmulPack.h"
#include "vecops/ops/details/matmul_pack/Selection.h"
#include "vecops/tensor/DataAccess.h"
#include "vecops/util/Math.h"

namespace vecops::ops {

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
        *this, [&](auto& active) VECOPS_INLINE_LAMBDA { execute(active); });
  }

  VECOPS_INLINE void operator()(kernel::WorkspaceView& workspace) const {
    ExecutionSession execution{workspace};
    (*this)(execution);
  }

private:
  VECOPS_INLINE void validate() const {
    VECOPS_ASSERT(input_.input_layout().shape()[0] >= 0 &&
                      input_.input_layout().shape()[1] >= 0,
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

} // namespace vecops::ops

#endif // VECOPS_OPS_MATMUL_PACK_H
