//
// Copyright (c) vecops contributors.
//

#ifndef VECOPS_OPS_DETAILS_TRANSPOSE_SELECTION_H
#define VECOPS_OPS_DETAILS_TRANSPOSE_SELECTION_H

#include <type_traits>

#include "vecops/kernel/details/transpose/Types.h"
#include "vecops/tensor/DataAccess.h"

/**
 * @file Selection.h
 * @brief Operation-level transpose implementation selection from operand types.
 *
 * This file intentionally lives in ops rather than kernel details: eligibility
 * depends on Tensor/Spec/Transform types and therefore includes DataAccess.
 * Moving it into the kernel backend would create the dependency cycle
 * `Backend -> DataAccess -> kernel/Transpose2D -> Backend`.
 */

namespace vecops::ops::transpose_details {

/**
 * @brief Whether a prepared operation can use the SME transpose backend.
 *
 * Selection is entirely type-driven: the operand category, rank, transform,
 * raw no-transform access, policy, and unit inner strides must all be known at
 * compile time. Source/destination memory dtypes may differ from ComputeType:
 * the SME leaf performs conversion at its load and/or store boundary. A
 * runtime-contiguous tensor whose type does not prove these properties remains
 * on the vector implementation.
 */
template <typename Input, typename Output,
          typename InputSpec, typename OutputSpec, typename Policy>
inline constexpr bool use_sme_v = [] {
#if defined(HAS_SME_FA64)
  if constexpr (
      InputSpec::InputTensor::Ndim != 2 ||
      OutputSpec::OutputTensor::Ndim != 2) {
    return false;
  } else if constexpr (
      std::same_as<Policy, kernel::transpose2d_policy::Automatic> &&
      tensor::is_tensor_v<std::remove_cvref_t<Input>> &&
      tensor::is_tensor_v<std::remove_cvref_t<Output>> &&
      std::same_as<typename InputSpec::TransformType, tensor::NoTransform> &&
      std::same_as<typename OutputSpec::TransformType, tensor::NoTransform> &&
      vec::Element<std::remove_cv_t<typename InputSpec::MemoryElement>> &&
      vec::Element<std::remove_cv_t<typename OutputSpec::MemoryElement>> &&
      std::same_as<
          tensor::stride_type_t<1, typename InputSpec::InputLayout>,
          meta::Const<1>> &&
      std::same_as<
          tensor::stride_type_t<1, typename OutputSpec::OutputLayout>,
          meta::Const<1>>) {
    using InputMemory =
        std::remove_cv_t<typename InputSpec::MemoryElement>;
    using OutputMemory =
        std::remove_cv_t<typename OutputSpec::MemoryElement>;
    using Compute = typename InputSpec::ComputeType;
    using M = tensor::size_type_t<0, typename InputSpec::InputLayout>;
    using N = tensor::size_type_t<1, typename InputSpec::InputLayout>;
    constexpr bool Converts =
        !std::same_as<InputMemory, Compute> ||
        !std::same_as<OutputMemory, Compute>;
    if constexpr (Converts && M::is_const && N::is_const) {
      constexpr nint_t Elements = M::value * N::value;
      // Streaming/ZA setup dominates tiny conversions. F64-source conversion
      // also needs a somewhat larger tile population to amortize its two-step
      // narrow path. Runtime extents stay SME-eligible so large Dynamic/Any
      // tensors retain the measured multi-x acceleration.
      if constexpr (Elements <= 256) return false;
      else if constexpr (sizeof(InputMemory) == 8 && Elements <= 8192)
        return false;
    }
    return true;
  }
#endif
  return false;
}();

/** Implementation tag selected for a fully typed transpose operation. */
template <typename Input, typename Output,
          typename InputSpec, typename OutputSpec, typename Policy>
using SelectedImplementation = std::conditional_t<
    use_sme_v<Input, Output, InputSpec, OutputSpec, Policy>,
    kernel::transpose2d_implementation::SME,
    kernel::transpose2d_implementation::Vector>;

} // namespace vecops::ops::transpose_details

#endif // VECOPS_OPS_DETAILS_TRANSPOSE_SELECTION_H
