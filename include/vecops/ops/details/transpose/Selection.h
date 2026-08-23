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
 * memory/compute dtype equality, policy, and unit inner strides must all be
 * known at compile time. A runtime-contiguous tensor whose type does not prove
 * these properties deliberately remains on the vector implementation.
 */
template <typename Input, typename Output,
          typename InputSpec, typename OutputSpec, typename Policy>
inline constexpr bool use_sme_v = [] {
#if defined(HAS_SME)
  if constexpr (
      InputSpec::InputTensor::Ndim != 2 ||
      OutputSpec::OutputTensor::Ndim != 2) {
    return false;
  } else if constexpr (
      std::same_as<Policy, kernel::transpose2d_policy::Automatic> &&
      tensor::is_tensor<std::remove_cvref_t<Input>> &&
      tensor::is_tensor<std::remove_cvref_t<Output>> &&
      std::same_as<typename InputSpec::TransformType, tensor::NoTransform> &&
      std::same_as<typename OutputSpec::TransformType, tensor::NoTransform> &&
      std::same_as<typename InputSpec::MemoryElement,
                   typename InputSpec::ComputeType> &&
      std::same_as<typename OutputSpec::MemoryElement,
                   typename OutputSpec::ComputeType> &&
      std::same_as<
          tensor::stride_type_t<1, typename InputSpec::InputLayout>,
          meta::Const<1>> &&
      std::same_as<
          tensor::stride_type_t<1, typename OutputSpec::OutputLayout>,
          meta::Const<1>>) {
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
