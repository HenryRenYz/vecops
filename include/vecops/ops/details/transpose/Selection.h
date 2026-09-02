#ifndef VECOPS_OPS_DETAILS_TRANSPOSE_SELECTION_H
#define VECOPS_OPS_DETAILS_TRANSPOSE_SELECTION_H

#include <type_traits>

#include "vecops/kernel/Transpose2D.h"
#include "vecops/tensor/DataAccess.h"

namespace vecops::ops::transpose_details {

/** Select SME only when operand and layout types prove it is applicable. */
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
      if constexpr (Elements <= 256) return false;
      else if constexpr (sizeof(InputMemory) == 8 && Elements <= 8192)
        return false;
    }
    return true;
  }
#endif
  return false;
}();

} // namespace vecops::ops::transpose_details

#endif // VECOPS_OPS_DETAILS_TRANSPOSE_SELECTION_H
