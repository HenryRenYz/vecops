/**
 * @file Argument.cpp
 * @brief Construction of scalar values and transient C ABI call frames.
 */

#include "vecops/runtime/Argument.h"

#include <type_traits>

namespace vecops::runtime {

Scalar Scalar::signed_integer(std::int64_t value, DType dtype) {
  Scalar result;
  result.dtype = dtype;
  result.value.i64 = value;
  return result;
}

Scalar Scalar::unsigned_integer(std::uint64_t value, DType dtype) {
  Scalar result;
  result.dtype = dtype;
  result.value.u64 = value;
  return result;
}

Scalar Scalar::floating(double value, DType dtype) {
  Scalar result;
  result.dtype = dtype;
  result.value.f64 = value;
  return result;
}

CallFrame::CallFrame(const ArgumentMetadata& arguments, void* workspace, std::uint64_t workspace_size,
                     const VecopsExecutionContext* context) {
  values_.reserve(arguments.size());
  for (const auto& value : arguments.values()) {
    // Copy outer ABI records into this frame.  Tensor shape/stride arrays stay
    // in ArgumentMetadata, whose documented lifetime must cover this frame.
    VecopsValue abi_value{};
    abi_value.struct_size = sizeof(VecopsValue);
    if (std::holds_alternative<std::monostate>(value)) {
      abi_value.kind = VECOPS_VALUE_NONE;
    } else if (const auto* tensor = std::get_if<TensorView>(&value)) {
      abi_value.kind = VECOPS_VALUE_TENSOR;
      abi_value.value.tensor = VecopsTensorView{
        .struct_size = sizeof(VecopsTensorView),
        .data = tensor->data,
        .byte_offset = tensor->byte_offset,
        .device_type = tensor->device_type,
        .device_index = tensor->device_index,
        .dtype = static_cast<std::uint32_t>(tensor->dtype),
        .rank = static_cast<std::uint32_t>(tensor->sizes.size()),
        .sizes = tensor->sizes.data(),
        .strides = tensor->strides.data(),
        .flags = tensor->flags,
      };
    } else {
      const auto& scalar = std::get<Scalar>(value);
      abi_value.kind = VECOPS_VALUE_SCALAR;
      abi_value.value.scalar = VecopsScalar{
        .struct_size = sizeof(VecopsScalar),
        .dtype = static_cast<std::uint32_t>(scalar.dtype),
        .value = scalar.value,
      };
    }
    values_.push_back(abi_value);
  }
  call_ = VecopsCall{
    .struct_size = sizeof(VecopsCall),
    .num_values = static_cast<std::uint32_t>(values_.size()),
    .values = values_.data(),
    .workspace = workspace,
    .workspace_size = workspace_size,
    .context = context,
  };
}

} // namespace vecops::runtime
