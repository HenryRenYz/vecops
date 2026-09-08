/**
 * @file Kernel.h
 * @brief Compact authoring surface for compiler-managed `__kernel__` files.
 */
#ifndef VECOPS_KERNEL_AUTHORING_H
#define VECOPS_KERNEL_AUTHORING_H

#include <initializer_list>

#include "vecops/Assertion.h"
#include "vecops/Meta.h"
#include "vecops/kernel/Workspace.h"
#include "vecops/tensor/Tensor.h"

namespace vecops {

using meta::cint;
using tensor::TensorLike;
using tensor::make_shape;
using tensor::make_strides;
using tensor::make_tensor;
using kernel::Workspace;
using kernel::WorkspaceView;

/** @brief Wildcard used by the runtime `assert_shape` authoring helper. */
struct AnyDimension {};
inline constexpr AnyDimension any{};

/** @brief One exact or wildcard runtime dimension assertion. */
class AssertDimension {
public:
  AssertDimension(AnyDimension)
    : any_(true) {
  }

  template <typename Value>
    requires(meta::ValueType<Value> || std::integral<std::remove_cvref_t<Value>>)
  AssertDimension(Value value)
    : value_(static_cast<nint_t>(value)) {
  }

  [[nodiscard]] bool accepts(nint_t value) const {
    return any_ || value == value_;
  }

private:
  bool any_ = false;
  nint_t value_ = 0;
};

/**
 * @brief Assert a tensor's runtime rank, shape, and strides.
 *
 * Generated adapters already encode the `KernelDef` metadata in the Tensor
 * type. This helper is an additional readable kernel-local contract and
 * supports the exact brace-list form used by source kernels.
 */
template <TensorLike Tensor>
inline void assert_shape(const Tensor& tensor, std::initializer_list<AssertDimension> shape,
                         std::initializer_list<AssertDimension> strides) {
  VECOPS_ASSERT(shape.size() == Tensor::Ndim, "assert_shape shape rank mismatch");
  VECOPS_ASSERT(strides.size() == Tensor::Ndim, "assert_shape stride rank mismatch");
  std::size_t axis = 0;
  for (const auto& expected : shape) {
    VECOPS_ASSERT(expected.accepts(tensor.size(static_cast<int>(axis))), "assert_shape size mismatch at axis %zu",
                  axis);
    ++axis;
  }
  axis = 0;
  for (const auto& expected : strides) {
    VECOPS_ASSERT(expected.accepts(tensor.stride(static_cast<int>(axis))), "assert_shape stride mismatch at axis %zu",
                  axis);
    ++axis;
  }
}

} // namespace vecops

#endif // VECOPS_KERNEL_AUTHORING_H
