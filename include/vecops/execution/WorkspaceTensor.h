// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
//
//

#ifndef VECOPS_EXECUTION_WORKSPACE_TENSOR_H
#define VECOPS_EXECUTION_WORKSPACE_TENSOR_H

/**
 * @file vecops/execution/WorkspaceTensor.h
 * @brief Typed tensor allocation helpers for WorkspaceContext.
 *
 * This header is an implementation boundary between tensor layouts and the
 * byte-oriented workspace planner. Kernel authors specify element type and
 * layout; storage-span calculation, overflow checks, and byte conversion stay
 * here. Returned tensors are non-owning views whose lifetime is the enclosing
 * WorkspaceContext scope.
 */

#include <algorithm>
#include <array>
#include <cstddef>
#include <limits>
#include <type_traits>
#include <utility>

#include "vecops/Assertion.h"
#include "vecops/execution/Parallel.h"
#include "vecops/execution/WorkspacePlan.h"
#include "vecops/tensor/Tensor.h"

namespace vecops::execution {

/** Placement and base-alignment policy for a typed workspace tensor. */
struct WorkspaceTensorOptions {
  nint_t alignment = vec::DEFAULT_ALIGNMENT;
  WorkspacePlacementPolicy placement = WorkspacePlacementPolicy::FastPreferred;
};

namespace workspace_tensor_details {

inline nint_t checked_add(nint_t left, nint_t right) {
  VECOPS_CHECK(left >= 0 && right >= 0, "workspace tensor sizes must be non-negative");
  VECOPS_CHECK(left <= std::numeric_limits<nint_t>::max() - right, "workspace tensor size addition overflow");
  return left + right;
}

inline nint_t checked_mul(nint_t left, nint_t right) {
  VECOPS_CHECK(left >= 0 && right >= 0, "workspace tensor sizes must be non-negative");
  VECOPS_CHECK(left == 0 || right <= std::numeric_limits<nint_t>::max() / left,
               "workspace tensor size multiplication overflow");
  return left * right;
}

inline void validate_options(const WorkspaceTensorOptions& options) {
  VECOPS_CHECK(options.alignment > 0 && (options.alignment & (options.alignment - 1)) == 0,
               "workspace tensor alignment must be a positive power of two");
}

/**
 * Return the number of elements covered by a writable layout.
 *
 * The current typed-workspace contract deliberately accepts only layouts that
 * can be proven non-overlapping by sorting non-trivial dimensions by stride.
 * This covers contiguous, padded, and permuted dense layouts. Negative strides
 * are rejected because WorkspaceContext returns the allocation base rather
 * than an offset logical-origin pointer.
 */
template <tensor::LayoutLike Layout>
nint_t writable_storage_span(const Layout& layout) {
  struct Axis {
    nint_t extent;
    nint_t stride;
  };

  std::array<Axis, Layout::Ndim> axes{};
  bool empty = false;
  for (int axis = 0; axis < Layout::Ndim; ++axis) {
    const nint_t extent = layout.shape()[axis];
    const nint_t stride = layout.strides()[axis];
    VECOPS_CHECK(extent >= 0, "workspace tensor extent must be non-negative");
    VECOPS_CHECK(stride >= 0, "workspace tensor does not support negative strides");
    axes[static_cast<std::size_t>(axis)] = {extent, stride};
    empty = empty || extent == 0;
  }
  if (empty)
    return 0;

  std::sort(axes.begin(), axes.end(), [](const Axis& left, const Axis& right) {
    // Unit dimensions never introduce overlap and are considered last so a
    // harmless zero stride on a singleton axis remains valid.
    if ((left.extent <= 1) != (right.extent <= 1))
      return left.extent > 1;
    return left.stride < right.stride;
  });

  nint_t span = 1;
  for (const auto [extent, stride] : axes) {
    if (extent <= 1)
      continue;
    VECOPS_CHECK(stride >= span, "workspace tensor layout is overlapping or cannot be proven non-overlapping");
    span = checked_add(span, checked_mul(extent - 1, stride));
  }
  return span;
}

template <typename T, tensor::LayoutLike Layout>
nint_t storage_bytes(const Layout& layout) {
  static_assert(!std::is_const_v<T>, "workspace tensors are writable and cannot have a const element type");
  static_assert(std::is_trivially_copyable_v<T>, "workspace tensor elements must be trivially copyable");
  static_assert((sizeof(T) & (sizeof(T) - 1)) == 0,
                "workspace tensor element sizes must be powers of two so replica strides remain element-aligned");
  return checked_mul(writable_storage_span(layout), static_cast<nint_t>(sizeof(T)));
}

template <typename T>
nint_t allocation_alignment(const WorkspaceTensorOptions& options) {
  validate_options(options);
  return std::max(options.alignment, static_cast<nint_t>(alignof(T)));
}

template <typename Replicas, tensor::LayoutLike Layout>
auto prepend_worker_layout(Replicas replicas, const Layout& inner, nint_t replica_stride_elements) {
  auto shape = tensor::details::insert_dim<0>(inner.shape(), replicas);
  auto strides = tensor::details::insert_dim<0>(inner.strides(), meta::Any{replica_stride_elements});
  return tensor::make_layout(std::move(shape), std::move(strides));
}

} // namespace workspace_tensor_details

template <nint_t Parallelism>
template <typename WorkerTensor>
constexpr decltype(auto) TaskContext<Parallelism>::local(WorkerTensor&& worker_tensor) const {
  using Tensor = std::remove_cvref_t<WorkerTensor>;
  static_assert(::vecops::tensor::BoundTensorLike<Tensor>,
                "TaskContext::local requires a bound worker tensor");
  static_assert(Tensor::Ndim >= 1,
                "TaskContext::local requires a leading worker-replica axis");
  using Replicas = std::remove_cvref_t<decltype(worker_tensor.template size<0>())>;
  if constexpr (Replicas::is_const) {
    static_assert(Replicas::value == Parallelism,
                  "worker tensor replica count must match TaskContext lane count");
  } else {
    VECOPS_CHECK(worker_tensor.template size<0>() == Parallelism,
                 "worker tensor replica count must match TaskContext lane count");
  }
  return std::forward<WorkerTensor>(worker_tensor)(lane_id(), ::vecops::tensor::ellipsis);
}

} // namespace vecops::execution

#endif // VECOPS_EXECUTION_WORKSPACE_TENSOR_H
