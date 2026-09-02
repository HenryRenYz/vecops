//
// Copyright (c) vecops contributors.
//

#ifndef VECOPS_MATMUL_DETAILS_FAMILIES_ARCHITECTURE_BATCH_PLANNER_H
#define VECOPS_MATMUL_DETAILS_FAMILIES_ARCHITECTURE_BATCH_PLANNER_H

#include "vecops/Meta.h"
#include "vecops/tensor/DataAccess.h"

namespace vecops::matmul::details {

/** Rank-three layout analysis and rank-two view construction. */
template <typename Invocation>
struct ArchitectureBatchPlanner {
  VECOPS_ALWAYS_INLINE static bool rows_flatten_enabled(const Invocation& op) {
    if constexpr (!Invocation::BatchRowsFlattenCandidate) {
      return false;
    } else {
      const auto& a_layout = op.a_.input_layout();
      const auto& ci_layout = op.c_input_.input_layout();
      const auto& co_layout = op.c_output_.output_layout();
      const nint_t batch = static_cast<nint_t>(
          tensor::size_value<0>(co_layout));
      const nint_t m = static_cast<nint_t>(op.m_);
      const nint_t n = static_cast<nint_t>(op.n_);
      const nint_t k = static_cast<nint_t>(op.k_);
      if (batch <= 1 || m <= 0 || n <= 0 || k <= 0 || m > 16)
        return false;
      if (batch > 64 / m) return false;
      const nint_t flat_m = batch * m;
      if (flat_m > 64) return false;

      const bool dense_a =
          static_cast<nint_t>(tensor::stride_value<2>(a_layout)) == 1 &&
          static_cast<nint_t>(tensor::stride_value<1>(a_layout)) == k &&
          static_cast<nint_t>(tensor::stride_value<0>(a_layout)) == m * k;
      const bool dense_co =
          static_cast<nint_t>(tensor::stride_value<2>(co_layout)) == 1 &&
          static_cast<nint_t>(tensor::stride_value<1>(co_layout)) == n &&
          static_cast<nint_t>(tensor::stride_value<0>(co_layout)) == m * n;
      const bool dense_ci =
          static_cast<nint_t>(tensor::stride_value<2>(ci_layout)) == 1 &&
          static_cast<nint_t>(tensor::stride_value<1>(ci_layout)) == n &&
          static_cast<nint_t>(tensor::stride_value<0>(ci_layout)) == m * n;
      const bool broadcast_ci =
          static_cast<nint_t>(tensor::stride_value<2>(ci_layout)) == 1 &&
          static_cast<nint_t>(tensor::stride_value<1>(ci_layout)) == 0 &&
          static_cast<nint_t>(tensor::stride_value<0>(ci_layout)) == 0;
      return dense_a && dense_co && (dense_ci || broadcast_ci);
    }
  }

  VECOPS_ALWAYS_INLINE static bool columns_flatten_enabled(const Invocation& op) {
    if constexpr (!Invocation::BatchColumnsFlattenCandidate) {
      return false;
    } else {
      const auto& a_layout = op.a_.input_layout();
      const auto& b_layout = op.b_.input_layout();
      const auto& ci_layout = op.c_input_.input_layout();
      const auto& co_layout = op.c_output_.output_layout();
      const nint_t batch = static_cast<nint_t>(
          tensor::size_value<0>(co_layout));
      const nint_t m = static_cast<nint_t>(op.m_);
      const nint_t n = static_cast<nint_t>(op.n_);
      const nint_t k = static_cast<nint_t>(op.k_);
      if (batch < 4 || m != 1 || n <= 0 || k <= 0) return false;
      if (n > 64 / batch) return false;

      const bool shared_a = [&] {
        if constexpr (Invocation::PackedAInput) {
          return true;
        } else {
          return static_cast<nint_t>(
                     tensor::stride_value<2>(a_layout)) == 1 &&
              static_cast<nint_t>(
                  tensor::stride_value<1>(a_layout)) == k &&
              static_cast<nint_t>(
                  tensor::stride_value<0>(a_layout)) == 0;
        }
      }();
      const nint_t b_row_stride = static_cast<nint_t>(
          tensor::stride_value<1>(b_layout));
      const bool mergeable_b =
          static_cast<nint_t>(tensor::stride_value<2>(b_layout)) == 1 &&
          b_row_stride >= k &&
          static_cast<nint_t>(tensor::stride_value<0>(b_layout)) ==
              n * b_row_stride;
      const bool dense_co =
          static_cast<nint_t>(tensor::stride_value<2>(co_layout)) == 1 &&
          static_cast<nint_t>(tensor::stride_value<0>(co_layout)) == n;
      const bool dense_ci =
          static_cast<nint_t>(tensor::stride_value<2>(ci_layout)) == 1 &&
          static_cast<nint_t>(tensor::stride_value<0>(ci_layout)) == n;
      const bool periodic_ci =
          Invocation::BatchColumnsPeriodicCInputCandidate &&
          static_cast<nint_t>(tensor::stride_value<2>(ci_layout)) == 1 &&
          static_cast<nint_t>(tensor::stride_value<1>(ci_layout)) == 0 &&
          static_cast<nint_t>(tensor::stride_value<0>(ci_layout)) == 0;
      return shared_a && mergeable_b && dense_co &&
          (dense_ci || periodic_ci);
    }
  }

  VECOPS_ALWAYS_INLINE static bool periodic_c_input_enabled(const Invocation& op) {
    if constexpr (!Invocation::BatchColumnsPeriodicCInputCandidate) {
      return false;
    } else {
      if (!columns_flatten_enabled(op)) return false;
      const auto& layout = op.c_input_.input_layout();
      return static_cast<nint_t>(tensor::stride_value<2>(layout)) == 1 &&
          static_cast<nint_t>(tensor::stride_value<1>(layout)) == 0 &&
          static_cast<nint_t>(tensor::stride_value<0>(layout)) == 0;
    }
  }

  VECOPS_ALWAYS_INLINE static nint_t periodic_c_input_bytes(const Invocation& op) {
    if constexpr (!Invocation::BatchColumnsPeriodicCInputCandidate) {
      return 0;
    } else {
      const nint_t batch = static_cast<nint_t>(tensor::size_value<0>(
          op.c_output_.output_layout()));
      return batch * static_cast<nint_t>(op.n_) *
          static_cast<nint_t>(sizeof(typename Invocation::AtomType::TAcc)) +
          63;
    }
  }

  template <typename Spec,
            meta::ValueType Rows, meta::ValueType Columns,
            meta::ValueType RowStride>
  VECOPS_ALWAYS_INLINE static auto flatten_input(
      const Spec& spec, Rows flat_rows, Columns columns,
      RowStride row_stride) {
    auto tensor = tensor::make_tensor(
        spec.tensor().data(), tensor::make_layout(
            tensor::make_shape(flat_rows, columns),
            tensor::make_strides(row_stride, meta::cint<1>)));
    return tensor::InputSpec<
        typename Spec::ComputeType, decltype(tensor),
        typename Spec::TransformType>{tensor, spec.transform()};
  }

  template <typename Spec,
            meta::ValueType Rows, meta::ValueType Columns,
            meta::ValueType RowStride>
  VECOPS_ALWAYS_INLINE static auto flatten_output(
      const Spec& spec, Rows flat_rows, Columns columns,
      RowStride row_stride) {
    auto tensor = tensor::make_tensor(
        spec.tensor().data(), tensor::make_layout(
            tensor::make_shape(flat_rows, columns),
            tensor::make_strides(row_stride, meta::cint<1>)));
    return tensor::OutputSpec<
        typename Spec::ComputeType, decltype(tensor),
        typename Spec::TransformType>{tensor, spec.transform()};
  }

  VECOPS_ALWAYS_INLINE static auto shared_b_leaf(const Invocation& op) {
    if constexpr (Invocation::PackedBInput) return op.b_;
    else return tensor::slice_view<0>(op.b_, 0);
  }

  VECOPS_ALWAYS_INLINE static auto shared_a_leaf(const Invocation& op) {
    if constexpr (Invocation::PackedAInput) return op.a_;
    else return tensor::slice_view<0>(op.a_, 0);
  }
};

} // namespace vecops::matmul::details

#endif // VECOPS_MATMUL_DETAILS_FAMILIES_ARCHITECTURE_BATCH_PLANNER_H
