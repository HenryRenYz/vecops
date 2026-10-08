// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
//
//

#ifndef VECOPS_MATMUL_DETAILS_FAMILIES_ARCHITECTURE_BATCH_PLANNER_H
#define VECOPS_MATMUL_DETAILS_FAMILIES_ARCHITECTURE_BATCH_PLANNER_H

/**
 * @file vecops/matmul/details/planning/families/ArchitectureBatchPlanner.h
 * @brief Rank-three batch analysis: runtime flatten gates and rank-two view
 *        construction for architecture-family invocations.
 *
 * The invocation decides statically which flatten shapes could apply (its
 * *Candidate constants); this planner re-checks those candidates against
 * the actual runtime layouts, because flatten legality ultimately depends
 * on operand strides.  It also rebuilds rank-two Input/OutputSpec views
 * over the same storage and extracts the shared (batch-broadcast) A/B
 * leaves the flattened paths run with.
 */

#include "vecops/Meta.h"
#include "vecops/tensor/DataAccess.h"

namespace vecops::matmul::details {

/** Rank-three layout analysis and rank-two view construction. */
template <typename Invocation>
struct ArchitectureBatchPlanner {
  /// Whether the batch-rows flatten (dense [batch, M, K] activations with a
  /// shared B may run as one [batch*M, K] rank-two product) is legal for
  /// this layout.
  VECOPS_ALWAYS_INLINE static bool rows_flatten_enabled(const Invocation& op) {
    if constexpr (!Invocation::BatchRowsFlattenCandidate) {
      return false;
    } else {
      const auto& a_layout = op.a_.input_layout();
      const auto& ci_layout = op.c_input_.input_layout();
      const auto& co_layout = op.c_output_.output_layout();
      const nint_t batch = static_cast<nint_t>(
          tensor::size<0>(co_layout));
      const nint_t m = static_cast<nint_t>(op.m_);
      const nint_t n = static_cast<nint_t>(op.n_);
      const nint_t k = static_cast<nint_t>(op.k_);
      // Flattening targets skinny decode-style batches: every item must fit
      // one resident tile row block (AMX tiles are 16 rows tall), and the
      // merged height stays within 64 rows — a handful of tiles — so the
      // merged problem remains the cheap skinny case rather than a
      // full-size product.  Both bounds below enforce flat_m <= 64.
      if (batch <= 1 || m <= 0 || n <= 0 || k <= 0 || m > 16)
        return false;
      if (batch > 64 / m) return false;
      const nint_t flat_m = batch * m;
      if (flat_m > 64) return false;

      // The flatten reinterprets storage as a row-major rank-two matrix:
      // A must be a dense [batch, m, k] block (strides {1, k, m*k}) so rows
      // of different batches are physically contiguous, and C likewise
      // ({1, n, m*n}).  A broadcast C input ({1, 0, 0}) is one row vector
      // repeated over the whole batch/row space; it stays correct under the
      // flatten because every flat row maps to that same vector.
      const bool dense_a =
          static_cast<nint_t>(tensor::stride<2>(a_layout)) == 1 &&
          static_cast<nint_t>(tensor::stride<1>(a_layout)) == k &&
          static_cast<nint_t>(tensor::stride<0>(a_layout)) == m * k;
      const bool dense_co =
          static_cast<nint_t>(tensor::stride<2>(co_layout)) == 1 &&
          static_cast<nint_t>(tensor::stride<1>(co_layout)) == n &&
          static_cast<nint_t>(tensor::stride<0>(co_layout)) == m * n;
      const bool dense_ci =
          static_cast<nint_t>(tensor::stride<2>(ci_layout)) == 1 &&
          static_cast<nint_t>(tensor::stride<1>(ci_layout)) == n &&
          static_cast<nint_t>(tensor::stride<0>(ci_layout)) == m * n;
      const bool broadcast_ci =
          static_cast<nint_t>(tensor::stride<2>(ci_layout)) == 1 &&
          static_cast<nint_t>(tensor::stride<1>(ci_layout)) == 0 &&
          static_cast<nint_t>(tensor::stride<0>(ci_layout)) == 0;
      return dense_a && dense_co && (dense_ci || broadcast_ci);
    }
  }

  /// Whether the batch-columns flatten (a shared A with M == 1 may run as
  /// one [1, batch*N] rank-two product) is legal for this layout.
  VECOPS_ALWAYS_INLINE static bool columns_flatten_enabled(const Invocation& op) {
    if constexpr (!Invocation::BatchColumnsFlattenCandidate) {
      return false;
    } else {
      const auto& a_layout = op.a_.input_layout();
      const auto& b_layout = op.b_.input_layout();
      const auto& ci_layout = op.c_input_.input_layout();
      const auto& co_layout = op.c_output_.output_layout();
      const nint_t batch = static_cast<nint_t>(
          tensor::size<0>(co_layout));
      const nint_t m = static_cast<nint_t>(op.m_);
      const nint_t n = static_cast<nint_t>(op.n_);
      const nint_t k = static_cast<nint_t>(op.k_);
      // M == 1 is what makes the physical orders match (see
      // BatchColumnsFlattenCandidate); a batch count below four does not
      // amortize the path switch, and the merged width is capped at 64
      // columns, symmetric to the rows flatten.
      if (batch < 4 || m != 1 || n <= 0 || k <= 0) return false;
      if (n > 64 / batch) return false;

      // shared_a: broadcast along batch ({1, k, 0}) or packed (packed
      // layouts carry no batch dimension at all).  mergeable_b: batch items
      // must be physically concatenatable — [batch, n, k] with batch stride
      // n * row_stride, where any per-row padding is still allowed.
      const bool shared_a = [&] {
        if constexpr (Invocation::PackedAInput) {
          return true;
        } else {
          return static_cast<nint_t>(
                     tensor::stride<2>(a_layout)) == 1 &&
              static_cast<nint_t>(
                  tensor::stride<1>(a_layout)) == k &&
              static_cast<nint_t>(
                  tensor::stride<0>(a_layout)) == 0;
        }
      }();
      const nint_t b_row_stride = static_cast<nint_t>(
          tensor::stride<1>(b_layout));
      const bool mergeable_b =
          static_cast<nint_t>(tensor::stride<2>(b_layout)) == 1 &&
          b_row_stride >= k &&
          static_cast<nint_t>(tensor::stride<0>(b_layout)) ==
              n * b_row_stride;
      // dense_co/dense_ci: C's batch stride must equal n so flat columns of
      // different batches are contiguous ({1, ?, n}).  periodic_ci: a
      // per-column C input broadcast over batch and row ({1, 0, 0}), which
      // the flattened run materializes instead (see periodic_c_input_*).
      const bool dense_co =
          static_cast<nint_t>(tensor::stride<2>(co_layout)) == 1 &&
          static_cast<nint_t>(tensor::stride<0>(co_layout)) == n;
      const bool dense_ci =
          static_cast<nint_t>(tensor::stride<2>(ci_layout)) == 1 &&
          static_cast<nint_t>(tensor::stride<0>(ci_layout)) == n;
      const bool periodic_ci =
          Invocation::BatchColumnsPeriodicCInputCandidate &&
          static_cast<nint_t>(tensor::stride<2>(ci_layout)) == 1 &&
          static_cast<nint_t>(tensor::stride<1>(ci_layout)) == 0 &&
          static_cast<nint_t>(tensor::stride<0>(ci_layout)) == 0;
      return shared_a && mergeable_b && dense_co &&
          (dense_ci || periodic_ci);
    }
  }

  /// Whether the flattened-columns run will materialize its periodic C
  /// input into a dense flat buffer (Layout cannot encode the required
  /// non-affine `flat_column % N` addressing).
  VECOPS_ALWAYS_INLINE static bool periodic_c_input_enabled(const Invocation& op) {
    if constexpr (!Invocation::BatchColumnsPeriodicCInputCandidate) {
      return false;
    } else {
      if (!columns_flatten_enabled(op)) return false;
      const auto& layout = op.c_input_.input_layout();
      return static_cast<nint_t>(tensor::stride<2>(layout)) == 1 &&
          static_cast<nint_t>(tensor::stride<1>(layout)) == 0 &&
          static_cast<nint_t>(tensor::stride<0>(layout)) == 0;
    }
  }

  /// Bytes for the dense [1, batch*N] C-input copy, rounded up to the
  /// 64-byte allocation granularity the run will use.
  VECOPS_ALWAYS_INLINE static nint_t periodic_c_input_bytes(const Invocation& op) {
    if constexpr (!Invocation::BatchColumnsPeriodicCInputCandidate) {
      return 0;
    } else {
      const nint_t batch = static_cast<nint_t>(tensor::size<0>(
          op.c_output_.output_layout()));
      return batch * static_cast<nint_t>(op.n_) *
          static_cast<nint_t>(sizeof(typename Invocation::AtomType::TAcc)) +
          63;
    }
  }

  /// Reinterpret the spec's storage as a rank-two [rows, columns] input
  /// with an explicit row stride.  ComputeType and TransformType are passed
  /// through unchanged so the flattened view keeps the original compute
  /// path; the caller picks the row stride from the source layout (0 for a
  /// broadcast C input is valid).
  template <typename Spec,
            meta::ValueType Rows, meta::ValueType Columns,
            meta::ValueType RowStride>
  VECOPS_ALWAYS_INLINE static auto flatten_input(
      const Spec& spec, Rows flat_rows, Columns columns,
      RowStride row_stride) {
    auto layout = tensor::make_layout(
        tensor::make_shape(flat_rows, columns),
        tensor::make_strides(row_stride, meta::cint<1>));
    auto view = [&]() VECOPS_INLINE_LAMBDA {
      using Element = typename Spec::InputTensor::ElementType;
      if constexpr (Spec::is_bound)
        return tensor::make_tensor(spec.tensor().data(), layout);
      else
        return tensor::make_unbound_tensor<Element>(layout);
    }();
    return spec.with_view_tensor_and_projection(
        std::move(view), tensor::identity_projection<decltype(view)::Ndim>());
  }

  /// Output-side twin of flatten_input().
  template <typename Spec,
            meta::ValueType Rows, meta::ValueType Columns,
            meta::ValueType RowStride>
  VECOPS_ALWAYS_INLINE static auto flatten_output(
      const Spec& spec, Rows flat_rows, Columns columns,
      RowStride row_stride) {
    auto layout = tensor::make_layout(
        tensor::make_shape(flat_rows, columns),
        tensor::make_strides(row_stride, meta::cint<1>));
    auto view = [&]() VECOPS_INLINE_LAMBDA {
      using Element = typename Spec::OutputTensor::ElementType;
      if constexpr (Spec::is_bound)
        return tensor::make_tensor(spec.tensor().data(), layout);
      else
        return tensor::make_unbound_tensor<Element>(layout);
    }();
    return spec.with_view_tensor_and_projection(
        std::move(view), tensor::identity_projection<decltype(view)::Ndim>());
  }

  /// The rank-two B shared by every batch item: packed inputs are already
  /// batch-free and return as-is; raw inputs slice batch 0, which the
  /// shared/broadcast gates upstream make equivalent to any other slice.
  VECOPS_ALWAYS_INLINE static auto shared_b_leaf(const Invocation& op) {
    if constexpr (Invocation::PackedBInput) return op.b_;
    else return tensor::slice_view<0>(op.b_, 0);
  }

  /// Rank-two A twin of shared_b_leaf().
  VECOPS_ALWAYS_INLINE static auto shared_a_leaf(const Invocation& op) {
    if constexpr (Invocation::PackedAInput) return op.a_;
    else return tensor::slice_view<0>(op.a_, 0);
  }
};

} // namespace vecops::matmul::details

#endif // VECOPS_MATMUL_DETAILS_FAMILIES_ARCHITECTURE_BATCH_PLANNER_H
