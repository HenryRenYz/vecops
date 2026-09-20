#ifndef VECOPS_OPS_DETAILS_ATTENTION_COMMON_H
#define VECOPS_OPS_DETAILS_ATTENTION_COMMON_H

#include <algorithm>
#include <concepts>
#include <limits>
#include <type_traits>
#include <utility>

#include "vecops/Assertion.h"
#include "vecops/CoreTypes.h"
#include "vecops/kernel/Loop.h"
#include "vecops/kernel/Workspace.h"
#include "vecops/ops/details/softmax/Exp.h"
#include "vecops/tensor/DataAccess.h"
#include "vecops/tensor/OptionalOperand.h"
#include "vecops/vec/Vec.h"

namespace vecops::ops::attention_details {

/** Causal-mask alignment follows PyTorch SDPA's upper-left/lower-right model.
 * A true keep predicate means that an element participates in attention.
 * @see https://docs.pytorch.org/docs/main/generated/torch.nn.functional.scaled_dot_product_attention.html
 * @see https://docs.pytorch.org/docs/main/generated/torch.nn.attention.bias.CausalBias.html
 */
enum class CausalMode {
  none,
  top_left,
  bottom_right,
};

template <typename Element, int Rank,
          tensor::TensorOf<Element, Rank> Buffer>
VECOPS_INLINE void validate_contiguous_buffer(
    const Buffer& buffer, const char* message) {
  VECOPS_ASSERT(buffer.is_contiguous(), message);
}

/** Normalize byte and C++ bool keep masks to a vectorizable compute boundary.
 * Reading a bool object representation through unsigned char is permitted by
 * the C++ aliasing rules; DataAccess then widens bytes to @p Compute so mask
 * vectors share the score Tag. Every nonzero value remains an active lane.
 * Transformed bool Specs are intentionally rejected because reinterpreting
 * them would discard the transform's semantic type.
 */
template <typename Compute, typename Operand>
  requires (!tensor::is_nullopt_v<Operand>)
VECOPS_INLINE auto as_mask_input(Operand&& operand) {
  using O = std::remove_cvref_t<Operand>;
  if constexpr (tensor::is_tensor_v<O>) {
    using Memory = std::remove_const_t<typename O::ElementType>;
    if constexpr (std::same_as<Memory, bool>) {
      using Byte = std::conditional_t<
          std::is_const_v<typename O::ElementType>, const uint8_t, uint8_t>;
      auto bytes = [&]() VECOPS_INLINE_LAMBDA {
        if constexpr (O::is_bound) {
          return tensor::make_tensor(
              reinterpret_cast<Byte*>(operand.data()), operand.layout());
        } else {
          using Layout = typename O::Layout;
          return tensor::Tensor<
              Byte, typename Layout::Shape, typename Layout::Strides,
              tensor::UnboundBinding>{
                tensor::UnboundBinding{operand.element_offset()},
                operand.layout()};
        }
      }();
      return tensor::input<Compute>(bytes);
    } else {
      return tensor::as_input_spec<Compute>(std::forward<Operand>(operand));
    }
  } else {
    using Memory = std::remove_const_t<typename O::MemoryElement>;
    if constexpr (std::same_as<Memory, bool>) {
      static_assert(std::same_as<typename O::TransformType, tensor::NoTransform>,
                    "transformed bool attention masks are not supported");
      using Tensor = typename O::InputTensor;
      using Byte = std::conditional_t<
          std::is_const_v<typename Tensor::ElementType>,
          const uint8_t, uint8_t>;
      auto bytes = [&]() VECOPS_INLINE_LAMBDA {
        if constexpr (O::is_bound) {
          return tensor::make_tensor(
              reinterpret_cast<Byte*>(operand.tensor().data()),
              operand.input_layout());
        } else {
          using Layout = typename Tensor::Layout;
          return tensor::Tensor<
              Byte, typename Layout::Shape, typename Layout::Strides,
              tensor::UnboundBinding>{
                tensor::UnboundBinding{
                    operand.tensor().element_offset()},
                operand.input_layout()};
        }
      }();
      return tensor::input<Compute>(bytes);
    } else {
      return tensor::as_input_spec<Compute>(std::forward<Operand>(operand));
    }
  }
}

template <typename Compute>
VECOPS_INLINE constexpr tensor::nullopt_t as_mask_input(
    tensor::nullopt_t) {
  return tensor::nullopt;
}


/** Load a contiguous mask segment once for scalar row-control consumers.
 *
 * The mask values are converted to @p Score in vector batches and retained in
 * @p activity; the return value counts nonzero lanes. Null masks require no
 * load and report every row active. This keeps small control loops from
 * issuing one DataAccess scalar load per row while preserving compile-time
 * optionality.
 */
template <typename Score, typename MaskAccess>
VECOPS_INLINE nint_t load_activity_segment(
    MaskAccess& mask, nint_t begin, nint_t count, Score* activity) {
  if constexpr (tensor::is_nullopt_v<MaskAccess>) {
    (void)mask;
    (void)begin;
    (void)activity;
    return count;
  } else {
    using Tag = vec::ScalableTag<Score, 0>;
    Tag tag{};
    nint_t active_count = 0;
    kernel::loop::fold<2, 1>(
        tag, count,
        [&](auto block_tag, nint_t row, auto active, const auto&)
            VECOPS_INLINE_LAMBDA {
          auto values = mask.load(
              block_tag, tensor::coord(begin + row), tensor::axis<0>,
              active, vec::opt::zero);
          auto keep = vec::cmpne(
              block_tag, values, vec::zeros(block_tag));
          active_count += vec::mask_count(block_tag, keep);
          vec::store(block_tag, activity + row, values, active);
        },
        kernel::loop::invariant(Score{}));
    return active_count;
  }
}

template <typename QLayout, typename KLayout, typename VLayout,
          typename OLayout>
VECOPS_INLINE void validate_dense_layouts(
    const QLayout& q, const KLayout& k, const VLayout& v,
    const OLayout& out) {
  static_assert(QLayout::Ndim == 2);
  static_assert(KLayout::Ndim == 2);
  static_assert(VLayout::Ndim == 2);
  static_assert(OLayout::Ndim == 2);
  const auto lq = tensor::size<0>(q);
  const auto lkv = tensor::size<0>(k);
  const auto dqk = tensor::size<1>(q);
  const auto dv = tensor::size<1>(v);
  VECOPS_ASSERT(lq > 0 && lkv > 0 && dqk > 0 && dv > 0,
                "attention dimensions must be non-empty");
  VECOPS_ASSERT(tensor::size<1>(k) == dqk,
                "attention Q/K head dimensions differ");
  VECOPS_ASSERT(tensor::size<0>(v) == lkv,
                "attention K/V sequence dimensions differ");
  VECOPS_ASSERT(tensor::size<0>(out) == lq && tensor::size<1>(out) == dv,
                "attention output shape mismatch");
}

template <tensor::InputSpecLike Spec>
VECOPS_INLINE void validate_vector_optional(
    const Spec& spec, nint_t n, const char* message) {
  static_assert(Spec::InputTensor::Ndim == 1);
  VECOPS_ASSERT(tensor::size<0>(spec.input_layout()) == n, message);
}

VECOPS_INLINE constexpr void validate_vector_optional(
    tensor::nullopt_t, nint_t, const char*) {}

template <tensor::InputSpecLike Spec>
VECOPS_INLINE void validate_matrix_optional(
    const Spec& spec, nint_t rows, nint_t columns,
    const char* message) {
  static_assert(Spec::InputTensor::Ndim == 2);
  VECOPS_ASSERT(tensor::size<0>(spec.input_layout()) == rows &&
                tensor::size<1>(spec.input_layout()) == columns, message);
}

VECOPS_INLINE constexpr void validate_matrix_optional(
    tensor::nullopt_t, nint_t, nint_t, const char*) {}

template <typename Score>
struct ScoreRowStats {
  Score maximum = -std::numeric_limits<Score>::infinity();
  bool any = false;
};

template <CausalMode Mode, vec::VectorTag Tag, typename Active,
          typename KeyMaskAccess, typename AttentionMaskAccess>
VECOPS_ALWAYS_INLINE auto attention_keep_mask(
    Tag tag, Active active, nint_t active_count, bool query_active,
    nint_t query_index, nint_t key_begin, nint_t offset,
    nint_t lq, nint_t lkv, KeyMaskAccess& key_mask,
    AttentionMaskAccess& attention_mask) {
  const auto zero = vec::zeros(tag);
  auto keep = query_active
      ? vec::mwhilelt(tag, nint_t{0}, active_count)
      : vec::mfalse(tag);
  if constexpr (Mode != CausalMode::none) {
    const nint_t last_key = Mode == CausalMode::top_left
        ? query_index
        : query_index + (lkv - lq);
    keep = vec::mask_and(
        tag, keep,
        vec::mwhilele(tag, key_begin + offset, last_key));
  }
  if constexpr (!tensor::is_nullopt_v<KeyMaskAccess>) {
    auto values = key_mask.load(
        tag, tensor::coord(key_begin + offset), tensor::axis<0>,
        active, vec::opt::zero);
    keep = vec::mask_and(
        tag, keep, vec::cmpne(tag, values, zero));
  }
  if constexpr (!tensor::is_nullopt_v<AttentionMaskAccess>) {
    auto values = attention_mask.load(
        tag, tensor::coord(query_index, key_begin + offset), tensor::axis<1>,
        active, vec::opt::zero);
    keep = vec::mask_and(
        tag, keep, vec::cmpne(tag, values, zero));
  }
  return keep;
}

/** Scale, bias and mask one contiguous score row with vector operations. */
template <CausalMode Mode, typename Score, typename KeyMaskAccess,
          typename AttentionMaskAccess, typename BiasAccess>
VECOPS_INLINE ScoreRowStats<Score> decorate_score_row(
    Score* scores, nint_t count, bool query_active,
    nint_t query_index, nint_t key_begin, nint_t lq, nint_t lkv,
    Score scale, KeyMaskAccess& key_mask,
    AttentionMaskAccess& attention_mask, BiasAccess& bias) {
  using Tag = vec::ScalableTag<Score, 0>;
  Tag tag{};
  ScoreRowStats<Score> stats;
  kernel::loop::fold<4, 1>(
      tag, count,
      [&](auto block_tag, nint_t offset, auto active,
          auto& maximum, const auto& scale_v,
          const auto& negative_infinity) VECOPS_INLINE_LAMBDA {
        const nint_t active_count = std::min(
            vec::size(block_tag), count - offset);
        auto keep = attention_keep_mask<Mode>(
            block_tag, active, active_count, query_active, query_index,
            key_begin, offset, lq, lkv, key_mask, attention_mask);
        auto values = vec::mul(
            block_tag,
            vec::load(
                block_tag, scores + offset, active, vec::opt::zero),
            scale_v);
        if constexpr (!tensor::is_nullopt_v<BiasAccess>) {
          values = vec::add(
              block_tag, values,
              bias.load(
                  block_tag,
                  tensor::coord(query_index, key_begin + offset),
                  tensor::axis<1>, active, vec::opt::zero));
        }
        values = vec::blend(
            block_tag, negative_infinity, keep, values);
        vec::store(block_tag, scores + offset, values, active);
        maximum = vec::max(maximum, values);
        stats.any = stats.any || vec::mask_any(block_tag, keep);
      },
      kernel::loop::reduce_max(stats.maximum),
      kernel::loop::invariant(scale),
      kernel::loop::invariant(
          -std::numeric_limits<Score>::infinity()));
  return stats;
}

/** Stable vector row softmax used by the materialized attention path.
 *
 * @p maximum is the maximum returned while decorating this same score row;
 * reusing it avoids a redundant read/reduction pass.
 *
 * Fully masked rows are represented by @p row_valid == false and produce an
 * all-zero probability row. This avoids the undefined `-inf - -inf` step and
 * makes the final attention output exactly zero for such rows.
 */
template <typename Score, typename Probability, vec::Accuracy ExpAccuracy>
VECOPS_INLINE void softmax_row(
    Score* scores, Probability* probabilities, nint_t n,
    bool row_valid, Score maximum) {
  if (!row_valid) {
    std::fill(probabilities, probabilities + n, Probability{});
    return;
  }
  using Tag = vec::ScalableTag<Score, 0>;
  Tag tag{};
  if (maximum == -std::numeric_limits<Score>::infinity()) {
    std::fill(probabilities, probabilities + n, Probability{});
    return;
  }
  Score sum{};
  kernel::loop::fold<4, 1>(
      tag, n,
      [&](auto block_tag, nint_t i, auto active,
          auto& sum_v, const auto& maximum_v,
          const auto& negative_infinity) VECOPS_INLINE_LAMBDA {
        auto values = vec::load(
            block_tag, scores + i, active,
            vec::opt::merge(negative_infinity));
        auto exponential = softmax_details::exp_neg_estimate_safe<
            Score, ExpAccuracy>(
            block_tag, vec::sub(block_tag, values, maximum_v), active);
        sum_v = vec::add(sum_v, exponential);
        vec::store(block_tag, scores + i, exponential, active);
      },
      kernel::loop::reduce_add(sum),
      kernel::loop::invariant(maximum),
      kernel::loop::invariant(
          -std::numeric_limits<Score>::infinity()));
  if (!(sum > Score{})) {
    std::fill(probabilities, probabilities + n, Probability{});
    return;
  }
  const Score reciprocal = Score{1} / sum;
  kernel::loop::fold<4, 1>(
      tag, n,
      [&](auto block_tag, nint_t i, auto active,
          const auto& reciprocal_v) VECOPS_INLINE_LAMBDA {
        auto values = vec::load(
            block_tag, scores + i, active, vec::opt::zero);
        values = vec::mul(block_tag, values, reciprocal_v);
        vec::store_convert(
            block_tag, probabilities + i, values, active);
      },
      kernel::loop::invariant(reciprocal));
}

} // namespace vecops::ops::attention_details

#endif // VECOPS_OPS_DETAILS_ATTENTION_COMMON_H
