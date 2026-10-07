// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
#ifndef VECOPS_OPS_DETAILS_SOFTMAX_EXP_H
#define VECOPS_OPS_DETAILS_SOFTMAX_EXP_H

#include <concepts>
#include <type_traits>

#include "vecops/CoreTypes.h"
#include "vecops/vec/Vec.h"

namespace vecops::ops::softmax_details {

// FP32 Estimate softmax avoids FEXPA: on 920F, particular attention scores
// make that instruction take orders of magnitude longer. A degree-nine
// polynomial approximates exp(x/128) on [-0.625, 0]; seven squarings recover
// exp(x). Centered scores below -80 have negligible softmax weight and are
// flushed to zero. Other accuracy tiers retain their numerical contract.
template <typename Score, vec::Accuracy ExpAccuracy, typename Tag,
          typename Value, typename Active>
VECOPS_INLINE auto exp_neg_estimate_safe(
    Tag tag, Value value, Active active) {
  if constexpr (ExpAccuracy == vec::Accuracy::Estimate &&
                std::same_as<Score, float32_t>) {
    const auto zero = vec::zeros(tag);
    const auto limit = vec::fill(tag, Score{-80});
    const auto clamped = vec::blend(
        tag, vec::max(tag, value, limit),
        vec::cmpne(tag, value, value), value);
    const auto x = vec::mul(tag, clamped, vec::fill(tag, Score{1.0f / 128.0f}));
    const auto one = vec::fill(tag, Score{1});
    auto result = vec::fill(tag, Score{1.0f / 362880.0f});
    result = vec::fmadd(tag, result, x, vec::fill(tag, Score{1.0f / 40320.0f}));
    result = vec::fmadd(tag, result, x, vec::fill(tag, Score{1.0f / 5040.0f}));
    result = vec::fmadd(tag, result, x, vec::fill(tag, Score{1.0f / 720.0f}));
    result = vec::fmadd(tag, result, x, vec::fill(tag, Score{1.0f / 120.0f}));
    result = vec::fmadd(tag, result, x, vec::fill(tag, Score{1.0f / 24.0f}));
    result = vec::fmadd(tag, result, x, vec::fill(tag, Score{1.0f / 6.0f}));
    result = vec::fmadd(tag, result, x, vec::fill(tag, Score{0.5f}));
    result = vec::fmadd(tag, result, x, one);
    result = vec::fmadd(tag, result, x, one);
    for (int i = 0; i < 7; ++i)
      result = vec::mul(tag, result, result);
    result = vec::blend(tag, result, vec::cmplt(tag, value, limit), zero);
    if constexpr (std::same_as<std::remove_cvref_t<Active>, vec::opt::First>) {
      const auto mask = vec::mwhilelt(tag, nint_t{0}, active.count);
      return vec::blend(tag, zero, mask, result);
    } else if constexpr (std::same_as<std::remove_cvref_t<Active>,
                                      vec::opt::Unmasked>) {
      return result;
    } else {
      return vec::blend(tag, zero, active.value, result);
    }
  } else {
    return vec::exp_neg(
        tag, value, vec::opt::math::accuracy<ExpAccuracy>, active,
        vec::opt::zero);
  }
}

} // namespace vecops::ops::softmax_details

#endif // VECOPS_OPS_DETAILS_SOFTMAX_EXP_H
