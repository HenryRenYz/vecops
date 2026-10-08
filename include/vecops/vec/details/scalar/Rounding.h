// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
#ifndef VECOPS_VEC_DETAILS_SCALAR_ROUNDING_H
#define VECOPS_VEC_DETAILS_SCALAR_ROUNDING_H

/** @file Rounding.h @brief Scalar word implementations for rounding. */

#include "vecops/vec/details/Rounding.h"
#include "vecops/vec/details/scalar/Basic.h"

namespace vecops::vec::details {

template <typename Op>
struct ScalarRoundingWordImpl {
  template <nint_t Index, FloatingTag Tag>
  static VECOPS_ALWAYS_INLINE NativeWordVec<Tag> call(
      Op, Tag tag, NativeWordVec<Tag> value) {
    using Traits = RepresentationTraits<ScalarBackend, Tag>;
    using T = ElementOf<Tag>;
    static_assert(Index >= 0 && Index < Traits::word_count);
    for (nint_t lane = 0; lane < Traits::word_lanes; ++lane)
      value[lane] = scalar_rounding_value<Op>(value[lane]);
    return value;
  }

  template <nint_t Index, FloatingTag Tag, typename Policy>
  static VECOPS_ALWAYS_INLINE NativeWordVec<Tag> call(
      Op op, Tag tag, NativeWordVec<Tag> value,
      NativeWordMask<Tag> mask, NativeWordVec<Tag> inactive, Policy) {
    return scalar_masked_merge<Index>(
        tag, call<Index>(op, tag, value), mask, inactive);
  }
};

#define VECOPS_VEC_REGISTER_SCALAR_ROUNDING(OpType)                    \
  template <>                                                          \
  struct NativeWordImpl<ScalarBackend, OpType>                         \
      : ScalarRoundingWordImpl<OpType> {}

VECOPS_VEC_REGISTER_SCALAR_ROUNDING(FloorOp);
VECOPS_VEC_REGISTER_SCALAR_ROUNDING(CeilOp);
VECOPS_VEC_REGISTER_SCALAR_ROUNDING(TruncOp);
VECOPS_VEC_REGISTER_SCALAR_ROUNDING(RoundOp);
VECOPS_VEC_REGISTER_SCALAR_ROUNDING(RoundEvenOp);
VECOPS_VEC_REGISTER_SCALAR_ROUNDING(NearbyIntOp);
VECOPS_VEC_REGISTER_SCALAR_ROUNDING(RintOp);

#undef VECOPS_VEC_REGISTER_SCALAR_ROUNDING

} // namespace vecops::vec::details

#endif // VECOPS_VEC_DETAILS_SCALAR_ROUNDING_H
