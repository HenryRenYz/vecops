//
// Copyright (c) vecops contributors.
//

#ifndef VECOPS_GEMM_DETAILS_PACKING_UTILITIES_H
#define VECOPS_GEMM_DETAILS_PACKING_UTILITIES_H

#include <limits>
#include <type_traits>

#include "vecops/Assertion.h"
#include "vecops/CoreTypes.h"
#include "vecops/Meta.h"
#include "vecops/tensor/Layout.h"
#include "vecops/util/Math.h"

namespace vecops::gemm::packing_details {

template <nint_t Value, nint_t Divisor>
VECOPS_INLINE constexpr auto ceil_div_value(
    meta::Const<Value>, meta::Const<Divisor>) {
  static_assert(Divisor > 0);
  return meta::cint<ceil_div(Value, Divisor)>;
}

template <nint_t Alignment, nint_t Low, nint_t High, nint_t Divisor>
VECOPS_INLINE constexpr auto ceil_div_value(
    meta::Dynamic<Alignment, Low, High> value,
    meta::Const<Divisor>) {
  static_assert(Divisor > 0);
  constexpr nint_t ResultAlignment =
      Alignment % Divisor == 0 ? Alignment / Divisor : 1;
  constexpr nint_t ResultLow = Low == meta::kLoInf
      ? meta::kLoInf : ceil_div(Low, Divisor);
  constexpr nint_t ResultHigh = High == meta::kHiInf
      ? meta::kHiInf : ceil_div(High, Divisor);
  return meta::Dynamic<ResultAlignment, ResultLow, ResultHigh>{
      ceil_div(static_cast<nint_t>(value), Divisor)};
}

template <meta::ValueType Value,
          nint_t Alignment, nint_t Low, nint_t High>
VECOPS_INLINE constexpr auto ceil_div_value(
    Value value, meta::Dynamic<Alignment, Low, High> divisor) {
  return meta::Any{ceil_div(
      static_cast<nint_t>(value), static_cast<nint_t>(divisor))};
}

VECOPS_INLINE constexpr void validate_packed_size(
    nint_t spatial, nint_t k, nint_t spatial_block, nint_t k_block) {
  VECOPS_ASSERT(spatial >= 0 && k >= 0,
                "matrix packing extents must be non-negative");
  VECOPS_ASSERT(spatial_block > 0 && k_block > 0,
                "matrix packing blocks must be positive");
  const nint_t spatial_groups = ceil_div(spatial, spatial_block);
  const nint_t k_groups = ceil_div(k, k_block);
  constexpr nint_t limit = std::numeric_limits<nint_t>::max();
  VECOPS_ASSERT(spatial_groups <= limit / spatial_block &&
                    k_groups <= limit / k_block,
                "packed matrix extent overflows nint_t");
  const nint_t padded_spatial = spatial_groups * spatial_block;
  const nint_t padded_k = k_groups * k_block;
  VECOPS_ASSERT(padded_spatial == 0 ||
                    padded_k <= limit / padded_spatial,
                "packed matrix element count overflows nint_t");
}

template <tensor::LayoutLike A, tensor::LayoutLike B>
VECOPS_INLINE bool same_layout_values(const A& a, const B& b) {
  if constexpr (A::Ndim != B::Ndim) {
    return false;
  } else {
    for (int d = 0; d < A::Ndim; ++d) {
      if (a.shape()[d] != b.shape()[d] ||
          a.strides()[d] != b.strides()[d]) {
        return false;
      }
    }
    return true;
  }
}

} // namespace vecops::gemm::packing_details

#endif // VECOPS_GEMM_DETAILS_PACKING_UTILITIES_H
