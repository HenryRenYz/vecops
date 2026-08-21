#ifndef VECOPS_VEC_CONVERSION_H
#define VECOPS_VEC_CONVERSION_H

#include <utility>

#include "vecops/vec/Basic.h"

namespace vecops::vec {

namespace details {
template <VectorTag ToTag, VectorTag FromTag>
consteval bool valid_mask_conversion();
template <VectorTag ToTag, VectorTag FromTag, typename... Options>
consteval bool valid_conversion_options();
}

/* **************************************************************************** */
//    Element type conversion                                             //
/* **************************************************************************** */

struct ConvertOp {
  template <VectorTag ToTag, VectorTag FromTag, typename... Options>
    requires (details::valid_conversion_options<
              ToTag, FromTag, Options...>())
  VECOPS_ALWAYS_INLINE Vec<ToTag> operator()(
      ToTag to, FromTag from, Vec<FromTag> value,
      Options&&... options) const;

  template <VectorTag ToTag, VectorTag FromTag>
    requires (details::valid_mask_conversion<ToTag, FromTag>())
  VECOPS_ALWAYS_INLINE Mask<ToTag> operator()(
      ToTag to, FromTag from, Mask<FromTag> value) const;
};

} // namespace vecops::vec

#include "vecops/vec/details/Dispatch.h"

#if defined(ARCH_X86_FAMILY) && !defined(CPU_CAPABILITY_GENERIC)
#include "vecops/vec/details/x86/Conversion.h"
#elif defined(CPU_CAPABILITY_SVE)
#include "vecops/vec/details/sve/Conversion.h"
#endif

#include "vecops/vec/details/Conversion.h"

namespace vecops::vec {

/**
 * Converts vector elements under orthogonal layout, value, and population
 * policies. The defaults are cvt::ordered and cvt::saturate.
 *
 * Ordered conversion sets output lane i from input lane i. Unordered permits
 * a stable backend-native permutation with a compositional contract. It is
 * ordered when source and destination elements have equal size; for any
 * compatible element types A, B, and C, A->B->C has the same lane provenance
 * as A->C, so A->B->A restores the original lane positions. This guarantee is
 * about ordering, not numerical round-trip equality. cvt::lane<P> operates on
 * equal-byte Tags: widening selects input lane ratio*i+P, while narrowing
 * writes converted input lane i to output lane ratio*i+P. Inactive narrowing
 * lanes use zero by default, or opt::zero/opt::merge(scalar)/
 * opt::merge(vector) when supplied.
 *
 * Saturating integer conversions clamp to the destination range; floating
 * conversions use the destination format's backend/scalar rounding behavior.
 * cvt::wrap is accepted only for integer narrowing and keeps the low
 * destination-width bits. Ordered/unordered forms require equal logical lane
 * shapes and reject population options. Lane conversion requires a 2x/4x/8x
 * element-width ratio; phase 1 is supported only for 2x conversion, and
 * widening lane conversion rejects population options. The explicit source
 * Tag is required because Vec representation alone loses subword shape.
 *
 * @see load_convert, store_convert for memory-side conversions.
 * @see bitcast for reinterpretation without conversion.
 */
template <VectorTag ToTag, VectorTag FromTag, typename... Options>
  requires (details::valid_conversion_options<
            ToTag, FromTag, Options...>())
VECOPS_ALWAYS_INLINE Vec<ToTag> ConvertOp::operator()(
    ToTag to, FromTag from, Vec<FromTag> value,
    Options&&... options) const {
  return details::execute(
      *this, to, from, value, std::forward<Options>(options)...);
}

/**
 * Converts a mask to a different element granularity without changing its
 * logical lanes. Output lane i equals input lane i. Both Tags must describe
 * the same fixed or scalable logical lane shape; the explicit source Tag is
 * required because a Mask does not encode its element type or logical extent.
 */
template <VectorTag ToTag, VectorTag FromTag>
  requires (details::valid_mask_conversion<ToTag, FromTag>())
VECOPS_ALWAYS_INLINE Mask<ToTag> ConvertOp::operator()(
    ToTag to, FromTag from, Mask<FromTag> value) const {
  return details::execute(*this, to, from, value);
}

inline constexpr ConvertOp convert{};

} // namespace vecops::vec

#endif // VECOPS_VEC_CONVERSION_H
