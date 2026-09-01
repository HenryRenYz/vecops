#ifndef VECOPS_VEC_WIDENINGDOT_H
#define VECOPS_VEC_WIDENINGDOT_H

/**
 * @file WideningDot.h
 * @brief Grouped widening dot products with an optional accumulator.
 */

#include <type_traits>

#include "vecops/vec/Arithmetic.h"
#include "vecops/vec/Conversion.h"

namespace vecops::vec {

namespace details {

template <VectorTag ToTag, VectorTag FromTag1, VectorTag FromTag2>
consteval bool valid_widening_dot();

template <VectorTag ToTag, TagInferableVector V>
struct InferredWideningDotSource;

template <typename ToTag, typename V>
concept WideningDotSourceInferable =
    VectorTag<ToTag> && TagInferableVector<V> && requires {
  typename InferredWideningDotSource<ToTag, V>::Tag;
};

template <VectorTag ToTag, TagInferableVector V>
  requires WideningDotSourceInferable<ToTag, V>
using InferredWideningDotSourceTag =
    typename InferredWideningDotSource<ToTag, V>::Tag;

} // namespace details

struct WideningDotOp {
  template <VectorTag ToTag, VectorTag FromTag1, VectorTag FromTag2>
    requires (details::valid_widening_dot<ToTag, FromTag1, FromTag2>())
  VECOPS_ALWAYS_INLINE Vec<ToTag> operator()(
      ToTag to, FromTag1 from1, FromTag2 from2,
      Vec<FromTag1> a, Vec<FromTag2> b) const;

  template <VectorTag ToTag, VectorTag FromTag1, VectorTag FromTag2>
    requires (details::valid_widening_dot<ToTag, FromTag1, FromTag2>())
  VECOPS_ALWAYS_INLINE Vec<ToTag> operator()(
      ToTag to, FromTag1 from1, FromTag2 from2,
      Vec<FromTag1> a, Vec<FromTag2> b, Vec<ToTag> c) const;

  template <VectorTag ToTag, TagInferableVector FromVec1,
            TagInferableVector FromVec2>
    requires (
        details::WideningDotSourceInferable<ToTag, FromVec1> &&
        details::WideningDotSourceInferable<ToTag, FromVec2> &&
        details::valid_widening_dot<
            ToTag,
            details::InferredWideningDotSourceTag<ToTag, FromVec1>,
            details::InferredWideningDotSourceTag<ToTag, FromVec2>>())
  VECOPS_ALWAYS_INLINE Vec<ToTag> operator()(
      ToTag to, FromVec1 a, FromVec2 b) const;

  template <VectorTag ToTag, TagInferableVector FromVec1,
            TagInferableVector FromVec2>
    requires (
        details::WideningDotSourceInferable<ToTag, FromVec1> &&
        details::WideningDotSourceInferable<ToTag, FromVec2> &&
        details::valid_widening_dot<
            ToTag,
            details::InferredWideningDotSourceTag<ToTag, FromVec1>,
            details::InferredWideningDotSourceTag<ToTag, FromVec2>>())
  VECOPS_ALWAYS_INLINE Vec<ToTag> operator()(
      ToTag to, FromVec1 a, FromVec2 b, Vec<ToTag> c) const;
};

inline constexpr WideningDotOp widening_dot{};

} // namespace vecops::vec

#include "vecops/vec/details/Dispatch.h"
#include "vecops/vec/details/WideningDot.h"

#if defined(ARCH_X86_FAMILY) && !defined(CPU_CAPABILITY_GENERIC)
#include "vecops/vec/details/x86/WideningDot.h"
#elif defined(CPU_CAPABILITY_SVE)
#include "vecops/vec/details/sve/WideningDot.h"
#endif

namespace vecops::vec {

/**
 * Computes grouped products with widening inputs. Equal element and Tag types
 * degenerate to mul. Otherwise all three Tags have equal logical byte spans,
 * the two inputs have equal element widths, and the destination element is a
 * wider type in the same arithmetic category. For
 * G=sizeof(To)/sizeof(From), result[k] is the sum of the G products formed
 * from input lanes G*k through G*k+G-1.
 *
 * Integer inputs retain their individual signedness when widened. Integer
 * multiplication and accumulation use destination-width modulo arithmetic.
 * Floating implementations may fuse or reassociate the per-group multiply-
 * adds, so different backends need not be bit-identical.
 */
template <VectorTag ToTag, VectorTag FromTag1, VectorTag FromTag2>
  requires (details::valid_widening_dot<ToTag, FromTag1, FromTag2>())
VECOPS_ALWAYS_INLINE Vec<ToTag> WideningDotOp::operator()(
    ToTag to, FromTag1 from1, FromTag2 from2,
    Vec<FromTag1> a, Vec<FromTag2> b) const {
  if constexpr (
      std::same_as<ToTag, FromTag1> &&
      std::same_as<ToTag, FromTag2>) {
    return mul(to, a, b);
  } else {
    return details::execute(*this, to, from1, from2, a, b);
  }
}

/**
 * Accumulating form of widening_dot. Equal element and Tag types degenerate
 * to fmadd; otherwise c[k] is added to the grouped widening dot product.
 */
template <VectorTag ToTag, VectorTag FromTag1, VectorTag FromTag2>
  requires (details::valid_widening_dot<ToTag, FromTag1, FromTag2>())
VECOPS_ALWAYS_INLINE Vec<ToTag> WideningDotOp::operator()(
    ToTag to, FromTag1 from1, FromTag2 from2,
    Vec<FromTag1> a, Vec<FromTag2> b, Vec<ToTag> c) const {
  if constexpr (
      std::same_as<ToTag, FromTag1> &&
      std::same_as<ToTag, FromTag2>) {
    return fmadd(to, a, b, c);
  } else {
    return details::execute(*this, to, from1, from2, a, b, c);
  }
}

/**
 * Infers each source element type from its Vec representation and reconstructs
 * the source logical extent from ToTag. This retains subword semantics that a
 * bare Vec-to-Tag reverse mapping cannot recover.
 */
template <VectorTag ToTag, TagInferableVector FromVec1,
          TagInferableVector FromVec2>
  requires (
      details::WideningDotSourceInferable<ToTag, FromVec1> &&
      details::WideningDotSourceInferable<ToTag, FromVec2> &&
      details::valid_widening_dot<
          ToTag,
          details::InferredWideningDotSourceTag<ToTag, FromVec1>,
          details::InferredWideningDotSourceTag<ToTag, FromVec2>>())
VECOPS_ALWAYS_INLINE Vec<ToTag> WideningDotOp::operator()(
    ToTag to, FromVec1 a, FromVec2 b) const {
  using FromTag1 = details::InferredWideningDotSourceTag<ToTag, FromVec1>;
  using FromTag2 = details::InferredWideningDotSourceTag<ToTag, FromVec2>;
  return (*this)(to, FromTag1{}, FromTag2{}, a, b);
}

/** Accumulating form of the source-Tag-inferred overload. */
template <VectorTag ToTag, TagInferableVector FromVec1,
          TagInferableVector FromVec2>
  requires (
      details::WideningDotSourceInferable<ToTag, FromVec1> &&
      details::WideningDotSourceInferable<ToTag, FromVec2> &&
      details::valid_widening_dot<
          ToTag,
          details::InferredWideningDotSourceTag<ToTag, FromVec1>,
          details::InferredWideningDotSourceTag<ToTag, FromVec2>>())
VECOPS_ALWAYS_INLINE Vec<ToTag> WideningDotOp::operator()(
    ToTag to, FromVec1 a, FromVec2 b, Vec<ToTag> c) const {
  using FromTag1 = details::InferredWideningDotSourceTag<ToTag, FromVec1>;
  using FromTag2 = details::InferredWideningDotSourceTag<ToTag, FromVec2>;
  return (*this)(to, FromTag1{}, FromTag2{}, a, b, c);
}

} // namespace vecops::vec

#endif // VECOPS_VEC_WIDENINGDOT_H
