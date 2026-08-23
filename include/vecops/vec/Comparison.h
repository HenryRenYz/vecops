#ifndef VECOPS_VEC_COMPARISON_H
#define VECOPS_VEC_COMPARISON_H

#include "vecops/vec/Basic.h"
#include "vecops/vec/Request.h"

namespace vecops::vec {

/* **************************************************************************** */
//    Value comparisons: cmpeq, cmpne, cmplt, cmpgt, cmple, cmpge       //
/* **************************************************************************** */

/**
 * Lane-wise equality.
 * Floating lanes follow C++ ordered comparison: NaN != NaN returns false,
 * NaN == anything returns false. The masked overload returns mask[i] &&
 * (a[i] == b[i]).
 * @see cmpne, cmplt, cmpgt, cmple, cmpge.
 */
struct CmpEqOp {
  template <VectorTag Tag> VECOPS_ALWAYS_INLINE Mask<Tag> operator()(
      Tag tag, Vec<Tag> a, Vec<Tag> b) const;
  template <VectorTag Tag, typename... Options>
    requires (sizeof...(Options) > 0)
  VECOPS_ALWAYS_INLINE Mask<Tag> operator()(
      Tag tag, Vec<Tag> a, Vec<Tag> b, Options&&... options) const;
  template <VectorTag Tag, Active A, Inactive I>
  VECOPS_ALWAYS_INLINE Mask<Tag> operator()(
      Tag tag, Vec<Tag> a, Vec<Tag> b,
      const OpRequest<Tag, A, I>& request) const;
  template <TagInferableVector V, typename... Options>
  VECOPS_ALWAYS_INLINE Mask<VecToTag<V>> operator()(
      V a, V b, Options&&... options) const {
    return (*this)(
        VecToTag<V>{}, a, b, std::forward<Options>(options)...);
  }
};

/**
 * Lane-wise inequality.
 * Unlike C++ semantics, this returns true when either operand is NaN
 * (i.e. NaN != anything is true, NaN != NaN is true). The masked overload
 * returns mask[i] && (a[i] != b[i]).
 * @see cmpeq, cmplt, cmpgt, cmple, cmpge.
 */
struct CmpNeOp {
  template <VectorTag Tag> VECOPS_ALWAYS_INLINE Mask<Tag> operator()(
      Tag tag, Vec<Tag> a, Vec<Tag> b) const;
  template <VectorTag Tag, typename... Options>
    requires (sizeof...(Options) > 0)
  VECOPS_ALWAYS_INLINE Mask<Tag> operator()(
      Tag tag, Vec<Tag> a, Vec<Tag> b, Options&&... options) const;
  template <VectorTag Tag, Active A, Inactive I>
  VECOPS_ALWAYS_INLINE Mask<Tag> operator()(
      Tag tag, Vec<Tag> a, Vec<Tag> b,
      const OpRequest<Tag, A, I>& request) const;
  template <TagInferableVector V, typename... Options>
  VECOPS_ALWAYS_INLINE Mask<VecToTag<V>> operator()(
      V a, V b, Options&&... options) const {
    return (*this)(
        VecToTag<V>{}, a, b, std::forward<Options>(options)...);
  }
};

/** Lane-wise less-than. The masked overload returns mask[i] && (a[i] < b[i]). */
struct CmpLtOp {
  template <VectorTag Tag> VECOPS_ALWAYS_INLINE Mask<Tag> operator()(
      Tag tag, Vec<Tag> a, Vec<Tag> b) const;
  template <VectorTag Tag, typename... Options>
    requires (sizeof...(Options) > 0)
  VECOPS_ALWAYS_INLINE Mask<Tag> operator()(
      Tag tag, Vec<Tag> a, Vec<Tag> b, Options&&... options) const;
  template <VectorTag Tag, Active A, Inactive I>
  VECOPS_ALWAYS_INLINE Mask<Tag> operator()(
      Tag tag, Vec<Tag> a, Vec<Tag> b,
      const OpRequest<Tag, A, I>& request) const;
  template <TagInferableVector V, typename... Options>
  VECOPS_ALWAYS_INLINE Mask<VecToTag<V>> operator()(
      V a, V b, Options&&... options) const {
    return (*this)(
        VecToTag<V>{}, a, b, std::forward<Options>(options)...);
  }
};

/** Lane-wise greater-than. The masked overload returns mask[i] && (a[i] > b[i]). */
struct CmpGtOp {
  template <VectorTag Tag> VECOPS_ALWAYS_INLINE Mask<Tag> operator()(
      Tag tag, Vec<Tag> a, Vec<Tag> b) const;
  template <VectorTag Tag, typename... Options>
    requires (sizeof...(Options) > 0)
  VECOPS_ALWAYS_INLINE Mask<Tag> operator()(
      Tag tag, Vec<Tag> a, Vec<Tag> b, Options&&... options) const;
  template <VectorTag Tag, Active A, Inactive I>
  VECOPS_ALWAYS_INLINE Mask<Tag> operator()(
      Tag tag, Vec<Tag> a, Vec<Tag> b,
      const OpRequest<Tag, A, I>& request) const;
  template <TagInferableVector V, typename... Options>
  VECOPS_ALWAYS_INLINE Mask<VecToTag<V>> operator()(
      V a, V b, Options&&... options) const {
    return (*this)(
        VecToTag<V>{}, a, b, std::forward<Options>(options)...);
  }
};

/** Lane-wise <= comparison. The masked overload returns mask[i] && (a[i] <= b[i]). */
struct CmpLeOp {
  template <VectorTag Tag> VECOPS_ALWAYS_INLINE Mask<Tag> operator()(
      Tag tag, Vec<Tag> a, Vec<Tag> b) const;
  template <VectorTag Tag, typename... Options>
    requires (sizeof...(Options) > 0)
  VECOPS_ALWAYS_INLINE Mask<Tag> operator()(
      Tag tag, Vec<Tag> a, Vec<Tag> b, Options&&... options) const;
  template <VectorTag Tag, Active A, Inactive I>
  VECOPS_ALWAYS_INLINE Mask<Tag> operator()(
      Tag tag, Vec<Tag> a, Vec<Tag> b,
      const OpRequest<Tag, A, I>& request) const;
  template <TagInferableVector V, typename... Options>
  VECOPS_ALWAYS_INLINE Mask<VecToTag<V>> operator()(
      V a, V b, Options&&... options) const {
    return (*this)(
        VecToTag<V>{}, a, b, std::forward<Options>(options)...);
  }
};

/** Lane-wise >= comparison. The masked overload returns mask[i] && (a[i] >= b[i]). */
struct CmpGeOp {
  template <VectorTag Tag> VECOPS_ALWAYS_INLINE Mask<Tag> operator()(
      Tag tag, Vec<Tag> a, Vec<Tag> b) const;
  template <VectorTag Tag, typename... Options>
    requires (sizeof...(Options) > 0)
  VECOPS_ALWAYS_INLINE Mask<Tag> operator()(
      Tag tag, Vec<Tag> a, Vec<Tag> b, Options&&... options) const;
  template <VectorTag Tag, Active A, Inactive I>
  VECOPS_ALWAYS_INLINE Mask<Tag> operator()(
      Tag tag, Vec<Tag> a, Vec<Tag> b,
      const OpRequest<Tag, A, I>& request) const;
  template <TagInferableVector V, typename... Options>
  VECOPS_ALWAYS_INLINE Mask<VecToTag<V>> operator()(
      V a, V b, Options&&... options) const {
    return (*this)(
        VecToTag<V>{}, a, b, std::forward<Options>(options)...);
  }
};

/* **************************************************************************** */
//    Floating-point classification: isnan, isposinf, isneginf, isinf   //
/* **************************************************************************** */

#define VECOPS_VEC_CLASSIFICATION_MEMBERS                                 \
  template <FloatingTag Tag>                                            \
  VECOPS_ALWAYS_INLINE Mask<Tag> operator()(Tag, Vec<Tag>) const;         \
  template <FloatingTag Tag, typename... Options>                       \
    requires (sizeof...(Options) > 0)                                   \
  VECOPS_ALWAYS_INLINE Mask<Tag> operator()(                              \
      Tag, Vec<Tag>, Options&&...) const;                                 \
  template <FloatingTag Tag, Active A, Inactive I>                        \
  VECOPS_ALWAYS_INLINE Mask<Tag> operator()(                              \
      Tag, Vec<Tag>, const OpRequest<Tag, A, I>&) const;                   \
  template <FloatingVectorValue V, typename... Options>                   \
  VECOPS_ALWAYS_INLINE Mask<VecToTag<V>> operator()(                     \
      V value, Options&&... options) const {                              \
    return (*this)(                                                       \
        VecToTag<V>{}, value, std::forward<Options>(options)...);        \
  }

/** True exactly for NaN lanes; the masked overload additionally gates by mask.
 * @see isposinf, isneginf, isinf.
 */
struct IsNanOp { VECOPS_VEC_CLASSIFICATION_MEMBERS; };
/** True exactly for +infinity lanes; the masked overload additionally gates by mask.
 * @see isneginf, isinf, isnan.
 */
struct IsPosInfOp { VECOPS_VEC_CLASSIFICATION_MEMBERS; };
/** True exactly for -infinity lanes; the masked overload additionally gates by mask.
 * @see isposinf, isinf, isnan.
 */
struct IsNegInfOp { VECOPS_VEC_CLASSIFICATION_MEMBERS; };
/**
 * True for either signed infinity; the masked overload additionally gates by
 * mask. Equivalent to logical OR of isposinf and isneginf.
 * @see isposinf, isneginf, isnan.
 */
struct IsInfOp { VECOPS_VEC_CLASSIFICATION_MEMBERS; };

#undef VECOPS_VEC_CLASSIFICATION_MEMBERS

} // namespace vecops::vec

#include "vecops/vec/details/Dispatch.h"
#include "vecops/vec/details/scalar/Comparison.h"

#if defined(ARCH_X86_FAMILY) && !defined(CPU_CAPABILITY_GENERIC)
#include "vecops/vec/details/x86/Comparison.h"
#elif defined(CPU_CAPABILITY_SVE)
#include "vecops/vec/details/sve/Comparison.h"
#endif

#include "vecops/vec/details/Comparison.h"

namespace vecops::vec {

#define VECOPS_VEC_DEFINE_BINARY_COMPARISON(OpType, Name, Expression)    \
  /**                                                                    \
   * Compares every logical lane according to this CPO.                   \
   * Floating comparisons follow C++ ordered comparison semantics,       \
   * except != is true when either operand is NaN.                        \
   */                                                                     \
  template <VectorTag Tag>                                                \
  VECOPS_ALWAYS_INLINE Mask<Tag> OpType::operator()(                      \
      Tag tag, Vec<Tag> a, Vec<Tag> b) const {                            \
    return details::execute(*this, tag, a, b);                            \
  }                                                                       \
  /**                                                                    \
   * `opt::unmasked` compares every lane. `opt::masked(mask)` selects active \
   * lanes. Inactive lanes are false by                                      \
   * default and with `opt::zero`; `opt::merge(mask_value)` preserves the  \
   * supplied mask lanes instead.                                         \
   */                                                                     \
  template <VectorTag Tag, typename... Options>                           \
    requires (sizeof...(Options) > 0)                                     \
  VECOPS_ALWAYS_INLINE Mask<Tag> OpType::operator()(                      \
      Tag tag, Vec<Tag> a, Vec<Tag> b, Options&&... options) const {      \
    return details::execute_comparison_options(                           \
        *this, tag, a, b, std::forward<Options>(options)...);             \
  }                                                                       \
  template <VectorTag Tag, Active A, Inactive I>                          \
  VECOPS_ALWAYS_INLINE Mask<Tag> OpType::operator()(                      \
      Tag tag, Vec<Tag> a, Vec<Tag> b,                                    \
      const OpRequest<Tag, A, I>& request) const {                        \
    return details::execute_comparison_request(                           \
        *this, tag, a, b, request);                                       \
  }                                                                       \
  inline constexpr OpType Name{}

VECOPS_VEC_DEFINE_BINARY_COMPARISON(CmpEqOp, cmpeq, a[i] == b[i]);
VECOPS_VEC_DEFINE_BINARY_COMPARISON(CmpNeOp, cmpne, a[i] != b[i]);
VECOPS_VEC_DEFINE_BINARY_COMPARISON(CmpLtOp, cmplt, a[i] < b[i]);
VECOPS_VEC_DEFINE_BINARY_COMPARISON(CmpGtOp, cmpgt, a[i] > b[i]);
VECOPS_VEC_DEFINE_BINARY_COMPARISON(CmpLeOp, cmple, a[i] <= b[i]);
VECOPS_VEC_DEFINE_BINARY_COMPARISON(CmpGeOp, cmpge, a[i] >= b[i]);

#undef VECOPS_VEC_DEFINE_BINARY_COMPARISON

#define VECOPS_VEC_DEFINE_CLASSIFICATION(OpType, Name, Description)      \
  /**                                                                    \
   * Classifies every logical floating-point lane; result[i] is true      \
   * exactly when value[i] has this CPO's classification. Finite values,  \
   * and other non-matching IEEE values produce false.                    \
   */                                                                     \
  template <FloatingTag Tag>                                            \
  VECOPS_ALWAYS_INLINE Mask<Tag> OpType::operator()(                      \
      Tag tag, Vec<Tag> value) const {                                    \
    return details::execute(*this, tag, value);                           \
  }                                                                       \
  /**                                                                    \
   * `opt::unmasked` classifies every lane. `opt::masked(mask)` selects     \
   * active lanes. Inactive lanes are false by                               \
   * default and with `opt::zero`; `opt::merge(mask_value)` preserves the  \
   * supplied mask lanes instead.                                         \
   */                                                                     \
  template <FloatingTag Tag, typename... Options>                       \
    requires (sizeof...(Options) > 0)                                   \
  VECOPS_ALWAYS_INLINE Mask<Tag> OpType::operator()(                      \
      Tag tag, Vec<Tag> value, Options&&... options) const {              \
    return details::execute_comparison_options(                           \
        *this, tag, value, std::forward<Options>(options)...);            \
  }                                                                       \
  template <FloatingTag Tag, Active A, Inactive I>                        \
  VECOPS_ALWAYS_INLINE Mask<Tag> OpType::operator()(                      \
      Tag tag, Vec<Tag> value,                                            \
      const OpRequest<Tag, A, I>& request) const {                        \
    return details::execute_comparison_request(                           \
        *this, tag, value, request);                                      \
  }                                                                       \
  inline constexpr OpType Name{}

VECOPS_VEC_DEFINE_CLASSIFICATION(IsNanOp, isnan, is a NaN);
VECOPS_VEC_DEFINE_CLASSIFICATION(IsPosInfOp, isposinf, is positive infinity);
VECOPS_VEC_DEFINE_CLASSIFICATION(IsNegInfOp, isneginf, is negative infinity);
VECOPS_VEC_DEFINE_CLASSIFICATION(IsInfOp, isinf, is either infinity);

#undef VECOPS_VEC_DEFINE_CLASSIFICATION

} // namespace vecops::vec

#endif // VECOPS_VEC_COMPARISON_H
