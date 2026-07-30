#ifndef VECOPS_VEC_ARITHMETIC_H
#define VECOPS_VEC_ARITHMETIC_H

#include "vecops/vec/Basic.h"

namespace vecops::vec {

/**
 * A VectorTag whose element type is one of the floating-point Element types
 * (bfloat16_t, float16_t, float32_t, float64_t).
 */
template <typename Tag>
concept FloatingTag =
    VectorTag<Tag> && ::vecops::IsFloatV<ElementOf<Tag>>;

namespace details {
/** Policy tag: preserve the first operand's lanes where inactive. */
struct PreserveArithmeticInactive {};
/** Policy tag: set inactive lanes to zero. */
struct ZeroArithmeticInactive {};
/** Policy tag: set inactive lanes from a supplied vector or scalar merge. */
struct MergeArithmeticInactive {};
} // namespace details

/* **************************************************************************** */
//    Binary arithmetic: add, sub, mul, div                               //
/* **************************************************************************** */

struct AddOp {
  template <VectorTag Tag>
  VECOPS_ALWAYS_INLINE Vec<Tag> operator()(
      Tag tag, Vec<Tag> a, Vec<Tag> b) const;

  template <VectorTag Tag, typename... Options>
    requires (sizeof...(Options) > 0)
  VECOPS_ALWAYS_INLINE Vec<Tag> operator()(
      Tag tag, Vec<Tag> a, Vec<Tag> b, Options&&... options) const;

  template <TagInferableVector V, typename... Options>
  VECOPS_ALWAYS_INLINE V operator()(
      V a, V b, Options&&... options) const {
    return (*this)(
        VecToTagT<V>{}, a, b, std::forward<Options>(options)...);
  }
};

struct SubOp {
  template <VectorTag Tag>
  VECOPS_ALWAYS_INLINE Vec<Tag> operator()(
      Tag tag, Vec<Tag> a, Vec<Tag> b) const;

  template <VectorTag Tag, typename... Options>
    requires (sizeof...(Options) > 0)
  VECOPS_ALWAYS_INLINE Vec<Tag> operator()(
      Tag tag, Vec<Tag> a, Vec<Tag> b, Options&&... options) const;

  template <TagInferableVector V, typename... Options>
  VECOPS_ALWAYS_INLINE V operator()(
      V a, V b, Options&&... options) const {
    return (*this)(
        VecToTagT<V>{}, a, b, std::forward<Options>(options)...);
  }
};

struct MulOp {
  template <VectorTag Tag>
  VECOPS_ALWAYS_INLINE Vec<Tag> operator()(
      Tag tag, Vec<Tag> a, Vec<Tag> b) const;

  template <VectorTag Tag, typename... Options>
    requires (sizeof...(Options) > 0)
  VECOPS_ALWAYS_INLINE Vec<Tag> operator()(
      Tag tag, Vec<Tag> a, Vec<Tag> b, Options&&... options) const;

  template <TagInferableVector V, typename... Options>
  VECOPS_ALWAYS_INLINE V operator()(
      V a, V b, Options&&... options) const {
    return (*this)(
        VecToTagT<V>{}, a, b, std::forward<Options>(options)...);
  }
};

struct DivOp {
  template <FloatingTag Tag>
  VECOPS_ALWAYS_INLINE Vec<Tag> operator()(
      Tag tag, Vec<Tag> a, Vec<Tag> b) const;

  template <FloatingTag Tag, typename... Options>
    requires (sizeof...(Options) > 0)
  VECOPS_ALWAYS_INLINE Vec<Tag> operator()(
      Tag tag, Vec<Tag> a, Vec<Tag> b, Options&&... options) const;

  template <FloatingVectorValue V, typename... Options>
  VECOPS_ALWAYS_INLINE V operator()(
      V a, V b, Options&&... options) const {
    return (*this)(
        VecToTagT<V>{}, a, b, std::forward<Options>(options)...);
  }
};

/* **************************************************************************** */
//    Extrema: min, max                                                   //
/* **************************************************************************** */

#define VECOPS_VEC_DECLARE_EXTREMA_OP(OpType)                           \
  struct OpType {                                                       \
    template <VectorTag Tag>                                           \
    VECOPS_ALWAYS_INLINE Vec<Tag> operator()(                           \
        Tag tag, Vec<Tag> a, Vec<Tag> b) const;                         \
    template <VectorTag Tag, typename... Options>                       \
      requires (sizeof...(Options) > 0)                                \
    VECOPS_ALWAYS_INLINE Vec<Tag> operator()(                           \
        Tag tag, Vec<Tag> a, Vec<Tag> b, Options&&... options) const;   \
    template <TagInferableVector V, typename... Options>                \
    VECOPS_ALWAYS_INLINE V operator()(                                  \
        V a, V b, Options&&... options) const {                         \
      return (*this)(                                                   \
          VecToTagT<V>{}, a, b, std::forward<Options>(options)...);     \
    }                                                                   \
  }

VECOPS_VEC_DECLARE_EXTREMA_OP(MinOp);
VECOPS_VEC_DECLARE_EXTREMA_OP(MaxOp);

#undef VECOPS_VEC_DECLARE_EXTREMA_OP

/* **************************************************************************** */
//    Unary arithmetic: neg, abs                                          //
/* **************************************************************************** */

#define VECOPS_VEC_DECLARE_UNARY_ARITHMETIC_OP(OpType)                 \
  struct OpType {                                                       \
    template <VectorTag Tag>                                           \
    VECOPS_ALWAYS_INLINE Vec<Tag> operator()(                           \
        Tag tag, Vec<Tag> value) const;                                 \
    template <VectorTag Tag, typename... Options>                       \
      requires (sizeof...(Options) > 0)                                \
    VECOPS_ALWAYS_INLINE Vec<Tag> operator()(                           \
        Tag tag, Vec<Tag> value, Options&&... options) const;           \
    template <TagInferableVector V, typename... Options>                \
    VECOPS_ALWAYS_INLINE V operator()(                                  \
        V value, Options&&... options) const {                          \
      return (*this)(                                                   \
          VecToTagT<V>{}, value, std::forward<Options>(options)...);    \
    }                                                                   \
  }

VECOPS_VEC_DECLARE_UNARY_ARITHMETIC_OP(NegOp);
VECOPS_VEC_DECLARE_UNARY_ARITHMETIC_OP(AbsOp);

#undef VECOPS_VEC_DECLARE_UNARY_ARITHMETIC_OP

/* **************************************************************************** */
//    Floating-point unary: sqrt, rcp, rsqrt                              //
/* **************************************************************************** */

#define VECOPS_VEC_DECLARE_FLOATING_UNARY_OP(OpType)                   \
  struct OpType {                                                       \
    template <FloatingTag Tag>                                         \
    VECOPS_ALWAYS_INLINE Vec<Tag> operator()(                           \
        Tag tag, Vec<Tag> value) const;                                 \
    template <FloatingTag Tag, typename... Options>                     \
      requires (sizeof...(Options) > 0)                                \
    VECOPS_ALWAYS_INLINE Vec<Tag> operator()(                           \
        Tag tag, Vec<Tag> value, Options&&... options) const;           \
    template <FloatingVectorValue V, typename... Options>               \
    VECOPS_ALWAYS_INLINE V operator()(                                  \
        V value, Options&&... options) const {                          \
      return (*this)(                                                   \
          VecToTagT<V>{}, value, std::forward<Options>(options)...);    \
    }                                                                   \
  }

VECOPS_VEC_DECLARE_FLOATING_UNARY_OP(SqrtOp);
VECOPS_VEC_DECLARE_FLOATING_UNARY_OP(RcpOp);
VECOPS_VEC_DECLARE_FLOATING_UNARY_OP(RsqrtOp);

#undef VECOPS_VEC_DECLARE_FLOATING_UNARY_OP

/* **************************************************************************** */
//    Fused multiply-add: fmadd, fmsub, fnmadd, fnmsub                   //
/* **************************************************************************** */

#define VECOPS_VEC_DECLARE_FMA_OP(OpType)                               \
  struct OpType {                                                       \
    template <VectorTag Tag>                                           \
    VECOPS_ALWAYS_INLINE Vec<Tag> operator()(                           \
        Tag tag, Vec<Tag> a, Vec<Tag> b, Vec<Tag> c) const;             \
    template <VectorTag Tag, typename... Options>                       \
      requires (sizeof...(Options) > 0)                                \
    VECOPS_ALWAYS_INLINE Vec<Tag> operator()(                           \
        Tag tag, Vec<Tag> a, Vec<Tag> b, Vec<Tag> c,                    \
        Options&&... options) const;                                    \
    template <TagInferableVector V, typename... Options>                \
    VECOPS_ALWAYS_INLINE V operator()(                                  \
        V a, V b, V c, Options&&... options) const {                    \
      return (*this)(                                                   \
          VecToTagT<V>{}, a, b, c, std::forward<Options>(options)...);  \
    }                                                                   \
  }

VECOPS_VEC_DECLARE_FMA_OP(FmaddOp);
VECOPS_VEC_DECLARE_FMA_OP(FmsubOp);
VECOPS_VEC_DECLARE_FMA_OP(FnmaddOp);
VECOPS_VEC_DECLARE_FMA_OP(FnmsubOp);

#undef VECOPS_VEC_DECLARE_FMA_OP

} // namespace vecops::vec

#include "vecops/vec/details/Dispatch.h"
#include "vecops/vec/details/scalar/Arithmetic.h"

#if defined(ARCH_X86_FAMILY) && !defined(CPU_CAPABILITY_GENERIC)
#include "vecops/vec/details/x86/Arithmetic.h"
#elif defined(CPU_CAPABILITY_SVE)
#include "vecops/vec/details/sve/Arithmetic.h"
#endif

#include "vecops/vec/details/Arithmetic.h"

namespace vecops::vec {

/**
 * Computes r[i] = a[i] + b[i] for every logical lane 0 <= i < size(tag).
 *
 * Integer addition wraps modulo 2^bits. float16_t and bfloat16_t results are
 * rounded back to their respective element formats. Multi-word Tags are
 * automatically batched unless a backend supplies a whole-Tag NativeImpl.
 *
 * @see sub, mul, div for other arithmetic operations.
 * @see opt::masked, opt::zero, opt::merge for masking options.
 */
template <VectorTag Tag>
VECOPS_ALWAYS_INLINE Vec<Tag> AddOp::operator()(
    Tag tag, Vec<Tag> a, Vec<Tag> b) const {
  return details::execute(*this, tag, a, b);
}

inline constexpr AddOp add{};

/**
 * Computes r[i] = a[i] + b[i] in lanes selected by exactly one
 * opt::masked(mask) option. Inactive lanes come from a by default,
 * ElementOf<Tag>{} with opt::zero, or the scalar/vector supplied by one
 * opt::merge(value). opt::zero and opt::merge are mutually exclusive.
 *
 * Integer addition wraps modulo 2^bits. Floating-point arithmetic follows the
 * backend's native format and rounding behavior; float16_t and bfloat16_t are
 * rounded back to their element formats.
 */
template <VectorTag Tag, typename... Options>
  requires (sizeof...(Options) > 0)
VECOPS_ALWAYS_INLINE Vec<Tag> AddOp::operator()(
    Tag tag, Vec<Tag> a, Vec<Tag> b, Options&&... options) const {
  return details::execute_arithmetic_options(
      *this, tag, a, b, std::forward<Options>(options)...);
}

/**
 * Computes r[i] = a[i] - b[i] for every logical lane 0 <= i < size(tag).
 *
 * Integer subtraction wraps modulo 2^bits. Floating-point arithmetic follows
 * the backend's native format and rounding behavior; float16_t and bfloat16_t
 * are rounded back to their element formats.
 */
template <VectorTag Tag>
VECOPS_ALWAYS_INLINE Vec<Tag> SubOp::operator()(
    Tag tag, Vec<Tag> a, Vec<Tag> b) const {
  return details::execute(*this, tag, a, b);
}

/**
 * Computes r[i] = a[i] - b[i] in lanes selected by exactly one
 * opt::masked(mask) option. Inactive lanes come from a by default, zero with
 * opt::zero, or one scalar/vector opt::merge(value). The population options
 * are mutually exclusive. Integer subtraction wraps modulo 2^bits; floating
 * results use the same format and rounding behavior as unmasked subtraction.
 */
template <VectorTag Tag, typename... Options>
  requires (sizeof...(Options) > 0)
VECOPS_ALWAYS_INLINE Vec<Tag> SubOp::operator()(
    Tag tag, Vec<Tag> a, Vec<Tag> b, Options&&... options) const {
  return details::execute_arithmetic_options(
      *this, tag, a, b, std::forward<Options>(options)...);
}

inline constexpr SubOp sub{};

/**
 * Computes r[i] = a[i] * b[i] for every logical lane 0 <= i < size(tag).
 *
 * Integer multiplication returns the low element-width bits (modulo 2^bits).
 * Floating-point arithmetic follows the backend's native format and rounding
 * behavior; float16_t and bfloat16_t are rounded back to their formats.
 */
template <VectorTag Tag>
VECOPS_ALWAYS_INLINE Vec<Tag> MulOp::operator()(
    Tag tag, Vec<Tag> a, Vec<Tag> b) const {
  return details::execute(*this, tag, a, b);
}

/**
 * Computes r[i] = a[i] * b[i] in lanes selected by exactly one
 * opt::masked(mask) option. Inactive lanes come from a by default, zero with
 * opt::zero, or one scalar/vector opt::merge(value). The population options
 * are mutually exclusive. Integer multiplication is modulo 2^bits; floating
 * results use the same format and rounding behavior as unmasked multiplication.
 */
template <VectorTag Tag, typename... Options>
  requires (sizeof...(Options) > 0)
VECOPS_ALWAYS_INLINE Vec<Tag> MulOp::operator()(
    Tag tag, Vec<Tag> a, Vec<Tag> b, Options&&... options) const {
  return details::execute_arithmetic_options(
      *this, tag, a, b, std::forward<Options>(options)...);
}

inline constexpr MulOp mul{};

/**
 * Computes floating-point r[i] = a[i] / b[i] in every logical lane.
 *
 * Only floating-point Tags are supported (FloatingTag constraint). Division
 * by zero produces a backend-dependent result (typically infinity or NaN).
 * Integer division is not available; use bit_shr for power-of-two division.
 *
 * @see rcp, rsqrt for reciprocal and reciprocal square root.
 */
template <FloatingTag Tag>
VECOPS_ALWAYS_INLINE Vec<Tag> DivOp::operator()(
    Tag tag, Vec<Tag> a, Vec<Tag> b) const {
  return details::execute(*this, tag, a, b);
}

/**
 * Computes division in lanes selected by exactly one `opt::masked(mask)`.
 * Inactive lanes preserve a by default, or use `opt::zero` / scalar-or-vector
 * `opt::merge`.
 */
template <FloatingTag Tag, typename... Options>
  requires (sizeof...(Options) > 0)
VECOPS_ALWAYS_INLINE Vec<Tag> DivOp::operator()(
    Tag tag, Vec<Tag> a, Vec<Tag> b, Options&&... options) const {
  return details::execute_arithmetic_options(
      *this, tag, a, b, std::forward<Options>(options)...);
}

inline constexpr DivOp div{};

#define VECOPS_VEC_DEFINE_EXTREMA_OP(OpType, Name)                     \
  template <VectorTag Tag>                                             \
  VECOPS_ALWAYS_INLINE Vec<Tag> OpType::operator()(                    \
      Tag tag, Vec<Tag> a, Vec<Tag> b) const {                         \
    return details::execute(*this, tag, a, b);                         \
  }                                                                    \
  template <VectorTag Tag, typename... Options>                        \
    requires (sizeof...(Options) > 0)                                 \
  VECOPS_ALWAYS_INLINE Vec<Tag> OpType::operator()(                    \
      Tag tag, Vec<Tag> a, Vec<Tag> b, Options&&... options) const {   \
    return details::execute_arithmetic_options(                        \
        *this, tag, a, b, std::forward<Options>(options)...);          \
  }                                                                    \
  inline constexpr OpType Name{}

/**
 * Computes the lane-wise minimum. Floating NaN and signed-zero selection
 * follows the active backend, matching the legacy operation. Filtered calls
 * require exactly one opt::masked(mask); inactive lanes preserve a by
 * default, or use opt::zero or one scalar/vector opt::merge value.
 */
VECOPS_VEC_DEFINE_EXTREMA_OP(MinOp, min);

/**
 * Computes the lane-wise maximum. Floating NaN and signed-zero selection
 * follows the active backend, matching the legacy operation. Filtered calls
 * require exactly one opt::masked(mask); inactive lanes preserve a by
 * default, or use opt::zero or one scalar/vector opt::merge value.
 */
VECOPS_VEC_DEFINE_EXTREMA_OP(MaxOp, max);

#undef VECOPS_VEC_DEFINE_EXTREMA_OP

#define VECOPS_VEC_DEFINE_UNARY_ARITHMETIC_OP(OpType, Name)            \
  template <VectorTag Tag>                                             \
  VECOPS_ALWAYS_INLINE Vec<Tag> OpType::operator()(                    \
      Tag tag, Vec<Tag> value) const {                                 \
    return details::execute(*this, tag, value);                        \
  }                                                                    \
  template <VectorTag Tag, typename... Options>                        \
    requires (sizeof...(Options) > 0)                                 \
  VECOPS_ALWAYS_INLINE Vec<Tag> OpType::operator()(                    \
      Tag tag, Vec<Tag> value, Options&&... options) const {           \
    return details::execute_unary_arithmetic_options(                  \
        *this, tag, value, std::forward<Options>(options)...);         \
  }                                                                    \
  inline constexpr OpType Name{}

/**
 * Toggles the sign of every lane. Floating values are transformed by toggling
 * only the sign bit, preserving zero signs and NaN payload bits. Integer
 * negation is modulo the element width. Filtered calls require exactly one
 * opt::masked(mask); inactive lanes preserve value by default or use
 * opt::zero / one scalar-or-vector opt::merge.
 */
VECOPS_VEC_DEFINE_UNARY_ARITHMETIC_OP(NegOp, neg);

/**
 * Clears the sign bit of floating lanes and computes modular integer absolute
 * value; consequently abs(INT_MIN) retains the INT_MIN bit pattern. Unsigned
 * lanes are unchanged. Filtered calls use the same Options population policy
 * as neg and never accept a positional mask/default argument.
 */
VECOPS_VEC_DEFINE_UNARY_ARITHMETIC_OP(AbsOp, abs);

/**
 * Computes the square root of every floating-point lane. Filtered calls
 * require exactly one opt::masked(mask); inactive lanes preserve the input by
 * default or use opt::zero / one scalar-or-vector opt::merge.
 */
template <FloatingTag Tag>
VECOPS_ALWAYS_INLINE Vec<Tag> SqrtOp::operator()(
    Tag tag, Vec<Tag> value) const {
  return details::execute(*this, tag, value);
}
template <FloatingTag Tag, typename... Options>
  requires (sizeof...(Options) > 0)
VECOPS_ALWAYS_INLINE Vec<Tag> SqrtOp::operator()(
    Tag tag, Vec<Tag> value, Options&&... options) const {
  return details::execute_unary_arithmetic_options(
      *this, tag, value, std::forward<Options>(options)...);
}
inline constexpr SqrtOp sqrt{};

/**
 * Computes a reciprocal (1/x) using the active backend's legacy instruction
 * tier. x86 and SVE estimate instructions remain estimates where previously
 * used — this is NOT a full-precision IEEE reciprocal. rcp(0) may produce
 * infinity or NaN depending on the backend.
 *
 * Filtered calls use the Options population policy and have no positional
 * mask/default overload.
 *
 * @see rsqrt for reciprocal square root.
 * @see div for full-precision floating-point division.
 */
template <FloatingTag Tag>
VECOPS_ALWAYS_INLINE Vec<Tag> RcpOp::operator()(
    Tag tag, Vec<Tag> value) const {
  return details::execute(*this, tag, value);
}
template <FloatingTag Tag, typename... Options>
  requires (sizeof...(Options) > 0)
VECOPS_ALWAYS_INLINE Vec<Tag> RcpOp::operator()(
    Tag tag, Vec<Tag> value, Options&&... options) const {
  return details::execute_unary_arithmetic_options(
      *this, tag, value, std::forward<Options>(options)...);
}
inline constexpr RcpOp rcp{};

/**
 * Computes a reciprocal square root (1/sqrt(x)) using the active backend's
 * legacy estimate instruction tier. Like rcp, this is NOT full-precision.
 * rsqrt(0) may produce infinity or NaN. Negative inputs may produce NaN.
 *
 * Filtered calls use exactly one opt::masked plus optional opt::zero or
 * scalar/vector opt::merge population.
 *
 * @see sqrt for full-precision square root.
 * @see rcp for reciprocal.
 */
template <FloatingTag Tag>
VECOPS_ALWAYS_INLINE Vec<Tag> RsqrtOp::operator()(
    Tag tag, Vec<Tag> value) const {
  return details::execute(*this, tag, value);
}
template <FloatingTag Tag, typename... Options>
  requires (sizeof...(Options) > 0)
VECOPS_ALWAYS_INLINE Vec<Tag> RsqrtOp::operator()(
    Tag tag, Vec<Tag> value, Options&&... options) const {
  return details::execute_unary_arithmetic_options(
      *this, tag, value, std::forward<Options>(options)...);
}
inline constexpr RsqrtOp rsqrt{};

#undef VECOPS_VEC_DEFINE_UNARY_ARITHMETIC_OP

#define VECOPS_VEC_DEFINE_FMA_OP(OpType, Name)                          \
  template <VectorTag Tag>                                               \
  VECOPS_ALWAYS_INLINE Vec<Tag> OpType::operator()(                      \
      Tag tag, Vec<Tag> a, Vec<Tag> b, Vec<Tag> c) const {              \
    return details::execute(*this, tag, a, b, c);                       \
  }                                                                      \
  template <VectorTag Tag, typename... Options>                          \
    requires (sizeof...(Options) > 0)                                    \
  VECOPS_ALWAYS_INLINE Vec<Tag> OpType::operator()(                      \
      Tag tag, Vec<Tag> a, Vec<Tag> b, Vec<Tag> c,                      \
      Options&&... options) const {                                      \
    return details::execute_ternary_arithmetic_options(                 \
        *this, tag, a, b, c, std::forward<Options>(options)...);        \
  }                                                                      \
  inline constexpr OpType Name{}

/**
 * Computes r[i] = a[i] * b[i] + c[i]. A backend-native fused instruction is
 * used when available; fallback paths may round the product separately
 * (no single-rounding guarantee). Integer lanes wrap modulo 2^bits.
 * Filtered calls require exactly one opt::masked(mask); inactive lanes
 * preserve a by default, or use opt::zero or one scalar/vector opt::merge.
 *
 * @see fmsub, fnmadd, fnmsub for other FMA variants.
 * @see mul, add for separate multiply-add.
 */
VECOPS_VEC_DEFINE_FMA_OP(FmaddOp, fmadd);

/**
 * Computes r[i] = a[i] * b[i] - c[i], with backend-native fusion when
 * available and modulo integer arithmetic. Filtered calls require exactly one
 * opt::masked(mask); inactive lanes preserve a by default, or use opt::zero
 * or one scalar/vector opt::merge value.
 *
 * @see fmadd, fnmadd, fnmsub for other FMA variants.
 */
VECOPS_VEC_DEFINE_FMA_OP(FmsubOp, fmsub);

/**
 * Computes r[i] = -(a[i] * b[i]) + c[i], with backend-native fusion when
 * available and modulo integer arithmetic. Filtered calls require exactly one
 * opt::masked(mask); inactive lanes preserve a by default, or use opt::zero
 * or one scalar/vector opt::merge value.
 *
 * @see fmadd, fmsub, fnmsub for other FMA variants.
 */
VECOPS_VEC_DEFINE_FMA_OP(FnmaddOp, fnmadd);

/**
 * Computes r[i] = -(a[i] * b[i]) - c[i], with backend-native fusion when
 * available and modulo integer arithmetic. Filtered calls require exactly one
 * opt::masked(mask); inactive lanes preserve a by default, or use opt::zero
 * or one scalar/vector opt::merge value. Fallback paths preserve IEEE signed
 * zero when negating a rounded product.
 *
 * @see fmadd, fmsub, fnmadd for other FMA variants.
 */
VECOPS_VEC_DEFINE_FMA_OP(FnmsubOp, fnmsub);

#undef VECOPS_VEC_DEFINE_FMA_OP

} // namespace vecops::vec

#endif // VECOPS_VEC_ARITHMETIC_H
