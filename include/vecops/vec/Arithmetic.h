#ifndef VECOPS_VEC_ARITHMETIC_H
#define VECOPS_VEC_ARITHMETIC_H

#include "vecops/vec/Basic.h"
#include "vecops/vec/Request.h"

namespace vecops::vec {

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
  template <VectorTag Tag, Active A, Inactive I>
  VECOPS_ALWAYS_INLINE Vec<Tag> operator()(
      Tag tag, Vec<Tag> a, Vec<Tag> b,
      const OpRequest<Tag, A, I>& request) const;

  template <TagInferableVector V, typename... Options>
  VECOPS_ALWAYS_INLINE V operator()(
      V a, V b, Options&&... options) const {
    return (*this)(
        VecToTag<V>{}, a, b, std::forward<Options>(options)...);
  }

  /**
   * Low-layer masked entry used by option dispatchers and backends:
   * (tag, values..., mask, inactive, policy) bypasses option parsing.
   */
  template <VectorTag Tag, typename Policy>
    requires (details::arithmetic_inactive_policy<Policy>)
  inline Vec<Tag> operator()(
      Tag tag, Vec<Tag> a, Vec<Tag> b, Mask<Tag> mask,
      Vec<Tag> inactive, Policy policy) const {
    return details::execute(*this, tag, a, b, mask, inactive, policy);
  }

  /**
   * Word-level entry for multi-word Tags; single-word Tags resolve to the
   * whole-Tag overloads above via execute()'s word_count == 1 branch. Used
   * by backend implementations and shared helpers.
   */
  template <VectorTag Tag>
    requires (details::multi_word_tag_v<Tag>)
  inline NativeWordVec<Tag> operator()(
      Tag tag, NativeWordVec<Tag> a, NativeWordVec<Tag> b) const {
    return details::execute_word<0, details::CurrentBackend>(*this, tag, a, b);
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
  template <VectorTag Tag, Active A, Inactive I>
  VECOPS_ALWAYS_INLINE Vec<Tag> operator()(
      Tag tag, Vec<Tag> a, Vec<Tag> b,
      const OpRequest<Tag, A, I>& request) const;

  template <TagInferableVector V, typename... Options>
  VECOPS_ALWAYS_INLINE V operator()(
      V a, V b, Options&&... options) const {
    return (*this)(
        VecToTag<V>{}, a, b, std::forward<Options>(options)...);
  }

  /**
   * Low-layer masked entry used by option dispatchers and backends:
   * (tag, values..., mask, inactive, policy) bypasses option parsing.
   */
  template <VectorTag Tag, typename Policy>
    requires (details::arithmetic_inactive_policy<Policy>)
  inline Vec<Tag> operator()(
      Tag tag, Vec<Tag> a, Vec<Tag> b, Mask<Tag> mask,
      Vec<Tag> inactive, Policy policy) const {
    return details::execute(*this, tag, a, b, mask, inactive, policy);
  }

  /**
   * Word-level entry for multi-word Tags; single-word Tags resolve to the
   * whole-Tag overloads above via execute()'s word_count == 1 branch. Used
   * by backend implementations and shared helpers.
   */
  template <VectorTag Tag>
    requires (details::multi_word_tag_v<Tag>)
  inline NativeWordVec<Tag> operator()(
      Tag tag, NativeWordVec<Tag> a, NativeWordVec<Tag> b) const {
    return details::execute_word<0, details::CurrentBackend>(*this, tag, a, b);
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
  template <VectorTag Tag, Active A, Inactive I>
  VECOPS_ALWAYS_INLINE Vec<Tag> operator()(
      Tag tag, Vec<Tag> a, Vec<Tag> b,
      const OpRequest<Tag, A, I>& request) const;

  template <TagInferableVector V, typename... Options>
  VECOPS_ALWAYS_INLINE V operator()(
      V a, V b, Options&&... options) const {
    return (*this)(
        VecToTag<V>{}, a, b, std::forward<Options>(options)...);
  }

  /**
   * Low-layer masked entry used by option dispatchers and backends:
   * (tag, values..., mask, inactive, policy) bypasses option parsing.
   */
  template <VectorTag Tag, typename Policy>
    requires (details::arithmetic_inactive_policy<Policy>)
  inline Vec<Tag> operator()(
      Tag tag, Vec<Tag> a, Vec<Tag> b, Mask<Tag> mask,
      Vec<Tag> inactive, Policy policy) const {
    return details::execute(*this, tag, a, b, mask, inactive, policy);
  }

  /**
   * Word-level entry for multi-word Tags; single-word Tags resolve to the
   * whole-Tag overloads above via execute()'s word_count == 1 branch. Used
   * by backend implementations and shared helpers.
   */
  template <VectorTag Tag>
    requires (details::multi_word_tag_v<Tag>)
  inline NativeWordVec<Tag> operator()(
      Tag tag, NativeWordVec<Tag> a, NativeWordVec<Tag> b) const {
    return details::execute_word<0, details::CurrentBackend>(*this, tag, a, b);
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
  template <FloatingTag Tag, Active A, Inactive I>
  VECOPS_ALWAYS_INLINE Vec<Tag> operator()(
      Tag tag, Vec<Tag> a, Vec<Tag> b,
      const OpRequest<Tag, A, I>& request) const;

  template <FloatingVectorValue V, typename... Options>
  VECOPS_ALWAYS_INLINE V operator()(
      V a, V b, Options&&... options) const {
    return (*this)(
        VecToTag<V>{}, a, b, std::forward<Options>(options)...);
  }

  /** Low-layer masked entry; see AddOp for the protocol. */
  template <FloatingTag Tag, typename Policy>
    requires (details::arithmetic_inactive_policy<Policy>)
  inline Vec<Tag> operator()(
      Tag tag, Vec<Tag> a, Vec<Tag> b, Mask<Tag> mask,
      Vec<Tag> inactive, Policy policy) const {
    return details::execute(*this, tag, a, b, mask, inactive, policy);
  }

  /** Word-level entry for multi-word Tags; see AddOp. */
  template <FloatingTag Tag>
    requires (details::multi_word_tag_v<Tag>)
  inline NativeWordVec<Tag> operator()(
      Tag tag, NativeWordVec<Tag> a, NativeWordVec<Tag> b) const {
    return details::execute_word<0, details::CurrentBackend>(*this, tag, a, b);
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
        Tag tag, Vec<Tag> a, Vec<Tag> b, Options&&... options) const; \
    template <VectorTag Tag, Active A, Inactive I>                       \
    VECOPS_ALWAYS_INLINE Vec<Tag> operator()(                           \
        Tag tag, Vec<Tag> a, Vec<Tag> b,                                          \
        const OpRequest<Tag, A, I>& request) const;   \
    template <TagInferableVector V, typename... Options>                \
    VECOPS_ALWAYS_INLINE V operator()(                                  \
        V a, V b, Options&&... options) const {                         \
      return (*this)(                                                   \
          VecToTag<V>{}, a, b, std::forward<Options>(options)...);     \
    }                                                                   \
    template <VectorTag Tag, typename Policy>                           \
      requires (details::arithmetic_inactive_policy<Policy>)            \
    inline Vec<Tag> operator()(                           \
        Tag tag, Vec<Tag> a, Vec<Tag> b, Mask<Tag> mask,                \
        Vec<Tag> inactive, Policy policy) const {                      \
      return details::execute(                                          \
          *this, tag, a, b, mask, inactive, policy);                    \
    }                                                                   \
    template <VectorTag Tag>                                            \
      requires (details::multi_word_tag_v<Tag>)                         \
    inline NativeWordVec<Tag> operator()(                \
        Tag tag, NativeWordVec<Tag> a, NativeWordVec<Tag> b) const {   \
      return details::execute_word<0, details::CurrentBackend>(                 \
          *this, tag, a, b);                                            \
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
        Tag tag, Vec<Tag> value, Options&&... options) const; \
    template <VectorTag Tag, Active A, Inactive I>                       \
    VECOPS_ALWAYS_INLINE Vec<Tag> operator()(                           \
        Tag tag, Vec<Tag> value,                                          \
        const OpRequest<Tag, A, I>& request) const;           \
    template <TagInferableVector V, typename... Options>                \
    VECOPS_ALWAYS_INLINE V operator()(                                  \
        V value, Options&&... options) const {                          \
      return (*this)(                                                   \
          VecToTag<V>{}, value, std::forward<Options>(options)...);    \
    }                                                                   \
    template <VectorTag Tag, typename Policy>                           \
      requires (details::arithmetic_inactive_policy<Policy>)            \
    inline Vec<Tag> operator()(                           \
        Tag tag, Vec<Tag> value, Mask<Tag> mask,                        \
        Vec<Tag> inactive, Policy policy) const {                      \
      return details::execute(                                          \
          *this, tag, value, mask, inactive, policy);                   \
    }                                                                   \
    template <VectorTag Tag>                                            \
      requires (details::multi_word_tag_v<Tag>)                         \
    inline NativeWordVec<Tag> operator()(                \
        Tag tag, NativeWordVec<Tag> value) const {                     \
      return details::execute_word<0, details::CurrentBackend>(                 \
          *this, tag, value);                                           \
    }                                                                   \
  }

VECOPS_VEC_DECLARE_UNARY_ARITHMETIC_OP(NegOp);
VECOPS_VEC_DECLARE_UNARY_ARITHMETIC_OP(AbsOp);

#undef VECOPS_VEC_DECLARE_UNARY_ARITHMETIC_OP

/* **************************************************************************** */
//    Floating-point unary: sqrt                                           //
/* **************************************************************************** */

#define VECOPS_VEC_DECLARE_FLOATING_UNARY_OP(OpType)                   \
  struct OpType {                                                       \
    template <FloatingTag Tag>                                         \
    VECOPS_ALWAYS_INLINE Vec<Tag> operator()(                           \
        Tag tag, Vec<Tag> value) const;                                 \
    template <FloatingTag Tag, typename... Options>                     \
      requires (sizeof...(Options) > 0)                                \
    VECOPS_ALWAYS_INLINE Vec<Tag> operator()(                           \
        Tag tag, Vec<Tag> value, Options&&... options) const; \
    template <FloatingTag Tag, Active A, Inactive I>                       \
    VECOPS_ALWAYS_INLINE Vec<Tag> operator()(                           \
        Tag tag, Vec<Tag> value,                                          \
        const OpRequest<Tag, A, I>& request) const;           \
    template <FloatingVectorValue V, typename... Options>               \
    VECOPS_ALWAYS_INLINE V operator()(                                  \
        V value, Options&&... options) const {                          \
      return (*this)(                                                   \
          VecToTag<V>{}, value, std::forward<Options>(options)...);    \
    }                                                                   \
    template <FloatingTag Tag, typename Policy>                         \
      requires (details::arithmetic_inactive_policy<Policy>)            \
    inline Vec<Tag> operator()(                           \
        Tag tag, Vec<Tag> value, Mask<Tag> mask,                        \
        Vec<Tag> inactive, Policy policy) const {                      \
      return details::execute(                                          \
          *this, tag, value, mask, inactive, policy);                   \
    }                                                                   \
    template <FloatingTag Tag>                                          \
      requires (details::multi_word_tag_v<Tag>)                         \
    inline NativeWordVec<Tag> operator()(                \
        Tag tag, NativeWordVec<Tag> value) const {                     \
      return details::execute_word<0, details::CurrentBackend>(                 \
          *this, tag, value);                                           \
    }                                                                   \
  }

VECOPS_VEC_DECLARE_FLOATING_UNARY_OP(SqrtOp);
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
        Options&&... options) const; \
    template <VectorTag Tag, Active A, Inactive I>                       \
    VECOPS_ALWAYS_INLINE Vec<Tag> operator()(                           \
        Tag tag, Vec<Tag> a, Vec<Tag> b, Vec<Tag> c,                    \
        const OpRequest<Tag, A, I>& request) const;                                    \
    template <TagInferableVector V, typename... Options>                \
    VECOPS_ALWAYS_INLINE V operator()(                                  \
        V a, V b, V c, Options&&... options) const {                    \
      return (*this)(                                                   \
          VecToTag<V>{}, a, b, c, std::forward<Options>(options)...);  \
    }                                                                   \
    template <VectorTag Tag, typename Policy>                           \
      requires (details::arithmetic_inactive_policy<Policy>)            \
    inline Vec<Tag> operator()(                           \
        Tag tag, Vec<Tag> a, Vec<Tag> b, Vec<Tag> c, Mask<Tag> mask,    \
        Vec<Tag> inactive, Policy policy) const {                      \
      return details::execute(                                          \
          *this, tag, a, b, c, mask, inactive, policy);                 \
    }                                                                   \
    template <VectorTag Tag>                                            \
      requires (details::multi_word_tag_v<Tag>)                         \
    inline NativeWordVec<Tag> operator()(                \
        Tag tag, NativeWordVec<Tag> a, NativeWordVec<Tag> b,            \
        NativeWordVec<Tag> c) const {                                  \
      return details::execute_word<0, details::CurrentBackend>(                 \
          *this, tag, a, b, c);                                         \
    }                                                                   \
  }

VECOPS_VEC_DECLARE_FMA_OP(FmaddOp);
VECOPS_VEC_DECLARE_FMA_OP(FmsubOp);
VECOPS_VEC_DECLARE_FMA_OP(FnmaddOp);
VECOPS_VEC_DECLARE_FMA_OP(FnmsubOp);

#undef VECOPS_VEC_DECLARE_FMA_OP

/**
 * Public entry-point variables. Declared before the backend includes below so
 * that details-layer implementations can call them by their short names
 * (details/Basic.h already follows this layout for fill/zeros/blend). The
 * operator() definitions live after the backend includes: backends must be
 * able to specialize dispatch before those definitions instantiate them.
 */
inline constexpr AddOp add{};
inline constexpr SubOp sub{};
inline constexpr MulOp mul{};
inline constexpr DivOp div{};
inline constexpr MinOp min{};
inline constexpr MaxOp max{};
inline constexpr NegOp neg{};
inline constexpr AbsOp abs{};
inline constexpr SqrtOp sqrt{};
inline constexpr FmaddOp fmadd{};
inline constexpr FmsubOp fmsub{};
inline constexpr FnmaddOp fnmadd{};
inline constexpr FnmsubOp fnmsub{};

} // namespace vecops::vec

#include "vecops/vec/details/Dispatch.h"
#include "vecops/vec/details/Arithmetic.h"
#include "vecops/vec/details/scalar/Arithmetic.h"

#if defined(ARCH_X86_FAMILY) && !defined(CPU_CAPABILITY_GENERIC)
#include "vecops/vec/details/x86/Arithmetic.h"
#elif defined(CPU_CAPABILITY_SVE)
#include "vecops/vec/details/sve/Arithmetic.h"
#endif

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
  template <VectorTag Tag, Active A, Inactive I>
  VECOPS_ALWAYS_INLINE Vec<Tag> AddOp::operator()(
      Tag tag, Vec<Tag> a, Vec<Tag> b,
      const OpRequest<Tag, A, I>& request) const {
    return details::execute_arithmetic_request(*this, tag, a, b, request);
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
  template <VectorTag Tag, Active A, Inactive I>
  VECOPS_ALWAYS_INLINE Vec<Tag> SubOp::operator()(
      Tag tag, Vec<Tag> a, Vec<Tag> b,
      const OpRequest<Tag, A, I>& request) const {
    return details::execute_arithmetic_request(*this, tag, a, b, request);
  }


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
  template <VectorTag Tag, Active A, Inactive I>
  VECOPS_ALWAYS_INLINE Vec<Tag> MulOp::operator()(
      Tag tag, Vec<Tag> a, Vec<Tag> b,
      const OpRequest<Tag, A, I>& request) const {
    return details::execute_arithmetic_request(*this, tag, a, b, request);
  }


/**
 * Computes floating-point r[i] = a[i] / b[i] in every logical lane.
 *
 * Only floating-point Tags are supported (FloatingTag constraint). Division
 * by zero produces a backend-dependent result (typically infinity or NaN).
 * Integer division is not available; use bit_shr for power-of-two division.
 *
 * @see Math.h rcp for a reciprocal with selectable accuracy.
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
  template <FloatingTag Tag, Active A, Inactive I>
  VECOPS_ALWAYS_INLINE Vec<Tag> DivOp::operator()(
      Tag tag, Vec<Tag> a, Vec<Tag> b,
      const OpRequest<Tag, A, I>& request) const {
    return details::execute_arithmetic_request(*this, tag, a, b, request);
  }


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
  template <VectorTag Tag, Active A, Inactive I>                       \
  VECOPS_ALWAYS_INLINE Vec<Tag> OpType::operator()(                    \
      Tag tag, Vec<Tag> a, Vec<Tag> b,                                 \
      const OpRequest<Tag, A, I>& request) const {                     \
    return details::execute_arithmetic_request(                        \
        *this, tag, a, b, request);                                    \
  }                                                                    \
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
  template <VectorTag Tag, Active A, Inactive I>                       \
  VECOPS_ALWAYS_INLINE Vec<Tag> OpType::operator()(                    \
      Tag tag, Vec<Tag> value,                                        \
      const OpRequest<Tag, A, I>& request) const {                     \
    return details::execute_unary_arithmetic_request(                  \
        *this, tag, value, request);                                  \
  }                                                                    \

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
  template <VectorTag Tag, Active A, Inactive I>                         \
  VECOPS_ALWAYS_INLINE Vec<Tag> OpType::operator()(                      \
      Tag tag, Vec<Tag> a, Vec<Tag> b, Vec<Tag> c,                       \
      const OpRequest<Tag, A, I>& request) const {                       \
    return details::execute_ternary_arithmetic_request(                  \
        *this, tag, a, b, c, request);                                   \
  }                                                                      \

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
