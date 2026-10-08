// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
#ifndef VECOPS_VEC_BIT_H
#define VECOPS_VEC_BIT_H

#include <concepts>
#include <utility>

#include "vecops/Meta.h"
#include "vecops/vec/Basic.h"

namespace vecops::vec {

/**
 * A VectorTag whose element type is integral (signed or unsigned).
 * All bitwise operations are defined on the unsigned bit pattern, so
 * signed element types participate in shifts, AND/OR/XOR/NOT with two's
 * complement bit semantics.
 */
template <typename Tag>
concept IntegerTag = VectorTag<Tag> && std::integral<ElementOf<Tag>>;

/* **************************************************************************** */
//    Bitwise binary: bit_and, bit_or, bit_xor, bit_andnot              //
/* **************************************************************************** */

#define VECOPS_VEC_DECLARE_BIT_BINARY(OpType)                            \
  struct OpType {                                                        \
    template <IntegerTag Tag>                                           \
    VECOPS_ALWAYS_INLINE Vec<Tag> operator()(                           \
        Tag tag, Vec<Tag> a, Vec<Tag> b) const;                         \
    template <IntegerTag Tag, typename... Options>                      \
      requires (sizeof...(Options) > 0)                                 \
    VECOPS_ALWAYS_INLINE Vec<Tag> operator()(                           \
        Tag tag, Vec<Tag> a, Vec<Tag> b, Options&&... options) const;   \
    template <IntegerVectorValue V, typename... Options>                \
    VECOPS_ALWAYS_INLINE V operator()(                                  \
        V a, V b, Options&&... options) const {                         \
      return (*this)(                                                   \
          VecToTag<V>{}, a, b, std::forward<Options>(options)...);     \
    }                                                                   \
    template <IntegerTag Tag>                                            \
      requires (details::multi_word_tag_v<Tag>)                         \
    inline NativeWordVec<Tag> operator()(                \
        Tag tag, NativeWordVec<Tag> a, NativeWordVec<Tag> b) const {   \
      return details::execute_word<0, details::CurrentBackend>(                 \
          *this, tag, a, b);                                            \
    }                                                                   \
  }

VECOPS_VEC_DECLARE_BIT_BINARY(BitAndOp);
VECOPS_VEC_DECLARE_BIT_BINARY(BitOrOp);
VECOPS_VEC_DECLARE_BIT_BINARY(BitXorOp);
VECOPS_VEC_DECLARE_BIT_BINARY(BitAndNotOp);

#undef VECOPS_VEC_DECLARE_BIT_BINARY

/* **************************************************************************** */
//    Bitwise unary: bit_not                                              //
/* **************************************************************************** */

struct BitNotOp {
  template <IntegerTag Tag>
  VECOPS_ALWAYS_INLINE Vec<Tag> operator()(Tag tag, Vec<Tag> value) const;

  template <IntegerTag Tag, typename... Options>
    requires (sizeof...(Options) > 0)
  VECOPS_ALWAYS_INLINE Vec<Tag> operator()(
      Tag tag, Vec<Tag> value, Options&&... options) const;

  template <IntegerVectorValue V, typename... Options>
  VECOPS_ALWAYS_INLINE V operator()(
      V value, Options&&... options) const {
    return (*this)(
        VecToTag<V>{}, value, std::forward<Options>(options)...);
  }

  /** Word-level entry for multi-word Tags; single-word Tags resolve to the
   * whole-Tag overloads above. */
  template <IntegerTag Tag>
    requires (details::multi_word_tag_v<Tag>)
  inline NativeWordVec<Tag> operator()(
      Tag tag, NativeWordVec<Tag> value) const {
    return details::execute_word<0, details::CurrentBackend>(*this, tag, value);
  }
};

/* **************************************************************************** */
//    Bit counts: popcount, countl_zero/one, countr_zero/one                  //
/* **************************************************************************** */

#define VECOPS_VEC_DECLARE_BIT_COUNT(OpType)                             \
  struct OpType {                                                        \
    template <IntegerTag Tag>                                           \
    VECOPS_ALWAYS_INLINE Vec<Tag> operator()(                           \
        Tag tag, Vec<Tag> value) const;                                 \
    template <IntegerTag Tag, typename... Options>                      \
      requires (sizeof...(Options) > 0)                                 \
    VECOPS_ALWAYS_INLINE Vec<Tag> operator()(                           \
        Tag tag, Vec<Tag> value, Options&&... options) const;           \
    template <IntegerVectorValue V, typename... Options>                \
    VECOPS_ALWAYS_INLINE V operator()(                                  \
        V value, Options&&... options) const {                          \
      return (*this)(                                                   \
          VecToTag<V>{}, value, std::forward<Options>(options)...);     \
    }                                                                   \
    template <IntegerTag Tag>                                           \
      requires (details::multi_word_tag_v<Tag>)                         \
    inline NativeWordVec<Tag> operator()(                               \
        Tag tag, NativeWordVec<Tag> value) const {                      \
      return details::execute_word<0, details::CurrentBackend>(         \
          *this, tag, value);                                           \
    }                                                                   \
  }

/** Counts one bits independently in every integer lane. */
VECOPS_VEC_DECLARE_BIT_COUNT(PopCountOp);
/** Counts leading zero bits; a zero lane returns its element bit width. */
VECOPS_VEC_DECLARE_BIT_COUNT(CountLeadingZeroOp);
/** Counts leading one bits; an all-one lane returns its element bit width. */
VECOPS_VEC_DECLARE_BIT_COUNT(CountLeadingOneOp);
/** Counts trailing zero bits; a zero lane returns its element bit width. */
VECOPS_VEC_DECLARE_BIT_COUNT(CountTrailingZeroOp);
/** Counts trailing one bits; an all-one lane returns its element bit width. */
VECOPS_VEC_DECLARE_BIT_COUNT(CountTrailingOneOp);

#undef VECOPS_VEC_DECLARE_BIT_COUNT

/* **************************************************************************** */
//    Bit shifts: shl, shr                                        //
/* **************************************************************************** */

struct BitShiftLeftOp {
  template <IntegerTag Tag>
  VECOPS_ALWAYS_INLINE Vec<Tag> operator()(
      Tag tag, Vec<Tag> value, int count) const;

  template <nint_t Count, IntegerTag Tag>
    requires (Count >= 0)
  VECOPS_ALWAYS_INLINE Vec<Tag> operator()(
      Tag tag, Vec<Tag> value, meta::Const<Count> count) const;

  template <IntegerTag Tag>
  VECOPS_ALWAYS_INLINE Vec<Tag> operator()(
      Tag tag, Vec<Tag> value, Vec<Tag> counts) const;

  template <IntegerTag Tag, typename... Options>
    requires (sizeof...(Options) > 0)
  VECOPS_ALWAYS_INLINE Vec<Tag> operator()(
      Tag tag, Vec<Tag> value, int count, Options&&... options) const;

  template <nint_t Count, IntegerTag Tag, typename... Options>
    requires (Count >= 0 && sizeof...(Options) > 0)
  VECOPS_ALWAYS_INLINE Vec<Tag> operator()(
      Tag tag, Vec<Tag> value, meta::Const<Count> count,
      Options&&... options) const;

  template <IntegerTag Tag, typename... Options>
    requires (sizeof...(Options) > 0)
  VECOPS_ALWAYS_INLINE Vec<Tag> operator()(
      Tag tag, Vec<Tag> value, Vec<Tag> counts,
      Options&&... options) const;

  template <IntegerVectorValue V, typename Count, typename... Options>
    requires requires(
        BitShiftLeftOp op, V value, Count&& count, Options&&... options) {
      op(
          VecToTag<V>{}, value, std::forward<Count>(count),
          std::forward<Options>(options)...);
    }
  VECOPS_ALWAYS_INLINE V operator()(
      V value, Count&& count, Options&&... options) const {
    return (*this)(
        VecToTag<V>{}, value, std::forward<Count>(count),
        std::forward<Options>(options)...);
  }

  /** Word-level int-count entry for multi-word Tags; single-word Tags
   * resolve to the whole-Tag overloads above. */
  template <IntegerTag Tag>
    requires (details::multi_word_tag_v<Tag>)
  inline NativeWordVec<Tag> operator()(
      Tag tag, NativeWordVec<Tag> value, int count) const {
    return details::execute_word<0, details::CurrentBackend>(
        *this, tag, value, count);
  }

  /** Word-level immediate-count entry for multi-word Tags. */
  template <nint_t Count, IntegerTag Tag>
    requires (Count >= 0 && details::multi_word_tag_v<Tag>)
  inline NativeWordVec<Tag> operator()(
      Tag tag, NativeWordVec<Tag> value, meta::Const<Count> count) const {
    return details::execute_word<0, details::CurrentBackend>(
        *this, tag, value, count);
  }

  /** Word-level per-lane-count entry for multi-word Tags. */
  template <IntegerTag Tag>
    requires (details::multi_word_tag_v<Tag>)
  inline NativeWordVec<Tag> operator()(
      Tag tag, NativeWordVec<Tag> value, NativeWordVec<Tag> counts) const {
    return details::execute_word<0, details::CurrentBackend>(
        *this, tag, value, counts);
  }
};

struct BitShiftRightOp {
  template <IntegerTag Tag>
  VECOPS_ALWAYS_INLINE Vec<Tag> operator()(
      Tag tag, Vec<Tag> value, int count) const;

  template <nint_t Count, IntegerTag Tag>
    requires (Count >= 0)
  VECOPS_ALWAYS_INLINE Vec<Tag> operator()(
      Tag tag, Vec<Tag> value, meta::Const<Count> count) const;

  template <IntegerTag Tag>
  VECOPS_ALWAYS_INLINE Vec<Tag> operator()(
      Tag tag, Vec<Tag> value, Vec<Tag> counts) const;

  template <IntegerTag Tag, typename... Options>
    requires (sizeof...(Options) > 0)
  VECOPS_ALWAYS_INLINE Vec<Tag> operator()(
      Tag tag, Vec<Tag> value, int count, Options&&... options) const;

  template <nint_t Count, IntegerTag Tag, typename... Options>
    requires (Count >= 0 && sizeof...(Options) > 0)
  VECOPS_ALWAYS_INLINE Vec<Tag> operator()(
      Tag tag, Vec<Tag> value, meta::Const<Count> count,
      Options&&... options) const;

  template <IntegerTag Tag, typename... Options>
    requires (sizeof...(Options) > 0)
  VECOPS_ALWAYS_INLINE Vec<Tag> operator()(
      Tag tag, Vec<Tag> value, Vec<Tag> counts,
      Options&&... options) const;

  template <IntegerVectorValue V, typename Count, typename... Options>
    requires requires(
        BitShiftRightOp op, V value, Count&& count, Options&&... options) {
      op(
          VecToTag<V>{}, value, std::forward<Count>(count),
          std::forward<Options>(options)...);
    }
  VECOPS_ALWAYS_INLINE V operator()(
      V value, Count&& count, Options&&... options) const {
    return (*this)(
        VecToTag<V>{}, value, std::forward<Count>(count),
        std::forward<Options>(options)...);
  }

  /** Word-level int-count entry for multi-word Tags; single-word Tags
   * resolve to the whole-Tag overloads above. */
  template <IntegerTag Tag>
    requires (details::multi_word_tag_v<Tag>)
  inline NativeWordVec<Tag> operator()(
      Tag tag, NativeWordVec<Tag> value, int count) const {
    return details::execute_word<0, details::CurrentBackend>(
        *this, tag, value, count);
  }

  /** Word-level immediate-count entry for multi-word Tags. */
  template <nint_t Count, IntegerTag Tag>
    requires (Count >= 0 && details::multi_word_tag_v<Tag>)
  inline NativeWordVec<Tag> operator()(
      Tag tag, NativeWordVec<Tag> value, meta::Const<Count> count) const {
    return details::execute_word<0, details::CurrentBackend>(
        *this, tag, value, count);
  }

  /** Word-level per-lane-count entry for multi-word Tags. */
  template <IntegerTag Tag>
    requires (details::multi_word_tag_v<Tag>)
  inline NativeWordVec<Tag> operator()(
      Tag tag, NativeWordVec<Tag> value, NativeWordVec<Tag> counts) const {
    return details::execute_word<0, details::CurrentBackend>(
        *this, tag, value, counts);
  }
};

/* **************************************************************************** */
//    Bit rotations: rotl, rotr                                               //
/* **************************************************************************** */

#define VECOPS_VEC_DECLARE_ROTATE(OpType)                                \
  struct OpType {                                                        \
    template <IntegerTag Tag>                                           \
    VECOPS_ALWAYS_INLINE Vec<Tag> operator()(                           \
        Tag tag, Vec<Tag> value, int count) const;                      \
    template <nint_t Count, IntegerTag Tag>                             \
    VECOPS_ALWAYS_INLINE Vec<Tag> operator()(                           \
        Tag tag, Vec<Tag> value, meta::Const<Count> count) const;       \
    template <IntegerTag Tag>                                           \
    VECOPS_ALWAYS_INLINE Vec<Tag> operator()(                           \
        Tag tag, Vec<Tag> value, Vec<Tag> counts) const;                \
    template <IntegerTag Tag, typename... Options>                      \
      requires (sizeof...(Options) > 0)                                 \
    VECOPS_ALWAYS_INLINE Vec<Tag> operator()(                           \
        Tag tag, Vec<Tag> value, int count, Options&&... options) const;\
    template <nint_t Count, IntegerTag Tag, typename... Options>        \
      requires (sizeof...(Options) > 0)                                 \
    VECOPS_ALWAYS_INLINE Vec<Tag> operator()(                           \
        Tag tag, Vec<Tag> value, meta::Const<Count> count,              \
        Options&&... options) const;                                    \
    template <IntegerTag Tag, typename... Options>                      \
      requires (sizeof...(Options) > 0)                                 \
    VECOPS_ALWAYS_INLINE Vec<Tag> operator()(                           \
        Tag tag, Vec<Tag> value, Vec<Tag> counts,                       \
        Options&&... options) const;                                    \
    template <IntegerVectorValue V, typename Count, typename... Options>\
      requires requires(                                                \
          OpType op, V value, Count&& count, Options&&... options) {    \
        op(                                                             \
            VecToTag<V>{}, value, std::forward<Count>(count),           \
            std::forward<Options>(options)...);                         \
      }                                                                 \
    VECOPS_ALWAYS_INLINE V operator()(                                  \
        V value, Count&& count, Options&&... options) const {           \
      return (*this)(                                                   \
          VecToTag<V>{}, value, std::forward<Count>(count),             \
          std::forward<Options>(options)...);                           \
    }                                                                   \
    template <IntegerTag Tag>                                           \
      requires (details::multi_word_tag_v<Tag>)                         \
    inline NativeWordVec<Tag> operator()(                               \
        Tag tag, NativeWordVec<Tag> value, int count) const {           \
      return details::execute_word<0, details::CurrentBackend>(         \
          *this, tag, value, count);                                    \
    }                                                                   \
    template <nint_t Count, IntegerTag Tag>                             \
      requires (details::multi_word_tag_v<Tag>)                         \
    inline NativeWordVec<Tag> operator()(                               \
        Tag tag, NativeWordVec<Tag> value, meta::Const<Count> count)    \
        const {                                                         \
      return details::execute_word<0, details::CurrentBackend>(         \
          *this, tag, value, count);                                    \
    }                                                                   \
    template <IntegerTag Tag>                                           \
      requires (details::multi_word_tag_v<Tag>)                         \
    inline NativeWordVec<Tag> operator()(                               \
        Tag tag, NativeWordVec<Tag> value, NativeWordVec<Tag> counts)   \
        const {                                                         \
      return details::execute_word<0, details::CurrentBackend>(         \
          *this, tag, value, counts);                                   \
    }                                                                   \
  }

/** Rotates each integer lane left by a modulo-width count. */
VECOPS_VEC_DECLARE_ROTATE(RotateLeftOp);
/** Rotates each integer lane right by a modulo-width count. */
VECOPS_VEC_DECLARE_ROTATE(RotateRightOp);

#undef VECOPS_VEC_DECLARE_ROTATE

/** Public entry-point variables, declared before the backend includes so
 * that details-layer implementations can call them by short names (same
 * layout as Basic.h and Arithmetic.h). */
inline constexpr BitAndOp bit_and{};
inline constexpr BitOrOp bit_or{};
inline constexpr BitXorOp bit_xor{};
inline constexpr BitAndNotOp bit_andnot{};
inline constexpr BitNotOp bit_not{};
inline constexpr BitShiftLeftOp shl{};
inline constexpr BitShiftRightOp shr{};
inline constexpr PopCountOp popcount{};
inline constexpr CountLeadingZeroOp countl_zero{};
inline constexpr CountLeadingOneOp countl_one{};
inline constexpr CountTrailingZeroOp countr_zero{};
inline constexpr CountTrailingOneOp countr_one{};
inline constexpr RotateLeftOp rotl{};
inline constexpr RotateRightOp rotr{};

} // namespace vecops::vec

#include "vecops/vec/details/Dispatch.h"
#include "vecops/vec/details/Bit.h"
#include "vecops/vec/details/scalar/Bit.h"

#if defined(ARCH_X86_FAMILY) && !defined(CPU_CAPABILITY_GENERIC)
#include "vecops/vec/details/x86/Bit.h"
#elif defined(CPU_CAPABILITY_SVE)
#include "vecops/vec/details/sve/Bit.h"
#endif

namespace vecops::vec {

/** Computes result[i] = a[i] & b[i] on every logical integer lane. */
template <IntegerTag Tag>
VECOPS_ALWAYS_INLINE Vec<Tag> BitAndOp::operator()(
    Tag tag, Vec<Tag> a, Vec<Tag> b) const {
  return details::execute(*this, tag, a, b);
}

/**
 * `opt::unmasked` computes every lane. With exactly one
 * `opt::masked(mask)`, computes a[i] & b[i] in active lanes. Inactive lanes
 * preserve a[i], or use the optional `opt::zero` or
 * `opt::merge(vector_or_scalar)` population policy.
 */
template <IntegerTag Tag, typename... Options>
  requires (sizeof...(Options) > 0)
VECOPS_ALWAYS_INLINE Vec<Tag> BitAndOp::operator()(
    Tag tag, Vec<Tag> a, Vec<Tag> b, Options&&... options) const {
  details::validate_bit_options<Tag, Options...>();
  if constexpr (
      details::option_count_v<details::IsUnmaskedOption, Options...> == 1) {
    return details::execute(*this, tag, a, b);
  } else {
    const auto inactive = details::bit_inactive_value(
      tag, a, std::forward<Options>(options)...);
    return details::execute(
      *this, tag, a, b,
      details::find_option<details::IsMaskedOption>(
          std::forward<Options>(options)...).value,
      inactive, details::bit_inactive_policy<Options...>());
  }
}

/** Computes result[i] = a[i] | b[i] on every logical integer lane. */
template <IntegerTag Tag>
VECOPS_ALWAYS_INLINE Vec<Tag> BitOrOp::operator()(
    Tag tag, Vec<Tag> a, Vec<Tag> b) const {
  return details::execute(*this, tag, a, b);
}

/**
 * `opt::unmasked` computes every lane. With exactly one
 * `opt::masked(mask)`, computes a[i] | b[i] in active
 * lanes. Inactive lanes preserve a[i], or use the optional `opt::zero` or
 * `opt::merge(vector_or_scalar)` population policy.
 */
template <IntegerTag Tag, typename... Options>
  requires (sizeof...(Options) > 0)
VECOPS_ALWAYS_INLINE Vec<Tag> BitOrOp::operator()(
    Tag tag, Vec<Tag> a, Vec<Tag> b, Options&&... options) const {
  details::validate_bit_options<Tag, Options...>();
  if constexpr (
      details::option_count_v<details::IsUnmaskedOption, Options...> == 1) {
    return details::execute(*this, tag, a, b);
  } else {
    const auto inactive = details::bit_inactive_value(
      tag, a, std::forward<Options>(options)...);
    return details::execute(
      *this, tag, a, b,
      details::find_option<details::IsMaskedOption>(
          std::forward<Options>(options)...).value,
      inactive, details::bit_inactive_policy<Options...>());
  }
}

/** Computes result[i] = a[i] ^ b[i] on every logical integer lane. */
template <IntegerTag Tag>
VECOPS_ALWAYS_INLINE Vec<Tag> BitXorOp::operator()(
    Tag tag, Vec<Tag> a, Vec<Tag> b) const {
  return details::execute(*this, tag, a, b);
}

/**
 * `opt::unmasked` computes every lane. With exactly one
 * `opt::masked(mask)`, computes a[i] ^ b[i] in active
 * lanes. Inactive lanes preserve a[i], or use the optional `opt::zero` or
 * `opt::merge(vector_or_scalar)` population policy.
 */
template <IntegerTag Tag, typename... Options>
  requires (sizeof...(Options) > 0)
VECOPS_ALWAYS_INLINE Vec<Tag> BitXorOp::operator()(
    Tag tag, Vec<Tag> a, Vec<Tag> b, Options&&... options) const {
  details::validate_bit_options<Tag, Options...>();
  if constexpr (
      details::option_count_v<details::IsUnmaskedOption, Options...> == 1) {
    return details::execute(*this, tag, a, b);
  } else {
    const auto inactive = details::bit_inactive_value(
      tag, a, std::forward<Options>(options)...);
    return details::execute(
      *this, tag, a, b,
      details::find_option<details::IsMaskedOption>(
          std::forward<Options>(options)...).value,
      inactive, details::bit_inactive_policy<Options...>());
  }
}

/** Computes result[i] = (~a[i]) & b[i] on every logical integer lane. */
template <IntegerTag Tag>
VECOPS_ALWAYS_INLINE Vec<Tag> BitAndNotOp::operator()(
    Tag tag, Vec<Tag> a, Vec<Tag> b) const {
  return details::execute(*this, tag, a, b);
}

/**
 * `opt::unmasked` computes every lane. With exactly one
 * `opt::masked(mask)`, computes (~a[i]) & b[i] in active
 * lanes. Inactive lanes preserve a[i], or use the optional `opt::zero` or
 * `opt::merge(vector_or_scalar)` population policy.
 */
template <IntegerTag Tag, typename... Options>
  requires (sizeof...(Options) > 0)
VECOPS_ALWAYS_INLINE Vec<Tag> BitAndNotOp::operator()(
    Tag tag, Vec<Tag> a, Vec<Tag> b, Options&&... options) const {
  details::validate_bit_options<Tag, Options...>();
  if constexpr (
      details::option_count_v<details::IsUnmaskedOption, Options...> == 1) {
    return details::execute(*this, tag, a, b);
  } else {
    const auto inactive = details::bit_inactive_value(
      tag, a, std::forward<Options>(options)...);
    return details::execute(
      *this, tag, a, b,
      details::find_option<details::IsMaskedOption>(
          std::forward<Options>(options)...).value,
      inactive, details::bit_inactive_policy<Options...>());
  }
}

/** Computes result[i] = ~value[i] on every logical integer lane. */
template <IntegerTag Tag>
VECOPS_ALWAYS_INLINE Vec<Tag> BitNotOp::operator()(
    Tag tag, Vec<Tag> value) const {
  return details::execute(*this, tag, value);
}

/**
 * `opt::unmasked` complements every lane. With exactly one
 * `opt::masked(mask)`, complements active lanes. Inactive
 * lanes preserve value[i], or use the optional `opt::zero` or
 * `opt::merge(vector_or_scalar)` population policy.
 */
template <IntegerTag Tag, typename... Options>
  requires (sizeof...(Options) > 0)
VECOPS_ALWAYS_INLINE Vec<Tag> BitNotOp::operator()(
    Tag tag, Vec<Tag> value, Options&&... options) const {
  details::validate_bit_options<Tag, Options...>();
  if constexpr (
      details::option_count_v<details::IsUnmaskedOption, Options...> == 1) {
    return details::execute(*this, tag, value);
  } else {
    const auto inactive = details::bit_inactive_value(
      tag, value, std::forward<Options>(options)...);
    return details::execute(
      *this, tag, value,
      details::find_option<details::IsMaskedOption>(
          std::forward<Options>(options)...).value,
      inactive, details::bit_inactive_policy<Options...>());
  }
}

/**
 * Shifts every lane left by count bits. The operation is defined on the
 * unsigned bit pattern of each lane, including for signed element types.
 * The count must be non-negative; passing a negative runtime scalar or a
 * per-lane vector containing a negative count has undefined behavior. A
 * negative meta::Const is rejected at compile time. Scalar counts at least
 * the element width produce zero in every lane; per-lane Vec counts produce
 * zero lane-by-lane. A meta::Const supplies a compile-time immediate, an int
 * supplies a runtime scalar, and a Vec count applies independently per lane.
 * @see shr for right shift.
 */
template <IntegerTag Tag>
VECOPS_ALWAYS_INLINE Vec<Tag> BitShiftLeftOp::operator()(
    Tag tag, Vec<Tag> value, int count) const {
  return details::execute(*this, tag, value, count);
}

template <nint_t Count, IntegerTag Tag>
  requires (Count >= 0)
VECOPS_ALWAYS_INLINE Vec<Tag> BitShiftLeftOp::operator()(
    Tag tag, Vec<Tag> value, meta::Const<Count> count) const {
  return details::execute(*this, tag, value, count);
}

template <IntegerTag Tag>
VECOPS_ALWAYS_INLINE Vec<Tag> BitShiftLeftOp::operator()(
    Tag tag, Vec<Tag> value, Vec<Tag> counts) const {
  return details::execute(*this, tag, value, counts);
}

/**
 * `opt::unmasked` applies the operation to every lane. With exactly one
 * `opt::masked(mask)`, applies the left-shift semantics to
 * active lanes. Inactive lanes preserve value[i], or use the optional
 * population policy. Counts must satisfy the same non-negative precondition
 * as the unmasked operation.
 */
template <IntegerTag Tag, typename... Options>
  requires (sizeof...(Options) > 0)
VECOPS_ALWAYS_INLINE Vec<Tag> BitShiftLeftOp::operator()(
    Tag tag, Vec<Tag> value, int count, Options&&... options) const {
  return details::execute_bit_shift_options(
      *this, tag, value, count, std::forward<Options>(options)...);
}

template <nint_t Count, IntegerTag Tag, typename... Options>
  requires (Count >= 0 && sizeof...(Options) > 0)
VECOPS_ALWAYS_INLINE Vec<Tag> BitShiftLeftOp::operator()(
    Tag tag, Vec<Tag> value, meta::Const<Count> count,
    Options&&... options) const {
  return details::execute_bit_shift_options(
      *this, tag, value, count, std::forward<Options>(options)...);
}

template <IntegerTag Tag, typename... Options>
  requires (sizeof...(Options) > 0)
VECOPS_ALWAYS_INLINE Vec<Tag> BitShiftLeftOp::operator()(
    Tag tag, Vec<Tag> value, Vec<Tag> counts, Options&&... options) const {
  return details::execute_bit_shift_options(
      *this, tag, value, counts, std::forward<Options>(options)...);
}

/**
 * Shifts every lane right by count bits. The count must be non-negative;
 * passing a negative runtime scalar or a per-lane vector containing a
 * negative count has undefined behavior, while a negative meta::Const is
 * rejected at compile time. Unsigned lanes shift logically to zero for
 * counts at least the element width. Signed lanes shift arithmetically and
 * become their sign fill for large counts. Counts may be a runtime int, a
 * meta::Const immediate, or a same-Tag Vec.
 * @see shl for left shift.
 */
template <IntegerTag Tag>
VECOPS_ALWAYS_INLINE Vec<Tag> BitShiftRightOp::operator()(
    Tag tag, Vec<Tag> value, int count) const {
  return details::execute(*this, tag, value, count);
}

template <nint_t Count, IntegerTag Tag>
  requires (Count >= 0)
VECOPS_ALWAYS_INLINE Vec<Tag> BitShiftRightOp::operator()(
    Tag tag, Vec<Tag> value, meta::Const<Count> count) const {
  return details::execute(*this, tag, value, count);
}

template <IntegerTag Tag>
VECOPS_ALWAYS_INLINE Vec<Tag> BitShiftRightOp::operator()(
    Tag tag, Vec<Tag> value, Vec<Tag> counts) const {
  return details::execute(*this, tag, value, counts);
}

/**
 * `opt::unmasked` applies the operation to every lane. With exactly one
 * `opt::masked(mask)`, applies the right-shift semantics to
 * active lanes. Inactive lanes preserve value[i], or use the optional
 * population policy. Counts must satisfy the same non-negative precondition
 * as the unmasked operation.
 */
template <IntegerTag Tag, typename... Options>
  requires (sizeof...(Options) > 0)
VECOPS_ALWAYS_INLINE Vec<Tag> BitShiftRightOp::operator()(
    Tag tag, Vec<Tag> value, int count, Options&&... options) const {
  return details::execute_bit_shift_options(
      *this, tag, value, count, std::forward<Options>(options)...);
}

template <nint_t Count, IntegerTag Tag, typename... Options>
  requires (Count >= 0 && sizeof...(Options) > 0)
VECOPS_ALWAYS_INLINE Vec<Tag> BitShiftRightOp::operator()(
    Tag tag, Vec<Tag> value, meta::Const<Count> count,
    Options&&... options) const {
  return details::execute_bit_shift_options(
      *this, tag, value, count, std::forward<Options>(options)...);
}

template <IntegerTag Tag, typename... Options>
  requires (sizeof...(Options) > 0)
VECOPS_ALWAYS_INLINE Vec<Tag> BitShiftRightOp::operator()(
    Tag tag, Vec<Tag> value, Vec<Tag> counts, Options&&... options) const {
  return details::execute_bit_shift_options(
      *this, tag, value, counts, std::forward<Options>(options)...);
}

/**
 * Applies an integer bit-count operation independently to every lane.
 * Signed elements are interpreted as their unsigned bit pattern. Leading
 * and trailing counts return the element bit width for the all-zero (or,
 * for one counts, all-one) input. Results retain the input Tag.
 */
#define VECOPS_VEC_DEFINE_BIT_COUNT(OpType)                              \
  template <IntegerTag Tag>                                             \
  VECOPS_ALWAYS_INLINE Vec<Tag> OpType::operator()(                     \
      Tag tag, Vec<Tag> value) const {                                  \
    return details::execute(*this, tag, value);                         \
  }                                                                     \
  template <IntegerTag Tag, typename... Options>                        \
    requires (sizeof...(Options) > 0)                                   \
  VECOPS_ALWAYS_INLINE Vec<Tag> OpType::operator()(                     \
      Tag tag, Vec<Tag> value, Options&&... options) const {            \
    return details::execute_bit_unary_options(                          \
        *this, tag, value, std::forward<Options>(options)...);          \
  }

VECOPS_VEC_DEFINE_BIT_COUNT(PopCountOp);
VECOPS_VEC_DEFINE_BIT_COUNT(CountLeadingZeroOp);
VECOPS_VEC_DEFINE_BIT_COUNT(CountLeadingOneOp);
VECOPS_VEC_DEFINE_BIT_COUNT(CountTrailingZeroOp);
VECOPS_VEC_DEFINE_BIT_COUNT(CountTrailingOneOp);

#undef VECOPS_VEC_DEFINE_BIT_COUNT

/**
 * Rotates each lane without mixing bits between lanes. Counts are reduced
 * modulo the element bit width. Negative counts rotate in the opposite
 * direction, matching std::rotl/std::rotr. Counts may be a compile-time
 * meta::Const, a runtime int shared by all lanes, or a same-Tag Vec.
 */
#define VECOPS_VEC_DEFINE_ROTATE(OpType)                                 \
  template <IntegerTag Tag>                                             \
  VECOPS_ALWAYS_INLINE Vec<Tag> OpType::operator()(                     \
      Tag tag, Vec<Tag> value, int count) const {                       \
    return details::execute(*this, tag, value, count);                  \
  }                                                                     \
  template <nint_t Count, IntegerTag Tag>                               \
  VECOPS_ALWAYS_INLINE Vec<Tag> OpType::operator()(                     \
      Tag tag, Vec<Tag> value, meta::Const<Count> count) const {        \
    return details::execute(*this, tag, value, count);                  \
  }                                                                     \
  template <IntegerTag Tag>                                             \
  VECOPS_ALWAYS_INLINE Vec<Tag> OpType::operator()(                     \
      Tag tag, Vec<Tag> value, Vec<Tag> counts) const {                 \
    return details::execute(*this, tag, value, counts);                 \
  }                                                                     \
  template <IntegerTag Tag, typename... Options>                        \
    requires (sizeof...(Options) > 0)                                   \
  VECOPS_ALWAYS_INLINE Vec<Tag> OpType::operator()(                     \
      Tag tag, Vec<Tag> value, int count, Options&&... options) const { \
    return details::execute_bit_shift_options(                          \
        *this, tag, value, count, std::forward<Options>(options)...);   \
  }                                                                     \
  template <nint_t Count, IntegerTag Tag, typename... Options>          \
    requires (sizeof...(Options) > 0)                                   \
  VECOPS_ALWAYS_INLINE Vec<Tag> OpType::operator()(                     \
      Tag tag, Vec<Tag> value, meta::Const<Count> count,                \
      Options&&... options) const {                                     \
    return details::execute_bit_shift_options(                          \
        *this, tag, value, count, std::forward<Options>(options)...);   \
  }                                                                     \
  template <IntegerTag Tag, typename... Options>                        \
    requires (sizeof...(Options) > 0)                                   \
  VECOPS_ALWAYS_INLINE Vec<Tag> OpType::operator()(                     \
      Tag tag, Vec<Tag> value, Vec<Tag> counts,                         \
      Options&&... options) const {                                     \
    return details::execute_bit_shift_options(                          \
        *this, tag, value, counts, std::forward<Options>(options)...);  \
  }

VECOPS_VEC_DEFINE_ROTATE(RotateLeftOp);
VECOPS_VEC_DEFINE_ROTATE(RotateRightOp);

#undef VECOPS_VEC_DEFINE_ROTATE

} // namespace vecops::vec

#endif // VECOPS_VEC_BIT_H
