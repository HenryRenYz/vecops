#ifndef VECOPS_VEC_BIT_H
#define VECOPS_VEC_BIT_H

#include <concepts>
#include <utility>

#include "vecops/gemm/Layout.h"
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
          VecToTagT<V>{}, a, b, std::forward<Options>(options)...);     \
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
        VecToTagT<V>{}, value, std::forward<Options>(options)...);
  }
};

/* **************************************************************************** */
//    Bit shifts: bit_shl, bit_shr                                        //
/* **************************************************************************** */

struct BitShiftLeftOp {
  template <IntegerTag Tag>
  VECOPS_ALWAYS_INLINE Vec<Tag> operator()(
      Tag tag, Vec<Tag> value, int count) const;

  template <nint_t Count, IntegerTag Tag>
  VECOPS_ALWAYS_INLINE Vec<Tag> operator()(
      Tag tag, Vec<Tag> value, gemm::Const<Count> count) const;

  template <IntegerTag Tag, nint_t Alignment, nint_t Lo, nint_t Hi>
  VECOPS_ALWAYS_INLINE Vec<Tag> operator()(
      Tag tag, Vec<Tag> value,
      gemm::Dynamic<Alignment, Lo, Hi> count) const;

  template <IntegerTag Tag>
  VECOPS_ALWAYS_INLINE Vec<Tag> operator()(
      Tag tag, Vec<Tag> value, Vec<Tag> counts) const;

  template <IntegerTag Tag, typename... Options>
    requires (sizeof...(Options) > 0)
  VECOPS_ALWAYS_INLINE Vec<Tag> operator()(
      Tag tag, Vec<Tag> value, int count, Options&&... options) const;

  template <nint_t Count, IntegerTag Tag, typename... Options>
    requires (sizeof...(Options) > 0)
  VECOPS_ALWAYS_INLINE Vec<Tag> operator()(
      Tag tag, Vec<Tag> value, gemm::Const<Count> count,
      Options&&... options) const;

  template <IntegerTag Tag, nint_t Alignment, nint_t Lo, nint_t Hi,
            typename... Options>
    requires (sizeof...(Options) > 0)
  VECOPS_ALWAYS_INLINE Vec<Tag> operator()(
      Tag tag, Vec<Tag> value, gemm::Dynamic<Alignment, Lo, Hi> count,
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
          VecToTagT<V>{}, value, std::forward<Count>(count),
          std::forward<Options>(options)...);
    }
  VECOPS_ALWAYS_INLINE V operator()(
      V value, Count&& count, Options&&... options) const {
    return (*this)(
        VecToTagT<V>{}, value, std::forward<Count>(count),
        std::forward<Options>(options)...);
  }
};

struct BitShiftRightOp {
  template <IntegerTag Tag>
  VECOPS_ALWAYS_INLINE Vec<Tag> operator()(
      Tag tag, Vec<Tag> value, int count) const;

  template <nint_t Count, IntegerTag Tag>
  VECOPS_ALWAYS_INLINE Vec<Tag> operator()(
      Tag tag, Vec<Tag> value, gemm::Const<Count> count) const;

  template <IntegerTag Tag, nint_t Alignment, nint_t Lo, nint_t Hi>
  VECOPS_ALWAYS_INLINE Vec<Tag> operator()(
      Tag tag, Vec<Tag> value,
      gemm::Dynamic<Alignment, Lo, Hi> count) const;

  template <IntegerTag Tag>
  VECOPS_ALWAYS_INLINE Vec<Tag> operator()(
      Tag tag, Vec<Tag> value, Vec<Tag> counts) const;

  template <IntegerTag Tag, typename... Options>
    requires (sizeof...(Options) > 0)
  VECOPS_ALWAYS_INLINE Vec<Tag> operator()(
      Tag tag, Vec<Tag> value, int count, Options&&... options) const;

  template <nint_t Count, IntegerTag Tag, typename... Options>
    requires (sizeof...(Options) > 0)
  VECOPS_ALWAYS_INLINE Vec<Tag> operator()(
      Tag tag, Vec<Tag> value, gemm::Const<Count> count,
      Options&&... options) const;

  template <IntegerTag Tag, nint_t Alignment, nint_t Lo, nint_t Hi,
            typename... Options>
    requires (sizeof...(Options) > 0)
  VECOPS_ALWAYS_INLINE Vec<Tag> operator()(
      Tag tag, Vec<Tag> value, gemm::Dynamic<Alignment, Lo, Hi> count,
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
          VecToTagT<V>{}, value, std::forward<Count>(count),
          std::forward<Options>(options)...);
    }
  VECOPS_ALWAYS_INLINE V operator()(
      V value, Count&& count, Options&&... options) const {
    return (*this)(
        VecToTagT<V>{}, value, std::forward<Count>(count),
        std::forward<Options>(options)...);
  }
};

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
      details::option_count<details::IsUnmaskedOption, Options...> == 1) {
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
      details::option_count<details::IsUnmaskedOption, Options...> == 1) {
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
      details::option_count<details::IsUnmaskedOption, Options...> == 1) {
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
      details::option_count<details::IsUnmaskedOption, Options...> == 1) {
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
      details::option_count<details::IsUnmaskedOption, Options...> == 1) {
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
 * Negative counts preserve the input (no-op). Scalar counts at least the
 * element width produce zero in every lane; per-lane Vec counts produce
 * zero lane-by-lane. A gemm::Const supplies a compile-time count,
 * gemm::Dynamic supplies a checked runtime count, and a Vec count applies
 * independently per lane. @see bit_shr for right shift.
 */
template <IntegerTag Tag>
VECOPS_ALWAYS_INLINE Vec<Tag> BitShiftLeftOp::operator()(
    Tag tag, Vec<Tag> value, int count) const {
  if (count < 0) return value;
  return details::execute(*this, tag, value, count);
}

template <nint_t Count, IntegerTag Tag>
VECOPS_ALWAYS_INLINE Vec<Tag> BitShiftLeftOp::operator()(
    Tag tag, Vec<Tag> value, gemm::Const<Count>) const {
  return (*this)(tag, value, static_cast<int>(Count));
}

template <IntegerTag Tag, nint_t Alignment, nint_t Lo, nint_t Hi>
VECOPS_ALWAYS_INLINE Vec<Tag> BitShiftLeftOp::operator()(
    Tag tag, Vec<Tag> value,
    gemm::Dynamic<Alignment, Lo, Hi> count) const {
  return (*this)(tag, value, static_cast<int>(
      static_cast<nint_t>(count)));
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
 * population policy. Negative counts preserve every lane.
 */
template <IntegerTag Tag, typename... Options>
  requires (sizeof...(Options) > 0)
VECOPS_ALWAYS_INLINE Vec<Tag> BitShiftLeftOp::operator()(
    Tag tag, Vec<Tag> value, int count, Options&&... options) const {
  if (count < 0) return value;
  return details::execute_bit_shift_options(
      *this, tag, value, count, std::forward<Options>(options)...);
}

template <nint_t Count, IntegerTag Tag, typename... Options>
  requires (sizeof...(Options) > 0)
VECOPS_ALWAYS_INLINE Vec<Tag> BitShiftLeftOp::operator()(
    Tag tag, Vec<Tag> value, gemm::Const<Count>, Options&&... options) const {
  return (*this)(
      tag, value, static_cast<int>(Count),
      std::forward<Options>(options)...);
}

template <IntegerTag Tag, nint_t Alignment, nint_t Lo, nint_t Hi,
          typename... Options>
  requires (sizeof...(Options) > 0)
VECOPS_ALWAYS_INLINE Vec<Tag> BitShiftLeftOp::operator()(
    Tag tag, Vec<Tag> value, gemm::Dynamic<Alignment, Lo, Hi> count,
    Options&&... options) const {
  return (*this)(
      tag, value, static_cast<int>(static_cast<nint_t>(count)),
      std::forward<Options>(options)...);
}

template <IntegerTag Tag, typename... Options>
  requires (sizeof...(Options) > 0)
VECOPS_ALWAYS_INLINE Vec<Tag> BitShiftLeftOp::operator()(
    Tag tag, Vec<Tag> value, Vec<Tag> counts, Options&&... options) const {
  return details::execute_bit_shift_options(
      *this, tag, value, counts, std::forward<Options>(options)...);
}

/**
 * Shifts every lane right by count bits. Negative counts preserve the input
 * (no-op). Unsigned lanes shift logically to zero for counts at least the
 * element width. Signed lanes shift arithmetically and become their sign
 * fill (all-ones for negative, all-zeros for non-negative) for large counts.
 * Counts may be scalar, gemm::Const, gemm::Dynamic, or a same-Tag Vec.
 * @see bit_shl for left shift.
 */
template <IntegerTag Tag>
VECOPS_ALWAYS_INLINE Vec<Tag> BitShiftRightOp::operator()(
    Tag tag, Vec<Tag> value, int count) const {
  if (count < 0) return value;
  return details::execute(*this, tag, value, count);
}

template <nint_t Count, IntegerTag Tag>
VECOPS_ALWAYS_INLINE Vec<Tag> BitShiftRightOp::operator()(
    Tag tag, Vec<Tag> value, gemm::Const<Count>) const {
  return (*this)(tag, value, static_cast<int>(Count));
}

template <IntegerTag Tag, nint_t Alignment, nint_t Lo, nint_t Hi>
VECOPS_ALWAYS_INLINE Vec<Tag> BitShiftRightOp::operator()(
    Tag tag, Vec<Tag> value,
    gemm::Dynamic<Alignment, Lo, Hi> count) const {
  return (*this)(tag, value, static_cast<int>(
      static_cast<nint_t>(count)));
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
 * population policy. Negative counts preserve every lane.
 */
template <IntegerTag Tag, typename... Options>
  requires (sizeof...(Options) > 0)
VECOPS_ALWAYS_INLINE Vec<Tag> BitShiftRightOp::operator()(
    Tag tag, Vec<Tag> value, int count, Options&&... options) const {
  if (count < 0) return value;
  return details::execute_bit_shift_options(
      *this, tag, value, count, std::forward<Options>(options)...);
}

template <nint_t Count, IntegerTag Tag, typename... Options>
  requires (sizeof...(Options) > 0)
VECOPS_ALWAYS_INLINE Vec<Tag> BitShiftRightOp::operator()(
    Tag tag, Vec<Tag> value, gemm::Const<Count>, Options&&... options) const {
  return (*this)(
      tag, value, static_cast<int>(Count),
      std::forward<Options>(options)...);
}

template <IntegerTag Tag, nint_t Alignment, nint_t Lo, nint_t Hi,
          typename... Options>
  requires (sizeof...(Options) > 0)
VECOPS_ALWAYS_INLINE Vec<Tag> BitShiftRightOp::operator()(
    Tag tag, Vec<Tag> value, gemm::Dynamic<Alignment, Lo, Hi> count,
    Options&&... options) const {
  return (*this)(
      tag, value, static_cast<int>(static_cast<nint_t>(count)),
      std::forward<Options>(options)...);
}

template <IntegerTag Tag, typename... Options>
  requires (sizeof...(Options) > 0)
VECOPS_ALWAYS_INLINE Vec<Tag> BitShiftRightOp::operator()(
    Tag tag, Vec<Tag> value, Vec<Tag> counts, Options&&... options) const {
  return details::execute_bit_shift_options(
      *this, tag, value, counts, std::forward<Options>(options)...);
}

inline constexpr BitAndOp bit_and{};
inline constexpr BitOrOp bit_or{};
inline constexpr BitXorOp bit_xor{};
inline constexpr BitAndNotOp bit_andnot{};
inline constexpr BitNotOp bit_not{};
inline constexpr BitShiftLeftOp bit_shl{};
inline constexpr BitShiftRightOp bit_shr{};

} // namespace vecops::vec

#endif // VECOPS_VEC_BIT_H
