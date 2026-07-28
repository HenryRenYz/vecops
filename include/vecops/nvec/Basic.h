#ifndef VECOPS_NVEC_BASIC_H
#define VECOPS_NVEC_BASIC_H

#include <cassert>
#include <limits>
#include <type_traits>
#include <utility>

#include "vecops/nvec/Options.h"

namespace vecops::nvec {

struct FillOp {
  template <VectorTag Tag>
  VECOPS_ALWAYS_INLINE Vec<Tag> operator()(
      Tag tag, ElementOf<Tag> value) const;

  template <VectorTag Tag, typename... Options>
    requires (sizeof...(Options) > 0)
  VECOPS_ALWAYS_INLINE Vec<Tag> operator()(
      Tag tag, ElementOf<Tag> value, Options&&... options) const;
};

struct MaskFillOp {
  template <VectorTag Tag>
  VECOPS_ALWAYS_INLINE Mask<Tag> operator()(Tag tag, bool value) const;
};

struct MaskWhileLtOp {
  template <VectorTag Tag>
  VECOPS_ALWAYS_INLINE Mask<Tag> operator()(
      Tag tag, nint_t a, nint_t b) const;
};

struct MaskWhileGeOp {
  template <VectorTag Tag>
  VECOPS_ALWAYS_INLINE Mask<Tag> operator()(
      Tag tag, nint_t a, nint_t b) const;
};

struct BlendOp {
  template <VectorTag Tag>
  VECOPS_ALWAYS_INLINE Vec<Tag> operator()(
      Tag tag, Vec<Tag> false_value, Mask<Tag> mask,
      Vec<Tag> true_value) const;
};

struct MaskAndOp {
  template <VectorTag Tag>
  VECOPS_ALWAYS_INLINE Mask<Tag> operator()(
      Tag tag, Mask<Tag> a, Mask<Tag> b) const;
};

struct MaskOrOp {
  template <VectorTag Tag>
  VECOPS_ALWAYS_INLINE Mask<Tag> operator()(
      Tag tag, Mask<Tag> a, Mask<Tag> b) const;
};

struct MaskXorOp {
  template <VectorTag Tag>
  VECOPS_ALWAYS_INLINE Mask<Tag> operator()(
      Tag tag, Mask<Tag> a, Mask<Tag> b) const;
};

struct MaskAndNotOp {
  template <VectorTag Tag>
  VECOPS_ALWAYS_INLINE Mask<Tag> operator()(
      Tag tag, Mask<Tag> a, Mask<Tag> b) const;
};

struct MaskNotOp {
  template <VectorTag Tag>
  VECOPS_ALWAYS_INLINE Mask<Tag> operator()(
      Tag tag, Mask<Tag> value) const;
};

struct GetVecLaneOp {};
struct SetVecLaneOp {};
struct GetMaskLaneOp {};
struct SetMaskLaneOp {};

struct GetOp {
  template <VectorTag Tag>
  VECOPS_ALWAYS_INLINE ElementOf<Tag> operator()(
      Tag tag, Vec<Tag> value, nint_t index) const;

  template <VectorTag Tag>
  VECOPS_ALWAYS_INLINE bool operator()(
      Tag tag, Mask<Tag> value, nint_t index) const;
};

struct SetOp {
  template <VectorTag Tag>
  VECOPS_ALWAYS_INLINE Vec<Tag> operator()(
      Tag tag, Vec<Tag> value, nint_t index, ElementOf<Tag> lane) const;

  template <VectorTag Tag>
  VECOPS_ALWAYS_INLINE Mask<Tag> operator()(
      Tag tag, Mask<Tag> value, nint_t index, bool lane) const;
};

struct BitCastOp {
  template <VectorTag ToTag, VectorTag FromTag>
  VECOPS_ALWAYS_INLINE Vec<ToTag> operator()(
      ToTag to, FromTag from, Vec<FromTag> value) const;
};

struct LowerOp {
  template <VectorTag Tag>
  VECOPS_ALWAYS_INLINE Vec<Half<Tag>> operator()(
      Tag tag, Vec<Tag> value) const;

  template <VectorTag Tag>
  VECOPS_ALWAYS_INLINE Mask<Half<Tag>> operator()(
      Tag tag, Mask<Tag> value) const;
};

struct UpperOp {
  template <VectorTag Tag>
  VECOPS_ALWAYS_INLINE Vec<Half<Tag>> operator()(
      Tag tag, Vec<Tag> value) const;

  template <VectorTag Tag>
  VECOPS_ALWAYS_INLINE Mask<Half<Tag>> operator()(
      Tag tag, Mask<Tag> value) const;
};

struct ConcatOp {
  template <VectorTag Tag>
  VECOPS_ALWAYS_INLINE Vec<Tag> operator()(
      Tag tag, Vec<Half<Tag>> lower_value,
      Vec<Half<Tag>> upper_value) const;

  template <VectorTag Tag>
  VECOPS_ALWAYS_INLINE Mask<Tag> operator()(
      Tag tag, Mask<Half<Tag>> lower_value,
      Mask<Half<Tag>> upper_value) const;
};

struct EvenOp {
  template <VectorTag Tag>
  VECOPS_ALWAYS_INLINE Vec<Half<Tag>> operator()(
      Tag tag, Vec<Tag> value) const;
};

struct OddOp {
  template <VectorTag Tag>
  VECOPS_ALWAYS_INLINE Vec<Half<Tag>> operator()(
      Tag tag, Vec<Tag> value) const;
};

struct ConcatEvenOp {
  template <VectorTag Tag>
  VECOPS_ALWAYS_INLINE Vec<Tag> operator()(
      Tag tag, Vec<Tag> a, Vec<Tag> b) const;
};

struct ConcatOddOp {
  template <VectorTag Tag>
  VECOPS_ALWAYS_INLINE Vec<Tag> operator()(
      Tag tag, Vec<Tag> a, Vec<Tag> b) const;
};

struct InterleaveOp {
  template <VectorTag Tag>
  VECOPS_ALWAYS_INLINE Vec<Tag> operator()(
      Tag tag, Vec<Half<Tag>> a, Vec<Half<Tag>> b) const;
};

struct InterleaveEvenOp {
  template <VectorTag Tag>
  VECOPS_ALWAYS_INLINE Vec<Tag> operator()(
      Tag tag, Vec<Tag> a, Vec<Tag> b) const;
};

struct InterleaveOddOp {
  template <VectorTag Tag>
  VECOPS_ALWAYS_INLINE Vec<Tag> operator()(
      Tag tag, Vec<Tag> a, Vec<Tag> b) const;
};

struct LocalInterleaveLowerOp {
  template <VectorTag Tag>
  VECOPS_ALWAYS_INLINE Vec<Tag> operator()(
      Tag tag, Vec<Tag> a, Vec<Tag> b) const;
};

struct LocalInterleaveUpperOp {
  template <VectorTag Tag>
  VECOPS_ALWAYS_INLINE Vec<Tag> operator()(
      Tag tag, Vec<Tag> a, Vec<Tag> b) const;
};

struct ShuffleOp {
  template <VectorTag Tag>
  VECOPS_ALWAYS_INLINE Vec<Tag> operator()(
      Tag tag, Vec<Tag> value, Vec<IndexTag<Tag>> indices) const;
};

struct LocalShuffleOp {
  template <VectorTag Tag>
  VECOPS_ALWAYS_INLINE Vec<Tag> operator()(
      Tag tag, Vec<Tag> value, Vec<IndexTag<Tag>> indices) const;

  template <int... Indices, VectorTag Tag>
  VECOPS_ALWAYS_INLINE Vec<Tag> operator()(
      Tag tag, Vec<Tag> value, opt::LaneOrder<Indices...>) const;

  template <VectorTag Tag, typename... Indices>
    requires (std::integral<std::remove_cvref_t<Indices>> && ...)
  VECOPS_ALWAYS_INLINE Vec<Tag> operator()(
      Tag tag, Vec<Tag> value, Indices... indices) const;
};

inline constexpr FillOp fill{};
inline constexpr MaskFillOp mfill{};
inline constexpr MaskWhileLtOp mwhilelt{};
inline constexpr MaskWhileGeOp mwhilege{};
inline constexpr BlendOp blend{};
inline constexpr MaskAndOp mask_and{};
inline constexpr MaskOrOp mask_or{};
inline constexpr MaskXorOp mask_xor{};
inline constexpr MaskAndNotOp mask_andnot{};
inline constexpr MaskNotOp mask_not{};
inline constexpr GetOp get{};
inline constexpr SetOp set{};
inline constexpr BitCastOp bitcast{};
inline constexpr LowerOp lower{};
inline constexpr UpperOp upper{};
inline constexpr ConcatOp concat{};
inline constexpr EvenOp even{};
inline constexpr OddOp odd{};
inline constexpr ConcatEvenOp concat_even{};
inline constexpr ConcatOddOp concat_odd{};
inline constexpr InterleaveOp interleave{};
inline constexpr InterleaveEvenOp interleave_even{};
inline constexpr InterleaveOddOp interleave_odd{};
inline constexpr LocalInterleaveLowerOp local_interleave_lower{};
inline constexpr LocalInterleaveUpperOp local_interleave_upper{};
inline constexpr ShuffleOp shuf{};
inline constexpr LocalShuffleOp local_shuf{};

struct ZerosOp {
  /** Returns a vector with result[i] = ElementOf<Tag>{} for every logical lane. */
  template <VectorTag Tag>
  VECOPS_ALWAYS_INLINE Vec<Tag> operator()(Tag tag) const {
    return fill(tag, ElementOf<Tag>{});
  }
};

struct MaskTrueOp {
  /** Returns a mask whose every logical lane is true. */
  template <VectorTag Tag>
  VECOPS_ALWAYS_INLINE Mask<Tag> operator()(Tag tag) const {
    return mfill(tag, true);
  }
};

struct MaskFalseOp {
  /** Returns a mask whose every logical lane is false. */
  template <VectorTag Tag>
  VECOPS_ALWAYS_INLINE Mask<Tag> operator()(Tag tag) const {
    return mfill(tag, false);
  }
};

struct MaskWhileLeOp {
  /** Returns mask i = (a + i <= b) for 0 <= i < size(tag). */
  template <VectorTag Tag>
  VECOPS_ALWAYS_INLINE Mask<Tag> operator()(
      Tag tag, nint_t a, nint_t b) const {
    return b == std::numeric_limits<nint_t>::max()
        ? mfill(tag, true)
        : mwhilelt(tag, a, b + 1);
  }
};

struct MaskWhileGtOp {
  /** Returns mask i = (a + i > b) for 0 <= i < size(tag). */
  template <VectorTag Tag>
  VECOPS_ALWAYS_INLINE Mask<Tag> operator()(
      Tag tag, nint_t a, nint_t b) const {
    return b == std::numeric_limits<nint_t>::max()
        ? mfill(tag, false)
        : mwhilege(tag, a, b + 1);
  }
};

inline constexpr ZerosOp zeros{};
inline constexpr MaskTrueOp mtrue{};
inline constexpr MaskFalseOp mfalse{};
inline constexpr MaskWhileLeOp mwhilele{};
inline constexpr MaskWhileGtOp mwhilegt{};

} // namespace vecops::nvec

// Backend specializations must precede the operator() definitions below so
// that details::execute() can resolve them during template instantiation.
#include "vecops/nvec/details/Dispatch.h"

#include "vecops/nvec/details/scalar/Basic.h"

#if defined(ARCH_X86_FAMILY) && !defined(CPU_CAPABILITY_GENERIC)
#include "vecops/nvec/details/x86/Basic.h"
#elif defined(CPU_CAPABILITY_SVE)
#include "vecops/nvec/details/sve/Basic.h"
#endif

#include "vecops/nvec/details/Basic.h"

namespace vecops::nvec {

/**
 * Returns a vector with result[i] = value for every logical lane.
 *
 * Floating-point values, including NaN payloads, infinities, subnormals and
 * signed zero, are copied bit-for-bit from value into every lane.
 */
template <VectorTag Tag>
VECOPS_ALWAYS_INLINE Vec<Tag> FillOp::operator()(
    Tag tag, ElementOf<Tag> value) const {
  return details::execute(*this, tag, value);
}

/**
 * Returns a conditionally filled vector.
 *
 * `opt::masked(mask)` activates lane i when mask[i] is true.
 * `opt::first(count)` activates 0 <= i < clamp(count, 0, size(tag)).
 * Inactive lanes are zero unless exactly one `opt::merge(vector_or_scalar)`
 * option is present. The active selectors are mutually exclusive.
 */
template <VectorTag Tag, typename... Options>
  requires (sizeof...(Options) > 0)
VECOPS_ALWAYS_INLINE Vec<Tag> FillOp::operator()(
    Tag tag, ElementOf<Tag> value, Options&&... options) const {
  static_assert(
      (details::is_fill_option_for<Tag, Options> && ...),
      "fill received an option with the wrong kind or value type");
  constexpr std::size_t masked_count =
      (std::size_t{0} + ... + std::size_t{details::is_masked_option<Options>});
  constexpr std::size_t first_count =
      (std::size_t{0} + ... + std::size_t{details::is_first_option<Options>});
  constexpr std::size_t vector_merge_count =
      (std::size_t{0} + ... +
       std::size_t{details::is_vector_merge_option<Options>});
  constexpr std::size_t scalar_merge_count =
      (std::size_t{0} + ... +
       std::size_t{details::is_scalar_merge_option<Options>});
  static_assert(
      masked_count + first_count == 1,
      "fill requires exactly one active-lane option");
  static_assert(
      vector_merge_count + scalar_merge_count <= 1,
      "fill accepts at most one merge option");

  const auto active_value = fill(tag, value);
  const auto inactive_value = [&]() -> Vec<Tag> {
    if constexpr (vector_merge_count == 1) {
      return details::find_option<details::IsVectorMergeOption>(
          std::forward<Options>(options)...).value;
    } else if constexpr (scalar_merge_count == 1) {
      return fill(
          tag,
          details::find_option<details::IsScalarMergeOption>(
              std::forward<Options>(options)...).value);
    } else {
      return zeros(tag);
    }
  }();

  if constexpr (masked_count == 1) {
    return blend(
        tag,
        inactive_value,
        details::find_option<details::IsMaskedOption>(
            std::forward<Options>(options)...).value,
        active_value);
  } else {
    const nint_t requested =
        details::find_option<details::IsFirstOption>(
            std::forward<Options>(options)...).count;
    const nint_t count = requested < 0
        ? 0
        : (requested > size(tag) ? size(tag) : requested);
    return blend(tag, inactive_value, mwhilelt(tag, 0, count), active_value);
  }
}

/** Returns a mask with result[i] = value for every logical lane. */
template <VectorTag Tag>
VECOPS_ALWAYS_INLINE Mask<Tag> MaskFillOp::operator()(
    Tag tag, bool value) const {
  return details::execute(*this, tag, value);
}

/** Returns a mask with result[i] = (a + i < b). */
template <VectorTag Tag>
VECOPS_ALWAYS_INLINE Mask<Tag> MaskWhileLtOp::operator()(
    Tag tag, nint_t a, nint_t b) const {
  return details::execute(*this, tag, a, b);
}

/** Returns a mask with result[i] = (a + i >= b). */
template <VectorTag Tag>
VECOPS_ALWAYS_INLINE Mask<Tag> MaskWhileGeOp::operator()(
    Tag tag, nint_t a, nint_t b) const {
  return details::execute(*this, tag, a, b);
}

/** Returns result[i] = mask[i] ? true_value[i] : false_value[i]. */
template <VectorTag Tag>
VECOPS_ALWAYS_INLINE Vec<Tag> BlendOp::operator()(
    Tag tag, Vec<Tag> false_value, Mask<Tag> mask,
    Vec<Tag> true_value) const {
  return details::execute(*this, tag, false_value, mask, true_value);
}

#define VECOPS_NVEC_DEFINE_MASK_BINARY_CALL(OpType)                        \
  template <VectorTag Tag>                                                \
  VECOPS_ALWAYS_INLINE Mask<Tag> OpType::operator()(                       \
      Tag tag, Mask<Tag> a, Mask<Tag> b) const {                          \
    return details::execute(*this, tag, a, b);                            \
  }

/** Returns result[i] = a[i] && b[i]. */
VECOPS_NVEC_DEFINE_MASK_BINARY_CALL(MaskAndOp)
/** Returns result[i] = a[i] || b[i]. */
VECOPS_NVEC_DEFINE_MASK_BINARY_CALL(MaskOrOp)
/** Returns result[i] = (a[i] != b[i]). */
VECOPS_NVEC_DEFINE_MASK_BINARY_CALL(MaskXorOp)
/** Returns result[i] = (!a[i]) && b[i]. */
VECOPS_NVEC_DEFINE_MASK_BINARY_CALL(MaskAndNotOp)

#undef VECOPS_NVEC_DEFINE_MASK_BINARY_CALL

/** Returns result[i] = !value[i]. */
template <VectorTag Tag>
VECOPS_ALWAYS_INLINE Mask<Tag> MaskNotOp::operator()(
    Tag tag, Mask<Tag> value) const {
  return details::execute(*this, tag, value);
}

/** Returns value[index], where 0 <= index < size(tag). */
template <VectorTag Tag>
VECOPS_ALWAYS_INLINE ElementOf<Tag> GetOp::operator()(
    Tag tag, Vec<Tag> value, nint_t index) const {
  assert(index >= 0 && index < size(tag));
  const nint_t word_lanes = native_word_size(tag);
  return details::visit_runtime_word<details::CurrentBackend>(
      tag,
      index / word_lanes,
      [&]<nint_t Index>() {
        return details::execute_word<Index, details::CurrentBackend>(
            GetVecLaneOp{}, tag, get_word<Index>(tag, value),
            index % word_lanes);
      });
}

/** Returns mask[index], where 0 <= index < size(tag). */
template <VectorTag Tag>
VECOPS_ALWAYS_INLINE bool GetOp::operator()(
    Tag tag, Mask<Tag> value, nint_t index) const {
  assert(index >= 0 && index < size(tag));
  const nint_t word_lanes = native_word_size(tag);
  return details::visit_runtime_word<details::CurrentBackend>(
      tag,
      index / word_lanes,
      [&]<nint_t Index>() {
        return details::execute_word<Index, details::CurrentBackend>(
            GetMaskLaneOp{}, tag, get_word<Index>(tag, value),
            index % word_lanes);
      });
}

/**
 * Returns value with result[index] = lane and every other logical lane
 * unchanged. The index must satisfy 0 <= index < size(tag).
 */
template <VectorTag Tag>
VECOPS_ALWAYS_INLINE Vec<Tag> SetOp::operator()(
    Tag tag, Vec<Tag> value, nint_t index, ElementOf<Tag> lane) const {
  assert(index >= 0 && index < size(tag));
  const nint_t word_lanes = native_word_size(tag);
  return details::visit_runtime_word<details::CurrentBackend>(
      tag,
      index / word_lanes,
      [&]<nint_t Index>() -> Vec<Tag> {
        return set_word<Index>(
            tag,
            value,
            details::execute_word<Index, details::CurrentBackend>(
                SetVecLaneOp{}, tag, get_word<Index>(tag, value),
                index % word_lanes, lane));
      });
}

/**
 * Returns mask with result[index] = lane and every other logical lane
 * unchanged. The index must satisfy 0 <= index < size(tag).
 */
template <VectorTag Tag>
VECOPS_ALWAYS_INLINE Mask<Tag> SetOp::operator()(
    Tag tag, Mask<Tag> value, nint_t index, bool lane) const {
  assert(index >= 0 && index < size(tag));
  const nint_t word_lanes = native_word_size(tag);
  return details::visit_runtime_word<details::CurrentBackend>(
      tag,
      index / word_lanes,
      [&]<nint_t Index>() -> Mask<Tag> {
        return set_word<Index>(
            tag,
            value,
            details::execute_word<Index, details::CurrentBackend>(
                SetMaskLaneOp{}, tag, get_word<Index>(tag, value),
                index % word_lanes, lane));
      });
}

/**
 * Reinterprets the complete logical bit sequence without conversion.
 *
 * Source and destination must have equal logical byte extents. Byte and bit
 * order are unchanged; only the lane partition and element interpretation
 * become those of ToTag. No floating-point operation or rounding occurs.
 */
template <VectorTag ToTag, VectorTag FromTag>
VECOPS_ALWAYS_INLINE Vec<ToTag> BitCastOp::operator()(
    ToTag to, FromTag from, Vec<FromTag> value) const {
  static_assert(
      details::same_logical_bytes<ToTag, FromTag>,
      "bitcast requires equal logical byte extents");
  return details::execute(*this, to, from, value);
}

/** Returns result[i] = value[i] for 0 <= i < size(tag) / 2. */
template <VectorTag Tag>
VECOPS_ALWAYS_INLINE Vec<Half<Tag>> LowerOp::operator()(
    Tag tag, Vec<Tag> value) const {
  return details::execute(*this, tag, value);
}

/** Returns result[i] = mask[i] for 0 <= i < size(tag) / 2. */
template <VectorTag Tag>
VECOPS_ALWAYS_INLINE Mask<Half<Tag>> LowerOp::operator()(
    Tag tag, Mask<Tag> value) const {
  return details::execute(*this, tag, value);
}

/** Returns result[i] = value[i + size(tag) / 2]. */
template <VectorTag Tag>
VECOPS_ALWAYS_INLINE Vec<Half<Tag>> UpperOp::operator()(
    Tag tag, Vec<Tag> value) const {
  return details::execute(*this, tag, value);
}

/** Returns result[i] = mask[i + size(tag) / 2]. */
template <VectorTag Tag>
VECOPS_ALWAYS_INLINE Mask<Half<Tag>> UpperOp::operator()(
    Tag tag, Mask<Tag> value) const {
  return details::execute(*this, tag, value);
}

/**
 * Concatenates two half vectors: result[i] = lower_value[i] in the lower
 * half, and result[i] = upper_value[i - size(tag)/2] in the upper half.
 */
template <VectorTag Tag>
VECOPS_ALWAYS_INLINE Vec<Tag> ConcatOp::operator()(
    Tag tag, Vec<Half<Tag>> lower_value,
    Vec<Half<Tag>> upper_value) const {
  return details::execute(*this, tag, lower_value, upper_value);
}

/** Concatenates two masks with the same lane ordering as vector concat. */
template <VectorTag Tag>
VECOPS_ALWAYS_INLINE Mask<Tag> ConcatOp::operator()(
    Tag tag, Mask<Half<Tag>> lower_value,
    Mask<Half<Tag>> upper_value) const {
  return details::execute(*this, tag, lower_value, upper_value);
}

/** Returns result[i] = value[2*i]. */
template <VectorTag Tag>
VECOPS_ALWAYS_INLINE Vec<Half<Tag>> EvenOp::operator()(
    Tag tag, Vec<Tag> value) const {
  return details::execute(*this, tag, value);
}

/** Returns result[i] = value[2*i + 1]. */
template <VectorTag Tag>
VECOPS_ALWAYS_INLINE Vec<Half<Tag>> OddOp::operator()(
    Tag tag, Vec<Tag> value) const {
  return details::execute(*this, tag, value);
}

/** Returns concat(tag, even(tag, a), even(tag, b)). */
template <VectorTag Tag>
VECOPS_ALWAYS_INLINE Vec<Tag> ConcatEvenOp::operator()(
    Tag tag, Vec<Tag> a, Vec<Tag> b) const {
  return details::execute(*this, tag, a, b);
}

/** Returns concat(tag, odd(tag, a), odd(tag, b)). */
template <VectorTag Tag>
VECOPS_ALWAYS_INLINE Vec<Tag> ConcatOddOp::operator()(
    Tag tag, Vec<Tag> a, Vec<Tag> b) const {
  return details::execute(*this, tag, a, b);
}

/** Returns result[2*i] = a[i] and result[2*i+1] = b[i]. */
template <VectorTag Tag>
VECOPS_ALWAYS_INLINE Vec<Tag> InterleaveOp::operator()(
    Tag tag, Vec<Half<Tag>> a, Vec<Half<Tag>> b) const {
  return details::execute(*this, tag, a, b);
}

/** Returns each pair as {a[2*k], b[2*k]}. */
template <VectorTag Tag>
VECOPS_ALWAYS_INLINE Vec<Tag> InterleaveEvenOp::operator()(
    Tag tag, Vec<Tag> a, Vec<Tag> b) const {
  return details::execute(*this, tag, a, b);
}

/** Returns each pair as {a[2*k+1], b[2*k+1]}. */
template <VectorTag Tag>
VECOPS_ALWAYS_INLINE Vec<Tag> InterleaveOddOp::operator()(
    Tag tag, Vec<Tag> a, Vec<Tag> b) const {
  return details::execute(*this, tag, a, b);
}

/**
 * Within every 16-byte block, interleaves the lower half of a with the lower
 * half of b. Blocks do not exchange lanes.
 */
template <VectorTag Tag>
VECOPS_ALWAYS_INLINE Vec<Tag> LocalInterleaveLowerOp::operator()(
    Tag tag, Vec<Tag> a, Vec<Tag> b) const {
  return details::execute(*this, tag, a, b);
}

/**
 * Within every 16-byte block, interleaves the upper half of a with the upper
 * half of b. Blocks do not exchange lanes.
 */
template <VectorTag Tag>
VECOPS_ALWAYS_INLINE Vec<Tag> LocalInterleaveUpperOp::operator()(
    Tag tag, Vec<Tag> a, Vec<Tag> b) const {
  return details::execute(*this, tag, a, b);
}

/**
 * Permutes lanes independently in each physical word. If W is
 * native_word_size(tag), result[k*W+i] = value[k*W+indices[k*W+i]]. Every
 * index must be in [0, W); words never exchange lanes.
 */
template <VectorTag Tag>
VECOPS_ALWAYS_INLINE Vec<Tag> ShuffleOp::operator()(
    Tag tag, Vec<Tag> value, Vec<IndexTag<Tag>> indices) const {
  return details::execute(*this, tag, value, indices);
}

/**
 * Permutes independently in each 16-byte block. For B=16/sizeof(T),
 * result[k*B+i] = value[k*B+indices[k*B+i]]. Every index must be in [0, B).
 */
template <VectorTag Tag>
VECOPS_ALWAYS_INLINE Vec<Tag> LocalShuffleOp::operator()(
    Tag tag, Vec<Tag> value, Vec<IndexTag<Tag>> indices) const {
  return details::execute(*this, tag, value, indices);
}

/**
 * Compile-time local shuffle. Indices are written in result-lane order and
 * the same 16-byte pattern is repeated in every block.
 */
template <int... Indices, VectorTag Tag>
VECOPS_ALWAYS_INLINE Vec<Tag> LocalShuffleOp::operator()(
    Tag tag, Vec<Tag> value, opt::LaneOrder<Indices...>) const {
  static_assert(
      sizeof...(Indices) == 16 / sizeof(ElementOf<Tag>),
      "local shuffle requires one index per lane in a 16-byte block");
  static_assert(
      ((Indices >= 0 && Indices < static_cast<int>(sizeof...(Indices))) && ...),
      "local shuffle index is outside its 16-byte block");
  return (*this)(tag, value, Indices...);
}

/** Runtime-scalar form of the repeating 16-byte local shuffle pattern. */
template <VectorTag Tag, typename... Indices>
  requires (std::integral<std::remove_cvref_t<Indices>> && ...)
VECOPS_ALWAYS_INLINE Vec<Tag> LocalShuffleOp::operator()(
    Tag tag, Vec<Tag> value, Indices... indices) const {
  constexpr nint_t pattern_lanes =
      16 / static_cast<nint_t>(sizeof(ElementOf<Tag>));
  static_assert(
      sizeof...(Indices) == static_cast<std::size_t>(pattern_lanes),
      "local shuffle requires one index per lane in a 16-byte block");
  const IndexElement<ElementOf<Tag>> pattern[] = {
      static_cast<IndexElement<ElementOf<Tag>>>(indices)...};
  auto index_value = fill(IndexTag<Tag>{}, IndexElement<ElementOf<Tag>>{});
  for (nint_t i = 0; i < size(tag); ++i) {
    assert(pattern[i % pattern_lanes] >= 0 &&
           pattern[i % pattern_lanes] < pattern_lanes);
    index_value = set(
        IndexTag<Tag>{}, index_value, i, pattern[i % pattern_lanes]);
  }
  return (*this)(tag, value, index_value);
}

} // namespace vecops::nvec

#endif // VECOPS_NVEC_BASIC_H
