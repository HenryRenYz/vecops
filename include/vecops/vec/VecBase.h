#ifndef VECOPS_VEC_VECBASE_H
#define VECOPS_VEC_VECBASE_H

#include <cassert>
#include <concepts>
#include <type_traits>

#include "vecops/CoreDefs.h"
#include "vecops/util/TypeTraits.h"
#include "vecops/vec/Tag.h"
#include "vecops/vec/details/Backend.h"
#include "vecops/vec/details/Representation.h"
#include "vecops/vec/details/scalar/Types.h"

#if defined(ARCH_X86_FAMILY) && !defined(CPU_CAPABILITY_GENERIC)
#include "vecops/vec/details/x86/Types.h"
#elif defined(CPU_CAPABILITY_SVE)
#include "vecops/vec/details/sve/Types.h"
#endif

namespace vecops::vec {

/**
 * Stable alignment used by shared workspaces and temporary buffers.
 *
 * Derived from the active SIMD width. When SIMD_WIDTH is negative (SVE), the
 * width is unknown at compile time so a fixed 64-byte alignment is used as a
 * conservative lower bound. Otherwise the alignment is max(SIMD_WIDTH/8, 8)
 * to ensure at least 8-byte alignment for 64-bit element types on narrow SIMD.
 */
inline constexpr nint_t DEFAULT_ALIGNMENT =
    SIMD_WIDTH < 0 ? 64 : (SIMD_WIDTH / 8 < 8 ? 8 : SIMD_WIDTH / 8);

namespace details {

template <VectorTag Tag>
using CurrentRepresentation = RepresentationTraits<
    CurrentBackend, std::remove_cvref_t<Tag>>;

template <typename Representation>
VECOPS_ALWAYS_INLINE constexpr nint_t representation_logical_lanes() {
  if constexpr (requires { Representation::logical_lanes(); }) {
    return Representation::logical_lanes();
  } else {
    return Representation::logical_lanes;
  }
}

template <typename Representation>
VECOPS_ALWAYS_INLINE constexpr nint_t representation_word_lanes() {
  if constexpr (requires { Representation::word_lanes(); }) {
    return Representation::word_lanes();
  } else {
    return Representation::word_lanes;
  }
}

} // namespace details

/**
 * Vector representation selected unambiguously from a complete Tag.
 * The concrete type is determined by the active backend (scalar, x86, or SVE).
 * @see Mask
 */
template <VectorTag Tag>
using Vec = typename details::CurrentRepresentation<Tag>::VecType;

/**
 * Mask representation selected unambiguously from a complete Tag.
 * @see Vec
 */
template <VectorTag Tag>
using Mask = typename details::CurrentRepresentation<Tag>::MaskType;

/** Physical vector type for one native SIMD word of the given Tag. */
template <VectorTag Tag>
using NativeWordVec = typename details::CurrentRepresentation<Tag>::WordVec;

/** Physical mask type for one native SIMD word of the given Tag. */
template <VectorTag Tag>
using NativeWordMask = typename details::CurrentRepresentation<Tag>::WordMask;

namespace details {

/** Backend customization point for runtime physical-word access. */
template <typename Backend, VectorTag Tag>
struct RuntimeWordAccess {
  static constexpr bool sized_vec = is_word_array_v<Vec<Tag>>;
  static constexpr bool sized_mask = is_word_array_v<Mask<Tag>>;

  static VECOPS_ALWAYS_INLINE constexpr NativeWordVec<Tag> get_vec(
      Tag, const Vec<Tag>& value, nint_t ordinal)
    requires sized_vec
  {
    return details::get_word(value, ordinal);
  }

  static VECOPS_ALWAYS_INLINE constexpr NativeWordVec<Tag> get_vec(
      Tag, Vec<Tag> value, nint_t ordinal)
    requires (!sized_vec)
  {
    return details::get_word(value, ordinal);
  }

  static VECOPS_ALWAYS_INLINE constexpr Vec<Tag> set_vec(
      Tag, Vec<Tag> value, nint_t ordinal, NativeWordVec<Tag> word) {
    return details::set_word(value, ordinal, word);
  }

  static VECOPS_ALWAYS_INLINE constexpr NativeWordMask<Tag> get_mask(
      Tag, const Mask<Tag>& value, nint_t ordinal)
    requires sized_mask
  {
    return details::get_word(value, ordinal);
  }

  static VECOPS_ALWAYS_INLINE constexpr NativeWordMask<Tag> get_mask(
      Tag, Mask<Tag> value, nint_t ordinal)
    requires (!sized_mask)
  {
    return details::get_word(value, ordinal);
  }

  static VECOPS_ALWAYS_INLINE constexpr Mask<Tag> set_mask(
      Tag, Mask<Tag> value, nint_t ordinal, NativeWordMask<Tag> word) {
    return details::set_word(value, ordinal, word);
  }
};

} // namespace details

/**
 * A VectorValue whose canonical physical Tag can be reconstructed via \ref
 * VecToTag. Most operations accept Tag-inferable vectors and internally
 * resolve VecToTag<V> to forward to the explicit-Tag overload.
 */
template <typename V>
concept TagInferableVector =
    VectorValue<V> && details::HasInferredTag<V>;

namespace details {

template <TagInferableVector V>
struct VecToTagImpl {
  using type = InferredTagOf<V>;
};

} // namespace details

/**
 * Maps a Vec representation to its canonical physical Tag.
 *
 * A representation does not retain a subword's logical extent. Consequently
 * this reverse mapping intentionally describes the complete physical extent;
 * callers that need explicit subword semantics must keep passing the original
 * Tag to the operation.
 */
template <TagInferableVector V>
using VecToTag = typename details::VecToTagImpl<std::remove_cvref_t<V>>::type;

/**
 * A Tag-inferable vector whose element type is integral (signed or unsigned).
 */
template <typename V>
concept IntegerVectorValue =
    TagInferableVector<V> && std::integral<ElementOf<VecToTag<V>>>;

/**
 * A Tag-inferable vector whose element type is floating-point.
 */
template <typename V>
concept FloatingVectorValue =
    TagInferableVector<V> && ::vecops::is_float_v<ElementOf<VecToTag<V>>>;

/**
 * Returns the exact logical lane count described by tag.
 *
 * For FixedTag this is a compile-time constant. For ScalableTag this is a
 * runtime value on sizeless SVE and a compile-time constant on fixed-width
 * backends (where VLS maps scalable requests to known widths).
 */
template <VectorTag Tag>
VECOPS_ALWAYS_INLINE constexpr nint_t size(Tag = {}) {
  if constexpr (is_fixed_tag_v<Tag>) {
    // FixedTag is usable as architecture-independent metadata even when the
    // current scalable SVE mode cannot materialize its Vec representation.
    return fixed_lanes_v<Tag>;
  } else {
    return details::representation_logical_lanes<
        details::CurrentRepresentation<Tag>>();
  }
}

/**
 * Returns the physical lane count in one backend word.
 *
 * This is the number of elements that fit in one native SIMD register width —
 * typically 16 bytes worth of lanes on scalar and x86, and a runtime
 * value on sizeless SVE.
 */
template <VectorTag Tag>
VECOPS_ALWAYS_INLINE constexpr nint_t native_word_size(Tag = {}) {
  return details::representation_word_lanes<
      details::CurrentRepresentation<Tag>>();
}

/**
 * Returns the number of physical backend words used by tag.
 *
 * Multi-word Tags cover vectors that span multiple SIMD registers (e.g.
 * ScalableTag<f32, 2> on AVX-512 is a 2-register, 1024-bit vector).
 */
template <VectorTag Tag>
VECOPS_ALWAYS_INLINE constexpr nint_t num_words(Tag = {}) {
  return details::CurrentRepresentation<Tag>::word_count;
}

/**
 * Required byte alignment for an aligned memory access.
 *
 * Formula: native_word_size(tag) * sizeof(ElementOf<Tag>). This is the
 * natural alignment of one physical SIMD word.
 */
template <VectorTag Tag>
VECOPS_ALWAYS_INLINE constexpr nint_t memory_alignment(Tag tag = {}) {
  return native_word_size(tag) *
      static_cast<nint_t>(sizeof(ElementOf<Tag>));
}

/**
 * Whether size(tag) depends on the runtime SVE vector length.
 * True only for SVE-backed scalable Tags on a sizeless SVE compilation mode.
 */
template <VectorTag Tag>
inline constexpr bool is_runtime_size_v =
    details::CurrentRepresentation<Tag>::is_runtime_size_v;

/**
 * Whether the logical extent occupies only part of its physical word.
 * True when FixedTag<T,N> has N < native_word_size, or when a ScalableTag
 * has a negative scale power (describing a subword fraction).
 */
template <VectorTag Tag>
inline constexpr bool is_subword_v =
    details::CurrentRepresentation<Tag>::is_subword_v;

/**
 * Returns a physical word from sized Scalar, x86, or VLS representations.
 * Sizeless SVE tuple access is provided by the SVE access layer.
 */
template <nint_t Index, VectorTag Tag>
  requires details::is_word_array_v<Vec<Tag>> &&
           requires(const Vec<Tag>& value) {
             details::get_word<Index>(value);
           }
VECOPS_ALWAYS_INLINE constexpr NativeWordVec<Tag> get_word(
    Tag, const Vec<Tag>& value) {
  return details::get_word<Index>(value);
}

template <nint_t Index, VectorTag Tag>
  requires (!details::is_word_array_v<Vec<Tag>>) && requires(Vec<Tag> value) {
    details::get_word<Index>(value);
  }
VECOPS_ALWAYS_INLINE constexpr NativeWordVec<Tag> get_word(
    Tag, Vec<Tag> value) {
  return details::get_word<Index>(value);
}

template <nint_t Index, VectorTag Tag>
  requires requires(Vec<Tag> value, NativeWordVec<Tag> word) {
    details::set_word<Index>(value, word);
  }
VECOPS_ALWAYS_INLINE constexpr Vec<Tag> set_word(
    Tag, Vec<Tag> value, NativeWordVec<Tag> word) {
  return details::set_word<Index>(value, word);
}

/** Returns a physical vector word selected by a runtime ordinal. */
template <VectorTag Tag>
  requires details::is_word_array_v<Vec<Tag>>
VECOPS_ALWAYS_INLINE constexpr NativeWordVec<Tag> get_word(
    Tag tag, const Vec<Tag>& value, nint_t ordinal) {
  assert(ordinal >= 0 && ordinal < num_words(tag));
  return details::RuntimeWordAccess<details::CurrentBackend, Tag>::get_vec(
      tag, value, ordinal);
}

template <VectorTag Tag>
  requires (!details::is_word_array_v<Vec<Tag>>)
VECOPS_ALWAYS_INLINE constexpr NativeWordVec<Tag> get_word(
    Tag tag, Vec<Tag> value, nint_t ordinal) {
  assert(ordinal >= 0 && ordinal < num_words(tag));
  return details::RuntimeWordAccess<details::CurrentBackend, Tag>::get_vec(
      tag, value, ordinal);
}

/** Replaces a physical vector word selected by a runtime ordinal. */
template <VectorTag Tag>
VECOPS_ALWAYS_INLINE constexpr Vec<Tag> set_word(
    Tag tag, Vec<Tag> value, nint_t ordinal, NativeWordVec<Tag> word) {
  assert(ordinal >= 0 && ordinal < num_words(tag));
  return details::RuntimeWordAccess<details::CurrentBackend, Tag>::set_vec(
      tag, value, ordinal, word);
}

/** Constructs a complete Vec directly from all of its physical words. */
template <VectorTag Tag, typename First, typename... Rest>
  requires std::same_as<std::remove_cvref_t<First>, NativeWordVec<Tag>> &&
           (std::same_as<std::remove_cvref_t<Rest>, NativeWordVec<Tag>> && ...)
VECOPS_ALWAYS_INLINE constexpr Vec<Tag> from_words(
    Tag, First first, Rest... rest) {
  constexpr nint_t count = details::CurrentRepresentation<Tag>::word_count;
  static_assert(1 + sizeof...(Rest) == static_cast<std::size_t>(count));
  if constexpr (count == 1) {
    return first;
  } else if constexpr (requires { sizeof(Vec<Tag>); }) {
    return Vec<Tag>{{first, rest...}};
  } else {
    using Result = decltype(details::make_word_group(first, rest...));
    static_assert(std::same_as<Result, Vec<Tag>>);
    return details::make_word_group(first, rest...);
  }
}

/** Returns a physical predicate word from a Mask selected by tag. */
template <nint_t Index, VectorTag Tag>
  requires details::is_word_array_v<Mask<Tag>> &&
           requires(const Mask<Tag>& value) {
             details::get_word<Index>(value);
           }
VECOPS_ALWAYS_INLINE constexpr NativeWordMask<Tag> get_word(
    Tag, const Mask<Tag>& value) {
  return details::get_word<Index>(value);
}

template <nint_t Index, VectorTag Tag>
  requires (!details::is_word_array_v<Mask<Tag>>) && requires(Mask<Tag> value) {
    details::get_word<Index>(value);
  }
VECOPS_ALWAYS_INLINE constexpr NativeWordMask<Tag> get_word(
    Tag, Mask<Tag> value) {
  return details::get_word<Index>(value);
}

/** Replaces a physical predicate word in a Mask selected by tag. */
template <nint_t Index, VectorTag Tag>
  requires requires(Mask<Tag> value, NativeWordMask<Tag> word) {
    details::set_word<Index>(value, word);
  }
VECOPS_ALWAYS_INLINE constexpr Mask<Tag> set_word(
    Tag, Mask<Tag> value, NativeWordMask<Tag> word) {
  return details::set_word<Index>(value, word);
}

/** Returns a physical predicate word selected by a runtime ordinal. */
template <VectorTag Tag>
  requires details::is_word_array_v<Mask<Tag>>
VECOPS_ALWAYS_INLINE constexpr NativeWordMask<Tag> get_word(
    Tag tag, const Mask<Tag>& value, nint_t ordinal) {
  assert(ordinal >= 0 && ordinal < num_words(tag));
  return details::RuntimeWordAccess<details::CurrentBackend, Tag>::get_mask(
      tag, value, ordinal);
}

template <VectorTag Tag>
  requires (!details::is_word_array_v<Mask<Tag>>)
VECOPS_ALWAYS_INLINE constexpr NativeWordMask<Tag> get_word(
    Tag tag, Mask<Tag> value, nint_t ordinal) {
  assert(ordinal >= 0 && ordinal < num_words(tag));
  return details::RuntimeWordAccess<details::CurrentBackend, Tag>::get_mask(
      tag, value, ordinal);
}

/** Replaces a physical predicate word selected by a runtime ordinal. */
template <VectorTag Tag>
VECOPS_ALWAYS_INLINE constexpr Mask<Tag> set_word(
    Tag tag, Mask<Tag> value, nint_t ordinal, NativeWordMask<Tag> word) {
  assert(ordinal >= 0 && ordinal < num_words(tag));
  return details::RuntimeWordAccess<details::CurrentBackend, Tag>::set_mask(
      tag, value, ordinal, word);
}

/** Constructs a complete Mask directly from all of its physical words. */
template <VectorTag Tag, typename First, typename... Rest>
  requires std::same_as<std::remove_cvref_t<First>, NativeWordMask<Tag>> &&
           (std::same_as<std::remove_cvref_t<Rest>, NativeWordMask<Tag>> && ...)
VECOPS_ALWAYS_INLINE constexpr Mask<Tag> mask_from_words(
    Tag, First first, Rest... rest) {
  constexpr nint_t count = details::CurrentRepresentation<Tag>::word_count;
  static_assert(1 + sizeof...(Rest) == static_cast<std::size_t>(count));
  if constexpr (count == 1) {
    return first;
  } else if constexpr (requires { sizeof(Mask<Tag>); }) {
    return Mask<Tag>{{first, rest...}};
  } else {
    using Result = decltype(details::make_word_group(first, rest...));
    static_assert(std::same_as<Result, Mask<Tag>>);
    return details::make_word_group(first, rest...);
  }
}

namespace details {

/**
 * Forward declarations of the dispatch entry points, defined in
 * details/Dispatch.h. Top-level headers declare their Op functors in the
 * declaration half, long before the sandwich middle includes Dispatch.h, and
 * the word-level / low-layer operator() entries written inline into those
 * functors call through these names. Every such call site is itself a
 * template, so it instantiates at its point of use — after the definitions
 * are visible — and links against them.
 */
template <VectorTag Tag, typename Op, typename... Args>
VECOPS_ALWAYS_INLINE constexpr decltype(auto) execute(
    Op op, Tag tag, Args&&... args);

template <nint_t Index, typename Backend, VectorTag Tag,
          typename Op, typename... Words>
VECOPS_ALWAYS_INLINE constexpr decltype(auto) execute_word(
    Op op, Tag tag, Words&&... words);

/**
 * Inactive-lane policy tags accepted by the low-layer masked entries
 * (tag, values..., mask, inactive, policy) that option dispatchers and
 * backends use to bypass option parsing.
 */
struct PreserveArithmeticInactive {};
/** Policy tag: set inactive lanes to zero. */
struct ZeroArithmeticInactive {};
/** Policy tag: set inactive lanes from a supplied vector or scalar merge. */
struct MergeArithmeticInactive {};

/** Concepts naming the policies accepted by low-layer masked entries. */
template <typename Policy>
concept arithmetic_inactive_policy =
    std::same_as<Policy, PreserveArithmeticInactive> ||
    std::same_as<Policy, ZeroArithmeticInactive> ||
    std::same_as<Policy, MergeArithmeticInactive>;

/** True when the Tag's value representation differs from one physical word. */
template <VectorTag Tag>
inline constexpr bool multi_word_tag_v =
    !std::same_as<Vec<Tag>, NativeWordVec<Tag>>;

} // namespace vecops::vec::details

} // namespace vecops::vec

#endif // VECOPS_VEC_VECBASE_H
