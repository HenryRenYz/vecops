#ifndef VECOPS_VEC_TAG_H
#define VECOPS_VEC_TAG_H

#include <concepts>
#include <cstddef>
#include <type_traits>

#include "vecops/CoreTypes.h"
#include "vecops/util/TypeTraits.h"

namespace vecops::vec {

namespace details {

template <typename T>
struct IsElement : std::false_type {};

template <> struct IsElement<bfloat16_t> : std::true_type {};
template <> struct IsElement<float16_t> : std::true_type {};
template <> struct IsElement<float32_t> : std::true_type {};
template <> struct IsElement<float64_t> : std::true_type {};
template <> struct IsElement<int8_t> : std::true_type {};
template <> struct IsElement<uint8_t> : std::true_type {};
template <> struct IsElement<int16_t> : std::true_type {};
template <> struct IsElement<uint16_t> : std::true_type {};
template <> struct IsElement<int32_t> : std::true_type {};
template <> struct IsElement<uint32_t> : std::true_type {};
template <> struct IsElement<int64_t> : std::true_type {};
template <> struct IsElement<uint64_t> : std::true_type {};

template <nint_t N>
struct FixedExtent {
  static_assert(
      N > 0 && (N & (N - 1)) == 0,
      "FixedTag lane count must be a positive power of two");
  static constexpr nint_t lanes = N;
};

template <int ScalePower>
struct ScalableExtent {
  static constexpr int scale_power_v = ScalePower;
};

/**
 * @file Tag.h
 * @brief Architecture-independent vector descriptor types and type-level
 * manipulators.
 *
 * The Tag system is the public entry point for describing a SIMD vector
 * request. It separates three concerns:
 *
 * - **Element type** via the @ref Element concept (12 types: bfloat16_t,
 *   float16_t, float32_t, float64_t, int8_t, uint8_t, int16_t, uint16_t,
 *   int32_t, uint32_t, int64_t, uint64_t).
 * - **Logical extent** via FixedTag<T,N> (compile-time lane count, N a power
 *   of two) or ScalableTag<T,P> (2^P native words, runtime lane count on SVE).
 * - **Type-level manipulation** via Rebind, ViewAs, Half, Twice, and the
 *   IndexTag family.
 *
 * A Tag is a pure compile-time descriptor — it carries no data and selects
 * the concrete Vec<Tag> / Mask<Tag> representation via the active backend.
 *
 * @see VecBase.h for the Vec/Mask representation types.
 * @see Options.h for operation options that reference Tags.
 */

/**
 *  Complete, architecture-independent description of a vector request.
 *
 *  Unlike a vector representation, this type retains logical subword extent.
 *  A descriptor therefore maps unambiguously to its Vec and Mask types, while
 *  the reverse mapping from Vec may be lossy.
 */
template <typename T, typename Extent>
struct VectorDescriptor {
  using ElementType = T;
  using ExtentType = Extent;
};

template <typename T>
struct IsVectorDescriptor : std::false_type {};

template <typename T, typename Extent>
struct IsVectorDescriptor<VectorDescriptor<T, Extent>> : std::true_type {};

template <typename Extent>
struct IsFixedExtent : std::false_type {};

template <nint_t N>
struct IsFixedExtent<FixedExtent<N>> : std::true_type {};

template <typename Extent>
struct IsScalableExtent : std::false_type {};

template <int ScalePower>
struct IsScalableExtent<ScalableExtent<ScalePower>> : std::true_type {};

consteval int element_size_shift(std::size_t from, std::size_t to) {
  int shift = 0;
  while (from < to) {
    from *= 2;
    ++shift;
  }
  while (from > to) {
    to *= 2;
    --shift;
  }
  return shift;
}

template <typename NewElement, typename Tag>
struct RebindImpl;

template <typename NewElement, typename OldElement, nint_t N>
struct RebindImpl<
    NewElement,
    VectorDescriptor<OldElement, FixedExtent<N>>> {
  using type = VectorDescriptor<NewElement, FixedExtent<N>>;
};

template <typename NewElement, typename OldElement, int ScalePower>
struct RebindImpl<
    NewElement,
    VectorDescriptor<OldElement, ScalableExtent<ScalePower>>> {
  using type = VectorDescriptor<
      NewElement,
      ScalableExtent<
          ScalePower + element_size_shift(sizeof(OldElement), sizeof(NewElement))>>;
};

template <typename NewElement, typename Tag>
struct ViewAsImpl;

template <typename NewElement, typename OldElement, nint_t N>
struct ViewAsImpl<
    NewElement,
    VectorDescriptor<OldElement, FixedExtent<N>>> {
  static constexpr std::size_t bytes = static_cast<std::size_t>(N) * sizeof(OldElement);
  static_assert(
      bytes % sizeof(NewElement) == 0,
      "ViewAs requires an integral number of output lanes");
  using type = VectorDescriptor<
      NewElement,
      FixedExtent<static_cast<nint_t>(bytes / sizeof(NewElement))>>;
};

template <typename NewElement, typename OldElement, int ScalePower>
struct ViewAsImpl<
    NewElement,
    VectorDescriptor<OldElement, ScalableExtent<ScalePower>>> {
  using type = VectorDescriptor<NewElement, ScalableExtent<ScalePower>>;
};

template <typename Tag>
struct HalfImpl;

template <typename T, nint_t N>
struct HalfImpl<VectorDescriptor<T, FixedExtent<N>>> {
  static_assert(N % 2 == 0, "Half requires an even fixed lane count");
  using type = VectorDescriptor<T, FixedExtent<N / 2>>;
};

template <typename T, int ScalePower>
struct HalfImpl<VectorDescriptor<T, ScalableExtent<ScalePower>>> {
  using type = VectorDescriptor<T, ScalableExtent<ScalePower - 1>>;
};

template <typename Tag>
struct TwiceImpl;

template <typename T, nint_t N>
struct TwiceImpl<VectorDescriptor<T, FixedExtent<N>>> {
  using type = VectorDescriptor<T, FixedExtent<N * 2>>;
};

template <typename T, int ScalePower>
struct TwiceImpl<VectorDescriptor<T, ScalableExtent<ScalePower>>> {
  using type = VectorDescriptor<T, ScalableExtent<ScalePower + 1>>;
};

template <typename T>
struct IndexElementImpl;

template <> struct IndexElementImpl<bfloat16_t> { using type = int16_t; };
template <> struct IndexElementImpl<float16_t> { using type = int16_t; };
template <> struct IndexElementImpl<float32_t> { using type = int32_t; };
template <> struct IndexElementImpl<float64_t> { using type = int64_t; };
template <> struct IndexElementImpl<int8_t> { using type = int8_t; };
template <> struct IndexElementImpl<uint8_t> { using type = int8_t; };
template <> struct IndexElementImpl<int16_t> { using type = int16_t; };
template <> struct IndexElementImpl<uint16_t> { using type = int16_t; };
template <> struct IndexElementImpl<int32_t> { using type = int32_t; };
template <> struct IndexElementImpl<uint32_t> { using type = int32_t; };
template <> struct IndexElementImpl<int64_t> { using type = int64_t; };
template <> struct IndexElementImpl<uint64_t> { using type = int64_t; };

} // namespace details

/**
 * Element types implemented by VecOps backends.
 *
 * The 12 supported types are: bfloat16_t, float16_t, float32_t, float64_t,
 * int8_t, uint8_t, int16_t, uint16_t, int32_t, uint32_t, int64_t, uint64_t.
 * cvref-qualified types are rejected (T must equal its unqualified form).
 */
template <typename T>
concept Element = details::IsElement<std::remove_cv_t<T>>::value &&
                  std::same_as<T, std::remove_cv_t<T>>;

/** A complete vector descriptor accepted by the public API. */
template <typename T>
concept VectorTag =
    details::IsVectorDescriptor<std::remove_cvref_t<T>>::value;

/**
 * Describes exactly N logical lanes of T.
 *
 * N must be a positive power of two and need not fill a hardware word. The
 * selected representation may contain inactive physical lanes, but the Tag
 * continues to retain exact N.
 */
template <Element T, nint_t N>
using FixedTag = details::VectorDescriptor<T, details::FixedExtent<N>>;

/**
 * Describes a target-relative vector containing 2^ScalePower native words.
 *
 * Negative powers describe a logical subword. On scalable SVE the lane count
 * is runtime-dependent; on fixed-width targets it is a compile-time constant.
 */
template <Element T, int ScalePower = 0>
using ScalableTag =
    details::VectorDescriptor<T, details::ScalableExtent<ScalePower>>;

/** Extracts the element type from a VectorTag descriptor. */
template <VectorTag Tag>
using ElementOf = typename std::remove_cvref_t<Tag>::ElementType;

/**
 * A VectorTag whose element type is one of the floating-point Element types
 * (bfloat16_t, float16_t, float32_t, float64_t).
 */
template <typename Tag>
concept FloatingTag =
    VectorTag<Tag> && ::vecops::is_float_v<ElementOf<Tag>>;

/** True when the Tag describes a compile-time-known lane count (FixedTag). */
template <VectorTag Tag>
inline constexpr bool is_fixed_tag_v = details::IsFixedExtent<
    typename std::remove_cvref_t<Tag>::ExtentType>::value;

/** True when the Tag describes a runtime-dependent lane count (ScalableTag). */
template <VectorTag Tag>
inline constexpr bool is_scalable_tag_v = details::IsScalableExtent<
    typename std::remove_cvref_t<Tag>::ExtentType>::value;

/** The compile-time logical lane count of a FixedTag. Requires is_fixed_tag_v<Tag>. */
template <VectorTag Tag>
  requires is_fixed_tag_v<Tag>
inline constexpr nint_t fixed_lanes_v =
    std::remove_cvref_t<Tag>::ExtentType::lanes;

/**
 * Base-2 exponent of the number of native words a ScalableTag covers.
 * Requires is_scalable_tag_v<Tag>.
 */
template <VectorTag Tag>
  requires is_scalable_tag_v<Tag>
inline constexpr int scale_power_v =
    std::remove_cvref_t<Tag>::ExtentType::scale_power_v;

/** Keeps the logical lane count while changing the element type. */
template <Element NewElement, VectorTag Tag>
using Rebind = typename details::RebindImpl<
    NewElement, std::remove_cvref_t<Tag>>::type;

/** Keeps the logical byte extent while changing the element type. */
template <Element NewElement, VectorTag Tag>
using ViewAs = typename details::ViewAsImpl<
    NewElement, std::remove_cvref_t<Tag>>::type;

/** Describes the lower or upper half of a Tag's logical lanes. */
template <VectorTag Tag>
using Half = typename details::HalfImpl<std::remove_cvref_t<Tag>>::type;

/** Describes twice a Tag's logical lanes without changing its element type. */
template <VectorTag Tag>
using Twice = typename details::TwiceImpl<std::remove_cvref_t<Tag>>::type;

/**
 * Signed integer element used to index lanes of T.
 *
 * Floating-point elements map to the same-width integer (e.g. float32_t →
 * int32_t); integer elements map to themselves. Unsigned elements map to
 * their signed counterpart (e.g. uint32_t → int32_t).
 */
template <Element T>
using IndexElement = typename details::IndexElementImpl<T>::type;

/**
 * Descriptor for one signed lane index per logical lane of Tag.
 * @see IndexElement
 */
template <VectorTag Tag>
using IndexTag = Rebind<IndexElement<ElementOf<Tag>>, Tag>;

namespace details {

/**
 * True when two complete Tags describe the same logical byte extent.
 * Used by bitcast validation: two Tags with equal byte spans can be
 * reinterpreted without data loss (within a single word), and scalable
 * Tags must share an identical scale power.
 */
template <VectorTag A, VectorTag B>
inline constexpr bool same_logical_bytes_v = [] {
  if constexpr (is_fixed_tag_v<A> && is_fixed_tag_v<B>) {
    return fixed_lanes_v<A> * static_cast<nint_t>(sizeof(ElementOf<A>)) ==
           fixed_lanes_v<B> * static_cast<nint_t>(sizeof(ElementOf<B>));
  } else if constexpr (is_scalable_tag_v<A> && is_scalable_tag_v<B>) {
    return scale_power_v<A> == scale_power_v<B>;
  } else {
    return false;
  }
}();

} // namespace details

} // namespace vecops::vec

#endif // VECOPS_VEC_TAG_H
