#ifndef VECOPS_VEC_OPTIONS_H
#define VECOPS_VEC_OPTIONS_H

#include <cstddef>
#include <type_traits>
#include <utility>

#include "vecops/Meta.h"
#include "vecops/vec/VecBase.h"

/**
 * @file Options.h
 * @brief Tag types that control operation behavior: masking, population,
 * conversion policy, and memory addressing.
 *
 * Options are organized in three namespaces:
 *
 * - **opt**: computation-level options for masking (masked/unmasked),
 *   inactive-lane population (zero/merge), math accuracy (opt::math), lane
 *   selection (first), lane ordering (LaneOrder), scale (Scale), and memory
 *   addressing (indexed, strided).
 *
 * - **cvt**: conversion-policy options controlling lane order
 *   (ordered/unordered/lane), value behavior (saturate/wrap), and narrowing
 *   phase.
 *
 * - **mem**: memory-access qualifiers for alignment (aligned/unaligned),
 *   temporality (temporal/non_temporal), narrow-store layout (packed/split),
 *   and prefetch hints.
 *
 * Options are validated at compile time by the detection traits in the
 * vec::details section at the end of this header. Each operation header
 * (Arithmetic.h, Bit.h, Memory.h, etc.) declares its own valid-option set
 * via constexpr predicates built from those traits.
 */

namespace vecops::vec {

/**
 * Compile-time accuracy policy shared by vector math operations and operators
 * which expose their math-accuracy choice.
 *
 * The meaning of each level, including its error metric and accuracy bound, is
 * defined by the operation that accepts the policy.  The levels select an
 * implementation contract; they do not impose a cross-operation or
 * cross-target ordering on measured accuracy or performance.  In particular,
 * a more relaxed implementation may be as accurate as a stricter one and is
 * not guaranteed to be faster on every target or input.
 */
enum class Accuracy { Strict, Fast, Estimate };

} // namespace vecops::vec

namespace vecops::vec::opt {

/** Compile-time options which select an operation-defined math accuracy. */
namespace math {

/** Associates a math option with one value of vec::Accuracy. */
template <Accuracy A>
struct AccuracyOption {
  static constexpr Accuracy value = A;
};

/** Generic accuracy option for forwarding a compile-time Accuracy value. */
template <Accuracy A>
inline constexpr AccuracyOption<A> accuracy{};

/** Requests the operation's strict accuracy contract. */
inline constexpr auto strict = accuracy<Accuracy::Strict>;

/** Requests the operation's fast accuracy contract, when distinguished. */
inline constexpr auto fast = accuracy<Accuracy::Fast>;

/** Requests the operation's estimate accuracy contract, when distinguished. */
inline constexpr auto estimate = accuracy<Accuracy::Estimate>;

} // namespace math

/**
 * Selects zero for every inactive output lane of a masked vector operation.
 * Mutually exclusive with any opt::merge option.
 */
struct Zero {};

inline constexpr Zero zero{};

/**
 * Explicitly selects every logical lane without applying a mask.
 * The counterpart of opt::masked; unmasked and masked in the same call is
 * a compile-time error.
 */
struct Unmasked {};

inline constexpr Unmasked unmasked{};

/**
 * Selects lanes in the logical element space of the receiving abstraction.
 * A higher-level accessor may translate the predicate before issuing the
 * eventual physical memory instruction.
 */
template <MaskValue M>
struct Masked {
  const M& value;
};

/**
 * Selects the lanes whose corresponding mask bit is true.
 *
 * The mask must be a named lvalue. Keeping only a reference is required for
 * sizeless SVE predicates, which cannot be stored as ordinary data members.
 */
template <MaskValue M>
VECOPS_ALWAYS_INLINE Masked<M> masked(const M& value) {
  return {value};
}

template <typename M>
  requires MaskValue<M> && (!std::is_lvalue_reference_v<M>)
Masked<std::remove_cvref_t<M>> masked(M&&) = delete;

/**
 * Selects the first count logical elements. count is clamped to [0, size(tag)]
 * at the call site: negative values act as 0, excess as size(tag).
 * @see mwhilelt
 */
struct First {
  nint_t count;
};

VECOPS_ALWAYS_INLINE constexpr First first(nint_t count) {
  return {count};
}

/** Supplies a vector value for inactive lanes. */
template <VectorValue V>
struct VectorMerge {
  const V& value;
};

/**
 * Preserves lanes from a named vector lvalue where an operation is inactive.
 * A reference wrapper also keeps this option usable with sizeless SVE values.
 */
template <VectorValue V>
VECOPS_ALWAYS_INLINE VectorMerge<V> merge(const V& value) {
  return {value};
}

template <typename V>
  requires VectorValue<V> && (!std::is_lvalue_reference_v<V>)
VectorMerge<std::remove_cvref_t<V>> merge(V&&) = delete;

/** Supplies a mask value for inactive lanes of a mask-producing operation. */
template <MaskValue M>
struct MaskMerge {
  const M& value;
};

template <MaskValue M>
VECOPS_ALWAYS_INLINE MaskMerge<M> merge(const M& value) {
  return {value};
}

template <typename M>
  requires MaskValue<M> && (!std::is_lvalue_reference_v<M>)
MaskMerge<std::remove_cvref_t<M>> merge(M&&) = delete;

/**
 * Supplies one scalar value broadcast to every inactive lane.
 * Valid only in operations that produce vector output (arithmetic, bit, fill,
 * load, etc.), not in mask-producing or reduction operations.
 */
template <Element T>
struct ScalarMerge {
  T value;
};

template <Element T>
VECOPS_ALWAYS_INLINE constexpr ScalarMerge<T> merge(T value) {
  return {value};
}

/**
 * Compile-time lane order for local_shuf operations.
 * All indices must be in [0, 16/sizeof(T)); the pattern repeats across every
 * 16-byte block of the input.
 * @see LocalShuffleOp
 */
template <int... Indices>
struct LaneOrder {};

template <int... Indices>
inline constexpr LaneOrder<Indices...> lanes{};

/** Compile-time byte scale for indexed memory addressing. */
template <int S>
struct Scale {
  static_assert(
      S == 1 || S == 2 || S == 4 || S == 8,
      "indexed scale must be 1, 2, 4, or 8");
  static constexpr int value = S;
};

template <int S>
inline constexpr Scale<S> scale{};

/**
 * Selects indexed element addressing with one signed i32/i64 index per lane.
 *
 * At the raw vec layer a scale of zero means `sizeof(memory element)` and an
 * explicit scale is a byte multiplier. A tensor accessor may instead lower
 * scale-zero logical offsets through its layout, so this option alone does
 * not guarantee a physical gather/scatter instruction. The named index vector
 * is retained by reference so this also works with sizeless SVE types.
 */
template <VectorValue V, int S = 0>
struct Indexed {
  static_assert(
      S == 0 || S == 1 || S == 2 || S == 4 || S == 8,
      "indexed scale must be 0 (auto), 1, 2, 4, or 8");
  static constexpr int scale = S;
  const V& indices;
};

template <VectorValue V>
VECOPS_ALWAYS_INLINE Indexed<V> indexed(const V& indices) {
  return {indices};
}

template <VectorValue V, int S>
VECOPS_ALWAYS_INLINE Indexed<V, S> indexed(
    const V& indices, Scale<S>) {
  return {indices};
}

template <typename V>
  requires VectorValue<std::remove_cvref_t<V>> &&
           (!std::is_lvalue_reference_v<V>)
Indexed<std::remove_cvref_t<V>> indexed(V&&) = delete;

template <typename V, int S>
  requires VectorValue<std::remove_cvref_t<V>> &&
           (!std::is_lvalue_reference_v<V>)
Indexed<std::remove_cvref_t<V>, S> indexed(V&&, Scale<S>) =
    delete;

namespace details {

template <typename>
struct IsStrideMetadata : std::false_type {};

template <nint_t N>
struct IsStrideMetadata<meta::Const<N>> : std::true_type {};

template <nint_t Alignment, nint_t Lo, nint_t Hi>
struct IsStrideMetadata<meta::Dynamic<Alignment, Lo, Hi>> : std::true_type {};

} // namespace details

/**
 * Selects a constant or runtime stride in the logical element space of the
 * receiving abstraction. Raw vec memory operations apply it to pointer
 * elements; a tensor accessor composes it with the selected tensor-axis
 * stride. The option alone does not determine load versus gather lowering.
 * Accepts meta::Const<N> for compile-time strides, meta::Dynamic for
 * bounded runtime strides, or a plain nint_t (via meta::Any). Strided and
 * indexed addressing are mutually exclusive and cannot be combined with
 * alignment options.
 */
template <typename Stride>
  requires details::IsStrideMetadata<std::remove_cvref_t<Stride>>::value
struct Strided {
  using StrideType = std::remove_cvref_t<Stride>;
  StrideType stride;
};

template <nint_t N>
VECOPS_ALWAYS_INLINE constexpr Strided<meta::Const<N>> strided(
    meta::Const<N> stride) {
  return {stride};
}

template <nint_t Alignment, nint_t Lo, nint_t Hi>
VECOPS_ALWAYS_INLINE constexpr Strided<meta::Dynamic<Alignment, Lo, Hi>>
strided(meta::Dynamic<Alignment, Lo, Hi> stride) {
  return {stride};
}

VECOPS_ALWAYS_INLINE constexpr Strided<meta::Any> strided(nint_t stride) {
  return {meta::Any{stride}};
}

} // namespace vecops::vec::opt

namespace vecops::vec {

// Addressing factories are also available unqualified alongside load/store.
using opt::indexed;
using opt::scale;
using opt::strided;

} // namespace vecops::vec

namespace vecops::vec::cvt {

/**
 * Preserves the natural logical lane order during conversion.
 * output[i] is derived from input[i] for equal-lane-shape Tags.
 * This is the default layout.
 */
struct Ordered {};

/**
 * Allows a stable backend-native lane permutation.
 *
 * Equal-width element conversion has ordered lane provenance. For any three
 * compatible element types A, B, and C, converting A->B->C has the same lane
 * provenance as converting A->C directly. Consequently A->B->A restores the
 * original lane positions. This contract concerns ordering only; value loss,
 * rounding, saturation, and wrapping follow the selected value policy.
 */
struct Unordered {};

/**
 * Selects or inserts one lane phase of a widening or narrowing conversion.
 *
 * For equal-byte Tags: widening selects input lane ratio*i+Phase, while
 * narrowing writes converted input lane i to output lane ratio*i+Phase.
 * Phase 0 is always supported; Phase 1 requires a 2x element-width ratio.
 */
template <int Phase>
struct Lane {
  static constexpr int phase = Phase;
};

/**
 * Clamps values that are outside a narrowing destination range.
 * For float→int: clamps to [min, max] of the integer destination type.
 * For int→narrower-int: clamps to the narrower range.
 * This is the default value policy.
 */
struct Saturate {};

/**
 * Keeps only the low destination-width bits of a narrowing integer
 * conversion. Valid only for integer-source, narrower-integer-destination
 * conversions. Equivalent to a C-style truncating cast.
 */
struct Wrap {};

inline constexpr Ordered ordered{};
inline constexpr Unordered unordered{};
template <int Phase>
inline constexpr Lane<Phase> lane{};
inline constexpr Saturate saturate{};
inline constexpr Wrap wrap{};
inline constexpr Wrap truncate{};

} // namespace vecops::vec::cvt

namespace vecops::vec::mem {

/** Makes no stronger alignment promise than alignof(element). This is the default. */
struct Unaligned {};

/**
 * Requires the base address to satisfy memory_alignment(tag) byte alignment.
 * An assertion validates alignment at runtime in debug builds.
 */
struct Aligned {};

/** Uses ordinary cacheable memory access semantics. This is the default. */
struct Temporal {};

/**
 * Requests a non-temporal (streaming) access when the selected backend
 * supports one. Silently degrades to a temporal access on unsupported
 * hardware.
 */
struct NonTemporal {};

/**
 *  store_convert narrowing layout: merge all source words into one destination
 *  word before writing.
 *
 *  When a narrowing store_convert spans multiple source words (e.g. two
 *  512-bit f32 words → one 512-bit bf16 word), packed converts every source
 *  half first, concatenates the resulting narrow vectors, and emits a single
 *  wide store.  This avoids multiple partial cache-line writes that would
 *  degrade write throughput on some hardware.
 *
 *  This is the default layout for store_convert.
 */
struct Packed {};

/**
 *  store_convert narrowing layout: write each source word's narrowed result
 *  independently, without merging.
 *
 *  Each source word is converted to its destination representation and stored
 *  immediately at the corresponding offset in the output buffer.  This gives
 *  the backend freedom to select the most natural word-level layout but may
 *  result in multiple narrower stores where packed would have issued one wide
 *  store.
 */
struct Split {};

inline constexpr Unaligned unaligned{};
inline constexpr Aligned aligned{};
inline constexpr Temporal temporal{};
inline constexpr NonTemporal non_temporal{};
inline constexpr Packed packed{};
inline constexpr Split split{};

/**
 * Requests that a prefetched cache line be placed in L1.
 * This is the default cache level for prefetch.
 */
struct PrefetchL1 {};

/** Requests that a prefetched cache line be placed in L2. */
struct PrefetchL2 {};

/**
 * Requests temporal prefetch behavior that keeps the cache line.
 * This is the default temporality for prefetch.
 */
struct PrefetchKeep {};

/**
 * Requests streaming (non-temporal) prefetch behavior.
 * The cache line is evicted after use, avoiding cache pollution.
 */
struct PrefetchStream {};

/**
 * Declares that a prefetch is intended for a subsequent read.
 * This is the default intent for prefetch.
 */
struct PrefetchRead {};

/** Declares that a prefetch is intended for a subsequent write. */
struct PrefetchWrite {};

inline constexpr PrefetchL1 prefetch_l1{};
inline constexpr PrefetchL2 prefetch_l2{};
inline constexpr PrefetchKeep prefetch_keep{};
inline constexpr PrefetchStream prefetch_stream{};
inline constexpr PrefetchRead prefetch_read{};
inline constexpr PrefetchWrite prefetch_write{};

} // namespace vecops::vec::mem

namespace vecops::vec::details {

/* **************************************************************************** */
//    Detection traits for operation-level options: masking, population         //
/* **************************************************************************** */

/** Detects opt::math::AccuracyOption<A> and exposes A as accuracy. */
template <typename>
struct IsMathAccuracyOption : std::false_type {};

template <Accuracy A>
struct IsMathAccuracyOption<opt::math::AccuracyOption<A>>
    : std::true_type {
  static constexpr Accuracy accuracy = A;
};

template <typename T>
inline constexpr bool is_math_accuracy_option_v =
    IsMathAccuracyOption<std::remove_cvref_t<T>>::value;

/** Detects opt::Masked<M> wrappers; the Value alias extracts the mask type. */
template <typename>
struct IsMaskedOption : std::false_type {};

template <MaskValue M>
struct IsMaskedOption<opt::Masked<M>> : std::true_type {
  using Value = M;
};

template <typename T>
inline constexpr bool is_masked_option_v =
    IsMaskedOption<std::remove_cvref_t<T>>::value;

template <typename T>
struct IsUnmaskedOption : std::bool_constant<
    std::same_as<std::remove_cvref_t<T>, opt::Unmasked>> {};

template <typename T>
inline constexpr bool is_unmasked_option_v =
    IsUnmaskedOption<std::remove_cvref_t<T>>::value;

template <typename T>
struct IsFirstOption : std::bool_constant<
    std::same_as<std::remove_cvref_t<T>, opt::First>> {};

template <typename T>
inline constexpr bool is_first_option_v = IsFirstOption<T>::value;

template <typename T>
struct IsZeroOption : std::bool_constant<
    std::same_as<std::remove_cvref_t<T>, opt::Zero>> {};

template <typename T>
inline constexpr bool is_zero_option_v = IsZeroOption<T>::value;

template <typename>
struct IsVectorMergeOption : std::false_type {};

template <VectorValue V>
struct IsVectorMergeOption<opt::VectorMerge<V>> : std::true_type {
  using Value = V;
};

template <typename T>
inline constexpr bool is_vector_merge_option_v =
    IsVectorMergeOption<std::remove_cvref_t<T>>::value;

template <typename>
struct IsMaskMergeOption : std::false_type {};

template <MaskValue M>
struct IsMaskMergeOption<opt::MaskMerge<M>> : std::true_type {
  using Value = M;
};

template <typename T>
inline constexpr bool is_mask_merge_option_v =
    IsMaskMergeOption<std::remove_cvref_t<T>>::value;

template <typename>
struct IsScalarMergeOption : std::false_type {};

template <Element T>
struct IsScalarMergeOption<opt::ScalarMerge<T>> : std::true_type {
  using Value = T;
};

template <typename T>
inline constexpr bool is_scalar_merge_option_v =
    IsScalarMergeOption<std::remove_cvref_t<T>>::value;

/* **************************************************************************** */
//    Tag-checked predicates                                                    //
/* **************************************************************************** */

/** True when the Option's mask type matches Mask<Tag>. */
template <VectorTag Tag, typename Option>
inline constexpr bool is_masked_option_for_v = [] {
  using Clean = std::remove_cvref_t<Option>;
  if constexpr (is_masked_option_v<Clean>) {
    return std::same_as<typename IsMaskedOption<Clean>::Value, Mask<Tag>>;
  } else {
    return false;
  }
}();

/** True when an option is a valid inactive-lane population policy for Tag.
 *  Accepts opt::zero, opt::VectorMerge<Tag>, or opt::ScalarMerge<ElementOf<Tag>>. */
template <VectorTag Tag, typename Option>
inline constexpr bool is_vector_population_option_for_v = [] {
  using Clean = std::remove_cvref_t<Option>;
  if constexpr (is_zero_option_v<Clean>) {
    return true;
  } else if constexpr (is_vector_merge_option_v<Clean>) {
    return std::same_as<typename IsVectorMergeOption<Clean>::Value, Vec<Tag>>;
  } else if constexpr (is_scalar_merge_option_v<Clean>) {
    return std::same_as<typename IsScalarMergeOption<Clean>::Value,
                        ElementOf<Tag>>;
  } else {
    return false;
  }
}();

/** True when an option is a valid mask-population policy: opt::zero or opt::MaskMerge<Tag>. */
template <VectorTag Tag, typename Option>
inline constexpr bool is_mask_population_option_for_v = [] {
  using Clean = std::remove_cvref_t<Option>;
  if constexpr (is_zero_option_v<Clean>) {
    return true;
  } else if constexpr (is_mask_merge_option_v<Clean>) {
    return std::same_as<typename IsMaskMergeOption<Clean>::Value, Mask<Tag>>;
  } else {
    return false;
  }
}();

/* **************************************************************************** */
//    Conversion option detection traits                                        //
/* **************************************************************************** */

template <typename T>
struct IsOrderedOption : std::bool_constant<
    std::same_as<std::remove_cvref_t<T>, cvt::Ordered>> {};

template <typename T>
inline constexpr bool is_ordered_option_v =
    IsOrderedOption<std::remove_cvref_t<T>>::value;

template <typename T>
struct IsUnorderedOption : std::bool_constant<
    std::same_as<std::remove_cvref_t<T>, cvt::Unordered>> {};

template <typename T>
inline constexpr bool is_unordered_option_v =
    IsUnorderedOption<std::remove_cvref_t<T>>::value;

template <typename>
struct IsLaneOption : std::false_type {};

template <typename T>
inline constexpr bool is_lane_option_v =
    IsLaneOption<std::remove_cvref_t<T>>::value;

template <int Phase>
struct IsLaneOption<cvt::Lane<Phase>> : std::true_type {
  static constexpr int phase = Phase;
};

template <typename T>
struct IsSaturateOption : std::bool_constant<
    std::same_as<std::remove_cvref_t<T>, cvt::Saturate>> {};

template <typename T>
inline constexpr bool is_saturate_option_v =
    IsSaturateOption<std::remove_cvref_t<T>>::value;

template <typename T>
struct IsWrapOption : std::bool_constant<
    std::same_as<std::remove_cvref_t<T>, cvt::Wrap>> {};

template <typename T>
inline constexpr bool is_wrap_option_v =
    IsWrapOption<std::remove_cvref_t<T>>::value;

/* **************************************************************************** */
//    Memory option detection traits                                            //
/* **************************************************************************** */

template <typename>
struct IsIndexedOption : std::false_type {};

template <VectorValue V, int Scale>
struct IsIndexedOption<opt::Indexed<V, Scale>> : std::true_type {
  using Value = V;
  static constexpr int scale = Scale;
};

template <typename>
struct IsStridedOption : std::false_type {};

template <typename Stride>
struct IsStridedOption<opt::Strided<Stride>> : std::true_type {
  using Value = Stride;
};

template <typename T>
struct IsUnalignedOption : std::bool_constant<
    std::same_as<std::remove_cvref_t<T>, mem::Unaligned>> {};

template <typename T>
struct IsAlignedOption : std::bool_constant<
    std::same_as<std::remove_cvref_t<T>, mem::Aligned>> {};

template <typename T>
struct IsTemporalOption : std::bool_constant<
    std::same_as<std::remove_cvref_t<T>, mem::Temporal>> {};

template <typename T>
struct IsNonTemporalOption : std::bool_constant<
    std::same_as<std::remove_cvref_t<T>, mem::NonTemporal>> {};

template <typename T>
struct IsPackedOption : std::bool_constant<
    std::same_as<std::remove_cvref_t<T>, mem::Packed>> {};

template <typename T>
struct IsSplitOption : std::bool_constant<
    std::same_as<std::remove_cvref_t<T>, mem::Split>> {};

/* **************************************************************************** */
//    Prefetch option detection traits                                          //
/* **************************************************************************** */

template <typename T>
struct IsPrefetchLocalityOption : std::bool_constant<
    std::same_as<std::remove_cvref_t<T>, mem::PrefetchL1> ||
    std::same_as<std::remove_cvref_t<T>, mem::PrefetchL2>> {};

template <typename T>
struct IsPrefetchTemporalityOption : std::bool_constant<
    std::same_as<std::remove_cvref_t<T>, mem::PrefetchKeep> ||
    std::same_as<std::remove_cvref_t<T>, mem::PrefetchStream>> {};

template <typename T>
struct IsPrefetchIntentOption : std::bool_constant<
    std::same_as<std::remove_cvref_t<T>, mem::PrefetchRead> ||
    std::same_as<std::remove_cvref_t<T>, mem::PrefetchWrite>> {};

template <typename T>
struct IsPrefetchOption : std::bool_constant<
    IsPrefetchLocalityOption<T>::value ||
    IsPrefetchTemporalityOption<T>::value ||
    IsPrefetchIntentOption<T>::value> {};

/* **************************************************************************** */
//    Memory option group predicates                                           //
/* **************************************************************************** */

template <typename T>
struct IsMemoryAlignmentOption : std::bool_constant<
    IsUnalignedOption<T>::value || IsAlignedOption<T>::value> {};

template <typename T>
struct IsMemoryTemporalityOption : std::bool_constant<
    IsTemporalOption<T>::value || IsNonTemporalOption<T>::value> {};

template <typename T>
struct IsMemoryActiveOption : std::bool_constant<
    IsMaskedOption<T>::value || IsUnmaskedOption<T>::value ||
    IsFirstOption<T>::value> {};

template <typename T>
struct IsMemoryPopulationOption : std::bool_constant<
    IsZeroOption<T>::value || IsVectorMergeOption<T>::value ||
    IsScalarMergeOption<T>::value> {};

template <typename T>
struct IsMemoryAddressingOption : std::bool_constant<
    IsIndexedOption<T>::value || IsStridedOption<T>::value> {};

/** store_convert narrowing layout: mem::Packed or mem::Split. */
template <typename T>
struct IsConversionMemoryPackingOption : std::bool_constant<
    IsPackedOption<T>::value || IsSplitOption<T>::value> {};

/* **************************************************************************** */
//    Option count and search utilities                                         //
/* **************************************************************************** */

/**
 * Compile-time count of Options matching a unary Predicate.
 * Each option's type is cleaned via std::remove_cvref_t before testing.
 */
template <template <typename> typename Predicate, typename... Options>
inline constexpr std::size_t option_count_v =
    (std::size_t{0} + ... +
     std::size_t{Predicate<std::remove_cvref_t<Options>>::value});

/**
 * Validates that a prefetch call selects at most one option from each
 * independent dimension: locality (L1/L2), temporality (keep/stream),
 * and intent (read/write).
 */
template <typename... Options>
consteval bool valid_prefetch_options() {
  return (IsPrefetchOption<std::remove_cvref_t<Options>>::value && ...) &&
      option_count_v<IsPrefetchLocalityOption, Options...> <= 1 &&
      option_count_v<IsPrefetchTemporalityOption, Options...> <= 1 &&
      option_count_v<IsPrefetchIntentOption, Options...> <= 1;
}

/**
 * Compile-time linear search returning a reference to the first option
 * that satisfies Predicate.  References are preserved for sizeless SVE
 * types which cannot be moved or copied.
 */
template <template <typename> typename Predicate,
          typename First, typename... Rest>
VECOPS_ALWAYS_INLINE constexpr decltype(auto) find_option(
    First&& first, Rest&&... rest) {
  if constexpr (Predicate<std::remove_cvref_t<First>>::value) {
    return std::forward<First>(first);
  } else {
    static_assert(sizeof...(Rest) > 0, "requested option is absent");
    return find_option<Predicate>(std::forward<Rest>(rest)...);
  }
}

/**
 * Type-level counterpart of `find_option`: names the cleaned type of the
 * first option satisfying Predicate, or `void` when absent. Used by the
 * Request resolver to fold an option pack once into resolved kind types.
 */
template <template <typename> typename Predicate, typename... Options>
struct FindOptionType {
  using type = void;
};

template <template <typename> typename Predicate,
          typename First, typename... Rest>
struct FindOptionType<Predicate, First, Rest...> {
  using type = std::conditional_t<
      Predicate<std::remove_cvref_t<First>>::value,
      std::remove_cvref_t<First>,
      typename FindOptionType<Predicate, Rest...>::type>;
};

template <template <typename> typename Predicate, typename... Options>
using find_option_type_t = typename FindOptionType<Predicate, Options...>::type;

} // namespace vecops::vec::details
#endif // VECOPS_VEC_OPTIONS_H
