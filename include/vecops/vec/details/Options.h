#ifndef VECOPS_VEC_DETAILS_OPTIONS_H
#define VECOPS_VEC_DETAILS_OPTIONS_H

#include <cstddef>
#include <type_traits>
#include <utility>

#include "vecops/vec/Options.h"

/**
 * @file Options.h
 * @brief Compile-time option detection and validation traits.
 *
 * Each operation family (arithmetic, bit, comparison, memory, etc.) declares
 * its valid option set via `constexpr` predicates. The option validation
 * layer uses these traits to:
 *
 * - Detect the kind of each option (masked, unmasked, merge, alignment, etc.)
 * - Validate that exactly one of mutually-exclusive groups is present
 * - Find and extract individual options from a variadic pack
 *
 * These traits are the foundation of the Options "type-safe tagged union"
 * pattern used throughout the public API.
 */

#include <cstddef>
#include <type_traits>
#include <utility>

#include "vecops/vec/Options.h"

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
inline constexpr bool is_math_accuracy_option =
    IsMathAccuracyOption<std::remove_cvref_t<T>>::value;

/** Detects opt::Masked<M> wrappers; the Value alias extracts the mask type. */
template <typename>
struct IsMaskedOption : std::false_type {};

template <MaskValue M>
struct IsMaskedOption<opt::Masked<M>> : std::true_type {
  using Value = M;
};

template <typename T>
inline constexpr bool is_masked_option =
    IsMaskedOption<std::remove_cvref_t<T>>::value;

template <typename T>
struct IsUnmaskedOption : std::bool_constant<
    std::same_as<std::remove_cvref_t<T>, opt::Unmasked>> {};

template <typename T>
inline constexpr bool is_unmasked_option = IsUnmaskedOption<T>::value;

template <typename T>
struct IsFirstOption : std::bool_constant<
    std::same_as<std::remove_cvref_t<T>, opt::First>> {};

template <typename T>
inline constexpr bool is_first_option = IsFirstOption<T>::value;

template <typename T>
struct IsZeroOption : std::bool_constant<
    std::same_as<std::remove_cvref_t<T>, opt::Zero>> {};

template <typename T>
inline constexpr bool is_zero_option = IsZeroOption<T>::value;

template <typename>
struct IsVectorMergeOption : std::false_type {};

template <VectorValue V>
struct IsVectorMergeOption<opt::VectorMerge<V>> : std::true_type {
  using Value = V;
};

template <typename T>
inline constexpr bool is_vector_merge_option =
    IsVectorMergeOption<std::remove_cvref_t<T>>::value;

template <typename>
struct IsMaskMergeOption : std::false_type {};

template <MaskValue M>
struct IsMaskMergeOption<opt::MaskMerge<M>> : std::true_type {
  using Value = M;
};

template <typename T>
inline constexpr bool is_mask_merge_option =
    IsMaskMergeOption<std::remove_cvref_t<T>>::value;

template <typename>
struct IsScalarMergeOption : std::false_type {};

template <Element T>
struct IsScalarMergeOption<opt::ScalarMerge<T>> : std::true_type {
  using Value = T;
};

template <typename T>
inline constexpr bool is_scalar_merge_option =
    IsScalarMergeOption<std::remove_cvref_t<T>>::value;

/* **************************************************************************** */
//    Tag-checked predicates                                                    //
/* **************************************************************************** */

/** True when the Option's mask type matches Mask<Tag>. */
template <VectorTag Tag, typename Option>
inline constexpr bool is_masked_option_for = [] {
  using Clean = std::remove_cvref_t<Option>;
  if constexpr (is_masked_option<Clean>) {
    return std::same_as<typename IsMaskedOption<Clean>::Value, Mask<Tag>>;
  } else {
    return false;
  }
}();

/** True when an option is a valid inactive-lane population policy for Tag.
 *  Accepts opt::zero, opt::VectorMerge<Tag>, or opt::ScalarMerge<ElementOf<Tag>>. */
template <VectorTag Tag, typename Option>
inline constexpr bool is_vector_population_option_for = [] {
  using Clean = std::remove_cvref_t<Option>;
  if constexpr (is_zero_option<Clean>) {
    return true;
  } else if constexpr (is_vector_merge_option<Clean>) {
    return std::same_as<typename IsVectorMergeOption<Clean>::Value, Vec<Tag>>;
  } else if constexpr (is_scalar_merge_option<Clean>) {
    return std::same_as<typename IsScalarMergeOption<Clean>::Value,
                        ElementOf<Tag>>;
  } else {
    return false;
  }
}();

/** True when an option is a valid mask-population policy: opt::zero or opt::MaskMerge<Tag>. */
template <VectorTag Tag, typename Option>
inline constexpr bool is_mask_population_option_for = [] {
  using Clean = std::remove_cvref_t<Option>;
  if constexpr (is_zero_option<Clean>) {
    return true;
  } else if constexpr (is_mask_merge_option<Clean>) {
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
struct IsUnorderedOption : std::bool_constant<
    std::same_as<std::remove_cvref_t<T>, cvt::Unordered>> {};

template <typename>
struct IsLaneOption : std::false_type {};

template <int Phase>
struct IsLaneOption<cvt::Lane<Phase>> : std::true_type {
  static constexpr int phase = Phase;
};

template <typename T>
struct IsSaturateOption : std::bool_constant<
    std::same_as<std::remove_cvref_t<T>, cvt::Saturate>> {};

template <typename T>
struct IsWrapOption : std::bool_constant<
    std::same_as<std::remove_cvref_t<T>, cvt::Wrap>> {};

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
//    Option count and search utilities                                         //
/* **************************************************************************** */

/**
 * Compile-time count of Options matching a unary Predicate.
 * Each option's type is cleaned via std::remove_cvref_t before testing.
 */
template <template <typename> typename Predicate, typename... Options>
inline constexpr std::size_t option_count =
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
      option_count<IsPrefetchLocalityOption, Options...> <= 1 &&
      option_count<IsPrefetchTemporalityOption, Options...> <= 1 &&
      option_count<IsPrefetchIntentOption, Options...> <= 1;
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

} // namespace vecops::vec::details

#endif // VECOPS_VEC_DETAILS_OPTIONS_H
