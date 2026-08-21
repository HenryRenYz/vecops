//
// Created by renyz on 2026/3/23.
//

#ifndef VECOPS_TYPETRAITS_H
#define VECOPS_TYPETRAITS_H

#include <type_traits>

#include "CoreTypes.h"

// Single home of the base element-type predicates (see docs/TemplateStyle.md).
// Whitelist semantics: only fixed-width integers and float32/64/float16/bfloat16
// qualify; long double, bool and char do not. All traits are snake_case + _v.

namespace vecops {
namespace details {

template <typename T, typename... TArgs>
struct IsAnyHelper {};

// Empty input type list defaults to false
template <typename T>
struct IsAnyHelper<T> {
  static constexpr bool value = false;
};

template <typename T, typename T1, typename... TArgs>
struct IsAnyHelper<T, T1, TArgs...> {
  static constexpr bool value = std::is_same_v<T, T1> || IsAnyHelper<T, TArgs...>::value;
};

} // namespace details

/** True iff T is one of TArgs. */
template <typename T, typename... TArgs>
inline constexpr bool is_any_v = details::IsAnyHelper<T, TArgs...>::value;

/** True iff T is none of TArgs (vacuously true for an empty list). */
template <typename T, typename ... TArgs>
inline constexpr bool is_none_v = !is_any_v<T, TArgs...> || sizeof...(TArgs) == 0;

/** True for the fixed-width integer element types (int8..uint64; no bool/char). */
template <typename T>
inline constexpr bool is_int_v =
    is_any_v<T, int8_t, uint8_t, int16_t, uint16_t, int32_t, uint32_t, int64_t, uint64_t>;

/** True for float16_t / bfloat16_t. */
template <typename T>
inline constexpr bool is_small_float_v = is_any_v<T, float16_t, bfloat16_t>;

/** True for float32_t / float64_t / float16_t / bfloat16_t (no long double). */
template <typename T>
inline constexpr bool is_float_v = is_any_v<T, float32_t, float64_t> || is_small_float_v<T>;

/** True for bfloat16_t. */
template <typename T>
inline constexpr bool is_bfloat16_v = std::is_same_v<T, bfloat16_t>;

/** True for float16_t. */
template <typename T>
inline constexpr bool is_float16_v = std::is_same_v<T, float16_t>;

/** True for the unsigned fixed-width integer element types. */
template <typename T>
inline constexpr bool is_unsigned_int_v = is_any_v<T, uint8_t, uint16_t, uint32_t, uint64_t>;

/**
 * @brief One branch in a compile-time first-match type selection chain.
 *
 * `chain_opt<Cond, T>` is intended to be used with `chain_if`. The first
 * option whose `Cond` is `true` supplies its `T` as the selected type.
 *
 * @code
 * using T = chain_if_t<
 *     chain_opt<false, int>,
 *     chain_opt<true, float>,
 *     chain_opt<true, double>>;  // T == float
 * @endcode
 *
 * @tparam Cond Whether this option matches.
 * @tparam T    Type returned if this option is the first match.
 */
template <bool Cond, typename T>
struct chain_opt {
  static constexpr bool value = Cond;
  using type = T;
};

/**
 * @brief Select the type from the first matching `chain_opt`.
 *
 * If no option matches, `chain_if<...>` intentionally has no `type`. This
 * mirrors `std::enable_if` and makes the helper useful in SFINAE contexts.
 * Overlapping conditions are resolved by source order: the first `true`
 * branch wins.
 *
 * @tparam Options Sequence of `chain_opt<Cond, T>` branches.
 */
template <typename... Options>
struct chain_if {};

template <typename T, typename... Rest>
struct chain_if<chain_opt<true, T>, Rest...> {
  using type = T;
};

template <typename T, typename... Rest>
struct chain_if<chain_opt<false, T>, Rest...> : chain_if<Rest...> {};

/**
 * @brief Convenience alias for `typename chain_if<...>::type`.
 */
template <typename... Options>
using chain_if_t = typename chain_if<Options...>::type;

} // namespace vecops

#endif //VECOPS_TYPETRAITS_H
