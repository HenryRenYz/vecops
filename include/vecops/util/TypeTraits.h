//
// Created by renyz on 2026/3/23.
//

#ifndef VECOPS_TYPETRAITS_H
#define VECOPS_TYPETRAITS_H

#include <type_traits>

#include "CoreTypes.h"

/**
 * Alias macro defining an constraint for a function, used as template parameter.
 * Used like:
 *   template <typename T, TL_IF(sizeof(T) == 4)>
 *   void something(T t, int a) { ... }
 */
#define TL_IF(...) std::enable_if_t<(__VA_ARGS__), bool> = true

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

template <typename T, typename... TArgs>
static constexpr bool is_any = details::IsAnyHelper<T, TArgs...>::value;

template <typename T, typename ... TArgs>
static constexpr bool is_none = !is_any<T, TArgs...> || sizeof...(TArgs) == 0;

template <typename Ta, typename Tb>
static constexpr bool is_same = is_any<Ta, Tb>;

template <typename T>
static constexpr bool is_int = is_any<T, int8_t, uint8_t, int16_t, uint16_t, int32_t, uint32_t, int64_t, uint64_t>;

template <typename T>
static constexpr bool is_small_float = is_any<T, float16_t, bfloat16_t>;

template <typename T>
static constexpr bool is_float = is_any<T, float32_t, float64_t> || is_small_float<T>;

template <typename T>
static constexpr bool is_signed_int = is_any<T, int8_t, int16_t, int32_t, int64_t>;

template <typename T>
static constexpr bool is_unsigned_int = is_any<T, uint8_t, uint16_t, uint32_t, uint64_t>;

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
