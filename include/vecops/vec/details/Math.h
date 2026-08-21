#ifndef VECOPS_VEC_DETAILS_MATH_H
#define VECOPS_VEC_DETAILS_MATH_H

/**
 * @file Math.h
 * @brief Backend-independent exponential batching and option dispatch.
 */

#include "vecops/vec/details/Arithmetic.h"

namespace vecops::vec::details {

template <Accuracy A, bool NegativeOnly>
struct EnableElementwiseWordBatching<ExpOp<A, NegativeOnly>>
    : std::true_type {};

template <
    typename Backend,
    Accuracy A,
    bool NegativeOnly,
    VectorTag Tag>
struct GenericImpl<Backend, ExpOp<A, NegativeOnly>, Tag>
    : UnaryArithmeticGenericImpl<
          Backend, ExpOp<A, NegativeOnly>, Tag> {};

/** Invokes f with every option except the compile-time math-accuracy option. */
template <typename F>
VECOPS_ALWAYS_INLINE decltype(auto) invoke_without_math_accuracy(F&& f) {
  return std::forward<F>(f)();
}

template <typename F, typename First, typename... Rest>
VECOPS_ALWAYS_INLINE decltype(auto) invoke_without_math_accuracy(
    F&& f, First&& first, Rest&&... rest) {
  if constexpr (is_math_accuracy_option<First>) {
    return invoke_without_math_accuracy(
        std::forward<F>(f), std::forward<Rest>(rest)...);
  } else {
    return invoke_without_math_accuracy(
        [&f, &first]<typename... Tail>(Tail&&... tail) -> decltype(auto) {
          return std::forward<F>(f)(
              std::forward<First>(first),
              std::forward<Tail>(tail)...);
        },
        std::forward<Rest>(rest)...);
  }
}

template <typename F, MaskValue Mask>
VECOPS_ALWAYS_INLINE decltype(auto) invoke_replacing_first(
    F&& f, const Mask&) {
  return std::forward<F>(f)();
}

template <typename F, MaskValue Mask, typename First, typename... Rest>
VECOPS_ALWAYS_INLINE decltype(auto) invoke_replacing_first(
    F&& f, const Mask& mask, First&& first, Rest&&... rest) {
  if constexpr (is_first_option<First>) {
    return invoke_replacing_first(
        [&f, &mask]<typename... Tail>(Tail&&... tail) -> decltype(auto) {
          return std::forward<F>(f)(
              opt::masked(mask), std::forward<Tail>(tail)...);
        },
        mask, std::forward<Rest>(rest)...);
  } else {
    return invoke_replacing_first(
        [&f, &first]<typename... Tail>(Tail&&... tail) -> decltype(auto) {
          return std::forward<F>(f)(
              std::forward<First>(first), std::forward<Tail>(tail)...);
        },
        mask, std::forward<Rest>(rest)...);
  }
}

/**
 * Selects one ExpOp specialization, removes the accuracy option, and reuses
 * the standard unary mask/population dispatcher for the remaining options.
 */
template <bool NegativeOnly, FloatingTag Tag, typename... Options>
VECOPS_ALWAYS_INLINE Vec<Tag> execute_exp_options(
    Tag tag, Vec<Tag> value, Options&&... options) {
  static_assert(valid_exp_options_for<Tag, Options...>());
  constexpr Accuracy accuracy = selected_math_accuracy<Options...>();
  const auto dispatch = [&]<typename... ArithmeticOptions>(
                            ArithmeticOptions&&... arithmetic_options)
      -> Vec<Tag> {
    if constexpr (sizeof...(ArithmeticOptions) == 0) {
      return execute(ExpOp<accuracy, NegativeOnly>{}, tag, value);
    } else if constexpr (
        option_count<IsFirstOption, ArithmeticOptions...> == 1) {
      const nint_t count = find_option<IsFirstOption>(
          arithmetic_options...).count;
      const auto mask = mwhilelt(tag, 0, count);
      return invoke_replacing_first(
          [&](auto&&... lowered_options) -> Vec<Tag> {
            return execute_unary_arithmetic_options(
                ExpOp<accuracy, NegativeOnly>{}, tag, value,
                std::forward<decltype(lowered_options)>(lowered_options)...);
          },
          mask, std::forward<ArithmeticOptions>(arithmetic_options)...);
    } else {
      return execute_unary_arithmetic_options(
          ExpOp<accuracy, NegativeOnly>{}, tag, value,
          std::forward<ArithmeticOptions>(arithmetic_options)...);
    }
  };
  return invoke_without_math_accuracy(
      dispatch, std::forward<Options>(options)...);
}

} // namespace vecops::vec::details

#endif // VECOPS_VEC_DETAILS_MATH_H
