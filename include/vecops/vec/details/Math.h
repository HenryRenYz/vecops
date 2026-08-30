#ifndef VECOPS_VEC_DETAILS_MATH_H
#define VECOPS_VEC_DETAILS_MATH_H

/**
 * @file Math.h
 * @brief Backend-independent exponential batching and option dispatch.
 */

#include "vecops/vec/details/Arithmetic.h"

namespace vecops::vec::details {

template <ExpBase Base, Accuracy A, bool NegativeOnly>
struct EnableElementwiseWordBatching<ExpOp<Base, A, NegativeOnly>>
    : std::true_type {};

template <
    typename Backend,
    ExpBase Base,
    Accuracy A,
    bool NegativeOnly,
    VectorTag Tag>
struct GenericImpl<Backend, ExpOp<Base, A, NegativeOnly>, Tag>
    : UnaryArithmeticGenericImpl<
          Backend, ExpOp<Base, A, NegativeOnly>, Tag> {};

template <Accuracy A>
struct EnableElementwiseWordBatching<RsqrtOp<A>> : std::true_type {};

template <Accuracy A>
struct EnableElementwiseWordBatching<RcpOp<A>> : std::true_type {};

template <typename Backend, Accuracy A, VectorTag Tag>
struct GenericImpl<Backend, RsqrtOp<A>, Tag>
    : UnaryArithmeticGenericImpl<Backend, RsqrtOp<A>, Tag> {};

template <typename Backend, Accuracy A, VectorTag Tag>
struct GenericImpl<Backend, RcpOp<A>, Tag>
    : UnaryArithmeticGenericImpl<Backend, RcpOp<A>, Tag> {};

template <LogBase Base, Accuracy A>
struct EnableElementwiseWordBatching<LogOp<Base, A>> : std::true_type {};

template <typename Backend, LogBase Base, Accuracy A, VectorTag Tag>
struct GenericImpl<Backend, LogOp<Base, A>, Tag>
    : UnaryArithmeticGenericImpl<Backend, LogOp<Base, A>, Tag> {};

/** Invokes f with every option except the compile-time math-accuracy option. */
template <typename F>
VECOPS_ALWAYS_INLINE decltype(auto) invoke_without_math_accuracy(F&& f) {
  return std::forward<F>(f)();
}

template <typename F, typename First, typename... Rest>
VECOPS_ALWAYS_INLINE decltype(auto) invoke_without_math_accuracy(
    F&& f, First&& first, Rest&&... rest) {
  if constexpr (is_math_accuracy_option_v<First>) {
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
  if constexpr (is_first_option_v<First>) {
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
template <ExpBase Base, bool NegativeOnly, FloatingTag Tag, typename... Options>
VECOPS_ALWAYS_INLINE Vec<Tag> execute_exp_options(
    Tag tag, Vec<Tag> value, Options&&... options) {
  static_assert(valid_exp_options_for<Tag, Options...>());
  constexpr Accuracy accuracy = selected_math_accuracy<Options...>();
  const auto dispatch = [&]<typename... ArithmeticOptions>(
                            ArithmeticOptions&&... arithmetic_options)
      -> Vec<Tag> {
    if constexpr (sizeof...(ArithmeticOptions) == 0) {
      return execute(ExpOp<Base, accuracy, NegativeOnly>{}, tag, value);
    } else if constexpr (
        option_count_v<IsFirstOption, ArithmeticOptions...> == 1) {
      const nint_t count = find_option<IsFirstOption>(
          arithmetic_options...).count;
      const auto mask = mwhilelt(tag, 0, count);
      return invoke_replacing_first(
          [&](auto&&... lowered_options) -> Vec<Tag> {
            return execute_unary_arithmetic_options(
                ExpOp<Base, accuracy, NegativeOnly>{}, tag, value,
                std::forward<decltype(lowered_options)>(lowered_options)...);
          },
          mask, std::forward<ArithmeticOptions>(arithmetic_options)...);
    } else {
      return execute_unary_arithmetic_options(
          ExpOp<Base, accuracy, NegativeOnly>{}, tag, value,
          std::forward<ArithmeticOptions>(arithmetic_options)...);
    }
  };
  return invoke_without_math_accuracy(
      dispatch, std::forward<Options>(options)...);
}

/**
 * Same dispatch shape as execute_exp_options, but for the unary math ops
 * whose token is a template over Accuracy alone (RsqrtOp, RcpOp). Selects
 * the tier, strips the accuracy option, and reuses the standard unary
 * mask/population dispatcher — including the opt::first-to-mask lowering —
 * for whatever remains.
 */
template <template <Accuracy> class OpToken, FloatingTag Tag, typename... Options>
VECOPS_ALWAYS_INLINE Vec<Tag> execute_unary_math_options(
    Tag tag, Vec<Tag> value, Options&&... options) {
  constexpr Accuracy accuracy = selected_math_accuracy<Options...>();
  const auto dispatch = [&]<typename... ArithmeticOptions>(
                            ArithmeticOptions&&... arithmetic_options)
      -> Vec<Tag> {
    if constexpr (sizeof...(ArithmeticOptions) == 0) {
      return execute(OpToken<accuracy>{}, tag, value);
    } else if constexpr (
        option_count_v<IsFirstOption, ArithmeticOptions...> == 1) {
      // opt::first is the only masking option in a valid pack; lower it to
      // a mask and dispatch the population explicitly. The branches are
      // spelled out (instead of re-entering execute_unary_arithmetic_options
      // through a replacement lambda) to keep the op token's compile-time
      // accuracy out of nested closure captures.
      const auto count = find_option<IsFirstOption>(
          arithmetic_options...).count;
      const auto mask = mwhilelt(tag, 0, count);
      constexpr std::size_t zero_count =
          option_count_v<IsZeroOption, ArithmeticOptions...>;
      constexpr std::size_t vector_merge_count =
          option_count_v<IsVectorMergeOption, ArithmeticOptions...>;
      constexpr std::size_t scalar_merge_count =
          option_count_v<IsScalarMergeOption, ArithmeticOptions...>;
      constexpr OpToken<accuracy> op{};
      if constexpr (zero_count == 1) {
        return execute(op, tag, value, mask, zeros(tag),
                       ZeroArithmeticInactive{});
      } else if constexpr (vector_merge_count == 1) {
        const auto& inactive = find_option<IsVectorMergeOption>(
            arithmetic_options...).value;
        return execute(
            op, tag, value, mask, inactive, MergeArithmeticInactive{});
      } else if constexpr (scalar_merge_count == 1) {
        const auto inactive = fill(
            tag, find_option<IsScalarMergeOption>(
                     arithmetic_options...).value);
        return execute(
            op, tag, value, mask, inactive, MergeArithmeticInactive{});
      } else {
        return execute(
            op, tag, value, mask, value, PreserveArithmeticInactive{});
      }
    } else {
      return execute_unary_arithmetic_options(
          OpToken<accuracy>{}, tag, value,
          std::forward<ArithmeticOptions>(arithmetic_options)...);
    }
  };
  return invoke_without_math_accuracy(
      dispatch, std::forward<Options>(options)...);
}

/**
 * Same dispatch shape for the logarithm family, whose token additionally
 * carries the LogBase. Selects the tier, strips the accuracy option, and
 * lowers opt::first to an explicitly dispatched population branch (the
 * spelled-out form avoids nested closure captures around the compile-time
 * accuracy, mirroring execute_unary_math_options).
 */
template <LogBase Base, FloatingTag Tag, typename... Options>
VECOPS_ALWAYS_INLINE Vec<Tag> execute_log_options(
    Tag tag, Vec<Tag> value, Options&&... options) {
  static_assert(valid_log_options_for<Tag, Options...>());
  constexpr Accuracy accuracy = selected_math_accuracy<Options...>();
  const auto dispatch = [&]<typename... ArithmeticOptions>(
                            ArithmeticOptions&&... arithmetic_options)
      -> Vec<Tag> {
    if constexpr (sizeof...(ArithmeticOptions) == 0) {
      return execute(LogOp<Base, accuracy>{}, tag, value);
    } else if constexpr (
        option_count_v<IsFirstOption, ArithmeticOptions...> == 1) {
      const auto count = find_option<IsFirstOption>(
          arithmetic_options...).count;
      const auto mask = mwhilelt(tag, 0, count);
      constexpr std::size_t zero_count =
          option_count_v<IsZeroOption, ArithmeticOptions...>;
      constexpr std::size_t vector_merge_count =
          option_count_v<IsVectorMergeOption, ArithmeticOptions...>;
      constexpr std::size_t scalar_merge_count =
          option_count_v<IsScalarMergeOption, ArithmeticOptions...>;
      constexpr LogOp<Base, accuracy> op{};
      if constexpr (zero_count == 1) {
        return execute(op, tag, value, mask, zeros(tag),
                       ZeroArithmeticInactive{});
      } else if constexpr (vector_merge_count == 1) {
        const auto& inactive = find_option<IsVectorMergeOption>(
            arithmetic_options...).value;
        return execute(
            op, tag, value, mask, inactive, MergeArithmeticInactive{});
      } else if constexpr (scalar_merge_count == 1) {
        const auto inactive = fill(
            tag, find_option<IsScalarMergeOption>(
                     arithmetic_options...).value);
        return execute(
            op, tag, value, mask, inactive, MergeArithmeticInactive{});
      } else {
        return execute(
            op, tag, value, mask, value, PreserveArithmeticInactive{});
      }
    } else {
      return execute_unary_arithmetic_options(
          LogOp<Base, accuracy>{}, tag, value,
          std::forward<ArithmeticOptions>(arithmetic_options)...);
    }
  };
  return invoke_without_math_accuracy(
      dispatch, std::forward<Options>(options)...);
}

} // namespace vecops::vec::details

#endif // VECOPS_VEC_DETAILS_MATH_H
