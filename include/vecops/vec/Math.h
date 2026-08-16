#ifndef VECOPS_VEC_MATH_H
#define VECOPS_VEC_MATH_H

/**
 * @file Math.h
 * @brief Exponential operations with compile-time accuracy options.
 *
 * `exp` and `exp_neg` are the canonical entry points.  With no math option
 * they use `Accuracy::Strict`; callers can select another implementation with
 * `opt::math::strict`, `opt::math::fast`, `opt::math::estimate`, or the generic
 * `opt::math::accuracy<A>` option.
 *
 * | Accuracy | Normal-input error contract |
 * |----------|-----------------------------|
 * | Strict   | ULP error <= 1              |
 * | Fast     | ULP error <= 4              |
 * | Estimate | ULP error <= 4 or relative error <= 0.006 |
 *
 * `exp_strict`, `exp_fast`, and `exp_est` are convenience forwarding CPOs.
 * Their `exp_neg_*` counterparts select the same accuracy while assuming every
 * active input lane is `x <= 0`.  Accuracy options are deliberately rejected by
 * these fixed-accuracy CPOs; use `exp` or `exp_neg` when forwarding an accuracy
 * selected by a template parameter.
 *
 * The negative-only family skips the overflow-test path. Active positive lanes
 * therefore have unspecified results. `VECOPS_MATH_ASSUME_VALID_INPUTS` also
 * skips NaN propagation checks on this path.
 *
 * When `VECOPS_PRESERVE_SUBNORMALS` is defined, Strict preserves representable
 * subnormal results. Fast and Estimate always flush subnormal outputs to zero.
 *
 * Accuracy is orthogonal to the unary masking options. Calls may combine one
 * accuracy option with exactly one `opt::masked(mask)` or `opt::unmasked` and
 * the usual `opt::zero`/`opt::merge` inactive-lane policy, in any order. A call
 * containing only an accuracy option is an ordinary unmasked call.
 */

#include "vecops/vec/Arithmetic.h"
#include "vecops/vec/Bit.h"
#include "vecops/vec/Comparison.h"

namespace vecops::vec {

namespace details {

/** One related implementation token for every accuracy/domain combination. */
template <Accuracy A, bool NegativeOnly>
struct ExpOp {};

template <FloatingTag Tag, typename... Options>
consteval bool valid_exp_options_for() {
  constexpr std::size_t accuracy_count =
      option_count<IsMathAccuracyOption, Options...>;
  if constexpr (accuracy_count > 1) return false;
  if constexpr (!((is_math_accuracy_option<Options> ||
                    is_arithmetic_option_for<Tag, Options>) && ...))
    return false;

  constexpr std::size_t arithmetic_count =
      sizeof...(Options) - accuracy_count;
  if constexpr (arithmetic_count == 0) return true;

  constexpr std::size_t masked_count =
      option_count<IsMaskedOption, Options...>;
  constexpr std::size_t unmasked_count =
      option_count<IsUnmaskedOption, Options...>;
  constexpr std::size_t zero_count = option_count<IsZeroOption, Options...>;
  constexpr std::size_t vector_merge_count =
      option_count<IsVectorMergeOption, Options...>;
  constexpr std::size_t scalar_merge_count =
      option_count<IsScalarMergeOption, Options...>;
  return masked_count + unmasked_count == 1 &&
      masked_count * unmasked_count == 0 && zero_count <= 1 &&
      vector_merge_count + scalar_merge_count <= 1 &&
      zero_count + vector_merge_count + scalar_merge_count <= 1;
}

template <typename... Options>
consteval Accuracy selected_math_accuracy() {
  Accuracy result = Accuracy::Strict;
  ([&] {
    using Option = std::remove_cvref_t<Options>;
    if constexpr (IsMathAccuracyOption<Option>::value)
      result = IsMathAccuracyOption<Option>::accuracy;
  }(), ...);
  return result;
}

template <typename... Options>
inline constexpr bool has_math_accuracy_option =
    option_count<IsMathAccuracyOption, Options...> != 0;

} // namespace details

/** Canonical exponential CPO, parameterized only by its valid input domain. */
template <bool NegativeOnly>
struct ExpCpo {
  template <FloatingTag Tag>
  VECOPS_ALWAYS_INLINE Vec<Tag> operator()(Tag tag, Vec<Tag> value) const;

  template <Accuracy A, FloatingTag Tag, typename... ArithmeticOptions>
    requires ((!details::is_math_accuracy_option<ArithmeticOptions>) && ... &&
              details::valid_exp_options_for<
                  Tag,
                  opt::math::AccuracyOption<A>,
                  ArithmeticOptions...>())
  VECOPS_ALWAYS_INLINE Vec<Tag> operator()(
      Tag tag,
      Vec<Tag> value,
      opt::math::AccuracyOption<A>,
      ArithmeticOptions&&... arithmetic_options) const;

  template <FloatingTag Tag, typename... Options>
    requires (sizeof...(Options) > 0 &&
              details::valid_exp_options_for<Tag, Options...>())
  VECOPS_ALWAYS_INLINE Vec<Tag> operator()(
      Tag tag, Vec<Tag> value, Options&&... options) const;

  template <FloatingVectorValue V, typename... Options>
    requires (details::valid_exp_options_for<VecToTagT<V>, Options...>())
  VECOPS_ALWAYS_INLINE V operator()(V value, Options&&... options) const {
    return (*this)(
        VecToTagT<V>{}, value, std::forward<Options>(options)...);
  }
};

/** Fixed-accuracy forwarding CPO used by exp_fast/exp_est/etc. */
template <Accuracy A, bool NegativeOnly>
struct FixedAccuracyExpCpo {
  template <FloatingTag Tag, typename... Options>
    requires (!details::has_math_accuracy_option<Options...> &&
              details::valid_exp_options_for<
                  Tag, Options..., decltype(opt::math::accuracy<A>)>())
  VECOPS_ALWAYS_INLINE Vec<Tag> operator()(
      Tag tag, Vec<Tag> value, Options&&... options) const {
    return ExpCpo<NegativeOnly>{}(
        tag, value, opt::math::accuracy<A>,
        std::forward<Options>(options)...);
  }

  template <FloatingVectorValue V, typename... Options>
    requires (!details::has_math_accuracy_option<Options...> &&
              details::valid_exp_options_for<
                  VecToTagT<V>, Options...,
                  decltype(opt::math::accuracy<A>)>())
  VECOPS_ALWAYS_INLINE V operator()(V value, Options&&... options) const {
    return (*this)(
        VecToTagT<V>{}, value, std::forward<Options>(options)...);
  }
};

} // namespace vecops::vec

#include "vecops/vec/details/Dispatch.h"
#include "vecops/vec/details/scalar/Math.h"

#if defined(ARCH_X86_FAMILY) && !defined(CPU_CAPABILITY_GENERIC)
#include "vecops/vec/details/x86/Math.h"
#elif defined(CPU_CAPABILITY_SVE)
#include "vecops/vec/details/sve/Math.h"
#endif

#include "vecops/vec/details/Math.h"

namespace vecops::vec {

template <bool NegativeOnly>
template <FloatingTag Tag>
VECOPS_ALWAYS_INLINE Vec<Tag> ExpCpo<NegativeOnly>::operator()(
    Tag tag, Vec<Tag> value) const {
  return details::execute(
      details::ExpOp<Accuracy::Strict, NegativeOnly>{}, tag, value);
}

template <bool NegativeOnly>
template <Accuracy A, FloatingTag Tag, typename... ArithmeticOptions>
  requires ((!details::is_math_accuracy_option<ArithmeticOptions>) && ... &&
            details::valid_exp_options_for<
                Tag,
                opt::math::AccuracyOption<A>,
                ArithmeticOptions...>())
VECOPS_ALWAYS_INLINE Vec<Tag> ExpCpo<NegativeOnly>::operator()(
    Tag tag,
    Vec<Tag> value,
    opt::math::AccuracyOption<A>,
    ArithmeticOptions&&... arithmetic_options) const {
  if constexpr (sizeof...(ArithmeticOptions) == 0) {
    return details::execute(details::ExpOp<A, NegativeOnly>{}, tag, value);
  } else {
    return details::execute_unary_arithmetic_options(
        details::ExpOp<A, NegativeOnly>{},
        tag,
        value,
        std::forward<ArithmeticOptions>(arithmetic_options)...);
  }
}

template <bool NegativeOnly>
template <FloatingTag Tag, typename... Options>
  requires (sizeof...(Options) > 0 &&
            details::valid_exp_options_for<Tag, Options...>())
VECOPS_ALWAYS_INLINE Vec<Tag> ExpCpo<NegativeOnly>::operator()(
    Tag tag, Vec<Tag> value, Options&&... options) const {
  return details::execute_exp_options<NegativeOnly>(
      tag, value, std::forward<Options>(options)...);
}

/** Full-domain exponential; defaults to opt::math::strict. */
inline constexpr ExpCpo<false> exp{};

/** Negative-only exponential; defaults to opt::math::strict. */
inline constexpr ExpCpo<true> exp_neg{};

/** Fixed Strict forwarding entry point for exp. */
inline constexpr FixedAccuracyExpCpo<Accuracy::Strict, false> exp_strict{};

/** Fixed Fast forwarding entry point for exp. */
inline constexpr FixedAccuracyExpCpo<Accuracy::Fast, false> exp_fast{};

/** Fixed Estimate forwarding entry point for exp. */
inline constexpr FixedAccuracyExpCpo<Accuracy::Estimate, false> exp_est{};

/** Fixed Strict forwarding entry point for exp_neg. */
inline constexpr FixedAccuracyExpCpo<Accuracy::Strict, true> exp_neg_strict{};

/** Fixed Fast forwarding entry point for exp_neg. */
inline constexpr FixedAccuracyExpCpo<Accuracy::Fast, true> exp_neg_fast{};

/** Fixed Estimate forwarding entry point for exp_neg. */
inline constexpr FixedAccuracyExpCpo<Accuracy::Estimate, true> exp_neg_est{};

} // namespace vecops::vec

#endif // VECOPS_VEC_MATH_H
