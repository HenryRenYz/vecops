#ifndef VECOPS_VEC_MATH_H
#define VECOPS_VEC_MATH_H

/**
 * @file Math.h
 * @brief The math module: exp, reciprocal, and logarithm operations with
 * compile-time accuracy options.
 *
 * `exp`, `exp2`, and `exp10` (plus their `*_neg` counterparts) are the
 * canonical exponential entry points.  With no math option they use `Accuracy::Strict`;
 * callers can select another implementation with `opt::math::strict`,
 * `opt::math::fast`, `opt::math::estimate`, or the generic
 * `opt::math::accuracy<A>` option.
 *
 * | Accuracy | Normal-input error contract |
 * |----------|-----------------------------|
 * | Strict   | ULP error <= 1              |
 * | Fast     | ULP error <= 4              |
 * | Estimate | ULP error <= 4 or relative error <= 0.006 |
 *
 * `exp_strict`, `exp_fast`, and `exp_est` (and the `exp2_*` / `exp10_*`
 * spellings) are convenience forwarding CPOs.  Their `*_neg_*` counterparts
 * select the same accuracy while assuming every active input lane is
 * `x <= 0`.  Accuracy options are deliberately rejected by these
 * fixed-accuracy CPOs; use the canonical CPO when forwarding an accuracy
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
 * accuracy option with exactly one `opt::masked(mask)`, `opt::first(count)`, or
 * `opt::unmasked` and the usual `opt::zero`/`opt::merge` inactive-lane policy,
 * in any order. `opt::first` is lowered to a backend mask. A call containing
 * only an accuracy option is an ordinary unmasked call.
 *
 * ***
 *
 * `rcp` and `rsqrt` are the reciprocal entry points of the same option
 * system: `rsqrt_strict` / `rsqrt_fast` / `rsqrt_est` and the `rcp_*`
 * spellings are the fixed-accuracy forwarding CPOs. Their per-tier
 * normal-input contracts, keyed to the estimate-plus-Newton ladder the
 * hardware offers (SVE FRSQRTE/FRECPE and FRSQRTS/FRECPS, x86
 * RCP14/RSQRT14 and RCPPS/RSQRTPS):
 *
 * | Accuracy | rcp / rsqrt contract (f32, f64)   | f16, bf16        |
 * |----------|-----------------------------------|------------------|
 * | Strict   | ULP error <= 1                    | ULP error <= 1   |
 * | Fast     | relative error <= 2^-15           | ULP error <= 2   |
 * | Estimate | relative error <= 2^-7            | relative error <= 2^-7 |
 *
 * The ladder mirrors common industry tiers: Estimate is the raw hardware
 * estimate (architecturally 2^-8 on SVE, tighter on x86), Fast adds one
 * Newton step (the classic NEON/SSE estimate-plus-refinement tier used for
 * game math and ML normalization kernels), and Strict refines to at most
 * 1 ULP — the SLEEF u10 class, tighter than CUDA's 2-ULP `rsqrtf`.
 *
 * Special inputs are tier-independent: rcp(+-0) = +-inf, rcp(+-inf) = +-0,
 * rsqrt(+-0) = +inf, rsqrt(x < 0) = NaN, and NaN propagates. When
 * `VECOPS_PRESERVE_SUBNORMALS` is defined, Strict produces gradual subnormal
 * results (rcp of huge inputs); Fast and Estimate always flush subnormal
 * outputs to zero. Accuracy options apply exactly as for the exponential
 * family, including combination with masking options.
 *
 * ***
 *
 * `log`, `log2`, and `log10` are the logarithm entry points of the same
 * option system, with the fixed-accuracy spellings `log_strict` / `log_fast`
 * / `log_est` (and the `log2_*` / `log10_*` forms). Logarithms have no
 * hardware estimate instruction to refine against, so the tiers ladder along
 * the two remaining cost axes — table lookups and polynomial degree — and
 * the top tier targets the accuracy class of Arm's own production SVE
 * libm routines (the optimized-routines implementations that ship in glibc
 * and ArmPL), not the 1-ULP reciprocal class:
 *
 * | Accuracy | f32 (log/log2/log10) | f64              | f16, bf16        |
 * |----------|----------------------|------------------|------------------|
 * | Strict   | ULP error <= 1       | ULP error <= 4   | ULP error <= 1   |
 * | Fast     | relative error <= 2^-13                | ULP error <= 2   |
 * | Estimate | relative error <= 2^-7                 | relative error <= 2^-7 |
 *
 * f64 Strict is the optimized-routines vector-libm class (their measured
 * maxima are 2.64/2.58/2.46 ULP for log/log2/log10; SLEEF's relaxed u35
 * tier is the same class), and pushing it to 1 ULP would need double-double
 * accumulation of the table terms. f32 Strict stays native: a degree-13
 * minimax on the [1/sqrt2, sqrt2) reduction window assembled Cody-Waite
 * style, with the k*log_b(2) term split so the product stays exact inside
 * the final FMA and every other rounding lands at or below the result
 * binade (measured 1 ULP on the SVE backend; the x86 backend evaluates
 * the same kernel natively). Fast is the table-free reduced-degree kernel
 * used by ML and statistics pipelines that only need ~1e-4 relative
 * accuracy; Estimate keeps three polynomial terms for the same 2^-7
 * budget as the reciprocal Estimate tier. On SVE, f16 Fast/Estimate run
 * native half-precision kernels while f16 Strict and bf16 widen through
 * the f32 pipeline, which rounds once into the narrow format; on x86
 * every f16 tier and bf16 widen through the f32 pipeline the same way.
 * The vector backends share the kernel constants (tables and polynomial
 * coefficients) from vec/details/Math.h, so all three bases run identical
 * instruction sequences with matching throughput; the scalar backend
 * alone evaluates the libm logarithm per lane.
 *
 * Special inputs are tier-independent: log(+-0) = -inf, log(+inf) = +inf,
 * log(x < 0) = NaN (including -inf), NaN propagates, and log(1) = +0
 * exactly. Logarithm outputs are never subnormal, so the tiers differ from
 * `VECOPS_PRESERVE_SUBNORMALS` builds only through their accuracy; subnormal
 * *inputs* are handled exactly in every tier. Accuracy options apply
 * exactly as for the exponential family, including combination with masking
 * options.
 */

#include <cstdint>

#include "vecops/vec/Arithmetic.h"
#include "vecops/vec/Bit.h"
#include "vecops/vec/Comparison.h"

namespace vecops::vec {

/** Exponential base distinguishing exp, exp2, and exp10 op tokens. */
enum class ExpBase : std::uint8_t {
  E,      //< e^x
  Base2,  //< 2^x
  Base10, //< 10^x
};

/** Logarithm base distinguishing log, log2, and log10 op tokens. */
enum class LogBase : std::uint8_t {
  E,      //< log_e
  Base2,  //< log_2
  Base10, //< log_10
};

namespace details {

/**
 * One related implementation token for every base/accuracy/domain
 * combination. The operator() entries below are the token-level call
 * surface used inside the details layer.
 */
template <ExpBase Base, Accuracy A, bool NegativeOnly>
struct ExpOp {
  template <FloatingTag Tag>
  VECOPS_ALWAYS_INLINE Vec<Tag> operator()(Tag tag, Vec<Tag> value) const {
    return details::execute(*this, tag, value);
  }

  /** Low-layer masked entry used by option dispatchers and backends. */
  template <FloatingTag Tag, typename Policy>
    requires (details::arithmetic_inactive_policy<Policy>)
  inline Vec<Tag> operator()(
      Tag tag, Vec<Tag> value, Mask<Tag> mask,
      Vec<Tag> inactive, Policy policy) const {
    return details::execute(*this, tag, value, mask, inactive, policy);
  }

  /**
   * Word-level entry for multi-word Tags; single-word Tags resolve to the
   * whole-Tag overload above via execute()'s word_count == 1 branch.
   */
  template <FloatingTag Tag>
    requires (details::multi_word_tag_v<Tag>)
  inline NativeWordVec<Tag> operator()(
      Tag tag, NativeWordVec<Tag> value) const {
    return details::execute_word<0, details::CurrentBackend>(*this, tag, value);
  }
};

/**
 * Token-level exp entry for internal use: every base/accuracy/domain
 * combination is spelled explicitly (exp<Base, A, NegativeOnly>(tag, value)).
 * Shadows the public vec::exp inside details, forcing internal callers to
 * name their accuracy tier instead of silently using the strict default.
 */
template <ExpBase Base, Accuracy A, bool NegativeOnly>
inline constexpr ExpOp<Base, A, NegativeOnly> exp{};

template <FloatingTag Tag, typename... Options>
consteval bool valid_exp_options_for() {
  constexpr std::size_t accuracy_count =
      option_count_v<IsMathAccuracyOption, Options...>;
  if constexpr (accuracy_count > 1) return false;
  if constexpr (!((is_math_accuracy_option_v<Options> ||
                    is_arithmetic_option_for_v<Tag, Options> ||
                    is_first_option_v<Options>) && ...))
    return false;

  constexpr std::size_t arithmetic_count =
      sizeof...(Options) - accuracy_count;
  if constexpr (arithmetic_count == 0) return true;

  constexpr std::size_t masked_count =
      option_count_v<IsMaskedOption, Options...>;
  constexpr std::size_t unmasked_count =
      option_count_v<IsUnmaskedOption, Options...>;
  constexpr std::size_t first_count =
      option_count_v<IsFirstOption, Options...>;
  constexpr std::size_t zero_count = option_count_v<IsZeroOption, Options...>;
  constexpr std::size_t vector_merge_count =
      option_count_v<IsVectorMergeOption, Options...>;
  constexpr std::size_t scalar_merge_count =
      option_count_v<IsScalarMergeOption, Options...>;
  return masked_count + unmasked_count + first_count == 1 &&
      masked_count * unmasked_count == 0 &&
      masked_count * first_count == 0 &&
      unmasked_count * first_count == 0 && zero_count <= 1 &&
      vector_merge_count + scalar_merge_count <= 1 &&
      zero_count + vector_merge_count + scalar_merge_count <= 1;
}

template <typename... Options>
consteval Accuracy selected_math_accuracy() {
  Accuracy result = Accuracy::Strict;
  ([&]() VECOPS_INLINE_LAMBDA {
    using Option = std::remove_cvref_t<Options>;
    if constexpr (is_math_accuracy_option_v<Option>)
      result = IsMathAccuracyOption<Option>::accuracy;
  }(), ...);
  return result;
}

template <typename... Options>
inline constexpr bool has_math_accuracy_option_v =
    option_count_v<IsMathAccuracyOption, Options...> != 0;

} // namespace details

/** Canonical exponential CPO, parameterized by base and valid input domain. */
template <ExpBase Base, bool NegativeOnly>
struct ExpCpo {
  template <FloatingTag Tag>
  VECOPS_ALWAYS_INLINE Vec<Tag> operator()(Tag tag, Vec<Tag> value) const;

  template <Accuracy A, FloatingTag Tag, typename... ArithmeticOptions>
    requires ((!details::is_math_accuracy_option_v<ArithmeticOptions>) && ... &&
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
    requires (details::valid_exp_options_for<VecToTag<V>, Options...>())
  VECOPS_ALWAYS_INLINE V operator()(V value, Options&&... options) const {
    return (*this)(
        VecToTag<V>{}, value, std::forward<Options>(options)...);
  }
};

/** Fixed-accuracy forwarding CPO used by exp_fast/exp_est/etc. */
template <ExpBase Base, Accuracy A, bool NegativeOnly>
struct FixedAccuracyExpCpo {
  template <FloatingTag Tag, typename... Options>
    requires (!details::has_math_accuracy_option_v<Options...> &&
              details::valid_exp_options_for<
                  Tag, Options..., decltype(opt::math::accuracy<A>)>())
  VECOPS_ALWAYS_INLINE Vec<Tag> operator()(
      Tag tag, Vec<Tag> value, Options&&... options) const {
    return ExpCpo<Base, NegativeOnly>{}(
        tag, value, opt::math::accuracy<A>,
        std::forward<Options>(options)...);
  }

  template <FloatingVectorValue V, typename... Options>
    requires (!details::has_math_accuracy_option_v<Options...> &&
              details::valid_exp_options_for<
                  VecToTag<V>, Options...,
                  decltype(opt::math::accuracy<A>)>())
  VECOPS_ALWAYS_INLINE V operator()(V value, Options&&... options) const {
    return (*this)(
        VecToTag<V>{}, value, std::forward<Options>(options)...);
  }
};

/* **************************************************************************** */
//    Reciprocal operations: rcp, rsqrt                                        //
/* **************************************************************************** */

namespace details {

/** One related implementation token for every reciprocal accuracy tier. */
template <Accuracy A>
struct RsqrtOp {
  template <FloatingTag Tag>
  VECOPS_ALWAYS_INLINE Vec<Tag> operator()(Tag tag, Vec<Tag> value) const {
    return details::execute(*this, tag, value);
  }

  /** Low-layer masked entry used by option dispatchers and backends. */
  template <FloatingTag Tag, typename Policy>
    requires (details::arithmetic_inactive_policy<Policy>)
  inline Vec<Tag> operator()(
      Tag tag, Vec<Tag> value, Mask<Tag> mask,
      Vec<Tag> inactive, Policy policy) const {
    return details::execute(*this, tag, value, mask, inactive, policy);
  }

  /**
   * Word-level entry for multi-word Tags; single-word Tags resolve to the
   * whole-Tag overload above via execute()'s word_count == 1 branch.
   */
  template <FloatingTag Tag>
    requires (details::multi_word_tag_v<Tag>)
  inline NativeWordVec<Tag> operator()(
      Tag tag, NativeWordVec<Tag> value) const {
    return details::execute_word<0, details::CurrentBackend>(*this, tag, value);
  }
};

template <Accuracy A>
struct RcpOp {
  template <FloatingTag Tag>
  VECOPS_ALWAYS_INLINE Vec<Tag> operator()(Tag tag, Vec<Tag> value) const {
    return details::execute(*this, tag, value);
  }

  /** Low-layer masked entry used by option dispatchers and backends. */
  template <FloatingTag Tag, typename Policy>
    requires (details::arithmetic_inactive_policy<Policy>)
  inline Vec<Tag> operator()(
      Tag tag, Vec<Tag> value, Mask<Tag> mask,
      Vec<Tag> inactive, Policy policy) const {
    return details::execute(*this, tag, value, mask, inactive, policy);
  }

  /**
   * Word-level entry for multi-word Tags; single-word Tags resolve to the
   * whole-Tag overload above via execute()'s word_count == 1 branch.
   */
  template <FloatingTag Tag>
    requires (details::multi_word_tag_v<Tag>)
  inline NativeWordVec<Tag> operator()(
      Tag tag, NativeWordVec<Tag> value) const {
    return details::execute_word<0, details::CurrentBackend>(*this, tag, value);
  }
};

/** Token-level entries for internal use; see details::exp above. */
template <Accuracy A>
inline constexpr RsqrtOp<A> rsqrt{};
template <Accuracy A>
inline constexpr RcpOp<A> rcp{};

/**
 * Option rules for rcp/rsqrt mirror the exp family: at most one accuracy
 * option, exactly one of masked/unmasked/first when masking at all, and at
 * most one zero/merge population option.
 */
template <FloatingTag Tag, typename... Options>
consteval bool valid_reciprocal_options_for() {
  return valid_exp_options_for<Tag, Options...>();
}

} // namespace details

/** Canonical unary-math CPO shared by rcp and rsqrt. */
template <template <Accuracy> class OpToken>
struct UnaryMathCpo {
  template <FloatingTag Tag>
  VECOPS_ALWAYS_INLINE Vec<Tag> operator()(Tag tag, Vec<Tag> value) const;

  template <Accuracy A, FloatingTag Tag, typename... ArithmeticOptions>
    requires ((!details::is_math_accuracy_option_v<ArithmeticOptions>) && ... &&
              details::valid_reciprocal_options_for<
                  Tag,
                  opt::math::AccuracyOption<A>,
                  ArithmeticOptions...>())
  VECOPS_ALWAYS_INLINE Vec<Tag> operator()(
      Tag tag,
      Vec<Tag> value,
      opt::math::AccuracyOption<A> accuracy,
      ArithmeticOptions&&... arithmetic_options) const;

  template <FloatingTag Tag, typename... Options>
    requires (sizeof...(Options) > 0 &&
              details::valid_reciprocal_options_for<Tag, Options...>())
  VECOPS_ALWAYS_INLINE Vec<Tag> operator()(
      Tag tag, Vec<Tag> value, Options&&... options) const;

  template <FloatingVectorValue V, typename... Options>
    requires (details::valid_reciprocal_options_for<VecToTag<V>, Options...>())
  VECOPS_ALWAYS_INLINE V operator()(V value, Options&&... options) const {
    return (*this)(
        VecToTag<V>{}, value, std::forward<Options>(options)...);
  }
};

/** Fixed-accuracy forwarding CPO used by rsqrt_fast/rcp_est/etc. */
template <template <Accuracy> class OpToken, Accuracy A>
struct FixedAccuracyUnaryMathCpo {
  template <FloatingTag Tag, typename... Options>
    requires (!details::has_math_accuracy_option_v<Options...> &&
              details::valid_reciprocal_options_for<
                  Tag, Options..., decltype(opt::math::accuracy<A>)>())
  VECOPS_ALWAYS_INLINE Vec<Tag> operator()(
      Tag tag, Vec<Tag> value, Options&&... options) const {
    return UnaryMathCpo<OpToken>{}(
        tag, value, opt::math::accuracy<A>,
        std::forward<Options>(options)...);
  }

  template <FloatingVectorValue V, typename... Options>
    requires (!details::has_math_accuracy_option_v<Options...> &&
              details::valid_reciprocal_options_for<
                  VecToTag<V>, Options...,
                  decltype(opt::math::accuracy<A>)>())
  VECOPS_ALWAYS_INLINE V operator()(V value, Options&&... options) const {
    return (*this)(
        VecToTag<V>{}, value, std::forward<Options>(options)...);
  }
};

/* **************************************************************************** */
//    Logarithm operations: log, log2, log10                                  //
/* **************************************************************************** */

namespace details {

/** One related implementation token for every log base/accuracy tier. */
template <LogBase Base, Accuracy A>
struct LogOp {
  template <FloatingTag Tag>
  VECOPS_ALWAYS_INLINE Vec<Tag> operator()(Tag tag, Vec<Tag> value) const {
    return details::execute(*this, tag, value);
  }

  /** Low-layer masked entry used by option dispatchers and backends. */
  template <FloatingTag Tag, typename Policy>
    requires (details::arithmetic_inactive_policy<Policy>)
  inline Vec<Tag> operator()(
      Tag tag, Vec<Tag> value, Mask<Tag> mask,
      Vec<Tag> inactive, Policy policy) const {
    return details::execute(*this, tag, value, mask, inactive, policy);
  }

  /**
   * Word-level entry for multi-word Tags; single-word Tags resolve to the
   * whole-Tag overload above via execute()'s word_count == 1 branch.
   */
  template <FloatingTag Tag>
    requires (details::multi_word_tag_v<Tag>)
  inline NativeWordVec<Tag> operator()(
      Tag tag, NativeWordVec<Tag> value) const {
    return details::execute_word<0, details::CurrentBackend>(*this, tag, value);
  }
};

/**
 * Token-level log entry for internal use:
 * log<Base, A>(tag, value). Shadows the public vec::log inside details; see
 * details::exp for the rationale.
 */
template <LogBase Base, Accuracy A>
inline constexpr LogOp<Base, A> log{};

/**
 * Option rules match the exp and reciprocal families: at most one accuracy
 * option, exactly one of masked/unmasked/first when masking at all, and at
 * most one zero/merge population option.
 */
template <FloatingTag Tag, typename... Options>
consteval bool valid_log_options_for() {
  return valid_exp_options_for<Tag, Options...>();
}

} // namespace details

/** Canonical logarithm CPO, parameterized by base. */
template <LogBase Base>
struct LogCpo {
  template <FloatingTag Tag>
  VECOPS_ALWAYS_INLINE Vec<Tag> operator()(Tag tag, Vec<Tag> value) const;

  template <Accuracy A, FloatingTag Tag, typename... ArithmeticOptions>
    requires ((!details::is_math_accuracy_option_v<ArithmeticOptions>) && ... &&
              details::valid_log_options_for<
                  Tag,
                  opt::math::AccuracyOption<A>,
                  ArithmeticOptions...>())
  VECOPS_ALWAYS_INLINE Vec<Tag> operator()(
      Tag tag,
      Vec<Tag> value,
      opt::math::AccuracyOption<A> accuracy,
      ArithmeticOptions&&... arithmetic_options) const;

  template <FloatingTag Tag, typename... Options>
    requires (sizeof...(Options) > 0 &&
              details::valid_log_options_for<Tag, Options...>())
  VECOPS_ALWAYS_INLINE Vec<Tag> operator()(
      Tag tag, Vec<Tag> value, Options&&... options) const;

  template <FloatingVectorValue V, typename... Options>
    requires (details::valid_log_options_for<VecToTag<V>, Options...>())
  VECOPS_ALWAYS_INLINE V operator()(V value, Options&&... options) const {
    return (*this)(
        VecToTag<V>{}, value, std::forward<Options>(options)...);
  }
};

/** Fixed-accuracy forwarding CPO used by log_fast/log10_est/etc. */
template <LogBase Base, Accuracy A>
struct FixedAccuracyLogCpo {
  template <FloatingTag Tag, typename... Options>
    requires (!details::has_math_accuracy_option_v<Options...> &&
              details::valid_log_options_for<
                  Tag, Options..., decltype(opt::math::accuracy<A>)>())
  VECOPS_ALWAYS_INLINE Vec<Tag> operator()(
      Tag tag, Vec<Tag> value, Options&&... options) const {
    return LogCpo<Base>{}(
        tag, value, opt::math::accuracy<A>,
        std::forward<Options>(options)...);
  }

  template <FloatingVectorValue V, typename... Options>
    requires (!details::has_math_accuracy_option_v<Options...> &&
              details::valid_log_options_for<
                  VecToTag<V>, Options...,
                  decltype(opt::math::accuracy<A>)>())
  VECOPS_ALWAYS_INLINE V operator()(V value, Options&&... options) const {
    return (*this)(
        VecToTag<V>{}, value, std::forward<Options>(options)...);
  }
};

/**
 * Public entry-point variables, declared before the backend includes so
 * that details-layer implementations can call them by short names (same
 * layout as Basic.h and Arithmetic.h). The CPO operator() definitions
 * stay after the backend includes.
 */
/** Full-domain exponential; defaults to opt::math::strict. */
inline constexpr ExpCpo<ExpBase::E, false> exp{};
/** Negative-only exponential; defaults to opt::math::strict. */
inline constexpr ExpCpo<ExpBase::E, true> exp_neg{};
/** Fixed Strict forwarding entry point for exp. */
inline constexpr FixedAccuracyExpCpo<ExpBase::E, Accuracy::Strict, false>
    exp_strict{};
/** Fixed Fast forwarding entry point for exp. */
inline constexpr FixedAccuracyExpCpo<ExpBase::E, Accuracy::Fast, false>
    exp_fast{};
/** Fixed Estimate forwarding entry point for exp. */
inline constexpr FixedAccuracyExpCpo<ExpBase::E, Accuracy::Estimate, false>
    exp_est{};
/** Fixed Strict forwarding entry point for exp_neg. */
inline constexpr FixedAccuracyExpCpo<ExpBase::E, Accuracy::Strict, true>
    exp_neg_strict{};
/** Fixed Fast forwarding entry point for exp_neg. */
inline constexpr FixedAccuracyExpCpo<ExpBase::E, Accuracy::Fast, true>
    exp_neg_fast{};
/** Fixed Estimate forwarding entry point for exp_neg. */
inline constexpr FixedAccuracyExpCpo<ExpBase::E, Accuracy::Estimate, true>
    exp_neg_est{};
/** Full-domain base-2 exponential; defaults to opt::math::strict. */
inline constexpr ExpCpo<ExpBase::Base2, false> exp2{};
/** Negative-only base-2 exponential; defaults to opt::math::strict. */
inline constexpr ExpCpo<ExpBase::Base2, true> exp2_neg{};
/** Fixed Strict forwarding entry point for exp2. */
inline constexpr FixedAccuracyExpCpo<ExpBase::Base2, Accuracy::Strict, false>
    exp2_strict{};
/** Fixed Fast forwarding entry point for exp2. */
inline constexpr FixedAccuracyExpCpo<ExpBase::Base2, Accuracy::Fast, false>
    exp2_fast{};
/** Fixed Estimate forwarding entry point for exp2. */
inline constexpr FixedAccuracyExpCpo<ExpBase::Base2, Accuracy::Estimate, false>
    exp2_est{};
/** Fixed Strict forwarding entry point for exp2_neg. */
inline constexpr FixedAccuracyExpCpo<ExpBase::Base2, Accuracy::Strict, true>
    exp2_neg_strict{};
/** Fixed Fast forwarding entry point for exp2_neg. */
inline constexpr FixedAccuracyExpCpo<ExpBase::Base2, Accuracy::Fast, true>
    exp2_neg_fast{};
/** Fixed Estimate forwarding entry point for exp2_neg. */
inline constexpr FixedAccuracyExpCpo<ExpBase::Base2, Accuracy::Estimate, true>
    exp2_neg_est{};
/** Full-domain base-10 exponential; defaults to opt::math::strict. */
inline constexpr ExpCpo<ExpBase::Base10, false> exp10{};
/** Negative-only base-10 exponential; defaults to opt::math::strict. */
inline constexpr ExpCpo<ExpBase::Base10, true> exp10_neg{};
/** Fixed Strict forwarding entry point for exp10. */
inline constexpr FixedAccuracyExpCpo<ExpBase::Base10, Accuracy::Strict, false>
    exp10_strict{};
/** Fixed Fast forwarding entry point for exp10. */
inline constexpr FixedAccuracyExpCpo<ExpBase::Base10, Accuracy::Fast, false>
    exp10_fast{};
/** Fixed Estimate forwarding entry point for exp10. */
inline constexpr FixedAccuracyExpCpo<ExpBase::Base10, Accuracy::Estimate, false>
    exp10_est{};
/** Fixed Strict forwarding entry point for exp10_neg. */
inline constexpr FixedAccuracyExpCpo<ExpBase::Base10, Accuracy::Strict, true>
    exp10_neg_strict{};
/** Fixed Fast forwarding entry point for exp10_neg. */
inline constexpr FixedAccuracyExpCpo<ExpBase::Base10, Accuracy::Fast, true>
    exp10_neg_fast{};
/** Fixed Estimate forwarding entry point for exp10_neg. */
inline constexpr FixedAccuracyExpCpo<ExpBase::Base10, Accuracy::Estimate, true>
    exp10_neg_est{};
/**
 * Computes the reciprocal square root 1/sqrt(x). With no math option the
 * call uses Accuracy::Strict; see the file header for the per-tier error
 * contracts. rsqrt(+-0) = +inf, rsqrt(x < 0) = NaN, rsqrt(+inf) = +0, NaN
 * propagates. Special inputs produce identical results in every tier.
 *
 * @see rcp for the plain reciprocal.
 * @see sqrt for the correctly-rounded square root.
 */
inline constexpr UnaryMathCpo<details::RsqrtOp> rsqrt{};
/** Fixed Strict forwarding entry point for rsqrt. */
inline constexpr FixedAccuracyUnaryMathCpo<details::RsqrtOp, Accuracy::Strict>
    rsqrt_strict{};
/** Fixed Fast forwarding entry point for rsqrt. */
inline constexpr FixedAccuracyUnaryMathCpo<details::RsqrtOp, Accuracy::Fast>
    rsqrt_fast{};
/** Fixed Estimate forwarding entry point for rsqrt. */
inline constexpr FixedAccuracyUnaryMathCpo<details::RsqrtOp, Accuracy::Estimate>
    rsqrt_est{};
/**
 * Computes the reciprocal 1/x. With no math option the call uses
 * Accuracy::Strict; see the file header for the per-tier error contracts.
 * rcp(+-0) = +-inf, rcp(+-inf) = +-0 with the sign of the input, NaN
 * propagates. Special inputs produce identical results in every tier.
 *
 * @see rsqrt for the reciprocal square root.
 * @see div for full-precision division.
 */
inline constexpr UnaryMathCpo<details::RcpOp> rcp{};
/** Fixed Strict forwarding entry point for rcp. */
inline constexpr FixedAccuracyUnaryMathCpo<details::RcpOp, Accuracy::Strict>
    rcp_strict{};
/** Fixed Fast forwarding entry point for rcp. */
inline constexpr FixedAccuracyUnaryMathCpo<details::RcpOp, Accuracy::Fast>
    rcp_fast{};
/** Fixed Estimate forwarding entry point for rcp. */
inline constexpr FixedAccuracyUnaryMathCpo<details::RcpOp, Accuracy::Estimate>
    rcp_est{};
/**
 * Computes the natural logarithm log_e(x). With no math option the call uses
 * Accuracy::Strict; see the file header for the per-tier error contracts.
 * log(+-0) = -inf, log(+inf) = +inf, log(x < 0) = NaN, log(1) = +0, and NaN
 * propagates. Special inputs produce identical results in every tier.
 *
 * @see log2, log10 for other bases.
 * @see exp for the inverse operation.
 */
inline constexpr LogCpo<LogBase::E> log{};
/** Fixed Strict forwarding entry point for log. */
inline constexpr FixedAccuracyLogCpo<LogBase::E, Accuracy::Strict>
    log_strict{};
/** Fixed Fast forwarding entry point for log. */
inline constexpr FixedAccuracyLogCpo<LogBase::E, Accuracy::Fast> log_fast{};
/** Fixed Estimate forwarding entry point for log. */
inline constexpr FixedAccuracyLogCpo<LogBase::E, Accuracy::Estimate>
    log_est{};
/**
 * Computes the base-2 logarithm log_2(x). With no math option the call uses
 * Accuracy::Strict; see the file header for the per-tier error contracts.
 * Special inputs match the log contract, with log2(2^k) exactly k.
 *
 * @see log, log10 for other bases.
 * @see exp2 for the inverse operation.
 */
inline constexpr LogCpo<LogBase::Base2> log2{};
/** Fixed Strict forwarding entry point for log2. */
inline constexpr FixedAccuracyLogCpo<LogBase::Base2, Accuracy::Strict>
    log2_strict{};
/** Fixed Fast forwarding entry point for log2. */
inline constexpr FixedAccuracyLogCpo<LogBase::Base2, Accuracy::Fast>
    log2_fast{};
/** Fixed Estimate forwarding entry point for log2. */
inline constexpr FixedAccuracyLogCpo<LogBase::Base2, Accuracy::Estimate>
    log2_est{};
/**
 * Computes the base-10 logarithm log_10(x). With no math option the call
 * uses Accuracy::Strict; see the file header for the per-tier error
 * contracts. Special inputs match the log contract.
 *
 * @see log, log2 for other bases.
 * @see exp10 for the inverse operation.
 */
inline constexpr LogCpo<LogBase::Base10> log10{};
/** Fixed Strict forwarding entry point for log10. */
inline constexpr FixedAccuracyLogCpo<LogBase::Base10, Accuracy::Strict>
    log10_strict{};
/** Fixed Fast forwarding entry point for log10. */
inline constexpr FixedAccuracyLogCpo<LogBase::Base10, Accuracy::Fast>
    log10_fast{};
/** Fixed Estimate forwarding entry point for log10. */
inline constexpr FixedAccuracyLogCpo<LogBase::Base10, Accuracy::Estimate>
    log10_est{};

} // namespace vecops::vec

#include "vecops/vec/details/Dispatch.h"
#include "vecops/vec/details/Math.h"
#include "vecops/vec/details/scalar/Math.h"

#if defined(ARCH_X86_FAMILY) && !defined(CPU_CAPABILITY_GENERIC)
#include "vecops/vec/details/x86/Math.h"
#elif defined(CPU_CAPABILITY_SVE)
#include "vecops/vec/details/sve/Math.h"
#endif

namespace vecops::vec {

template <ExpBase Base, bool NegativeOnly>
template <FloatingTag Tag>
VECOPS_ALWAYS_INLINE Vec<Tag> ExpCpo<Base, NegativeOnly>::operator()(
    Tag tag, Vec<Tag> value) const {
  return details::execute(
      details::ExpOp<Base, Accuracy::Strict, NegativeOnly>{}, tag, value);
}

template <ExpBase Base, bool NegativeOnly>
template <Accuracy A, FloatingTag Tag, typename... ArithmeticOptions>
  requires ((!details::is_math_accuracy_option_v<ArithmeticOptions>) && ... &&
            details::valid_exp_options_for<
                Tag,
                opt::math::AccuracyOption<A>,
                ArithmeticOptions...>())
VECOPS_ALWAYS_INLINE Vec<Tag> ExpCpo<Base, NegativeOnly>::operator()(
    Tag tag,
    Vec<Tag> value,
    opt::math::AccuracyOption<A> accuracy,
    ArithmeticOptions&&... arithmetic_options) const {
  (void)accuracy;
  if constexpr (sizeof...(ArithmeticOptions) == 0) {
    return details::execute(
        details::ExpOp<Base, A, NegativeOnly>{}, tag, value);
  } else if constexpr (
      details::option_count_v<
          details::IsFirstOption, ArithmeticOptions...> == 1) {
    const nint_t count = details::find_option<details::IsFirstOption>(
        arithmetic_options...).count;
    const auto mask = mwhilelt(tag, 0, count);
    return details::invoke_replacing_first(
        [&](auto&&... lowered_options) -> Vec<Tag> {
          return details::execute_unary_arithmetic_options(
              details::ExpOp<Base, A, NegativeOnly>{}, tag, value,
              std::forward<decltype(lowered_options)>(lowered_options)...);
        },
        mask, std::forward<ArithmeticOptions>(arithmetic_options)...);
  } else {
    return details::execute_unary_arithmetic_options(
        details::ExpOp<Base, A, NegativeOnly>{}, tag, value,
        std::forward<ArithmeticOptions>(arithmetic_options)...);
  }
}

template <ExpBase Base, bool NegativeOnly>
template <FloatingTag Tag, typename... Options>
  requires (sizeof...(Options) > 0 &&
            details::valid_exp_options_for<Tag, Options...>())
VECOPS_ALWAYS_INLINE Vec<Tag> ExpCpo<Base, NegativeOnly>::operator()(
    Tag tag, Vec<Tag> value, Options&&... options) const {
  return details::execute_exp_options<Base, NegativeOnly>(
      tag, value, std::forward<Options>(options)...);
}

























/* **************************************************************************** */
//    rcp / rsqrt definitions                                                  //
/* **************************************************************************** */

template <template <Accuracy> class OpToken>
template <FloatingTag Tag>
VECOPS_ALWAYS_INLINE Vec<Tag> UnaryMathCpo<OpToken>::operator()(
    Tag tag, Vec<Tag> value) const {
  return details::execute(OpToken<Accuracy::Strict>{}, tag, value);
}

template <template <Accuracy> class OpToken>
template <Accuracy A, FloatingTag Tag, typename... ArithmeticOptions>
  requires ((!details::is_math_accuracy_option_v<ArithmeticOptions>) && ... &&
            details::valid_reciprocal_options_for<
                Tag,
                opt::math::AccuracyOption<A>,
                ArithmeticOptions...>())
VECOPS_ALWAYS_INLINE Vec<Tag> UnaryMathCpo<OpToken>::operator()(
    Tag tag,
    Vec<Tag> value,
    opt::math::AccuracyOption<A> accuracy,
    ArithmeticOptions&&... arithmetic_options) const {
  // Re-read the tier from the option type instead of relying on the
  // deduced A through deep inline chains.
  constexpr Accuracy tier =
      details::IsMathAccuracyOption<opt::math::AccuracyOption<A>>::accuracy;
  if constexpr (sizeof...(ArithmeticOptions) == 0) {
    return details::execute(OpToken<tier>{}, tag, value);
  } else if constexpr (
      details::option_count_v<
          details::IsFirstOption, ArithmeticOptions...> == 1) {
    // opt::first is the only masking option in a valid pack; lower it
    // explicitly (details::execute_unary_math_options has the spelled-out
    // population branches).
    return details::execute_unary_math_options<OpToken>(
        tag, value, opt::math::accuracy<tier>,
        std::forward<ArithmeticOptions>(arithmetic_options)...);
  } else {
    return details::execute_unary_arithmetic_options(
        OpToken<tier>{}, tag, value,
        std::forward<ArithmeticOptions>(arithmetic_options)...);
  }
}

template <template <Accuracy> class OpToken>
template <FloatingTag Tag, typename... Options>
  requires (sizeof...(Options) > 0 &&
            details::valid_reciprocal_options_for<Tag, Options...>())
VECOPS_ALWAYS_INLINE Vec<Tag> UnaryMathCpo<OpToken>::operator()(
    Tag tag, Vec<Tag> value, Options&&... options) const {
  return details::execute_unary_math_options<OpToken>(
      tag, value, std::forward<Options>(options)...);
}









/* **************************************************************************** */
//    log / log2 / log10 definitions                                           //
/* **************************************************************************** */

template <LogBase Base>
template <FloatingTag Tag>
VECOPS_ALWAYS_INLINE Vec<Tag> LogCpo<Base>::operator()(
    Tag tag, Vec<Tag> value) const {
  return details::execute(details::LogOp<Base, Accuracy::Strict>{}, tag,
                          value);
}

template <LogBase Base>
template <Accuracy A, FloatingTag Tag, typename... ArithmeticOptions>
  requires ((!details::is_math_accuracy_option_v<ArithmeticOptions>) && ... &&
            details::valid_log_options_for<
                Tag,
                opt::math::AccuracyOption<A>,
                ArithmeticOptions...>())
VECOPS_ALWAYS_INLINE Vec<Tag> LogCpo<Base>::operator()(
    Tag tag,
    Vec<Tag> value,
    opt::math::AccuracyOption<A> accuracy,
    ArithmeticOptions&&... arithmetic_options) const {
  // Re-read the tier from the option type instead of relying on the deduced
  // A through deep inline chains (same hardening as the reciprocal CPO).
  constexpr Accuracy tier =
      details::IsMathAccuracyOption<opt::math::AccuracyOption<A>>::accuracy;
  if constexpr (sizeof...(ArithmeticOptions) == 0) {
    return details::execute(details::LogOp<Base, tier>{}, tag, value);
  } else if constexpr (
      details::option_count_v<
          details::IsFirstOption, ArithmeticOptions...> == 1) {
    // opt::first is the only masking option in a valid pack; lower it
    // explicitly (details::execute_log_options has the spelled-out
    // population branches).
    return details::execute_log_options<Base>(
        tag, value, opt::math::accuracy<tier>,
        std::forward<ArithmeticOptions>(arithmetic_options)...);
  } else {
    return details::execute_unary_arithmetic_options(
        details::LogOp<Base, tier>{}, tag, value,
        std::forward<ArithmeticOptions>(arithmetic_options)...);
  }
}

template <LogBase Base>
template <FloatingTag Tag, typename... Options>
  requires (sizeof...(Options) > 0 &&
            details::valid_log_options_for<Tag, Options...>())
VECOPS_ALWAYS_INLINE Vec<Tag> LogCpo<Base>::operator()(
    Tag tag, Vec<Tag> value, Options&&... options) const {
  return details::execute_log_options<Base>(
      tag, value, std::forward<Options>(options)...);
}













} // namespace vecops::vec

#endif // VECOPS_VEC_MATH_H
