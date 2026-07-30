#ifndef VECOPS_VEC_MATH_H
#define VECOPS_VEC_MATH_H

/**
 * @file Math.h
 * @brief Exponential operations with multiple precision tiers.
 *
 * Six callable CPOs implement `exp` across three precision tiers and two
 * input-domain variants:
 *
 * | CPO         | Tier     | Input domain |
 * |-------------|----------|-------------|
 * | exp         | Strict   | full range   |
 * | exp_fast    | Fast     | full range   |
 * | exp_est     | Estimate | full range   |
 * | exp_neg     | Strict   | `x <= 0`     |
 * | exp_neg_fast| Fast     | `x <= 0`     |
 * | exp_neg_est | Estimate | `x <= 0`     |
 *
 * @section precision Precision contracts
 *
 * The three tiers define the maximum allowed error for normal (non-overflow,
 * non-subnormal) inputs.  All contracts are verified against a high-precision
 * reference (std::exp in double precision) during testing:
 *
 * - **Strict** (exp, exp_neg): ULP error ≤ 1.
 *   On SVE the scalar-style approximation (FEXPA with polynomial residual)
 *   achieves ~1 ULP for f32/f64/f16/bf16.
 *
 * - **Fast** (exp_fast, exp_neg_fast): ULP error ≤ 4.
 *   On SVE f16 this uses native fp16 FEXPA with a split-constant range
 *   reduction; other types use the same polynomial path as Strict with a
 *   reduced-quality residual or the same path where Strict already meets
 *   the Fast contract.
 *
 * - **Estimate** (exp_est, exp_neg_est): ULP error ≤ 4 **or** relative
 *   error ≤ 0.006 (whichever is looser).
 *   This is the fastest tier.  On x86 and SVE it uses the ISA-native
 *   estimate instruction (e.g. SVE FEXPA) with a minimal linear correction.
 *   The mixed ULP/relative criterion keeps the error bound meaningful
 *   across all element types: 0.6% relative error is the target for
 *   neural-network exponent usage, while 4 ULP serves as the output
 *   quantisation floor.
 *
 * @section neg Variants (exp_neg*)
 *
 * The `_neg` family assumes every active input lane satisfies `x <= 0`.
 * This is the typical softmax/layer-norm pattern where the input has
 * already been shifted by `x - row_max`.  These variants skip the
 * overflow-test branch and are measurably faster than the general entry
 * points on the same tier.
 *
 * Active lanes with `x > 0` produce **unspecified** results — the
 * behaviour is backend-dependent and may return infinity, NaN, or an
 * arbitrary finite value.  The `VECOPS_MATH_ASSUME_VALID_INPUTS` macro
 * additionally skips NaN propagation checks for the negative-only path.
 *
 * @section subnormals Subnormal handling
 *
 * When `VECOPS_PRESERVE_SUBNORMALS` is defined, the Strict tier preserves
 * subnormal (gradual underflow) results.  Without it (the default), and
 * on all Fast/Estimate tiers, subnormal outputs are flushed to zero.
 * On SVE bf16 the Strict tier additionally falls back to a higher-precision
 * path under this macro to avoid double-rounding artifacts.
 *
 * @section options Filtered calls
 *
 * Filtered calls accept exactly one `opt::masked(mask)` option, plus the
 * standard inactive-lane population policy (`opt::zero`, `opt::merge`).
 * The masking and population behaviour follows the same pattern as the
 * unary arithmetic operations (see Arithmetic.h).
 *
 * @see rcp, rsqrt for reciprocal and reciprocal-square-root estimate operations.
 * @see Arithmetic.h for the common masked-option pattern used by exp.
 */

#include "vecops/vec/Arithmetic.h"
#include "vecops/vec/Bit.h"
#include "vecops/vec/Comparison.h"

namespace vecops::vec {

namespace details {
/**
 * Precision tier selector used internally to forward the tier from
 * operation type (e.g. ExpFastOp → Fast) to the backend implementation.
 */
enum class ExpTier { Strict, Fast, Estimate };
}

#define VECOPS_VEC_DECLARE_EXP_OP(OpType)                              \
  struct OpType {                                                       \
    template <FloatingTag Tag>                                         \
    VECOPS_ALWAYS_INLINE Vec<Tag> operator()(                           \
        Tag tag, Vec<Tag> value) const;                                 \
    template <FloatingTag Tag, typename... Options>                     \
      requires (sizeof...(Options) > 0)                                \
    VECOPS_ALWAYS_INLINE Vec<Tag> operator()(                           \
        Tag tag, Vec<Tag> value, Options&&... options) const;           \
    template <FloatingVectorValue V, typename... Options>               \
    VECOPS_ALWAYS_INLINE V operator()(                                  \
        V value, Options&&... options) const {                          \
      return (*this)(                                                   \
          VecToTagT<V>{}, value, std::forward<Options>(options)...);    \
    }                                                                   \
  }

VECOPS_VEC_DECLARE_EXP_OP(ExpOp);
VECOPS_VEC_DECLARE_EXP_OP(ExpFastOp);
VECOPS_VEC_DECLARE_EXP_OP(ExpEstOp);
VECOPS_VEC_DECLARE_EXP_OP(ExpNegOp);
VECOPS_VEC_DECLARE_EXP_OP(ExpNegFastOp);
VECOPS_VEC_DECLARE_EXP_OP(ExpNegEstOp);

#undef VECOPS_VEC_DECLARE_EXP_OP

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

/* **************************************************************************** */
//    Exponential: exp, exp_fast, exp_est and negative-only variants     //
/* **************************************************************************** */

#define VECOPS_VEC_DEFINE_EXP_OP(OpType, Name)                         \
  template <FloatingTag Tag>                                          \
  VECOPS_ALWAYS_INLINE Vec<Tag> OpType::operator()(                    \
      Tag tag, Vec<Tag> value) const {                                 \
    return details::execute(*this, tag, value);                        \
  }                                                                    \
  template <FloatingTag Tag, typename... Options>                      \
    requires (sizeof...(Options) > 0)                                 \
  VECOPS_ALWAYS_INLINE Vec<Tag> OpType::operator()(                    \
      Tag tag, Vec<Tag> value, Options&&... options) const {           \
    return details::execute_unary_arithmetic_options(                  \
        *this, tag, value, std::forward<Options>(options)...);         \
  }                                                                    \
  inline constexpr OpType Name{}

/**
 * Strict exponential.  ULP error ≤ 1 for all normal inputs.
 * Filtered calls require exactly one opt::masked, plus the usual
 * inactive-lane population options.
 *
 * @see exp_fast, exp_est for the faster-but-less-accurate tiers.
 * @see exp_neg for the negative-only Strict variant.
 */
VECOPS_VEC_DEFINE_EXP_OP(ExpOp, exp);

/**
 * Fast exponential.  ULP error ≤ 4 for all normal inputs.
 * On SVE this uses the same polynomial path as Strict where it already
 * meets the Fast contract; on f16 it uses native fp16 FEXPA with a
 * split-constant range reduction.
 *
 * @see exp, exp_est for the other precision tiers.
 * @see exp_neg_fast for the negative-only Fast variant.
 */
VECOPS_VEC_DEFINE_EXP_OP(ExpFastOp, exp_fast);

/**
 * Estimate exponential.  ULP error ≤ 4 OR relative error ≤ 0.006
 * (whichever is looser).  Uses ISA-native estimate instructions (FEXPA
 * on SVE, approximate RCP-based path on x86) with a minimal correction.
 *
 * @see exp, exp_fast for the higher-quality tiers.
 * @see exp_neg_est for the negative-only Estimate variant.
 */
VECOPS_VEC_DEFINE_EXP_OP(ExpEstOp, exp_est);

/**
 * Strict exponential for inputs known to be `x <= 0`.
 * Same ULP ≤ 1 precision as exp but skips the overflow check path.
 * Active lanes with `x > 0` produce unspecified results.
 *
 * @see exp for the general Strict variant.
 * @see exp_neg_fast, exp_neg_est for the negative-only Fast/Estimate tiers.
 */
VECOPS_VEC_DEFINE_EXP_OP(ExpNegOp, exp_neg);

/**
 * Fast exponential for inputs known to be `x <= 0`.
 * Same ULP ≤ 4 precision as exp_fast but skips the overflow check path.
 * Active lanes with `x > 0` produce unspecified results.
 *
 * @see exp_neg for the negative-only Strict variant.
 * @see exp_neg_est for the negative-only Estimate variant.
 */
VECOPS_VEC_DEFINE_EXP_OP(ExpNegFastOp, exp_neg_fast);

/**
 * Estimate exponential for inputs known to be `x <= 0`.
 * Same ULP ≤ 4 / relative ≤ 0.006 precision as exp_est but skips the
 * overflow check path.  Active lanes with `x > 0` produce unspecified
 * results.
 *
 * @see exp_neg_fast for the negative-only Fast variant.
 * @see exp_neg for the negative-only Strict variant.
 */
VECOPS_VEC_DEFINE_EXP_OP(ExpNegEstOp, exp_neg_est);

#undef VECOPS_VEC_DEFINE_EXP_OP

} // namespace vecops::vec

#endif // VECOPS_VEC_MATH_H
