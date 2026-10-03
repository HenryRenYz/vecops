#ifndef VECOPS_VEC_DETAILS_SVE_MATH_POLICY_H
#define VECOPS_VEC_DETAILS_SVE_MATH_POLICY_H

#include "vecops/CoreDefs.h"

/**
 * @file Policy.h
 * @brief Placement policy for vector-only SVE math exceptional paths.
 *
 * Define VECOPS_SVE_INLINE_MATH_SPECIALS=1 before any vecops header in a
 * translation unit whose vector math can execute inside a manual SME owner.
 * Every exceptional path must then remain inline: an ordinary SVE PCS call
 * may invalidate streaming vector/ZA state. Ordinary SVE builds keep these
 * cold paths outlined to control code size. This changes placement only,
 * not accuracy, saturation, NaN or subnormal behavior. Use one consistent
 * setting across translation units sharing inline/template definitions.
 * Scalar libm fallbacks (e.g. large-argument trig) are not made SME-safe by
 * this policy; manual owners must still pass the no-call disassembly gate.
 */
#if defined(VECOPS_SVE_INLINE_MATH_SPECIALS) && VECOPS_SVE_INLINE_MATH_SPECIALS
#define VECOPS_SVE_MATH_SPECIAL VECOPS_ALWAYS_INLINE
#else
#define VECOPS_SVE_MATH_SPECIAL VECOPS_NOINLINE inline
#endif

#endif // VECOPS_VEC_DETAILS_SVE_MATH_POLICY_H
