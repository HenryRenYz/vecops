#ifndef VECOPS_VEC_DETAILS_SCALAR_MATH_H
#define VECOPS_VEC_DETAILS_SCALAR_MATH_H

/**
 * @file Math.h
 * @brief Scalar backend math hub.
 *
 * Backends split per-family implementations into sibling files under
 * vec/details/scalar/math/; this header only aggregates them so the public
 * include chain stays a single entry point per backend.
 */

#include "vecops/vec/details/scalar/math/Exp.h"
#include "vecops/vec/details/scalar/math/Reciprocal.h"

#endif // VECOPS_VEC_DETAILS_SCALAR_MATH_H
