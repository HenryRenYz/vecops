// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
#ifndef VECOPS_VEC_DETAILS_SVE_MATH_H
#define VECOPS_VEC_DETAILS_SVE_MATH_H

/**
 * @file Math.h
 * @brief SVE backend math hub.
 *
 * Backends split per-family implementations into sibling files under
 * vec/details/sve/math/; this header only aggregates them so the public
 * include chain stays a single entry point per backend.
 */

#include "vecops/vec/details/sve/math/Exp.h"
#include "vecops/vec/details/sve/math/Reciprocal.h"
#include "vecops/vec/details/sve/math/Log.h"
#include "vecops/vec/details/sve/math/Trig.h"

#endif // VECOPS_VEC_DETAILS_SVE_MATH_H
