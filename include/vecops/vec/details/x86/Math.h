// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
#ifndef VECOPS_VEC_DETAILS_X86_MATH_H
#define VECOPS_VEC_DETAILS_X86_MATH_H

/**
 * @file Math.h
 * @brief x86 backend math hub.
 *
 * Backends split per-family implementations into sibling files under
 * vec/details/x86/math/; this header only aggregates them so the public
 * include chain stays a single entry point per backend.
 */

#include "vecops/vec/details/x86/math/Exp.h"
#include "vecops/vec/details/x86/math/Reciprocal.h"
#include "vecops/vec/details/x86/math/Log.h"
#include "vecops/vec/details/x86/math/Trig.h"

#endif // VECOPS_VEC_DETAILS_X86_MATH_H
