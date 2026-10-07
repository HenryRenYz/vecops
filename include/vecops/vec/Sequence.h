// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
#ifndef VECOPS_VEC_SEQUENCE_H
#define VECOPS_VEC_SEQUENCE_H

/**
 * @file Sequence.h
 * @brief Arithmetic lane-sequence generation.
 */

#include "vecops/vec/Arithmetic.h"

namespace vecops::vec {

struct IotaOp {
  template <VectorTag Tag>
  VECOPS_ALWAYS_INLINE Vec<Tag> operator()(
      Tag tag, ElementOf<Tag> start) const;

  template <VectorTag Tag>
  VECOPS_ALWAYS_INLINE Vec<Tag> operator()(
      Tag tag, ElementOf<Tag> start, ElementOf<Tag> step) const;
};

inline constexpr IotaOp iota{};

} // namespace vecops::vec

#include "vecops/vec/details/Dispatch.h"
#include "vecops/vec/details/Sequence.h"
#include "vecops/vec/details/scalar/Sequence.h"

#if defined(ARCH_X86_FAMILY) && !defined(CPU_CAPABILITY_GENERIC)
#include "vecops/vec/details/x86/Sequence.h"
#elif defined(CPU_CAPABILITY_SVE)
#include "vecops/vec/details/sve/Sequence.h"
#endif

namespace vecops::vec {

/**
 * Returns `result[i] = start + i` for every logical lane. Integer arithmetic
 * wraps modulo the element width. Floating arithmetic is evaluated in the
 * element's normal compute format and rounded back to that element type.
 */
template <VectorTag Tag>
VECOPS_ALWAYS_INLINE Vec<Tag> IotaOp::operator()(
    Tag tag, ElementOf<Tag> start) const {
  return details::execute(*this, tag, start);
}

/**
 * Returns `result[i] = start + i * step` for every logical lane. The lane
 * ordinal is global across all physical words of a multi-word Tag.
 */
template <VectorTag Tag>
VECOPS_ALWAYS_INLINE Vec<Tag> IotaOp::operator()(
    Tag tag, ElementOf<Tag> start, ElementOf<Tag> step) const {
  return details::execute(*this, tag, start, step);
}

} // namespace vecops::vec

#endif // VECOPS_VEC_SEQUENCE_H
