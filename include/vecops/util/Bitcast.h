// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT

#ifndef VECOPS_BITCAST_H
#define VECOPS_BITCAST_H

#include <bit>

#include "vecops/CoreDefs.h"

namespace vecops {

#if defined(__cpp_lib_bit_cast)

template <typename TOut, typename TIn>
VECOPS_INLINE constexpr TOut bitcast(TIn v) {
  return std::bit_cast<TOut>(v);
}

#else
// BiSheng 5.1 ships a standard library without std::bit_cast. The union
// form is the fallback; every use involves same-size trivially-copyable
// scalar pairs where it is the conventional implementation.
template <typename TOut, typename TIn>
VECOPS_INLINE constexpr TOut bitcast(TIn v) {
  union { TIn in; TOut out; } u { .in = v };
  return u.out;
}
#endif

} // namespace vecops

#endif //VECOPS_BITCAST_H
