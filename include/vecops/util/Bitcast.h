//
// Created by HenryRenYz on 2026/6/7.
//

#ifndef VECOPS_BITCAST_H
#define VECOPS_BITCAST_H

#include "vecops/CoreDefs.h"

namespace vecops {

template <typename TOut, typename TIn>
VECOPS_INLINE constexpr TOut bitcast(TIn v) {
  union { TIn in; TOut out; } u { .in = v };
  return u.out;
}

} // namespace vecops

#endif //VECOPS_BITCAST_H
