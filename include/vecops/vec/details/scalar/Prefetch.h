// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
#ifndef VECOPS_VEC_DETAILS_SCALAR_PREFETCH_H
#define VECOPS_VEC_DETAILS_SCALAR_PREFETCH_H

#include "vecops/vec/details/Dispatch.h"

namespace vecops::vec::details {

template <VectorTag Tag>
struct NativeImpl<ScalarBackend, PrefetchOp, Tag> {
  template <typename... Options>
  static VECOPS_ALWAYS_INLINE void call(
      PrefetchOp, Tag, const ElementOf<Tag>*, Options&&...) {
    // The scalar backend deliberately treats prefetch as a no-op.
  }
};

} // namespace vecops::vec::details

#endif // VECOPS_VEC_DETAILS_SCALAR_PREFETCH_H
