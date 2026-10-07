// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
#ifndef VECOPS_VEC_DETAILS_SVE_PREFETCH_H
#define VECOPS_VEC_DETAILS_SVE_PREFETCH_H

/**
 * @file Prefetch.h
 * @brief SVE backend implementations for prefetch operations.
 */

#if !defined(HAS_SVE)
#error "This header requires an SVE target"
#endif

#include <type_traits>

#include "vecops/vec/details/Dispatch.h"

namespace vecops::vec::details {


/* **************************************************************************** */
//                       Prefetch word implementations                        //
/* **************************************************************************** */

template <VectorTag Tag>
struct NativeImpl<SVEBackend, PrefetchOp, Tag> {
  template <typename... Options>
  static VECOPS_ALWAYS_INLINE void call(
      PrefetchOp, Tag, const ElementOf<Tag>* pointer, Options&&...) {
    constexpr bool l2 =
        (std::same_as<std::remove_cvref_t<Options>,
                      mem::PrefetchL2> || ...);
    constexpr bool stream =
        (std::same_as<std::remove_cvref_t<Options>,
                      mem::PrefetchStream> || ...);
    constexpr bool write =
        (std::same_as<std::remove_cvref_t<Options>,
                      mem::PrefetchWrite> || ...);
    constexpr enum svprfop operation = [] {
      if constexpr (write && l2 && stream) return SV_PSTL2STRM;
      else if constexpr (write && l2) return SV_PSTL2KEEP;
      else if constexpr (write && stream) return SV_PSTL1STRM;
      else if constexpr (write) return SV_PSTL1KEEP;
      else if constexpr (l2 && stream) return SV_PLDL2STRM;
      else if constexpr (l2) return SV_PLDL2KEEP;
      else if constexpr (stream) return SV_PLDL1STRM;
      else return SV_PLDL1KEEP;
    }();

    if constexpr (sizeof(ElementOf<Tag>) == 1)
      svprfb(svptrue_b8(), pointer, operation);
    else if constexpr (sizeof(ElementOf<Tag>) == 2)
      svprfh(svptrue_b16(), pointer, operation);
    else if constexpr (sizeof(ElementOf<Tag>) == 4)
      svprfw(svptrue_b32(), pointer, operation);
    else
      svprfd(svptrue_b64(), pointer, operation);
  }
};

} // namespace vecops::vec::details

#endif // VECOPS_VEC_DETAILS_SVE_PREFETCH_H
