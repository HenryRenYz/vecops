#ifndef VECOPS_VEC_DETAILS_X86_PREFETCH_H
#define VECOPS_VEC_DETAILS_X86_PREFETCH_H

/**
 * @file Prefetch.h
 * @brief x86 backend implementations for prefetch operations.
 */

#if !defined(ARCH_X86_FAMILY)
#error "This header requires an x86 target"
#endif

#include <type_traits>

#include "vecops/vec/details/Dispatch.h"

namespace vecops::vec::details {


/* **************************************************************************** */
//                       Prefetch word implementations                        //
/* **************************************************************************** */

template <VectorTag Tag>
struct NativeImpl<X86Backend, PrefetchOp, Tag> {
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
    constexpr int hint = write || (!l2 && stream)
        ? _MM_HINT_NTA : l2 ? _MM_HINT_T1 : _MM_HINT_T0;
    _mm_prefetch(reinterpret_cast<const char*>(pointer), hint);
  }
};

} // namespace vecops::vec::details

#endif // VECOPS_VEC_DETAILS_X86_PREFETCH_H
