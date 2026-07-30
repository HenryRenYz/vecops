#ifndef VECOPS_VEC_PREFETCH_H
#define VECOPS_VEC_PREFETCH_H

#include <utility>

#include "vecops/vec/Options.h"
#include "vecops/vec/details/Options.h"

namespace vecops::vec {

struct PrefetchOp {
  template <VectorTag Tag, typename... Options>
    requires (details::valid_prefetch_options<Options...>())
  VECOPS_ALWAYS_INLINE void operator()(
      Tag tag, const ElementOf<Tag>* pointer, Options&&... options) const;
};

} // namespace vecops::vec

#include "vecops/vec/details/Dispatch.h"
#include "vecops/vec/details/scalar/Prefetch.h"

#if defined(ARCH_X86_FAMILY) && !defined(CPU_CAPABILITY_GENERIC)
#include "vecops/vec/details/x86/Prefetch.h"
#elif defined(CPU_CAPABILITY_SVE)
#include "vecops/vec/details/sve/Prefetch.h"
#endif

namespace vecops::vec {

/**
 * Prefetches the cache line containing pointer for a future contiguous access.
 *
 * The three independent option dimensions default to `mem::prefetch_l1`,
 * `mem::prefetch_keep`, and `mem::prefetch_read`. Backends silently map an
 * unsupported combination to the closest available hint. Prefetching is only
 * a performance hint and does not make the pointed-to object accessible.
 *
 * @see load for actual memory loads.
 * @see mem::PrefetchL1, mem::PrefetchL2, mem::PrefetchKeep,
 *      mem::PrefetchStream, mem::PrefetchRead, mem::PrefetchWrite.
 */
template <VectorTag Tag, typename... Options>
  requires (details::valid_prefetch_options<Options...>())
VECOPS_ALWAYS_INLINE void PrefetchOp::operator()(
    Tag tag, const ElementOf<Tag>* pointer, Options&&... options) const {
  details::execute(
      *this, tag, pointer, std::forward<Options>(options)...);
}

inline constexpr PrefetchOp prefetch{};

} // namespace vecops::vec

#endif // VECOPS_VEC_PREFETCH_H
