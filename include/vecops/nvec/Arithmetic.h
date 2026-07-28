#ifndef VECOPS_NVEC_ARITHMETIC_H
#define VECOPS_NVEC_ARITHMETIC_H

#include "vecops/nvec/Basic.h"

namespace vecops::nvec {

struct AddOp {
  template <VectorTag Tag>
  VECOPS_ALWAYS_INLINE Vec<Tag> operator()(
      Tag tag, Vec<Tag> a, Vec<Tag> b) const;
};

} // namespace vecops::nvec

#include "vecops/nvec/details/Dispatch.h"
#include "vecops/nvec/details/scalar/Arithmetic.h"

#if defined(ARCH_X86_FAMILY) && !defined(CPU_CAPABILITY_GENERIC)
#include "vecops/nvec/details/x86/Arithmetic.h"
#elif defined(CPU_CAPABILITY_SVE)
#include "vecops/nvec/details/sve/Arithmetic.h"
#endif

#include "vecops/nvec/details/Arithmetic.h"

namespace vecops::nvec {

/**
 * Computes r[i] = a[i] + b[i] for every logical lane 0 <= i < size(tag).
 *
 * Integer addition wraps modulo 2^bits. float16_t and bfloat16_t results are
 * rounded back to their respective element formats. Multi-word Tags are
 * automatically batched unless a backend supplies a whole-Tag NativeImpl.
 */
template <VectorTag Tag>
VECOPS_ALWAYS_INLINE Vec<Tag> AddOp::operator()(
    Tag tag, Vec<Tag> a, Vec<Tag> b) const {
  return details::execute(*this, tag, a, b);
}

inline constexpr AddOp add{};

} // namespace vecops::nvec

#endif // VECOPS_NVEC_ARITHMETIC_H
