#ifndef VECOPS_NVEC_OPTIONS_H
#define VECOPS_NVEC_OPTIONS_H

#include <type_traits>

#include "vecops/nvec/VecBase.h"

namespace vecops::nvec::opt {

/** Selects lanes for a policy-bearing vector operation. */
template <MaskValue M>
struct Masked {
  const M& value;
};

/**
 * Selects the lanes whose corresponding mask bit is true.
 *
 * The mask must be a named lvalue. Keeping only a reference is required for
 * sizeless SVE predicates, which cannot be stored as ordinary data members.
 */
template <MaskValue M>
VECOPS_ALWAYS_INLINE Masked<M> masked(const M& value) {
  return {value};
}

template <typename M>
  requires MaskValue<M> && (!std::is_lvalue_reference_v<M>)
Masked<std::remove_cvref_t<M>> masked(M&&) = delete;

/** Selects the first count logical lanes, after clamping to [0, size(tag)]. */
struct First {
  nint_t count;
};

VECOPS_ALWAYS_INLINE constexpr First first(nint_t count) {
  return {count};
}

/** Supplies a vector value for inactive lanes. */
template <VectorValue V>
struct VectorMerge {
  const V& value;
};

/**
 * Preserves lanes from a named vector lvalue where an operation is inactive.
 * A reference wrapper also keeps this option usable with sizeless SVE values.
 */
template <VectorValue V>
VECOPS_ALWAYS_INLINE VectorMerge<V> merge(const V& value) {
  return {value};
}

template <typename V>
  requires VectorValue<V> && (!std::is_lvalue_reference_v<V>)
VectorMerge<std::remove_cvref_t<V>> merge(V&&) = delete;

/** Supplies one scalar value for every inactive lane. */
template <Element T>
struct ScalarMerge {
  T value;
};

template <Element T>
VECOPS_ALWAYS_INLINE constexpr ScalarMerge<T> merge(T value) {
  return {value};
}

/** Compile-time lane order for local shuffle operations. */
template <int... Indices>
struct LaneOrder {};

template <int... Indices>
inline constexpr LaneOrder<Indices...> lanes{};

} // namespace vecops::nvec::opt

#endif // VECOPS_NVEC_OPTIONS_H
