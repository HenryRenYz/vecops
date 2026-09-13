//
// Copyright (c) vecops contributors.
//

#ifndef VECOPS_MATMUL_DETAILS_PROBLEM_MAPPER_H
#define VECOPS_MATMUL_DETAILS_PROBLEM_MAPPER_H

#include <type_traits>

#include "vecops/Assertion.h"
#include "vecops/Meta.h"

namespace vecops::matmul::details {

template <meta::ValueType M, meta::ValueType N, meta::ValueType K>
struct MappedProblem {
  [[no_unique_address]] M m;
  [[no_unique_address]] N n;
  [[no_unique_address]] K k;
};

/**
 * Normalize and validate mathematical extents without choosing an algorithm.
 * Runtime extents are retyped after validation so the non-negative matmul
 * contract remains visible to downstream tiling and backend dispatch.
 */
struct ProblemMapper {
private:
  template <meta::ValueType Extent>
  VECOPS_INLINE static auto nonnegative(Extent extent) {
    using E = std::remove_cvref_t<Extent>;
    if constexpr (!E::is_runtime ||
                  meta::lower_bound_at_least_v<E, 0>) {
      return extent;
    } else {
      return meta::Dynamic<
          E::alignment, 0, meta::upper_bound_v<E>>{
              static_cast<nint_t>(extent)};
    }
  }

public:
  template <meta::ValueType M, meta::ValueType N, meta::ValueType K>
  VECOPS_INLINE static auto map(M m, N n, K k) {
    VECOPS_ASSERT(static_cast<nint_t>(m) >= 0 &&
                  static_cast<nint_t>(n) >= 0 &&
                  static_cast<nint_t>(k) >= 0,
                  "matmul extents must be non-negative");
    auto mapped_m = nonnegative(m);
    auto mapped_n = nonnegative(n);
    auto mapped_k = nonnegative(k);
    return MappedProblem<
        decltype(mapped_m), decltype(mapped_n), decltype(mapped_k)>{
            mapped_m, mapped_n, mapped_k};
  }
};

} // namespace vecops::matmul::details

#endif // VECOPS_MATMUL_DETAILS_PROBLEM_MAPPER_H
