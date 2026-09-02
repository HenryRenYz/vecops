//
// Copyright (c) vecops contributors.
//

#ifndef VECOPS_MATMUL_DETAILS_PROBLEM_MAPPER_H
#define VECOPS_MATMUL_DETAILS_PROBLEM_MAPPER_H

#include "vecops/Assertion.h"
#include "vecops/Meta.h"

namespace vecops::matmul::details {

template <meta::ValueType M, meta::ValueType N, meta::ValueType K>
struct MappedProblem {
  [[no_unique_address]] M m;
  [[no_unique_address]] N n;
  [[no_unique_address]] K k;
};

/** Normalize and validate mathematical extents without choosing an algorithm. */
struct ProblemMapper {
  template <meta::ValueType M, meta::ValueType N, meta::ValueType K>
  VECOPS_INLINE static auto map(M m, N n, K k) {
    VECOPS_ASSERT(static_cast<nint_t>(m) >= 0 &&
                  static_cast<nint_t>(n) >= 0 &&
                  static_cast<nint_t>(k) >= 0,
                  "matmul extents must be non-negative");
    return MappedProblem<M, N, K>{m, n, k};
  }
};

} // namespace vecops::matmul::details

#endif // VECOPS_MATMUL_DETAILS_PROBLEM_MAPPER_H
