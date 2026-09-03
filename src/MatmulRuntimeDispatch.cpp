//
// Copyright (c) vecops contributors.
//
// Runtime matmul family predicates live in one non-ISA object so every
// template instantiation shares the same branch body.

#include "vecops/matmul/details/kernel/RuntimeDispatch.h"

namespace vecops::kernel::matmul_details {

VECOPS_NOINLINE bool runtime_amx_small_vector_profitable(
    SmallVectorShape shape, nint_t m, nint_t n, nint_t k) {
  return runtime_dispatch_rules::amx_small_vector_profitable(
      shape, m, n, k);
}

VECOPS_NOINLINE bool runtime_sme_small_vector_profitable(
    SmallVectorShape shape, nint_t m, nint_t n, nint_t k) {
  return runtime_dispatch_rules::sme_small_vector_profitable(
      shape, m, n, k);
}

VECOPS_NOINLINE bool runtime_small_area_profitable(
    nint_t m, nint_t n, nint_t k, nint_t limit) {
  return k > 0 && runtime_dispatch_rules::area_at_most(m, n, limit);
}

VECOPS_NOINLINE bool runtime_amx_residual_split_profitable(
    nint_t m, nint_t n, nint_t k) {
  return runtime_dispatch_rules::amx_residual_split_profitable(m, n, k);
}

} // namespace vecops::kernel::matmul_details
