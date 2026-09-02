//
// Copyright (c) vecops contributors.
//

#ifndef VECOPS_MATMUL_DETAILS_BACKEND_H
#define VECOPS_MATMUL_DETAILS_BACKEND_H

#include "vecops/Features.h"
#include "vecops/execution/ExecutionSession.h"
#include "vecops/matmul/Family.h"
#include "vecops/matmul/details/kernel/Types.h"

namespace vecops::kernel::matmul_details {

template <typename Implementation>
struct Backend {
  using ResourceRequirements = execution::details::ResourceSet<>;
  static constexpr int ProblemRank = 2;

  static nint_t scratch_bytes() { return 0; }

  template <typename... Args>
  VECOPS_ALWAYS_INLINE static void run(Args&&...) {
    static_assert(
        execution::details::dependent_false_v<Implementation, Args...>,
        "matrix-multiply backend is unavailable for this target");
  }
};

} // namespace vecops::kernel::matmul_details

#if defined(HAS_AMX_TILE)
#include "vecops/matmul/details/kernel/amx/Backend.h"
#endif

#if defined(HAS_SME)
#include "vecops/matmul/details/kernel/sme/Backend.h"
#endif

#endif // VECOPS_MATMUL_DETAILS_BACKEND_H
