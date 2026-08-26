//
// Copyright (c) vecops contributors.
//

#ifndef VECOPS_KERNEL_DETAILS_MATMUL_BACKEND_H
#define VECOPS_KERNEL_DETAILS_MATMUL_BACKEND_H

#include "vecops/Features.h"
#include "vecops/execution/ExecutionSession.h"
#include "vecops/kernel/details/matmul/Types.h"

namespace vecops::kernel::matmul_details {

template <typename Implementation>
struct Backend {
  using ResourceRequirements = execution::details::ResourceSet<>;

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
#include "vecops/kernel/details/matmul/amx/Backend.h"
#endif

#if defined(HAS_SME)
#include "vecops/kernel/details/matmul/sme/Backend.h"
#endif

#endif // VECOPS_KERNEL_DETAILS_MATMUL_BACKEND_H
