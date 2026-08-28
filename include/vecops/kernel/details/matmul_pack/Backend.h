//
// Copyright (c) vecops contributors.
//

#ifndef VECOPS_KERNEL_DETAILS_MATMUL_PACK_BACKEND_H
#define VECOPS_KERNEL_DETAILS_MATMUL_PACK_BACKEND_H

#include "vecops/Features.h"
#include "vecops/execution/ExecutionSession.h"
#include "vecops/kernel/details/matmul_pack/Types.h"

namespace vecops::kernel::matmul_pack_details {

template <typename Format, typename Implementation>
struct Backend {
  using ResourceRequirements = execution::details::ResourceSet<>;

  template <typename InputSpec, typename OutputSpec>
  static constexpr bool eligible = false;

  template <gemm::Atom Atom, gemm::Operand Side, typename... Args>
  VECOPS_ALWAYS_INLINE static void run(Args&&...) {
    static_assert(
        execution::details::dependent_false_v<Format, Implementation, Args...>,
        "matmul packing backend is unavailable for this target");
  }
};

} // namespace vecops::kernel::matmul_pack_details

#if defined(ARCH_X86_FAMILY)
#include "vecops/kernel/details/matmul_pack/amx/Backend.h"
#endif

#if defined(HAS_SME_FA64)
#include "vecops/kernel/details/matmul_pack/sme/Backend.h"
#endif

#endif // VECOPS_KERNEL_DETAILS_MATMUL_PACK_BACKEND_H
