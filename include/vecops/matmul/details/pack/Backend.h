//
// Copyright (c) vecops contributors.
//

#ifndef VECOPS_MATMUL_DETAILS_PACK_BACKEND_H
#define VECOPS_MATMUL_DETAILS_PACK_BACKEND_H

#include "vecops/Features.h"
#include "vecops/execution/ExecutionSession.h"
#include "vecops/matmul/details/pack/Types.h"

namespace vecops::kernel::matmul_pack_details {

template <typename Format, typename Implementation>
struct Backend {
  using ResourceRequirements = execution::details::ResourceSet<>;
  static constexpr bool supports_column_compensation = false;

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
#include "vecops/matmul/details/pack/amx/Backend.h"
#endif

#if defined(HAS_SME)
#include "vecops/matmul/details/pack/sme/Backend.h"
#endif

#endif // VECOPS_MATMUL_DETAILS_PACK_BACKEND_H
