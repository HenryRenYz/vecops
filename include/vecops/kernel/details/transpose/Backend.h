//
// Copyright (c) vecops contributors.
//

#ifndef VECOPS_KERNEL_DETAILS_TRANSPOSE_BACKEND_H
#define VECOPS_KERNEL_DETAILS_TRANSPOSE_BACKEND_H

#include "vecops/execution/ExecutionSession.h"
#include "vecops/kernel/details/transpose/Types.h"
#include "vecops/platform/Capabilities.h"

/**
 * @file Backend.h
 * @brief Compile-time Transpose2D backend selection and implementation mapping.
 *
 * The vector backend is chosen from target capability macros and may delegate
 * to the shared generic-vector network. SME is a separate implementation with
 * a distinct execution-resource contract. Only the selected platform headers
 * are included; unsupported implementations fail during template instantiation
 * instead of falling back at runtime.
 */

#if defined(CPU_CAPABILITY_SVE)
#include "vecops/kernel/details/transpose/sve/Backend.h"
#elif defined(ARCH_X86_FAMILY) && !defined(CPU_CAPABILITY_GENERIC)
#include "vecops/kernel/details/transpose/x86/Backend.h"
#else
#include "vecops/kernel/details/transpose/scalar/Backend.h"
#endif

#if defined(HAS_SME)
#include "vecops/kernel/details/transpose/sme/Transpose2D.h"
#endif

namespace vecops::kernel::transpose2d_details {

template <typename Implementation>
/** Map a public implementation tag to its backend and resource requirements. */
struct ImplementationBackend;

#if defined(CPU_CAPABILITY_SVE)
/** Vector transpose backend selected for the current translation unit. */
using CurrentVectorBackend = SVEBackend;
#elif defined(ARCH_X86_FAMILY) && !defined(CPU_CAPABILITY_GENERIC)
using CurrentVectorBackend = X86Backend;
#else
using CurrentVectorBackend = ScalarBackend;
#endif

template <>
/** Vector implementation mapped to the selected platform vector backend. */
struct ImplementationBackend<transpose2d_implementation::Vector>
    : CurrentVectorBackend {
  using ResourceRequirements =
      typename execution::details::current_backend_t::DefaultRequirements;
};

#if defined(HAS_SME)
template <>
/** SME implementation mapped to the streaming/ZA backend. */
struct ImplementationBackend<transpose2d_implementation::SME> : SMEBackend {};
#else
template <>
/** Unsupported SME placeholder producing a dependent compile-time diagnostic. */
struct ImplementationBackend<transpose2d_implementation::SME> {
  using ResourceRequirements = execution::details::ResourceSet<>;

  template <int SrcRow, int SrcCol, int DstRow, int DstCol,
            typename... Args>
  /** Fail only when an unavailable SME implementation is actually invoked. */
  VECOPS_ALWAYS_INLINE static void run(Args&&...) {
    (void)SrcRow;
    (void)SrcCol;
    (void)DstRow;
    (void)DstCol;
    static_assert(
        execution::details::dependent_false_v<Args...>,
        "SME transpose implementation requires an SME target");
  }
};
#endif

} // namespace vecops::kernel::transpose2d_details

#endif // VECOPS_KERNEL_DETAILS_TRANSPOSE_BACKEND_H
