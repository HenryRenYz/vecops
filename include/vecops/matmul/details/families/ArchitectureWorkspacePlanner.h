//
// Copyright (c) vecops contributors.
//

#ifndef VECOPS_MATMUL_DETAILS_FAMILIES_ARCHITECTURE_WORKSPACE_PLANNER_H
#define VECOPS_MATMUL_DETAILS_FAMILIES_ARCHITECTURE_WORKSPACE_PLANNER_H

#include "vecops/matmul/details/Kernel.h"

namespace vecops::ops::matmul_details {

/** Workspace accounting for packing, backend scratch, and batch materialization. */
template <typename Invocation>
struct ArchitectureWorkspacePlanner {
  VECOPS_ALWAYS_INLINE static nint_t required(const Invocation& op) {
    nint_t bytes = kernel::matmul_implementation::scratch_bytes<
        typename Invocation::Implementation>();
    if constexpr (Invocation::CompileTimeAutoPacking) {
      if constexpr (Invocation::SMEBatchRowsFullPackCandidate) {
        bytes += op.batch_rows_flatten_enabled()
            ? op.batch_rows_packed_a_bytes()
            : op.template auto_packed_bytes<
                  ::vecops::matmul::Operand::A>(op.a_);
      } else {
        bytes += op.template auto_packed_bytes<
            ::vecops::matmul::Operand::A>(op.a_);
      }
      bytes += op.template auto_packed_bytes<
          ::vecops::matmul::Operand::B>(op.b_);
    }
    if (op.batch_rows_pack_a_enabled())
      bytes += op.batch_rows_packed_a_bytes();
    if (op.batch_columns_periodic_c_input_enabled())
      bytes += op.batch_columns_periodic_c_input_bytes();
    return bytes;
  }
};

} // namespace vecops::ops::matmul_details

#endif // VECOPS_MATMUL_DETAILS_FAMILIES_ARCHITECTURE_WORKSPACE_PLANNER_H
