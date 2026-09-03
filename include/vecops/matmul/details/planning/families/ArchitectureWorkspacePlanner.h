//
// Copyright (c) vecops contributors.
//

#ifndef VECOPS_MATMUL_DETAILS_FAMILIES_ARCHITECTURE_WORKSPACE_PLANNER_H
#define VECOPS_MATMUL_DETAILS_FAMILIES_ARCHITECTURE_WORKSPACE_PLANNER_H

/**
 * @file vecops/matmul/details/planning/families/ArchitectureWorkspacePlanner.h
 * @brief Total workspace bound for one architecture-family invocation.
 *
 * The bound must cover every allocation the invocation can make at run time:
 * backend scratch, compile-time online packing, the optional flattened
 * batch-rows packed-A variant, and the periodic C-input materialization.
 * The runtime gates (batch_rows_pack_a_enabled and friends) are evaluated
 * once when the invocation is constructed, so a later required_workspace()
 * query and the actual run always see the same decisions.
 */

#include "vecops/matmul/details/kernel/Kernel.h"

namespace vecops::matmul::details {

/** Workspace accounting for packing, backend scratch, and batch materialization. */
template <typename Invocation>
struct ArchitectureWorkspacePlanner {
  /// Byte bound covering every execution branch of @p op (see the file
  /// comment for the coverage contract).
  VECOPS_ALWAYS_INLINE static nint_t required(const Invocation& op) {
    nint_t bytes = kernel::matmul_implementation::scratch_bytes<
        typename Invocation::Implementation>();
    if constexpr (Invocation::CompileTimeAutoPacking) {
      if constexpr (Invocation::SMEBatchRowsFullPackCandidate) {
        // The full-pack variant swaps the ordinary packed-A bytes for the
        // flattened [batch*M, K] packed A when flattening will actually run.
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
    // The two additions below are runtime-gated but their gates were cached
    // by the constructor, keeping this a deterministic pure query.
    if (op.batch_rows_pack_a_enabled())
      bytes += op.batch_rows_packed_a_bytes();
    if (op.batch_columns_periodic_c_input_enabled())
      bytes += op.batch_columns_periodic_c_input_bytes();
    return bytes;
  }
};

} // namespace vecops::matmul::details

#endif // VECOPS_MATMUL_DETAILS_FAMILIES_ARCHITECTURE_WORKSPACE_PLANNER_H
