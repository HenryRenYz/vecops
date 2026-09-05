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
 * backend scratch, the bounded online-packing alternatives, the optional
 * flattened batch-rows packed-A variant, and periodic C-input materialization.
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
    nint_t whole_packing_bytes = 0;
    // Reserve only the operand types selected from Meta bounds, plus the one
    // possible native-AMX B alternative.  Dynamic extents no longer retain a
    // runtime union of raw/A/B/AB variants.
    if constexpr (Invocation::MayAutoPackA) {
      if constexpr (Invocation::SMEBatchRowsFullPackCandidate) {
        // The two-sided flattened SME branch needs a complete [batch*M,K]
        // packed A rather than the ordinary one-leaf staging buffer.
        whole_packing_bytes +=
            op.batch_rows_flatten_enabled() && op.online_packs_b()
            ? op.batch_rows_packed_a_bytes()
            : op.template auto_packed_bytes<
                  ::vecops::matmul::Operand::A>(op.a_);
      } else {
        whole_packing_bytes += op.template auto_packed_bytes<
            ::vecops::matmul::Operand::A>(op.a_);
      }
    }
    if constexpr (Invocation::MayAutoPackB)
      whole_packing_bytes += op.template auto_packed_bytes<
          ::vecops::matmul::Operand::B>(op.b_);
    // The bounded WholeProblem panel route and complete online packing are
    // mutually exclusive. Reserve the larger alternative instead of adding
    // both lifetimes; scratch above is shared by either branch.
    if (op.whole_b_panel_enabled())
      bytes += op.whole_b_panel_bytes();
    else
      bytes += whole_packing_bytes;
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
