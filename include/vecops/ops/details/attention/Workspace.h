// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
#ifndef VECOPS_OPS_DETAILS_ATTENTION_WORKSPACE_H
#define VECOPS_OPS_DETAILS_ATTENTION_WORKSPACE_H
namespace vecops::ops::attention_details {
/** Logical attention buffers, independent of their physical memory tier. */
struct WorkspacePart {
  enum : unsigned {
    General,
    Scores,
    Probabilities,
    Output,
    RowMax,
    RowSum,
    RowScale,
    PackedQ,
    PackedK,
    PackedV,
    Count
  };
};
} // namespace vecops::ops::attention_details
#endif
