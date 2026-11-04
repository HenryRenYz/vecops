// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
#ifndef VECOPS_MATMUL_WORKSPACE_H
#define VECOPS_MATMUL_WORKSPACE_H
/** @file Workspace.h @brief Logical allocation sites shared by Matmul planning and execution. */
namespace vecops::matmul {
struct WorkspacePart {
  enum : unsigned { Backend, PackA, PackB, Auxiliary, Count };
};
} // namespace vecops::matmul
#endif
