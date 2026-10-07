// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
/** @file TestKernel.cpp @brief Minimal compiler-managed `__kernel__` source. */

#include "vecops/Kernel.h"
#include "vecops/execution/WorkspaceContext.h"

using namespace vecops;

constexpr auto B = spec::B;
constexpr auto D = spec::D;
constexpr auto stride_D = spec::stride_D;
using ComputeType = spec::ComputeType;

void __kernel__(execution::WorkspaceContext& workspace, TensorLike auto input, TensorLike auto output, float64_t factor,
                bool add_one) {
  assert_shape(input, {B, D}, {stride_D, cint<1>});
  assert_shape(output, {B, D}, {any, cint<1>});
  auto phase = workspace.serial_scope("compute");
  auto scalar = workspace.request("scalar", {.bytes = sizeof(ComputeType), .alignment = alignof(ComputeType)});
  auto* scratch = static_cast<ComputeType*>(scalar.replica());
  for (nint_t row = 0; row < B; ++row) {
    for (nint_t column = 0; column < D; ++column) {
      *scratch = static_cast<ComputeType>(input(row, column) * factor + (add_one ? 1.0 : 0.0));
      output(row, column) = *scratch;
    }
  }
}
