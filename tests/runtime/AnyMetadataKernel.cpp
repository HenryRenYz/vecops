// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
/** @file AnyMetadataKernel.cpp @brief Runtime-valued unconstrained metadata probe. */

#include "vecops/Kernel.h"

using namespace vecops;

void __kernel__(TensorLike auto input, TensorLike auto output) {
  assert_shape(input, {any, any}, {any, cint<1>});
  assert_shape(output, {any, any}, {any, cint<1>});
  for (nint_t row = 0; row < input.size(0); ++row)
    for (nint_t column = 0; column < input.size(1); ++column)
      output(row, column) = input(row, column) + 1.0f;
}
