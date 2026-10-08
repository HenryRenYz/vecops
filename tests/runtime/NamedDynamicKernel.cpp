// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
/** @file NamedDynamicKernel.cpp @brief Runtime-valued named-dimension probe. */

#include "vecops/Kernel.h"

using namespace vecops;

using B = spec::B;
constexpr auto D = spec::D;

void __kernel__(TensorLike auto input, TensorLike auto output) {
  const auto batch = B{input.size(0)};
  assert_shape(input, {vecops::any, D}, {D, cint<1>});
  assert_shape(output, {vecops::any, D}, {D, cint<1>});
  for (nint_t row = 0; row < batch; ++row)
    for (nint_t column = 0; column < D; ++column)
      output(row, column) = input(row, column) + 1.0f;
}
