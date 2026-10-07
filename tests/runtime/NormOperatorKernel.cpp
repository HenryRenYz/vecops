// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
/** @file NormOperatorKernel.cpp @brief Compiler-managed normalization kernel. */

#include "vecops/Kernel.h"
#include "vecops/ops/LayerNorm.h"
#include "vecops/ops/Softmax.h"

using namespace vecops;

constexpr auto B = spec::B;
constexpr auto D = spec::D;
constexpr auto Op = spec::Op;
using ComputeType = spec::ComputeType;

void __kernel__(TensorLike auto input, auto scale, auto bias, TensorLike auto output, float64_t eps) {
  assert_shape(input, {B, D}, {D, cint<1>});
  assert_shape(output, {B, D}, {D, cint<1>});
  if constexpr (nint_t(Op) == 1) {
    auto operation = ops::layer_norm(ops::LayerNormConfig<ComputeType>{static_cast<ComputeType>(eps)});
    operation(input, scale, bias, output);
  } else {
    static_assert(nint_t(Op) == 2);
    ops::softmax(ops::SoftmaxConfig<ComputeType>{})(input, output);
  }
}
