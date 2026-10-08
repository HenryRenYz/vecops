// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
/** Benchmark-only typed reference; workspace is reused outside the repeat loop. */
#include "vecops/Kernel.h"
#include "vecops/execution/Parallel.h"
#include "vecops/kernel/Workspace.h"
#include "vecops/ops/LayerNorm.h"
#include "vecops/ops/Softmax.h"

using namespace vecops;
using ComputeType = spec::ComputeType;

void __kernel__(TensorLike auto input, auto scale, auto bias,
                TensorLike auto output, nint_t repeat) {
  auto invoke = [&](auto operation) VECOPS_INLINE_LAMBDA {
    auto x = tensor::input<ComputeType>(input);
    auto y = tensor::output<ComputeType>(output);
    if constexpr (nint_t(spec::Op) == 1) {
      auto with_parameters = [&](auto gamma, auto beta) VECOPS_INLINE_LAMBDA {
        kernel::Workspace storage(operation.required_workspace(x, gamma, beta, y));
        auto workspace = storage.view();
        for (nint_t iteration = 0; iteration < repeat; ++iteration) {
          operation(workspace, x, gamma, beta, y);
          asm volatile("" ::: "memory");
        }
      };
      if constexpr (bool(spec::HasGamma) && bool(spec::HasBeta)) {
        with_parameters(tensor::input<ComputeType>(scale), tensor::input<ComputeType>(bias));
      } else if constexpr (bool(spec::HasGamma)) {
        with_parameters(tensor::input<ComputeType>(scale), tensor::nullopt);
      } else if constexpr (bool(spec::HasBeta)) {
        with_parameters(tensor::nullopt, tensor::input<ComputeType>(bias));
      } else {
        with_parameters(tensor::nullopt, tensor::nullopt);
      }
    } else {
      kernel::Workspace storage(operation.required_workspace(x, y));
      auto workspace = storage.view();
      for (nint_t iteration = 0; iteration < repeat; ++iteration) {
        operation(workspace, x, y);
        asm volatile("" ::: "memory");
      }
    }
  };
  if constexpr (nint_t(spec::Op) == 1) {
    invoke(ops::layer_norm(ops::LayerNormConfig<ComputeType>{ComputeType(1e-5)}));
  } else {
    constexpr auto Accuracy = static_cast<vec::Accuracy>(nint_t(spec::Accuracy));
    invoke(ops::softmax(ops::SoftmaxConfig<ComputeType, Accuracy>{}));
  }
}
