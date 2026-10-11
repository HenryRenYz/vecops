// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
// Same atom, blocks and metadata as ops/AttentionBench.cpp. Workspace is
// supplied by the benchmark, so timed iterations do not allocate its storage.
#include "vecops/Kernel.h"
#include "vecops/ops/Attention.h"

using namespace vecops;
#if defined(ARCH_X86_FAMILY)
using BenchAtom = matmul::AMX_BF16F32;
#elif defined(HAS_SME)
using BenchAtom = matmul::SME_BF16F32;
#else
#error "AttentionBench requires AMX or SME"
#endif

void __kernel__(tensor::TensorLike auto q, tensor::TensorLike auto k,
                tensor::TensorLike auto v, tensor::TensorLike auto out,
                tensor::TensorLike auto scratch, tensor::TensorLike auto info,
                double scale, nint_t repeat) {
  constexpr auto strategy = static_cast<ops::SDPAStrategy>(nint_t(spec::Strategy));
  using Config = ops::AttentionConfig<ops::MatmulConfig<BenchAtom>, 32, 32,
      ops::AttentionCausalMode::none, strategy>;
  auto operation = ops::scaled_dot_product_attention(Config{});
  // Planning can be expensive. AttentionBench does it before timing; this
  // benchmark likewise probes once with repeat=0, never on a timed call.
  if (repeat == 0) {
    info.data()[0] = operation.required_workspace(q, k, v, out);
    return;
  }
  kernel::WorkspaceView workspace(scratch.data(), tensor::size<0>(scratch));
  for (nint_t iteration = 0; iteration < repeat; ++iteration) {
    workspace.reset();
    operation(workspace, q, k, v, out, static_cast<float>(scale));
    asm volatile("" ::: "memory");
  }
}
