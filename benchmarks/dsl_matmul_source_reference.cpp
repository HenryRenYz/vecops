// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
// Source-catalog baseline: preserve the public Matmul family/policy selector,
// rank-three/shared-operand strategies and public MatmulPack implementation.
// Offline packing uses the separate OnlyPack specialization, never the timed
// Matmul call. Online modes pack once per iteration before the public operator.
// "raw" means no externally prepared packed input. MatmulConfig retains its
// default automatic packing policy, which may pack inside the timed operator.
#define __kernel__ fusion_raw_reference_kernel
#include "dsl_matmul_fusion_reference.cpp"
#undef __kernel__
#include "vecops/ops/MatmulPack.h"

namespace source_benchmark {
template <int Family>
constexpr auto family_type() {
  using namespace vecops::matmul;
  if constexpr (Family == 1)
    return std::type_identity<family_selection::Require<kernel_family::SmallVector>>{};
  else if constexpr (Family == 2)
    return std::type_identity<family_selection::Require<kernel_family::PackedDot>>{};
  else if constexpr (Family == 3)
    return std::type_identity<family_selection::Require<kernel_family::RuntimeQuantInt8>>{};
  else {
    static_assert(Family == 0, "unknown source-catalog family selector");
    return std::type_identity<family_selection::Automatic>{};
  }
}

template <int Policy>
constexpr auto policy_type() {
  using namespace vecops::kernel;
  if constexpr (Policy == 1)
    return std::type_identity<loop::tile2d_policy::ExactCover>{};
  else if constexpr (Policy == 2)
    return std::type_identity<loop::tile2d_policy::FourRegions>{};
  else {
    static_assert(Policy == 0, "unknown source-catalog tile policy");
    return std::type_identity<matmul_policy::Automatic>{};
  }
}

template <typename Tensor>
VECOPS_ALWAYS_INLINE auto matrix(const Tensor& tensor) {
  if constexpr (Tensor::Ndim == 3)
    return vecops::tensor::make_tensor(
      tensor.data(),
      vecops::tensor::make_layout(
        vecops::tensor::make_shape(vecops::tensor::size<1>(tensor), vecops::tensor::size<2>(tensor)),
        vecops::tensor::make_strides(vecops::tensor::stride<1>(tensor), vecops::tensor::stride<2>(tensor))));
  else
    return tensor;
}
} // namespace source_benchmark

void __kernel__(vecops::tensor::TensorLike auto A, vecops::tensor::TensorLike auto B, vecops::tensor::TensorLike auto C,
                vecops::tensor::TensorLike auto D, vecops::tensor::TensorLike auto S,
                vecops::tensor::TensorLike auto SC, vecops::tensor::TensorLike auto SQ,
                vecops::tensor::TensorLike auto RQ, vecops::tensor::TensorLike auto RD,
                vecops::tensor::TensorLike auto W, vecops::tensor::TensorLike auto Z,
                vecops::tensor::TensorLike auto AP, vecops::tensor::TensorLike auto BP, std::int64_t repeat) {
  using namespace vecops;
  using Atom = BenchmarkAtom<spec::NativeA, spec::NativeB>;
  using Acc = typename Atom::TAcc;
  using Family = typename decltype(source_benchmark::family_type<spec::ReferenceFamily>())::type;
  using Policy = typename decltype(source_benchmark::policy_type<spec::ReferencePolicy>())::type;
  constexpr auto input_a = static_cast<fusion_benchmark::Input>(decltype(spec::InputPipelineA)::value);
  constexpr auto input_b = static_cast<fusion_benchmark::Input>(decltype(spec::InputPipelineB)::value);
  constexpr auto output = static_cast<fusion_benchmark::Output>(decltype(spec::OutputPipeline)::value);
  constexpr int rank = decltype(A)::Ndim;
  static_assert(rank == 2 || (!spec::PackA && (!spec::PackB || spec::SharedB)),
                "source catalog registers rank-three packing only for shared B");
  const auto row_stride = fusion_benchmark::row_batch_stride(A);
  const auto bt = tensor::transpose<rank - 2, rank - 1>(B);
  const auto a_raw = fusion_benchmark::input<true, typename Atom::TA, input_a>(A, RQ, Z, row_stride);
  const auto b_raw = fusion_benchmark::input<false, typename Atom::TB, input_b>(bt, RQ, Z, row_stride);
  const auto ap = tensor::make_tensor(
    AP.data(), matmul::packed_layout<Atom, matmul::Operand::A>(source_benchmark::matrix(A).layout()));
  const auto bp = tensor::make_tensor(
    BP.data(), matmul::packed_layout<Atom, matmul::Operand::B>(source_benchmark::matrix(bt).layout()));
  execution::ExecutionSession packing_execution;
  auto pack = [&]() VECOPS_INLINE_LAMBDA {
    if constexpr (spec::PackA)
      ops::matmul_pack(ops::MatmulPackConfig<Atom, matmul::Operand::A>{})(packing_execution, a_raw,
                                                                          tensor::output<typename Atom::TA>(ap));
    if constexpr (spec::PackB) {
      const auto matrix_b =
        fusion_benchmark::input<false, typename Atom::TB, input_b>(source_benchmark::matrix(bt), RQ, Z, row_stride);
      ops::matmul_pack(ops::MatmulPackConfig<Atom, matmul::Operand::B>{})(packing_execution, matrix_b,
                                                                          tensor::output<typename Atom::TB>(bp));
    }
  };
  if constexpr (spec::OnlyPack) {
    for (std::int64_t iteration = 0; iteration < repeat; ++iteration) {
      pack();
      asm volatile("" ::: "memory");
    }
  } else {
    const auto a = [&]() VECOPS_INLINE_LAMBDA {
      if constexpr (spec::PackA)
        return tensor::input<typename Atom::TA>(ap);
      else
        return a_raw;
    }();
    const auto b = [&]() VECOPS_INLINE_LAMBDA {
      if constexpr (spec::PackB)
        return tensor::input<typename Atom::TB>(bp);
      else
        return b_raw;
    }();
    const auto out = fusion_benchmark::output<Acc, output>(C, S, SC, SQ, RD, W, Z, row_stride);
    using Config = ops::MatmulConfig<Atom, Family, Policy, matmul::GenericTiledTuning<>,
                                     platform::SystemCacheInfoProvider, spec::SwapAB>;
    const auto operation = ops::matmul(Config{});
    const auto m = tensor::size<rank - 2>(A);
    const auto n = tensor::size<rank - 1>(B);
    const auto k = tensor::size<rank - 1>(A);
    auto run = [&](const auto&... prior) VECOPS_INLINE_LAMBDA {
      kernel::Workspace storage(operation.required_workspace(m, n, k, a, b, prior..., out));
      auto workspace = storage.view();
      execution::ExecutionSession execution{workspace};
      for (std::int64_t iteration = 0; iteration < repeat; ++iteration) {
        if constexpr (spec::OnlinePack)
          pack();
        operation(execution, m, n, k, a, b, prior..., out);
        asm volatile("" ::: "memory");
      }
    };
    if constexpr (spec::HasPrior)
      run(tensor::input<Acc>(fusion_benchmark::prior<spec::BroadcastPrior>(D, m, n)));
    else
      run();
  }
}
