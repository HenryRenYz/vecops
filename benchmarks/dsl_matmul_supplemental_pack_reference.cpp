// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
// Exact public/forced packing selectors for the supplemental registration ledger.
#include "vecops/Kernel.h"
#include "vecops/ops/MatmulPack.h"
#include "dsl_matmul_atom.h"

template<class Atom, vecops::matmul::Operand Side, class Implementation,
         class Input, class Output>
VECOPS_ALWAYS_INLINE void supplemental_forced_pack(
    vecops::execution::ExecutionSession& execution,
    const Input& input, const Output& output) {
  struct Requirement {
    using ResourceRequirements =
      vecops::kernel::matmul_pack_implementation::resource_requirements_t<Atom, Side, Implementation>;
  } requirement;
  execution.with_resources(requirement, [&](auto& active) VECOPS_INLINE_LAMBDA {
    using Packing = vecops::matmul::packing_t<Atom, Side>;
    using InputPolicy = vecops::tensor::InputAccessPolicy<
      Packing::VectorAxis, 1, vecops::tensor::AccessPlan::direct>;
    using OutputPolicy = vecops::tensor::OutputAccessPolicy<
      Output::OutputTensor::Ndim - 1, vecops::tensor::AccessPlan::direct>;
    vecops::kernel::with_operands(active,
      vecops::tensor::operand(input, InputPolicy{}),
      vecops::tensor::operand(output, OutputPolicy{}),
      [&](auto& source, auto& destination) VECOPS_INLINE_LAMBDA {
        vecops::kernel::matmul_pack_bound<Atom, Side>(active, source, destination, Implementation{});
        destination.commit();
      });
  });
}

void __kernel__(vecops::tensor::TensorLike auto A,
                vecops::tensor::TensorLike auto P, std::int64_t repeat) {
  using Native = vecops::spec::Native;
  using Atom = BenchmarkAtom<vecops::spec::NativeA, vecops::spec::NativeB>;
  constexpr auto side = vecops::spec::SideB ? vecops::matmul::Operand::B : vecops::matmul::Operand::A;
  auto source = [&]() VECOPS_INLINE_LAMBDA {
    if constexpr (vecops::spec::SideB) return vecops::tensor::transpose<0,1>(A);
    else return A;
  }();
  auto input = [&]() VECOPS_INLINE_LAMBDA {
    if constexpr (vecops::spec::Pipeline == 1) {
      auto transform = vecops::tensor::make_elementwise_vec_transform<float,float>(
        [](auto tag, auto value) VECOPS_KERNEL_LAMBDA {
          return vecops::vec::mul(tag,value,vecops::vec::fill(tag,4.0f));
        });
      return vecops::tensor::input<Native>(source,transform);
    } else if constexpr (vecops::spec::Pipeline == 2) {
      auto transform = vecops::tensor::make_elementwise_vec_transform<Native,Native>(
        [](auto tag, auto value) VECOPS_KERNEL_LAMBDA {
          return vecops::vec::add(tag,value,vecops::vec::fill(tag,Native{1}));
        });
      return vecops::tensor::input<Native>(source,transform);
    } else return vecops::tensor::input<Native>(source);
  }();
  auto target = vecops::tensor::make_tensor(P.data(),
    vecops::matmul::packed_layout<Atom,side>(source.layout()));
  auto output = vecops::tensor::output<Native>(target);
  vecops::execution::ExecutionSession execution;
  for (std::int64_t iteration=0; iteration<repeat; ++iteration) {
    if constexpr (vecops::spec::Implementation == 0) {
      auto operation = vecops::ops::matmul_pack(vecops::ops::MatmulPackConfig<Atom,side>{});
      operation(execution,input,output);
    } else if constexpr (vecops::spec::Implementation == 1) {
      supplemental_forced_pack<Atom,side,vecops::kernel::matmul_pack_implementation::Vector>(execution,input,output);
#if defined(HAS_SME)
    } else if constexpr (vecops::spec::Implementation == 2) {
      supplemental_forced_pack<Atom,side,vecops::kernel::matmul_pack_implementation::SMEPostprocess>(execution,input,output);
#if defined(HAS_SME_F64F64)
    } else if constexpr (vecops::spec::Implementation == 3) {
      supplemental_forced_pack<Atom,side,vecops::kernel::matmul_pack_implementation::SMEFP32ToFP64>(execution,input,output);
#endif
    } else if constexpr (vecops::spec::Implementation == 4) {
      supplemental_forced_pack<Atom,side,vecops::kernel::matmul_pack_implementation::SMEStagedTransform>(execution,input,output);
    } else if constexpr (vecops::spec::Implementation == 5) {
      supplemental_forced_pack<Atom,side,vecops::kernel::matmul_pack_implementation::SMEStagedFP16ToFP32>(execution,input,output);
#endif
    }
    asm volatile("" ::: "memory");
  }
}
