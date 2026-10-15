// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
// Preserve MatmulPackBench's SeparateSum/Fused algorithms and mutable sidecar.
#include "vecops/Kernel.h"
#include "vecops/ops/MatmulPack.h"
#include "dsl_matmul_atom.h"

void __kernel__(vecops::tensor::TensorLike auto A,
                vecops::tensor::TensorLike auto P,
                vecops::tensor::TensorLike auto D, std::int64_t repeat) {
  using namespace vecops;
  using Atom = matmul::AMX_I8I32<uint8_t, int8_t>;
  const auto source = tensor::transpose<0,1>(A);
  const auto input = [&]() VECOPS_INLINE_LAMBDA {
    if constexpr (spec::Transform) {
      auto transform = tensor::make_elementwise_vec_transform<float32_t,float32_t>(
          [](auto tag, auto value) VECOPS_KERNEL_LAMBDA {
            return vec::mul(tag,value,vec::fill(tag,float32_t{4}));
          });
      return tensor::input<int8_t>(source,transform);
    } else return tensor::input<int8_t>(source);
  }();
  const auto output = tensor::make_tensor(P.data(),
      matmul::packed_layout<Atom,matmul::Operand::B>(source.layout()));
  execution::ExecutionSession execution;
  for (std::int64_t iteration=0;iteration<repeat;++iteration) {
    if constexpr (spec::CompensationMode==1) {
      matmul::details::run_matmul_pack<Atom,matmul::Operand::B>(execution,input,output);
      // The original benchmark's scalar loop bounds are registration constants
      // for both metadata modes. Keep them distinct from input tensor metadata.
      for (nint_t row=0;row<spec::LogicalN;++row) {
        int32_t sum=0;
        for (nint_t kk=0;kk<spec::LogicalK;++kk) {
          const auto value=source(row,kk);
          if constexpr (spec::Transform) sum+=static_cast<int8_t>(value*4.0f);
          else sum+=static_cast<int8_t>(value);
        }
        D(row)=-int32_t{3}*sum;
      }
    } else {
      static_assert(spec::CompensationMode==2);
      matmul::details::run_matmul_pack_b_compensated<Atom>(execution,input,output,D,int32_t{3});
    }
    asm volatile("" ::: "memory");
  }
}
