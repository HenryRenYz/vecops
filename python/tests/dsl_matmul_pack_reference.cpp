// Public packed ABI conformance oracle; no private packing helpers.
#include "vecops/Kernel.h"
#include "vecops/ops/MatmulPack.h"
void __kernel__(vecops::tensor::TensorLike auto A,vecops::tensor::TensorLike auto P) {
  using T=vecops::spec::Native;
  using Atom=std::conditional_t<std::same_as<T,vecops::bfloat16_t>,vecops::matmul::AMX_BF16F32,
                               vecops::matmul::AMX_I8I32<std::int8_t,std::int8_t>>;
  constexpr auto side=vecops::spec::SideB ? vecops::matmul::Operand::B : vecops::matmul::Operand::A;
  using Packing=vecops::matmul::packing_t<Atom,side>;
  auto source=[&]() VECOPS_INLINE_LAMBDA {
    if constexpr (vecops::spec::SideB) return vecops::tensor::transpose<0,1>(A);
    else return A;
  }();
  auto target=vecops::tensor::make_tensor(P.data(),vecops::matmul::packed_layout<Atom,side>(source.layout()));
  vecops::execution::ExecutionSession execution;
  vecops::ops::matmul_pack(vecops::ops::MatmulPackConfig<Atom,side>{})(execution,source,target);
}
