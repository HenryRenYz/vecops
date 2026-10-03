// Paired public MatmulPack baseline, including optional load/convert/affine.
#include "vecops/Kernel.h"
#include "vecops/ops/MatmulPack.h"

void __kernel__(vecops::tensor::TensorLike auto A,vecops::tensor::TensorLike auto P,
                std::int64_t repeat) {
  using Native=vecops::spec::Native;
  using Atom=std::conditional_t<std::same_as<Native,vecops::bfloat16_t>,vecops::matmul::AMX_BF16F32,
                               vecops::matmul::AMX_I8I32<std::int8_t,std::int8_t>>;
  constexpr auto side=vecops::spec::SideB ? vecops::matmul::Operand::B : vecops::matmul::Operand::A;
  auto source=[&]() VECOPS_INLINE_LAMBDA {
    if constexpr (vecops::spec::SideB) return vecops::tensor::transpose<0,1>(A);
    else return A;
  }();
  auto input=[&]() VECOPS_INLINE_LAMBDA {
    if constexpr (vecops::spec::Transform) {
      auto transform=vecops::tensor::make_elementwise_vec_transform<float,float>(
        [](auto tag,auto value) VECOPS_KERNEL_LAMBDA {
          return vecops::vec::add(tag,vecops::vec::mul(tag,value,vecops::vec::fill(tag,2.0f)),vecops::vec::fill(tag,1.0f));
        });
      return vecops::tensor::input<Native>(source,transform);
    } else return vecops::tensor::input<Native>(source);
  }();
  auto target=vecops::tensor::make_tensor(P.data(),vecops::matmul::packed_layout<Atom,side>(source.layout()));
  auto operation=vecops::ops::matmul_pack(vecops::ops::MatmulPackConfig<Atom,side>{});
  vecops::execution::ExecutionSession execution;
  for (std::int64_t iteration=0;iteration<repeat;++iteration) {
    operation(execution,input,vecops::tensor::output<Native>(target));
    asm volatile("" ::: "memory");
  }
}
