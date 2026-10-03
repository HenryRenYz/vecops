// Paired prepacked baseline. Packing is timed separately, not in this kernel.
#define __kernel__ raw_matmul_reference_kernel
#include "dsl_matmul_reference.cpp"
#undef __kernel__
#include "vecops/matmul/Packing.h"

void __kernel__(vecops::tensor::TensorLike auto A,vecops::tensor::TensorLike auto B,
                vecops::tensor::TensorLike auto AP,vecops::tensor::TensorLike auto BP,
                vecops::tensor::TensorLike auto C,vecops::tensor::TensorLike auto D,std::int64_t repeat) {
  using TA=vecops::spec::NativeA;
  using TB=vecops::spec::NativeB;
  using Atom=std::conditional_t<std::same_as<TA,vecops::bfloat16_t>,vecops::matmul::AMX_BF16F32,
    vecops::matmul::AMX_I8I32<std::conditional_t<std::same_as<TA,vecops::bfloat16_t>,std::int8_t,TA>,
                           std::conditional_t<std::same_as<TB,vecops::bfloat16_t>,std::int8_t,TB>>>;
  using Acc=typename Atom::TAcc;
  auto bt=vecops::tensor::transpose<0,1>(B);
  auto a=[&]() VECOPS_INLINE_LAMBDA {
    if constexpr (vecops::spec::PackA) {
      auto tensor=vecops::tensor::make_tensor(AP.data(),vecops::matmul::packed_layout<Atom,vecops::matmul::Operand::A>(A.layout()));
      return vecops::tensor::input<TA>(tensor);
    } else return benchmark_input<TA>(A);
  }();
  auto b=[&]() VECOPS_INLINE_LAMBDA {
    if constexpr (vecops::spec::PackB) {
      auto tensor=vecops::tensor::make_tensor(BP.data(),vecops::matmul::packed_layout<Atom,vecops::matmul::Operand::B>(bt.layout()));
      return vecops::tensor::input<TB>(tensor);
    } else return benchmark_input<TB>(bt);
  }();
  auto out=benchmark_output<Acc>(C);
  auto operation=vecops::ops::matmul(vecops::ops::MatmulConfig<Atom>{});
  auto m=vecops::tensor::size<0>(A),n=vecops::tensor::size<1>(B),k=vecops::tensor::size<1>(A);
  if constexpr (vecops::spec::Prologue) {
    auto prior_tensor=[&]() VECOPS_INLINE_LAMBDA {
      if constexpr (vecops::spec::Prologue==2)
        return vecops::tensor::make_tensor(D.data(),vecops::tensor::make_layout(
          vecops::tensor::make_shape(m,n),vecops::tensor::make_strides(vecops::cint<0>,vecops::tensor::stride<1>(D))));
      else return D;
    }();
    auto prior=vecops::tensor::input<Acc>(prior_tensor);
    vecops::kernel::Workspace storage(operation.required_workspace(m,n,k,a,b,prior,out));
    auto workspace=storage.view();
    vecops::execution::ExecutionSession execution{workspace};
    for (std::int64_t i=0;i<repeat;++i) {
      operation(execution,m,n,k,a,b,prior,out);
      asm volatile("" ::: "memory");
    }
  } else {
    vecops::kernel::Workspace storage(operation.required_workspace(m,n,k,a,b,out));
    auto workspace=storage.view();
    vecops::execution::ExecutionSession execution{workspace};
    for (std::int64_t i=0;i<repeat;++i) {
      operation(execution,m,n,k,a,b,out);
      asm volatile("" ::: "memory");
    }
  }
}
