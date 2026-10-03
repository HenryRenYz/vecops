// Paired baseline: public Matmul configuration, transforms and reusable scope.
#include "vecops/Kernel.h"
#include "vecops/ops/Matmul.h"
#include "vecops/kernel/Workspace.h"
#include "vecops/execution/ExecutionSession.h"

template<class Compute, class Tensor>
auto benchmark_input(const Tensor& tensor) {
  if constexpr (vecops::spec::InputScale == 1) {
    return vecops::tensor::input<Compute>(tensor);
  } else {
    auto transform=vecops::tensor::make_elementwise_vec_transform<vecops::float32_t,vecops::float32_t>(
      [](auto tag,auto value) VECOPS_KERNEL_LAMBDA {
        return vecops::vec::add(tag,vecops::vec::mul(tag,value,vecops::vec::fill(tag,2.0f)),vecops::vec::fill(tag,1.0f));
      });
    return vecops::tensor::input<Compute>(tensor,transform);
  }
}

template<class Acc,class Tensor>
auto benchmark_output(const Tensor& tensor) {
  if constexpr (vecops::spec::Epilogue == 0) {
    return vecops::tensor::output<Acc>(tensor);
  } else {
    auto transform=vecops::tensor::make_elementwise_vec_transform<vecops::float32_t,vecops::float32_t>(
      [](auto tag,auto value) VECOPS_KERNEL_LAMBDA {
        if constexpr (vecops::spec::Epilogue == 1)
          return vecops::vec::max(tag,value,vecops::vec::zeros(tag));
        else if constexpr (vecops::spec::Epilogue == 2)
          return vecops::vec::min(tag,vecops::vec::max(tag,value,vecops::vec::fill(tag,-0.25f)),vecops::vec::fill(tag,0.25f));
        else if constexpr (vecops::spec::Epilogue == 3)
          return vecops::vec::mul(tag,value,vecops::vec::fill(tag,0.5f));
        else
          return vecops::vec::div(tag,value,vecops::vec::add(tag,vecops::vec::fill(tag,1.0f),vecops::vec::exp(tag,vecops::vec::neg(tag,value))));
      });
    return vecops::tensor::output<Acc>(tensor,transform);
  }
}

void __kernel__(vecops::tensor::TensorLike auto A,vecops::tensor::TensorLike auto B,
                vecops::tensor::TensorLike auto C,vecops::tensor::TensorLike auto D,
                std::int64_t repeat) {
  using TA=vecops::spec::NativeA;
  using TB=vecops::spec::NativeB;
  using Atom=std::conditional_t<std::same_as<TA,vecops::bfloat16_t>,vecops::matmul::AMX_BF16F32,
    vecops::matmul::AMX_I8I32<std::conditional_t<std::same_as<TA,vecops::bfloat16_t>,std::int8_t,TA>,
                           std::conditional_t<std::same_as<TB,vecops::bfloat16_t>,std::int8_t,TB>>>;
  using Acc=typename Atom::TAcc;
  auto a=benchmark_input<TA>(A);
  auto b=benchmark_input<TB>(vecops::tensor::transpose<0,1>(B));
  auto out=benchmark_output<Acc>(C);
  auto operation=vecops::ops::matmul(vecops::ops::MatmulConfig<Atom>{});
  const auto m=vecops::tensor::size<0>(A);
  const auto n=vecops::tensor::size<1>(B);
  const auto k=vecops::tensor::size<1>(A);
  if constexpr (vecops::spec::Prologue) {
    auto prior_tensor = [&]() VECOPS_INLINE_LAMBDA {
      if constexpr (vecops::spec::Prologue == 2) {
        return vecops::tensor::make_tensor(D.data(), vecops::tensor::make_layout(
          vecops::tensor::make_shape(m,n),
          vecops::tensor::make_strides(vecops::cint<0>,vecops::tensor::stride<1>(D))));
      } else {
        return D;
      }
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
