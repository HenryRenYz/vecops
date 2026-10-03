// Native-loop baseline, same operation/configuration as TransposeBenchCommon.h.
#include "vecops/Kernel.h"
#include "vecops/ops/Transpose.h"
#include "vecops/execution/ExecutionSession.h"

void __kernel__(vecops::tensor::TensorLike auto A,
                vecops::tensor::TensorLike auto C, std::int64_t repeat) {
  vecops::execution::ExecutionSession execution;
  auto operation = vecops::ops::transpose(vecops::ops::TransposeConfig<vecops::spec::Compute>{});
  for (std::int64_t iteration = 0; iteration < repeat; ++iteration) {
    operation(execution, A, C);
    asm volatile("" ::: "memory");
  }
}
