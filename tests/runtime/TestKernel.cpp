/** @file TestKernel.cpp @brief Minimal compiler-managed `__kernel__` source. */

#include "vecops/Kernel.h"

using namespace vecops;

constexpr auto B = spec::B;
constexpr auto D = spec::D;
constexpr auto stride_D = spec::stride_D;
using ComputeType = spec::ComputeType;

void __kernel__(TensorLike auto input, TensorLike auto output, float64_t factor, bool add_one) {
  assert_shape(input, {B, D}, {stride_D, cint<1>});
  assert_shape(output, {B, D}, {any, cint<1>});
  for (nint_t row = 0; row < B; ++row) {
    for (nint_t column = 0; column < D; ++column) {
      output(row, column) = static_cast<ComputeType>(input(row, column) * factor + (add_one ? 1.0 : 0.0));
    }
  }
}
