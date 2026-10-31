// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
#define VECOPS_RECORDABLE_KERNEL 1
#include "vecops/execution/KernelPlan.h"
using namespace vecops;
using namespace vecops::execution;

void __kernel__(KernelBuilder& k, auto input, auto output, auto factor, auto add_one) {
  auto temp = k.tensor<spec::ComputeType>("temp", input.layout());
  constexpr nint_t P = static_cast<nint_t>(spec::Parallelism);
  k.grid<P>("scale", meta::cint<0>, input.template size<0>(), meta::cint<1>, kernel_bind(input, temp, factor),
            [](auto, const auto& item, auto input, auto temp, double factor) {
              for (nint_t row = item.begin(); row < item.end(); ++row)
                for (nint_t column = 0; column < input.size(1); ++column)
                  temp(row, column) = input(row, column) * factor;
            });
  k.grid<P>("epilogue", meta::cint<0>, input.template size<0>(), meta::cint<1>, kernel_bind(temp, output, add_one),
            [](auto, const auto& item, auto temp, auto output, bool add_one) {
              for (nint_t row = item.begin(); row < item.end(); ++row)
                for (nint_t column = 0; column < output.size(1); ++column)
                  output(row, column) = temp(row, column) + (add_one ? 1 : 0);
            });
}
