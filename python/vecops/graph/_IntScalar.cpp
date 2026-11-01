// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
// Scalar integer control in a resident tape; preserve signed wrap exactly.
#define VECOPS_RECORDABLE_KERNEL 1
#include "vecops/Kernel.h"
#include "vecops/execution/KernelPlan.h"
#include <bit>
#include <cstdint>
using namespace vecops;
constexpr auto Parallelism=spec::Parallelism;
void __kernel__(execution::KernelBuilder& builder,auto input,auto,auto output,auto offset) {
  builder.grid<Parallelism>("integer-scalar-add",nint_t{0},cint<1>,cint<1>,
      execution::kernel_bind(input,output,offset),
      [](execution::TaskContext<Parallelism>,const auto&,auto input,auto output,int64_t offset) VECOPS_INLINE_LAMBDA_NOEXCEPT {
        const auto bits=std::bit_cast<uint64_t>(static_cast<int64_t>(*input.data()))+
                        std::bit_cast<uint64_t>(offset);
        *output.data()=std::bit_cast<int64_t>(bits);
      });
}
