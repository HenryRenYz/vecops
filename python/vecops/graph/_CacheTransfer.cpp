// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
/** @file _CacheTransfer.cpp @brief Prepared, bit-preserving CPU tier transfer. */
#define VECOPS_RECORDABLE_KERNEL 1
#include "vecops/Kernel.h"
#include "vecops/execution/KernelPlan.h"
#include "vecops/vec/Vec.h"
#include <algorithm>
using namespace vecops;
void __kernel__(execution::KernelBuilder& builder, auto source, auto target) {
  // The process provider may have fewer lanes (e.g. an ATen-only backend).
  // TransferWorkers bounds this grid; it never enlarges the provider contract.
  constexpr nint_t P = std::min(static_cast<nint_t>(spec::Parallelism),
                                 static_cast<nint_t>(spec::TransferWorkers));
  builder.grid<P>("transfer", nint_t{0}, source.size(0), meta::cint<65536>,
    execution::kernel_bind(execution::read(source), execution::write(target)),
    [](auto, const auto& item, auto source, auto target) VECOPS_INLINE_LAMBDA_NOEXCEPT {
      using Tag = vec::ScalableTag<uint8_t,0>;
      const auto tag = Tag{};
      const nint_t width = vec::size(tag);
      nint_t i = item.begin();
      for (; i + 4*width <= item.end(); i += 4*width) {
        auto a = vec::load(tag,source.data()+i);
        auto b = vec::load(tag,source.data()+i+width);
        auto c = vec::load(tag,source.data()+i+2*width);
        auto d = vec::load(tag,source.data()+i+3*width);
        vec::store(tag,target.data()+i,a); vec::store(tag,target.data()+i+width,b);
        vec::store(tag,target.data()+i+2*width,c); vec::store(tag,target.data()+i+3*width,d);
      }
      for (; i < item.end(); i += width) {
        const auto active = vec::opt::first(std::min(width,item.end()-i));
        auto value = vec::load(tag,source.data()+i,active,vec::opt::zero);
        vec::store(tag,target.data()+i,value,active);
      }
    });
}
