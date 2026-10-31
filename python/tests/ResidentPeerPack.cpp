// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
// Test reader: a deliberately delayed rank checks snapshot retirement across
// repeated equal-shaped collectives, using the ordinary recordable ABI.
#define VECOPS_RECORDABLE_KERNEL 1
#include "vecops/Kernel.h"
#include "vecops/execution/KernelPlan.h"
#include <chrono>
#include <thread>
#include <stdexcept>
using namespace vecops;
constexpr nint_t P = spec::Parallelism;
void __kernel__(execution::KernelBuilder& builder, auto pointers, auto output) {
  builder.single("delayed_reader",execution::kernel_bind(),[] {
    if constexpr (static_cast<bool>(spec::Delay))
      std::this_thread::sleep_for(std::chrono::milliseconds(2));
  });
  builder.grid<P>("read_snapshots",nint_t{0},output.size(0),meta::cint<1>,
    execution::kernel_bind(pointers,output),
    [](execution::TaskContext<P>,const auto& item,auto pointers,auto output) {
      const auto peer=item.begin();
      const auto* source=reinterpret_cast<const float*>(static_cast<uintptr_t>(pointers.data()[peer]));
      if constexpr (static_cast<bool>(spec::Delay))
        if(output.size(1)>0 && source[0]<0)
          throw std::runtime_error("injected snapshot reader failure");
      for(nint_t i=0;i<output.size(1);++i)output.data()[peer*output.stride(0)+i]=source[i];
    });
}
