// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
/** Bounded short penultimate-axis reduction for prepared graph execution.
 * Assign each output row/panel to one task; no shared output initialization,
 * atomic reduction, temporary allocation, or second pool is required.
 * Float accumulation has an explicit 16-value block boundary. Dtype narrowing
 * remains at the original sum output, before any following division/epilogue.
 */
#define VECOPS_RECORDABLE_KERNEL 1
#include "vecops/Kernel.h"
#include "vecops/execution/KernelPlan.h"
#include "vecops/vec/Vec.h"
#include <algorithm>
using namespace vecops;
using Tag=vec::ScalableTag<float32_t,0>;
constexpr auto Parallelism=spec::Parallelism;
void __kernel__(execution::KernelBuilder& builder,auto input,auto,auto output,auto) {
  constexpr int Rank=decltype(input)::Ndim;
  const auto columns=input.template size<Rank-1>();
  const auto count=input.template size<Rank-2>();
  nint_t rows=1;
  for(int axis=0;axis<Rank-2;++axis)rows*=input.size(axis);
  constexpr nint_t Panel=128;
  const auto panels=ceil_div(columns,cint<Panel>);
  builder.grid<Parallelism>("short-axis-sum",nint_t{0},rows*panels,cint<1>,
    execution::kernel_bind(input,output,columns,count,panels),
    [](execution::TaskContext<Parallelism>,const auto& task,auto input,auto output,auto columns,auto count,auto panels) VECOPS_INLINE_LAMBDA_NOEXCEPT {
      const auto row=task.begin()/panels;
      nint_t index=row,offset=0;
      for(int axis=decltype(input)::Ndim-3;axis>=0;--axis){offset+=(index%input.size(axis))*input.stride(axis);index/=input.size(axis);}
      const auto first=(task.begin()%panels)*Panel;
      const auto last=std::min<nint_t>(first+Panel,columns);
      const auto width=vec::size(Tag{});
      for(nint_t column=first;column<last;column+=width){
        auto active=vec::opt::first(std::min<nint_t>(width,last-column));
        auto total=vec::fill(Tag{},0.f);
        for(nint_t start=0;start<count;start+=16){
          auto partial=vec::fill(Tag{},0.f);
          for(nint_t r=start;r<std::min<nint_t>(start+16,count);++r)
            partial=vec::add(Tag{},partial,vec::load_convert(Tag{},input.data()+offset+r*input.stride(decltype(input)::Ndim-2)+column,active,vec::opt::zero));
          total=vec::add(Tag{},total,partial);
        }
        vec::store_convert(Tag{},output.data()+row*columns+column,total,active);
      }
    });
}
