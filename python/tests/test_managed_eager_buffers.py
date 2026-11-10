# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Real eager native calls reuse the exact model-planned kernel workspace."""
import gc

import pytest
import torch

from test_graph_resource_program import runtime, committed


def test_eager_registration_uses_planned_addresses_and_rejects_unbound_replay(runtime,tmp_path):
  import vecops.language as T
  from vecops import _schema_bridge
  from vecops.graph.torch import tensor_metadata
  from vecops.planner.compiler import GraphMemoryCompiler, native_workspace_requirements, activate_native_buffers
  from vecops.memory import native
  from dsl_test_utils import make_compiler
  source=tmp_path/'managed-eager.cpp'
  source.write_text('''
// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
#define VECOPS_RECORDABLE_KERNEL 1
#include "vecops/Kernel.h"
#include "vecops/execution/KernelPlan.h"
using namespace vecops;
void __kernel__(execution::KernelBuilder& b,auto x,auto y,auto address) {
  auto temporary=b.tensor<float>("temporary",tensor::make_shape(meta::cint<64>));
  b.grid<1>("define",0,64,1,execution::kernel_bind(x,execution::overwrite(temporary),address),
    [](auto,auto work,auto x,auto temporary,auto address) {
      temporary(work.begin())=x(work.begin())+1;
      if(work.begin()==0) address(0)=reinterpret_cast<int64_t>(temporary.data());
    });
  b.grid<1>("read",0,64,1,execution::kernel_bind(execution::read(temporary),y),
    [](auto,auto work,auto temporary,auto y) { y(work.begin())=temporary(work.begin())*2; });
}
''')
  compiler=make_compiler(target='Native',jobs=1,cache_dir=tmp_path/'cache',build_dir=tmp_path/'build')
  @T.jit(out_idx=[1,2],compiler=compiler)
  def factory():
    @T.template(source=str(source),recipe_id='managed-eager-buffer-authority-v1')
    def kernel(A:T.Tensor((64,),(1,),T.float32), B:T.Tensor((64,),(1,),T.float32),
               address:T.Tensor((1,),(1,),T.int64)):
      pass
    return kernel
  kernel=factory()
  function=kernel.explicit_torch_function()
  torch.set_num_threads(1)
  x=torch.ones(64); out=torch.empty_like(x); address=torch.empty(1,dtype=torch.int64)
  function(x,out,address)  # Materialize the source specialization and bridge.
  assert committed(runtime)==0
  operation,specializations=kernel.explicit_torch_binding()
  lease,_,_,_=_schema_bridge.prepare_replay_plan(operation,(x,out,address,*specializations),tensor_metadata,
                                                describe_only=True,for_eager=True)
  assert lease.buffer_requirements[0]['retained'] is False
  lease.select()
  with pytest.raises(RuntimeError,match='not active'):
    function(x,out,address)
  view=runtime.view(runtime.current_scope())
  requirements=native_workspace_requirements(lease,prefix='uncaptured')
  program=GraphMemoryCompiler(runtime,view,runtime.snapshot(view),native_module=native).compile(requirements)
  prepared=program.prepare(runtime,view)
  prepared.materialize()
  activate_native_buffers(lease,prepared,prefix='uncaptured')
  binding=program.bindings[requirements[0].id]
  expected=prepared.addresses.slot(binding.slot).value+binding.view_offset_bytes
  for increment in (0,3):
    function(x+increment,out,address)
    assert address.item()==expected
    torch.testing.assert_close(out,(x+increment+1)*2,rtol=0,atol=0)
  lease.close()
  prepared.close()
  del lease,prepared
  gc.collect()
  assert committed(runtime)==0


def test_scoped_eager_retained_factory_reinitializes_after_alias_reuse(runtime,tmp_path):
  import ctypes
  import vecops.language as T
  from vecops import _schema_bridge
  from vecops.graph.torch import tensor_metadata
  from vecops.planner.compiler import GraphMemoryCompiler,ScopedNativeLease
  from vecops.memory import native
  from dsl_test_utils import make_compiler
  source=tmp_path/'scoped-retained.cpp'
  source.write_text('''
// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
#define VECOPS_RECORDABLE_KERNEL 1
#include "vecops/Kernel.h"
#include "vecops/execution/KernelPlan.h"
using namespace vecops;
void __kernel__(execution::KernelBuilder& b,auto x,auto y,auto address) {
  auto storage=b.tensor<float>("initialized-state",tensor::make_shape(meta::cint<1>));
  auto state=b.prepare(execution::kernel_bind(storage),[](auto storage) {
    storage(0)=7.f; return storage.data();
  });
  b.grid<1>("read",0,64,1,execution::kernel_bind(state,x,y,address),
    [](auto,auto work,auto state,auto x,auto y,auto address) {
      y(work.begin())=x(work.begin())+state[0];
      if(work.begin()==0) address(0)=reinterpret_cast<int64_t>(state);
    });
}
''')
  build=make_compiler(target='Native',jobs=1,cache_dir=tmp_path/'cache',build_dir=tmp_path/'build')
  @T.jit(out_idx=[1,2],compiler=build)
  def factory():
    @T.template(source=str(source),recipe_id='scoped-eager-initialized-state-v1')
    def kernel(A:T.Tensor((64,),(1,),T.float32),B:T.Tensor((64,),(1,),T.float32),
               address:T.Tensor((1,),(1,),T.int64)):pass
    return kernel
  kernel=factory();function=kernel.explicit_torch_function()
  torch.set_num_threads(1)
  x=torch.arange(64,dtype=torch.float32);out=torch.empty_like(x);address=torch.empty(1,dtype=torch.int64)
  function(x,out,address)
  operation,specs=kernel.explicit_torch_binding()
  lease,_,_,_=_schema_bridge.prepare_replay_plan(operation,(x,out,address,*specs),tensor_metadata,
      describe_only=True,for_eager=True)
  assert lease.buffer_requirements[0]['retained'] is True
  class OtherPhase:
    buffer_requirements=lease.buffer_requirements
    def activate_buffers(self,mapping,owner):
      self.owner=owner
      self.address=next(iter(mapping.values()))['address']
      ctypes.c_float.from_address(self.address).value=666
    def suspend_buffers(self):self.owner=None
  other=OtherPhase()
  first=ScopedNativeLease(lease,prefix='first',lifetime=(0,1))
  second=ScopedNativeLease(other,prefix='second',lifetime=(1,2))
  view=runtime.view(runtime.current_scope())
  program=GraphMemoryCompiler(runtime,view,runtime.snapshot(view),native_module=native).compile(
      first.requirements()+second.requirements())
  prepared=program.prepare(runtime,view).materialize()
  first.activate(prepared)
  function(x,out,address)
  torch.testing.assert_close(out,x+7,rtol=0,atol=0)
  first.suspend()
  lease.select()
  with pytest.raises(RuntimeError,match='not active'):function(x,out,address)
  second.activate(prepared)
  assert other.address==address.item() and ctypes.c_float.from_address(other.address).value==666
  second.suspend()
  first.activate(prepared)
  function(x,out,address)
  torch.testing.assert_close(out,x+7,rtol=0,atol=0)
  first.suspend();lease.close();prepared.close()
  del lease,prepared
  gc.collect()
  assert committed(runtime)==0
