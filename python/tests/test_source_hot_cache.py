# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Repeating source signatures and workspace-context replacement."""
import pytest
import torch
import vecops
import vecops.language as T


@pytest.fixture(scope='module')
def source_call(tmp_path_factory):
  path = tmp_path_factory.mktemp('source-hot-cache')/'kernel.cpp'
  path.write_text('''
// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
#define VECOPS_RECORDABLE_KERNEL 1
#include "vecops/Kernel.h"
#include "vecops/execution/KernelPlan.h"
using namespace vecops;
using namespace vecops::tensor;
constexpr nint_t P = spec::Parallelism;
void __kernel__(execution::KernelBuilder& b, auto x, auto y) {
  auto temp = b.tensor<float32_t>("temp", make_shape(x.size(0)));
  b.grid<P>("first", nint_t{0}, x.size(0), cint<1>, execution::kernel_bind(x, temp),
      [](execution::TaskContext<P>, const auto& work, auto x, auto temp) {
        auto i = work.begin(); temp.data()[i] = x.data()[i] + 1.f;
      });
  b.grid<P>("second", nint_t{0}, x.size(0), cint<1>, execution::kernel_bind(temp, y),
      [](execution::TaskContext<P>, const auto& work, auto temp, auto y) {
        auto i = work.begin(); y.data()[i] = temp.data()[i] * 2.f;
      });
}
''')
  @T.jit(out_idx=[1])
  def factory():
    N = T.dynamic(lower=1)
    @T.template(source=str(path), recipe_id='source-hot-cache-context-test')
    def kernel(x: T.Tensor((N,), (1,), T.float32),
               y: T.Tensor((N,), (1,), T.float32)):
      pass
    return kernel
  torch.set_num_threads(4)
  compiled=factory()
  invoke=compiled.explicit_torch_function()
  # Make every test independently describable; bridge preparation is not a
  # side effect supplied by another test's ordering.
  invoke(torch.arange(4.),torch.empty(4))
  class SourceCall:
    def __call__(self,*args):return invoke(*args)
    def operation(self):return compiled.explicit_torch_binding()[0]
  return SourceCall()


def test_more_than_eight_repeating_metadata_signatures_keep_prepared_plans(source_call):
  arrays = [(torch.arange(float(n)), torch.empty(n)) for n in range(17, 31)]
  with torch.inference_mode():
    for x,y in arrays:
      source_call(x,y)
    vecops._C.reset_bridge_hot_cache_stats()
    for x,y in arrays:
      source_call(x,y)
      torch.testing.assert_close(y,(x+1)*2,rtol=0,atol=0)
    stats = vecops._C.bridge_hot_cache_stats()
    if stats['enabled'] and stats['capacity'] >= len(arrays):
      assert stats['hits'] == len(arrays)
      assert stats['prepares'] == 0 and stats['evictions'] == 0


def _described_eager(source_call,x,y):
  from types import SimpleNamespace
  from vecops import _schema_bridge
  from vecops.graph.torch import tensor_metadata
  descriptors=tuple(SimpleNamespace(metadata=tensor_metadata(value)) for value in (x,y))
  prepared=_schema_bridge.prepare_replay_plan(source_call.operation(),descriptors,lambda value:value.metadata,
      describe_only=True,for_eager=True)
  assert prepared is not None
  prepared[0].model_retained=True
  return prepared[0]


def _bind_eager(lease,runtime):
  from vecops import planner
  view=runtime.view(runtime.current_scope())
  requirements=planner.native_workspace_requirements(lease,prefix='native')
  program=planner.GraphMemoryCompiler(runtime,view,runtime.snapshot(view),native_module=vecops.memory.native).compile(requirements)
  prepared=program.prepare(runtime,view).materialize()
  planner.activate_native_buffers(lease,prepared,prefix='native')
  return prepared


def test_identical_metadata_can_adopt_a_new_resource_program(source_call):
  from test_graph_replay import _resource_runtime,_committed
  x,y=torch.arange(37.),torch.empty(37)
  runtime=_resource_runtime()
  with torch.inference_mode():
    for offset in range(2):
      lease=_described_eager(source_call,x,y)
      prepared=_bind_eager(lease,runtime)
      source_call(x+offset,y)
      torch.testing.assert_close(y,(x+offset+1)*2,rtol=0,atol=0)
      lease.close();prepared.close()
      del lease,prepared
      assert _committed(runtime)==0


def test_native_workspace_and_eager_tensor_use_one_capacity_authority(source_call):
  from test_graph_replay import _resource_runtime,_committed
  from vecops.memory import Runtime,Pool
  runtime=_resource_runtime();view=runtime.view(runtime.current_scope())
  wrapped=Runtime(runtime)
  arena=wrapped.create_arena('eager',allowed_pools=['uma'],hard_limit=1<<20,
      grow_by_bytes=4096,retain_idle_bytes=0,page_bytes=4096)
  pool=Pool(arena)
  held=pool.empty((32,),dtype=torch.float32).fill_(19)
  x,y=torch.arange(37.),torch.empty(37)
  with torch.inference_mode():
    lease=_described_eager(source_call,x,y)
    prepared=_bind_eager(lease,runtime)
    assert _committed(runtime)==8192
    for offset in range(3):
      source_call(x+offset,y)
      torch.testing.assert_close(y,(x+offset+1)*2,rtol=0,atol=0)
      torch.testing.assert_close(held,torch.full_like(held,19),rtol=0,atol=0)
      assert _committed(runtime)==8192
    lease.close();prepared.close()
  del lease,prepared,held
  import gc
  gc.collect();pool.trim(0)
  assert _committed(runtime)==0


def test_ordinary_native_eager_and_pool_share_the_active_runtime(source_call):
  from test_graph_replay import _resource_runtime,_committed
  from vecops.memory import Runtime, Pool, use_runtime
  import gc
  runtime=_resource_runtime()
  arena=Runtime(runtime).create_arena('eager',allowed_pools=['uma'],hard_limit=1<<20,
      grow_by_bytes=4096,retain_idle_bytes=0,page_bytes=4096)
  held=Pool(arena).empty((32,),dtype=torch.float32).fill_(19)
  x,y=torch.arange(37.),torch.empty(37)
  with use_runtime(runtime),torch.inference_mode():
    source_call(x,y)
    assert _committed(runtime)==8192
    source_call(x+1,y)
    assert _committed(runtime)==8192
    torch.testing.assert_close(y,(x+2)*2,rtol=0,atol=0)
    torch.testing.assert_close(held,torch.full_like(held,19),rtol=0,atol=0)
    vecops._C._clear_eager_workspace_cache()
    assert _committed(runtime)==4096
    source_call(x,y)
    assert _committed(runtime)==8192
  assert _committed(runtime)==4096
  del held;gc.collect();arena.trim(0)
  assert _committed(runtime)==0


def test_exactly_full_native_workspace_can_replay_and_scope_switch_refunds(source_call):
  from test_graph_replay import _resource_runtime,_committed
  from vecops.memory import use_runtime
  first,second=_resource_runtime(),_resource_runtime()
  x,y=torch.arange(float((1<<20)//4)),torch.empty((1<<20)//4)
  with use_runtime(first),torch.inference_mode():
    source_call(x,y)
    assert _committed(first)==1<<20
    source_call(x+1,y)
    torch.testing.assert_close(y,(x+2)*2,rtol=0,atol=0)
    with use_runtime(second):
      assert _committed(first)==0
      source_call(x,y)
      assert _committed(second)==1<<20
    assert _committed(second)==0
    source_call(x,y)
    assert _committed(first)==1<<20
  assert _committed(first)==_committed(second)==0


def test_ordinary_native_eager_does_not_fall_back_around_capacity(source_call):
  from test_graph_replay import _resource_runtime,_committed
  from vecops.memory import use_runtime
  runtime=_resource_runtime()
  x,y=torch.ones((1<<20)//4+1),torch.empty((1<<20)//4+1)
  with use_runtime(runtime),torch.inference_mode():
    with pytest.raises(RuntimeError,match='capacity|workspace|placement|arena|fit|allocation'):
      source_call(x,y)
    assert _committed(runtime)==0
    source_call(x[:37],y[:37])
    torch.testing.assert_close(y[:37],torch.full_like(y[:37],4),rtol=0,atol=0)
    assert _committed(runtime)==4096
  assert _committed(runtime)==0


def test_native_runtime_scopes_require_lifo_on_the_calling_thread():
  from test_graph_replay import _resource_runtime,_committed
  from concurrent.futures import ThreadPoolExecutor
  native=vecops.memory.native
  runtime=_resource_runtime()
  first=native._WorkspaceRuntimeScope(runtime)
  second=native._WorkspaceRuntimeScope(runtime)
  try:
    with pytest.raises(RuntimeError,match='LIFO'):
      first.close()
    with ThreadPoolExecutor(1) as executor:
      with pytest.raises(RuntimeError,match='entering thread'):
        executor.submit(second.close).result()
  finally:
    second.close();first.close()
  assert _committed(runtime)==0


def test_identical_eager_signatures_switch_between_live_program_owners(source_call):
  from test_graph_replay import _resource_runtime,_committed
  runtime=_resource_runtime()
  x,y=torch.arange(37.),torch.empty(37)
  first=_described_eager(source_call,x,y)
  first_program=_bind_eager(first,runtime)
  second=_described_eager(source_call,x,y)
  second_program=_bind_eager(second,runtime)
  try:
    assert _committed(runtime)==8192
    for selected,other,offset in [(first,second,3),(second,first,7),(first,second,11)]:
      selected.select()
      other.unselect()
      selected_program=first_program if selected is first else second_program
      other_program=second_program if selected is first else first_program
      symbol='native/'+selected.buffer_requirements[0]['name']
      active=selected_program.tensor(symbol,(37,),torch.float32)
      inactive=other_program.tensor(symbol,(37,),torch.float32)
      active.fill_(-777);inactive.fill_(-777)
      source_call(x+offset,y)
      torch.testing.assert_close(y,(x+offset+1)*2,rtol=0,atol=0)
      torch.testing.assert_close(active,x+offset+1,rtol=0,atol=0)
      torch.testing.assert_close(inactive,torch.full_like(inactive,-777),rtol=0,atol=0)
      del active,inactive
      assert _committed(runtime)==8192
    # Closing one owner leaves the other selectable without reallocating it.
    first.close();first_program.close()
    second.select();source_call(x,y)
    torch.testing.assert_close(y,(x+1)*2,rtol=0,atol=0)
    assert _committed(runtime)==4096
  finally:
    first.close();second.close()
    first_program.close();second_program.close()
  assert _committed(runtime)==0
