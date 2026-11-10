# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Real capture/lowering consumes resource recipes and native workspace plans."""
import gc

import pytest
import torch

from vecops.graph import GraphCapture
from vecops.memory import native as m
from vecops.planner.compiler import BufferRequirement, GraphMemoryCompiler, ProgramCall, requirements_from_graph


def value(name, **fields):
  result = getattr(m, name)()
  for key, item in fields.items():
    setattr(result, key, item)
  return result


@pytest.fixture
def runtime(monkeypatch):
  monkeypatch.setenv('VECOPS_NUM_THREADS', '1')
  monkeypatch.setenv('VECOPS_THREAD_POOL', 'native')
  graph = value('ResourceGraph', agents=[value('Agent', id='executor', capabilities=['host_load_store'], address_space='host')],
    pools=[value('MemoryPool', id='opaque-bank', capacity_bytes=1<<20, address_space='host', granularity_bytes=4096,
                 page_sizes_bytes=[4096], capacity_group='capacity')],
    capacity_groups=[value('CapacityGroup', id='capacity', capacity_bytes=1<<20)],
    access=[value('AccessConnection', id='direct', rule='arbitrary', agent='executor', pool='opaque-bank',
                  access=[m.AccessMode.read,m.AccessMode.write], mapping=m.MappingMode.direct)])
  discovery = value('LinuxDiscovery', graph=graph,
    scope=value('Scope', executable_agents=['executor'], allocatable_pools=['opaque-bank'],
                accessible_pools=['opaque-bank'], execution_slots={'executor':1}),
    pool_bindings=[value('LinuxPoolBinding',pool='opaque-bank',numa_nodes=[])],
    pool_available_bytes={'opaque-bank':1<<20})
  return m.make_linux_runtime(discovery,value('LinuxRuntimeOptions',participant='test'))


def committed(runtime):
  gc.collect()
  view=runtime.view(runtime.current_scope())
  return sum(group.committed_bytes for group in runtime.snapshot(view).groups)


def test_real_native_workspace_describe_compile_activate_and_replay(runtime,tmp_path,monkeypatch):
  import vecops.language as T
  from dsl_test_utils import make_compiler
  compiler=make_compiler(target='Native',jobs=1,cache_dir=tmp_path/'kernel',build_dir=tmp_path/'build')
  source=tmp_path/'resource.cpp'
  source.write_text('''
// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
#define VECOPS_RECORDABLE_KERNEL 1
#include "vecops/Kernel.h"
#include "vecops/execution/KernelPlan.h"
using namespace vecops;
void __kernel__(execution::KernelBuilder& b, auto x, auto y) {
  auto scratch=b.tensor<float>("scratch",tensor::make_shape(meta::cint<64>));
  auto other=b.tensor<float>("scratch",tensor::make_shape(meta::cint<32>));
  b.grid<1>("define",0,64,1,execution::kernel_bind(x,execution::overwrite(scratch)),
    [](auto,auto work,auto x,auto scratch) {scratch(work.begin())=x(work.begin())+1;});
  b.grid<1>("other",0,32,1,execution::kernel_bind(x,execution::overwrite(other)),
    [](auto,auto work,auto x,auto other) {other(work.begin())=x(work.begin())+2;});
  b.grid<1>("read",0,64,1,execution::kernel_bind(execution::read(scratch),execution::read(other),y),
    [](auto,auto work,auto scratch,auto other,auto y) {
      y(work.begin())=scratch(work.begin())*2+other(work.begin()%32);
    });
}
''')
  @T.jit(out_idx=[1],compiler=compiler)
  def factory():
    @T.template(source=str(source),recipe_id='resource-workspace-native-v1')
    def kernel(A:T.Tensor((64,),(1,),T.float32),C:T.Tensor((64,),(1,),T.float32)):
      pass
    return kernel
  kernel=factory()
  torch.set_num_threads(1)
  native_call=kernel.explicit_torch_function()
  def function(x,out):
    native_call(x,out)
    return out
  capture=GraphCapture('resource-native-workspace')
  capture.capture_torch(function,torch.ones(64),torch.empty(64))
  assert committed(runtime)==0
  executable=capture.compile_torch(runtime=runtime)
  prepared=executable.native_memory_program
  assert prepared is not None
  assert any(requirement.id.startswith('kernel:') for requirement in prepared.program.requirements)
  assert committed(runtime)>0
  def forbidden(*args,**kwargs):
    raise AssertionError('Python kernel entered during native replay')
  monkeypatch.setattr(kernel._core,'torch',forbidden)
  x=torch.arange(64,dtype=torch.float32)
  out=torch.empty_like(x)
  executable(x,out)
  torch.testing.assert_close(out,(x+1)*2+(x[:32]+2).repeat(2),rtol=0,atol=0)
  executable(x+3,out)
  torch.testing.assert_close(out,(x+4)*2+(x[:32]+5).repeat(2),rtol=0,atol=0)
  assert not {'prim::PythonOp','prim::CallFunction'}.intersection(executable.ir)
  del prepared,executable
  assert committed(runtime)==0


def test_explicit_resource_program_binds_factory_output_before_native_replay(runtime):
  def function(x):
    out=torch.empty_like(x)
    out.copy_(x)
    return out
  with torch.inference_mode():
    capture=GraphCapture('resource-owned-output')
    capture.capture_torch(function,torch.ones(16))
    view=runtime.view(runtime.current_scope())
    requirements=requirements_from_graph(capture._torch_execution_graph,owned_outputs=True)
    # Borrowed inputs remain external; only the result gets a physical slot.
    engine=GraphMemoryCompiler(runtime,view,runtime.snapshot(view),native_module=m)
    program=engine.compile(requirements)
    prepared=program.prepare(runtime,view)
    borrowed={req.id:torch.ones(16) for req in requirements if not req.allocated}
    prepared.materialize(external=borrowed)
    executable=capture.compile_torch(runtime=runtime,memory_program=prepared)
    output=executable(torch.arange(16,dtype=torch.float32))
    torch.testing.assert_close(output,torch.arange(16,dtype=torch.float32),rtol=0,atol=0)
    assert executable.planned_factory_count>0
    prepared.close()
    del prepared,executable
    assert torch.equal(output,torch.arange(16,dtype=torch.float32))


def test_model_program_call_maps_local_tensor_ids_and_reuses_serial_storage(runtime):
  def function(x):
    out=torch.empty_like(x)
    out.copy_(x)
    return out
  with torch.inference_mode():
    captures=[]
    for name in ('first','second'):
      capture=GraphCapture(name)
      capture.capture_torch(function,torch.ones(16))
      captures.append(capture)
    requirements={str(i):capture.memory_requirements(owned_outputs=True) for i,capture in enumerate(captures)}
    view=runtime.view(runtime.current_scope())
    program=GraphMemoryCompiler(runtime,view,runtime.snapshot(view),native_module=m).compile_model(
        requirements,[ProgramCall('a','0'),ProgramCall('b','1')],serialized=True)
    prepared=program.prepare(runtime,view)
    external={req.id:torch.ones(16) for req in program.requirements if not req.allocated}
    prepared.materialize(external=external)
    executables=[capture.compile_torch(runtime=runtime,memory_program=prepared,memory_call=call)
                 for capture,call in zip(captures,('a','b'))]
    for index,executable in enumerate(executables):
      expected=torch.arange(16,dtype=torch.float32)+index
      torch.testing.assert_close(executable(expected),expected)
    assert sum(program.peak_bytes_by_capacity_group.values())==4096


def test_lowered_plan_separates_managed_factories_from_framework_demands(runtime):
  def function(x):
    temporary=torch.empty_like(x)
    temporary.copy_(x)
    return torch.softmax(temporary,dim=-1)
  with torch.inference_mode():
    capture=GraphCapture('resource-lowered-policy')
    capture.capture_torch(function,torch.ones(16))
    plan=capture.prepare_torch()
    assert committed(runtime)==0
    assert len(plan._tensor_requirements)==1
    assert plan.framework_requirements
    assert {req.id for req in plan.framework_requirements}.isdisjoint(
        req.id for req in plan.managed_requirements)
    view=runtime.view(runtime.current_scope())
    recipe=plan.framework_budget_recipe(view.graph.pools[0],participant=runtime.participant)
    assert recipe.backings and not recipe.mappings and not recipe.addresses.slots
    snapshot=runtime.snapshot(view)
    lease=runtime.reserve(view,recipe,snapshot.epoch)
    assert sum(group.unbacked_reserved_bytes for group in runtime.snapshot(view).groups)==4096
    assert committed(runtime)==0
    del lease
    assert sum(group.unbacked_reserved_bytes for group in runtime.snapshot(view).groups)==0
    executable=plan.compile(runtime=runtime)
    expected=torch.arange(16,dtype=torch.float32)
    torch.testing.assert_close(executable(expected),torch.softmax(expected,dim=-1))
    assert executable.prepared_graph_plan is plan


def test_prepared_pointwise_donation_plan_matches_final_storage_requirements(runtime):
  def function(x):
    a=x+1
    b=a*2
    output=torch.empty_like(b)
    output.copy_(b)
    return output
  with torch.inference_mode():
    capture=GraphCapture('resource-donation-cold-plan')
    capture.capture_torch(function,torch.ones(16))
    original=capture._torch_execution_graph.to_dict()
    plan=capture.prepare_torch(native_pointwise=True,native_donation=True,borrow_outputs=True)
    assert committed(runtime)==0
    assert capture._torch_execution_graph.to_dict()==original
    managed={req.id for req in plan.managed_requirements}
    expected={plan.execution.tensors[key].storage_id for key in plan.scratch_ids}
    assert expected<=managed
    assert not {req.id for req in plan.framework_requirements}&managed
    view=runtime.view(runtime.current_scope())
    program=GraphMemoryCompiler(runtime,view,runtime.snapshot(view),native_module=m).compile(plan.managed_requirements)
    prepared=program.prepare(runtime,view).materialize()
    executable=plan.compile(runtime=runtime,memory_program=prepared)
    assert executable.graph is plan.execution
    for i in range(3):
      x=torch.arange(16,dtype=torch.float32)+i
      torch.testing.assert_close(executable(x),(x+1)*2,rtol=0,atol=0)
    assert set(prepared.program.bindings)==managed


def test_metadata_prepare_precedes_real_payload_or_virtual_binding(runtime):
  from vecops.graph._metadata import metadata_mode
  def function(x):
    output=torch.empty_like(x)
    output.copy_(x)
    return output
  with metadata_mode():
    fake=torch.empty(16)
    capture=GraphCapture('resource-va-prototype')
    capture.capture_torch_metadata(function,fake)
  plan=capture.prepare_torch(borrow_outputs=True)
  assert committed(runtime)==0
  view=runtime.view(runtime.current_scope())
  compiler=GraphMemoryCompiler(runtime,view,runtime.snapshot(view),native_module=m)
  program=compiler.compile(plan.managed_requirements)
  prepared=program.prepare(runtime,view)
  input_program=compiler.compile([BufferRequirement('input-prototype',64)])
  input_va=input_program.prepare(runtime,view)
  prototype=input_va.tensor('input-prototype',(16,),torch.float32)
  assert committed(runtime)==0
  # Bind an already reserved VA prototype only after final offset planning.
  plan.bind_metadata((prototype,))
  prepared.materialize()
  executable=plan.compile(runtime=runtime,memory_program=prepared)
  x=torch.arange(16,dtype=torch.float32)
  torch.testing.assert_close(executable(x),x,rtol=0,atol=0)


def test_metadata_state_source_identities_survive_weak_conversion_cache_collection():
  import gc
  import torch
  from vecops.graph import GraphCapture
  from vecops.graph._metadata import metadata_mode
  class Model(torch.nn.Module):
    def __init__(self):
      super().__init__()
      for index in range(64):self.register_buffer(f'state{index}',torch.ones(index+1))
    def forward(self,x):return x+self.state0
  owner=Model();capture=GraphCapture('many-metadata-state-roots')
  with metadata_mode() as mode:
    capture.capture_torch_metadata(owner,mode.from_tensor(torch.ones(1)))
    gc.collect()
    assert len(capture._metadata_sources)==64
    for _name,real in owner.named_buffers():
      fake=mode.from_tensor(real)
      assert capture._metadata_sources[fake.untyped_storage()._cdata] is real


def test_produced_storage_identity_receipts_belong_to_graph_storage_not_live_token_table():
  import gc
  import torch
  from vecops.graph import GraphCapture
  from vecops.graph._metadata import metadata_mode
  def function(x):
    for index in range(64):result=torch.sin(x)*(index+1)
    return result
  with metadata_mode():
    capture=GraphCapture('historical-storage-receipts')
    capture.capture_torch_metadata(function,torch.empty(7))
  gc.collect()
  allocated={storage.id for storage in capture._torch_execution_graph.storages.values()
             if storage.allocation_event is not None}
  receipts=capture._torch_replay_tape.produced_storage_receipts
  assert set(receipts)==allocated
  assert len({id(receipt) for receipt in receipts.values()})==len(receipts)


def test_metadata_many_equal_shape_weights_bind_correct_values_not_recycled_tokens():
  import torch
  from vecops.graph import GraphCapture
  from vecops.graph._metadata import metadata_mode
  class Model(torch.nn.Module):
    def __init__(self):
      super().__init__()
      for index in range(64):self.register_buffer(f'state{index}',torch.tensor([float(index+1)]))
    def forward(self,x):return x+self.state0+self.state63
  owner=Model();example=torch.tensor([3.0]);capture=GraphCapture('equal-shape-metadata-state')
  with metadata_mode() as mode:capture.capture_torch_metadata(owner,mode.from_tensor(example))
  plan=capture.prepare_torch()
  plan.bind_metadata((example,))
  executable=plan.compile()
  torch.testing.assert_close(executable(example),torch.tensor([68.0]),rtol=0,atol=0)


def test_prepared_placement_query_is_relative_certified_and_detached(runtime):
  view=runtime.view(runtime.current_scope())
  requirements=(BufferRequirement('a',200,lifetimes=((0,1),)),
                BufferRequirement('b',300,lifetimes=((0,1),)),
                BufferRequirement('input',128,boundary='external',ownership='borrowed',external_role='input'))
  program=GraphMemoryCompiler(runtime,view,runtime.snapshot(view)).compile(requirements)
  prepared=program.prepare(runtime,view)
  with pytest.raises(RuntimeError,match='live materialized'):prepared.placement('b')
  prepared.materialize(external={'input':torch.empty(32)})
  endpoint=prepared.placement('b',byte_offset=13,size_bytes=101)
  assert endpoint.address_space=='host'
  assert [(s.offset_bytes,s.size_bytes,s.pool) for s in endpoint.segments]==[(0,101,'opaque-bank')]
  # Mutating a returned observation cannot alter the prepared recipe.
  segment=endpoint.segments[0];segment.pool='untrusted';endpoint.segments=[segment]
  assert prepared.placement('b').segments[0].pool=='opaque-bank'
  with pytest.raises(ValueError,match='borrowed external'):prepared.placement('input')
  with pytest.raises(ValueError,match='exact'):prepared.placement('b',byte_offset=True)
  with pytest.raises(ValueError,match='exceeds'):prepared.placement('b',byte_offset=200,size_bytes=101)
  prepared.close()
  with pytest.raises(RuntimeError,match='live materialized'):prepared.placement('b')


def test_planned_placement_is_available_without_capacity_or_physical_backing(runtime):
  view=runtime.view(runtime.current_scope())
  prepared=GraphMemoryCompiler(runtime,view,runtime.snapshot(view)).compile(
      [BufferRequirement('buffer',300)]).prepare(runtime,view)
  try:
    endpoint=prepared.planned_placement('buffer',byte_offset=17,size_bytes=101)
    assert [(s.offset_bytes,s.size_bytes,s.pool)for s in endpoint.segments]==[(0,101,'opaque-bank')]
    assert prepared.binding is None and prepared.backings is None and prepared._lease is None
    assert sum(g.committed_bytes+g.unbacked_reserved_bytes for g in runtime.snapshot(view).groups)==0
    with pytest.raises(RuntimeError,match='live materialized'):prepared.placement('buffer')
  finally:prepared.close()
  with pytest.raises(RuntimeError,match='live program'):prepared.planned_placement('buffer')


def test_model_transport_provision_deduplicates_without_short_circuit():
  from types import SimpleNamespace
  from vecops.graph import provision_communication_capacity
  calls=[]
  def lease(identity,grows):
    def provision():calls.append(identity);return grows
    return SimpleNamespace(capacity_provider_id=identity,provision_capacity=provision)
  a,b=lease('opaque-a',True),lease('opaque-b',False)
  plans=[SimpleNamespace(communication_providers=lambda:(a,b)),
         SimpleNamespace(communication_providers=lambda:(a,))]
  assert provision_communication_capacity(plans) is True
  assert calls==['opaque-a','opaque-b']



def test_eager_transport_is_provisioned_without_any_captured_graph():
  from types import SimpleNamespace
  from vecops.graph import provision_communication_capacity
  calls=[]
  provider=SimpleNamespace(capacity_provider_id='default-group',
      provision_capacity=lambda:(calls.append('eager') or True))
  assert provision_communication_capacity([],providers=[provider]) is True
  assert calls==['eager']
