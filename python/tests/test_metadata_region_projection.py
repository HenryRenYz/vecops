# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""A regional executable is a view of the first semantic trace, not a rerun."""
import copy
from contextlib import nullcontext
import pytest
import torch
from vecops.graph import GraphCapture
from vecops.graph.torch import register_inputs, register_module_state, record_torch
from vecops.graph._metadata import (metadata_mode, MetadataRegionProjection,
                                   bind_projected_sources, bind_metadata)
from test_graph_resource_program import runtime as resource_runtime


@pytest.mark.parametrize('inference', [False, True])
def test_projected_binding_preserves_source_inference_mode_and_storage_aliases(inference):
  class Owner(torch.nn.Module):
    def forward(self,x):return x*self.weight+1
  owner=Owner()
  with torch.inference_mode() if inference else nullcontext():
    base=torch.arange(64,dtype=torch.float32,requires_grad=not inference)
    owner.register_buffer('weight',base[3:27:2])
  x=torch.arange(12,dtype=torch.float32)
  parent=GraphCapture('inference-source-rebinding')
  with torch.inference_mode(),metadata_mode()as mode:
    fake=mode.from_tensor(x)
    with parent:
      register_inputs(parent,(fake,));register_module_state(parent,owner,fake_mode=mode)
      with record_torch(parent):
        projection=MetadataRegionProjection(parent,owner,(fake,))
        capture=projection.finish(owner(fake))
    parent.finish()
  # Relocation changes the storage, preserving a noncontiguous source and a
  # nonzero storage offset. Bind outside inference mode, as AF3 preparation does.
  with torch.inference_mode() if inference else nullcontext():
    replacement=torch.arange(64,dtype=torch.float32,requires_grad=not inference)*2
    owner.weight=replacement[3:27:2]
  assert not torch.is_inference_mode_enabled()
  bind_projected_sources(capture,(x,))
  tape=capture._torch_replay_tape
  assert tape._region_source_bindings
  for key in tape._region_source_bindings:
    alias=tape.tensors[key]
    assert torch._C._is_alias_of(alias,owner.weight)
    assert alias.shape==owner.weight.shape and alias.stride()==(2,)
    assert alias.storage_offset()==3
    assert torch.is_inference(alias)==inference
    assert alias.requires_grad==(not inference)
  bind_metadata(capture,(x,))
  assert not torch.is_inference_mode_enabled()


def test_projection_rebinds_relocated_state_without_a_second_forward():
  class Owner(torch.nn.Module):
    def __init__(self):
      super().__init__(); self.weight=torch.nn.Parameter(torch.arange(4,dtype=torch.float32)+2)
      self.calls=0
    def forward(self,x):
      self.calls+=1
      return x*self.weight+1
  owner=Owner(); x=torch.arange(4,dtype=torch.float32)+1
  parent=GraphCapture('whole')
  with torch.inference_mode(), metadata_mode() as mode:
    fake=mode.from_tensor(x)
    with parent:
      register_inputs(parent,(fake,)); register_module_state(parent,owner,fake_mode=mode)
      with record_torch(parent):
        projection=MetadataRegionProjection(parent,owner,(fake,),name='first-pass')
        result=owner(fake)
        capture=projection.finish(result)
        # Later events may extend/mutate parent liveness, never the snapshot.
        snapshot=copy.deepcopy(capture._torch_execution_graph.to_dict())
        discarded=result.sin()
    whole=parent.finish()
  assert owner.calls==1
  assert snapshot==capture._torch_execution_graph.to_dict()
  assert len(capture._torch_replay_tape.instructions)==2
  assert len(whole.events)>len(capture._torch_execution_graph.events)
  arena=torch.empty(20);arena[7:11].copy_(torch.tensor([10.,20.,30.,40.]))
  owner.weight=torch.nn.Parameter(arena[7:11])
  with torch.inference_mode():
    bind_projected_sources(capture,(x,))
    bind_metadata(capture,(x,))
    executable=capture.compile_torch(borrow_readonly_state=True)
    actual=executable(x)
  torch.testing.assert_close(actual,torch.tensor([11.,41.,91.,161.]),rtol=0,atol=0)
  assert owner.calls==1


def test_projection_rejects_overlapping_capture_and_changed_state():
  owner=torch.nn.Linear(4,4,bias=False);x=torch.ones(1,4)
  parent=GraphCapture('whole')
  with torch.inference_mode(),metadata_mode()as mode:
    fake=mode.from_tensor(x)
    with parent:
      register_inputs(parent,(fake,));register_module_state(parent,owner,fake_mode=mode)
      with record_torch(parent):
        p=MetadataRegionProjection(parent,owner,(fake,))
        with pytest.raises(ValueError,match='overlap'):
          MetadataRegionProjection(parent,owner,(fake,))
        result=owner(fake);capture=p.finish(result)
    parent.finish()
  owner.weight=torch.nn.Parameter(torch.ones(4,5))
  with pytest.raises(ValueError,match='geometry changed'):
    bind_projected_sources(capture,(x,))


def test_projection_keeps_sampled_loop_timeline_dense_after_factory_filtering():
  from vecops.planner.core import storage_traffic
  owner=torch.nn.Identity();x=torch.arange(4,dtype=torch.float32)
  parent=GraphCapture('whole-with-sampled-loop')
  with torch.inference_mode(),metadata_mode()as mode:
    fake=mode.from_tensor(x)
    with parent:
      register_inputs(parent,(fake,))
      with record_torch(parent):
        ignored=fake+3 # The region also starts at a nonzero parent index.
        projection=MetadataRegionProjection(parent,owner,(fake,))
        loop=parent.declare_loop('sampled-layers',4)
        result=fake
        for index in (0,1):
          with loop.iteration(index,inputs=(result,))as state:
            temporary=torch.empty_like(result)
            temporary.copy_(result+1)
            result=temporary*2
            state['outputs']=(result,)
        loop.close()
        parent_indexes=[event.index for event in parent.events]
        capture=projection.finish(result)
    whole=parent.finish()
  regional=capture._torch_execution_graph
  assert [event.index for event in regional.events]==list(range(1,len(regional.events)+1))
  assert len(whole.events)>len(regional.events)
  assert len(regional.loops)==1 and regional.loops[0].observed_iterations==[0,1]
  assert storage_traffic(regional)
  # Filtering/reindexing the regional view must never change its parent.
  assert [event.index for event in whole.events]==parent_indexes
  with torch.inference_mode():
    bind_projected_sources(capture,(x,))
    bind_metadata(capture,(x,))
    executable=capture.compile_torch(native_pointwise=True,borrow_outputs=True,native_donation=True)
    torch.testing.assert_close(executable(x),x*4+6,rtol=0,atol=0)


def test_projection_preserves_region_owned_routed_output_storage(resource_runtime):
  from torch.utils._python_dispatch import TorchDispatchMode
  class Router(TorchDispatchMode):
    def __torch_dispatch__(self,function,types,args=(),kwargs=None):
      if function is torch.ops.aten.neg.default:
        return torch.empty_like(args[0])
      return function(*args,**(kwargs or {}))
  owner=torch.nn.Identity();x=torch.arange(4,dtype=torch.float32)
  parent=GraphCapture('routed-output-ownership')
  with torch.inference_mode(),metadata_mode()as mode:
    fake=mode.from_tensor(x)
    with parent:
      register_inputs(parent,(fake,))
      with record_torch(parent),Router():
        projection=MetadataRegionProjection(parent,owner,(fake,))
        with record_torch(parent):
          result=-fake
          capture=projection.finish(result)
    whole=parent.finish()
  regional=capture._torch_execution_graph
  output=regional.tensors[regional.outputs[0]]
  storage=regional.storages[output.storage_id]
  assert not storage.external
  assert storage.allocation_event==output.producer_event
  assert storage.allocation_event in {event.id for event in regional.events}
  # The router's private factory is represented by the semantic outer ATen op.
  # Both views must preserve its fresh storage and the same producer relation.
  assert [event.name for event in whole.events] == ['aten.neg.default']
  original=whole.storages[output.storage_id]
  assert not original.external and original.allocation_event==storage.allocation_event
  assert output.id in whole.events[0].outputs
  assert whole.tensors[output.id].producer_event==original.allocation_event
  view=resource_runtime.view(resource_runtime.current_scope())
  def committed():
    return sum(group.committed_bytes for group in resource_runtime.snapshot(view).groups)
  assert committed()==0
  with torch.inference_mode():
    bind_projected_sources(capture,(x,));bind_metadata(capture,(x,))
    executable=capture.compile_torch(runtime=resource_runtime,native_pointwise=True,
        borrow_outputs=True,native_donation=True)
    assert committed()==4096
    result=executable(x)
    assert not torch._C._is_alias_of(result,x)
    torch.testing.assert_close(result,-x,rtol=0,atol=0)


def test_projection_records_unpublished_native_source_schema(monkeypatch):
  from types import SimpleNamespace
  import vecops
  from vecops import _precompile,_schema_bridge
  from vecops.graph import current_capture
  from vecops.graph._metadata import resolve_projected_functions
  from vecops.graph._replay import _reject_tensor_data_escapes
  definition=vecops.KernelDef('source',[
      vecops.TensorDef('A',[4],[1],dtype=vecops.float32),
      vecops.TensorDef('C',[4],[1],dtype=vecops.float32,access=vecops.TensorAccess.output),
  ])._native()
  registration=SimpleNamespace(library='projection_pending',name='source',source=None,
      kernel_def=definition,spec_order=None,return_outputs=False)
  wrapper=_schema_bridge._public_wrapper('projection_pending','source',definition,
      SimpleNamespace(_handle=123),None,registration,[False],[None])
  monkeypatch.setattr(_precompile,'compile_batch',lambda requests,**kwargs:
      _precompile.PrecompileResult(len(requests),len(requests),1,0.))
  monkeypatch.setattr(_schema_bridge,'compile_pending_torch_bridges',lambda **kwargs:0)
  owner=torch.nn.Identity();observed={}
  def model(x):
    parent=current_capture()
    p=MetadataRegionProjection(parent,owner,(x,));observed['projection']=p
    with record_torch(parent):
      with _reject_tensor_data_escapes():
        result=wrapper(x,torch.empty_like(x))
      observed['capture']=p.finish(result)
    return result
  def observe(reg,args):observed['projection'].record_native(reg,args)
  whole=vecops.precompile(model,torch.ones(4),capture_graph=True,
                          _torch_call_observer=observe,_graph_state_owner=owner)
  tape=observed['capture']._torch_replay_tape
  assert [x.function._schema.name for x in tape.instructions]==['aten::empty_like','projection_pending::source']
  assert tape.instructions[-1].result is None
  assert whole.prepared==1
  library=torch.library.Library('projection_pending','DEF')
  library.define(str(tape.instructions[-1].function._schema).split('::',1)[1])
  resolve_projected_functions(observed['capture'])
  assert tape.instructions[-1].function is torch.ops.projection_pending.source.default


def test_projection_records_router_outer_operator_once(monkeypatch):
  from types import SimpleNamespace
  import vecops
  from vecops import _precompile,_schema_bridge
  from vecops.graph import current_capture
  from torch.utils._python_dispatch import TorchDispatchMode
  definition=vecops.KernelDef('route',[
      vecops.TensorDef('A',[4],[1],dtype=vecops.float32),
      vecops.TensorDef('C',[4],[1],dtype=vecops.float32,access=vecops.TensorAccess.output),
  ])._native()
  reg=SimpleNamespace(library='projection_route',name='route',source=None,
      kernel_def=definition,spec_order=None,return_outputs=False)
  wrapper=_schema_bridge._public_wrapper('projection_route','route',definition,
      SimpleNamespace(_handle=124),None,reg,[False],[None])
  monkeypatch.setattr(_precompile,'compile_batch',lambda requests,**kwargs:
      _precompile.PrecompileResult(len(requests),len(requests),1,0.))
  monkeypatch.setattr(_schema_bridge,'compile_pending_torch_bridges',lambda **kwargs:0)
  class Router(TorchDispatchMode):
    def __torch_dispatch__(self,operation,types,args=(),kwargs=None):
      if operation is torch.ops.aten.neg.default:
        return wrapper(args[0],torch.empty_like(args[0]))
      return operation(*args,**(kwargs or {}))
  owner=torch.nn.Identity();seen={}
  def model(x):
    parent=current_capture()
    with Router():
      p=MetadataRegionProjection(parent,owner,(x,));seen['projection']=p
      with record_torch(parent):
        result=-x;seen['capture']=p.finish(result)
    return result
  def observe(reg,args):seen['projection'].record_native(reg,args)
  value=torch.arange(4,dtype=torch.float32)+1
  result=vecops.precompile(model,value,capture_graph=True,_graph_state_owner=owner,
                           _torch_call_observer=observe)
  capture=seen['capture']
  assert result.prepared==1 # the routed native request is still compiled
  assert [x.function._schema.name for x in capture._torch_replay_tape.instructions]==['aten::neg']
  with torch.inference_mode():
    bind_metadata(capture,(value,))
    executable=capture.compile_torch()
    torch.testing.assert_close(executable(value),-value,rtol=0,atol=0)
