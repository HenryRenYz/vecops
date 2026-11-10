# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Cold consumer queries preserve logical groups and never create native plans."""
from types import SimpleNamespace

import pytest


def test_deferred_and_bound_consumers_share_address_free_inventory(monkeypatch):
  from vecops.distributed import _template,tuning
  from vecops.graph._replay import _Instruction,_Ref
  from vecops.graph.ir import TensorMetadata
  group=object();calls=[]
  recipe=dict(operation='vecops_comm::consume',operator_name='opaque::contraction',specs=[8,36])
  monkeypatch.setattr(_template,'group_for',lambda value:group)
  monkeypatch.setattr(_template,'describe',lambda operation,handle:calls.append(handle)or recipe)
  operation=SimpleNamespace(_schema=SimpleNamespace(name='vecops_comm::consume'))
  tensors={name:SimpleNamespace(metadata=TensorMetadata(shape=shape,strides=strides,
      dtype='torch.bfloat16',device='cpu'))for name,shape,strides in (
      ('source',(128,96,768),(73728,768,1)),('output',(128,96,768),(73728,768,1)))}
  deferred=_Instruction('event',operation,(None,_Ref('output'),_Ref('source')),None,None)
  tape=SimpleNamespace(instructions=[deferred],_collective_relocations={'0':recipe})
  capture=SimpleNamespace(_torch_replay_tape=tape,_torch_execution_graph=SimpleNamespace(tensors=tensors))
  result=tuning.consumer_requests([capture,capture])
  assert len(result)==1 and result[0].group is group and not calls
  assert result[0].operator_name=='opaque::contraction'and result[0].specs==(8,36)
  assert result[0].source.shape==(128,96,768)and result[0].output.dtype=='torch.bfloat16'
  assert not hasattr(result[0],'address')and not hasattr(result[0],'tensor')
  assert tape.instructions[0]is deferred and tape._collective_relocations=={'0':recipe}
  bound=_Instruction('event',operation,(91,_Ref('output'),_Ref('source')),None,None)
  tape.instructions=[bound];tape._collective_relocations={}
  assert tuning.consumer_requests([capture])==result and calls==[91]


def test_inventory_preserves_observed_order_and_rejects_stale_contract(monkeypatch):
  from vecops.distributed import _template,tuning
  from vecops.graph._replay import _Instruction,_Ref
  from vecops.graph.ir import TensorMetadata
  group=object()
  def existing(recipe):
    if recipe.get('context')!='live':raise ValueError('platform/runtime contract changed')
    return group
  monkeypatch.setattr(_template,'group_for',existing)
  operation=SimpleNamespace(_schema=SimpleNamespace(name='vecops_comm::consume'))
  tensor=SimpleNamespace(metadata=TensorMetadata(shape=(16,),strides=(1,),dtype='torch.float32',device='cpu'))
  instructions=[_Instruction(str(i),operation,(None,_Ref('x'),_Ref('x')),None,None)for i in range(12)]
  recipes={str(i):dict(operation='vecops_comm::consume',operator_name='opaque::case'+str(i),
      specs=[],context='live')for i in range(12)}
  tape=SimpleNamespace(instructions=instructions,_collective_relocations=dict(sorted(recipes.items())))
  capture=SimpleNamespace(_torch_replay_tape=tape,_torch_execution_graph=SimpleNamespace(tensors={'x':tensor}))
  assert [r.operator_name for r in tuning.consumer_requests([capture])]==['opaque::case'+str(i)for i in range(12)]
  recipes['0']['context']='old'
  with pytest.raises(ValueError,match='platform/runtime'):tuning.consumer_requests([capture])
  recipes['0']['context']='live';tape._collective_relocations={'00':recipes['0']}
  with pytest.raises(ValueError,match='instruction index'):tuning.consumer_requests([capture])


def test_calibration_options_are_detached_policy_facts(monkeypatch):
  import vecops.distributed as vd
  from vecops.distributed import _routing,tuning
  planner=SimpleNamespace(policy={'algorithm':'auto'},consumer_routes={'measured':[1]},
      compiler=SimpleNamespace(topology=SimpleNamespace(partitions=((0,0,1,1),)),
          candidates=lambda:((0,),(1,))))
  monkeypatch.setattr(vd,'backend',lambda group:object())
  monkeypatch.setattr(_routing,'for_backend',lambda native:planner)
  result=tuning.consumer_calibration_options()
  assert result['installed_consumer_profiles']==1 and result['candidates']==[[0],[1]]
  result['policy']['algorithm']='flat';result['partitions'][0][0]=99
  assert planner.policy['algorithm']=='auto'and planner.compiler.topology.partitions==((0,0,1,1),)
