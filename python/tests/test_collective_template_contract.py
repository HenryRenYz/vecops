# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Logical groups cannot silently bind to a different communicator or platform."""
from types import SimpleNamespace
import json

import pytest


def test_collective_recipe_retains_membership_not_handle(monkeypatch):
  import torch.distributed as dist
  import vecops.distributed as vd
  from vecops.distributed import _template
  class Group:
    pass
  world,subgroup,duplicate=Group(),Group(),Group()
  groups={world:{0:0,1:1,2:2},subgroup:{0:0,2:1}}
  context={'value':'platform-one'}
  monkeypatch.setattr(_template,'_plans',{})
  monkeypatch.setattr(dist,'is_initialized',lambda:True)
  monkeypatch.setattr(dist,'get_backend',lambda group=None:'vecops')
  monkeypatch.setattr(dist,'get_process_group_ranks',lambda group:list(groups[group]))
  monkeypatch.setattr(dist.distributed_c10d,'_get_default_group',lambda:world)
  monkeypatch.setattr(dist.distributed_c10d,'_world',SimpleNamespace(pg_group_ranks=groups))
  monkeypatch.setattr(vd,'backend',lambda group=None:SimpleNamespace(tuning_context=context['value']))
  _template.remember('vecops_comm::run',123456789,world,kind='all_gather',options={})
  recipe=json.loads(json.dumps(_template.describe('vecops_comm::run',123456789)))
  assert '123456789'not in json.dumps(recipe)
  assert _template.group_for(recipe)is world
  context['value']='platform-two'
  with pytest.raises(ValueError,match='platform/runtime'):_template.group_for(recipe)
  context['value']='platform-one'
  _template.remember('vecops_comm::consume',987654321,subgroup,operator_name='native::consumer',specs=[3])
  recipe=_template.describe('vecops_comm::consume',987654321)
  assert _template.group_for(recipe)is subgroup
  groups[duplicate]={0:0,2:1}
  with pytest.raises(ValueError,match='unambiguous'):_template.group_for(recipe)
  with pytest.raises(ValueError,match='no relocatable'):_template.describe('vecops_comm::run',999)


def test_canonical_json_keeps_live_collective_preparation_order(monkeypatch):
  """Digit-boundary ordinals retain their original allocation/descriptor order."""
  import torch
  import vecops.distributed as vd
  from vecops.distributed import _template
  from vecops.graph._replay import _Instruction, _Ref
  from vecops.graph.ir import TensorMetadata

  prepared=[]
  group=object()
  def prepare(kind, outputs, inputs, out_splits, in_splits, root):
    prepared.append(inputs[0].shape[0])
    return 900+len(prepared)
  monkeypatch.setattr(_template,'group_for',lambda recipe:group)
  monkeypatch.setattr(_template,'remember',lambda *a,**k:None)
  monkeypatch.setattr(vd,'backend',lambda selected:SimpleNamespace(prepare=prepare))
  monkeypatch.setattr(vd,'native_operators',lambda selected:{})
  operation=SimpleNamespace(_schema=SimpleNamespace(name='vecops_comm::run'))
  instructions=[None]*102
  tensors={}
  recipes={}
  for index in (2,10,101):
    ref=_Ref(str(index))
    tensors[ref.id]=SimpleNamespace(metadata=TensorMetadata(
        shape=(index,),strides=(1,),dtype=str(torch.float32),device='cpu'))
    instructions[index]=_Instruction(str(index),operation,(None,[ref],[ref]),None,None)
    recipes[str(index)]=dict(operation='vecops_comm::run',kind='all_gather',
        options=dict(copy_levels=None,output_splits=[],input_splits=[],root=0))
  tape=SimpleNamespace(instructions=instructions,native_operators={},
      _collective_relocations=json.loads(json.dumps(recipes,sort_keys=True)))
  _template.rebind(tape,SimpleNamespace(tensors=tensors))
  assert prepared==[2,10,101]
  assert [tape.instructions[i].arguments[0]for i in (2,10,101)]==[901,902,903]
  assert tape._collective_relocations=={}
