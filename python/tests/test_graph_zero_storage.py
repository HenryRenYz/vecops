# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Zero-byte storage keeps alias metadata without adding physical backing."""
from types import SimpleNamespace

import pytest
import torch

from vecops.graph.ir import StorageValue, TensorMetadata, TensorValue
from vecops.graph._replay_scratch import allocate
from vecops.memory import native
from vecops.planner import BufferRequirement, GraphMemoryCompiler


@pytest.fixture
def resources():
  runtime = native.discover_linux_runtime()
  view = runtime.view(runtime.current_scope())
  return runtime,view


def prepare(resources, requirements=()):
  runtime,view = resources
  compiled = GraphMemoryCompiler(runtime,view,runtime.snapshot(view),native_module=native).compile(requirements)
  prepared = compiled.prepare(runtime,view)
  prepared.materialize()
  return prepared


def committed(resources):
  runtime,view = resources
  return sum(group.committed_bytes for group in runtime.snapshot(view).groups)


def graph(*rows):
  tensors,storages = {},{}
  for identity,storage_id,storage_bytes,shape,strides,offset in rows:
    metadata = TensorMetadata(shape,strides,'torch.float32','cpu',element_size=4,
                              storage_offset=offset)
    tensors[identity] = TensorValue(identity,metadata,storage_id)
    storage = storages.setdefault(storage_id,StorageValue(storage_id,storage_bytes))
    assert storage.bytes == storage_bytes
    storage.tensors.append(identity)
  return SimpleNamespace(tensors=tensors,storages=storages)


def test_linked_zero_storage_preserves_alias_identity_offsets_and_zero_budget(resources):
  runtime,view = resources
  # A linked Program may already own unrelated pages. An empty-only owner still
  # reports zero bytes and must not request another page or require a slot key.
  program = prepare(resources,[BufferRequirement('unrelated',16,
      allowed_pools=(view.allocatable_pools[0],))])
  before = committed(resources)
  execution = graph(('a','empty',0,(0,7),(7,1),0),
                    ('b','empty',0,(0,3),(7,1),5),
                    ('c','another-empty',0,(0,),(1,),2))
  with torch.device('meta'):
    views,owner,required = allocate(execution,tuple(execution.tensors),runtime=runtime,
                                  view=view,program=program)
  assert all(tensor.device.type == 'cpu' for tensor in views.values())
  assert owner.prepared is program and owner.required_bytes == required == 0
  assert committed(resources) == before
  assert owner.buffers['empty'].untyped_storage().nbytes() == 0
  assert views['a'].untyped_storage()._cdata == views['b'].untyped_storage()._cdata
  assert views['a'].untyped_storage()._cdata != views['c'].untyped_storage()._cdata
  assert views['b'].storage_offset() == 5 and views['b'].stride() == (7,1)
  assert views['c'].storage_offset() == 2
  program.close()


def test_standalone_zero_storage_has_no_backing_recipe_or_charge(resources):
  runtime,view = resources
  execution = graph(('a','empty',0,(0,),(1,),0))
  before = committed(resources)
  views,owner,required = allocate(execution,['a'],runtime=runtime,view=view)
  assert views['a'].numel() == required == owner.required_bytes == 0
  assert not owner.prepared.recipe.backings and not owner.prepared.recipe.mappings
  assert committed(resources) == before
  owner.prepared.close()


def test_zero_storage_rejects_nonempty_view(resources):
  runtime,view = resources
  execution = graph(('a','empty',0,(1,),(1,),0))
  before = committed(resources)
  with pytest.raises(ValueError,match='zero-byte.*nonempty'):
    allocate(execution,['a'],runtime=runtime,view=view)
  assert committed(resources) == before


def test_empty_slice_of_nonempty_storage_still_requires_linked_binding(resources):
  runtime,view = resources
  execution = graph(('a','payload',16,(0,),(1,),2))
  program = prepare(resources)
  with pytest.raises(ValueError,match='absent from the linked memory program'):
    allocate(execution,['a'],runtime=runtime,view=view,program=program)
  program.close()
