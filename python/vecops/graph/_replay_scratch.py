# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Bind replay tensor/workspace storage through resource-model byte recipes.

A supplied PreparedProgram may contain the entire model's linked scratch plan.
It is shared by all serialized calls; per-graph requirements do not create an
additional physical arena. Standalone graphs compile their own exact recipe.
"""
from dataclasses import dataclass


@dataclass
class ResourceOwners:
  prepared: object
  buffers: dict
  required_bytes: int

  def numel(self):
    return self.required_bytes

  def placement_stats(self):
    return dict(self.prepared.program.peak_bytes_by_capacity_group)


def allocate(execution, identities, *, runtime=None, view=None, program=None,
             binding_map=None, allow_escaping=False):
  import torch
  from vecops.planner.compiler import BufferRequirement, GraphMemoryCompiler, PreparedProgram
  groups = {}
  for identity in identities:
    tensor = execution.tensors[identity]
    storage = execution.storages[tensor.storage_id]
    if storage.external or storage.persistent or (not allow_escaping and
        any(execution.tensors[x].escapes for x in storage.tensors)):
      raise ValueError('scratch requires internal storage; retained outputs need explicit ownership')
    groups.setdefault(tensor.storage_id, []).append(identity)
  origins, requirements, zero_storages = {}, [], set()
  for storage_id, refs in groups.items():
    storage = execution.storages[storage_id]
    if storage.bytes == 0:
      if any(0 not in execution.tensors[x].metadata.shape for x in refs):
        raise ValueError('zero-byte graph storage has a nonempty tensor view: '+storage_id)
      zero_storages.add(storage_id)
      continue
    origin = min(execution.tensors[x].metadata.storage_offset * execution.tensors[x].metadata.element_size for x in refs)
    size = 0
    for identity in refs:
      metadata = execution.tensors[identity].metadata
      if any(stride < 0 for stride in metadata.strides):
        raise ValueError('negative graph buffer strides are unsupported')
      extent = 0 if 0 in metadata.shape else 1 + sum((d-1)*s for d,s in zip(metadata.shape,metadata.strides))
      size = max(size,metadata.storage_offset*metadata.element_size-origin+extent*metadata.element_size)
    origins[storage_id] = origin
    requirements.append(BufferRequirement(storage_id,max(1,size),
        lifetimes=((0,1),) if allow_escaping else ((storage.lifetime_start,storage.lifetime_end+1),),
        boundary='external' if allow_escaping else 'internal',external_role='output' if allow_escaping else None))
  if runtime is None:
    from vecops import memory
    runtime = memory.current_runtime() or memory.default_runtime()
  native_runtime = getattr(runtime,'native',runtime)
  if view is None:
    view = native_runtime.view(native_runtime.current_scope())
  if program is None:
    from vecops.memory import native
    compiled = GraphMemoryCompiler(native_runtime,view,native_runtime.snapshot(view),native_module=native).compile(requirements)
    prepared = compiled.prepare(native_runtime,view)
    prepared.materialize()
  elif isinstance(program,PreparedProgram):
    prepared = program
  else:
    prepared = program.prepare(native_runtime,view)
    prepared.materialize()
  names = binding_map or {}
  buffers = {}
  views = {}
  for storage_id,refs in groups.items():
    if storage_id in zero_storages:
      # Empty storage has metadata and alias identity, but no physical bytes or
      # linked page binding. Retain one canonical StorageImpl for all its views.
      base = torch.empty(0,dtype=torch.uint8,device='cpu')
      buffers[storage_id] = base
      for identity in refs:
        metadata = execution.tensors[identity].metadata
        if metadata.storage_offset < 0 or any(stride < 0 for stride in metadata.strides):
          raise ValueError('negative graph buffer offsets or strides are unsupported')
        dtype = getattr(torch,metadata.dtype.removeprefix('torch.'))
        views[identity] = base.view(dtype).as_strided(metadata.shape,metadata.strides,
            storage_offset=metadata.storage_offset)
      continue
    key = names.get(storage_id,storage_id)
    if key not in prepared.program.bindings:
      raise ValueError('graph storage is absent from the linked memory program: '+key)
    for identity in refs:
      metadata = execution.tensors[identity].metadata
      delta = metadata.storage_offset*metadata.element_size-origins[storage_id]
      dtype = getattr(torch,metadata.dtype.removeprefix('torch.'))
      views[identity] = prepared.tensor(key,metadata.shape,dtype,metadata.strides,byte_offset=delta)
    buffers[storage_id] = prepared.tensor(key,(prepared.program.bindings[key].size_bytes,),torch.uint8)
  capacity = sum(prepared.program.peak_bytes_by_capacity_group.values()) if requirements else 0
  return views,ResourceOwners(prepared,buffers,capacity),capacity
