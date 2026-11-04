# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Bind graph alias groups through the common native buffer planner.

Graph intermediates and operator workspace coexist and therefore use distinct
serialization groups. Escaping outputs receive storage-owned leases; they are
never covered by a transient reuse assertion. No placement occurs on replay.
"""
from dataclasses import dataclass


@dataclass
class _Owners:
  buffers: dict
  required_bytes: int

  def numel(self):
    return self.required_bytes

  def placement_stats(self):
    result = {}
    for tier in ('fast','slow'):
      end = total = 0
      for b in sorted((x for x in self.buffers.values() if x.tier == tier),key=lambda x:x.offset):
        total += max(0,b.offset+b.size-max(end,b.offset))
        end = max(end,b.offset+b.size)
      result[tier] = total
    return result


def allocate(execution, identities, *, allocator=None, allow_escaping=False):
  import torch
  from vecops.planner.buffers import BufferRequirement, current_session, plan_buffers
  groups = {}
  for identity in identities:
    tensor = execution.tensors[identity]
    storage = execution.storages[tensor.storage_id]
    if (storage.external or storage.persistent or (not allow_escaping and
        any(execution.tensors[x].escapes for x in storage.tensors))):
      raise ValueError('static scratch must be internal, nonpersistent and nonescaping')
    groups.setdefault(tensor.storage_id, []).append(identity)
  from vecops.planner.core import storage_traffic
  traffic = storage_traffic(execution)
  origins, requirements = {}, []
  for storage_id, refs in groups.items():
    storage = execution.storages[storage_id]
    origin = min(execution.tensors[x].metadata.storage_offset * execution.tensors[x].metadata.element_size for x in refs)
    size = 0
    for ref in refs:
      m = execution.tensors[ref].metadata
      if any(x < 0 for x in m.strides):
        raise ValueError('negative graph buffer strides are unsupported')
      extent = 0 if any(x == 0 for x in m.shape) else 1 + sum((x-1)*s for x,s in zip(m.shape,m.strides))
      size = max(size, m.storage_offset*m.element_size-origin + extent*m.element_size)
    origins[storage_id] = origin
    # All retained outputs coexist beyond this graph's final event.
    lifetime = (0, 1) if allow_escaping else (storage.lifetime_start, storage.lifetime_end + 1)
    requirements.append(BufferRequirement(storage_id, max(1, size), lifetimes=(lifetime,),
        traffic_bytes=traffic.get(storage_id, max(1,size))))
  placement = plan_buffers(requirements, fast_capacity=0, slow_capacity=(1 << 63)-1)
  capacity = placement['slow_bytes']
  managed = current_session() if allocator is None else None
  if managed is not None:
    buffers = managed.bind(requirements, reuse_group='' if allow_escaping else 'graph')
    raw = _Owners(buffers, capacity)
    storages = {key:torch.frombuffer(buffer, dtype=torch.uint8).untyped_storage()
                for key,buffer in buffers.items()}
    offsets = {key:0 for key in groups}
  else:
    raw = torch.empty(capacity, dtype=torch.uint8) if allocator is None else allocator(capacity)
    if (type(raw) is not torch.Tensor or raw.device.type != 'cpu' or raw.dtype != torch.uint8
        or raw.ndim != 1 or not raw.is_contiguous() or raw.storage_offset() != 0 or raw.numel() < capacity):
      raise ValueError('scratch allocator must return a contiguous CPU uint8 buffer of sufficient capacity at offset zero')
    storages = {key:raw.untyped_storage() for key in groups}
    offsets = {key:value['offset'] for key,value in placement['buffers'].items()}
  views = {}
  for identity in identities:
    tensor = execution.tensors[identity]
    m = tensor.metadata
    byte_offset = offsets[tensor.storage_id] + m.storage_offset*m.element_size-origins[tensor.storage_id]
    if byte_offset % m.element_size:
      raise ValueError('misaligned static replay scratch')
    views[identity] = torch.empty(0,dtype=getattr(torch,m.dtype.removeprefix('torch.'))).set_(
        storages[tensor.storage_id], byte_offset//m.element_size, m.shape, m.strides)
  return views, raw, capacity
