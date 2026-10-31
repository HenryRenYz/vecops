# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Private static scratch for nonescaping replay tensors.

Intervals cover the entire captured storage, including all aliases. A slot is
reused only after the previous storage's last use, never at the same event.
This prevents output/input overlap during an operation that consumes its input
for the last time. Outputs and external/persistent storage are never placed.
"""


def allocate(execution, identities, *, allocator=None, allow_escaping=False):
  import torch
  groups = {}
  for identity in identities:
    tensor = execution.tensors[identity]
    storage = execution.storages[tensor.storage_id]
    if (storage.external or storage.persistent or (not allow_escaping and
        any(execution.tensors[x].escapes for x in storage.tensors))):
      raise ValueError('static scratch must be internal, nonpersistent and nonescaping')
    groups.setdefault(tensor.storage_id, []).append(identity)
  active, free, offsets = [], [], {}
  capacity = 0
  align = lambda x:(x + 63) // 64 * 64
  for storage_id, refs in sorted(groups.items(), key=lambda kv:execution.storages[kv[0]].lifetime_start):
    storage = execution.storages[storage_id]
    expired = [x for x in active if x[0] < storage.lifetime_start]
    active = [x for x in active if x[0] >= storage.lifetime_start]
    free.extend((offset, size) for _, offset, size in expired)
    merged = []
    for offset, size in sorted(free):
      if merged and merged[-1][0] + merged[-1][1] == offset:
        merged[-1] = (merged[-1][0], merged[-1][1] + size)
      else:
        merged.append((offset, size))
    free = merged
    origin = min(execution.tensors[x].metadata.storage_offset * execution.tensors[x].metadata.element_size for x in refs)
    size = 0
    for ref in refs:
      m = execution.tensors[ref].metadata
      extent = 0 if any(x == 0 for x in m.shape) else 1 + sum((x-1)*s for x,s in zip(m.shape,m.strides))
      size = max(size, m.storage_offset*m.element_size-origin + extent*m.element_size)
    size = align(max(1, size))
    index = next((i for i,(_,n) in enumerate(free) if n >= size), None)
    if index is None:
      offset = capacity
      capacity += size
    else:
      offset, available = free.pop(index)
      if available > size:
        free.append((offset+size, available-size))
    active.append((storage.lifetime_end, offset, size))
    offsets[storage_id] = (offset, origin)
  raw = torch.empty(capacity, dtype=torch.uint8) if allocator is None else allocator(capacity)
  if (type(raw) is not torch.Tensor or raw.device.type != 'cpu' or raw.dtype != torch.uint8
      or raw.ndim != 1 or not raw.is_contiguous() or raw.storage_offset() != 0 or raw.numel() < capacity):
    raise ValueError('scratch allocator must return a contiguous CPU uint8 buffer of sufficient capacity at offset zero')
  views = {}
  # Storage identity/capacity observations are forbidden during capture. One
  # private owner backs these views; captured live aliases keep relative offsets.
  storages = {key:raw.untyped_storage() for key in groups}
  for identity in identities:
    tensor = execution.tensors[identity]
    m = tensor.metadata
    offset, origin = offsets[tensor.storage_id]
    byte_offset = offset + m.storage_offset*m.element_size-origin
    if byte_offset % m.element_size:
      raise ValueError('misaligned static replay scratch')
    views[identity] = torch.empty(0,dtype=getattr(torch,m.dtype.removeprefix('torch.'))).set_(
        storages[tensor.storage_id], byte_offset//m.element_size, m.shape, m.strides)
  return views, raw, capacity
