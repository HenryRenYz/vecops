# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Compile-time planner arena isolation and explicitly immutable state leases.

Relocation preserves actual overlapping byte spans, strides and within-component
view relationships. Storage-capacity/identity/absolute-offset observations are
unsupported; disconnected spans need not retain a shared StorageImpl.
"""
from __future__ import annotations
from dataclasses import dataclass

@dataclass
class ArenaLease:
  owner: object
  raw: object
  system: object
  domain: int | None
  placement: str
  fallback: str | None
  alignment: int
  large_pages: bool
  label: str
  state_ranges: tuple = ()


def _state_ranges(session, begin):
  """Only copied, dedicated, immutable module-state slots may be borrowed."""
  from vecops.graph.ir import TensorKind
  from vecops.planner.ir import MemoryTier
  placements=session.plan.placements
  occupied=sorted((p.offset,p.offset+p.bytes,key)for key,p in placements.items()
                 if p.tier==MemoryTier.fast and p.offset is not None and p.bytes)
  prefixes=[];maximum=0
  for low,high,key in occupied:
    prefixes.append(maximum);maximum=max(maximum,high)
  indices={key:index for index,(_,_,key)in enumerate(occupied)}
  result={}
  for tensor in session.graph.tensors.values():
    key=tensor.storage_id;p=placements[key]
    if not (tensor.kind in {TensorKind.parameter,TensorKind.buffer}
            and key in session._copied_storages and p.tier==MemoryTier.fast
            and p.replicated and p.read_only
            and p.annotations.get('eager_external_replica') is True
            and p.offset is not None and p.bytes):
      continue
    index=indices[key];low,high,_=occupied[index]
    # A state slot must never overlap another slot, even at disjoint epochs.
    if prefixes[index]>low or (index+1<len(occupied)and occupied[index+1][0]<high):
      continue
    result[key]=(begin+low,begin+high,key)
  return tuple(sorted(result.values()))


def capture_leases(session):
  if session is None:return []
  import torch
  entries=[('fast',session.fast_owner,session._fast_owner_bytes),
           ('slow',session.slow_owner,session._slow_owner_bytes)]
  pool=getattr(session,'_dynamic_pool',None)
  if pool is not None:entries.append(('dynamic',pool.owner,pool.capacity))
  result=[]
  for label,owner,size in entries:
    if owner is None or not size:continue
    raw=torch.frombuffer(owner,dtype=torch.uint8,count=size)
    result.append(ArenaLease(owner,raw,session.system,session.domain,
      'default' if label=='slow' else 'high_bandwidth',
      'default' if label=='slow' or session.allow_fast_fallback else None,
      max(64,session.plan.machine.alignment),session.large_pages,label,
      _state_ranges(session,raw.data_ptr())if label=='fast'else ()))
  return result


def _storage_span(tensor):
  storage=tensor.untyped_storage()
  return storage.data_ptr(),storage.data_ptr()+storage.nbytes()


def _tensor_span(tensor):
  storage=tensor.untyped_storage()
  start=storage.data_ptr()+tensor.storage_offset()*tensor.element_size()
  if not tensor.numel():return start,start
  extent=sum((size-1)*stride for size,stride in zip(tensor.shape,tensor.stride()))+1
  return start,start+extent*tensor.element_size()


def _components(spans,begin,end,alignment):
  nonempty=sorted((max(begin,low//alignment*alignment),
                   min(end,(high+alignment-1)//alignment*alignment))
                  for low,high in spans if high>low)
  result=[]
  for low,high in nonempty:
    if result and low<=result[-1][1]:result[-1][1]=max(result[-1][1],high)
    else:result.append([low,high])
  # Empty anchors never truncate a nonempty component or its overlap extent.
  for low,high in spans:
    if low==high and not any(a<=low<=b for a,b in result):result.append([low,low])
  return sorted(result)


def isolate_bindings(bindings,inputs,leases,*,borrow_readonly_state=False,
                     tensor_spans=None,written_ids=()):
  """Return isolated bindings, owners, successor leases and allocator metadata.

  Immutable borrowing is opt-in. The validated state ranges remain exclusively
  read-only for the executable lifetime, including access through old aliases.
  Source owner pinning survives session close and module ``.data`` restoration.
  """
  import torch
  result=dict(bindings);owners=[];next_leases=[];isolated=set();rows=[]
  spans_meta=tensor_spans or {}
  writes=[spans_meta[key]for key in written_ids if key in spans_meta]
  for lease in leases:
    begin=lease.raw.data_ptr();end=begin+lease.raw.numel()
    selected={}
    for key,tensor in bindings.items():
      low,high=_storage_span(tensor)
      if high>begin and low<end:
        if low<begin or high>end:raise ValueError('graph binding storage partially overlaps a planner arena')
        selected[key]=tensor
    if not selected:continue
    for tensor in inputs:
      low,high=_tensor_span(tensor)
      # Views of one planner allocation can occupy disjoint live byte ranges.
      # Only an actual input/binding overlap needs staging: isolated bindings
      # are private snapshots (or proven dedicated immutable state), so sharing
      # a large original StorageImpl alone does not create a replay dependency.
      if any(y>low and x<high for x,y in map(_tensor_span, selected.values())):
        raise ValueError('graph input aliases a captured planner arena; input staging is required')
    borrowed={};borrow_ranges={}
    if borrow_readonly_state:
      for key,tensor in selected.items():
        low,high=_tensor_span(tensor)
        for a,b,state in lease.state_ranges:
          if low<high and a<=low and high<=b:
            if any(y>a and x<b for x,y in writes):
              raise ValueError('graph writes through an alias of borrowed immutable planner state')
            borrowed[key]=tensor;borrow_ranges[state]=(a,b);break
    copied={key:tensor for key,tensor in selected.items()if key not in borrowed}
    # Any write alias of a borrowed state range is forbidden, even if that
    # alias belongs to a different StorageImpl or is not a bound argument.
    if any(y>a and x<b for x,y in writes for a,b in borrow_ranges.values()):
      raise ValueError('graph writes through an alias of borrowed immutable planner state')
    if borrowed:
      owners.append(lease.owner);next_leases.append(lease);isolated.update(borrowed)
    components=_components([_tensor_span(t)for t in copied.values()],begin,end,lease.alignment)
    offsets=[];size=0
    for low,high in components:
      offsets.append(size);size+=max(lease.alignment,high-low)
    component_rows=[]
    if size:
      owner=lease.system.allocate(size,placement=lease.placement,domain=lease.domain,
        fallback=lease.fallback,alignment=lease.alignment,large_pages=lease.large_pages)
      raw=torch.frombuffer(owner,dtype=torch.uint8,count=size)
      for (low,high),offset in zip(components,offsets):
        if high>low:raw.narrow(0,offset,high-low).copy_(lease.raw.narrow(0,low-begin,high-low))
        component_rows.append({'source_offset_bytes':low-begin,'bytes':high-low,'private_offset_bytes':offset})
      storages={}
      for key,tensor in copied.items():
        low,high=_tensor_span(tensor)
        index=next(i for i,(a,b)in enumerate(components)if a<=low and high<=b)
        a,b=components[index];offset=offsets[index]
        original=tensor.untyped_storage();identity=original._cdata
        if identity not in storages:
          # Each original StorageImpl keeps one independent StorageImpl, even
          # across disjoint relocated components. Different original storages
          # may still physically overlap through this shared compact owner.
          storages[identity]=torch.frombuffer(owner,dtype=torch.uint8,count=size).untyped_storage()
        storage=storages[identity]
        byte_offset=offset+low-a
        if byte_offset<0 or byte_offset%tensor.element_size():raise ValueError('planner arena rebasing would misalign a graph tensor')
        value=torch.empty(0,dtype=tensor.dtype).set_(storage,byte_offset//tensor.element_size(),tuple(tensor.shape),tuple(tensor.stride()))
        value.requires_grad_(tensor.requires_grad);result[key]=value;isolated.add(key)
      owners.append(owner)
      next_leases.append(ArenaLease(owner,raw,lease.system,lease.domain,lease.placement,
        lease.fallback,lease.alignment,lease.large_pages,lease.label))
    rows.append({'arena':lease.label,'source_capacity_bytes':lease.raw.numel(),
      'private_bytes':size,'source_offset_bytes':min((x[0]for x in components),default=begin)-begin,
      'bound_tensors':len(selected),'copied_bound_tensors':len(copied),
      'borrowed_bound_tensors':len(borrowed),'borrowed_state_bytes':sum(b-a for a,b in borrow_ranges.values()),
      'borrowed_state_ranges':[{'storage_id':key,'source_offset_bytes':a-begin,'bytes':b-a}for key,(a,b)in sorted(borrow_ranges.items())],
      'components':component_rows,'placement':lease.placement,'requested_domain':lease.domain,
      'target_id':getattr(lease.owner if not size else owner,'target_id',None),
      'large_page_bytes':getattr(lease.owner if not size else owner,'large_page_bytes',None)})
  return result,owners,next_leases,isolated,rows
