# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Prepared readonly-cache transfer with explicit resource-program destinations.

Original buffers keep their placement, offset, address and canonical contents. Only
fixed, privately owned initializer outputs are eligible. The consumer proves
read-only use; the caller additionally promises invocation-scoped immutability
across other graphs. Transfers are explicit native vec grids, never SDMA/libc.
"""
from functools import lru_cache
import threading
import os


@lru_cache(None)
def _copy_kernel(parallelism, vector_token):
  # JIT specialization observes this context when the factory is constructed.
  # A fresh graph owner may use a different team or ARM vector length.
  from vecops import language as T
  @T.jit(out_idx=[1])
  def factory():
    N = T.dynamic(lower=0)
    TransferWorkers = T.Const(parallelism)
    @T.template(source='_CacheTransfer.cpp',recipe_id='prepared-cache-transfer-v1')
    def kernel(source:T.Tensor((N,),(1,),T.uint8),target:T.Tensor((N,),(1,),T.uint8)):
      TransferWorkers
    return kernel
  compiled = factory()
  compiled.explicit_torch_function()
  return compiled


def _owned(producer):
  """Return canonical logical buffers, never an entire shared slab envelope."""
  owner=getattr(producer,'_retained_output_owner',None)
  if not hasattr(owner,'buffers'):
    return []
  protected=producer._protected_storages
  return [(key,buffer,buffer.untyped_storage().data_ptr())
          for key,buffer in owner.buffers.items() if key not in protected]


def fixed_values(producer,values):
  import torch
  owned=_owned(producer)
  result=[]
  for value in values:
    if not isinstance(value,torch.Tensor):
      result.append(None)
      continue
    storage=value.untyped_storage()
    match=next((buffer for _,buffer,address in owned if storage.data_ptr()==address
                and storage.nbytes()<=buffer.numel()),None)
    result.append(value if match is not None else None)
  return tuple(result)


def _physical_ranges(descriptor, offset, size, *, required_access=None):
  """Clip complete backend-certified spans; VA inequality is not independence."""
  stop=offset+size
  cursor=offset
  result=[]
  for span in descriptor.spans:
    begin=max(offset,span.offset_bytes)
    end=min(stop,span.offset_bytes+span.size_bytes)
    if begin>=end:
      continue
    if begin!=cursor or not span.physical_identity:
      raise ValueError('output transfer requires complete physical independence receipts')
    if required_access is not None and required_access not in span.access:
      raise ValueError('output transfer destination requires a writable physical binding')
    physical=span.physical_offset_bytes+begin-span.offset_bytes
    result.append((span.physical_identity,physical,physical+end-begin))
    cursor=end
  if cursor!=stop:
    raise ValueError('output transfer requires complete physical independence receipts')
  return tuple(result)


def _intersects(left,right):
  return any(a==b and low<end and begin<high
      for a,low,high in left for b,begin,end in right)


def _check_targets(target_program,groups,targets,consumer,input_offset,*,values,producer):
  """Certify independent destinations before constructing any copy executable.

  Actual binding receipts cover remapped aliases even at different addresses.
  External source/consumer receipts come from the public import descriptor;
  unknown identity is rejected rather than interpreted as a separate allocation.
  All observations are cold. Replay performs no page query or mapping operation.
  """
  from vecops.memory import native as m
  runtime,view=target_program.runtime,target_program.view
  context=m.InvocationContext()
  stamp=m.InvocationStamp();stamp.id='output-transfer-alias-validation'
  stamp.binding_generation=target_program.binding.generation
  context.stamp=stamp
  environment=runtime.resolve_operands(view,target_program.binding,context)
  descriptors={memory.symbol:memory for memory in environment.memories}
  target_ranges=[]
  for entry,target in zip(groups,targets):
    binding=target_program.program.bindings[entry['key']]
    if binding.slot not in descriptors:
      raise ValueError('output transfer destination requires an owned physical binding')
    ranges=_physical_ranges(descriptors[binding.slot],binding.view_offset_bytes,target.numel(),
        required_access=m.AccessMode.write)
    if any(_intersects(ranges,previous) for previous in target_ranges):
      raise ValueError('distinct cache sources require physically independent targets')
    target_ranges.append(ranges)

  def external(address,size,owner):
    if not size:
      return ()
    pointer=m.VirtualAddress();pointer.address_space='host';pointer.value=address
    memory=m.ExternalMemoryView('output-transfer-live-range',pointer,size,owner,read_only=True)
    descriptor=runtime.import_memory(view,memory).describe('output-transfer-live-range')
    return _physical_ranges(descriptor,0,size)

  import torch
  source_storages={tensor.untyped_storage()._cdata:tensor.untyped_storage()
      for tensor in values if isinstance(tensor,torch.Tensor)}
  live=[external(storage.data_ptr(),storage.nbytes(),storage) for storage in source_storages.values()]
  for span in getattr(producer,'_bound_storage_spans',()):
    if span is not None:
      live.append(external(*span,producer))
  redirected={input_offset+index for entry in groups for index in entry['indices']}
  for index,span in enumerate(consumer._input_storage_spans):
    if index not in redirected and span is not None:
      live.append(external(*span,consumer))
  for span in consumer._bound_storage_spans:
    if span is not None:
      live.append(external(*span,consumer))
  if any(_intersects(target,source) for target in target_ranges for source in live):
    raise ValueError('output transfer target aliases a live source or consumer operand')


class OutputTransfer:
  """One invocation-boundary copy recipe and retained resource views.

  replay() runs once after the initializer, before steady calls. It checks the
  fixed source addresses and returns a replacement cache tuple. No allocation,
  capture, placement, mapping or tensor-view construction occurs on replay.
  Canonical outputs are never mutated. Instances use their graph's reentrancy
  guard and must be serialized with their producer and consumer by the caller.
  """
  def __init__(self, executable, sources, targets, replacements, addresses, owners, rows):
    self.executable,self.inputs = executable,tuple((*sources,*targets))
    self.replacements,self.addresses = replacements,addresses
    self.owners,self.rows = owners,rows
    self.bytes = sum(row['bytes'] for row in rows)
    self.calls = 0
    self.thread = threading.get_ident()

  def replay(self, values):
    if threading.get_ident()!=self.thread:
      raise RuntimeError('output transfer must use its preparation thread')
    if any(values[index].data_ptr()!=address for index,address in self.addresses.items()):
      raise RuntimeError('initializer output transfer source address changed; prepare again')
    self.executable.run_tensors(self.inputs)
    self.calls += 1
    return tuple(self.replacements.get(index,value) for index,value in enumerate(values))


def prepare(producer,values,consumer,*,input_offset,target_program,target_bindings):
  """Bind explicit immutable cache destinations from a model resource program.

  The caller selects and materializes the destination recipe before this call.
  This helper performs no placement search or automatic cache promotion. Source
  aliases and consumer writes are checked before the native byte-copy executable
  is built. Targets retain their actual physical owners.
  Distinct source storages require physically disjoint targets, also disjoint
  from every source and unredirected live consumer operand. Backend identity
  receipts certify this even across different virtual mappings; incomplete
  receipts require an independently certified destination before preparation.
  """
  import torch
  from vecops.graph import GraphCapture
  if target_program is None:
    raise ValueError('output transfer requires an explicit materialized target program')
  if target_program.binding is None:
    raise ValueError('output transfer targets must be materialized')
  targets_by_index=dict(target_bindings or {})
  if not targets_by_index:
    raise ValueError('output transfer requires explicit target buffer bindings')
  owned=_owned(producer)
  sources,targets,rows,replacements,addresses=[],[],[],{},{}
  writes={consumer.graph.tensors[use.tensor_id].storage_id for event in consumer.graph.events
          for use in event.uses if use.access.value in {'write','read_write'}}
  groups={}
  for index,key in targets_by_index.items():
    if not isinstance(index,int) or index<0 or index>=len(values):
      raise ValueError('output transfer names an invalid cache index')
    value=values[index]
    if not isinstance(value,torch.Tensor) or not value.numel():
      raise ValueError('output transfer requires a nonempty fixed tensor')
    storage=value.untyped_storage()
    if not any(storage.data_ptr()==address and storage.nbytes()<=buffer.numel()
               for _,buffer,address in owned):
      raise ValueError('output transfer source is not privately owned by its producer')
    ref=consumer._input_refs[input_offset+index]
    identity=consumer.graph.tensors[ref.id].storage_id
    if identity in writes or identity in consumer._protected_storages:
      raise ValueError('output transfer consumer may modify or asynchronously retain this cache')
    group=groups.setdefault(storage._cdata,dict(storage=storage,key=key,indices=[],identities=set()))
    if group['key']!=key:
      raise ValueError('aliases of one cache require one destination buffer')
    group['indices'].append(index)
    group['identities'].add(identity)
  for entry in groups.values():
    allowed={input_offset+index for index in entry['indices']}
    if any(hasattr(ref,'id') and consumer.graph.tensors[ref.id].storage_id in entry['identities']
           and index not in allowed for index,ref in enumerate(consumer._input_refs)):
      raise ValueError('every cache input alias must be redirected together')
    start,size=entry['storage'].data_ptr(),entry['storage'].nbytes()
    def overlap(span):
      return span is not None and span[1]>0 and start<span[0]+span[1] and span[0]<start+size
    if (any(index not in allowed and overlap(span) for index,span in enumerate(consumer._input_storage_spans))
        or any(map(overlap,consumer._bound_storage_spans))):
      raise ValueError('cache destination would break a physical alias with another operand')
    source=torch.empty(0,dtype=torch.uint8, device='cpu').set_(entry['storage'],0,(size,),(1,))
    target=target_program.tensor(entry['key'],(size,),torch.uint8)
    sources.append(source);targets.append(target)
    for index in entry['indices']:
      old=values[index]
      replacements[index]=target_program.tensor(entry['key'],old.shape,old.dtype,old.stride(),
          byte_offset=old.storage_offset()*old.element_size())
      addresses[index]=old.data_ptr()
    rows.append(dict(indices=entry['indices'],bytes=size,target_buffer=entry['key']))
  _check_targets(target_program,list(groups.values()),targets,consumer,input_offset,
      values=values,producer=producer)
  import vecops
  from vecops._arm_vector_state import get_vector_state
  parallelism=int(os.environ.get('VECOPS_NUM_THREADS',torch.get_num_threads()))
  compiled=_copy_kernel(parallelism,get_vector_state().env_token)
  with vecops.collect_compile_requests() as collector:
    for source,target in zip(sources,targets):
      meta=vecops.TensorMeta(tuple(source.shape),strides=(1,),dtype='uint8')
      compiled(meta,meta)
  vecops.compile_batch(collector.requests,execution_parallelism=parallelism)
  operation=compiled.explicit_torch_function()
  def copy(sources,targets):
    for source,target in zip(sources,targets):
      operation(source,target)
  capture=GraphCapture(consumer.graph.name+':output-transfer')
  capture.capture_torch(copy,tuple(sources),tuple(targets))
  executable=capture.compile_torch(runtime=target_program.runtime)
  return OutputTransfer(executable,sources,targets,replacements,addresses,[target_program],rows)
