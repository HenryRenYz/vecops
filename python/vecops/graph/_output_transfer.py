# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Prepared readonly-cache transfer using free capacity in the common session.

Original buffers keep their tier, offset, address and canonical contents. Only
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
  import torch
  owner = getattr(producer,'_retained_output_owner',None)
  if not getattr(producer,'_ownership_proof',False) or not hasattr(owner,'buffers'):
    return []
  protected = producer._protected_storages
  return [(key,buffer,torch.frombuffer(buffer,dtype=torch.uint8).data_ptr())
          for key,buffer in owner.buffers.items() if key not in protected]


def fixed_values(producer, values):
  """Retain only outputs whose entire storage has fixed private ownership."""
  import torch
  owned = _owned(producer)
  result = []
  for value in values:
    if not isinstance(value,torch.Tensor):
      result.append(None); continue
    storage = value.untyped_storage()
    match = next((buffer for _,buffer,address in owned
                  if storage.data_ptr()==address and storage.nbytes()<=buffer.size),None)
    result.append(value if match is not None else None)
  return tuple(result)


class OutputTransfer:
  """One invocation-boundary copy recipe and retained fast views.

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


def prepare(producer, values, consumer, *, input_offset, expected_replays):
  """Plan into currently free fast space, preserving the existing placement.

  The caller promises these cache values are immutable until the next producer
  invocation, including uses outside the consumer graph. Consumer writes and
  opaque/collective effects are rejected conservatively. Zero-read caches are
  excluded. Scores are static read-byte estimates, not measured traffic or a
  guarantee of optimal tier choice. This is an opt-in preparation operation.
  """
  import torch
  from vecops.graph import GraphCapture
  from vecops.planner.buffers import BufferRequirement,current_session
  from vecops.planner.core import storage_traffic
  session = current_session()
  if (expected_replays < 2 or session is None or producer._buffer_session is not session
      or consumer._buffer_session is not session or not consumer._ownership_proof
      or not session.stats()['fast_capacity']):
    return None
  owned = _owned(producer)
  traffic = storage_traffic(consumer.graph)
  writes = {consumer.graph.tensors[use.tensor_id].storage_id for event in consumer.graph.events
            for use in event.uses if use.access.value in {'write','read_write'}}
  groups = {}
  for index,value in enumerate(values):
    if value is None or not value.numel():continue
    storage = value.untyped_storage()
    match = next(((key,buffer) for key,buffer,address in owned
                 if storage.data_ptr()==address and storage.nbytes()<=buffer.size),None)
    if match is None or match[1].tier!='slow':continue
    ref = consumer._input_refs[input_offset+index]
    identity = consumer.graph.tensors[ref.id].storage_id
    entry = groups.setdefault(storage._cdata,dict(storage=storage,match=match,indices=[],storages=set()))
    entry['indices'].append(index);entry['storages'].add(identity)
  candidates = []
  for entry in groups.values():
    identities = entry['storages']
    if identities & (writes | consumer._protected_storages):continue
    # A partial rebind would break input alias guards or leave another reader
    # using a stale canonical buffer. Include every input alias or skip it.
    allowed = {input_offset+index for index in entry['indices']}
    if any(ref.id in consumer.graph.tensors and
           consumer.graph.tensors[ref.id].storage_id in identities and index not in allowed
           for index,ref in enumerate(consumer._input_refs) if hasattr(ref,'id')):
      continue
    start, size = entry['storage'].data_ptr(), entry['storage'].nbytes()
    def overlap(span):
      return span is not None and span[1] > 0 and start < span[0]+span[1] and span[0] < start+size
    if (any(index not in allowed and overlap(span)
            for index,span in enumerate(consumer._input_storage_spans))
        or any(map(overlap,consumer._bound_storage_spans))):
      continue
    reads = sum(traffic.get(key,0) for key in identities)
    size = entry['storage'].nbytes()
    if reads <= 0:continue
    entry.update(bytes=size,reads=reads)
    candidates.append(entry)
  candidates.sort(key=lambda x:(-x['reads']/x['bytes'],-x['bytes'],x['match'][0]))
  sources,targets,owners,rows,replacements,addresses = [],[],[],[],{},{}
  for entry in candidates:
    size = entry['bytes']
    if size > session.stats()['largest_fast_free']:continue
    name = 'output-transfer:'+consumer.graph.name+':'+entry['match'][0]
    buffer = session.bind([BufferRequirement(name,size,placement='required',
        traffic_bytes=entry['reads']*expected_replays)])[name]
    owners.append(buffer)
    storage = torch.frombuffer(memoryview(buffer)[:size],dtype=torch.uint8).untyped_storage()
    sources.append(torch.empty(0,dtype=torch.uint8).set_(entry['storage'],0,(size,),(1,)))
    targets.append(torch.empty(0,dtype=torch.uint8).set_(storage,0,(size,),(1,)))
    for index in entry['indices']:
      old = values[index]
      replacements[index] = torch.empty(0,dtype=old.dtype).set_(storage,old.storage_offset(),old.shape,old.stride())
      addresses[index] = old.data_ptr()
    rows.append(dict(indices=entry['indices'],bytes=size,read_bytes_per_replay=entry['reads'],
                     source_tier='slow',target_tier='fast',target_offset=buffer.offset))
  if not rows:return None
  import vecops
  from vecops._arm_vector_state import get_vector_state
  parallelism = int(os.environ.get('VECOPS_NUM_THREADS', torch.get_num_threads()))
  compiled = _copy_kernel(parallelism, get_vector_state().env_token)
  # Compiler job count and execution width are independent. Prepare every
  # concrete capacity before capture, as for graph pointwise kernels.
  with vecops.collect_compile_requests() as collector:
    for source, target in zip(sources, targets):
      meta = vecops.TensorMeta(tuple(source.shape), strides=(1,), dtype='uint8')
      compiled(meta, meta)
  vecops.compile_batch(collector.requests, execution_parallelism=parallelism)
  operation = compiled.explicit_torch_function()
  def copy(sources,targets):
    for source,target in zip(sources,targets):operation(source,target)
  capture = GraphCapture(consumer.graph.name+':output-transfer')
  capture.capture_torch(copy,tuple(sources),tuple(targets))
  executable = capture.compile_torch()
  return OutputTransfer(executable,sources,targets,replacements,addresses,owners,rows)
