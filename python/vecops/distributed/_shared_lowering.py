# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Cold shared-page lowering for ordinary, private graph collectives."""
from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import statistics


_OPERATION = 'vecops_comm::run'
_VIEW = {
    'aten::alias', 'aten::detach', 'aten::view', 'aten::_unsafe_view',
    'aten::reshape', 'aten::reshape_as', 'aten::slice', 'aten::narrow',
    'aten::select', 'aten::split', 'aten::split_with_sizes', 'aten::unbind',
    'aten::unsqueeze', 'aten::squeeze', 'aten::transpose', 'aten::permute',
    'aten::flatten', 'aten::unflatten', 'aten::movedim',
}


def _json(value):
  return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def profile_manifest():
  """Read once at group construction, then attest the same bytes on all ranks."""
  if not (name := os.environ.get('VECOPS_SHARED_PAGE_PROFILE')):
    return {}, None
  data = Path(name).expanduser().read_bytes()
  obj = json.loads(data)
  if obj.get('schema') != 'vecops-shared-page/1' or not isinstance(obj.get('cases'), dict):
    raise ValueError('invalid shared-page calibration profile')
  return obj['cases'], hashlib.sha256(data).hexdigest()


def _choose(samples, operation, minimum_gain):
  if not isinstance(samples, dict):
    return None
  def elapsed(name):
    row = samples.get(name)
    if (not isinstance(row, list) or len(row) < 3 or
        any(type(x) not in (int, float) or not math.isfinite(x) or x <= 0 for x in row)):
      return None
    return row
  if not (base := elapsed('copy')):
    return None
  choices = []
  for name in ('source', 'destination') if operation == 'all_to_all' else ('source',):
    if (row := elapsed(name)) and len(row) == len(base):
      # All batches must confirm the requested gain. A lucky median cannot
      # admit a route whose slowest batch overlaps the copy baseline.
      if max(row) < min(base)*(1-minimum_gain):
        choices.append((statistics.median(row), name))
  return min(choices)[1] if choices else None


def _refs(value):
  from vecops.graph._replay import _Ref
  if isinstance(value, _Ref):
    yield value
  elif isinstance(value, (list, tuple)):
    for item in value:
      yield from _refs(item)


@dataclass(frozen=True)
class _Candidate:
  operation: str
  ordinal: int
  first: int
  last: int
  source: str
  target: str
  block_bytes: int
  dtype: str
  stripe_layout: tuple[int,bool,bool]
  references: tuple
  contract: dict


def _artifact_hash(path):
  from ._routing import _binary_hashes
  path = Path(path).resolve()
  stat = path.stat()
  receipt = str(path),stat.st_ino,stat.st_size,stat.st_mtime_ns
  if receipt not in _binary_hashes:
    with path.open('rb') as stream:
      _binary_hashes[receipt] = hashlib.file_digest(stream,'sha256').hexdigest()
  return _binary_hashes[receipt]


def _candidate(plan, ordinal, recipe, page, prepared_endpoints, workspaces, uses, ordinals, alias_nodes, reject):
  import torch
  from vecops import _schema_bridge
  from vecops.graph._pointwise import contiguous_layout
  from vecops.graph._replay import _Ref
  tape = plan.capture._torch_replay_tape
  execution = plan.execution
  item = tape.instructions[ordinal]
  outputs, inputs = item.arguments[1:3]
  kind = recipe['kind']
  world = len(recipe['ranks'])
  options = recipe['options']
  if world < 2 or options['copy_levels'] is not None or options['root']:
    return reject('group or explicit collective options')
  operation = 'all_to_all' if kind.startswith('all_to_all') else 'all_gather'
  if (not inputs or not outputs or
      not all(isinstance(x, _Ref) for x in (*inputs, *outputs))):
    return reject('tensor arity')

  def blocks(refs, count, splits=()):
    rows = [execution.tensors[ref.id] for ref in refs]
    root = rows[0].storage_id
    metadata = [row.metadata for row in rows]
    if any(row.storage_id != root for row in rows):
      return
    if any(m.dtype != metadata[0].dtype or m.device != 'cpu' or m.requires_grad or
           m.layout != 'torch.strided' or type(m.logical_bytes) is not int or not m.logical_bytes or
           not all(type(n) is int for n in (*m.shape,*m.strides,m.storage_offset,m.element_size))
           for m in metadata):
      return
    total = sum(m.logical_bytes for m in metadata)
    if total % count:
      return
    block = total//count
    if block % page or execution.storages[root].bytes != total:
      return
    stripe = 0
    if len(rows) == count:
      if all(contiguous_layout(m.shape,m.strides) for m in metadata):
        if any(m.logical_bytes != block or m.storage_offset*m.element_size != i*block
               for i,m in enumerate(metadata)):
          return
      else:
        # Exact partition of a dense [outer, peers, inner] storage.
        # A peer view selects one page-aligned run from each outer row.
        first = metadata[0]
        suffix, inner = len(first.shape), 1
        while suffix and (first.shape[suffix-1] == 1 or first.strides[suffix-1] == inner):
          inner *= first.shape[suffix-1]
          suffix -= 1
        stripe = inner*first.element_size
        pitch = inner*count
        if count == 1 or not suffix or stripe % page:
          return
        for length,stride in reversed(tuple(zip(first.shape[:suffix],first.strides[:suffix]))):
          if length != 1 and stride != pitch:
            return
          pitch *= length
        if any(m.shape != first.shape or m.strides != first.strides or m.logical_bytes != block
               or m.storage_offset != i*inner for i,m in enumerate(metadata)):
          return
    elif len(rows) != 1 or metadata[0].storage_offset or not contiguous_layout(metadata[0].shape,metadata[0].strides):
      return
    if splits:
      m = metadata[0]
      unit = m.element_size if kind == 'all_gatherv' else math.prod(m.shape[1:])*m.element_size
      if len(splits) != count or any(value*unit != block for value in splits):
        return
    return root, block, metadata[0].dtype, stripe

  source = blocks(inputs, world if operation == 'all_to_all' else 1, options['input_splits'])
  target = blocks(outputs, world, options['output_splits'])
  if source is None or target is None or source[1:3] != target[1:3] or source[0] == target[0]:
    return reject('rank blocks do not partition distinct page-aligned storages')
  if source[3] and target[3] and source[3] != target[3]:
    return reject('incompatible interleaved stripe sizes')
  src, block_bytes, dtype, source_stripe = source
  dst = target[0]
  stripe_layout = (source_stripe or target[3],bool(source_stripe),bool(target[3]))
  for storage_id in (src, dst):
    storage = execution.storages[storage_id]
    if (storage.external or storage.persistent or storage_id not in plan._owned_storages or
        any(execution.tensors[x].escapes or x in plan.replacements for x in storage.tensors)):
      return reject('collective storage is not private and graph-owned')

  src_uses = [(n,u) for n,u in uses[src] if n != ordinal]
  dst_uses = [(n,u) for n,u in uses[dst] if n != ordinal]
  if (len(src_uses) != 1 or not dst_uses or
      any(n <= ordinal or u.access.value != 'read' for n,u in dst_uses)):
    return reject('requires one producer and invocation-local read-only consumers')
  first, writer = src_uses[0]
  last = max(n for n,_ in dst_uses)
  m = execution.tensors[writer.tensor_id].metadata
  if (first >= ordinal or not writer.access.writes or m.storage_offset or
      m.logical_bytes != execution.storages[src].bytes or not contiguous_layout(m.shape,m.strides)):
    return reject('producer does not define the full contiguous storage')

  # Require recorded native effects between the producer and last consumer.
  # View-only instructions are allowed and preserve the captured byte geometry.
  # Keeping other dispatch and collectives outside this interval also prevents
  # a failed opaque operation from stranding a shared lease on the peers.
  interval, references, addresses = [], set(), [plan.prepared_cache[ordinal][0].abi_address]
  for index in range(first,last+1):
    if index == ordinal:
      interval.append({'collective':kind})
      continue
    instruction = tape.instructions[index]
    name = instruction.function._schema.name
    args = plan.native_rewrites.get(index,instruction.arguments)
    if index in plan.pointwise:
      compiled,a,b,result,alpha = plan.pointwise[index]
      function,specs = compiled.explicit_torch_binding()
      name,args = function._schema.name,(a,b,result,alpha,*specs)
    elif (index in ordinals and index in alias_nodes) or (name in {'aten::empty','aten::empty_strided','aten::empty_like'}
        and (result := tuple(_refs(instruction.result)))
        and all(ref.id in plan.static_factories for ref in result)):
      continue
    elif name.startswith('aten::') or instruction.function._schema.returns:
      return reject('unfused operator in producer-consumer interval: '+name)
    prepared = plan.prepared_cache.get(index)
    registration = _schema_bridge._registrations.get(tuple(name.split('::',1)))
    if prepared is None or registration is None or registration.library_path is None:
      return reject('operator has no native plan in producer-consumer interval: '+name)
    addresses.append(prepared[0].abi_address)
    if index == first:
      # A write-only out parameter is the producer's complete data definition.
      if not any(isinstance(ref,_Ref) and ref.id == writer.tensor_id and flag == 2
                 for ref,flag in zip(args,prepared[1])):
        return reject('producer lacks a write-only ABI operand')
    if any(isinstance(ref,_Ref) and execution.tensors[ref.id].storage_id == dst and flag != 1
           for ref,flag in zip(args,prepared[1])):
      return reject('consumer ABI does not grant read-only access')
    workspace = workspaces(index,prepared[0])
    if workspace is None:
      return reject('native workspace placement unavailable')
    def contract(arg):
      if isinstance(arg,_Ref):
        tensor = execution.tensors[arg.id]
        m = tensor.metadata
        role = 'source' if tensor.storage_id == src else 'target' if tensor.storage_id == dst else 'other'
        if role != 'other':
          if (m.dtype != dtype or m.requires_grad or not all(
              type(n) is int and n >= 0 for n in (*m.shape,*m.strides,m.storage_offset,m.element_size))):
            raise ValueError('incompatible shared tensor alias')
          span = (0 if 0 in m.shape else 1+sum((n-1)*s for n,s in zip(m.shape,m.strides)))
          if (m.storage_offset+span)*m.element_size > execution.storages[tensor.storage_id].bytes:
            raise ValueError('shared tensor alias extends beyond its storage')
        references.add(arg.id)
        return {'tensor':[m.shape,m.strides,m.dtype,m.storage_offset], 'role':role,
                'placement':prepared_endpoints(arg)}
      if arg is None or type(arg) in (bool,int,str) or (type(arg) is float and math.isfinite(arg)):
        return arg
      if isinstance(arg, torch.dtype):
        return ('torch.dtype',str(arg))
      raise ValueError('non-static native operand in shared-page candidate')
    try:
      interval.append({'kernel':name,'bridge_binary':_artifact_hash(registration.library_path),
                       'workspace':workspace,
                       'arguments':[contract(arg) for arg in args]})
    except (ValueError,OSError):
      return reject('non-static kernel arguments')

  from . import load_communication
  artifacts = load_communication().shared_exchange_artifacts(addresses)
  if not artifacts:
    return reject('producer and consumers do not share one resident execution team')
  try:
    for entry,path in zip((entry for entry in interval if 'kernel' in entry),artifacts[1:],strict=True):
      entry['kernel_binary'] = _artifact_hash(path)
  except OSError:
    return reject('native artifact could not be identified')
  return _Candidate(operation,ordinal,first,last,src,dst,block_bytes,dtype,stripe_layout,tuple(sorted(references)),
      {'kind':kind,'operation':operation,'bytes':block_bytes,'dtype':dtype,
       'interval':interval,'stripe_layout':stripe_layout,
       'copy_route':list(plan.prepared_cache[ordinal][0].native.route_modes)})


def prepare(plan, placement, workspaces, runtime, view, alias_nodes):
  """Return selected cold leases; keys cover all ranks' geometry and placement."""
  from . import PageExchange, backend
  from ._routing import for_backend
  from ._template import live_group
  from .tuning import profile_context, _complete_performance_context
  import torch

  audit = os.environ.get('VECOPS_SHARED_PAGE_AUDIT')
  tape = plan.capture._torch_replay_tape
  execution = plan.execution
  indexed = [(index,i) for index,i in enumerate(tape.instructions) if i.function._schema.name == _OPERATION]
  if not indexed:
    return (), {}
  groups = {i.arguments[0]:live_group(_OPERATION,i.arguments[0]) for _,i in indexed}
  groups = {handle:(group,backend(group)) for handle,group in groups.items() if group is not None}
  planners = {handle:for_backend(native) for handle,(_,native) in groups.items()}
  if not audit and not any(p and p.shared_profile for p in planners.values()):
    return (), {}
  view_ordinals, uses = set(), defaultdict(list)
  events = {event.id:event for event in execution.events}
  for index,instruction in enumerate(tape.instructions):
    event = events[instruction.event_id]
    if instruction.function._schema.name in {'aten::empty','aten::empty_strided','aten::empty_like'}:
      outputs = tuple(_refs(instruction.result))
      if outputs and all(ref.id in plan.static_factories for ref in outputs):
        continue
    if (instruction.function._schema.name in _VIEW and
        (instruction.function._schema.name == 'aten::reshape_as' or not tuple(_refs(instruction.arguments[1:])))):
      inputs = {execution.tensors[ref.id].storage_id for ref in _refs(instruction.arguments[:1])}
      outputs = tuple(_refs(instruction.result))
      if (outputs and all(execution.tensors[ref.id].storage_id in inputs for ref in outputs) and
          not any(use.access.writes for use in event.uses)):
        view_ordinals.add(index)
        continue
    for use in event.uses:
      if use.access.value not in {'allocate','alias'}:
        uses[execution.tensors[use.tensor_id].storage_id].append((index,use))
  selected, measurements, regions, last = [], {}, {}, -1
  for ordinal,instruction in indexed:
    handle = instruction.arguments[0]
    planner = planners.get(handle)
    prepared = plan.prepared_cache.get(ordinal)
    if planner is None:
      continue
    from ._template import _plans
    recipe = _plans[(_OPERATION,handle)][1]
    recipe = {**recipe,'ranks':range(planner.size)}
    if recipe['kind'] not in {'all_gather','all_gather_into_tensor','all_gatherv','all_to_all','all_to_all_single'}:
      continue
    group,native = groups[handle]
    reasons = []
    candidate = (_candidate(plan,ordinal,recipe,native.payload_page,placement,workspaces,uses,
                            view_ordinals,alias_nodes,reasons.append)
                 if prepared is not None else None)
    row = None
    if candidate is not None and candidate.first > last:
      row = {**candidate.contract, 'payload_pool':native.payload_pool}
    if planner.shared_profile or planner.shared_audit:
      step = planner.shared_ordinal
      planner.shared_ordinal += 1
      key = f'vecops/shared-lowering/{step}/'
      planner.store.set(key+str(planner.rank),_json({'kind':recipe['kind'],'candidate':row}))
      peers = [json.loads(planner.store.get(key+str(rank))) for rank in range(planner.size)]
      if any(p['kind'] != recipe['kind'] for p in peers):
        raise ValueError('shared-page collective preparation order differs between ranks')
      rows = [p['candidate'] for p in peers]
    else:
      rows = [row]
    complete = (all(rows) and len({(r['bytes'],r['dtype'],tuple(r['stripe_layout'])) for r in rows}) == 1 and
                all(all(arg.get('placement') is not None for entry in r['interval'] for arg in entry.get('arguments',[])
                        if isinstance(arg,dict)) for r in rows))
    context = profile_context(native) if complete else None
    case = (hashlib.sha256(_json({'context':context,'rows':rows}).encode()).hexdigest()
            if context and _complete_performance_context(context) else None)
    route = (_choose(planner.shared_profile.get(case),candidate.operation,
                     planner.policy.get('minimum_gain',.05)) if case is not None else None)
    measure = planner.shared_measure
    recording = bool(measure and case and (not planner.shared_cases or case in planner.shared_cases))
    if recording:
      route = measure if measure in {'source','destination'} and (
          measure == 'source' or candidate.operation == 'all_to_all') else None
      measurements[ordinal] = (candidate,case,'copy',planner.rank,planner.size)
      last = candidate.last
    def record(selected=None, observed=None, backing=None):
      if audit:
        with open(f'{audit}.{planner.rank}.jsonl','a') as stream:
          print(_json({'graph':plan.capture.name,'ordinal':ordinal,'kind':recipe['kind'],
                       'local_candidate':candidate is not None,'case':case,'rows':rows,
                       'selected':selected,'observed_shared_pages':observed,
                       'shared_backing':backing,'reasons':reasons}),file=stream)
    if route is None:
      record()
      continue
    reuse_key = (planner,candidate.operation,candidate.block_bytes,candidate.dtype,
                 candidate.stripe_layout,route)
    if reuse_key not in regions:
      # The copy-path program still owns its admission. Charge each distinct
      # shared layout once; disjoint windows reuse its bounded lease slot.
      charged = candidate.block_bytes*(planner.size if candidate.operation == 'all_to_all' else 1)
      available = next((pool.allocatable_bytes for pool in runtime.snapshot(view,[]).pools
                        if pool.pool == native.payload_pool),0)
      if not native.preparation_consensus(available >= charged):
        reasons.append('insufficient shared backing capacity')
        record()
        continue
      dtype = getattr(torch,candidate.dtype.removeprefix('torch.'))
      try:
        region = PageExchange((candidate.block_bytes//torch.empty((),dtype=dtype).element_size(),),
            dtype,group=group,operation=candidate.operation,placement=route,_stripe_layout=candidate.stripe_layout)
      except RuntimeError as error:
        if 'shared-page backing admission failed on a peer' not in str(error):
          raise
        reasons.append('shared backing admission failed')
        record()
        continue
      observed = _observe_pages(runtime,view,region) if audit else None
      regions[reuse_key] = (region,observed,len(regions))
    region,observed,backing = regions[reuse_key]
    if recording:
      measurements[ordinal] = (candidate,case,route,planner.rank,planner.size)
    selected.append((candidate,region,case,route))
    last = candidate.last
    record(route,observed,backing)
  return selected, measurements


def _observe_pages(runtime, view, region):
  from vecops.memory import native
  result = {}
  for role,tensor in (('producer',region.producer()),('consumer',region.consumer())):
    address = native.VirtualAddress()
    address.address_space, address.value = 'host', tensor.data_ptr()
    external = native.ExternalMemoryView('shared-page-audit-'+role,address,
        tensor.untyped_storage().nbytes(),tensor,None,role == 'consumer')
    try:
      observation = runtime.observe_external(view,external)
    except (RuntimeError,ValueError):
      result[role] = None
      continue
    result[role] = ([[segment.pool,segment.size_bytes] for segment in observation.observed_placement]
                    if observation.complete else None)
  return result
