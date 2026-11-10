# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Address-free capture templates. No tensor payloads or Python pickles."""
from __future__ import annotations

import weakref

import torch
from torch.utils._pytree import tree_flatten, treespec_dumps, treespec_loads

from ._replay import _check_context, _Instruction, _Ref, _Tape
from .ir import ExecutionGraph


class TemplateUnsupported(ValueError):
  """A capture cannot be relocated without guessing its dependencies."""


def _tracked_operator(function):
  if function._schema.name.split('::',1)[0] in {'aten','prims','alphafold3_native','vecops_language'}:
    return True
  from vecops import _schema_bridge
  registration = _schema_bridge._registrations.get(tuple(function._schema.name.split('::',1)))
  return registration is not None and registration.materialized and bool(registration.library_path)


def encode(value):
  if isinstance(value, _Ref):
    return {'ref': value.id}
  if isinstance(value, (tuple, list)):
    return {type(value).__name__: [encode(x) for x in value]}
  if isinstance(value, dict):
    return {'dict': [[k, encode(v)] for k, v in value.items()]}
  if isinstance(value, (torch.dtype, torch.layout, torch.memory_format)):
    return {'torch': str(value).removeprefix('torch.')}
  if isinstance(value, torch.device):
    return {'device': str(value)}
  if value is None or type(value) in (str, int, float, bool):
    return value
  raise TemplateUnsupported(f'unsupported value: {type(value).__name__}')


def decode(value):
  if not isinstance(value, dict):
    return value
  if len(value) != 1:
    raise ValueError('invalid template value')
  tag, item = next(iter(value.items()))
  if tag == 'ref':
    return _Ref(item)
  if tag == 'tuple':
    return tuple(map(decode, item))
  if tag == 'list':
    return list(map(decode, item))
  if tag == 'dict':
    return {k: decode(v) for k, v in item}
  if tag == 'device':
    return torch.device(item)
  if tag == 'torch':
    result = getattr(torch, item)
    if isinstance(result, (torch.dtype, torch.layout, torch.memory_format)):
      return result
  raise ValueError(f'invalid template tag: {tag}')


def sources(owner, args):
  leaves, _ = tree_flatten((args, {}))
  result = {f'input:{i}': x for i, x in enumerate(leaves) if isinstance(x, torch.Tensor)}
  result.update((f'parameter:{n}', x) for n, x in owner.named_parameters(remove_duplicate=False))
  result.update((f'buffer:{n}', x) for n, x in owner.named_buffers(remove_duplicate=False))
  # Plain tensor attributes are also model-owned state. Invocation ValueCache
  # contents are deliberately excluded; those must be explicit graph slots.
  for module_name, module in owner.named_modules():
    for name, value in vars(module).items():
      if isinstance(value, torch.Tensor):
        result[f'attribute:{module_name}:{name}'] = value
  return result


def metadata(x):
  return [list(x.shape), list(x.stride()), str(x.dtype), str(x.device),
          x.storage_offset(), x.untyped_storage().nbytes(), x.requires_grad]


def _alias_contract(tensors):
  """Canonical storage groups and overlap components in O(n log n)."""
  groups = {}
  for key, tensor in tensors.items():
    storage = tensor.untyped_storage()
    row = groups.setdefault(storage._cdata, [[], storage.data_ptr(), storage.nbytes()])
    row[0].append(key)
  rows = [(sorted(keys), start, length) for keys, start, length in groups.values()]
  identity_groups = sorted(keys for keys, _, _ in rows if len(keys) > 1)
  intervals = sorted((start, start+length, keys) for keys, start, length in rows if length)
  components, active, end = [], [], 0
  def flush():
    if len(active) > 1:
      base = active[0][0]
      components.append(sorted([[keys, start-base] for start, _, keys in active]))
  for row in intervals:
    if active and row[0] >= end:
      flush()
      active = []
    active.append(row)
    end = max(end, row[1]) if len(active) > 1 else row[1]
  flush()
  return {'storage_groups': identity_groups, 'overlaps': sorted(components)}


def _binding_key(tensor):
  return (tensor.untyped_storage()._cdata, tuple(tensor.shape), tuple(tensor.stride()),
          tensor.dtype, tensor.device, tensor.storage_offset(),
          tensor.untyped_storage().nbytes(), tensor.requires_grad)


def export_capture(capture, owner, args):
  tape = capture._torch_replay_tape
  planned_state = False
  graph = capture._torch_execution_graph
  available = sources(owner, args)
  by_identity = {id(value): name for name, value in available.items()}
  by_layout = {_binding_key(value): name for name, value in available.items()}
  storage_sources = {}
  for name, value in available.items():
    if value.is_contiguous():
      start = value.storage_offset()*value.element_size()
      storage_sources.setdefault(value.untyped_storage()._cdata, []).append(
          (name, value, start, start+value.numel()*value.element_size()))
  bindings = {}
  input_bindings = {ref.id: f'input:{i}' for i, ref in enumerate(tape.input_refs) if isinstance(ref, _Ref)}
  for identity, tensor in tape.tensors.items():
    match = input_bindings.get(identity) or by_identity.get(id(tensor)) or by_layout.get(_binding_key(tensor))
    if match is None:
      start = tensor.storage_offset()*tensor.element_size()
      end = start + (0 if not tensor.numel() else 1+sum((n-1)*stride for n,stride in zip(tensor.shape,tensor.stride())))*tensor.element_size()
      match = next((name for name, value, low, high in storage_sources.get(tensor.untyped_storage()._cdata, ())
                    if low <= start and end <= high), None)
    if match is None:
      target_storage = tensor.untyped_storage()
      low = target_storage.data_ptr()
      high = low+target_storage.nbytes()
      access_low, access_high = tape.tensor_spans[identity]
      for name, value in available.items():
        storage = value.untyped_storage()
        begin = storage.data_ptr()
        logical = begin+value.storage_offset()*value.element_size()
        if (value.is_contiguous() and begin <= low and high <= begin+storage.nbytes()
            and logical <= access_low and access_high <= logical+value.numel()*value.element_size()):
          match = name
          break
    if match is None:
      uses = [str(x.function._schema) for x in tape.instructions if identity in repr(x.arguments)][:2]
      info = graph.tensors[identity]
      raise TemplateUnsupported(f'unbound external tensor {identity} {tuple(tensor.shape)} name={info.name} kind={info.kind} uses={uses}')
    source = available[match]
    bindings[identity] = [match, metadata(source), metadata(tensor),
                         tensor.untyped_storage().data_ptr()-source.untyped_storage().data_ptr(),
                         tensor.untyped_storage()._cdata == source.untyped_storage()._cdata]
  instructions, collectives = [], {}
  for ordinal,x in enumerate(tape.instructions):
    schema = x.function._schema
    arguments = x.arguments
    if schema.name in {'vecops_comm::run','vecops_comm::consume'}:
      from vecops.distributed._template import describe
      collectives[str(ordinal)] = describe(schema.name, arguments[0])
      arguments = (None,*arguments[1:])
    elif not _tracked_operator(x.function):
      raise TemplateUnsupported(f'operator implementation/owner is not tracked: {schema.name}')
    instructions.append([x.event_id, schema.name, schema.overload_name, str(schema),
                         encode(arguments), encode(x.result), x.matmul_precision])
  # Only lengths are needed; logical starts follow IR storage offsets. Allocator
  # address reuse between distinct temporary lifetimes is not an alias proof.
  spans = {key: end-start for key, (start, end) in tape.tensor_spans.items()}
  return {'version': 1, 'planned_state': planned_state, 'collectives': collectives,
              'pointwise_recipe':getattr(capture,'_pointwise_recipe',None),
              'graph': graph.to_dict(), 'bindings': bindings,
              'aliases': _alias_contract(tape.tensors),
              'input_aliases': _alias_contract({k:v for k,v in available.items() if k.startswith('input:')}),
              'instructions': instructions,
              'input_refs': encode(tape.input_refs), 'input_spec': treespec_dumps(tape.input_spec),
              'output_leaves': encode(tape.output_leaves), 'output_spec': treespec_dumps(tape.output_spec),
              'spans': spans}


def restore_capture(capture, payload, owner, args):
  _check_context()
  if payload['version'] != 1:
    raise ValueError('unsupported capture template version')
  session = None
  if bool(payload.get('planned_state')) != (session is not None):
    raise ValueError('template planner ownership context changed')
  graph = ExecutionGraph.from_dict(payload['graph'])
  available = sources(owner, args)
  if _alias_contract({k:v for k,v in available.items() if k.startswith('input:')}) != payload['input_aliases']:
    raise ValueError('template input storage alias contract changed')
  tensors = {}
  storage_views = {}
  for identity, (name, source_meta, target_meta, delta, shared) in payload['bindings'].items():
    source = available[name]
    if metadata(source) != source_meta:
      raise ValueError(f'template binding metadata changed: {name}')
    if shared and metadata(source) == target_meta:
      value = source
    else:
      shape, stride, dtype, device, offset, capacity, requires_grad = target_meta
      dtype = getattr(torch, dtype.removeprefix('torch.'))
      storage = source.untyped_storage()
      if not shared:
        storage_id = graph.tensors[identity].storage_id
        if storage_id not in storage_views:
          if delta < 0 or delta+capacity > storage.nbytes():
            raise ValueError('template buffer view exceeds current source storage')
          # Recreate a distinct StorageImpl over the current allocation. NumPy
          # supplies the buffer protocol only; no tensor contents are copied,
          # hashed, persisted or used to decide capture structure.
          if capacity:
            raw = torch.empty(0, dtype=torch.uint8, device='cpu').set_(storage, 0, (storage.nbytes(),), (1,))
            view = memoryview(raw.numpy())[delta:delta+capacity]
            storage_views[storage_id] = torch.frombuffer(view, dtype=torch.uint8).untyped_storage()
          else:
            storage_views[storage_id] = torch.empty(0, dtype=torch.uint8, device='cpu').untyped_storage()
        storage = storage_views[storage_id]
      value = torch.empty(0, dtype=dtype, device=device, requires_grad=requires_grad)
      value.set_(storage, offset, shape, stride)
    if metadata(value) != target_meta:
      raise ValueError(f'template view metadata changed: {name}')
    tensors[identity] = value
  if _alias_contract(tensors) != payload['aliases']:
    raise ValueError('template external storage alias contract changed')
  bases = {}
  for identity, value in tensors.items():
    storage = graph.tensors[identity].storage_id
    base = value.untyped_storage().data_ptr()
    if storage in bases and bases[storage] != base:
      raise ValueError('template storage binding changed')
    bases[storage] = base
  # Virtual spans for unbound temporaries are used only by conservative Python
  # alias analysis. Real executor addresses are allocated during lowering.
  virtual = 1 << 100
  for identity, storage in graph.storages.items():
    if identity not in bases:
      bases[identity] = virtual
      virtual += max(int(storage.bytes or 0), 1) + (1 << 40)
  tape = _Tape.__new__(_Tape)
  tape.capture = weakref.proxy(capture)
  tape.input_leaves, spec = tree_flatten((args, {}))
  tape.input_spec = treespec_loads(payload['input_spec'])
  if spec != tape.input_spec:
    raise ValueError('template input pytree changed')
  tape.input_refs = decode(payload['input_refs'])
  tape.output_spec = treespec_loads(payload['output_spec'])
  tape.output_leaves = decode(payload['output_leaves'])
  tape.tensors = tensors
  tape.storage_bindings = {k: x.detach() for k, x in tensors.items()}
  tape.tensor_spans = {}
  for identity, length in payload['spans'].items():
    tensor = graph.tensors[identity]
    start = bases[tensor.storage_id] + tensor.metadata.storage_offset * tensor.metadata.element_size
    tape.tensor_spans[identity] = (start, start + length)
  tape.instructions = []
  for event, name, overload, schema, arguments, result, precision in payload['instructions']:
    namespace, op = name.split('::')
    function = getattr(getattr(getattr(torch.ops, namespace), op), overload or 'default')
    if str(function._schema) != schema:
      raise ValueError(f'template operator schema changed: {name}')
    tape.instructions.append(_Instruction(event, function, decode(arguments), decode(result), precision))
  tape.module_owner = owner
  tape.native_operators = {}
  # Acquire leases from the current plan, after the caller has rebound current
  # model state. The payload never contains an arena owner or a plan pointer.
  tape.session_owner = session
  tape.session_scopes = None if session is None else list(session._scope)
  tape.arena_leases = []
  tape.produced = set()
  tape.produced_storage_origins = {}
  tape._collective_relocations = payload.get('collectives',{})
  if tape._collective_relocations:
    from vecops.distributed._template import rebind
    rebind(tape, graph)
  capture._pointwise_recipe=payload.get('pointwise_recipe')
  capture._torch_replay_tape = tape
  capture._torch_execution_graph = graph
  capture._finished = True
  return capture


def export_output_layout(values, owner, args):
  """Persist post-lowering slot geometry, including aliases but no payloads."""
  leaves,spec=tree_flatten(values)
  available=sources(owner,args)
  storage_sources={x.untyped_storage()._cdata:name for name,x in available.items()}
  objects,storages={},{}
  result=[]
  for value in leaves:
    if not isinstance(value,torch.Tensor):
      result.append({'value':encode(value)})
      continue
    storage=value.untyped_storage()._cdata
    result.append(dict(metadata=metadata(value),object=objects.setdefault(id(value),len(objects)),
        storage=storages.setdefault(storage,len(storages)),source=storage_sources.get(storage)))
  return dict(spec=treespec_dumps(spec),leaves=result)


def restore_metadata_capture(capture, payload, owner, args, *, output_layout=None):
  """Restore an address-free tape under strict FakeTensorMode, without replay.

  Only shared StorageImpl views are currently supported. Exotic overlapping
  buffers and communication operators conservatively request a live capture.
  Real current-state and input bindings are checked later by bind_metadata.
  """
  from torch._guards import detect_fake_mode
  from torch._subclasses.fake_tensor import is_fake
  mode = detect_fake_mode()
  if mode is None or not getattr(mode, '_vecops_metadata_no_numeric', False):
    raise ValueError('metadata template requires strict metadata_mode()')
  if payload['version'] != 1:
    raise ValueError('unsupported metadata template version')
  session = None
  if bool(payload.get('planned_state')) != (session is not None):
    raise ValueError('metadata template planner context changed')
  graph = ExecutionGraph.from_dict(payload['graph'])
  available = sources(owner, args)
  # Inputs are fresh metadata prototypes, whose distinct allocations cannot
  # reproduce unrelated overlapping StorageImpl objects. Model-owned real
  # state can share a planner arena across dtypes: validate those exact relative
  # overlap components before constructing Fake views, without reading bytes.
  if payload['input_aliases']['overlaps']:
    raise TemplateUnsupported('metadata input storage overlap requires live capture')
  state={identity:available[row[0]] for identity,row in payload['bindings'].items()
         if not row[0].startswith('input:')}
  selected=set(state)
  expected={'storage_groups':[], 'overlaps':[]}
  for group in payload['aliases']['storage_groups']:
    keys=sorted(selected.intersection(group))
    if len(keys)>1:expected['storage_groups'].append(keys)
  for component in payload['aliases']['overlaps']:
    rows=[(sorted(selected.intersection(keys)),offset)for keys,offset in component]
    rows=[(keys,offset)for keys,offset in rows if keys]
    if len(rows)>1:
      low=min(offset for _,offset in rows)
      expected['overlaps'].append(sorted([[keys,offset-low]for keys,offset in rows]))
  expected={key:sorted(value)for key,value in expected.items()}
  if _alias_contract(state)!=expected:
    raise ValueError('metadata template model-state physical alias contract changed')
  tensors = {}
  capture._metadata_sources = {}
  input_refs = decode(payload['input_refs'])
  leaves, spec = tree_flatten((args, {}))
  input_spec = treespec_loads(payload['input_spec'])
  if spec != input_spec:
    raise ValueError('metadata template input pytree changed')
  for identity, (name, source_meta, target_meta, delta, shared) in payload['bindings'].items():
    source = available[name]
    if metadata(source) != source_meta:
      raise ValueError(f'metadata template binding geometry changed: {name}')
    if not shared or delta:
      raise TemplateUnsupported('metadata template requires a shared storage view')
    fake = source if is_fake(source) else mode.from_tensor(source)
    shape, stride, dtype, device, offset, capacity, requires_grad = target_meta
    if metadata(fake) != target_meta:
      if capacity != fake.untyped_storage().nbytes() or str(fake.device) != device:
        raise ValueError('metadata template source storage capacity changed')
      fake = torch.empty(0, dtype=getattr(torch, dtype.removeprefix('torch.')),
                         device=device, requires_grad=requires_grad).set_(
                             fake.untyped_storage(), offset, shape, stride)
    if not is_fake(source):
      capture._metadata_sources[fake.untyped_storage()._cdata] = source
    elif identity not in {x.id for x in input_refs if isinstance(x, _Ref)}:
      raise TemplateUnsupported('metadata template needs an explicit input view binding')
    tensors[identity] = fake
  groups = {}
  for identity, tensor in tensors.items():
    groups.setdefault(tensor.untyped_storage()._cdata, []).append(identity)
  if sorted(sorted(v) for v in groups.values() if len(v)>1) != payload['aliases']['storage_groups']:
    raise ValueError('metadata template external alias contract changed')
  input_groups = {}
  for name, tensor in available.items():
    if name.startswith('input:'):
      input_groups.setdefault(tensor.untyped_storage()._cdata, []).append(name)
  if sorted(sorted(v) for v in input_groups.values() if len(v)>1) != payload['input_aliases']['storage_groups']:
    raise ValueError('metadata template input alias contract changed')
  tape = _Tape.__new__(_Tape)
  tape.capture = weakref.proxy(capture)
  tape.input_leaves, tape.input_spec, tape.input_refs = leaves, input_spec, input_refs
  tape.output_spec = treespec_loads(payload['output_spec'])
  tape.output_leaves = decode(payload['output_leaves'])
  tape.tensors = tensors
  tape.storage_bindings = {}
  tape.tensor_spans = {}
  bases = {key: (i+1)*(1<<100) for i,key in enumerate(graph.storages)}
  for identity, length in payload['spans'].items():
    info = graph.tensors[identity]
    start = bases[info.storage_id] + info.metadata.storage_offset*info.metadata.element_size
    tape.tensor_spans[identity] = (start, start+length)
  tape.instructions = []
  for event,name,overload,schema,arguments,result,precision in payload['instructions']:
    namespace,op = name.split('::')
    function = getattr(getattr(getattr(torch.ops,namespace),op),overload or 'default')
    ordinal = len(tape.instructions)
    recipe = payload.get('collectives',{}).get(str(ordinal))
    if recipe is not None:
      if name not in {'vecops_comm::run','vecops_comm::consume'} or recipe.get('operation') != name:
        raise ValueError('metadata collective relocation schema changed')
    elif not _tracked_operator(function):
      raise TemplateUnsupported('metadata template operator requires live capture')
    if str(function._schema) != schema:
      raise ValueError(f'metadata template operator schema changed: {name}')
    tape.instructions.append(_Instruction(event,function,decode(arguments),decode(result),precision))
  tape.module_owner, tape.native_operators = owner, {}
  tape._collective_relocations = payload.get('collectives',{})
  tape.session_owner, tape.session_scopes = session, None if session is None else list(session._scope)
  tape.arena_leases = []
  tape.produced, tape.produced_storage_origins = set(), {}
  tape.metadata_only = True
  capture._torch_replay_tape, capture._torch_execution_graph = tape, graph
  capture._finished = True
  if output_layout is not None:
    # A lowered output slot can have more capacity than its logical Fake
    # intermediate. The steady tape consumes that actual initializer geometry.
    roots,objects={},{}
    result=[]
    for row in output_layout['leaves']:
      if 'value' in row:
        result.append(decode(row['value']))
        continue
      if row['object'] not in objects:
        shape,strides,dtype,device,offset,capacity,requires_grad=row['metadata']
        if row['storage'] not in roots:
          if row['source'] is not None:
            source=available[row['source']]
            fake=source if is_fake(source) else mode.from_tensor(source)
            if fake.untyped_storage().nbytes()!=capacity:
              raise ValueError('metadata output source capacity changed')
            roots[row['storage']]=fake.untyped_storage()
          else:
            roots[row['storage']]=torch.empty(capacity,dtype=torch.uint8,device=device).untyped_storage()
        objects[row['object']]=torch.empty(0,dtype=getattr(torch,dtype.removeprefix('torch.')),
            device=device,requires_grad=requires_grad).set_(roots[row['storage']],offset,shape,strides)
      result.append(objects[row['object']])
    from torch.utils._pytree import tree_unflatten
    return tree_unflatten(result,treespec_loads(output_layout['spec']))
  # Only metadata storage is constructed; initializer outputs carry explicit
  # slots and remain input-dependent computations at the first real replay.
  roots = {graph.tensors[k].storage_id:v.untyped_storage() for k,v in tensors.items()}
  def output(value):
    if not isinstance(value,_Ref):
      return value
    if value.id in tensors:
      return tensors[value.id]
    info = graph.tensors[value.id];m=info.metadata
    if info.storage_id not in roots:
      roots[info.storage_id]=torch.empty(graph.storages[info.storage_id].bytes,
          dtype=torch.uint8,device=m.device).untyped_storage()
    return torch.empty(0,dtype=getattr(torch,m.dtype.removeprefix('torch.')),device=m.device).set_(
        roots[info.storage_id],m.storage_offset,m.shape,m.strides)
  from torch.utils._pytree import tree_unflatten
  return tree_unflatten([output(x) for x in tape.output_leaves],tape.output_spec)
