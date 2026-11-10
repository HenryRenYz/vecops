# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""FakeTensor executable recording and storage-only runtime binding.

Recording never submits a CPU kernel. Binding allocates/views storage, without
running the recorded program; input-dependent values are computed by replay.
Unknown external tensors are rejected rather than filled with invented data.
"""
from dataclasses import replace

import torch
from torch.utils._pytree import tree_flatten, tree_unflatten
from torch._subclasses.fake_tensor import is_fake

from ._replay import _Tape, _Ref


def metadata_mode():
  """Disable CPU fallback and tensor constant folding during shape tracing."""
  from torch._subclasses.fake_tensor import FakeTensorMode
  class MetadataMode(FakeTensorMode):
    def may_turn_const(self, tensor):
      return False
  mode = MetadataMode(allow_non_fake_inputs=True, allow_fallback_kernels=False)
  mode._vecops_metadata_no_numeric = True
  return mode


def capture_metadata(capture, function, args, kwargs):
  from torch._guards import detect_fake_mode
  from .torch import record_torch, register_inputs, register_module_state, mark_graph_outputs
  from ._replay import _reject_tensor_data_escapes
  mode = detect_fake_mode()
  if mode is None or not getattr(mode, '_vecops_metadata_no_numeric', False):
    raise RuntimeError('metadata capture requires the strict metadata_mode() context')
  if capture.events or capture._finished:
    raise RuntimeError('metadata capture requires a fresh graph')
  owner = function if hasattr(function, 'named_parameters') else getattr(function, '__self__', None)
  owner = getattr(owner, 'owner', owner)
  capture._metadata_sources = {}
  # Fake conversion caches are weak. Keep these metadata-only owners alive:
  # otherwise StorageImpl addresses can be recycled during this loop, and a
  # later source silently overwrites another parameter's alias identity.
  capture._metadata_source_fakes = []
  if owner is not None:
    from ._template import sources
    for real in sources(owner, ()).values():
      if not is_fake(real):
        fake = mode.from_tensor(real)
        capture._metadata_source_fakes.append(fake)
        capture._metadata_sources[fake.untyped_storage()._cdata] = real
  with capture:
    register_inputs(capture, (args, kwargs))
    if owner is not None:
      register_module_state(capture, owner, fake_mode=mode)
    tape = _Tape(capture, args, kwargs, metadata_only=True)
    tape.module_owner = owner
    capture._torch_replay_tape = tape
    with torch.inference_mode(), record_torch(capture), _reject_tensor_data_escapes():
      result = function(*args, **kwargs)
    mark_graph_outputs(capture, result)
    leaves, tape.output_spec = tree_flatten(result)
    tape.output_leaves = tape.encode(leaves)
  capture._torch_execution_graph = capture.finish()
  return result


def _view(storage, tensor, *, offset=None):
  return torch.empty(0, dtype=tensor.dtype, device='cpu').set_(storage,
      tensor.storage_offset() if offset is None else offset, tuple(tensor.shape), tuple(tensor.stride()))


def prototypes(values, shared=None):
  """Allocate uninitialized shape/alias prototypes; never read their contents."""
  shared = {} if shared is None else shared
  def one(value):
    if not isinstance(value, torch.Tensor) or not is_fake(value):
      return value
    storage = value.untyped_storage()
    if storage._cdata not in shared:
      shared[storage._cdata] = torch.empty(storage.nbytes(), dtype=torch.uint8, device='cpu').untyped_storage()
    return _view(shared[storage._cdata], value)
  leaves, spec = tree_flatten(values)
  return tree_unflatten([one(x) for x in leaves], spec)


def bind_metadata(capture, args, kwargs=None):
  """Replace metadata-only external bindings with current real storage."""
  tape = capture._torch_replay_tape
  if not getattr(tape, 'metadata_only', False):
    raise ValueError('not a metadata executable capture')
  kwargs = kwargs or {}
  leaves, spec = tree_flatten((args, kwargs))
  if spec != tape.input_spec:
    raise ValueError('metadata binding pytree changed')
  execution = capture._torch_execution_graph
  bindings = {ref.id:real for ref, real in zip(tape.input_refs, leaves) if isinstance(ref, _Ref)}
  roots = {}
  for identity, fake in tape.tensors.items():
    if identity in bindings:
      real = bindings[identity]
    elif not is_fake(fake):
      real = fake
    else:
      source = capture._metadata_sources.get(fake.untyped_storage()._cdata)
      if source is None:
        # FakeTensor preserves a real constant only for explicit literal
        # construction, not general input-dependent arithmetic.
        source = getattr(fake, 'constant', None)
        if source is None:
          raise ValueError(f'unbound metadata external tensor {identity}: {tuple(fake.shape)}')
      real = _view(source.untyped_storage(), fake)
    if tuple(real.shape) != tuple(fake.shape) or real.dtype != fake.dtype or tuple(real.stride()) != tuple(fake.stride()):
      raise ValueError(f'metadata binding geometry changed: {identity}')
    bindings[identity] = real
    info = execution.tensors[identity]
    old_offset = info.metadata.storage_offset
    root = (real.untyped_storage().data_ptr(), real.storage_offset()-old_offset,
            real.untyped_storage().nbytes())
    previous = roots.setdefault(info.storage_id,root)
    if previous != root:
      raise ValueError('metadata input/state storage alias contract changed')
  for identity, info in execution.tensors.items():
    root = roots.get(info.storage_id)
    if root is None:
      continue
    base, delta, capacity = root
    info.metadata = replace(info.metadata, storage_offset=info.metadata.storage_offset+delta)
    execution.storages[info.storage_id].bytes = capacity
    length = tape.tensor_spans[identity][1]-tape.tensor_spans[identity][0] if identity in tape.tensor_spans else 0
    start = base+info.metadata.storage_offset*info.metadata.element_size
    tape.tensor_spans[identity] = (start,start+length)
  tape.input_leaves = leaves
  tape.tensors = bindings
  tape.storage_bindings = {key:value.detach() for key,value in bindings.items()}
  tape.session_owner = None
  tape.session_scopes = None
  tape.arena_leases = []
  if getattr(tape, '_collective_relocations', None):
    from vecops.distributed._template import rebind
    rebind(tape, execution)
  return capture


def output_prototypes(execution, tape, values, graph, examples):
  """Use fixed output slots where lowered; otherwise allocate metadata storage."""
  if hasattr(tape,'resource_output_prototypes'):
    return tape.resource_output_prototypes
  input_positions = {value:i for i,value in enumerate(graph.inputs())}
  roots = {}
  for identity, value in values.items():
    if value in input_positions:
      real = examples[input_positions[value]]
      info = execution.tensors[identity]
      roots.setdefault(info.storage_id, (real.untyped_storage(), real.storage_offset()-info.metadata.storage_offset))
  result = []
  for ref in tape.output_leaves:
    if not isinstance(ref, _Ref):
      result.append(ref)
      continue
    info = execution.tensors[ref.id]
    m = info.metadata
    if info.storage_id not in roots:
      roots[info.storage_id] = (torch.empty(execution.storages[info.storage_id].bytes,dtype=torch.uint8, device='cpu').untyped_storage(),0)
    storage, delta = roots[info.storage_id]
    result.append(torch.empty(0,dtype=getattr(torch,m.dtype.removeprefix('torch.')), device='cpu').set_(
        storage,m.storage_offset+delta,m.shape,m.strides))
  return tree_unflatten(result,tape.output_spec)

class MetadataRegionProjection:
  """Record a region tape from an existing semantic trace, without rerunning it.

  The parent remains the sole GraphCapture and dispatch recorder. This object
  only adds the executable instruction view of the same events. No parent
  tensor/lifetime records are modified when the regional view is finalized.
  """
  def __init__(self, parent, owner, args, kwargs=None, *, name='metadata-region'):
    from torch._guards import detect_fake_mode
    from ._template import sources
    mode = detect_fake_mode()
    if mode is None or not getattr(mode, '_vecops_metadata_no_numeric', False):
      raise ValueError('region projection requires strict metadata_mode()')
    if getattr(parent, '_torch_replay_tape', None) is not None:
      raise ValueError('region projections cannot overlap')
    self.parent, self.owner, self.name = parent, owner, name
    self.args, self.kwargs = args, kwargs or {}
    self.start = len(parent.events)
    self.available = sources(owner, args)
    # Convert state under the same mode as the full model, preserving aliases.
    # These are metadata objects; no tensor constants or CPU fallback is used.
    self.symbols = [(key, value, value if is_fake(value) else mode.from_tensor(value))
                    for key, value in self.available.items()]
    self.tape = _Tape(parent, args, self.kwargs, metadata_only=True)
    self.tape.module_owner = owner
    self.tape._region_projection = True
    self.finished = False
    parent._torch_replay_tape = self.tape

  def record_native(self, registration, arguments):
    """Attach an unmaterialized source call to the event collection just emitted."""
    if getattr(self.parent, '_torch_recording_dispatch_depth', 0):
      return
    from types import SimpleNamespace
    from .. import _schema_bridge
    signature = _schema_bridge._bridge_signature_identity(registration)
    schema = torch._C.parse_schema(signature.replace('vecops_signature(',
        f'{registration.library}::{registration.name}(', 1))
    function = SimpleNamespace(_schema=schema, tags=(), _region_unresolved=True)
    event = self.parent.events[-1]
    if event.kind != 'operator' or event.name != schema.name:
      raise ValueError('source observer is not attached to its semantic event')
    outputs = [value for parameter, value in zip(registration.kernel_def.inputs, arguments)
               if getattr(parameter, 'output', False)]
    result = (outputs[0] if len(outputs)==1 else tuple(outputs)) if registration.return_outputs else None
    self.tape.record(function, tuple(arguments), {}, result, event)

  def finish(self, result):
    from dataclasses import replace
    import copy
    from .capture import GraphCapture
    from .ir import TensorKind
    if self.finished or self.parent._torch_replay_tape is not self.tape:
      raise ValueError('metadata region projection is not active')
    # Result encoding must happen while parent tensor objects are still live.
    leaves, self.tape.output_spec = tree_flatten(result)
    self.tape.output_leaves = self.tape.encode(leaves)
    del self.parent._torch_replay_tape
    self.finished = True
    instruction_events = {instruction.event_id for instruction in self.tape.instructions}
    region_events = self.parent.events[self.start:]
    region_event_ids = {event.id for event in region_events}
    events = copy.deepcopy([event for event in region_events
                            if event.kind != 'operator' or event.id in instruction_events])
    # A projection removes request-only/router factory events. Give the
    # detached view its own dense timeline: sampled loop ranges and liveness
    # operate on these indexes, while stable event IDs keep every reference.
    for index, event in enumerate(events, start=1):
      event.index = index
    event_ids = {event.id for event in events}
    selected = set(self.tape.tensors)
    selected.update(ref.id for ref in self.tape.output_leaves if isinstance(ref, _Ref))
    for event in events:
      if event.kind == 'operator':
        selected.update(use.tensor_id for use in event.uses)
        selected.update(event.outputs)
    pending = list(selected)
    while pending:
      alias = self.parent.tensors[pending.pop()].alias_of
      if alias is not None and alias not in selected:
        selected.add(alias); pending.append(alias)
    for event in events:
      event.uses = [use for use in event.uses if use.tensor_id in selected]
      event.outputs = [key for key in event.outputs if key in selected]
    capture = GraphCapture(self.name, attributes=dict(self.parent.attributes))
    capture.events = events
    capture.tensors = {key:replace(self.parent.tensors[key],
        annotations=dict(self.parent.tensors[key].annotations)) for key in self.parent.tensors if key in selected}
    external = set(self.tape.tensors)
    outputs = {ref.id for ref in self.tape.output_leaves if isinstance(ref, _Ref)}
    produced_events={}
    def produced(value,event):
      if isinstance(value,_Ref):produced_events.setdefault(value.id,event)
      elif isinstance(value,(tuple,list)):
        for item in value:produced(item,event)
      elif isinstance(value,dict):
        for item in value.values():produced(item,event)
    for instruction in self.tape.instructions:
      produced(instruction.result,instruction.event_id)
    for key, tensor in capture.tensors.items():
      if tensor.producer_event not in instruction_events and key not in external and key in produced_events:
        tensor.producer_event=produced_events[key]
      if tensor.producer_event not in event_ids:
        tensor.producer_event = None
        if tensor.kind not in (TensorKind.parameter, TensorKind.buffer):
          tensor.kind = TensorKind.input
      # Escapes/retention outside this region are not outputs of this tape.
      tensor.escapes = key in outputs
    storage_ids = dict.fromkeys(t.storage_id for t in capture.tensors.values())
    event_order = {event.id:event.index for event in events}
    for key in storage_ids:
      old = self.parent.storages[key]
      allocated_here = old.allocation_event in region_event_ids
      allocation = old.allocation_event if old.allocation_event in event_ids else None
      if allocated_here and allocation is None:
        # A router can create its output through a Torch factory
        # hidden below the outer dispatcher op. Filtering that factory must
        # not turn a region-owned functional result into external storage.
        producers = {capture.tensors[item].producer_event for item in old.tensors
                     if item in selected and capture.tensors[item].producer_event in event_ids}
        allocation = min(producers,key=event_order.__getitem__) if producers else None
      capture.storages[key] = replace(old,
          tensors=[t for t in old.tensors if t in selected],
          allocation_event=allocation,
          external=old.external or not allocated_here or any(t in external for t in old.tensors),
          annotations=dict(old.annotations))
    capture.loops = [copy.deepcopy(loop)for loop in self.parent.loops
                     if loop.begin_event in event_ids and loop.end_event in event_ids]
    capture.inputs = list(self.tape.tensors)
    capture.outputs = list(dict.fromkeys(ref.id for ref in self.tape.output_leaves if isinstance(ref, _Ref)))
    capture._torch_execution_graph = capture.finish()
    capture._torch_replay_tape = self.tape
    import weakref
    self.tape.capture = weakref.proxy(capture)
    self.tape._region_source_bindings = self._source_bindings()
    self.tape._region_owner = self.owner
    self.capture = capture
    return capture

  def _source_bindings(self):
    """Resolve entering state to logical owner paths before arena relocation."""
    from ._template import TemplateUnsupported
    inputs = {ref.id for ref in self.tape.input_refs if isinstance(ref, _Ref)}
    result = {}
    by_object={}
    by_storage={}
    def span(tensor):
      begin=tensor.storage_offset()
      length=0 if not tensor.numel() else 1+sum((n-1)*stride for n,stride in zip(tensor.shape,tensor.stride()))
      return begin,begin+length
    for row in self.symbols:
      name,source,fake=row
      for value in (source,fake):
        by_object[id(value)]=row
        key=(is_fake(value),value.untyped_storage()._cdata,value.dtype)
        by_storage.setdefault(key,[]).append((row,value,span(value)))
    for identity, tensor in self.tape.tensors.items():
      if identity in inputs:
        continue
      match=by_object.get(id(tensor))
      if match is None:
        low,high=span(tensor)
        key=(is_fake(tensor),tensor.untyped_storage()._cdata,tensor.dtype)
        match=next((row for row,value,(start,end)in by_storage.get(key,())
                    if start<=low and high<=end),None)
      if match is None:
        raise TemplateUnsupported(f'regional state has no symbolic binding: {identity}')
      name,source,fake=match
      candidate=fake if is_fake(tensor) else source
      delta=tensor.storage_offset()-candidate.storage_offset()
      result[identity]=(name,tuple(source.shape),tuple(source.stride()),source.dtype,
                        source.device,delta,tuple(tensor.shape),tuple(tensor.stride()))
    return result

  def abort(self):
    if getattr(self.parent, '_torch_replay_tape', None) is self.tape:
      del self.parent._torch_replay_tape
    self.finished = True


def bind_projected_sources(capture, args=()):
  """Rebind an in-memory discovery tape after parameter arena relocation."""
  from ._template import sources
  tape = capture._torch_replay_tape
  bindings = getattr(tape, '_region_source_bindings', None)
  if bindings is None:
    return
  available = sources(tape._region_owner, args)
  for identity, (name, shape, strides, dtype, device, delta, view_shape, view_strides) in bindings.items():
    source = available[name]
    if (tuple(source.shape),tuple(source.stride()),source.dtype,source.device) != (shape,strides,dtype,device):
      raise ValueError('discovered region state geometry changed before binding')
    tape.tensors[identity] = torch.as_strided(source, view_shape, view_strides,
                                            source.storage_offset()+delta)
  resolve_projected_functions(capture)


def resolve_projected_functions(capture):
  tape=capture._torch_replay_tape
  for instruction in tape.instructions:
    function = instruction.function
    if getattr(function, '_region_unresolved', False):
      schema = function._schema
      namespace, name = schema.name.split('::', 1)
      actual = getattr(getattr(getattr(torch.ops, namespace), name), schema.overload_name or 'default')
      if str(actual._schema) != str(schema):
        raise ValueError('source operator schema changed during discovery')
      instruction.function = actual
