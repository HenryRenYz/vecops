# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Logical collective recipes; process-local handles never enter graph caches.

Only layout-only prepared run/consume plans are relocatable. Existing groups
must already exist; restoring a cache never creates a collective process group.
PageExchange has external mapping lifetime and deliberately remains unsupported.
"""
from dataclasses import replace
import weakref

_plans = {}


def group_contract(group=None):
  """Logical membership/platform attestation without a process group address."""
  import torch.distributed as dist
  if not dist.is_initialized():
    return None
  group = group or dist.distributed_c10d._get_default_group()
  name = str(dist.get_backend(group)).lower()
  context = None
  if name == 'vecops':
    from . import backend
    import hashlib
    context = hashlib.sha256(backend(group).tuning_context.encode()).hexdigest()
  return dict(ranks=dist.get_process_group_ranks(group), backend=name,
              default=group is dist.distributed_c10d._get_default_group(), context=context)


def remember(operation, handle, group, **recipe):
  import torch.distributed as dist
  group = group or dist.distributed_c10d._get_default_group()
  _plans[(operation, int(handle))] = (weakref.ref(group), recipe)


def routing_for(operation,handle):
  """Return the current group's cold compiler; no process handle is persisted."""
  entry=_plans.get((operation,int(handle)))
  if entry is None or entry[0]() is None:return None
  from . import backend
  from ._routing import for_backend
  return for_backend(backend(entry[0]()))


def describe(operation, handle):
  import torch.distributed as dist
  entry = _plans.get((operation, int(handle)))
  if entry is None or entry[0]() is None:
    raise ValueError('collective plan has no relocatable logical recipe')
  group = entry[0]()
  return dict(version=1, operation=operation, **group_contract(group), **entry[1])


def group_for(recipe):
  import torch.distributed as dist
  if recipe.get('version') != 1 or recipe.get('backend') != 'vecops':
    raise ValueError('unsupported collective template contract')
  ranks = recipe['ranks']
  if not isinstance(ranks, list) or not ranks or any(type(x) is not int for x in ranks):
    raise ValueError('invalid collective template membership')
  matches = [group for group, membership in dist.distributed_c10d._world.pg_group_ranks.items()
             if [rank for rank, _ in sorted(membership.items(), key=lambda x:x[1])] == ranks
             and str(dist.get_backend(group)).lower() == recipe['backend']]
  # Multiple groups with identical ranks can have different routing/epochs.
  # Default world is unambiguous; subgroups need a unique existing binding.
  world = dist.distributed_c10d._get_default_group()
  if recipe.get('default') is True:
    if world not in matches:
      raise ValueError('collective template world membership changed')
    selected = world
  else:
    matches = [group for group in matches if group is not world]
    if len(matches) != 1:
      raise ValueError('collective template requires an unambiguous existing group')
    selected = matches[0]
  if group_contract(selected)['context'] != recipe.get('context'):
    raise ValueError('collective template platform/runtime contract changed')
  return selected


def rebind(tape, execution):
  """Create local metadata plans, never submit a collective or inspect values."""
  import torch
  from . import backend, native_operators, load_communication
  from vecops import _schema_bridge
  from vecops.graph._replay import _Ref

  def meta(value):
    if not isinstance(value, _Ref):
      raise ValueError('collective template tensor reference changed')
    m = execution.tensors[value.id].metadata
    return torch.empty_strided(m.shape, m.strides, dtype=getattr(torch,m.dtype.removeprefix('torch.')),
                               device='meta')

  relocations = []
  seen = set()
  for ordinal, recipe in getattr(tape, '_collective_relocations', {}).items():
    index = int(ordinal)
    if (str(index) != str(ordinal) or not 0 <= index < len(tape.instructions)
        or index in seen):
      raise ValueError('invalid collective relocation instruction index')
    seen.add(index)
    instruction = tape.instructions[index]
    operation = instruction.function._schema.name
    if operation != recipe['operation'] or instruction.arguments[0] is not None:
      raise ValueError('collective relocation instruction changed')
    relocations.append((index, recipe))
  # Canonical JSON sorts string keys ("10" before "2"). Preparing in that
  # order changes native descriptor/allocation order relative to live capture.
  # Validate indices before creating any plan, then follow instruction order.
  for index, recipe in sorted(relocations, key=lambda item: item[0]):
    instruction = tape.instructions[index]
    operation = instruction.function._schema.name
    group = group_for(recipe)
    arguments = instruction.arguments
    if operation == 'vecops_comm::run':
      outputs, inputs = [meta(x) for x in arguments[1]], [meta(x) for x in arguments[2]]
      options = dict(recipe['options'])
      levels = options.pop('copy_levels')
      if levels is None:
        handle = backend(group).prepare(recipe['kind'], outputs, inputs,
            options['output_splits'], options['input_splits'], options['root'])
      else:
        handle = backend(group).prepare_routed(recipe['kind'], outputs, inputs, levels)
      remember(operation, handle, group, kind=recipe['kind'], options=recipe['options'])
    elif operation == 'vecops_comm::consume':
      name = recipe['operator_name']
      registration = _schema_bridge._registrations.get(tuple(name.split('::',1)))
      if registration is None or not registration.materialized or not registration.library_path:
        raise ValueError('collective consumer native registration is unavailable')
      handle = backend(group).prepare_consumer(meta(arguments[2]), meta(arguments[1]),
                                               name, recipe['specs'])
      remember(operation, handle, group, operator_name=name, specs=recipe['specs'])
      tape.native_operators[operation] = (load_communication(), group, registration)
    else:
      raise ValueError('unsupported collective relocation operation')
    tape.native_operators.update({k:v for k,v in native_operators(group).items()
                                  if k not in tape.native_operators})
    tape.instructions[index] = replace(instruction, arguments=(handle,*arguments[1:]))
  tape._collective_relocations = {}


def record_deferred_consumer(source, output, operator_name, specs, group, owners):
  """Collect a consumer kernel and logical effect before its DSO is published."""
  from .. import _precompile, _schema_bridge
  from ..graph import current_capture, AccessKind
  capture = current_capture()
  tape = getattr(capture, '_torch_replay_tape', None) if capture is not None else None
  if (_precompile._active_collector is None
      or not getattr(tape, '_region_projection', False)):
    return False
  import torch
  import torch.distributed as dist
  from .. import TensorMeta
  from ..graph.torch import register_tensor
  group = group or dist.distributed_c10d._get_default_group()
  contract = group_contract(group)
  if not contract or contract['backend'] != 'vecops':
    raise ValueError('deferred consumer requires an initialized native group')
  registration = _schema_bridge._registrations.get(tuple(operator_name.split('::',1)))
  if registration is None or len(registration.kernel_def.inputs) != 2:
    raise ValueError('deferred consumer has no validated two-tensor declaration')
  symbols = _schema_bridge._ordered_values(registration.kernel_def, registration.spec_order)
  if len(symbols) != len(specs):
    raise ValueError('deferred consumer specialization contract changed')
  addresses = TensorMeta((len(contract['ranks']),), dtype='int64')
  target = TensorMeta(tuple(output.shape), strides=tuple(output.stride()), dtype=output.dtype)
  # Compiler work is not part of either semantic/executable model view.
  from ..graph._discovery import _compiler_bookkeeping
  with _compiler_bookkeeping():
    from .. import _C
    values={key:_C.DType(value) if registration.kernel_def.values[key]==_C.DTypeValue else value
            for key,value in zip(symbols,specs)}
    _precompile.maybe_collect_call(registration.operator, registration.kernel_def,
        (addresses,target), values, name=operator_name, source=registration.source)
  for tensor in (source,output):
    try:capture.tensor_id(tensor)
    except KeyError:register_tensor(capture,tensor,kind='external',external=True)
  from . import load_communication
  function = torch.ops.vecops_comm.consume.default
  event = capture.record_operator(function._schema.name,
      inputs=[(output,AccessKind.read_write,'output'),(source,AccessKind.read,'input')],outputs=[],
      attributes={'schema':str(function._schema),'deferred_collective':True})
  ordinal = len(tape.instructions)
  tape.record(function,(None,output,source),{},None,event)
  relocations = getattr(tape,'_collective_relocations',None)
  if relocations is None:relocations=tape._collective_relocations={}
  relocations[str(ordinal)] = dict(version=1, operation=function._schema.name,
      **contract, operator_name=operator_name, specs=list(specs))
  tape.native_operators[function._schema.name] = (load_communication(),group,tuple(owners))
  return True
