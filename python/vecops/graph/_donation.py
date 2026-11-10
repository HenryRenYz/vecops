# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Cold ownership/liveness proof for audited native inplace contracts.

Only a private planning IR changes. Storage lifetimes and accessed byte spans
follow the union of every donated logical version. Eager schemas stay intact.
"""
from dataclasses import replace


def plan_reuse(execution, instructions, pointwise, mutable, protected, spans,
               native_contracts=None, native_outputs=None, factories=()):
  # Only storage membership, tensor.storage_id and storage lifetime/capacity
  # change below. Events, metadata and annotations are read-only during replay
  # lowering; recursively cloning them dominates large-graph binding time.
  execution = replace(execution,
      tensors={key:replace(value) for key,value in execution.tensors.items()},
      storages={key:replace(value,tensors=list(value.tensors))
                for key,value in execution.storages.items()})
  spans = dict(spans)
  native_contracts, native_outputs = native_contracts or {}, native_outputs or {}
  events = {event.id: event.index for event in execution.events}
  owned, donations, replacements, rewritten = set(), {}, {}, {}
  factory_ids = set(factories)
  # These schema arguments describe geometry only. In particular AF3 returns
  # output.reshape_as(old_input), which must not keep old input *data* alive.
  # Keep self reads, writes, unknown operators and escaping aliases conservative.
  template_events = {instruction.event_id for instruction in instructions
                     if instruction.function._schema.name in {
                         'aten::view_as', 'aten::reshape_as', 'aten::expand_as'}}
  def accesses_data(event, use):
    return (use.access.value not in {'alias', 'allocate'} and not (
        event.id in template_events and use.argument == 'other' and use.access.value == 'read'))
  last_data_use = {}
  first_storage_use = {}
  for event in execution.events:
    for use in event.uses:
      if accesses_data(event, use):last_data_use[use.tensor_id] = event.index
      if use.access.value not in {'alias','allocate'}:
        storage = execution.tensors[use.tensor_id].storage_id
        first_storage_use[storage] = min(first_storage_use.get(storage,event.index),event.index)

  def signature(tensor):
    m = tensor.metadata
    return m.shape, m.strides, m.dtype, m.device, m.storage_offset

  def references(value):
    if hasattr(value, 'id'):
      yield value
    elif isinstance(value, (list, tuple)):
      for child in value:
        yield from references(child)

  for ordinal, instruction in enumerate(instructions):
    index = events[instruction.event_id]
    candidates = []
    if ordinal in pointwise and ordinal not in mutable:
      _, a, b, output, _ = pointwise[ordinal]
      candidates.extend((source, output, (a, b), True) for source in (a, b))
    for source_index, output_index in native_contracts.get(ordinal, ()):
      source, output = instruction.arguments[source_index], instruction.arguments[output_index]
      if not hasattr(source, 'id') or not hasattr(output, 'id') or output.id not in factory_ids:
        continue
      operands = tuple(ref for i, value in enumerate(instruction.arguments) if i != output_index
                       for ref in references(value))
      candidates.append((source, output, operands, False))
    for source, output, operands, same_index in candidates:
      if source.id in protected or output.id in protected:
        continue
      tensor, result = execution.tensors[source.id], execution.tensors[output.id]
      storage, result_storage = execution.storages[tensor.storage_id], execution.storages[result.storage_id]
      # Views inherit storage ownership. Tracking only the producing Tensor ID
      # unnecessarily rejects the reshape/view chains used by native kernels.
      if not any(identity in owned for identity in storage.tensors):
        continue
      if (storage.id == result.storage_id or storage.external or storage.persistent
          or tensor.metadata.requires_grad or result.metadata.requires_grad
          or result_storage.external or result_storage.persistent
          or any(execution.tensors[x].escapes for x in storage.tensors)
          or any(x in protected for x in storage.tensors + result_storage.tensors)
          or max((last_data_use.get(identity, -1) for identity in storage.tensors), default=-1) != index
          or signature(tensor) != signature(result)):
        continue
      # Native contracts exclude *all* other aliases. Same-index pointwise
      # kernels also support an identical second input view, but no broadcasts.
      alias_operands = [ref for ref in operands if execution.tensors[ref.id].storage_id == storage.id]
      if any((not same_index and ref is not source) or
             signature(execution.tensors[ref.id]) != signature(tensor) for ref in alias_operands):
        continue
      # An out-style buffer may exist earlier, but must still be uninitialized.
      # Even metadata uses stay conservative here: redirecting an out buffer
      # before the input view is defined can create a forward SSA reference.
      if not same_index and first_storage_use.get(result_storage.id,index) < index:
        continue
      delta = spans[source.id][0] - spans[output.id][0]
      for identity in result_storage.tensors:
        execution.tensors[identity].storage_id = storage.id
        low, high = spans[identity]
        spans[identity] = low + delta, high + delta
      storage.tensors.extend(result_storage.tensors)
      storage.lifetime_end = max(storage.lifetime_end, result_storage.lifetime_end)
      storage.bytes = max(storage.bytes, result_storage.bytes)
      if result_storage.id in first_storage_use:
        first = first_storage_use.pop(result_storage.id)
        first_storage_use[storage.id] = min(first_storage_use.get(storage.id,first),first)
      del execution.storages[result_storage.id]
      replacements[output.id] = source
      if same_index:
        compiled, a, b, _, alpha = pointwise[ordinal]
        pointwise[ordinal] = compiled, a, b, source, alpha
        donations[ordinal] = output, source
      else:
        rewritten[ordinal] = tuple(source if hasattr(value, 'id') and value.id == output.id else value
                                   for value in instruction.arguments)
      break
    if ordinal in pointwise and ordinal not in mutable:
      # Track the logical produced version, including a donated result.
      original = donations.get(ordinal)
      owned.add(original[0].id if original else pointwise[ordinal][3].id)
    owned.update(ref.id for ref in native_outputs.get(ordinal, ()))
  return execution, donations, replacements, rewritten, spans
