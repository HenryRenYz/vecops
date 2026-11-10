# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Validation and human-readable facts for resource memory programs."""

def validate_plan(program,queries,view):
  queries=getattr(queries,'native',queries)
  queries.validate_recipe(view,program.memory)
  slots={slot.id:slot for slot in program.memory.addresses.slots}
  for identity,binding in program.bindings.items():
    if binding.slot is None:
      if binding.external is None:
        raise ValueError('buffer has neither owned storage nor an external binding: '+identity)
      continue
    if binding.slot not in slots or binding.view_offset_bytes+binding.size_bytes>slots[binding.slot].size_bytes:
      raise ValueError('buffer exceeds its virtual slot: '+identity)
  return program


def plan_summary(program):
  recipe=program.memory
  return {'buffers':len(program.bindings),'virtual_bytes':sum(r.bytes for r in recipe.addresses.regions),
          'physical_bytes':sum(b.bytes for b in recipe.backings),
          'capacity_groups':dict(program.peak_bytes_by_capacity_group),
          'budget_epoch':program.budget_epoch,'resource_signature':program.resource_signature,
          'unknown_cost':program.used_conservative_fallback}
