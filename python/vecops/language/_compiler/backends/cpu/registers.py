# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Closed fixed-register placement; large logical values keep generic lowering."""

from dataclasses import dataclass, replace
from types import MappingProxyType

from ...._library.registers import transpose_dwords_512, transpose_registers
from ....dsl import Vector
from ... import ir as I
from ...dtypes import ELEMENT_BYTES
from ...frontend.helpers import expand_register_helper
from ...ir.traversal import nodes, walk_unique
from ...ir.vector import RegisterProgram
from ...operations import Trait
from ...operations import current as snapshot
from ...passes.manager import register_pass


def register_transpose(rows, columns, element_bytes, *, dtype=None):
  """Bind the portable DSL library recipe to selected physical geometry."""
  return expand_register_helper(
    transpose_registers,
    (Vector(dtype or f"uint{element_bytes * 8}", N=columns),) * rows,
    catalog=snapshot(),
  )


def x86_dword_transpose():
  return expand_register_helper(transpose_dwords_512, (Vector("uint32", N=16),) * 16, catalog=snapshot())


@dataclass(frozen=True)
class CPURegisterPlan:
  program: RegisterProgram
  tags: object


def closed_register_program(value, catalog, limit):
  """Return the original typed closed DAG when every logical register fits.

  Input producers remain atomic under the existing trait restrictions. This
  query is shared by placement and prepared verification; it never retags or
  clones historical expressions. Cyclic register-geometry edges fail
  conservatively; atomic inputs retain the existing trait restrictions.
  """
  inputs, operations, seen, active = [], [], set(), set()

  def visit(expression):
    if id(expression) in seen:
      return True
    if id(expression) in active or not isinstance(expression.type, I.TileType) or len(expression.type.shape) != 1:
      return False
    extent = I.constant(expression.type.shape[0])
    if (
      type(extent) is not int
      or extent <= 0
      or extent & (extent - 1)
      or extent * ELEMENT_BYTES[expression.type.dtype] > limit
    ):
      return False
    active.add(id(expression))
    if catalog.get(expression).family == "register_geometry":
      if any(not visit(argument) for argument in expression.args):
        return False
    else:
      if any(
        catalog.has(e, Trait.REGISTER_TRANSFORM | Trait.REDUCTION)
        or catalog.has(e, Trait.STATE_REFERENCE)
        and e.op != "state_ref"
        for e in walk_unique(expression)
      ):
        return False
      inputs.append(expression)
    active.remove(id(expression))
    seen.add(id(expression))
    operations.append(expression)
    return True

  if catalog.get(value).family != "register_geometry" or not visit(value):
    return None
  return RegisterProgram(tuple(inputs), tuple(operations), (value,))


@register_pass("cpu-plan-register-programs", stage="cpu-memory", before="cpu-plan-state")
def plan_register_programs(prepared, context):
  target, catalog = context.environment.codegen_target, context.environment.operations
  # Ordinary VL can legalize an explicit complete register. SVL and cost
  # profiles do not establish the ordinary vector representation.
  limit = 16 if target.isa in {"sve", "sme"} else (target.vector_bits or 0) // 8
  state = context.environment.vector_state
  if target.isa in {"sve", "sme"} and state is not None and state.sve_bytes is not None:
    # Explicit complete logical registers may occupy one actual native word.
    # Their printer uses matching scalable tags, not oversized VLA FixedTags.
    limit = state.sve_bytes
  if target.isa in {"scalar", "neon"}:
    limit = 0
  plans = {}
  for statement in nodes(prepared.vectors.kernel.body):
    store = isinstance(statement, I.Store)
    candidate = store or isinstance(statement, (I.DeclareState, I.UpdateState))
    value = statement.value if candidate else None
    prefix = False
    if (
      value is not None
      and value.op == "value_transform"
      and value.data == ("slice", (0,))
      and len(value.args) == 2
      and I.constant(value.args[1]) == 0
    ):
      # A bounded store may consume a prefix of a complete register DAG.
      # Keep its arithmetic closed; predicate only the memory boundary.
      value = value.args[0]
      prefix = True
    if (
      not candidate
      or store
      and (statement.mask is not None or len(statement.destination.shape) != 1)
      or catalog.get(value).family != "register_geometry"
      or store
      and not prefix
      and not I.same_expr(statement.destination.shape[0], statement.value.type.shape[0])
    ):
      continue
    program = closed_register_program(value, catalog, limit)
    if program is not None:
      operations = program.operations
      tags = {}
      for expression in operations:
        extent = I.constant(expression.type.shape[0])
        dtype = expression.type.dtype
        byte_count = extent * ELEMENT_BYTES[dtype]
        if target.isa in {"sve", "sme"} and state is not None and state.sve_bytes is not None and byte_count > 16:
          power = byte_count.bit_length() - state.sve_bytes.bit_length()
          tags[id(expression)] = I.VectorTag(pow=power, basis_dtype=dtype)
        else:
          tags[id(expression)] = expression.type.tag
      if target.isa in {"sve", "sme"}:
        supported = True
        for expression in operations:
          for domain in context.environment.implementations.get(expression.op).half_domains:
            domain_value = expression if domain == -1 else expression.args[domain]
            descriptor = tags[id(domain_value)]
            if (
              descriptor.N is None and (16 // ELEMENT_BYTES[domain_value.type.dtype]) >> max(0, 1 - descriptor.pow) == 0
            ):
              supported = False
        if not supported:
          # Actual VL may fit the result while C++ Half<Tag> is unrepresentable
          # at the architectural minimum. Retain exact generic legalization,
          # rather than emitting a tag whose type cannot be instantiated.
          continue
      plans[id(statement) if store else id(value)] = CPURegisterPlan(program, MappingProxyType(tags))
  # A multi-output permutation is one DAG. Printing a separate program at
  # each store repeats its shared input conversion and butterfly nodes; the
  # intervening stores can prevent the native compiler from recovering CSE.
  # Memory independence is versioned at emission, retaining sequential
  # behavior for overlapping destinations or inputs.
  for group in nodes(prepared.vectors.kernel.body):
    if not isinstance(group, I.ElementwiseGroup) or len(group.stores) < 2:
      continue
    members = [plans.get(id(store)) for store in group.stores]
    if any(plan is None for plan in members):
      continue
    if any(
      not I.same_expr(store.destination.shape[0], group.stores[0].destination.shape[0]) for store in group.stores[1:]
    ):
      continue
    # Unbounded materialized producers retain independent output banks:
    # GCC's large Dynamic cache kernels otherwise spill more temporaries.
    # Exact tile caches and original converting producers still share their
    # DAG; repeated loads across independent stores hurt fringe conversion.
    reads = tuple(load for store in group.stores for load in I.loads_unique(store.value))
    cached = all(
      node.op in {"load", "identity", "bitcast", "value_transform"}
      for plan in members
      for value in plan.program.inputs
      for node in walk_unique(value)
      if isinstance(node.type, I.TileType)
    )
    unbounded = any(
      any(
        I.constant(extent) is None
        for extent in (load.data.region.root_shape if load.data.region is not None else load.data.shape)
      )
      for load in reads
    )
    if reads and cached and unbounded and all(load.data.owned for load in reads):
      continue
    outputs = tuple(plan.program.outputs[0] for plan in members)
    first = outputs[0]
    if any(
      value.type.dtype != first.type.dtype
      or value.type.mask != first.type.mask
      or not I.same_expr(value.type.shape[0], first.type.shape[0])
      or members[index].tags[id(value)] != members[0].tags[id(first)]
      for index, value in enumerate(outputs)
    ):
      continue
    inputs, operations, tags, input_ids, operation_ids = [], [], {}, set(), set()
    for plan in members:
      for value in plan.program.inputs:
        if id(value) not in input_ids:
          inputs.append(value)
          input_ids.add(id(value))
      for value in plan.program.operations:
        if id(value) not in operation_ids:
          operations.append(value)
          operation_ids.add(id(value))
      tags.update(plan.tags)
    plans[id(group)] = CPURegisterPlan(
      RegisterProgram(tuple(inputs), tuple(operations), outputs), MappingProxyType(tags)
    )
  return replace(prepared, register_programs=MappingProxyType(plans))
