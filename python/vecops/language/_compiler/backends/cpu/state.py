# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""CPU state placement: independent register pieces or private scalar memory."""

import itertools
import math
from dataclasses import dataclass, replace
from types import MappingProxyType

from ... import ir as I
from ...dtypes import ELEMENT_BYTES
from ...operations import Trait
from ...passes.manager import register_pass


@dataclass(frozen=True)
class StatePiece:
  coordinates: tuple[int, ...]
  lanes: int
  scalable: bool = False
  power: int = 0
  full: bool = False


@dataclass(frozen=True)
class StatePlan:
  state: I.TileState
  view: I.View
  pieces: tuple[StatePiece, ...]
  closed_fixed: bool = False


def closed_fixed_lanes(state, statements, programs, environment):
  """Prove one fixed physical representation for every original state write.

  A register transform alone does not prove this: its complete typed program,
  including every update, must fit a known x86 register. This query neither
  changes expression identities nor relies on the current state placement.
  """
  from ...ir.vector import RegisterProgram
  from .registers import CPURegisterPlan, closed_register_program

  target = environment.codegen_target
  if (
    target.architecture != "x86"
    or target.isa in {"scalar", "neon"}
    or type(target.vector_bits) is not int
    or target.vector_bits <= 0
  ):
    return None
  limit = target.vector_bits // 8

  def fixed_type(typ):
    return (
      type(typ) is I.VectorType
      and not typ.mask
      and type(typ.tag) is I.VectorTag
      and type(typ.tag.N) is int
      and typ.tag.N > 0
      and not typ.tag.N & (typ.tag.N - 1)
      and typ.tag.pow == 0
      and typ.tag.domain == "simd"
      and len(typ.shape) == 1
      and I.constant(typ.shape[0]) == typ.tag.N
      and typ.dtype in ELEMENT_BYTES
      and typ.tag.basis_dtype in ELEMENT_BYTES
      and typ.tag.N * ELEMENT_BYTES[typ.dtype] <= limit
    )

  typ = state.type
  if not fixed_type(typ) or sum(type(s) is I.DeclareState for s in statements) != 1:
    return None
  catalog = environment.operations
  for statement in statements:
    if type(statement) not in {I.DeclareState, I.UpdateState} or statement.state is not state:
      return None
    value = statement.value
    plan = programs.get(id(value))
    if (
      type(value) is not I.Expr
      or not fixed_type(value.type)
      or value.type.dtype != typ.dtype
      or value.type.mask != typ.mask
      or value.type.tag != typ.tag
      or type(plan) is not CPURegisterPlan
      or type(plan.program) is not RegisterProgram
      or len(plan.program.outputs) != 1
      or plan.program.outputs[0] is not value
    ):
      return None
    # Recompute the actual closed DAG rather than trusting its key or marker.
    confirmed = closed_register_program(value, catalog, limit)
    if confirmed is None:
      return None
    inputs, operations = confirmed.inputs, confirmed.operations
    if (
      len(inputs) != len(plan.program.inputs)
      or any(a is not b for a, b in zip(inputs, plan.program.inputs))
      or len(operations) != len(plan.program.operations)
      or any(a is not b for a, b in zip(operations, plan.program.operations))
    ):
      return None
    for expression in operations:
      if (
        type(expression) is not I.Expr
        or not fixed_type(expression.type)
        or plan.tags.get(id(expression)) != expression.type.tag
      ):
        return None
      if expression.op in {"bitreinterpret", "bitcast"}:
        if len(expression.args) != 1:
          return None
        source = expression.args[0].type
        if source.tag.N * ELEMENT_BYTES[source.dtype] != expression.type.tag.N * ELEMENT_BYTES[expression.type.dtype]:
          return None
  return typ.tag.N


def state_view(state):
  shape = state.type.shape
  strides = [I.literal(1)] * len(shape)
  for axis in reversed(range(len(shape) - 1)):
    strides[axis] = I.binary("mul", strides[axis + 1], shape[axis + 1])
  dtype = "uint8" if state.type.mask else state.type.dtype
  return I.View(state.name, dtype, shape, tuple(strides), I.literal(0), owned=True)


def stored_value(state, value):
  if not state.type.mask:
    return value
  typ = I.TileType("uint8", state.type.shape)
  return I.Expr(typ, "where", (value, I.literal(1, "uint8"), I.literal(0, "uint8")))


@register_pass("cpu-plan-state", stage="cpu-memory", after="cpu-plan-register-programs", before="cpu-plan-grids")
def plan_state(prepared, context):
  budget = context.options.get("cpu_state_register_budget", 12)
  if isinstance(budget, bool) or not isinstance(budget, int) or not 0 <= budget <= 64:
    raise ValueError("cpu_state_register_budget must be an integer in [0,64]")
  plans = {}
  from ...ir.traversal import children, nodes, values, walk_unique

  matrix_states = set()
  word_states = set()
  sve_word_states = set()
  state_links = {}
  initial_word_bytes = {}
  call_states = set()
  definitions = {}
  for statement in nodes(prepared.vectors.kernel.body):
    if isinstance(statement, (I.DeclareState, I.UpdateState)):
      definitions.setdefault(statement.state.name, []).append(statement)
    if isinstance(statement, I.ExitRegion) and statement.outlined:
      contained = tuple(nodes(statement.body))
      local = {s.state.name for s in contained if isinstance(s, I.DeclareState)}
      call_states.update(s.name for s in contained if isinstance(s, I.TileState) and s.name not in local)

  def matrices(block):
    for statement in block.statements:
      if isinstance(statement, (I.DeclareState, I.UpdateState)):
        expressions = tuple(walk_unique(statement.value))
        name = statement.state.name
        width = max(
          (ELEMENT_BYTES[e.type.dtype] for e in expressions if isinstance(e.type, I.TileType)),
          default=ELEMENT_BYTES[statement.state.type.dtype],
        )
        initial_word_bytes[name] = max(initial_word_bytes.get(name, 1), width)
        for expression in expressions:
          if context.environment.operations.get(expression).family == "sve_register":
            sve_word_states.add(name)
            for source in expression.args:
              for reference in walk_unique(source):
                typ = reference.type
                if (
                  reference.op == "state_ref"
                  and isinstance(typ, I.VectorType)
                  and typ.tag.N is None
                  and typ.tag.pow == 0
                  and typ.tag.domain == "simd"
                  and typ.tag.basis_dtype == typ.dtype
                ):
                  sve_word_states.add(reference.data.name)
        for reference in expressions:
          if reference.op == "state_ref":
            state_links.setdefault(name, set()).add(reference.data.name)
            state_links.setdefault(reference.data.name, set()).add(name)
      if isinstance(statement, (I.DeclareState, I.UpdateState)) and any(
        context.environment.operations.has(e, Trait.REGISTER_TRANSFORM) for e in walk_unique(statement.value)
      ):
        word_states.add(statement.state.name)
      if isinstance(statement, (I.DeclareState, I.UpdateState)) and any(
        e.op == "dot" for e in walk_unique(statement.value)
      ):
        matrix_states.add(statement.state.name)
      value = getattr(statement, "value", None)
      if isinstance(value, I.Expr):
        matrix_states.update(
          ref.data.name
          for product in walk_unique(value)
          if product.op == "dot"
          for accumulator in product.args[2:]
          for ref in walk_unique(accumulator)
          if ref.op == "state_ref"
        )
      for child in children(statement):
        matrices(child)

  matrices(prepared.vectors.kernel.body)

  # A piece must fit the widest *vector* intermediate, not just the stored
  # element. Scalar literals do not introduce another lane representation.
  # Include consumers as well as updates so a widening cast never turns one
  # local into a multiword Vec aggregate during emission.
  word_bytes = dict(initial_word_bytes)
  for root in values(prepared.vectors.kernel.body):
    expressions = tuple(walk_unique(root))
    width = max((ELEMENT_BYTES[e.type.dtype] for e in expressions if isinstance(e.type, I.TileType)), default=1)
    names = {e.data.name for e in expressions if e.op == "state_ref"}
    for name in names:
      word_bytes[name] = max(word_bytes.get(name, 1), width)
  # Snapshots and simultaneous assignments share a lane schedule. Otherwise
  # widening one consumer would cause get/set repacking in every recurrence.
  seen_states = set()
  for name in state_links:
    if name in seen_states:
      continue
    component, pending = set(), [name]
    while pending:
      current = pending.pop()
      if current in component:
        continue
      component.add(current)
      pending.extend(state_links.get(current, ()))
    seen_states.update(component)
    width = max(word_bytes.get(member, 1) for member in component)
    for member in component:
      word_bytes[member] = width
  target = context.environment.codegen_target
  sve = target.isa in {"sve", "sme"}
  vector_state = context.environment.vector_state
  sve_bytes = vector_state.sve_bytes if sve and vector_state is not None else None
  # Runtime vector geometry is a code-generation input, unlike cost-profile
  # overrides. VLA FixedTag is nevertheless limited to the architectural
  # 128-bit minimum: a captured VL does not enable -msve-vector-bits.
  native_bytes = sve_bytes or (target.vector_bits or 128) // 8
  fixed_bytes = 16 if sve else native_bytes

  # Branch merges and helper snapshots can sit between a typed ISA word and
  # its consumer. Retain that geometry through same-dtype, complete-word
  # aliases; unrelated partial vectors and scalar producers stay conservative.
  declarations = {
    statement.state.name: statement.state.type
    for statement in nodes(prepared.vectors.kernel.body)
    if isinstance(statement, I.DeclareState)
  }
  pending = list(sve_word_states)
  while pending:
    name = pending.pop()
    source_type = declarations.get(name)
    for neighbor in state_links.get(name, ()):
      typ = declarations.get(neighbor)
      if (
        neighbor not in sve_word_states
        and source_type is not None
        and isinstance(typ, I.VectorType)
        and not typ.mask
        and typ.dtype == source_type.dtype
        and typ.tag.N is None
        and typ.tag.pow == 0
        and typ.tag.domain == "simd"
        and typ.tag.basis_dtype == typ.dtype
        and len(typ.shape) == 1
        and I.same_expr(typ.shape[0], I.Expr(I.ScalarType("int64"), "lanes", data=(typ.dtype, 0)))
      ):
        sve_word_states.add(neighbor)
        pending.append(neighbor)

  def visit(block):
    for statement in block.statements:
      if isinstance(statement, I.DeclareState):
        state = statement.state
        shape = tuple(I.constant(s) for s in state.type.shape)
        pieces = []
        typ = state.type
        widest = max(word_bytes.get(state.name, 1), ELEMENT_BYTES[typ.dtype])
        if budget and state.name not in matrix_states and isinstance(typ, I.VectorType) and typ.tag.N is None:
          # The logical lane count retains the descriptor's basis after a
          # cast. Wider elements require more native words for the same lanes.
          basis = typ.tag.basis_dtype or typ.dtype
          words = typ.tag.pow + widest.bit_length() - ELEMENT_BYTES[basis].bit_length()
          count = 1 << max(0, words) if words <= budget.bit_length() else budget + 1
          if count <= budget:
            power = ELEMENT_BYTES[typ.dtype].bit_length() - widest.bit_length()
            pieces = [StatePiece((i,), 0, True, power) for i in range(count)]
        # Fixed pieces always fit every intermediate in a native register.
        largest = 1 if target.isa in {"scalar", "neon"} else max(1, fixed_bytes // widest)
        if state.name in word_states:
          # Partial FixedTags change the shuffle word/block width. Keep native
          # scalable words with clipped active counts, or stage scalar memory.
          explicit_word = (
            isinstance(typ, I.VectorType)
            and not typ.mask
            and typ.tag.N is None
            and typ.tag.pow == 0
            and typ.tag.domain == "simd"
            and typ.tag.basis_dtype == typ.dtype
            and len(pieces) == 1
            and pieces[0].scalable
            and pieces[0].power == 0
          )
          if explicit_word:
            # Register transforms do not invalidate an already proven native
            # vector boundary merely because its active prefix is dynamic.
            pass
          elif budget and len(shape) == 1 and type(shape[0]) is int and shape[0] > 0:
            word_lanes = max(1, native_bytes // ELEMENT_BYTES[typ.dtype])
            count = (shape[0] + word_lanes - 1) // word_lanes
            pieces = [StatePiece((i,), 0, True) for i in range(count)] if count <= budget else []
          else:
            pieces = []
        closed_fixed = False
        if (
          budget
          and state.name not in matrix_states
          and len(pieces) == 1
          and pieces[0].scalable
          and pieces[0].coordinates == (0,)
          and pieces[0].power == 0
        ):
          lanes = closed_fixed_lanes(state, definitions[state.name], prepared.register_programs, context.environment)
          if lanes is not None:
            pieces = [StatePiece((0,), lanes)]
            closed_fixed = True
        if (
          not pieces
          and state.name not in word_states
          and budget
          and state.name not in matrix_states
          and shape
          and all(isinstance(s, int) and 0 < s <= 4096 for s in shape)
        ):
          if sve_bytes is not None:
            # Keep the captured native width without illegal VLA FixedTags.
            # Smaller scalable powers leave headroom for widening values;
            # each piece still emits a single named native register.
            largest = max(1, sve_bytes // widest)
            power = ELEMENT_BYTES[typ.dtype].bit_length() - widest.bit_length()
          count = math.prod(shape[:-1]) * ((shape[-1] + largest - 1) // largest)
          if count <= budget:
            for row in itertools.product(*(range(s) for s in shape[:-1])):
              if sve_bytes is not None:
                for word, offset in enumerate(range(0, shape[-1], largest)):
                  lanes = min(largest, shape[-1] - offset)
                  pieces.append(StatePiece(row + (word,), lanes, True, power, lanes == largest))
                continue
              offset = 0
              while offset < shape[-1]:
                lanes = 1 << min((shape[-1] - offset).bit_length() - 1, largest.bit_length() - 1)
                pieces.append(StatePiece(row + (offset,), lanes))
                offset += lanes
        # Segment operations change lane geometry while retaining one whole
        # ISA register. Their typed complete-word inputs/results must not be
        # split by the widest element type of the connected state graph.
        if sve and budget and state.name in sve_word_states:
          pieces = [StatePiece((0,), 0, True, 0)]
        # Values crossing a private call use scalar storage. In particular,
        # sizeless SVE registers cannot be members of a C++ closure object.
        # States local to the outlined body retain ordinary register planning.
        if len(pieces) > budget or state.name in call_states:
          pieces = []
          closed_fixed = False
        plans[state.name] = StatePlan(state, state_view(state), tuple(pieces), closed_fixed)
        placement = f"{len(pieces)} independent register pieces" if pieces else "private scalar scratch"
        prepared.vectors.kernel.diagnostics.append(f"tile state {state.name}: {placement}")
      for child in children(statement):
        visit(child)

  visit(prepared.vectors.kernel.body)
  return replace(prepared, states=MappingProxyType(plans))
