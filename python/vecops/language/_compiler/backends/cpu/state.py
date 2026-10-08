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


@register_pass("cpu-plan-state", stage="cpu-memory", before="cpu-plan-grids")
def plan_state(prepared, context):
  budget = context.options.get("cpu_state_register_budget", 12)
  if isinstance(budget, bool) or not isinstance(budget, int) or not 0 <= budget <= 64:
    raise ValueError("cpu_state_register_budget must be an integer in [0,64]")
  plans = {}
  from ...ir.traversal import children, values, walk_unique

  matrix_states = set()
  word_states = set()
  state_links = {}
  initial_word_bytes = {}

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
          if budget and len(shape) == 1 and type(shape[0]) is int and shape[0] > 0:
            word_lanes = max(1, native_bytes // ELEMENT_BYTES[typ.dtype])
            count = (shape[0] + word_lanes - 1) // word_lanes
            pieces = [StatePiece((i,), 0, True) for i in range(count)] if count <= budget else []
          else:
            pieces = []
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
        if len(pieces) > budget:
          pieces = []
        plans[state.name] = StatePlan(state, state_view(state), tuple(pieces))
        placement = f"{len(pieces)} independent register pieces" if pieces else "private scalar scratch"
        prepared.vectors.kernel.diagnostics.append(f"tile state {state.name}: {placement}")
      for child in children(statement):
        visit(child)

  visit(prepared.vectors.kernel.body)
  return replace(prepared, states=MappingProxyType(plans))
