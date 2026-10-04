"""CPU state placement: independent register pieces or private scalar memory."""

import itertools
import math
from dataclasses import dataclass, replace
from types import MappingProxyType

from ... import ir as I
from ...dtypes import ELEMENT_BYTES
from ...passes.manager import register_pass
from ...operations import Trait


@dataclass(frozen=True)
class StatePiece:
  coordinates: tuple[int, ...]
  lanes: int
  scalable: bool = False


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
  from ...ir.traversal import children, walk

  matrix_states = set()
  word_states = set()

  def matrices(block):
    for statement in block.statements:
      if isinstance(statement, (I.DeclareState, I.UpdateState)) and any(
        context.environment.operations.has(e, Trait.REGISTER_TRANSFORM) for e in walk(statement.value)
      ):
        word_states.add(statement.state.name)
      if isinstance(statement, (I.DeclareState, I.UpdateState)) and any(e.op == "dot" for e in walk(statement.value)):
        matrix_states.add(statement.state.name)
      value = getattr(statement, "value", None)
      if isinstance(value, I.Expr):
        matrix_states.update(
          ref.data.name
          for product in walk(value)
          if product.op == "dot"
          for accumulator in product.args[2:]
          for ref in walk(accumulator)
          if ref.op == "state_ref"
        )
      for child in children(statement):
        matrices(child)

  matrices(prepared.vectors.kernel.body)

  def visit(block):
    for statement in block.statements:
      if isinstance(statement, I.DeclareState):
        state = statement.state
        shape = tuple(I.constant(s) for s in state.type.shape)
        pieces = []
        typ = state.type
        if budget and state.name not in matrix_states and isinstance(typ, I.VectorType) and typ.tag.N is None:
          # The logical lane count retains the descriptor's basis after a
          # cast. Wider elements require more native words for the same lanes.
          basis = typ.tag.basis_dtype or typ.dtype
          power = typ.tag.pow + ELEMENT_BYTES[typ.dtype].bit_length() - ELEMENT_BYTES[basis].bit_length()
          count = 1 << max(0, power) if power <= budget.bit_length() else budget + 1
          if count <= budget:
            pieces = [StatePiece((i,), 0, True) for i in range(count)]
        # FixedTag pieces fit one native register on the selected x86 ISA,
        # or the architectural 128-bit minimum in vector-length-agnostic SVE.
        bits = context.environment.target.vector_bits or 128
        largest = (
          1
          if context.environment.target.isa in {"scalar", "neon"}
          else max(1, bits // (8 * ELEMENT_BYTES[state.type.dtype]))
        )
        if state.name in word_states:
          # Partial FixedTags change the shuffle word/block width. Keep native
          # scalable words with clipped active counts, or stage scalar memory.
          if budget and len(shape) == 1 and type(shape[0]) is int and shape[0] > 0:
            count = (shape[0] + largest - 1) // largest
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
          count = math.prod(shape[:-1]) * ((shape[-1] + largest - 1) // largest)
          if count <= budget:
            for row in itertools.product(*(range(s) for s in shape[:-1])):
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
