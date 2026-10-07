"""Closed fixed-register placement; large logical values keep generic lowering."""

from dataclasses import dataclass, replace
from types import MappingProxyType

from ...._library.registers import pack_interleave, transpose_dwords_512, transpose_registers
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


@register_pass("cpu-plan-register-programs", stage="cpu-memory", after="cpu-plan-state")
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
    if (
      not isinstance(statement, I.Store)
      or statement.mask is not None
      or len(statement.destination.shape) != 1
      or catalog.get(statement.value).family != "register_geometry"
      or not I.same_expr(statement.destination.shape[0], statement.value.type.shape[0])
    ):
      continue
    inputs, operations, seen = [], [], set()

    def visit(expression, seen=seen, inputs=inputs, operations=operations):
      if id(expression) in seen:
        return True
      if not isinstance(expression.type, I.TileType) or len(expression.type.shape) != 1:
        return False
      extent = I.constant(expression.type.shape[0])
      if (
        type(extent) is not int
        or extent <= 0
        or extent & (extent - 1)
        or extent * ELEMENT_BYTES[expression.type.dtype] > limit
      ):
        return False
      if catalog.get(expression).family == "register_geometry":
        if any(not visit(argument) for argument in expression.args):
          return False
      else:
        if any(
          catalog.has(e, Trait.REGISTER_TRANSFORM | Trait.REDUCTION | Trait.STATE_REFERENCE)
          for e in walk_unique(expression)
        ):
          return False
        inputs.append(expression)
      seen.add(id(expression))
      operations.append(expression)
      return True

    if visit(statement.value):
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
            value = expression if domain == -1 else expression.args[domain]
            descriptor = tags[id(value)]
            if descriptor.N is None and (16 // ELEMENT_BYTES[value.type.dtype]) >> max(0, 1 - descriptor.pow) == 0:
              supported = False
        if not supported:
          # Actual VL may fit the result while C++ Half<Tag> is unrepresentable
          # at the architectural minimum. Retain exact generic legalization,
          # rather than emitting a tag whose type cannot be instantiated.
          continue
      plans[id(statement)] = CPURegisterPlan(
        RegisterProgram(tuple(inputs), tuple(operations), (statement.value,)), MappingProxyType(tags)
      )
  groups = {}
  # The same source program serves raw SME operands, explicit packing, and
  # AMX's fixed spatial columns. No native VL is guessed: scalable tags retain
  # their domain/power through the pure DAG until terminal instruction spelling.
  graph = tuple(nodes(prepared.vectors.kernel.body))
  packed = any(isinstance(node, I.Pack) or isinstance(node, I.Expr) and node.op == "dot" for node in graph)
  dtypes = {node.type.dtype for node in graph if isinstance(node, I.Expr)}
  dtypes.update(node.dtype for node in graph if isinstance(node, I.View))
  for dtype in sorted(dtypes & {"int8", "uint8", "float16", "bfloat16"}) if packed else ():
    count = 4 // ELEMENT_BYTES[dtype]
    for lanes, power in ((16, 0), (None, -(count.bit_length() - 1))):
      groups[dtype, lanes, power] = expand_register_helper(
        pack_interleave, (Vector(dtype, N=lanes, pow=power),) * count, catalog=catalog, prefix="pack_group"
      )
  return replace(prepared, register_programs=MappingProxyType(plans), packing_groups=MappingProxyType(groups))
