# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Representation verifiers shared by compiler phase boundaries.

Checks are linear in the owned DAG, do not rewrite it, and permit unresolved
dynamic constraints. Frontend semantic checks and backend hardware preflight
remain responsible for their stronger target/lexical contracts.
"""

from . import kernel as I
from .matmul import Pack
from .state import DeclareState, UpdateState
from .traversal import nodes
from .vector import VectorRegion


class IRVerificationError(ValueError):
  pass


def require(condition, message):
  if not condition:
    raise IRVerificationError(message)


def verify_kernel(kernel, *, terminal=False):
  require(isinstance(kernel, I.KernelIR), "kernel verifier requires KernelIR")
  if terminal:
    # BindTile registers historical load identities; it does not evaluate its
    # expression. Orientation may leave an original product descriptor there
    # while its executable transposed consumer has already been lowered.
    # Check executable ownership separately: a shared descriptor used by an
    # actual statement must still be rejected, even if also bound as history.
    for value in nodes(kernel.body, skip=(I.BindTile,)):
      if isinstance(value, I.Expr):
        require(value.op != "dot", "unlowered dot remains at the emission boundary")
      elif isinstance(value, Pack):
        raise IRVerificationError("unlowered packing remains at the emission boundary")
  for value in nodes(kernel.body):
    if isinstance(value, I.Expr):
      require(isinstance(value.type, (I.ScalarType, I.TileType)), "expression has an invalid value type")
      require(isinstance(value.op, str) and bool(value.op), "expression requires an operation name")
      require(
        isinstance(value.args, tuple) and all(isinstance(a, I.Expr) for a in value.args),
        f"{value.op}: operands must be an immutable expression tuple",
      )
      if value.op == "arange" and isinstance(value.type, I.TileType):
        axis = value.data if value.data is not None else len(value.type.shape) - 1
        require(type(axis) is int and 0 <= axis < len(value.type.shape), "index generator axis is outside its rank")
    elif isinstance(value, I.Block):
      require(isinstance(value.statements, list), "block statements must be an ordered list")
    elif isinstance(value, (I.TileType, I.View)):
      require(isinstance(value.shape, tuple), "logical shape must be an immutable tuple")
      require(
        all(isinstance(s, I.Expr) and isinstance(s.type, I.ScalarType) and "int" in s.type.dtype for s in value.shape),
        "logical extents must be scalar integer expressions",
      )
      if isinstance(value, I.View):
        require(len(value.shape) == len(value.strides), "view shape/stride rank mismatch")
    elif isinstance(value, I.For):
      require(I.constant(value.step) != 0, "loop step cannot be zero")
      require(type(value.unroll) is int and value.unroll > 0, "loop unroll must be positive")
    elif isinstance(value, I.Store):
      require(value.destination.writable, "store destination is input-only")
      require(isinstance(value.value, I.Expr), "store requires a computation value")
    elif isinstance(value, (DeclareState, UpdateState)):
      require(isinstance(value.value.type, I.TileType), "tile state requires a tile/vector value")
      typ, other = value.state.type, value.value.type
      require(
        typ.dtype == other.dtype and typ.mask == other.mask and len(typ.shape) == len(other.shape),
        "tile state dtype/predicate/rank mismatch",
      )
      require(
        all(
          I.constant(a) is None or I.constant(b) is None or I.constant(a) == I.constant(b)
          for a, b in zip(typ.shape, other.shape)
        ),
        "tile state has incompatible constant extents",
      )
  return kernel


def verify_vector(region):
  require(isinstance(region, VectorRegion), "vector verifier requires VectorRegion")
  count = len(region.operations)
  for index, operation in enumerate(region.operations):
    require(operation.index == index, "vector operation indices are not contiguous")
    require(
      all(operand is None or type(operand) is int and 0 <= operand < index for operand in operation.operands),
      "vector operand must precede its consumer",
    )
    require(isinstance(operation.type, I.VectorType), "vector operation has an unlegalized type")
  require(
    all(root is None or type(root) is int and 0 <= root < count for root in region.root_indices),
    "vector region contains an invalid root index",
  )
  require(type(region.unroll) is int and region.unroll > 0, "vector unroll must be positive")
  return region
