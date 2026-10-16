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


def verify_prepared(prepared):
  verify_kernel(prepared.vectors.kernel, terminal=True)
  seen = set()
  for region in (
    *prepared.vectors.regions,
    *prepared.stores.values(),
    *prepared.snapshots.values(),
    *prepared.axes.values(),
    *prepared.reductions.values(),
    *prepared.primitives.values(),
    *prepared.state_regions.values(),
  ):
    if id(region) not in seen:
      verify_vector(region)
      seen.add(id(region))
  storage = prepared.storage
  if storage is not None:
    slots = {}
    for interval in storage.intervals:
      slot = storage.slots[interval.storage]
      for previous in slots.get(slot, ()):
        require(
          previous.dtype == interval.dtype and previous.scope == interval.scope,
          "scratch slot crosses dtype/ownership scopes",
        )
        require(
          previous.last < interval.first or interval.last < previous.first,
          "scratch slot has overlapping live intervals",
        )
      slots.setdefault(slot, []).append(interval)
  verify_private_captures(prepared)
  return prepared


def verify_private_captures(prepared):
  """Reject stale, mutable or out-of-scope positive value-capture proofs."""
  from ...types import TensorType
  from ..backends.cpu.capabilities import CPP_TYPES
  from ..operations import Trait, current
  from .collective import TeamReduce
  from .state import ExitRegion
  from .traversal import children, nodes

  kernel = prepared.vectors.kernel
  definitions, excluded = {}, set()
  for node in nodes(kernel.body):
    if isinstance(node, (I.Assign, I.For)):
      definitions.setdefault(node.target, []).append(node)
    if isinstance(node, (I.For, TeamReduce)):
      excluded.add(node.target)
    elif isinstance(node, I.Grid):
      excluded.update(node.targets)
  immutable = {
    name for name, sites in definitions.items()
    if len(sites) == 1 and isinstance(sites[0], I.Assign) and sites[0].declare
    and isinstance(sites[0].value.type, I.ScalarType) and sites[0].value.type.dtype in CPP_TYPES
    and name not in excluded
  }
  parameters = {
    name for name, typ in kernel.parameters.items()
    if not isinstance(typ, TensorType) and name not in definitions and name not in excluded
  }
  tensors = {name: typ for name, typ in kernel.parameters.items() if isinstance(typ, TensorType)}
  metadata = {
    f"{name}.{operation}({axis})"
    for name, typ in tensors.items() for operation in ("size", "stride") for axis in range(len(typ.shape))
  }
  seen = set()
  manager = getattr(prepared.vectors, "manager", None)
  environment = (getattr(manager, "options", {}) or {}).get("_compiler_environment")
  catalog = environment.operations if environment is not None else current()

  def read_references(region):
    found, symbols = {}, set()

    def visit(value):
      for node in nodes(value):
        if not isinstance(node, I.Expr):
          continue
        if catalog.has(node, Trait.MEMORY_READ):
          found[id(node)] = node
        if node.op == "symbol" and node.data not in symbols:
          symbols.add(node.data)
          if node.data in kernel.bindings:
            visit(kernel.bindings[node.data])

    visit(region.body)
    return found

  def block(body, available, hardware=False):
    available = set(available)
    for statement in body.statements:
      if isinstance(statement, ExitRegion) and statement.outlined:
        plan = prepared.private_captures.get(id(statement))
        require(plan is not None and plan.region is statement, "private capture plan has a missing/stale subject")
        seen.add(id(statement))
        for names in (plan.scalars, plan.metadata, plan.pointers):
          require(len(set(names)) == len(names), "private capture plan contains duplicate bindings")
        require(set(plan.scalars) <= available, "private scalar capture is mutable or lacks lexical dominance")
        require(set(plan.metadata) <= metadata, "private metadata capture is not an ABI snapshot")
        require(set(plan.pointers) <= tensors.keys(), "private pointer capture is not an ABI tensor")
        reads = read_references(statement)
        require(len({id(read) for read in plan.snapshot_reads}) == len(plan.snapshot_reads),
                "private snapshot capture contains duplicate reads")
        require(all(reads.get(id(read)) is read for read in plan.snapshot_reads),
                "private snapshot capture has a detached/non-read subject")
        require(not hardware or not (plan.scalars or plan.metadata or plan.pointers or plan.snapshot_reads),
                "private value capture crosses an active hardware scope")
      for child in children(statement):
        block(child, available, hardware or isinstance(statement, I.HardwareRegion))
      assignments = statement.assignments if isinstance(statement, I.ReductionGroup) else (statement,)
      for assignment in assignments:
        if isinstance(assignment, I.Assign) and assignment.declare and assignment.target in immutable:
          available.add(assignment.target)

  block(kernel.body, parameters)
  require(set(prepared.private_captures) == seen, "private capture plan contains a detached region")
  return prepared
