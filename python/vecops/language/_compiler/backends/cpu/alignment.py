# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Prove stable store geometry, then plan native-word alignment peeling.

The proof is intersected over lexical occurrences. Peeling only changes a
single pointwise store's physical chunks: alias snapshots, historical readers,
logical coordinates and the public tensor's strides retain their owners.
"""

from dataclasses import dataclass, replace
from types import MappingProxyType

from ... import ir as I
from ...dtypes import ELEMENT_BYTES
from ...ir.traversal import nodes
from ...operations import Trait
from ...passes.manager import register_pass


@dataclass(frozen=True)
class StoreAlignmentProof:
  store: I.Store
  destination: I.View
  value: I.Expr
  minimum_columns: int
  occurrences: int


@dataclass(frozen=True)
class AlignedStore:
  region: object
  proof: StoreAlignmentProof


def make_alignment_collector(kernel):
  """Reuse the final memory traversal's temporary scalar/ABI environment."""
  failed, counts, minima, stores = set(), {}, {}, {}
  blocked, seen = set(), set()
  for region in nodes(kernel.body):
    if isinstance(region, (I.Grid, I.Team, I.HardwareRegion, I.ReductionGroup)):
      blocked.update(id(node) for node in nodes(region, seen) if isinstance(node, I.Store))

  def eligible(store, scope):
    view = store.destination
    if (
      type(view) is not I.View or view.owned or not view.shape
      or len(view.shape) != len(view.strides)
      or scope.abi.get(view.storage) != view.dtype
      or view.dtype not in ELEMENT_BYTES or store.mask is not None
      or store.vector_extent is not None
    ):
      return None
    counts = [scope.range(value) for value in view.shape]
    strides = [scope.range(value) for value in view.strides]
    offset = scope.range(view.offset)
    if (
      any(value is None or value[0] < 0 for value in (*counts, *strides))
      or offset is None or offset[0] < 0 or strides[-1] != (1, 1)
    ):
      return None
    # Extend empty dimensions to one only for this conservative magnitude
    # bound. No pointer is formed on the corresponding empty runtime path.
    end = view.offset
    for extent, stride in zip(view.shape, view.strides):
      distance = I.binary("max", I.literal(0), I.binary("sub", extent, I.literal(1)))
      end = I.binary("add", end, I.binary("mul", distance, stride))
    # Positive bounded strides need not be affine (N * runtime_pitch).
    # The triangle/typed-overflow proof suffices; each summand is nonnegative.
    safe_end = scope.safe(I.binary("mul", end, I.literal(ELEMENT_BYTES[view.dtype])))
    return counts[-1][0] if safe_end else None

  def observe(statement, scope):
    candidates = statement.stores if type(statement) is I.ElementwiseGroup else (statement,)
    for store in candidates:
      if type(store) is not I.Store:
        continue
      identity = id(store)
      stores[identity] = store
      counts[identity] = counts.get(identity, 0) + 1
      width = None if identity in blocked else eligible(store, scope)
      if width is None:
        failed.add(identity)
      else:
        minima[identity] = min(minima.get(identity, width), width)

  def finish():
    return MappingProxyType({
      identity: StoreAlignmentProof(store, store.destination, store.value, minima[identity], counts[identity])
      for identity, store in stores.items() if identity not in failed
    })

  return observe, finish


def select_aligned_stores(prepared, proofs, environment):
  """Select wide, ordinary pointwise traversals with a single store owner.

  At least eight native words amortize a masked head and tail. Explicit
  vector contracts, reductions, register permutations, state and converting
  stores retain their existing schedules. Unknown geometry retains fallback.
  """
  if not proofs:
    return {}
  target = environment.codegen_target
  if target.architecture != "x86" or target.isa != "avx512" or not target.vector_bits:
    return {}
  catalog = environment.operations
  result, visited = {}, set()
  regions = (
    *prepared.vectors.plans.values(), *prepared.vectors.regions,
    *prepared.stores.values(), *prepared.snapshots.values(), *prepared.axes.values(),
    *prepared.reductions.values(), *prepared.primitives.values(), *prepared.state_regions.values(),
  )
  boundaries = Trait.REDUCTION | Trait.REGISTER_TRANSFORM | Trait.VECTOR_BOUNDARY | Trait.STATE_REFERENCE
  for region in regions:
    if id(region) in visited:
      continue
    visited.add(id(region))
    if (
      not region.shape or region.base_power != 0 or not region.full_chunks
      or region.single_chunk or region.iteration_axis is not None
      or region.initializations or region.finalizations or len(region.effects) != 1
    ):
      continue
    effect = region.effects[0]
    proof = proofs.get(id(effect.store))
    if (
      effect.kind != "store_convert" or proof is None
      or effect.store is not proof.store or proof.store.destination is not proof.destination
      or proof.store.value is not proof.value or proof.store.mask is not None
      or region.dtype != proof.destination.dtype
      or len(region.shape) != len(proof.destination.shape)
      or any(a is not b for a, b in zip(region.shape, proof.destination.shape))
    ):
      continue
    size = ELEMENT_BYTES[region.dtype]
    if proof.minimum_columns * size < 8 * (target.vector_bits // 8):
      continue
    if any(
      isinstance(op.expression.type, I.VectorType)
      or getattr(op.expression.type, "mask", False)
      or catalog.has(op.expression, boundaries)
      or catalog.get(op.expression).effects not in {"none", "read"}
      or catalog.has(op.expression, Trait.MEMORY_READ)
      and (op.expression.op != "load" or type(op.expression.data) is not I.View)
      or isinstance(op.expression.type, I.TileType)
      and op.expression.op != "load" and not catalog.has(op.expression, Trait.ELEMENTWISE)
      for op in region.operations
    ):
      continue
    result[id(region)] = AlignedStore(region, proof)
  return result


@register_pass("cpu-plan-store-alignment", stage="cpu-memory", after="cpu-plan-memory-proofs")
def plan_store_alignment(prepared, context):
  enabled = context.options.get("cpu_align_stores", True)
  if type(enabled) is not bool:
    raise TypeError("cpu_align_stores must be bool")
  plans = select_aligned_stores(prepared, prepared.store_alignment_proofs, context.environment) if enabled else {}
  return replace(prepared, aligned_stores=MappingProxyType(plans))
