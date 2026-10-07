"""Prove private lane permutations for complete pointwise memory regions.

Tile expressions remain logically ordered. A complete converting load and
its converting store can share the backend's native permutation only when
every intervening operation commutes with that permutation. This proof is
separate from target costs, and never applies to reduction banks or tails.
Emission additionally checks the identity-based capture/materialization maps:
the original physical memory dtype is not sufficient for an older snapshot.
The default automatic policy has a separate, conservative profitability
gate. Broad forced layouts remain experimental: legality does not imply
that native pair instructions are profitable on a particular CPU.
"""

from dataclasses import dataclass, replace
from types import MappingProxyType

from ... import ir as I
from ...ir.traversal import walk_unique
from ...ir.vector import VectorRegion
from ...operations import Trait
from ...passes.manager import PassResult, register_pass
from .cost.register_layout import assess_register_layout


@dataclass(frozen=True)
class PointwiseRegisterLayout:
  """A region-local permutation proof, not a changed public Vector type."""

  name: str
  expressions: frozenset[int]
  loads: tuple[I.Expr, ...]


def prove_pointwise_layout(region, catalog):
  """Return a proof only for one closed, F32 pointwise converting store.

  TRANSPOSE_STREAMABLE is the central permutation-equivariance promise.
  ELEMENTWISE alone is insufficient: e.g. index-generating operations are
  elementwise but their values depend on the logical lane position.
  The shared region inventory can also contain scheduled hardware plans,
  including after re-preparing an already emitted VectorKernel. Those are
  not pointwise vector regions and cannot participate in this proof.
  """
  if (
    not isinstance(region, VectorRegion)
    or region.dtype != "float32"
    or not region.shape
    or not region.full_chunks
    or region.single_chunk
    or region.iteration_axis is not None
    or region.initializations
    or region.finalizations
    or len(region.effects) != 1
  ):
    return None
  effect = region.effects[0]
  if effect.kind != "store_convert" or effect.store is None:
    return None
  store = effect.store
  if (
    store.mask is not None
    or not store.destination.shape
    or store.destination.dtype not in {"float16", "bfloat16"}
    or store.destination.owned
    or I.constant(store.destination.strides[-1]) != 1
    or any(operation.opaque or operation.type.mask for operation in region.operations)
  ):
    return None
  forbidden = (
    Trait.VECTOR_BOUNDARY
    | Trait.STATE_REFERENCE
    | Trait.REGISTER_TRANSFORM
    | Trait.REDUCTION
    | Trait.UNSTABLE_EXTENT
    | Trait.REFERENCE
  )
  expressions, loads = {}, {}
  for root in region.roots:
    for value in walk_unique(root):
      expressions[id(value)] = value
      definition = catalog.get(value)
      if isinstance(value.type, I.ScalarType):
        # Already-computed uniform statistics/raw ABI scalars can be filled
        # into either lane layout. Do not pull a scalar memory/state read or
        # an effectful helper into the pointwise proof.
        if (
          definition.effects != "none"
          # A ScalarType local is already one uniform value for this body,
          # even when the surrounding serial loop changes it on its next
          # iteration. No scalar writes occur inside a pointwise region.
          or catalog.has(value, (forbidden & ~(Trait.REFERENCE | Trait.UNSTABLE_EXTENT)) | Trait.MEMORY_READ)
          or (catalog.has(value, Trait.REFERENCE | Trait.UNSTABLE_EXTENT) and bool(value.args))
          or not catalog.has(value, Trait.ELEMENTWISE | Trait.RUNTIME_SHAPE)
        ):
          return None
        continue
      if (
        not isinstance(value.type, I.TileType)
        or value.type.dtype != "float32"
        or value.type.mask
        or catalog.has(value, forbidden & ~Trait.UNSTABLE_EXTENT)
      ):
        return None
      if value.op == "load":
        view = value.data
        if (
          value.args
          or view.dtype not in {"float16", "bfloat16"}
          or view.owned
          or not view.shape
          or len(view.shape) > len(region.shape)
        ):
          return None
        # Leading-axis broadcasting is uniform for the SIMD axis. A unit
        # SIMD extent is also uniform; no vector memory permutation is needed
        # for that existing scalar-fill load path.
        broadcast = I.constant(view.shape[-1]) == 1
        if not broadcast and (I.constant(view.strides[-1]) != 1 or not I.same_expr(view.shape[-1], region.shape[-1])):
          return None
        loads[id(value)] = value
      elif (
        definition.effects != "none"
        or catalog.has(value, Trait.UNSTABLE_EXTENT)
        or not catalog.has(value, Trait.ELEMENTWISE)
        or not catalog.has(value, Trait.TRANSPOSE_STREAMABLE)
      ):
        return None
  if not loads:
    return None
  return PointwiseRegisterLayout("native-16-to-32", frozenset(expressions), tuple(loads.values()))


@register_pass("cpu-plan-pointwise-register-layout", stage="cpu-memory", after="cpu-plan-transpose")
def plan_register_layout(prepared, context):
  policy = context.options.get("cpu_pointwise_register_layout_policy", "auto")
  if not isinstance(policy, str) or policy not in {"off", "auto"}:
    raise ValueError("cpu_pointwise_register_layout_policy must be 'off' or 'auto'")
  if "cpu_pointwise_register_layout" in context.options:
    enabled = context.options["cpu_pointwise_register_layout"]
    if not isinstance(enabled, bool):
      raise TypeError("cpu_pointwise_register_layout must be bool")
    policy = "force" if enabled else "off"
  target = context.environment.codegen_target
  if policy == "off" or target.architecture != "arm" or target.isa not in {"sve", "sme"}:
    return PassResult(prepared, changed=False)
  regions = {}
  for region in (
    *prepared.vectors.regions,
    *prepared.stores.values(),
    *prepared.axes.values(),
    *prepared.snapshots.values(),
    *prepared.reductions.values(),
  ):
    regions[id(region)] = region
  layouts, decisions = {}, {}
  for identity, region in regions.items():
    proof = prove_pointwise_layout(region, context.environment.operations)
    if proof is not None:
      if policy == "auto":
        decision = assess_register_layout(region, proof, context.environment.costs)
        decisions[identity] = decision
        if not decision.selected:
          continue
      layouts[identity] = proof
  return replace(
    prepared, register_layouts=MappingProxyType(layouts), register_layout_decisions=MappingProxyType(decisions)
  )
