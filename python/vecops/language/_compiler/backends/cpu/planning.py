"""Prepare every CPU emission alternative before printing C++.

Alias fallbacks, snapshots and axis-reduction tails used to run compiler passes
inside the emitter. The catalog separates that work from source printing. Its
unused alternatives are not reported as emitted regions; identifier binding
for scratch buffers does not re-run lowering or optimization.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, replace
from types import MappingProxyType

from ... import ir as I
from ...ir.iteration import AxisResult
from ...lowering.tile_to_vector import lower_axis, lower_axis_fold, lower_reduction, lower_region, lower_stores
from ...operations import Trait, has
from ...pipeline import cpu_memory_pipeline


def snapshot_strides(view):
  """Keep a proven unit-stride axis in dense snapshots.

  Row-major snapshots of K-contiguous B introduce a gather alternative in
  every otherwise contiguous dot load, even when the operands do not alias.
  Preserve that axis without retaining source padding, negative strides or
  overlapping/broadcast storage. Unknown layouts retain the row-major default.
  """
  rank = len(view.shape)
  unit = next((axis for axis in reversed(range(rank)) if I.constant(view.strides[axis]) == 1), rank - 1)
  order = tuple(axis for axis in range(rank) if axis != unit) + (unit,)
  strides, extent = [None] * rank, I.literal(1)
  for axis in reversed(order):
    strides[axis] = extent
    extent = I.binary("mul", extent, view.shape[axis])
  return tuple(strides)


@dataclass(frozen=True)
class PreparedKernel:
  vectors: object
  stores: object
  axes: object
  snapshots: object
  reductions: object
  transposes: object
  matmuls: object
  packing_permutation: object | None = None
  packing_permutation_x86: object | None = None
  dense_packs: object = field(default_factory=dict)
  grouped_epilogues: frozenset = frozenset()
  grids: object = field(default_factory=dict)
  grid_regions: object = field(default_factory=dict)
  states: object = field(default_factory=dict)
  state_regions: object = field(default_factory=dict)
  primitives: object = field(default_factory=dict)
  register_layouts: object = field(default_factory=dict)
  register_layout_decisions: object = field(default_factory=dict)
  pass_trace: list = field(default_factory=list)

  def store(self, statement):
    return self.stores[id(statement)]

  def axis(self, expression, unroll):
    return self.axes[(id(expression), unroll)]

  def reduction(self, expressions, stores=()):
    return self.reductions[(tuple(map(id, expressions)), tuple(map(id, stores)))]

  def snapshot(self, load, storage):
    template = self.snapshots[id(load)]
    original = template.effects[0].store
    statement = replace(original, destination=replace(original.destination, storage=storage))
    region = replace(template, effects=tuple(replace(effect, store=statement) for effect in template.effects))
    return statement, region


def prepare(vectors, *, memory_manager=None):
  """Lower all latent regions using the same per-compilation pass snapshot."""
  manager = vectors.manager
  stores, axes, snapshots, reductions, transposes, matmuls = {}, {}, {}, {}, {}, {}
  seen = set()
  state_regions = {}
  primitives = {}

  def visit(value):
    if isinstance(value, (tuple, list)):
      for item in value:
        visit(item)
      return
    if not isinstance(value, I.Node) or id(value) in seen:
      return
    seen.add(id(value))
    if isinstance(value, I.IndexedStore):
      indices = I.Expr(I.shaped_type("int64", value.indices.type.shape, value.indices.type), "cast", (value.indices,))
      roots = (indices, value.value) + ((value.mask,) if value.mask is not None else ())
      primitives[id(value)] = lower_region(
        value.indices.type.shape, value.value.type.dtype, roots, unroll=1, manager=manager
      )
    if isinstance(value, (I.DeclareState, I.UpdateState)):
      from .state import state_view, stored_value

      store = I.Store(state_view(value.state), stored_value(value.state, value.value))
      state_regions[id(value)] = lower_stores((store,), manager=manager)
      state_regions[(id(value), "register")] = lower_region(
        value.state.type.shape, value.state.type.dtype, (value.value,), unroll=1, manager=manager
      )
    if isinstance(value, I.Store) and value.destination.shape and id(value) not in vectors.plans:
      stores[id(value)] = lower_stores((value,), opaque=vectors.kernel.materializations, manager=manager)
    if isinstance(value, I.Expr):
      if has(value, Trait.MASK_REDUCTION):
        source = value.args[0]
        primitives[id(value)] = lower_region(
          source.type.shape, source.type.dtype, (source,), unroll=value.data, manager=manager
        )
      if value.op == "axis_reduce":
        axis, operation, unroll = value.data
        for count in dict.fromkeys((unroll, 1)):
          axes[(id(value), count)] = (
            lower_axis(value.args[0], value.type.dtype, axis, operation, count, manager=manager)
            if not isinstance(value.data, AxisResult)
            else lower_axis_fold(value.data.fold, value.type.dtype, count, manager=manager)
          )
      elif has(value, Trait.SCALAR_REDUCTION) and id(value) not in vectors.plans:
        reductions[((id(value),), ())] = lower_reduction((value,), manager=manager)
      elif has(value, Trait.MEMORY_READ) and isinstance(value.type, I.TileType):
        shape = value.type.shape
        destination = I.View(
          "snapshot",
          value.type.dtype,
          shape,
          snapshot_strides(value.data)
          if value.op == "load"
          else snapshot_strides(I.View("", value.type.dtype, shape, (I.literal(1),) * len(shape), I.literal(0))),
          I.literal(0),
          owned=True,
        )
        snapshots[id(value)] = lower_stores(
          (I.Store(destination, value),), opaque=vectors.kernel.materializations, manager=manager
        )
    for descriptor in fields(value):
      visit(getattr(value, descriptor.name))

  visit(vectors.kernel.body)
  visit(tuple(vectors.kernel.bindings.values()))
  # Registration changes after this point cannot change a source emission.
  prepared = PreparedKernel(
    vectors,
    *(MappingProxyType(plans) for plans in (stores, axes, snapshots, reductions, transposes, matmuls)),
    state_regions=MappingProxyType(state_regions),
    primitives=MappingProxyType(primitives),
  )
  prepared = (memory_manager or cpu_memory_pipeline(options=manager.options)).run(prepared)
  return prepared
