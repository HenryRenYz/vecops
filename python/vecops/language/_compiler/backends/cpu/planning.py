"""Prepare every CPU emission alternative before printing C++.

Alias fallbacks, snapshots and axis-reduction tails used to run compiler passes
inside the emitter. The catalog separates that work from source printing. Its
unused alternatives are not reported as emitted regions; identifier binding
for scratch buffers does not re-run lowering or optimization.
"""

from dataclasses import dataclass, field, fields, replace
from types import MappingProxyType
from ... import ir as I
from ...lowering.tile_to_vector import lower_stores, lower_axis, lower_reduction
from ...pipeline import cpu_memory_pipeline
from ...operations import has, Trait


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

  def visit(value):
    if isinstance(value, (tuple, list)):
      for item in value:
        visit(item)
      return
    if not isinstance(value, I.Node) or id(value) in seen:
      return
    seen.add(id(value))
    if isinstance(value, I.Store) and value.destination.shape and id(value) not in vectors.plans:
      stores[id(value)] = lower_stores((value,), opaque=vectors.kernel.materializations, manager=manager)
    if isinstance(value, I.Expr):
      if value.op == "axis_reduce":
        axis, operation, unroll = value.data
        for count in dict.fromkeys((unroll, 1)):
          axes[(id(value), count)] = lower_axis(
            value.args[0], value.type.dtype, axis, operation, count, manager=manager
          )
      elif has(value, Trait.SCALAR_REDUCTION) and id(value) not in vectors.plans:
        reductions[((id(value),), ())] = lower_reduction((value,), manager=manager)
      elif value.op == "load" and value.data.shape:
        shape = value.data.shape
        destination = I.View(
          "snapshot", value.type.dtype, shape, snapshot_strides(value.data), I.literal(0), owned=True
        )
        snapshots[id(value)] = lower_stores(
          (I.Store(destination, value),), opaque=vectors.kernel.materializations, manager=manager
        )
    for field in fields(value):
      visit(getattr(value, field.name))

  visit(vectors.kernel.body)
  visit(tuple(vectors.kernel.bindings.values()))
  # Registration changes after this point cannot change a source emission.
  prepared = PreparedKernel(
    vectors, *(MappingProxyType(plans) for plans in (stores, axes, snapshots, reductions, transposes, matmuls))
  )
  return (memory_manager or cpu_memory_pipeline(options=manager.options)).run(prepared)
