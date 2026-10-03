"""Tile consumers lower to explicit vector regions before target emission."""

import math
from .. import ir as I
from ..ir import vector as V
from ..dtypes import ELEMENT_BYTES
from ..passes.manager import PassManager
from ..operations import Trait, snapshot


def lower_region(shape, dtype, roots, *, unroll=2, opaque=(), manager=None, run_passes=True):
  operations, memo = [], {}
  opaque = set(opaque)
  catalog = manager.options["_compiler_environment"].operations if manager is not None else snapshot()

  def add(expression):
    if not isinstance(expression.type, I.TileType):
      return None
    if id(expression) in memo:
      return memo[id(expression)]
    boundary = catalog.has(expression, Trait.VECTOR_BOUNDARY) or id(expression) in opaque
    operands = tuple(None for _ in expression.args) if boundary else tuple(add(a) for a in expression.args)
    index = len(operations)
    typ = I.VectorType(
      expression.type.dtype, expression.type.shape, expression.type.mask, I.VectorTag(basis_dtype=dtype)
    )
    operations.append(V.VectorOp(index, expression, operands, typ, boundary))
    memo[id(expression)] = index
    return index

  root_indices = tuple(add(root) for root in roots)
  region = V.VectorRegion(shape, dtype, unroll, tuple(roots), operations, root_indices)
  tagged = [
    op.expression.type
    for op in operations
    if isinstance(op.expression.type, I.VectorType) and op.expression.type.tag.N is None
  ]
  if tagged:
    # Keep the source lane basis through conversions. Large virtual powers may
    # subsequently stream through smaller physical vectors on a chosen target.
    region.base_power = max(
      typ.tag.pow
      + int(math.log2(ELEMENT_BYTES[dtype]))
      - int(math.log2(ELEMENT_BYTES[typ.tag.basis_dtype or typ.dtype]))
      for typ in tagged
    )
  return (manager or PassManager(stage="vector")).run(region) if run_passes else region


def store_value(store):
  value = store.value
  return value.args[0] if value.op == "cast" and value.type.dtype == store.destination.dtype else value


def store_roots(stores):
  return tuple(
    root
    for store in stores
    for root in ((store_value(store), store.mask) if store.mask is not None else (store_value(store),))
  )


def attach_stores(region, stores, *, offset=0):
  effects = list(region.effects)
  for index, store in enumerate(stores):
    mask_root = offset + 1 if store.mask is not None else -1
    effects.append(
      V.VectorEffect(
        "store_convert",
        index=index,
        value_root=offset,
        mask_root=mask_root,
        dtype=store_value(store).type.dtype,
        store=store,
      )
    )
    offset += 2 if store.mask is not None else 1
  region.effects = tuple(effects)
  return region


def lower_reduction(expressions, *, stores=(), manager=None):
  source = expressions[0].args[0]
  roots = tuple(e.args[0] for e in expressions) + store_roots(stores)
  manager = manager or PassManager(stage="vector")
  catalog = manager.options["_compiler_environment"].operations
  region = lower_region(
    source.type.shape, source.type.dtype, roots, unroll=expressions[0].data, manager=manager, run_passes=False
  )
  region.initializations = tuple(
    V.VectorEffect("acc_init", catalog.get(e).reduction, i, dtype=e.type.dtype) for i, e in enumerate(expressions)
  )
  region.effects = tuple(
    V.VectorEffect("accumulate", catalog.get(e).reduction, i, i, dtype=e.type.dtype) for i, e in enumerate(expressions)
  )
  region.finalizations = tuple(
    V.VectorEffect("horizontal_reduce", catalog.get(e).reduction, i, dtype=e.type.dtype)
    for i, e in enumerate(expressions)
  )
  attach_stores(region, stores, offset=len(expressions))
  return (manager or PassManager(stage="vector")).run(region)


def lower_stores(stores, *, opaque=(), manager=None):
  dtype = store_value(stores[0]).type.dtype
  dtype = stores[0].destination.dtype if dtype == "bool" else dtype
  region = lower_region(
    stores[0].destination.shape, dtype, store_roots(stores), opaque=opaque, manager=manager, run_passes=False
  )
  attach_stores(region, stores)
  return (manager or PassManager(stage="vector")).run(region)


def lower_axis(source, dtype, axis, operation, unroll, *, manager=None):
  region = lower_region(source.type.shape, dtype, (source,), unroll=unroll, manager=manager, run_passes=False)
  region.iteration_axis = axis
  region.initializations = (V.VectorEffect("acc_init", operation, dtype=dtype),)
  region.effects = (V.VectorEffect("accumulate", operation, value_root=0, dtype=dtype),)
  region.finalizations = (V.VectorEffect("merge_vector_accumulators", operation, dtype=dtype),)
  return (manager or PassManager(stage="vector")).run(region)


def lower_kernel(ir, *, manager=None):
  manager = manager or PassManager(stage="vector")
  result = V.VectorKernel(ir, manager)

  def remember(identity, region):
    result.plans[identity] = region
    result.regions.append(region)

  def expressions(value):
    if (
      manager.options["_compiler_environment"].operations.has(value, Trait.SCALAR_REDUCTION)
      and id(value) not in result.plans
    ):
      remember(id(value), lower_reduction((value,), manager=manager))
    for argument in value.args:
      expressions(argument)

  def block(body):
    for statement in body.statements:
      if isinstance(statement, I.UnrolledFor):
        block(statement.copies)
        block(statement.loop.body)
      elif isinstance(statement, I.ReductionGroup):
        remember(
          id(statement),
          lower_reduction(tuple(a.value for a in statement.assignments), stores=statement.stores, manager=manager),
        )
      elif isinstance(statement, (I.Store, I.ElementwiseGroup)):
        stores = (statement,) if isinstance(statement, I.Store) else statement.stores
        if stores[0].destination.shape:
          remember(id(statement), lower_stores(stores, opaque=ir.materializations, manager=manager))
      for name in ("value", "condition", "mask"):
        value = getattr(statement, name, None)
        if isinstance(value, I.Expr):
          expressions(value)
      for name in ("body", "otherwise"):
        child = getattr(statement, name, None)
        if isinstance(child, I.Block):
          block(child)

  block(ir.body)
  ir.vector_regions = result.regions
  return result
