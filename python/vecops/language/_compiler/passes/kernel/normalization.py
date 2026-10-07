"""Normalize row statistics without hiding the elementwise consumer DAG.

A sole row-separable store can stream statistics and its epilogue together.
Other consumers retain materialized statistics outside matrix/packing loops.
Both paths share executable scalar and reduction IR construction.
"""

from dataclasses import fields, replace

from ... import ir as I
from ...frontend.expression_helper import expand_expression_helper
from ...ir.traversal import children, nodes, walk_unique
from ...operations import Trait
from ..manager import register_pass


def _slice_row(value, row, catalog):
  if isinstance(value.type, I.ScalarType) or len(value.type.shape) == 1:
    return value
  if len(value.type.shape) != 2:
    raise TypeError("row normalization requires scalar or rank-two elementwise operands")
  if value.op != "load" and not catalog.has(value, Trait.ELEMENTWISE):
    # Index/layout/state boundaries own their coordinates. Do not slice their
    # children as though an expanded row predicate were a column predicate.
    typ = I.TileType(value.type.dtype, (value.type.shape[1],), value.type.mask)
    index = I.literal(0) if I.constant(value.type.shape[0]) == 1 else row
    return I.Expr(typ, "value_transform", (value,), ("row", index))
  data = value.data
  if value.op == "load":
    index = I.literal(0) if I.constant(data.shape[0]) == 1 else row
    data = replace(
      data,
      shape=(data.shape[1],),
      strides=(data.strides[1],),
      offset=I.binary("add", data.offset, I.binary("mul", index, data.strides[0])),
    )
  return replace(
    value,
    type=I.TileType(value.type.dtype, (value.type.shape[1],), value.type.mask),
    args=tuple(_slice_row(argument, row, catalog) for argument in value.args),
    data=data,
  )


def _exponent(value, accuracy, prefix, context, bindings):
  if value.type.dtype != "float32" or accuracy != "estimate":
    return I.Expr(value.type, "exp_neg", (value,), accuracy)
  # Rank-one helpers and both rank-two schedules parse the same safe F32
  # polynomial, not another implementation hidden in a pass.
  from ...._library.math import softmax_exp_neg

  result, prelude = expand_expression_helper(
    softmax_exp_neg,
    (value, accuracy),
    catalog=context.environment.operations,
    prefix=prefix,
  )
  bindings.extend(prelude)
  return result


def _inverse(value, accuracy, prefix, context, bindings):
  # Keep rank-one and both rank-two schedules on one visible DSL algorithm.
  # A strict softmax normalization factor is IEEE 1/sum, not a rewrite of
  # the explicit rcp primitive's native estimate/refinement contract.
  from ...._library.math import softmax_inverse

  result, prelude = expand_expression_helper(
    softmax_inverse,
    (value, accuracy),
    catalog=context.environment.operations,
    prefix=prefix,
  )
  bindings.extend(prelude)
  return result


def _row_statistics(value, row, prefix, context, *, preserve_capture=False):
  """Build statistics and an optional producer reusable within this row.

  A helper expansion can introduce lexical frozen references that CSE must
  not equate with references from another expansion. Preserve the producer's
  identity for a same-row consumer instead; the generic cost/reuse planner
  still decides whether it should be cached. A materialized-statistics
  schedule must not use this row-local producer after the row loop exits.
  """
  source, eps = value.args
  kind, variance, unroll, accuracy = value.data
  dtype, (_, width) = source.type.dtype, source.type.shape
  if preserve_capture:
    # A sliced load has another identity, so it cannot find a snapshot saved
    # for the original captured tile before an intervening write. Project the
    # original value instead: the shared row-transform lowering retains its
    # operand identity and selects that capture's saved data/layout as needed.
    index = I.literal(0) if I.constant(source.type.shape[0]) == 1 else row
    x = I.Expr(I.TileType(dtype, (width,), source.type.mask), "value_transform", (source,), ("row", index))
  else:
    x = _slice_row(source, row, context.environment.operations)
  invariant, body, producer = [], [], None

  def reduce(operation, expression):
    return I.Expr(I.ScalarType(dtype), "reduce_" + operation, (expression,), unroll)

  def local(name):
    return I.Expr(I.ScalarType(dtype), "local", data=prefix + "_" + name)

  def scalar(expression):
    return I.Expr(I.ScalarType(dtype), "cast", (expression,))

  if kind == "rmsnorm":
    var = I.binary("div", reduce("add", I.binary("mul", x, x)), scalar(width))
    inv = local("invstd")
    body.append(
      I.Assign(inv.data, I.Expr(I.ScalarType(dtype), "rsqrt", (I.binary("add", var, scalar(eps)),), accuracy))
    )
    center, scale = scalar(I.literal(0)), inv
  elif kind == "layernorm":
    inverse_count = local("inv_count")
    invariant.append(I.Assign(inverse_count.data, I.binary("div", scalar(I.literal(1)), scalar(width))))
    mean = local("mean")
    body.append(I.Assign(mean.data, I.binary("mul", reduce("add", x), inverse_count)))
    centered = I.binary("sub", x, mean)
    var = (
      I.binary(
        "max",
        I.binary(
          "sub",
          I.binary("mul", reduce("add", I.binary("mul", x, x)), inverse_count),
          I.binary("mul", mean, mean),
        ),
        scalar(I.literal(0)),
      )
      if variance == "moments"
      else I.binary("mul", reduce("add", I.binary("mul", centered, centered)), inverse_count)
    )
    inv = local("invstd")
    body.append(
      I.Assign(inv.data, I.Expr(I.ScalarType(dtype), "rsqrt", (I.binary("add", var, scalar(eps)),), "strict"))
    )
    center, scale = mean, inv
  else:
    maximum = local("maximum")
    body.append(I.Assign(maximum.data, reduce("max", x)))
    if kind == "masked_softmax":
      safe = local("safe_maximum")
      body.append(
        I.Assign(
          safe.data,
          I.Expr(
            I.ScalarType(dtype),
            "where",
            (I.binary("eq", maximum, I.literal(float("-inf"), dtype)), scalar(I.literal(0)), maximum),
          ),
        )
      )
      maximum = safe
    exponential = _exponent(I.binary("sub", x, maximum), accuracy, prefix + "_exp", context, body)
    producer = exponential
    total = reduce("add", exponential)
    if kind == "masked_softmax":
      denominator = local("denominator")
      body.append(I.Assign(denominator.data, total))
      total = I.Expr(
        I.ScalarType(dtype),
        "where",
        (I.binary("gt", denominator, scalar(I.literal(0))), denominator, scalar(I.literal(1))),
      )
    inv = local("inverse")
    inverse = _inverse(total, accuracy, prefix + "_reciprocal", context, body)
    body.append(I.Assign(inv.data, inverse))
    center, scale = maximum, inv
  return invariant, body, center, scale, x, producer


def _normalization_tail(source, center, scale, kind, accuracy, prefix, context, bindings, *, producer=None):
  tail = producer
  if tail is None:
    tail = source if kind == "rmsnorm" else I.binary("sub", source, center)
    if kind in {"softmax", "masked_softmax"}:
      tail = _exponent(tail, accuracy, prefix + "_tail", context, bindings)
  return I.binary("mul", tail, scale)


def _streamable(value, normalization, shape, catalog):
  """Prove row-separability; unknown transforms remain materialized."""
  for expression in walk_unique(value):
    if expression is normalization:
      continue
    definition = catalog.get(expression)
    if (
      definition.effects not in {"none", "read"}
      or catalog.has(
        expression, Trait.STATE_REFERENCE | Trait.REDUCTION | Trait.REGISTER_TRANSFORM | Trait.VECTOR_BOUNDARY
      )
      or expression.op != "load"
      and not isinstance(expression.type, I.ScalarType)
      and not catalog.has(expression, Trait.ELEMENTWISE)
    ):
      return False
    if not isinstance(expression.type, I.ScalarType):
      own_shape = expression.type.shape
      if not 1 <= len(own_shape) <= 2 or any(
        I.constant(a) != 1 and not I.same_expr(a, b) for a, b in zip(reversed(own_shape), reversed(shape))
      ):
        return False
  return True


@register_pass(
  "stream-row-normalization",
  after=("normalize-reduction-layout",),
  before=("decompose-row-normalization",),
)
def stream_row_normalization(ir, context):
  """Fuse a sole direct store, guarded by full-region read/write disjointness."""
  if not context.options.get("row_normalization_streaming", True):
    return ir
  catalog, consumers, bindings, read_consumers, serial = context.environment.operations, {}, {}, {}, 0

  def collect(block):
    for index, statement in enumerate(block.statements):
      if isinstance(statement, I.BindTile):
        for load in I.loads_unique(statement.value):
          bindings.setdefault(id(load), (id(block), index))
      if not isinstance(statement, I.BindTile):
        # BindTile is lazy-load provenance, not an evaluated tile consumer.
        for field in fields(statement):
          if field.name not in {"body", "otherwise", "loop", "copies"}:
            for node in nodes(getattr(statement, field.name)):
              if isinstance(node, I.Expr) and node.op == "row_normalize":
                consumers.setdefault(id(node), set()).add(id(statement))
              if isinstance(node, I.Expr) and catalog.has(node, Trait.MEMORY_READ):
                read_consumers.setdefault(id(node), set()).add(id(statement))
      for child in children(statement):
        collect(child)

  collect(ir.body)

  def visit(block):
    nonlocal serial
    candidates = {}
    released_bindings = set()
    for index, statement in enumerate(block.statements):
      for child in children(statement):
        visit(child)
      if not isinstance(statement, I.Store) or statement.mask is not None or len(statement.destination.shape) != 2:
        continue
      normalizations = {id(e): e for e in walk_unique(statement.value) if e.op == "row_normalize"}
      if len(normalizations) != 1:
        continue
      identity, normalization = next(iter(normalizations.items()))
      shape = normalization.type.shape
      if (
        consumers.get(identity) != {id(statement)}
        or len(shape) != 2
        or not all(I.same_expr(a, b) for a, b in zip(shape, statement.destination.shape))
        or not _streamable(statement.value, normalization, shape, catalog)
      ):
        continue
      loads = tuple(I.loads_unique(statement.value))
      captured = [bindings[id(load)] for load in loads if id(load) in bindings]
      if any(owner != id(block) for owner, _ in captured):
        # An enclosing captured tile may already have a preserved snapshot.
        # Keep its original evaluation/storage boundary in this first pass.
        continue
      first = min((position for _, position in captured), default=index)
      if any(not isinstance(previous, I.BindTile) for previous in block.statements[first:index]):
        # Do not move normalization through writes, scalar state updates or
        # other effects between a lazy tile capture and its sole consumer.
        continue
      serial += 1
      prefix = f"dsl_o_norm_stream_{serial}"
      row = I.Expr(I.ScalarType("int64"), "local", data=prefix + "_row")
      invariant, body, center, scale, source, producer = _row_statistics(normalization, row, prefix, context)
      kind, _, _, accuracy = normalization.data
      tail = _normalization_tail(source, center, scale, kind, accuracy, prefix, context, body, producer=producer)
      row_values = {}

      def substitute(value, normalization=normalization, tail=tail, row=row, row_values=row_values):
        if id(value) in row_values:
          return row_values[id(value)]
        if value is normalization:
          result = tail
        elif isinstance(value.type, I.ScalarType):
          result = value
        elif value.op == "load":
          result = _slice_row(value, row, catalog)
        else:
          result = replace(
            value,
            type=I.TileType(value.type.dtype, (value.type.shape[-1],), value.type.mask),
            args=tuple(substitute(argument) for argument in value.args),
          )
        row_values[id(value)] = result
        return result

      destination = statement.destination
      sink = replace(
        statement,
        destination=replace(
          destination,
          shape=(destination.shape[1],),
          strides=(destination.strides[1],),
          offset=I.binary("add", destination.offset, I.binary("mul", row, destination.strides[0])),
        ),
        value=substitute(statement.value),
      )
      body.append(sink)
      # Per-row writable checks cannot prove that distinct output rows do
      # not overlap. Invalid full layouts must take the original diagnostic.
      guard = I.Expr(I.ScalarType("bool"), "view_nonoverlap", data=destination)
      views = set()
      for load in loads:
        view = load.data
        if (view.owned or destination.owned) and view.storage != destination.storage or repr(view) in views:
          continue
        views.add(repr(view))
        guard = I.binary("and", guard, I.Expr(I.ScalarType("bool"), "view_disjoint", data=(view, destination)))
      fused = I.Block([*invariant, I.For(row.data, I.literal(0), shape[0], I.literal(1), I.Block(body), unroll=1)])
      candidates[id(statement)] = (normalization, I.If(guard, fused, I.Block([statement])))
      if context.options.get("row_normalization_release_captures", True) and all(
        read_consumers.get(id(load)) == {id(statement)} for load in loads
      ):
        # Only this sink consumes these reads, and the interval from their
        # earliest local capture contains no writes, scalar assignments or
        # other effects. The guard can therefore dominate their preservation:
        # disjoint rows need no snapshots, while the original fallback store
        # still snapshots overlapping/shifted operands before its writes.
        # Do not release captures observed by another statement or scope.
        private_reads = {id(load) for load in loads}
        consumer_values = {id(value) for value in walk_unique(statement.value)}
        for previous in block.statements[first:index]:
          own_reads = tuple(I.loads_unique(previous.value))
          if (
            own_reads and all(id(load) in private_reads for load in own_reads) and id(previous.value) in consumer_values
          ):
            # Being backed by a private load alone does not make an unused
            # dot, state/collective or architecture boundary disposable.
            # Only captures in the validated, pure sink DAG are released.
            # Its row_normalize boundary is the operation being rewritten;
            # all other boundaries and mutable scalar references stay put.
            safe = True
            for value in walk_unique(previous.value):
              if value is normalization or id(value) in private_reads:
                continue
              if (
                catalog.get(value).effects != "none"
                or catalog.has(value, Trait.STATE_REFERENCE | Trait.VECTOR_BOUNDARY | Trait.REGISTER_TRANSFORM)
                or not catalog.has(value, Trait.ELEMENTWISE | Trait.SCALAR_REDUCTION | Trait.RUNTIME_SHAPE | Trait.CSE)
                or catalog.has(value, Trait.UNSTABLE_EXTENT)
                and not catalog.has(value, Trait.SCALAR_REDUCTION)
              ):
                safe = False
                break
            if not safe:
              continue
            released_bindings.add(id(previous))
      ir.diagnostics.append(
        "row normalization streaming: scalar statistics and sole epilogue fused under whole-region non-alias guard"
      )
    if not candidates:
      return
    normalized = {id(value): value for value, _ in candidates.values()}
    provenance_values = {}

    def provenance(value):
      if id(value) in provenance_values:
        return provenance_values[id(value)]
      if id(value) in normalized:
        result = normalized[id(value)].args[0]
      else:
        args = tuple(provenance(argument) for argument in value.args)
        result = replace(value, args=args) if any(a is not b for a, b in zip(args, value.args)) else value
      provenance_values[id(value)] = result
      return result

    block.statements = [
      candidates[id(statement)][1]
      if id(statement) in candidates
      else I.BindTile(provenance(statement.value))
      if isinstance(statement, I.BindTile)
      else statement
      for statement in block.statements
      if id(statement) not in released_bindings
    ]

  visit(ir.body)
  return ir


@register_pass("decompose-row-normalization", before=("canonicalize-memory",))
def decompose_row_normalization(ir, context):
  memo, counter = {}, 0
  captures = {}

  def collect_captures(block):
    for index, statement in enumerate(block.statements):
      if isinstance(statement, I.BindTile):
        for load in I.loads_unique(statement.value):
          captures.setdefault(id(load), (id(block), index))
      for child in children(statement):
        collect_captures(child)

  collect_captures(ir.body)

  def visit(block):
    nonlocal counter
    result = []

    def rewrite(value):
      nonlocal counter
      if isinstance(value, I.Expr):
        if id(value) in memo:
          return memo[id(value)]
        args = tuple(rewrite(argument) for argument in value.args)
        if value.op != "row_normalize":
          memo[id(value)] = replace(value, args=args)
          return memo[id(value)]
        captured = [captures[id(load)] for load in I.loads_unique(value.args[0]) if id(load) in captures]
        preserve_capture = any(
          owner != id(block) or any(not isinstance(previous, I.BindTile) for previous in block.statements[first:index])
          for owner, first in captured
        )
        normalized = replace(value, args=args)
        source = args[0]
        kind, _, _, accuracy = value.data
        dtype, (height, _) = source.type.dtype, source.type.shape
        counter += 1
        prefix = f"dsl_o_norm_{counter}"
        row = I.Expr(I.ScalarType("int64"), "local", data=prefix + "_row")
        statistics = []
        for name in ("scale",) if kind == "rmsnorm" else ("center", "scale"):
          view = I.View(prefix + "_" + name, dtype, (height,), (I.literal(1),), I.literal(0), owned=True)
          statistics.append(view)
          result.append(I.Allocate(view))
        # The statistics loop ends before this schedule's consumers run.
        # Only stored center/scale values may escape, not its row producer.
        invariant, body, center, scale, _, _ = _row_statistics(
          normalized, row, prefix, context, preserve_capture=preserve_capture
        )
        result.extend(invariant)
        for view, expression in zip(statistics, ((scale,) if kind == "rmsnorm" else (center, scale))):
          body.append(I.Store(replace(view, shape=(), strides=(), offset=row), expression))
        result.append(I.For(row.data, I.literal(0), height, I.literal(1), I.Block(body), unroll=1))
        loaded = []
        for view in statistics:
          broadcast = replace(view, shape=(height, I.literal(1)), strides=(I.literal(1), I.literal(0)))
          loaded.append(I.Expr(I.TileType(dtype, broadcast.shape), "load", data=broadcast))
        center = I.literal(0, dtype) if kind == "rmsnorm" else loaded[0]
        memo[id(value)] = _normalization_tail(source, center, loaded[-1], kind, accuracy, prefix, context, result)
        return memo[id(value)]
      if isinstance(value, I.Block):
        visit(value)
        return value
      if isinstance(value, tuple):
        return tuple(rewrite(v) for v in value)
      if isinstance(value, I.Node):
        return replace(value, **{field.name: rewrite(getattr(value, field.name)) for field in fields(value)})
      return value

    for index, statement in enumerate(block.statements):
      result.append(rewrite(statement))
    block.statements = result

  visit(ir.body)
  return ir
