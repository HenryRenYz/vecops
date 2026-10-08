# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Schedule source-parsed pure row regions without recognizing algorithms.

A sole row-separable store can stream statistics and its epilogue together.
Other consumers retain materialized statistics outside matrix/packing loops.
Both paths share executable scalar and reduction IR construction.
"""

from dataclasses import fields, replace

from ... import ir as I
from ...ir.traversal import children, nodes, walk_unique
from ...operations import Trait
from ..manager import register_pass


def _slice_row(value, row, catalog):
  if isinstance(value.type, I.ScalarType) or len(value.type.shape) == 1:
    return value
  if len(value.type.shape) != 2:
    raise TypeError("row projection requires scalar or rank-two elementwise operands")
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


def _row_program(value, row, prefix, context, *, preserve_capture=False):
  """Instantiate one hygienic program and hoist its row-invariant scalars."""
  source = value.args[0]
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
  recipe, memo = value.data, {}
  names = {s.target: prefix + "_" + s.target for s in recipe.body.statements if isinstance(s, I.Assign)}

  def clone(node):
    if node is recipe.argument:
      return x
    if isinstance(node, I.Expr):
      if id(node) not in memo:
        data = names.get(node.data, node.data) if node.op in {"local", "frozen"} else clone(node.data)
        memo[id(node)] = replace(node, args=tuple(clone(a) for a in node.args), data=data)
      return memo[id(node)]
    if isinstance(node, tuple):
      return tuple(clone(item) for item in node)
    if isinstance(node, I.Node):
      changes = {f.name: clone(getattr(node, f.name)) for f in fields(node)}
      if isinstance(node, I.Assign):
        changes["target"] = names[node.target]
      return replace(node, **changes)
    return node

  invariant, body, fixed = [], [], set()
  for statement in recipe.body.statements:
    statement = clone(statement)
    if isinstance(statement, I.Assign) and all(
      not isinstance(e.type, I.TileType)
      and not context.environment.operations.has(e, Trait.REDUCTION | Trait.MEMORY_READ | Trait.STATE_REFERENCE)
      and (e.op not in {"local", "frozen"} or e.data not in names.values() or e.data in fixed)
      for e in walk_unique(statement.value)
    ):
      invariant.append(statement)
      fixed.add(statement.target)
    else:
      body.append(statement)
  return invariant, body, clone(recipe.result), x


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
  "stream-row-maps",
  after=("normalize-reduction-layout",),
  before=("decompose-row-maps",),
)
def stream_row_maps(ir, context):
  """Fuse a sole direct store, guarded by full-region read/write disjointness."""
  if not context.options.get("row_map_streaming", context.options.get("row_normalization_streaming", True)):
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
              if isinstance(node, I.Expr) and node.op == "map_rows":
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
      normalizations = {id(e): e for e in walk_unique(statement.value) if e.op == "map_rows"}
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
      prefix = f"dsl_o_map_stream_{serial}"
      row = I.Expr(I.ScalarType("int64"), "local", data=prefix + "_row")
      invariant, body, tail, _ = _row_program(normalization, row, prefix, context)
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
      if context.options.get(
        "row_map_release_captures", context.options.get("row_normalization_release_captures", True)
      ) and all(read_consumers.get(id(load)) == {id(statement)} for load in loads):
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
            # Its map_rows boundary is the operation being rewritten;
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
        "row-map streaming: scalar intermediates and sole consumer fused under whole-region non-alias guard"
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


@register_pass("decompose-row-maps", before=("canonicalize-memory",))
def decompose_row_maps(ir, context):
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
        if value.op != "map_rows":
          memo[id(value)] = replace(value, args=args)
          return memo[id(value)]
        captured = [captures[id(load)] for load in I.loads_unique(value.args[0]) if id(load) in captures]
        preserve_capture = any(
          owner != id(block) or any(not isinstance(previous, I.BindTile) for previous in block.statements[first:index])
          for owner, first in captured
        )
        normalized = replace(value, args=args)
        source = args[0]
        height = source.type.shape[0]
        counter += 1
        prefix = f"dsl_o_map_{counter}"
        row = I.Expr(I.ScalarType("int64"), "local", data=prefix + "_row")
        invariant, body, tail, row_source = _row_program(
          normalized, row, prefix, context, preserve_capture=preserve_capture
        )
        definitions = {s.target: s.value for s in body if isinstance(s, I.Assign)}

        def reference(expression):
          seen = set()
          while expression.op in {"local", "frozen"} and expression.data in definitions:
            if expression.data in seen:
              raise TypeError("cyclic row scalar definition")
            seen.add(expression.data)
            alias = definitions[expression.data]
            if alias.op not in {"local", "frozen"}:
              break
            expression = alias
          return expression

        statistics = {}
        for expression in walk_unique(tail):
          expression = reference(expression)
          if expression.op in {"local", "frozen"} and expression.data in definitions:
            statistics.setdefault(expression.data, expression)
        broadcasts = {}
        for name, expression in statistics.items():
          view = I.View(name + "_rows", expression.type.dtype, (height,), (I.literal(1),), I.literal(0), owned=True)
          result.append(I.Allocate(view))
          body.append(I.Store(replace(view, shape=(), strides=(), offset=row), expression))
          broadcast = replace(view, shape=(height, I.literal(1)), strides=(I.literal(1), I.literal(0)))
          broadcasts[name] = I.Expr(I.TileType(view.dtype, broadcast.shape), "load", data=broadcast)
        result.extend(invariant)
        result.append(I.For(row.data, I.literal(0), height, I.literal(1), I.Block(body), unroll=1))
        lifted = {}

        def lift(expression):
          if expression is row_source:
            return source
          if id(expression) not in lifted:
            ref = reference(expression)
            if ref.op in {"local", "frozen"} and ref.data in broadcasts:
              answer = broadcasts[ref.data]
            else:
              arguments = tuple(lift(a) for a in expression.args)
              shaped = [a.type for a in arguments if isinstance(a.type, I.TileType)]
              typ = expression.type
              if shaped:
                shape = shaped[0].shape
                for own in shaped[1:]:
                  shape = I.broadcast_shape(shape, own.shape)
                mask = getattr(typ, "mask", False) or context.environment.operations.has(expression, Trait.COMPARISON)
                typ = I.TileType(typ.dtype, shape, mask)
              answer = replace(expression, type=typ, args=arguments)
            lifted[id(expression)] = answer
          return lifted[id(expression)]

        memo[id(value)] = lift(tail)
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
