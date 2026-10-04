"""Hoist row statistics; leave normalization's elementwise tail visible.

The same DAG survives inlined T.func helpers. A matrix-product/packing policy
can fuse the tail, but never recompute its reductions for every microtile.
"""

from dataclasses import fields, replace

from ... import ir as I
from ...operations import Trait
from ..manager import register_pass


@register_pass("decompose-row-normalization", before=("canonicalize-memory",))
def decompose_row_normalization(ir, context):
  memo, counter = {}, 0

  def visit(block):
    nonlocal counter
    result = []

    def rewrite(value):
      nonlocal counter
      if isinstance(value, I.Expr):
        if id(value) in memo:
          return memo[id(value)]
        args = tuple(rewrite(a) for a in value.args)
        if value.op != "row_normalize":
          memo[id(value)] = replace(value, args=args)
          return memo[id(value)]
        source, eps = args
        kind, variance, unroll, accuracy = value.data
        dtype = source.type.dtype
        m, k = source.type.shape
        counter += 1
        prefix = f"dsl_o_norm_{counter}"
        row = I.Expr(I.ScalarType("int64"), "local", data=prefix + "_row")
        statistics = []
        for name in ("scale",) if kind == "rmsnorm" else ("center", "scale"):
          view = I.View(prefix + "_" + name, dtype, (m,), (I.literal(1),), I.literal(0), owned=True)
          statistics.append(view)
          result.append(I.Allocate(view))

        def slice_row(e):
          if isinstance(e.type, I.ScalarType):
            return e
          if len(e.type.shape) == 1:
            return e
          if len(e.type.shape) != 2:
            raise TypeError("row normalization requires scalar or rank-two elementwise operands")
          if e.op != "load" and not context.environment.operations.has(e, Trait.ELEMENTWISE):
            # A shape/index/state boundary owns its coordinates. Slicing its
            # children as though they were elementwise would silently turn
            # e.g. an expanded row predicate into a column predicate.
            typ = I.TileType(e.type.dtype, (e.type.shape[1],), e.type.mask)
            index = I.literal(0) if I.constant(e.type.shape[0]) == 1 else row
            return I.Expr(typ, "value_transform", (e,), ("row", index))
          data = e.data
          if e.op == "load":
            index = I.literal(0) if I.constant(data.shape[0]) == 1 else row
            data = replace(
              data,
              shape=(data.shape[1],),
              strides=(data.strides[1],),
              offset=I.binary("add", data.offset, I.binary("mul", index, data.strides[0])),
            )
          return replace(
            e,
            type=I.TileType(e.type.dtype, (e.type.shape[1],), e.type.mask),
            args=tuple(slice_row(a) for a in e.args),
            data=data,
          )

        x = slice_row(source)
        reduce = lambda op, e: I.Expr(I.ScalarType(dtype), "reduce_" + op, (e,), unroll)
        local = lambda name: I.Expr(I.ScalarType(dtype), "local", data=prefix + "_" + name)
        scalar = lambda e: I.Expr(I.ScalarType(dtype), "cast", (e,))
        body = []
        # Keep the statistics outside any consumer's matrix/packing loops.
        if kind == "rmsnorm":
          variance = I.binary("div", reduce("add", I.binary("mul", x, x)), scalar(k))
          inv = local("invstd")
          body.append(
            I.Assign(
              inv.data, I.Expr(I.ScalarType(dtype), "rsqrt", (I.binary("add", variance, scalar(eps)),), accuracy)
            )
          )
          center, scale = scalar(I.literal(0)), inv
        elif kind == "layernorm":
          mean = local("mean")
          body.append(I.Assign(mean.data, I.binary("div", reduce("add", x), k)))
          centered = I.binary("sub", x, mean)
          var = (
            I.binary(
              "max",
              I.binary("sub", I.binary("div", reduce("add", I.binary("mul", x, x)), k), I.binary("mul", mean, mean)),
              scalar(I.literal(0)),
            )
            if variance == "moments"
            else I.binary("div", reduce("add", I.binary("mul", centered, centered)), k)
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
            # Empty/all-masked rows have maximum=-inf: exp(-inf-0)=0.
            safe = local("safe_maximum")
            body.append(
              I.Assign(
                safe.data,
                I.Expr(
                  I.ScalarType(dtype),
                  "where",
                  (
                    I.binary("eq", maximum, I.literal(float("-inf"), dtype)),
                    scalar(I.literal(0)),
                    maximum,
                  ),
                ),
              )
            )
            maximum = safe
          exponent = I.Expr(x.type, "exp_neg", (I.binary("sub", x, maximum),), accuracy)
          total = reduce("add", exponent)
          if kind == "masked_softmax":
            denominator = local("denominator")
            body.append(I.Assign(denominator.data, total))
            total = I.Expr(
              I.ScalarType(dtype),
              "where",
              (
                I.binary("gt", denominator, scalar(I.literal(0))),
                denominator,
                scalar(I.literal(1)),
              ),
            )
          inv = local("inverse")
          body.append(I.Assign(inv.data, I.Expr(I.ScalarType(dtype), "rcp", (total,), accuracy)))
          center, scale = maximum, inv
        for view, e in zip(statistics, ((scale,) if kind == "rmsnorm" else (center, scale))):
          body.append(I.Store(replace(view, shape=(), strides=(), offset=row), e))
        result.append(I.For(row.data, I.literal(0), m, I.literal(1), I.Block(body), unroll=1))
        loaded = []
        for view in statistics:
          broadcast = replace(view, shape=(m, I.literal(1)), strides=(I.literal(1), I.literal(0)))
          loaded.append(I.Expr(I.TileType(dtype, broadcast.shape), "load", data=broadcast))
        tail = source if kind == "rmsnorm" else I.binary("sub", source, loaded[0])
        if kind in {"softmax", "masked_softmax"}:
          tail = I.Expr(tail.type, "exp_neg", (tail,), accuracy)
        memo[id(value)] = I.binary("mul", tail, loaded[-1])
        return memo[id(value)]
      if isinstance(value, I.Block):
        visit(value)
        return value
      if isinstance(value, tuple):
        return tuple(rewrite(v) for v in value)
      if isinstance(value, I.Node):
        return replace(value, **{f.name: rewrite(getattr(value, f.name)) for f in fields(value)})
      return value

    for statement in block.statements:
      result.append(rewrite(statement))
    block.statements = result

  visit(ir.body)
  return ir
