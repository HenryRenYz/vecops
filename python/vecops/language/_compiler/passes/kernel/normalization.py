"""Hoist row statistics; leave normalization's elementwise tail visible.

The same DAG survives inlined T.func helpers. A matrix-product/packing policy
can fuse the tail, but never recompute its reductions for every microtile.
"""

from dataclasses import fields, replace
from ... import ir as I
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
        kind, variance, unroll = value.data
        dtype = source.type.dtype
        m, k = source.type.shape
        counter += 1
        prefix = f"dsl_o_norm_{counter}"
        row = I.Expr(I.ScalarType("int64"), "local", data=prefix + "_row")
        statistics = []
        for name in ("center", "scale"):
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
            e, type=I.TileType(e.type.dtype, (e.type.shape[1],)), args=tuple(slice_row(a) for a in e.args), data=data
          )

        x = slice_row(source)
        reduce = lambda op, e: I.Expr(I.ScalarType(dtype), "reduce_" + op, (e,), unroll)
        local = lambda name: I.Expr(I.ScalarType(dtype), "local", data=prefix + "_" + name)
        scalar = lambda e: I.Expr(I.ScalarType(dtype), "cast", (e,))
        body = []
        # Keep the statistics outside any consumer's matrix/packing loops.
        if kind == "layernorm":
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
          exponent = I.Expr(x.type, "exp_neg", (I.binary("sub", x, maximum),), "strict")
          inv = local("inverse")
          body.append(I.Assign(inv.data, I.Expr(I.ScalarType(dtype), "rcp", (reduce("add", exponent),), "strict")))
          center, scale = maximum, inv
        for view, e in zip(statistics, (center, scale)):
          body.append(I.Store(replace(view, shape=(), strides=(), offset=row), e))
        result.append(I.For(row.data, I.literal(0), m, I.literal(1), I.Block(body), unroll=1))
        loaded = []
        for view in statistics:
          broadcast = replace(view, shape=(m, I.literal(1)), strides=(I.literal(1), I.literal(0)))
          loaded.append(I.Expr(I.TileType(dtype, broadcast.shape), "load", data=broadcast))
        tail = I.binary("sub", source, loaded[0])
        if kind == "softmax":
          tail = I.Expr(tail.type, "exp_neg", (tail,), "strict")
        memo[id(value)] = I.binary("mul", tail, loaded[1])
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
