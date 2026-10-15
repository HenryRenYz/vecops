# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Preserve matrix products as fusion boundaries, eliminate redundant sinks."""

from dataclasses import replace

from ... import ir as I
from ...ir.traversal import children, walk
from ..manager import register_pass


@register_pass("fuse-dot-epilogue", after=("expression-cse",), before=("fuse-elementwise",))
def fuse_dot_epilogue(ir, context):
  """Drop an overwritten product/captured-result sink before its epilogue.

  Batched helpers return a load of their private complete result. Treat that
  captured value like a product when the next store uses the same expression;
  Fresh reads of the output retain the first store's observable effect.
  Private-result fusion also retains fresh external reads: an external input
  can alias the output and observe the helper's original write.
  """
  def same_view(a, b):
    return (
      a.storage == b.storage
      and a.dtype == b.dtype
      and len(a.shape) == len(b.shape)
      and all(I.same_expr(x, y) for x, y in zip((*a.shape, *a.strides, a.offset), (*b.shape, *b.strides, b.offset)))
    )

  def visit(block):
    result = []
    for statement in block.statements:
      for child in children(statement):
        visit(child)
      previous = next((s for s in reversed(result) if not isinstance(s, I.BindTile)), None)
      if (
        isinstance(previous, I.Store)
        and (previous.value.op == "dot" or
             previous.value.op == "load" and not previous.value.args and
             previous.value.data.owned and
             previous.value.data.storage != previous.destination.storage)
        and previous.mask is None
        and isinstance(statement, I.Store)
        and statement.mask is None
        and same_view(previous.destination, statement.destination)
        and any(e is previous.value for e in walk(statement.value))
      ):
        product_loads = {id(load) for load in I.loads(previous.value)}
        captured_before_sink = set()
        if previous.value.op == "load":
          for earlier in result:
            if earlier is previous:
              break
            if isinstance(earlier, I.BindTile):
              captured_before_sink.update(id(load) for load in I.loads(earlier.value))
        # A fresh read of C after the first effect observes the computed C.
        # It must not be silently changed into a read of the old C.
        if not any(
          load.data.storage == previous.destination.storage and id(load) not in product_loads
          for load in I.loads(statement.value)
        ) and (previous.value.op == "dot" or not any(
          not load.data.owned and id(load) not in product_loads and id(load) not in captured_before_sink
          for load in I.loads(statement.value)
        )):
          result.remove(previous)
      result.append(statement)
    block.statements = result

  visit(ir.body)
  return ir


@register_pass("orient-matmul", after=("fuse-dot-epilogue",), before=("fuse-elementwise",))
def orient_matmul(ir, context):
  """Orient the complete producer/consumer DAG using proven layout metadata.

  `output` retains the previous column-contiguous policy. `target` additionally
  uses AMX role/shape costs, and leaves eligible SME zero-epilogue products to
  their direct vertical stores. Unknown layouts never become layout promises.
  """
  policy = context.options.get("cpu_matmul_orientation", "output")
  if policy not in {"output", "target"}:
    raise ValueError("cpu_matmul_orientation must be output or target")

  def should_swap(statement, products):
    destination = statement.destination
    column = I.constant(destination.strides[0]) == 1 and I.constant(destination.strides[1]) != 1
    if policy == "output":
      return column
    target = context.environment.codegen_target
    if len(products) != 1:
      return column
    product = products[0]
    if target.isa == "sme" and statement.value is product and destination.dtype == product.type.dtype:
      return False
    a, b = product.args[:2]
    if (target.architecture != "x86" or target.isa == "scalar" or
        a.type.dtype not in {"bfloat16", "float16", "int8", "uint8"} or
        b.type.dtype not in {"bfloat16", "float16", "int8", "uint8"} or
        product.data.implementation not in {"auto", "amx"}):
      return column
    def backing(value):
      while value.op == "cast":
        value = value.args[0]
      return value.data if value.op == "load" and not value.args else None
    left, right = backing(a), backing(b)
    if left is None or right is None:
      return column
    a_row = I.constant(left.strides[1]) == 1
    a_column = I.constant(left.strides[0]) == 1
    # C++ exposes B as [N,K]; DSL uses [K,N]. Its BRow means K-contiguous.
    b_k = I.constant(right.strides[0]) == 1
    m, n, k = map(I.constant, (*product.type.shape, a.type.shape[1]))
    if column and a_column and b_k:
      return True
    if column and a_row and b_k and type(m) is int and type(n) is int and n >= m:
      return True
    if (I.constant(destination.strides[1]) == 1 and a_column and b_k and
        all(type(v) is int for v in (m, n, k)) and m >= 4 and 2 * m >= n and k >= 16):
      return True
    return False

  def visit(block):
    result = []
    for statement in block.statements:
      for child in children(statement):
        visit(child)
      if (
        isinstance(statement, I.Store)
        and len(statement.destination.shape) == 2
      ):
        products = [e for e in walk(statement.value) if e.op == "dot"]
        if (
          products
          and all(e.data.tiled and e.data.swap for e in products)
          and should_swap(statement, products)
          and not any(
            e.op == "packed_load" or isinstance(e.type, I.TileType) and len(e.type.shape) not in (1, 2)
            for e in walk(statement.value)
          )
        ):
          destination = replace(
            statement.destination, shape=statement.destination.shape[::-1], strides=statement.destination.strides[::-1],
            region=replace(statement.destination.region, axes=statement.destination.region.axes[::-1])
                   if statement.destination.region is not None else None,
          )
          statement = replace(
            statement,
            destination=destination,
            value=I.transpose_matrix(statement.value),
            mask=I.transpose_matrix(statement.mask) if statement.mask is not None else None,
          )
      result.append(statement)
    block.statements = result

  visit(ir.body)
  return ir


@register_pass("materialize-dependent-dots", after=("fuse-dot-epilogue",), before=("orient-matmul",))
def materialize_dependent_dots(ir, context):
  """Keep one register-resident product per sink, including pack/reduce sinks.

  Product DAGs nested in a dot or packing operation need a concrete region.
  With sibling products, retain the last product's epilogue fusion and save
  only the other products. BindTile is provenance, not an eager computation.
  """
  serial = 0

  def visit(block):
    nonlocal serial
    output = []
    for statement in block.statements:
      for child in children(statement):
        visit(child)
      replacements = {}
      memo = {}

      def rewrite(value):
        if id(value) in replacements:
          return replacements[id(value)]
        if id(value) not in memo:
          args = tuple(rewrite(a) for a in value.args)
          memo[id(value)] = replace(value, args=args) if any(a is not b for a, b in zip(args, value.args)) else value
        return memo[id(value)]

      def materialize(value):
        nonlocal serial
        if id(value) not in replacements:
          serial += 1
          view = I.View(
            f"dsl_o_dot_result_{serial}",
            value.type.dtype,
            value.type.shape,
            (value.type.shape[1], I.literal(1)),
            I.literal(0),
            owned=True,
          )
          output.extend((I.Allocate(view), I.Store(view, rewrite(value))))
          replacements[id(value)] = I.Expr(value.type, "load", data=view)
          memo.clear()

      def nested(value, in_dot=False, force=False):
        for argument in value.args:
          nested(argument, in_dot or value.op == "dot", force)
        if value.op == "dot" and (in_dot or force):
          materialize(value)

      if isinstance(statement, (I.Store, I.DeclareState, I.UpdateState)):
        nested(statement.value)
        products = {id(e): e for e in walk(rewrite(statement.value)) if e.op == "dot"}
        for product in tuple(products.values())[:-1]:
          materialize(product)
        if replacements:
          statement = replace(statement, value=rewrite(statement.value))
      elif isinstance(statement, I.Pack):
        nested(statement.source, force=True)
        if replacements:
          statement = replace(statement, source=rewrite(statement.source))
      elif isinstance(statement, I.Assign):
        nested(statement.value, force=True)
        if replacements:
          statement = replace(statement, value=rewrite(statement.value))
      output.append(statement)
    block.statements = output

  visit(ir.body)
  return ir
