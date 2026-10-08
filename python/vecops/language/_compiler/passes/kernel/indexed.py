# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Prove dense SIMD induction accesses before memory/vector legalization.

Only clipped unit-step logical indices over the complete rank-one view are
recognized. Arbitrary indices, masks and offset ranges keep indexed checks;
physical stride remains a memory-planner concern (load/store or gather).
"""

from dataclasses import fields, replace

from ... import ir as I
from ...analysis.loops import trivial_true
from ...ir.traversal import children, nodes, walk
from ...operations import Trait
from ..manager import register_pass


@register_pass("normalize-simd-access", after=("expression-cse",), before=("normalize-iteration-regions",))
def normalize_simd_access(ir, context):
  def visit(block):
    for loop in block.statements:
      for child in children(loop):
        visit(child)
      if not isinstance(loop, I.For) or loop.kind not in {"simd", "parallel"}:
        continue
      definitions = {s.target: s.value for s in nodes(loop.body) if isinstance(s, I.Assign) and s.declare}
      changed = {s.target for s in nodes(loop.body) if isinstance(s, I.Assign) and not s.declare}
      memo = {}
      metadata = set()
      catalog = context.environment.operations
      for s in loop.body.statements:
        if (
          isinstance(s, I.Assign)
          and s.declare
          and s.value.type.dtype == "int64"
          and all(
            catalog.get(e).effects == "none"
            and not catalog.has(e, Trait.REDUCTION)
            and (not catalog.has(e, Trait.REFERENCE) or e.data == loop.target or e.data in metadata)
            for e in walk(s.value)
          )
        ):
          metadata.add(s.target)

      def expand(e, definitions=definitions, changed=changed):
        if e.op == "frozen" and e.data in definitions and e.data not in changed:
          return expand(definitions[e.data])
        return replace(e, args=tuple(expand(a) for a in e.args))

      def project(view, index, loop=loop, expand=expand):
        if len(view.shape) != 1 or not I.same_expr(view.shape[0], loop.stop) or I.constant(loop.start) != 0:
          return None
        if index.op != "arange" or len(index.type.shape) != 1:
          return None
        start, width = expand(index.args[0]), expand(index.type.shape[0])
        expected = I.binary(
          "min", loop.step, I.binary("sub", loop.stop, I.Expr(I.ScalarType("int64"), "local", data=loop.target))
        )
        if start.op != "local" or start.data != loop.target or not I.same_expr(width, expected):
          return None
        offset = I.binary("add", view.offset, I.binary("mul", start, view.strides[0]))
        root = view.region
        region = replace(root, starts=(start,), widths=(width,)) if root else None
        return replace(view, shape=(width,), offset=offset, region=region)

      def rewrite(value, memo=memo, metadata=metadata, definitions=definitions, project=project):
        if isinstance(value, I.Expr):
          if id(value) not in memo:
            if value.op == "frozen" and value.data in metadata:
              memo[id(value)] = rewrite(definitions[value.data])
              return memo[id(value)]
            result = replace(
              value, args=tuple(rewrite(a) for a in value.args), type=rewrite(value.type), data=rewrite(value.data)
            )
            if result.op == "gather" and len(result.args) == 3:
              view = project(result.data, result.args[0])
              if view is not None:
                mask, other = result.args[1:]
                memory_args = () if I.constant(mask) is True and I.constant(other) == 0 else (mask, other)
                result = replace(result, op="load", args=memory_args, data=view)
            memo[id(value)] = result
          return memo[id(value)]
        if isinstance(value, tuple):
          return tuple(rewrite(v) for v in value)
        if isinstance(value, I.Block):
          return I.Block([rewrite(v) for v in value.statements])
        if isinstance(value, I.IndexedStore) and value.unique:
          view = project(value.destination, value.indices)
          if view is not None:
            return I.Store(view, rewrite(value.value), rewrite(value.mask), vector_extent=value.indices.type.tag)
        if isinstance(value, I.Node):
          return replace(value, **{f.name: rewrite(getattr(value, f.name)) for f in fields(value)})
        return value

      loop.body.statements = [
        s
        for s in rewrite(loop.body).statements
        if not isinstance(s, I.Assert) or not trivial_true(expand(s.condition))
        if not isinstance(s, I.Assign) or s.target not in metadata
      ]

  visit(ir.body)
  return ir
