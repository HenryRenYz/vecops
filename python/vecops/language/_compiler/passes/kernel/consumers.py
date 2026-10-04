"""Pointwise consumers of loop-produced private storage fuse at the write site.

Prefix recurrences retain their order, including exclusive scans. External
aliasing selects the original materialized program. This is a loop/storage
transform, not a separate implementation for every scan combiner.
"""

from dataclasses import replace

from ... import ir as I
from ...analysis.loops import trivial_true
from ...ir.traversal import children, nodes, walk
from ...operations import Trait
from ..manager import register_pass


@register_pass("fuse-loop-consumers", after=("fuse-loop-regions",), before=("unroll-serial",))
def fuse_loop_consumers(ir, context):
  if not context.options.get("loop_fusion", True):
    return ir
  catalog = context.environment.operations

  def visit(block):
    for statement in block.statements:
      for child in children(statement):
        visit(child)
    index = 1
    while index < len(block.statements):
      producer, consumer = block.statements[index - 1 : index + 1]
      if not isinstance(producer, I.For) or not isinstance(consumer, I.Store) or not consumer.destination.shape:
        index += 1
        continue
      reads = {id(e): e for e in walk(consumer.value) if e.op == "load" and e.data.owned}
      if len(reads) != 1 or any(
        not isinstance(e.type, I.ScalarType) and e.op != "load" and not catalog.has(e, Trait.ELEMENTWISE)
        for e in walk(consumer.value)
      ):
        index += 1
        continue
      load = next(iter(reads.values()))
      view = load.data
      if I.constant(view.offset) != 0:
        index += 1
        continue
      producer_nodes = tuple(nodes(producer.body))
      changed = {s.target for s in producer_nodes if isinstance(s, I.Assign) and not s.declare}
      if (
        any(
          isinstance(
            s,
            (
              I.Control,
              I.If,
              I.While,
              I.UpdateState,
              I.DeclareState,
              I.IndexedStore,
              I.HardwareRegion,
              I.ArchitectureCall,
            ),
          )
          or isinstance(s, I.Store)
          and s.destination.storage != view.storage
          or isinstance(s, I.Assert)
          and not trivial_true(s.condition)
          or type(s).__module__.endswith(("collective", "pipeline"))
          for s in producer_nodes
        )
        or any(catalog.has(e, Trait.REFERENCE) and e.data in changed for e in walk(consumer.value))
        or consumer.mask is not None
      ):
        index += 1
        continue
      writes = [s for s in nodes(producer.body) if isinstance(s, I.Store) and s.destination.storage == view.storage]
      if (
        len(writes) != 1
        or writes[0].destination.shape
        or len(view.shape) != len(consumer.destination.shape)
        or not all(I.same_expr(a, b) for a, b in zip(view.shape, consumer.destination.shape))
      ):
        index += 1
        continue
      later = I.Block(block.statements[index + 1 :])
      if any(isinstance(e, I.View) and e.storage == view.storage for e in nodes(later)):
        index += 1
        continue
      site = writes[0]
      coords = tuple(
        I.binary("mod", I.binary("floordiv", site.destination.offset, stride), extent)
        for stride, extent in zip(view.strides, view.shape)
      )

      def scalarize(e, view=view, site=site, coords=coords):
        if isinstance(e.type, I.ScalarType):
          return e
        if e.op == "load":
          if e.data.storage == view.storage:
            return I.Expr(I.ScalarType(view.dtype), "cast", (site.value,))
          offset = e.data.offset
          own = coords[len(coords) - len(e.data.shape) :]
          for coord, extent, stride in zip(own, e.data.shape, e.data.strides):
            coord = (
              I.literal(0)
              if I.constant(extent) == 1
              else coord
              if I.constant(extent) is not None
              else I.Expr(I.ScalarType("int64"), "where", (I.binary("eq", extent, I.literal(1)), I.literal(0), coord))
            )
            offset = I.binary("add", offset, I.binary("mul", coord, stride))
          return I.Expr(
            I.ScalarType(e.type.dtype), "load", e.args, replace(e.data, shape=(), strides=(), offset=offset)
          )
        return replace(
          e, type=I.ScalarType("bool" if e.type.mask else e.type.dtype), args=tuple(scalarize(a) for a in e.args)
        )

      destination = consumer.destination
      offset = destination.offset
      for coord, stride in zip(coords, destination.strides):
        offset = I.binary("add", offset, I.binary("mul", coord, stride))
      sink = replace(
        consumer,
        destination=replace(destination, shape=(), strides=(), offset=offset),
        value=scalarize(consumer.value),
        mask=scalarize(consumer.mask) if consumer.mask is not None else None,
      )

      def rewrite(body, site=site, sink=sink):
        result = []
        for s in body.statements:
          if s is site:
            result.append(sink)
          elif isinstance(s, I.For):
            result.append(replace(s, body=rewrite(s.body)))
          else:
            result.append(s)
        return I.Block(result)

      fused = replace(producer, body=rewrite(producer.body))
      guard = I.literal(True)
      seen = set()
      induction_names = {producer.target, *(s.target for s in producer_nodes if isinstance(s, I.For))}
      for e in nodes((producer.body, consumer.value, consumer.mask)):
        if not isinstance(e, I.Expr) or e.op != "load" or e.data.owned or e.data.storage in seen:
          continue
        seen.add(e.data.storage)
        # Fold/scan input expressions retain whole-view provenance.
        source = e.data
        if source.region is not None:
          source = replace(
            source, shape=source.region.root_shape, strides=source.region.root_strides, offset=I.literal(0)
          )
        if any(
          v.op == "local" and v.data in induction_names
          for n in (source.offset, *source.shape, *source.strides)
          for v in walk(n)
        ):
          guard = None
          break
        guard = I.binary("and", guard, I.Expr(I.ScalarType("bool"), "view_disjoint", data=(source, destination)))
      if guard is None:
        index += 1
        continue
      allocation_index = next(
        (
          n
          for n, s in enumerate(block.statements[: index - 1])
          if isinstance(s, I.Allocate) and s.view.storage == view.storage
        ),
        None,
      )
      fallback = [producer, consumer]
      if allocation_index is None:
        index += 1
        continue
      if any(
        isinstance(e, I.View) and e.storage == view.storage
        for e in nodes(I.Block(block.statements[allocation_index + 1 : index - 1]))
      ):
        # Removing storage is legal only when no earlier phase observes it.
        index += 1
        continue
      allocated = block.statements[allocation_index].view
      if len(allocated.shape) != len(view.shape) or not all(
        I.same_expr(a, b) for a, b in zip((*allocated.shape, *allocated.strides), (*view.shape, *view.strides))
      ):
        index += 1
        continue
      fallback.insert(0, block.statements.pop(allocation_index))
      index -= 1
      replacement = fused if I.constant(guard) is True else I.If(guard, I.Block([fused]), I.Block(fallback))
      block.statements[index - 1 : index + 1] = [replacement]
      ir.diagnostics.append(
        "loop consumer fusion: prefix/pointwise state retained; private intermediate removed on non-alias path"
      )

  visit(ir.body)
  return ir
