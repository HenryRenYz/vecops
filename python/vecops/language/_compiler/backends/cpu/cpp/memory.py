# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""C++ emission: memory."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
from functools import wraps

from .... import ir as I
from ....analysis.facts import abi_metadata_constants
from ....dtypes import ELEMENT_BYTES
from ....ir.traversal import walk_unique
from ....operations import Trait
from ..broadcasts import BroadcastProof, ScalarGeometry, broadcast_key
from ..capabilities import CPP_TYPES


def scoped_scalar_geometry(emit):
  """Bind prepared geometry to exactly one store/group emission invocation."""

  @wraps(emit)
  def scoped(self, subject, *args, **kwargs):
    with self.scalar_geometry(subject):
      return emit(self, subject, *args, **kwargs)

  return scoped


class MemoryEmitter:
  @scoped_scalar_geometry
  def single_snapshot_store(self, statement, future, emit):
    """Keep a single fresh owner entirely off the nonalias execution path.

    Compute the original guard once, at its original position. Restrict this
    to one ordinary read and no prior capture/materialization: multiple span
    checks may throw, and must not move across another snapshot allocation.
    """
    from ..storage import SnapshotScope

    storage = getattr(self.prepared, "storage", None)
    proof = getattr(storage, "snapshot_scopes", {}).get(id(statement))
    if (
      type(proof) is not SnapshotScope
      or proof.subject is not statement
      or type(statement) is not I.Store
      or len(proof.reads) != 1
      or self.live_loads(future)
      or self.snapshots
      or self.materialized
      or self.reduction_materialized
    ):
      return False
    policy = self.vector_ir.manager.options.get("store_snapshot_regions", True)
    if type(policy) is not bool:
      raise TypeError("store_snapshot_regions must be bool")
    load = proof.reads[0]
    if not policy or not self.snapshot_required(load, statement.destination):
      return False
    condition = self.snapshot_condition(load, statement.destination, False)
    snapshots, materialized = self.snapshots, self.materialized
    reductions, cache = self.reduction_materialized, self.chunk_cache
    provenance = getattr(self, "store_snapshot_provenance", None)
    try:
      self.opening(f"if (!({condition})) /* fresh store snapshots absent */")
      self.snapshots, self.materialized = dict(snapshots), dict(materialized)
      self.reduction_materialized, self.chunk_cache = dict(reductions), {}
      self.store_snapshot_provenance = {key: value for key, value in (provenance or {}).items() if key != id(statement)}
      try:
        emit()
      finally:
        self.close()
      self.opening("else /* retain current and historical store snapshots */")
      self.snapshots, self.materialized = dict(snapshots), dict(materialized)
      self.reduction_materialized, self.chunk_cache = dict(reductions), {}
      self.store_snapshot_provenance = {key: value for key, value in (provenance or {}).items() if key != id(statement)}
      try:
        self.snapshot_contents(load, condition)
        emit()
      finally:
        self.close()
    finally:
      self.snapshots, self.materialized = snapshots, materialized
      self.reduction_materialized, self.chunk_cache = reductions, cache
      if provenance is None:
        del self.store_snapshot_provenance
      else:
        self.store_snapshot_provenance = provenance
    return True

  @contextmanager
  def snapshot_lifetime(self, subject, future):
    """Keep only proved consumer-local implicit owners inside this block.

    Protecting an active future read can create/rebind a buffer for a later
    consumer. Such emission retains its existing outer scope even when this
    consumer's own reads are local. Historical outer bindings are never erased.
    """
    from ..storage import SnapshotScope

    storage = getattr(self.prepared, "storage", None)
    proof = getattr(storage, "snapshot_scopes", {}).get(id(subject))
    if type(proof) is not SnapshotScope or proof.subject is not subject or self.live_loads(future):
      yield
      return
    snapshots, active = self.snapshots, self.active_loads
    materialized, reductions, cache = self.materialized, self.reduction_materialized, self.chunk_cache
    provenance = getattr(self, "store_snapshot_provenance", None)
    self.snapshots, self.active_loads = dict(snapshots), dict(active)
    self.materialized, self.reduction_materialized, self.chunk_cache = dict(materialized), dict(reductions), {}
    self.store_snapshot_provenance = dict(provenance or {})
    self.opening("/* consumer-local snapshot lifetime */")
    try:
      yield
    finally:
      self.close()
      self.snapshots, self.active_loads = snapshots, active
      self.materialized, self.reduction_materialized, self.chunk_cache = materialized, reductions, cache
      if provenance is None:
        del self.store_snapshot_provenance
      else:
        self.store_snapshot_provenance = provenance

  @contextmanager
  def scalar_geometry(self, subject):
    """Consume equality classes without rewriting retained Tile/View objects.

    A nested fallback store replaces the group's context rather than inheriting
    unrelated equalities. Branch-local vectors cannot escape the context.
    """
    previous = getattr(self, "active_scalar_classes", {})
    previous_aliases = getattr(self, "active_scalar_aliases", {})
    cache = self.chunk_cache
    plan = getattr(self.prepared, "scalar_geometry", {}).get(id(subject))
    classes, aliases = {}, {}
    if type(plan) is ScalarGeometry and plan.subject is subject:
      for component, representative in zip(plan.classes, plan.representatives):
        for expression in component:
          classes[id(expression)] = component
          if representative is not None and expression is not representative:
            aliases[id(expression)] = (expression, representative)
    self.active_scalar_classes = classes
    self.active_scalar_aliases = aliases
    self.chunk_cache = {}
    try:
      yield
    finally:
      self.active_scalar_classes = previous
      self.active_scalar_aliases = previous_aliases
      self.chunk_cache = cache

  def same_geometry(self, first, second):
    """Query only unconditional identity/literals or the current certificate."""
    if I.same_extent(first, second):
      return True
    component = getattr(self, "active_scalar_classes", {}).get(id(first), ())
    return any(expression is first for expression in component) and any(
      expression is second for expression in component
    )

  def live_loads(self, required):
    """Visit retained reads in capture order, independent of object addresses.

    Python ids are membership keys only. Iterating their set changed alias
    preparation order between equivalent lowerings and changed native code.
    Keep the scheduling deterministic when tuning it. Snapshotting never writes caller memory;
    an immutable key tuple preserves the original independent-read semantics.
    """
    required = set(required)
    return tuple(identity for identity in self.active_loads if identity in required)

  def conversion_layout_option(self):
    """A private paired load/store layout exists only in a proven full body."""
    return ", vecops::vec::cvt::unordered" if getattr(self, "pointwise_register_layout", None) else ""

  def storage_pointer(self, storage):
    """Use an entry pointer for ABI tensors; resolve resizable scratch live."""
    return self.parameter_pointers.get(storage, f"{storage}.data()")

  def pointer(self, view, coordinates=()):
    base = self.storage_pointer(view.storage)
    offset = self.scalar(view.offset)
    terms = [offset]
    for coordinate, stride in zip(coordinates, view.strides):
      terms.append(f"({coordinate}) * ({self.scalar(stride)})")
    return f"({base} + ({' + '.join(terms)}))"

  def coordinates(self, shape, root_shape, coords):
    offset = len(root_shape) - len(shape)
    if offset < 0:
      raise TypeError("a higher-rank tile cannot broadcast to a lower-rank consumer")
    result = []
    for axis, extent in enumerate(shape):
      coordinate = coords[offset + axis]
      if I.constant(extent) == 1:
        coordinate = "0"
      elif not self.same_geometry(extent, root_shape[offset + axis]) and I.constant(extent) is None:
        coordinate = f"(({self.scalar(extent)}) == 1 ? 0 : ({coordinate}))"
      result.append(coordinate)
    return tuple(result)

  def full_fixed_access(self, tag, active):
    """A fixed piece is full only when its exact logical count is known.

    Do not propagate a blanket full-vector flag into scalar reductions or
    scalable/tail consumers nested in the same expression.
    """
    return tag in self.fixed_tag_lanes and str(self.fixed_tag_lanes[tag]) == str(active)

  def memory_options(self, mask, tag, root_shape, coords, active, axis):
    if mask is None:
      return (
        "vecops::vec::opt::unmasked"
        if getattr(self, "full_vector", False) or self.full_fixed_access(tag, active)
        else f"vecops::vec::opt::first({active})"
      ), None
    predicate = self.vector(mask, tag, root_shape, coords, active, axis)
    name = self.unique("memory_mask")
    self.line(
      f"auto {name} = vecops::vec::mask_and({tag}{{}}, vecops::vec::mwhilelt({tag}{{}}, 0, {active}), {predicate});"
    )
    return f"vecops::vec::opt::masked({name})", name

  @scoped_scalar_geometry
  def store(self, statement, *, plan=None):
    destination, value = statement.destination, statement.value
    if not destination.shape:
      if statement.mask is not None:
        self.opening(f"if ({self.scalar(statement.mask)})")
      self.line(
        f"*{self.pointer(destination)} = {self.scalar_convert(destination.dtype, self.scalar(value), value.type.dtype)};"
      )
      if statement.mask is not None:
        self.close()
      return
    self.check_store(statement)
    if self.register_store(statement):
      return
    if self.transpose_store(statement, self.prepared.transposes.get(id(statement))):
      return
    self.streaming_store(statement, plan=plan)

  def streaming_store(self, statement, *, plan=None):
    plan = plan or self.vector_ir.plans.get(id(statement))
    if plan is None:
      plan = self.prepared.store(statement)
      self.emitted_regions.append(plan)
    elif id(statement) not in self.vector_ir.plans:
      self.emitted_regions.append(plan)
    fresh = self.fresh_store_snapshots(statement, plan)

    def emit():
      self.traverse_region(
        plan, lambda tags, coords, active, roots, state: self.write_effects(plan, tags, coords, active, roots)
      )

    if not fresh:
      emit()
      return
    # Only this store's newly created conditional captures are false under
    # this guard. An older capture may hold the value from before another
    # write, even when the *current* source and destination are disjoint.
    # Never substitute its original memory or clear unrelated mappings.
    snapshots = self.snapshots
    materialized = dict(self.materialized)
    reduction_materialized = dict(self.reduction_materialized)
    cache = self.chunk_cache
    guards = dict.fromkeys(entry[1] for entry in fresh.values())
    self.opening("if (!(" + " || ".join(guards) + ")) /* fresh store snapshots absent */")
    try:
      self.snapshots = {identity: entry for identity, entry in snapshots.items() if identity not in fresh}
      self.materialized = dict(materialized)
      self.reduction_materialized = dict(reduction_materialized)
      self.chunk_cache = {}
      emit()
      self.close()
      self.opening("else /* retain current and historical store snapshots */")
      self.snapshots = dict(snapshots)
      self.materialized = dict(materialized)
      self.reduction_materialized = dict(reduction_materialized)
      self.chunk_cache = {}
      emit()
      self.close()
    finally:
      # Branch-local proxy expressions and C++ locals cannot escape their
      # scopes. Captures/materializations from outside remain available to
      # every later store, including nested allocation and hardware regions.
      self.snapshots = snapshots
      self.materialized = materialized
      self.reduction_materialized = reduction_materialized
      self.chunk_cache = cache

  def fresh_store_snapshots(self, statement, plan):
    """Return only guarded captures first created for this exact store.

    Version only row-independent elementwise memory DAGs. Reduction/state,
    register transforms and opaque operations keep their existing lowering;
    their branch-local state or materialization lifetimes need a separate
    legality proof. The operation catalog, not a target-specific opcode
    whitelist, supplies the semantic boundary.
    """
    policy = self.vector_ir.manager.options.get("store_snapshot_regions", True)
    if not isinstance(policy, bool):
      raise TypeError("store_snapshot_regions must be bool")
    if not policy:
      return {}
    captured = getattr(self, "store_snapshot_provenance", {}).get(id(statement))
    if captured is None or captured[0] is not statement:
      return {}
    boundaries = Trait.REDUCTION | Trait.REGISTER_TRANSFORM | Trait.VECTOR_BOUNDARY | Trait.STATE_REFERENCE
    catalog = self.environment.operations
    loads = set()
    for root in plan.roots:
      for expression in walk_unique(root):
        if (
          catalog.get(expression).effects not in {"none", "read"}
          or catalog.has(expression, boundaries)
          or isinstance(expression.type, I.TileType)
          and expression.op != "load"
          and not catalog.has(expression, Trait.ELEMENTWISE)
        ):
          return {}
        if expression.op == "load":
          loads.add(id(expression))
    return {
      identity: entry
      for identity, entry in captured[1].items()
      if identity in loads and self.snapshots.get(identity) is entry
    }

  def check_store(self, statement):
    destination, value = statement.destination, statement.value
    shape = ", ".join(self.scalar(v) for v in destination.shape)
    strides = ", ".join(self.scalar(v) for v in destination.strides)
    transpose = self.prepared.transposes.get(id(statement))
    expected, dense = I.literal(1), True
    for extent, stride in zip(reversed(destination.shape), reversed(destination.strides)):
      dense = dense and I.same_expr(stride, expected)
      expected = I.binary("mul", expected, extent)
    if dense and statement.vector_extent is not None and len(destination.shape) == 1:
      # A clipped SIMD induction is nonnegative and bounded by one virtual
      # vector. Rechecking its rank-one element product in the hot loop can
      # outline checked_shape and force every live Vec to spill across calls.
      self.line("/* clipped SIMD extent: dense writable size already proven */")
    elif dense or transpose is not None and transpose.dense_destination:
      self.line(f"{self.checked_shape(shape)}; /* proven dense writable layout */")
    else:
      self.line(f"vecops_dsl::check_writable<{len(destination.shape)}>({{{shape}}}, {{{strides}}});")
    for operand in (value, statement.mask):
      if operand is None or not isinstance(operand.type, I.TileType):
        continue
      if len(operand.type.shape) > len(destination.shape):
        raise TypeError("store source cannot have a higher rank than destination")
      for source_extent, destination_extent in zip(reversed(operand.type.shape), reversed(destination.shape)):
        if I.constant(source_extent) == 1 or I.same_extent(source_extent, destination_extent):
          continue
        proof = getattr(self.prepared, "broadcast_proofs", {}).get(
          broadcast_key(statement, source_extent, destination_extent)
        )
        if type(proof) is BroadcastProof and (
          proof.store is statement and proof.source is source_extent and proof.destination is destination_extent
        ):
          continue
        if source_extent.op == destination_extent.op == "literal":
          raise TypeError("store source does not broadcast to the destination shape")
        self.line(f"vecops_dsl::check_broadcast({self.scalar(source_extent)}, {self.scalar(destination_extent)});")

  def write_effects(self, region, tags, coordinates, active, roots):
    effects = [effect for effect in region.effects if effect.kind == "store_convert"]
    stores = tuple(effect.store for effect in effects)
    values = [
      value
      for effect in effects
      for value in (
        (roots[effect.value_root], roots[effect.mask_root]) if effect.mask_root >= 0 else (roots[effect.value_root],)
      )
    ]
    self.write_roots(stores, tags, coordinates, active, values)

  def write_roots(self, stores, tags, coordinates, active, roots):
    index = 0
    for statement in stores:
      values = roots[index]
      index += 1
      masks = roots[index] if statement.mask is not None else None
      if masks is not None:
        index += 1
      destination = statement.destination
      value = statement.value
      dtype = (
        value.args[0] if value.op == "cast" and value.type.dtype == statement.destination.dtype else value
      ).type.dtype
      for slot, (base, coords, count) in enumerate(zip(tags, coordinates, active)):
        tag = self.rebound_tag(dtype, base)
        self.opening(f"if ({count} > 0)")
        options = (
          "vecops::vec::opt::unmasked"
          if getattr(self, "full_vector", False) or self.full_fixed_access(tag, count)
          else f"vecops::vec::opt::first({count})"
        )
        options += self.conversion_layout_option()
        if masks is not None:
          mask = (
            I.Expr(statement.mask.type, "vector_ref", data=masks[slot])
            if isinstance(statement.mask.type, I.TileType)
            else statement.mask
          )
          options, _ = self.memory_options(mask, tag, destination.shape, coords, count, len(destination.shape) - 1)
        pointer = self.pointer(destination, coords)
        if I.constant(destination.strides[-1]) == 1:
          self.line(f"vecops::vec::store_convert({tag}{{}}, {pointer}, {values[slot]}, {options});")
        else:
          stride = self.scalar(destination.strides[-1])
          self.opening(f"if (({stride}) == 1)")
          self.line(f"vecops::vec::store_convert({tag}{{}}, {pointer}, {values[slot]}, {options});")
          self.close()
          self.opening("else")
          self.line(
            f"vecops::vec::store_convert({tag}{{}}, {pointer}, {values[slot]}, {options}, vecops::vec::opt::strided({stride}));"
          )
          self.close()
        self.close()

  def expression_loads(self, block):
    result = {}
    for statement in block.statements:
      if isinstance(statement, I.ReductionGroup):
        result.update(self.expression_loads(I.Block([*statement.assignments, *statement.stores])))
      if isinstance(statement, I.ElementwiseGroup):
        result.update(self.expression_loads(I.Block(list(statement.stores))))
      if isinstance(statement, I.UnrolledFor):
        result.update(self.expression_loads(statement.copies))
        result.update(self.expression_loads(statement.loop.body))
      for value in (
        getattr(statement, "value", None),
        getattr(statement, "condition", None),
        getattr(statement, "mask", None),
        getattr(statement, "indices", None),
      ):
        if isinstance(value, I.Expr):
          for load in I.loads_unique(value):
            result[id(load)] = load
      for name in ("body", "otherwise"):
        child = getattr(statement, name, None)
        if isinstance(child, I.Block):
          result.update(self.expression_loads(child))
    return result

  def overlap(self, source, destination):
    if self.disjoint_proven(source, destination):
      return "false"

    def span(view):
      shape = ", ".join(self.scalar(v) for v in view.shape)
      strides = ", ".join(self.scalar(v) for v in view.strides)
      return f"vecops_dsl::span({self.pointer(view)}, {{{shape}}}, {{{strides}}}, sizeof({CPP_TYPES[view.dtype]}))"

    expression = f"vecops_dsl::overlap({span(source)}, {span(destination)})"
    return self.invariant_alias_predicate(source, destination, expression)

  def invariant_alias_predicate(self, source, destination, expression):
    """Remember invariant checks lazily, retaining first-use trap ordering.

    Only the predicate is reused: historical contents and scratch lifetimes
    still follow the ordinary snapshot plan. Workers never share this mutable
    state. Bound the number of invocation-local scalar pairs to avoid source
    and register growth for kernels with many distinct constant subviews.
    """
    if (not getattr(self, "serial_guard_depth", 0) or self.parallel_depth or self.hardware_scope is not None
        or self.hardware_scalar_bindings
        or self.environment.codegen_target.architecture != "x86"
        or getattr(self, "active_scalar_aliases", {})):
      return expression
    plan = getattr(self.prepared, "entry_geometry", None)
    if (plan is None or not plan.matches(self.ir, self.environment.operations)
        or not plan.view(source) or not plan.view(destination)):
      return expression
    name = self.invariant_alias_predicates.get(expression)
    if name is None:
      if len(self.invariant_alias_predicates) >= 16:
        return expression
      name = self.unique("invariant_alias")
      self.invariant_alias_predicates[expression] = name
    return f"([&]() VECOPS_INLINE_LAMBDA {{ if (!{name}_ready) {{ {name}_value = {expression}; {name}_ready = true; }} return {name}_value; }}())"

  def register_disjoint_proven(self, a, b, *, group=None):
    if group is not None:
      from ..partitions import partition_key

      proof = getattr(self.prepared, "partition_proofs", {}).get(partition_key(group, a, b))
      if (
        proof is not None
        and proof.group is group
        and (proof.first is a and proof.second is b or proof.first is b and proof.second is a)
      ):
        return True
    # Private scratch cannot alias a different live storage. This is the same
    # ownership fact used by snapshot(); do not add runtime span arithmetic
    # for independently allocated permutation inputs and outputs.
    if a.storage != b.storage and (a.owned or b.owned):
      return True
    if a.storage == b.storage and a.owned and b.owned and a.region is not None and b.region is not None:
      root = self.storage_views.get(a.storage)
      if root is not None and root.owned and root.dtype == a.dtype == b.dtype:
        expected, dense = I.literal(1), True
        for extent, stride in zip(reversed(root.shape), reversed(root.strides)):
          dense = dense and I.same_expr(stride, expected)
          expected = I.binary("mul", expected, extent)
        regions = (a.region, b.region)
        same_root = all(
          len(region.root_shape) == len(root.shape)
          and all(
            I.same_expr(x, y) for x, y in zip((*region.root_shape, *region.root_strides), (*root.shape, *root.strides))
          )
          for region in regions
        )
        if (
          dense
          and same_root
          and all(region.affine_partition and all(step == 1 for step in region.steps) for region in regions)
        ):
          for start_a, start_b, width_a, width_b, clipped_a, clipped_b in zip(
            a.region.exact_starts or a.region.starts,
            b.region.exact_starts or b.region.starts,
            a.region.widths,
            b.region.widths,
            a.region.clipped,
            b.region.clipped,
          ):
            if clipped_a or clipped_b:
              continue
            difference = I.constant(I.binary("sub", start_b, start_a))
            wa, wb = I.constant(width_a), I.constant(width_b)
            if (
              all(type(v) is int for v in (difference, wa, wb))
              and wa >= 0
              and wb >= 0
              and (difference >= wa or -difference >= wb)
            ):
              return True
    return self.disjoint_proven(a, b)

  def disjoint_proven(self, a, b):
    proof = getattr(self, "disjoint_roots", {}).get(tuple(sorted((a.storage, b.storage))))
    if proof is None:
      return False

    definitions = getattr(getattr(self, "prepared", None), "immutable_geometry", {})
    kernel = getattr(self, "ir", None)
    constants = getattr(self, "abi_metadata_constants", None)
    if constants is None:
      constants = abi_metadata_constants(kernel)
    bindings = getattr(kernel, "bindings", {})
    raw = {v.data for v in bindings.values() if v.op == "raw"}
    for name, parameter in getattr(kernel, "parameters", {}).items():
      if hasattr(parameter, "shape"):
        raw.update(f"{name}.{kind}({axis})" for kind in ("size", "stride") for axis in range(len(parameter.shape)))
      else:
        raw.add(name)
    catalog = getattr(getattr(self, "environment", None), "operations", None)
    memo = {}

    def stable(value):
      if value in memo:
        return memo[value]
      memo[value] = None
      if value.op == "literal":
        result = value
      elif value.op == "raw" and value.data in raw:
        result = (
          I.literal(constants[value.data])
          if not value.args and type(value.type) is I.ScalarType
          and value.type.dtype == "int64" and value.data in constants
          else value
        )
      elif value.op == "symbol" and value.data in bindings:
        result = stable(bindings[value.data])
      elif value.op in {"local", "frozen"} and value.data in definitions:
        result = stable(definitions[value.data])
      elif (
        value.args
        and isinstance(value.type, I.ScalarType)
        and catalog is not None
        and catalog.get(value).effects == "none"
        and not catalog.has(value, Trait.REFERENCE | Trait.MEMORY_READ | Trait.STATE_REFERENCE | Trait.CAPTURE_SCALAR)
      ):
        args = tuple(stable(v) for v in value.args)
        result = replace(value, args=args) if all(v is not None for v in args) else None
      else:
        result = None
      memo[value] = result
      return result

    def same(a, b):
      a, b = stable(a), stable(b)
      return a is not None and b is not None and I.same_expr(a, b)

    def same_layout(shape, strides, root):
      return len(shape) == len(root.shape) and all(
        same(x, y) for x, y in zip((*shape, *strides), (*root.shape, *root.strides))
      )

    def covered(own, root):
      width = ELEMENT_BYTES.get(own.dtype)
      if own.storage != root.storage or width is None or width != ELEMENT_BYTES.get(root.dtype):
        return False
      if same_layout(own.shape, own.strides, root) and same(own.offset, root.offset):
        return True
      region = own.region
      if region is None:
        return False
      source = region.span_source
      if (
        source is not None
        and source.region is None
        and source.storage == own.storage
        and source.dtype == own.dtype
        and covered(source, root)
      ):
        return True
      if not same_layout(region.root_shape, region.root_strides, root):
        return False
      # A load may capture its offset while logical coordinates remain lazy.
      # Reconstructing the origin by subtracting those coordinates is unsound.
      return region.root_offset is not None and same(region.root_offset, root.offset)

    # Both roles matter even when two guarded windows share one storage name.
    return (covered(a, proof[0]) and covered(b, proof[1])) or (covered(a, proof[1]) and covered(b, proof[0]))

  def same_mapping(self, a, b):
    if len(a.shape) != len(b.shape) or a.dtype != b.dtype:
      return "false"
    terms = [f"static_cast<const void*>({self.pointer(a)}) == static_cast<const void*>({self.pointer(b)})"]
    terms += [f"({self.scalar(x)}) == ({self.scalar(y)})" for x, y in zip(a.shape, b.shape)]
    terms += [f"({self.scalar(x)}) == ({self.scalar(y)})" for x, y in zip(a.strides, b.strides)]
    return "(" + " && ".join(terms) + ")"

  def snapshot_required(self, load, destination):
    """The existing static exclusions, before any runtime guard is printed."""
    view = load.data
    if self.disjoint_proven(view, destination):
      return False
    if self.verified_grid is not None and view.storage != destination.storage:
      # The enclosing runtime guard proved disjoint parameter root spans.
      # Worker-private allocations cannot alias a parameter or another slot.
      known = self.verified_grid.reads | self.verified_grid.writes | self.verified_grid.private
      if view.storage in known and destination.storage in known:
        return False
    return not (not view.shape or (view.owned or destination.owned) and view.storage != destination.storage)

  def snapshot_condition(self, load, destination, must_preserve):
    """Print one original guard without constructing its conditional owner."""
    view = load.data
    condition = self.unique("alias")
    overlap = self.overlap(view, destination)
    if not must_preserve:
      overlap += f" && !{self.same_mapping(view, destination)}"
    old = self.snapshots.get(id(load))
    if old is not None:
      overlap = f"({old[1]}) || ({overlap})"
    nonempty = " && ".join(f"({self.scalar(n)}) > 0" for n in view.shape + destination.shape)
    self.line(f"const bool {condition} = ({nonempty}) && ({overlap});")
    return condition

  def snapshot_contents(self, load, condition):
    """Preserve an existing capture before allocating another private owner.

    A true historical guard already owns this read's values. With the same
    layout, its serial consumer can retain that owner and copy only when the
    earlier guard was false. The alias checks stay at their original sites.
    Worker regions require their own owners, including when an enclosing
    serial region has a currently empty snapshot buffer.
    """
    old = self.snapshots.get(id(load))
    layout = self.prepared.snapshots[id(load)].effects[0].store.destination
    reuse = self.reusable_snapshot_owner(old, layout)
    name = old[0].storage if reuse else self.unique("snapshot")
    dtype = load.type.dtype
    if not reuse:
      self.line(f"vecops_dsl::Buffer<{CPP_TYPES[dtype]}> {name};")
    shape = load.type.shape
    statement, plan = self.prepared.snapshot(load, name)
    saved = statement.destination
    copy_condition = f"({condition}) && !({old[1]})" if reuse else condition
    self.opening(f"if ({copy_condition})")
    count = ", ".join(self.scalar(v) for v in shape)
    self.line(f"{name}.resize({self.checked_shape(count)});")
    self.store(statement, plan=plan)
    self.close()
    self.snapshots[id(load)] = (saved, condition)

  def reusable_snapshot_owner(self, old, layout):
    """Apply the measured x86 policy only to identical private layouts."""
    if old is None or self.parallel_depth or self.environment.codegen_target.architecture != "x86":
      return False
    saved = old[0]
    return (
      saved.owned and layout.owned and saved.dtype == layout.dtype
      and len(saved.shape) == len(layout.shape) and len(saved.strides) == len(layout.strides)
      and all(I.same_expr(a, b) for a, b in zip(
        (*saved.shape, *saved.strides, saved.offset), (*layout.shape, *layout.strides, layout.offset)
      ))
    )

  def snapshot(self, load, destination, must_preserve):
    if self.snapshot_required(load, destination):
      condition = self.snapshot_condition(load, destination, must_preserve)
      self.snapshot_contents(load, condition)

  @scoped_scalar_geometry
  def protect_store(self, statement, future):
    current = {id(load): load for load in I.loads_unique(statement.value)}
    if statement.mask is not None:
      current.update({id(load): load for load in I.loads_unique(statement.mask)})
    preserved = self.live_loads(future)
    preserved_ids = set(preserved)
    # Matrix inputs are re-read across output tiles. Even identical mappings
    # are not an elementwise-safe in-place update (C=A/B needs a snapshot).
    matrix_inputs = {
      id(load)
      for value in walk_unique(statement.value)
      if value.op == "dot"
      for operand in value.args[:2]
      for load in I.loads_unique(operand)
    }
    transformed_inputs = {
      id(load)
      for value in walk_unique(statement.value)
      if value.op == "value_transform" or self.environment.operations.has(value, Trait.REGISTER_TRANSFORM)
      for load in I.loads_unique(value)
    }
    for identity in preserved:
      current[identity] = self.active_loads[identity]
    fresh = {}
    for identity, load in current.items():
      historical = identity in self.snapshots
      self.snapshot(
        load,
        statement.destination,
        identity in preserved_ids or identity in matrix_inputs or identity in transformed_inputs or load.op == "gather",
      )
      if not historical and identity in self.snapshots:
        fresh[identity] = self.snapshots[identity]
    # Retain the statement, not just its id: synthesized snapshot/reduction
    # statements can otherwise be collected and their Python ids reused.
    if not hasattr(self, "store_snapshot_provenance"):
      self.store_snapshot_provenance = {}
    self.store_snapshot_provenance[id(statement)] = (statement, fresh)

  def stores(self, block):
    for statement in block.statements:
      if isinstance(statement, (I.Store, I.IndexedStore)):
        yield statement
      if isinstance(statement, (I.ReductionGroup, I.ElementwiseGroup)):
        yield from statement.stores
      if isinstance(statement, I.UnrolledFor):
        yield from self.stores(statement.loop.body)
      for name in ("body", "otherwise"):
        child = getattr(statement, name, None)
        if isinstance(child, I.Block):
          yield from self.stores(child)

  def protect_region(self, region, future):
    dependencies = self.expression_loads(region)
    dependencies.update(future)
    destinations = {s.destination.storage for s in self.stores(region)}
    for identity in self.live_loads(dependencies):
      for destination in sorted(destinations):
        root = self.storage_views.get(destination)
        if root is not None:
          self.snapshot(self.active_loads[identity], root, True)

  def used_storage(self, block):
    result = set()
    for statement in block.statements:
      if isinstance(statement, I.Pack):
        result.add(statement.destination.view.storage)
        result.update(load.data.storage for load in I.loads_unique(statement.source))
      if isinstance(statement, I.ReductionGroup):
        result.update(self.used_storage(I.Block([*statement.assignments, *statement.stores])))
      if isinstance(statement, I.ElementwiseGroup):
        result.update(self.used_storage(I.Block(list(statement.stores))))
      if isinstance(statement, I.UnrolledFor):
        result.update(self.used_storage(statement.copies))
        result.update(self.used_storage(statement.loop.body))
      for value in (
        getattr(statement, "value", None),
        getattr(statement, "condition", None),
        getattr(statement, "mask", None),
        getattr(statement, "indices", None),
      ):
        if isinstance(value, I.Expr):
          result.update(load.data.storage for load in I.loads_unique(value))
          # Physical packed operands and memory-proof metadata carry views
          # in their payload, not in an ordinary logical load edge.
          from ....ir.traversal import nodes

          result.update(node.storage for node in nodes(value) if isinstance(node, I.View))
      if isinstance(statement, (I.Store, I.IndexedStore)):
        result.add(statement.destination.storage)
      if isinstance(statement, I.ArchitectureCall) and statement.view is not None:
        result.add(statement.view.storage)
      for name in ("body", "otherwise"):
        child = getattr(statement, name, None)
        if isinstance(child, I.Block):
          result.update(self.used_storage(child))
    return result
