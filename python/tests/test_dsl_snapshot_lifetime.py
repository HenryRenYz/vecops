# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Implicit snapshot owners may end only after their final proven consumer."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import vecops
import vecops.language as T
from dsl_test_utils import make_compiler
from vecops.language._compiler import ir as I
from vecops.language._compiler.backends.cpu.backend import CPUBackend
from vecops.language._compiler.backends.cpu.cpp.memory import MemoryEmitter
from vecops.language._compiler.backends.cpu.storage import SnapshotScope, StoragePlan, snapshot_lifetime_scopes
from vecops.language._compiler.backends.cpu.verification import verify_snapshot_scopes
from vecops.language._compiler.environment import freeze


def view(name):
  return I.View(name, "float32", (I.literal(17),), (I.literal(1),), I.literal(0))


def read(name="A"):
  source = view(name)
  return I.Expr(I.TileType(source.dtype, source.shape), "load", data=source)


@pytest.mark.parametrize("change", ("same", "shape", "stride", "offset", "dtype", "borrowed", "rank", "worker", "arm"))
def test_snapshot_owner_reuse_rejects_different_layouts_and_shared_worker_owners(change):
  emitter = MemoryEmitter()
  emitter.parallel_depth = int(change == "worker")
  emitter.environment = SimpleNamespace(codegen_target=SimpleNamespace(architecture="arm" if change == "arm" else "x86"))
  saved = replace(view("saved"), owned=True)
  layout = replace(saved, storage="next")
  if change == "shape":
    layout = replace(layout, shape=(I.literal(16),))
  elif change == "stride":
    layout = replace(layout, strides=(I.literal(-1),))
  elif change == "offset":
    layout = replace(layout, offset=I.literal(1))
  elif change == "dtype":
    layout = replace(layout, dtype="int32")
  elif change == "borrowed":
    saved = replace(saved, owned=False)
  elif change == "rank":
    layout = replace(layout, shape=(I.literal(1), *layout.shape), strides=(I.literal(17), *layout.strides))
  assert emitter.reusable_snapshot_owner((saved, "historical_guard"), layout) == (change == "same")
  assert not emitter.reusable_snapshot_owner(None, layout)


def plans(statements):
  kernel = I.KernelIR(I.Block(statements), {}, {})
  return kernel, snapshot_lifetime_scopes(kernel, freeze(target="Scalar").operations)


def test_independent_reads_end_at_each_consumer_but_shared_history_does_not():
  first, second = read(), read()
  a, b = I.Store(view("C"), first), I.Store(view("D"), second)
  _, result = plans([I.BindTile(first), a, I.BindTile(second), b])
  assert result[id(a)].reads == (first,)
  assert result[id(b)].reads == (second,)
  later = I.Store(view("D"), first)
  _, result = plans([I.BindTile(first), a, later])
  assert not result


def test_group_owns_all_internal_consumers_but_not_an_external_historical_use():
  old = read()
  stores = (I.Store(view("C"), old), I.Store(view("D"), old))
  group = I.ElementwiseGroup(stores)
  _, result = plans([I.BindTile(old), group])
  assert set(result) == {id(group)}
  assert result[id(group)].reads == (old,)
  _, result = plans([group, I.Store(view("E"), old)])
  assert not result


@pytest.mark.parametrize("grouped", (False, True))
def test_shared_subject_occurrences_are_not_a_single_consumer(grouped):
  old = read()
  store = I.Store(view("C"), old)
  subject = I.ElementwiseGroup((store,)) if grouped else store
  _, result = plans([subject, I.If(I.literal(True), I.Block([subject]), I.Block())])
  assert not result


@pytest.mark.parametrize("owner", ("outlined", "hardware", "grid", "unrolled"))
def test_capture_worker_and_specialized_regions_keep_original_owner_scope(owner):
  old = read()
  body = I.Block([I.Store(view("C"), old)])
  if owner == "outlined":
    statement = I.ExitRegion("exit", body, outlined=True)
  elif owner == "hardware":
    statement = I.HardwareRegion("amx", body)
  elif owner == "grid":
    statement = I.Grid(("p",), (I.literal(1),), body)
  else:
    statement = I.UnrolledFor(I.For("i", I.literal(0), I.literal(1), I.literal(1), body), body)
  _, result = plans([I.BindTile(old), statement])
  assert not result


def test_metadata_reads_and_unknown_ownership_block_shortening():
  old = read()
  store = I.Store(view("C"), old)
  # Even a reference carried through owned type/data must count as another use.
  metadata_user = I.Expr(I.ScalarType("int64"), "raw", data=(old,))
  _, result = plans([store, I.Assign("metadata", metadata_user)])
  assert not result

  class Hidden:
    def __getattribute__(self, name):
      raise AssertionError("unknown ownership must not execute custom access")

  unknown = I.Expr(I.ScalarType("int64"), "raw", data=Hidden())
  _, result = plans([store, I.Assign("unknown", unknown)])
  assert not result


def test_unknown_expr_and_node_subclasses_are_not_enumerated():
  class UnknownExpr(I.Expr):
    def __getattribute__(self, name):
      raise AssertionError("unknown Expr subclass was inspected")

  class UnknownStore(I.Store):
    def __getattribute__(self, name):
      raise AssertionError("unknown Node subclass was inspected")

  old = read()
  _, result = plans([I.Store(view("C"), UnknownExpr(old.type, "load", data=old.data))])
  assert not result
  _, result = plans([UnknownStore(view("C"), old)])
  assert not result


def test_cyclic_ownership_retains_existing_lifetimes():
  old = read()
  cyclic = I.Expr(old.type, "add", (old,))
  object.__setattr__(cyclic, "args", (cyclic,))
  _, result = plans([I.Store(view("C"), cyclic)])
  assert not result


def test_many_independent_reads_do_not_disable_other_consumer_scopes():
  stores = [I.Store(view(f"C{i}"), read(f"A{i}")) for i in range(257)]
  _, result = plans(stores)
  assert set(result) == {id(store) for store in stores}
  assert all(result[id(store)].reads == (store.value,) for store in stores)


def test_shared_dag_escape_reaches_every_read_without_transitive_read_sets():
  from vecops.language._compiler.backends.cpu.storage import _SnapshotOwners

  catalog = freeze(target="Scalar").operations
  loads = [read(f"A{i}") for i in range(129)]
  value = loads[0]
  for load in loads[1:]:
    value = I.Expr(value.type, "add", (value, load))
  first, second = I.Store(view("C"), value), I.Store(view("D"), value)
  _, result = plans([first])
  assert set(result) == {id(first)}
  assert {id(v) for v in result[id(first)].reads} == {id(v) for v in loads}
  _, result = plans([first, second])
  assert not result
  query = _SnapshotOwners(catalog)
  assert query.inspect(value)
  # A shared diamond has one record per actual node, not one read-set per path.
  query.consume(value, 1, 1)
  query.consume(value, 2, 2)
  query.consume(value, 3, 3)
  assert all(owners == (query.escaped, query.escaped) for owners in query.owners.values())
  assert all(len(record) == 2 for record in query.graph.values())


def test_group_owner_and_exact_store_owner_are_distinct_dimensions():
  first, second = read("A"), read("B")
  a, b = I.Store(view("C"), first), I.Store(view("D"), second)
  group = I.ElementwiseGroup((a, b))
  _, result = plans([group])
  assert set(result) == {id(group), id(a), id(b)}
  assert result[id(a)].reads == (first,)
  assert result[id(b)].reads == (second,)
  assert {id(v) for v in result[id(group)].reads} == {id(first), id(second)}
  _, result = plans([group, I.Store(view("E"), first)])
  assert set(result) == {id(b)}


def test_verifier_rechecks_actual_users_and_strong_historical_subjects():
  old = read()
  store = I.Store(view("C"), old)
  kernel, result = plans([store])
  prepared = SimpleNamespace(vectors=SimpleNamespace(kernel=kernel), storage=StoragePlan({}, {}, (), result))
  verify_snapshot_scopes(prepared)
  proof = result[id(store)]
  for changed in (
    replace(proof, reads=(read(),)),
    replace(proof, occurrences=2),
    replace(proof, subject=replace(store)),
  ):
    prepared.storage = replace(prepared.storage, snapshot_scopes={id(store): changed})
    with pytest.raises((TypeError, ValueError), match="snapshot"):
      verify_snapshot_scopes(prepared)
  prepared.storage = replace(prepared.storage, snapshot_scopes=result)
  kernel.body.statements.append(I.Store(view("D"), old))
  with pytest.raises((TypeError, ValueError), match="escapes"):
    verify_snapshot_scopes(prepared)


def emitter_for(store):
  emitter = MemoryEmitter()
  proof = SnapshotScope(store, tuple(I.loads_unique(store.value)), 1)
  emitter.prepared = SimpleNamespace(storage=StoragePlan({}, {}, (), {id(store): proof}))
  emitter.snapshots = {100: ("outer_view", "outer_alias")}
  emitter.active_loads = {100: read("outer")}
  emitter.materialized = {200: "outer_materialization"}
  emitter.reduction_materialized = {300: "outer_reduction"}
  emitter.chunk_cache = {400: "outer_cpp_local"}
  emitter.store_snapshot_provenance = {500: "outer_provenance"}
  emitter.lines = []
  emitter.opening = lambda line: emitter.lines.append(line + " {")
  emitter.close = lambda: emitter.lines.append("}")
  return emitter


def test_emission_restores_every_outer_binding_on_normal_and_exceptional_exit():
  store = I.Store(view("C"), read())
  for failing in (False, True):
    emitter = emitter_for(store)
    names = (
      "snapshots",
      "active_loads",
      "materialized",
      "reduction_materialized",
      "chunk_cache",
      "store_snapshot_provenance",
    )
    before = tuple(getattr(emitter, name) for name in names)
    try:
      with emitter.snapshot_lifetime(store, {}):
        assert not emitter.chunk_cache
        for name in names:
          getattr(emitter, name)[999] = "inner_cpp_binding"
        if failing:
          raise RuntimeError("emit failure")
    except RuntimeError:
      assert failing
    assert all(getattr(emitter, name) is original for name, original in zip(names, before))
    assert all(999 not in mapping for mapping in before)
    assert emitter.lines == ["/* consumer-local snapshot lifetime */ {", "}"]


def test_preserved_future_read_keeps_outer_scope_even_for_local_own_reads():
  store = I.Store(view("C"), read())
  emitter = emitter_for(store)
  before = emitter.snapshots
  with emitter.snapshot_lifetime(store, {100: emitter.active_loads[100]}):
    assert emitter.snapshots is before
  assert not emitter.lines


def test_single_read_guard_is_evaluated_once_and_only_slow_branch_owns_buffer():
  store = I.Store(view("C"), read())
  emitter = emitter_for(store)
  emitter.snapshots, emitter.materialized, emitter.reduction_materialized = {}, {}, {}
  emitter.vector_ir = SimpleNamespace(manager=SimpleNamespace(options={}))
  emitter.active_scalar_classes, emitter.active_scalar_aliases = {}, {}
  emitter.snapshot_required = lambda *_: True
  guards, copies, seen = [], [], []

  def condition(*args):
    guards.append(args)
    return "original_alias"

  def contents(load, alias):
    copies.append((load, alias))
    emitter.snapshots[id(load)] = ("slow_buffer", alias)

  emitter.snapshot_condition, emitter.snapshot_contents = condition, contents
  before = (
    emitter.snapshots,
    emitter.materialized,
    emitter.reduction_materialized,
    emitter.chunk_cache,
    emitter.store_snapshot_provenance,
  )

  def emit():
    seen.append(dict(emitter.snapshots))
    emitter.materialized[999] = "branch_local"
    assert not emitter.chunk_cache

  assert emitter.single_snapshot_store(store, {}, emit)
  assert len(guards) == len(copies) == 1
  assert not seen[0] and seen[1][id(copies[0][0])] == ("slow_buffer", "original_alias")
  after = (
    emitter.snapshots,
    emitter.materialized,
    emitter.reduction_materialized,
    emitter.chunk_cache,
    emitter.store_snapshot_provenance,
  )
  assert all(a is b for a, b in zip(before, after))
  assert 999 not in emitter.materialized


@pytest.mark.parametrize("blocked", ("historical", "materialized", "future", "multiple"))
def test_single_guard_versioning_never_shortens_existing_or_multiple_owners(blocked):
  store = I.Store(view("C"), read())
  emitter = emitter_for(store)
  emitter.snapshots, emitter.materialized, emitter.reduction_materialized = {}, {}, {}
  future = {}
  if blocked == "historical":
    emitter.snapshots[100] = ("old", "alias")
  elif blocked == "materialized":
    emitter.materialized[100] = "old"
  elif blocked == "future":
    future = {100: emitter.active_loads[100]}
  else:
    proof = emitter.prepared.storage.snapshot_scopes[id(store)]
    emitter.prepared.storage = replace(
      emitter.prepared.storage, snapshot_scopes={id(store): replace(proof, reads=(*proof.reads, read()))}
    )
  assert not emitter.single_snapshot_store(store, future, lambda: pytest.fail("must retain original emission"))
  assert not emitter.lines


def test_single_guard_branch_exception_restores_capture_and_cache_maps():
  store = I.Store(view("C"), read())
  emitter = emitter_for(store)
  emitter.snapshots, emitter.materialized, emitter.reduction_materialized = {}, {}, {}
  emitter.vector_ir = SimpleNamespace(manager=SimpleNamespace(options={}))
  emitter.snapshot_required = lambda *_: True
  emitter.snapshot_condition = lambda *_: "alias"
  emitter.snapshot_contents = lambda *_: pytest.fail("slow branch not emitted after error")
  before = (
    emitter.snapshots,
    emitter.materialized,
    emitter.reduction_materialized,
    emitter.chunk_cache,
    emitter.store_snapshot_provenance,
  )

  def fail():
    emitter.snapshots[999] = ("invalid", "local")
    raise RuntimeError("emitter failed")

  with pytest.raises(RuntimeError, match="emitter failed"):
    emitter.single_snapshot_store(store, {}, fail)
  after = (
    emitter.snapshots,
    emitter.materialized,
    emitter.reduction_materialized,
    emitter.chunk_cache,
    emitter.store_snapshot_provenance,
  )
  assert all(a is b for a, b in zip(before, after))
  assert emitter.lines[-1] == "}"


def template(kind):
  M, N = T.dynamic(lower=0), T.dynamic(lower=0)
  S0, S1 = T.dynamic(), T.dynamic()

  @T.func(inline=False)
  def helper(source, destination):
    T.store(destination, source + 2)

  @T.template
  def kernel(
    A: T.Tensor((M, N), (S0, S1), "float32", access="inout"),
    C: T.Tensor((M, N), (S0, S1), "float32", access="output"),
    D: T.Tensor((M, N), (S0, S1), "float32", access="output"),
  ):
    if kind == "captured":
      original = T.load(A)
      T.store(A, 42)
      T.store(C, original + 1)
      T.compiler_fence()
      T.store(D, original + 2)
    elif kind == "outlined":
      original = T.load(A)
      T.store(A, 42)
      helper(original, C)
      T.store(D, original + 3)
    else:
      for row in T.serial(M, unroll=1):
        original = T.load(A[row, :])
        T.store(C[row, :], original + 1)
        T.compiler_fence()
        other = original if kind == "shared" else T.load(A[row, :])
        T.store(D[row, :], other + 2)

  return kernel


def test_public_independent_stores_gain_scopes_and_shared_history_keeps_owner():
  local = CPUBackend().lower(template("independent").ir, (), target="Scalar")
  assert local.backend_ir.storage.snapshot_scopes
  assert "/* consumer-local snapshot lifetime */" in local.source
  assert "vecops_dsl::Buffer<float> dsl_c_snapshot_" in local.source
  shared = CPUBackend().lower(template("shared").ir, (), target="Scalar")
  assert not shared.backend_ir.storage.snapshot_scopes
  assert "/* consumer-local snapshot lifetime */" not in shared.source
  disabled = CPUBackend().lower(
    template("independent").ir, (), target="Scalar", options={"cpu_snapshot_lifetime_scopes": False}
  )
  assert not disabled.backend_ir.storage.snapshot_scopes
  assert "/* consumer-local snapshot lifetime */" not in disabled.source
  for invalid in (0, 1, None, "true"):
    with pytest.raises(TypeError, match="cpu_snapshot_lifetime_scopes must be bool"):
      CPUBackend().lower(
        template("independent").ir, (), target="Scalar", options={"cpu_snapshot_lifetime_scopes": invalid}
      )


@pytest.fixture(scope="module")
def native_snapshot_kernels(tmp_path_factory):
  directory = tmp_path_factory.mktemp("snapshot-lifetime")
  compiler = make_compiler(jobs=2, output="quiet", cache_dir=directory / "cache", build_dir=directory / "build")
  result = {}
  with vecops.collect_compile_requests() as collector:
    for kind in ("independent", "shared", "captured", "outlined"):
      for enabled in (False, True):
        compiled = T.compile(template(kind), compiler=compiler, pass_options={"cpu_snapshot_lifetime_scopes": enabled})
        metadata = (vecops.TensorMeta((3, 17), dtype="float32"),) * 3
        compiled.operator(*metadata, **compiled._core.specialization_values)
        result[kind, enabled] = compiled
  vecops.compile_batch(collector.requests, parallelism=2, execution_parallelism=1)
  return result


@pytest.mark.parametrize("kind", ("independent", "shared", "captured", "outlined"))
@pytest.mark.parametrize("enabled", (False, True))
def test_native_aliases_history_helpers_fullowners_and_two_poisons(native_snapshot_kernels, kind, enabled):
  compiled = native_snapshot_kernels[kind, enabled]
  for width in (0, 1, 17, 33):
    for layout in ("dense", "shifted", "inplace", "late_inplace", "late_shifted", "both", "negative"):
      for poison in (-90.0, 195.0):
        n = 3 * width
        owner = np.full(3 * n + 16, poison, np.float32)
        a = owner[4 : 4 + n].reshape(3, width)
        c = owner[4 + n : 4 + 2 * n].reshape(3, width)
        d = owner[4 + 2 * n : 4 + 3 * n].reshape(3, width)
        a[:] = np.arange(n, dtype=np.float32).reshape(3, width) / 4
        if layout == "shifted":
          c = owner[5 : 5 + n].reshape(3, width)
        elif layout == "inplace":
          c = a
        elif layout == "late_inplace":
          d = a
        elif layout == "late_shifted":
          d = owner[5 : 5 + n].reshape(3, width)
        elif layout == "both":
          c, d = a, a
        elif layout == "negative":
          a, c, d = a[:, ::-1], c[:, ::-1], d[:, ::-1]
        expected = owner.copy()

        def equivalent(array, owner=owner, expected=expected):
          offset = array.ctypes.data - owner.ctypes.data
          return np.ndarray(array.shape, dtype=array.dtype, buffer=expected, offset=offset, strides=array.strides)

        aa, cc, dd = (equivalent(value) for value in (a, c, d))
        if kind in {"captured", "outlined"}:
          original = aa.copy()
          aa[:] = 42
          cc[:] = original + (2 if kind == "outlined" else 1)
          dd[:] = original + (3 if kind == "outlined" else 2)
        else:
          for row in range(3):
            original = aa[row].copy()
            cc[row] = original + 1
            dd[row] = (original if kind == "shared" else aa[row].copy()) + 2
        compiled.operator.native(a, c, d, **compiled._core.specialization_values)
        np.testing.assert_array_equal(owner, expected, err_msg=f"{kind}/{enabled}/{width}/{layout}/{poison}")


def loop_capture_template(kind):
  M, N = T.dynamic(lower=0), T.dynamic(lower=0)

  @T.template
  def kernel(
    A: T.Tensor((M, N), dtype="float32", access="inout"),
    C: T.Tensor((M, N), dtype="float32", access="output"),
    repeat: int,
  ):
    captured = T.load(A)
    if kind == "for":
      for iteration in T.serial(repeat, unroll=1):
        T.store(C, captured + T.cast(iteration, T.float32))
    else:
      iteration = 0
      while iteration < repeat:
        T.store(C, captured + T.cast(iteration, T.float32))
        iteration += 1

  return kernel


@pytest.mark.parametrize("kind", ("for", "while"))
def test_loop_capture_keeps_outer_snapshot_before_local_consumer_scope(kind):
  artifact = CPUBackend().lower(loop_capture_template(kind).ir, (), target="Scalar")
  marker = artifact.source.index("/* consumer-local snapshot lifetime */")
  assert "vecops_dsl::Buffer<float> dsl_c_snapshot_" in artifact.source[:marker]
  assert "dsl_c_snapshot_" in artifact.source[marker:]
  assert artifact.backend_ir.storage.snapshot_scopes


@pytest.fixture(scope="module")
def native_loop_capture_kernels(tmp_path_factory):
  directory = tmp_path_factory.mktemp("loop-snapshot-lifetime")
  compiler = make_compiler(jobs=2, output="quiet", cache_dir=directory / "cache", build_dir=directory / "build")
  result = {}
  with vecops.collect_compile_requests() as collector:
    for kind in ("for", "while"):
      for enabled in (False, True):
        compiled = T.compile(
          loop_capture_template(kind), compiler=compiler, pass_options={"cpu_snapshot_lifetime_scopes": enabled}
        )
        compiled.operator(
          vecops.TensorMeta((3, 17), dtype="float32"),
          vecops.TensorMeta((3, 17), dtype="float32"),
          3,
          **compiled._core.specialization_values,
        )
        result[kind, enabled] = compiled
  vecops.compile_batch(collector.requests, parallelism=2, execution_parallelism=1)
  return result


@pytest.mark.parametrize("kind", ("for", "while"))
@pytest.mark.parametrize("enabled", (False, True))
def test_native_loop_capture_survives_all_repeated_overlapping_consumers(native_loop_capture_kernels, kind, enabled):
  compiled = native_loop_capture_kernels[kind, enabled]
  for width in (0, 1, 17, 33):
    n = 3 * width
    for shifted in (False, True):
      for poison in (-90.0, 195.0):
        owner = np.full(n + 9, poison, np.float32)
        # Tensor's default layout is dense row-major, including a zero row
        # pitch for N=0. NumPy's empty reshape can choose a nonzero row pitch.
        a = np.ndarray((3, width), dtype=np.float32, buffer=owner, offset=16, strides=(width * 4, 4))
        a[:] = np.arange(n, dtype=np.float32).reshape(3, width) / 4
        c = np.ndarray(
          (3, width), dtype=np.float32, buffer=owner, offset=(4 + int(shifted)) * 4, strides=(width * 4, 4)
        )
        original = a.copy()
        expected = owner.copy()
        expected[4 + int(shifted) : 4 + int(shifted) + n] = (original + 2).reshape(-1)
        compiled.operator.native(a, c, 3, **compiled._core.specialization_values)
        np.testing.assert_array_equal(owner, expected, err_msg=f"{kind}/{enabled}/{width}/{shifted}/{poison}")
