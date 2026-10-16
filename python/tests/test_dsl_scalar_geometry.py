# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Total scalar certificates retain historical metadata and runtime aliasing."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import vecops
import vecops.language as T
from dsl_test_utils import make_compiler
from vecops.language._compiler import ir as I
from vecops.language._compiler.backends.cpu.broadcasts import ScalarGeometry, collect_broadcast_analysis
from vecops.language._compiler.backends.cpu.cpp.expressions import ExpressionsEmitter
from vecops.language._compiler.backends.cpu.cpp.memory import MemoryEmitter
from vecops.language._compiler.environment import freeze


def local(name):
  return I.Expr(I.ScalarType("int64"), "local", data=name)


def example(*, offset=None, stride=None, repeated=False):
  i, off, step = local("i"), local("captured_offset"), local("captured_stride")
  zero, one = I.literal(0), I.literal(1)
  target_step = I.binary("add", i, one)
  source = I.View("A", "float32", (I.literal(4),), (step,), off)
  destination = replace(source, storage="C", offset=i, strides=(target_step,))
  load = I.Expr(I.TileType("float32", source.shape), "load", data=source)
  value = I.Expr(load.type, "add", (load, load)) if repeated else load
  mask = I.Expr(I.TileType("bool", source.shape), "gt", (load, I.literal(0.0))) if repeated else None
  store = I.Store(destination, value, mask)
  body = I.Block([
    I.Assign("captured_offset", offset if offset is not None else I.binary("add", i, zero)),
    I.Assign("captured_stride", stride if stride is not None else I.binary("add", i, one)), store,
  ])
  kernel = I.KernelIR(I.Block([I.For("i", zero, I.literal(5), one, body)]), {}, {})
  return kernel, store, (off, i), (step, target_step), load


class Emitter(MemoryEmitter, ExpressionsEmitter):
  pass


def emitter_for(kernel):
  analysis = collect_broadcast_analysis(kernel)
  e = Emitter()
  e.prepared = SimpleNamespace(scalar_geometry=analysis.geometry)
  e.chunk_cache = {"outer": "outer_vector"}
  e.environment = freeze(target="Scalar")
  e.hardware_scalar_bindings, e.hardware_scope = {}, None
  e.parameter_pointers = {"A": "A_ptr", "C": "C_ptr"}
  e.active_loads, e.snapshots = {}, {}
  return e


def test_offset_stride_certificates_change_actual_alias_expressions_without_proving_disjointness():
  kernel, store, offset, stride, _ = example(repeated=True)
  e = emitter_for(kernel)
  assert type(e.prepared.scalar_geometry[id(store)]) is ScalarGeometry
  before = e.same_mapping(store.value.args[0].data, store.destination)
  with e.scalar_geometry(store):
    assert e.same_geometry(*offset) and e.same_geometry(*stride)
    assert e.scalar(offset[0]) == e.scalar(offset[1]) == "captured_offset"
    assert e.scalar(stride[0]) == e.scalar(stride[1]) == "captured_stride"
    after = e.same_mapping(store.value.args[0].data, store.destination)
    assert after != before
    assert "A_ptr" in after and "C_ptr" in after and "static_cast<const void*>" in after
    assert "vecops_dsl::overlap" in e.overlap(store.value.args[0].data, store.destination)
    assert not e.disjoint_proven(store.value.args[0].data, store.destination)
  assert not e.same_geometry(*offset) and not e.same_geometry(*stride)


@pytest.mark.parametrize("grouped", [False, True])
def test_shared_store_metadata_intersects_every_lexical_domain(grouped):
  kernel, store, offset, _, _ = example()
  if grouped:
    kernel.body.statements[0].body.statements[-1] = I.ElementwiseGroup((store, store))
  first = kernel.body.statements[0]
  bad = I.Block([I.Assign("captured_offset", I.literal(99)),
                 I.Assign("captured_stride", I.binary("add", local("i"), I.literal(1))),
                 first.body.statements[-1]])
  kernel.body.statements.append(replace(first, body=bad))
  e = emitter_for(kernel)
  with e.scalar_geometry(store):
    assert not e.same_geometry(*offset)


@pytest.mark.parametrize("mutation", ["counter", "dependency", "opaque", "mask"])
def test_writes_and_opaque_metadata_do_not_reinterpret_historical_captures(mutation):
  kernel, store, offset, _, _ = example()
  body = kernel.body.statements[0].body
  if mutation == "counter":
    body.statements.insert(2, I.Assign("i", I.literal(3), False))
  elif mutation == "dependency":
    body.statements.insert(0, I.Assign("base", local("i")))
    body.statements[1] = I.Assign("captured_offset", local("base"))
    body.statements.insert(3, I.Assign("base", I.literal(99), False))
    store = replace(store, destination=replace(store.destination, offset=local("base")))
    body.statements[-1] = store
    offset = (offset[0], store.destination.offset)
  elif mutation == "opaque":
    body.statements.insert(2, I.Assign("opaque", I.Expr(I.ScalarType("int64"), "raw", data="mutate()")))
  else:
    store = replace(store, mask=I.Expr(I.TileType("bool", store.destination.shape), "unknown_effect"))
    body.statements[-1] = store
  e = emitter_for(kernel)
  with e.scalar_geometry(store):
    assert not e.same_geometry(*offset)


def unsafe_metadata():
  i = local("i")
  yield local("unknown")
  yield I.Expr(I.ScalarType("int64"), "raw", data="unvalidated_metadata")
  yield I.Expr(I.ScalarType("int64"), "unknown_effect", (i,))
  yield I.Expr(I.ScalarType("int64"), "load", data=I.View("M", "int64", (), (), I.literal(0)))
  for op, divisor in [("floordiv", 0), ("mod", 0), ("floordiv", -1)]:
    yield I.Expr(I.ScalarType("int64"), op, (i, I.literal(divisor)))
  for dtype in ["uint64", "int32", "float32", "bool"]:
    yield I.Expr(I.ScalarType(dtype), "literal", data=1)
  overflow = I.Expr(I.ScalarType("int64"), "add", (I.literal((1 << 63) - 1), I.literal(1)))
  yield overflow
  yield I.Expr(I.ScalarType("int64"), "sub", (overflow, I.literal(1)))


@pytest.mark.parametrize("unsafe", list(unsafe_metadata()))
def test_unknown_effectful_trapping_or_unbounded_address_metadata_is_not_certified(unsafe):
  kernel, store, offset, _, _ = example(offset=unsafe)
  e = emitter_for(kernel)
  with e.scalar_geometry(store):
    assert not e.same_geometry(*offset)


def test_negative_stride_equality_still_keeps_runtime_span_and_pointer_comparison():
  kernel, store, _, stride, load = example(stride=I.literal(-2))
  store = replace(store, destination=replace(store.destination, strides=(I.binary("sub", I.literal(0), I.literal(2)),)))
  kernel.body.statements[0].body.statements[-1] = store
  e = emitter_for(kernel)
  with e.scalar_geometry(store):
    assert e.same_geometry(stride[0], store.destination.strides[0])
    assert "vecops_dsl::overlap" in e.overlap(load.data, store.destination)
    assert "A_ptr" in e.same_mapping(load.data, store.destination)
    assert "C_ptr" in e.same_mapping(load.data, store.destination)


def test_protect_store_nested_fallback_and_exception_restore_group_aliases_and_cache():
  kernel, store, offset, _, _ = example()
  group = I.ElementwiseGroup((store,))
  kernel.body.statements[0].body.statements[-1] = group
  e = emitter_for(kernel)
  outer_cache = e.chunk_cache
  observations = []

  def capture(_load, _destination, _preserve):
    observations.append(e.same_geometry(*offset))
    e.chunk_cache["inside"] = "branch_vector"
    if not observations[-1]:
      raise RuntimeError("fallback abort")

  e.snapshot = capture
  with e.scalar_geometry(group):
    group_cache = e.chunk_cache
    e.protect_store(store, {})
    assert e.chunk_cache is group_cache and not group_cache
    assert e.same_geometry(*offset)
    with pytest.raises(RuntimeError, match="fallback abort"):
      e.protect_store(replace(store), {})
    assert e.chunk_cache is group_cache and e.same_geometry(*offset)
  assert observations == [True, False]
  assert e.chunk_cache is outer_cache and not e.active_scalar_aliases


@pytest.mark.parametrize("boundary", ["dot", "value_transform", "gather", "future"])
def test_scalar_certificates_preserve_matrix_transform_gather_and_future_read_captures(boundary):
  kernel, store, _, _, load = example()
  if boundary == "dot":
    value = I.Expr(load.type, "dot", (load, load, I.Expr(load.type, "literal", data=0.0)))
  elif boundary == "value_transform":
    value = I.Expr(load.type, boundary, (load,), data=("transpose", ()))
  elif boundary == "gather":
    load = I.Expr(load.type, boundary, data=load.data)
    value = load
  else:
    value = store.value
  store = replace(store, value=value)
  kernel.body.statements[0].body.statements[-1] = store
  e = emitter_for(kernel)
  e.active_loads = {id(load): load}
  captured = []
  e.snapshot = lambda reader, destination, preserve: captured.append((reader, preserve))
  e.protect_store(store, {id(load): load} if boundary == "future" else {})
  assert captured and all(preserve for _, preserve in captured)


def sliced_template(history, masked):
  n = T.dynamic(lower=0)

  @T.template
  def kernel(A: T.Tensor((n,), (1,), T.float32, access="inout"),
             C: T.Tensor((n,), (1,), T.float32, access="output"),
             D: T.Tensor((n,), (1,), T.float32, access="output")):
    if history:
      original = T.load(A)
      T.store(A, original + 3)
    for start in T.serial(0, n, 7, unroll=1):
      value = T.load(A[start:start + 7])
      if masked:
        T.store(C[start:start + 7], value + 2, mask=value > 0)
      else:
        T.store(C[start:start + 7], value + 2)
    if history:
      T.store(D, original + 5)

  return kernel


@pytest.fixture(scope="module")
def native_sliced(tmp_path_factory):
  p = tmp_path_factory.mktemp("scalar-geometry")
  compiler = make_compiler(jobs=4, output="quiet", cache_dir=p / "cache", build_dir=p / "build")
  kernels = {}
  with vecops.collect_compile_requests() as collector:
    for history in [False, True]:
      for masked in [False, True]:
        compiled = T.compile(sliced_template(history, masked), compiler=compiler)
        metadata = (vecops.TensorMeta((17,), strides=(1,), dtype="float32"),) * 3
        compiled.operator(*metadata, **compiled._core.specialization_values)
        kernels[history, masked] = compiled
  vecops.compile_batch(collector.requests, parallelism=4, execution_parallelism=1)
  return kernels


@pytest.mark.parametrize("history,masked", [(False, False), (False, True), (True, False), (True, True)])
def test_native_sliced_aliases_match_sequential_copy_oracle_and_full_owner_canaries(native_sliced, history, masked):
  for width in [0, 1, 6, 7, 8, 17, 33]:
    for layout in ["disjoint", "inplace", "forward_overlap", "backward_overlap", "one_owner"]:
      owners = [np.arange(width + 4, dtype=np.float32) - 8,
                np.full(width + 4, -19, dtype=np.float32), np.full(width + 4, -31, dtype=np.float32)]
      expected = [owner.copy() for owner in owners]

      def views(storage):
        a = storage[0][1:width + 1]
        c = storage[1][1:width + 1]
        d = storage[2][1:width + 1]
        if layout == "inplace":
          c = a
        elif layout in {"forward_overlap", "one_owner"}:
          c = storage[0][2:width + 2]
        elif layout == "backward_overlap":
          c = storage[0][:width]
        if layout == "one_owner":
          d = storage[0][:width]
        return a, c, d

      a, c, d = views(owners)
      ea, ec, ed = views(expected)
      original = ea.copy()
      if history:
        ea[:] = original + np.float32(3)
      for start in range(0, width, 7):
        stop = min(start + 7, width)
        value = ea[start:stop].copy()
        if masked:
          np.copyto(ec[start:stop], value + np.float32(2), where=value > 0)
        else:
          ec[start:stop] = value + np.float32(2)
      if history:
        ed[:] = original + np.float32(5)
      native_sliced[history, masked].operator.native(a, c, d, **native_sliced[history, masked]._core.specialization_values)
      for actual_owner, expected_owner in zip(owners, expected):
        np.testing.assert_array_equal(actual_owner, expected_owner, err_msg=f"{width=}/{layout=}/{history=}/{masked=}")
