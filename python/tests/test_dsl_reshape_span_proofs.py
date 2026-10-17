# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Checked reshapes retain alias guards, including bounded-prefix guards."""

from types import SimpleNamespace

import numpy as np
import pytest
import vecops.language as T
from dsl_test_utils import make_compiler
from vecops.language._compiler import ir as I
from vecops.language._compiler.backends.cpu.backend import CPUBackend
from vecops.language._compiler.backends.cpu.cpp.memory import MemoryEmitter
from vecops.language._compiler.operations import current


def declaration(narrow):
  m, n, capacity = T.dynamic(lower=1), T.dynamic(lower=1), T.dynamic(lower=1)

  @T.template
  def kernel(
    A: T.Tensor((m, n), (T.dynamic(), T.dynamic()), "float32"),
    B: T.Tensor((capacity,), dtype="float32", access="output"),
  ):
    assert B.shape[0] >= m * n, "output prefix too small"
    prefix = B[: m * n]
    if narrow:
      guard = prefix[m * n - 1 : m * n]
    else:
      guard = prefix
    if T.view_disjoint(A, guard):
      matrix = T.reshape(prefix, (m, n))
      for row in T.serial(m):
        T.store(matrix[row, :], T.load(A[row, :]))
    else:
      staged = T.alloc((m, n), dtype="float32")
      T.store(staged, T.load(A))
      T.store(T.reshape(prefix, (m, n)), T.load(staged))

  return kernel


def guard_body(source):
  start = source.index("if (!(vecops_dsl::overlap")
  opening = source.rfind("{", start, source.index("\n", start))
  depth = 0
  for end in range(opening, len(source)):
    depth += (source[end] == "{") - (source[end] == "}")
    if depth == 0:
      return source[opening : end + 1]
  raise AssertionError("guard body did not close")


@pytest.mark.parametrize("target", ("AVX512", "SVE", "Scalar"))
@pytest.mark.parametrize("narrow", (False, True))
def test_checked_reshape_guard_covers_only_its_actual_source_span(target, narrow):
  artifact = CPUBackend().lower(declaration(narrow).ir, (), target=target)
  body = guard_body(artifact.source)
  assert ("vecops_dsl::Buffer<float>" in body) == narrow


@pytest.mark.parametrize("narrow", (False, True))
def test_native_reshape_guards_aliases_strides_and_unused_capacity(narrow, tmp_path):
  compiler = make_compiler(jobs=2, cache_dir=tmp_path / "cache", build_dir=tmp_path / "build")
  kernel = T.compile(declaration(narrow), compiler=compiler)
  for m, n in ((3, 5), (2, 17), (1, 65)):
    size = m * n
    for layout in ("separate", "overlap", "reverse_overlap"):
      owner = np.arange(2 * size + 32, dtype=np.float32)
      expected = owner.copy()
      a = owner[:size].reshape(m, n)
      ea = expected[:size].reshape(m, n)
      if layout == "reverse_overlap":
        a, ea = a[::-1, ::-1], ea[::-1, ::-1]
      if layout == "separate":
        output = np.full(size + 7, -99, dtype=np.float32)
        eo = output.copy()
      else:
        output = owner[2 : 2 + size + 7]
        eo = expected[2 : 2 + size + 7]
      if layout == "separate" or not narrow:
        eo[:size] = ea.copy().ravel()
      else:
        # The narrow guard is disjoint, but later writes overlap later rows.
        for row in range(m):
          eo[row * n : (row + 1) * n] = ea[row].copy()
      kernel.operator.native(a, output, **kernel._core.specialization_values)
      np.testing.assert_array_equal(owner, expected)
      np.testing.assert_array_equal(output, eo)


def view(storage, dtype, count, offset=0, *, origin=None):
  lit = I.literal
  region = None
  if origin is not None:
    region = I.ViewRegion(
      (lit(16),), (lit(1),), (lit(offset - origin),), (lit(count),), (1,), (0,), (False,), root_offset=lit(origin)
    )
  return I.View(storage, dtype, (lit(count),), (lit(1),), lit(offset), region=region)


def emitter(a, b):
  result = MemoryEmitter()
  result.disjoint_roots = {tuple(sorted((a.storage, b.storage))): (a, b)}
  result.storage_views = {a.storage: a, b.storage: b}
  return result


def test_equal_element_counts_do_not_imply_equal_byte_spans():
  a, b = view("A", "float32", 16), view("B", "uint8", 16)
  e = emitter(a, b)
  assert not e.disjoint_proven(a, view("B", "float32", 16))
  assert e.disjoint_proven(a, view("B", "int8", 16))


def test_same_storage_windows_must_match_opposite_guard_roles():
  a, b = view("X", "float32", 16), view("X", "float32", 16, 16)
  e = emitter(a, b)
  assert e.disjoint_proven(a, b)
  assert e.disjoint_proven(b, a)
  assert not e.disjoint_proven(a, a)
  assert not e.disjoint_proven(b, b)


def test_logical_root_sizes_do_not_hide_a_different_physical_origin():
  a, b = view("A", "float32", 16, 2), view("B", "float32", 16)
  e = emitter(a, b)
  assert not e.disjoint_proven(view("A", "float32", 16, origin=0), b)


def test_bounded_subviews_keep_their_nonzero_physical_root_origin():
  a, b = view("A", "float32", 16, 5), view("B", "float32", 16, 3)
  e = emitter(a, b)
  assert e.disjoint_proven(view("A", "float32", 4, 9, origin=5), view("B", "float32", 2, 6, origin=3))
  assert not e.disjoint_proven(view("A", "float32", 4, 10, origin=6), view("B", "float32", 2, 6, origin=3))


def test_span_metadata_can_expand_immutable_capture_definitions():
  a, b = view("A", "float32", 16), view("B", "float32", 16)
  e = emitter(a, b)
  e.ir = SimpleNamespace(bindings={}, parameters={})
  e.environment = SimpleNamespace(operations=current())
  e.prepared = SimpleNamespace(
    immutable_geometry={"n": I.literal(16), "captured": I.Expr(I.ScalarType("int64"), "local", data="n")}
  )
  own = I.View(
    "A", "float32", (I.Expr(I.ScalarType("int64"), "frozen", data="captured"),), (I.literal(1),), I.literal(0)
  )
  assert e.disjoint_proven(own, b)


def test_captures_of_mutable_coordinates_do_not_cover_guarded_spans():
  count = I.Expr(I.ScalarType("int64"), "local", data="n")
  a = I.View("A", "float32", (count,), (I.literal(1),), I.literal(0))
  b = view("B", "float32", 16)
  e = emitter(a, b)
  e.ir = SimpleNamespace(bindings={}, parameters={})
  e.environment = SimpleNamespace(operations=current())
  e.prepared = SimpleNamespace(immutable_geometry={"captured": count})
  own = I.View(
    "A", "float32", (I.Expr(I.ScalarType("int64"), "frozen", data="captured"),), (I.literal(1),), I.literal(0)
  )
  assert not e.disjoint_proven(own, b)
  assert not e.disjoint_proven(a, b)
