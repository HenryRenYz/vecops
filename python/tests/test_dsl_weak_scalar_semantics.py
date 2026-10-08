# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Literal-promotion hints are not boolean or helper-storage dtype qualifiers."""

import pytest
import vecops.language as T
from dsl_test_utils import make_compiler
from vecops.language._compiler import ir as I
from vecops.language._compiler.frontend.parser import lower
from vecops.language._compiler.ir.traversal import nodes, walk_unique


@pytest.mark.parametrize("operation", ("static_print", "print"))
@pytest.mark.parametrize("flush_kind", ("default", "weak", "typed", "runtime"))
def test_print_flush_accepts_boolean_literals_and_runtime_scalars(operation, flush_kind, capsys):
  emit = getattr(T, operation)

  @T.template
  def kernel(C: T.Tensor((1,), dtype="float32", access="output"), *, flush: bool = False):
    if flush_kind == "default":
      emit("weak-bool-ok")
    elif flush_kind == "weak":
      emit("weak-bool-ok", flush=True)
    elif flush_kind == "typed":
      emit("weak-bool-ok", flush=T.cast(False, "bool"))
    else:
      emit("weak-bool-ok", flush=flush)
    T.store(C, T.full((1,), 0))

  # Static printing retains its literal-only contract; a cast expression is
  # typed correctly, but is not evaluated by the static-print frontend.
  if operation == "static_print" and flush_kind in {"runtime", "typed"}:
    with pytest.raises(TypeError, match="static_print arguments must be compile-time values"):
      lower(kernel.ir)
    return
  result = lower(kernel.ir)
  if operation == "static_print":
    assert capsys.readouterr().out == "weak-bool-ok\n"
  else:
    statement = next(item for item in nodes(result.body) if isinstance(item, I.Print))
    assert isinstance(statement.flush.type, I.ScalarType) and statement.flush.type.dtype == "bool"


@pytest.mark.parametrize("kind", ("integer", "tile"))
def test_print_flush_still_rejects_non_boolean_scalar_storage(kind):
  @T.template
  def kernel(A: T.Tensor((3,), dtype="bool")):
    if kind == "integer":
      T.print("invalid", flush=1)
    else:
      T.print("invalid", flush=T.load(A))

  with pytest.raises(TypeError, match="print flush must be boolean"):
    lower(kernel.ir)


def _loop_helper_kernel(dtype, one, zero, first_weak):
  @T.func
  def choose(A):
    for i in T.serial(1):
      if T.load(A[0]) > 0:
        if first_weak:
          return one
        return T.cast(one, dtype)
    if first_weak:
      return T.cast(zero, dtype)
    return zero

  @T.template
  def kernel(A: T.Tensor((1,), dtype="float32"), C: T.Tensor((1,), dtype=dtype, access="output")):
    T.store(C[0], choose(A))

  return kernel


@pytest.mark.parametrize("dtype,one,zero", (("bool", True, False), ("int64", 1, 0), ("float64", 1.0, 0.0)))
@pytest.mark.parametrize("first_weak", (False, True))
def test_loop_helper_returns_same_dtype_with_mixed_literal_strength(dtype, one, zero, first_weak):
  kernel = _loop_helper_kernel(dtype, one, zero, first_weak)
  result = lower(kernel.ir)
  store = next(item for item in nodes(result.body) if isinstance(item, I.Store))
  assert isinstance(store.value.type, I.ScalarType) and store.value.type.dtype == dtype
  assert not store.value.type.weak
  assert all(not value.type.weak for value in walk_unique(store.value) if value.op == "local")


@pytest.fixture(scope="module")
def compiler(tmp_path_factory):
  root = tmp_path_factory.mktemp("weak-helper-return")
  return make_compiler(jobs=4, cache_dir=root / "cache", build_dir=root / "build")


@pytest.mark.parametrize("dtype,one,zero", (("bool", True, False), ("int64", 1, 0), ("float64", 1.0, 0.0)))
@pytest.mark.parametrize("first_weak", (False, True))
def test_native_loop_helper_returns_mixed_strength_scalar_dtype(dtype, one, zero, first_weak, compiler):
  import numpy as np

  kernel = T.compile(_loop_helper_kernel(dtype, one, zero, first_weak), compiler=compiler)
  for value in (-1.0, 0.0, 1.0):
    a, c = np.array((value,), dtype="float32"), np.empty(1, dtype=dtype)
    kernel(a, c)
    np.testing.assert_array_equal(c, np.array((one if value > 0 else zero,), dtype=dtype))


def test_loop_helper_return_dtype_changes_are_still_rejected():
  @T.func
  def choose(A):
    for i in T.serial(1):
      if T.load(A[0]) > 0:
        return 1
    return T.cast(0, "int32")

  @T.template
  def kernel(A: T.Tensor((1,), dtype="float32"), C: T.Tensor((1,), dtype="int64", access="output")):
    T.store(C[0], choose(A))

  with pytest.raises(TypeError, match="T.func return dtype changes"):
    lower(kernel.ir)
