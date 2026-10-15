# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Storage placement is planned before emission and respects value lifetimes."""

import numpy as np
import pytest
import vecops.language as T
from dsl_test_utils import make_compiler
from vecops.language._compiler import ir as I
from vecops.language._compiler.backends.cpu.planning import PreparedKernel
from vecops.language._compiler.backends.cpu.storage import plan_storage
from vecops.language._compiler.ir.vector import VectorKernel
from vecops.language._compiler.ir.verification import verify_prepared
from vecops.language._compiler.passes.manager import PassContext


def storage(name, width=17, dtype="float32", *, owned=True):
  return I.View(name, dtype, (I.literal(width),), (I.literal(1),), I.literal(0), owned=owned)


def read(view):
  return I.Expr(I.TileType(view.dtype, view.shape), "load", data=view)


def prepared(body):
  kernel = I.KernelIR(I.Block(body), {}, {})
  return PreparedKernel(VectorKernel(kernel), {}, {}, {}, {}, {})


def test_nonoverlapping_compatible_buffers_share_capacity_without_rewriting_views():
  first, second = storage("first", 32), storage("second", 17)
  c, d = storage("C", 32, owned=False), storage("D", 17, owned=False)
  ir = prepared([I.Allocate(first), I.Store(c, read(first)), I.Allocate(second), I.Store(d, read(second))])
  result = plan_storage(ir, PassContext("cpu-memory"))
  assert result.storage.slots == {"first": "first", "second": "first"}
  assert result.vectors.kernel.body.statements[2].view.storage == "second"
  verify_prepared(result)


def test_later_historical_read_prevents_storage_reuse():
  first, second = storage("first"), storage("second")
  old = read(first)
  c, d = storage("C", owned=False), storage("D", owned=False)
  ir = prepared([I.Allocate(first), I.BindTile(old), I.Allocate(second), I.Store(d, read(second)), I.Store(c, old)])
  result = plan_storage(ir, PassContext("cpu-memory"))
  assert result.storage.slots == {"first": "first", "second": "second"}


@pytest.mark.parametrize("kind", ["dtype", "worker", "branch", "disabled"])
def test_incompatible_owners_and_types_never_share_slots(kind):
  first = storage("first")
  second = storage("second", dtype="float64" if kind == "dtype" else "float32")
  body = [
    I.Allocate(first),
    I.Store(storage("C", owned=False), read(first)),
    I.Allocate(second),
    I.Store(storage("D", dtype=second.dtype, owned=False), read(second)),
  ]
  if kind == "worker":
    body = [I.Grid(("program",), (I.literal(4),), I.Block(body))]
  elif kind == "branch":
    flag = I.Expr(I.ScalarType("bool"), "raw", data="flag")
    body = [I.If(flag, I.Block(body[:2]), I.Block()), *body[2:]]
  result = plan_storage(prepared(body), PassContext("cpu-memory", {"cpu_scratch_reuse": kind != "disabled"}))
  assert result.storage.slots.get("first", "first") != result.storage.slots.get("second", "second")


def template(width):
  N = T.dynamic(lower=0)
  S0, S1 = T.dynamic(), T.dynamic()

  @T.template
  def program(
    A: T.Tensor((N, width), (S0, S1), "float32"),
    C: T.Tensor((N, width), dtype="float32", access="output"),
    D: T.Tensor((N, width), dtype="float32", access="output"),
  ):
    for row in T.serial(N, unroll=1):
      first = T.alloc((width,), dtype="float32")
      T.store(first, T.load(A[row, :]) + 1)
      T.store(C[row, :], T.load(first) * 2)
    T.compiler_fence()
    for row in T.serial(N, unroll=1):
      second = T.alloc((width,), dtype="float32")
      T.store(second, T.load(A[row, :]) - 1)
      T.store(D[row, :], T.load(second) * 3)

  return program


def test_native_scratch_reuse_and_independent_historical_reads(tmp_path):
  compiler = make_compiler(jobs=2, cache_dir=tmp_path / "cache", build_dir=tmp_path / "build")
  for reuse in (False, True):
    compiled = T.compile(
      template(17), compiler=compiler, pass_options={"cpu_scratch_reuse": reuse}, debug=T.DebugOptions(memory=True)
    )
    slots = compiled.backend_ir.storage.slots
    assert len(slots) == 2
    assert len(set(slots.values())) == (1 if reuse else 2)
    for count in (0, 3, 5):
      a = np.arange(count * 17, dtype=np.float32).reshape(count, 17)[:, ::-1]
      c, d = np.empty_like(a), np.empty_like(a)
      compiled(a, c, d)
      np.testing.assert_array_equal(c, (a + 1) * 2)
      np.testing.assert_array_equal(d, (a - 1) * 3)
    profile = compiled.profile(a, c, d, warmup=0, repeat=1, memory=True, mode="native")
    assert profile.memory_summary["instrumented_peak_bytes"] == 17 * 4 * (1 if reuse else 2)
