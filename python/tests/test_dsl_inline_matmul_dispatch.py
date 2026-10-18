# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Inline matrix dispatch preserves native and SIMD tails, seeds and aliases."""

import pytest
import vecops.language as T
from test_dsl_outline_vector_fallback import (
  contains_disjoint_guard,
  nodes,
  numeric_template,
  template,
)
from vecops.language._compiler import ir as I
from vecops.language._compiler.backends.cpu.backend import CPUBackend


def test_inline_dispatch_keeps_architectural_owners_inside_the_kernel():
  artifact = CPUBackend().lower(
    template().ir, (), target="AVX512", options={"cpu_outline_vector_fallback": False},
  )
  walked = list(nodes(artifact.kernel.body))
  assert any(isinstance(node, I.HardwareRegion) and node.kind == "amx" for node in walked)
  calls = [node for node in walked if isinstance(node, I.ExitRegion) and node.outlined]
  # The alias-preserving SIMD dispatcher has its own boundary. Native owners
  # and the proven-disjoint branch remain in the caller.
  assert calls
  assert all(not contains_disjoint_guard(call.body) for call in calls)
  assert all(not any(isinstance(node, I.HardwareRegion) for node in nodes(call.body)) for call in calls)
  assert not any(isinstance(node, I.ExitRegion) and node.outlined for call in calls for node in nodes(call.body))


@pytest.fixture(scope="module")
def native_inline_dispatch(tmp_path_factory):
  import vecops
  from dsl_test_utils import amx_available_for_test, make_compiler

  if not amx_available_for_test():
    pytest.skip("requires an AMX target to exercise native matrix dispatch")
  root = tmp_path_factory.mktemp("inline-matrix-dispatch")
  compiled = T.compile(
    numeric_template(),
    compiler=make_compiler(jobs=2, cache_dir=root / "cache", build_dir=root / "build"),
    pass_options={"cpu_outline_vector_fallback": False},
  )
  arguments = (
    vecops.TensorMeta((33, 65), dtype="bfloat16"),
    vecops.TensorMeta((65, 35), dtype="bfloat16"),
    vecops.TensorMeta((1, 35), dtype="float32"),
    vecops.TensorMeta((33, 35), dtype="float32", access=vecops.TensorAccess.output),
  )
  with vecops.collect_compile_requests() as collector:
    compiled.operator(*arguments, **compiled._core.specialization_values)
  assert len(collector.requests) == 1
  vecops.compile_batch(collector.requests, parallelism=2, execution_parallelism=1)
  return compiled


@pytest.mark.parametrize(
  "m,n,k,alias",
  [
    (1, 8, 0, "prior"),
    (1, 8, 1, "prior"),
    (1, 8, 65, "prior"),
    (8, 1, 65, "prior"),
    (1, 8, 65, "A"),
    (1, 8, 65, "B"),
    (32, 32, 17, "prior"),
    (33, 35, 65, "prior"),
    (33, 35, 65, "A"),
    (33, 35, 65, "B"),
  ],
)
def test_native_inline_dispatch_preserves_tail_seed_epilogue_and_alias(native_inline_dispatch, m, n, k, alias):
  from test_dsl_outline_vector_fallback import (
    test_native_outlined_fallback_preserves_tail_seed_epilogue_and_alias as check_numeric,
  )

  # Retain the independent FP64 oracle, mutable aliased Prior, A/B shared
  # storage and untouched-byte canaries for the explicitly inline boundary.
  check_numeric(native_inline_dispatch, m, n, k, alias)


def test_online_amx_packing_boundary_does_not_escape_to_standalone_packers(monkeypatch):
  from vecops.language._compiler.backends.cpu import packing

  def packed_template(compute):
    m, n, k = (T.dynamic(lower=0) for _ in range(3))

    @T.template
    def kernel(
      A: T.Tensor((m, k), dtype="bfloat16"),
      B: T.Tensor((k, n), dtype="bfloat16"),
      P: T.Tensor((4096,), dtype="bfloat16", access="output"),
      C: T.Tensor((m, n), dtype="float32", access="output"),
    ):
      T.pack(B, P, side="B", format="amx")
      if compute:
        b = T.packed(P, (k, n), side="B", format="amx")
        T.dot((T.load(A), b), C, implementation="amx")

    return kernel

  observed = []
  original = packing.expand_helper

  class PackingCaptured(Exception):
    pass

  def record(helper, *args, **kwargs):
    _, body = original(helper, *args, **kwargs)
    calls = [node for node in nodes(I.Block(body)) if isinstance(node, I.ExitRegion) and node.outlined]
    observed.append(bool(calls))
    assert all(not any(isinstance(node, I.HardwareRegion) for node in nodes(call.body)) for call in calls)
    raise PackingCaptured

  monkeypatch.setattr(packing, "expand_helper", record)
  for compute in (False, True, False):
    with pytest.raises(PackingCaptured):
      CPUBackend().lower(packed_template(compute).ir, (), target="AVX512")
  assert observed == [False, True, False]


def test_native_online_amx_pack_calls_preserve_layouts_aliases_padding_and_canaries(tmp_path):
  import numpy as np
  from dsl_test_utils import amx_available_for_test, make_compiler

  if not amx_available_for_test():
    pytest.skip("requires native AMX compute and online packing")
  pytest.importorskip("ml_dtypes")
  m, n, k, capacity = (T.dynamic(lower=0) for _ in range(4))
  rs, cs = T.dynamic(), T.dynamic()

  @T.template
  def kernel(
    A: T.Tensor((m, k), dtype="bfloat16"),
    B: T.Tensor((k, n), (rs, cs), dtype="bfloat16"),
    P: T.Tensor((capacity,), dtype="bfloat16", access="output"),
    C: T.Tensor((m, n), dtype="float32", access="output"),
  ):
    T.pack(B, P, side="B", format="amx")
    packed = T.packed(P, (k, n), side="B", format="amx")
    T.dot((T.load(A), packed), C, implementation="amx")

  compiled = T.compile(
    kernel,
    compiler=make_compiler(jobs=2, cache_dir=tmp_path / "cache", build_dir=tmp_path / "build"),
  )
  dtype = np.dtype("bfloat16")
  for reduction, columns, layout in ((32, 32, "row"), (32, 32, "column"), (33, 19, "reverse"), (0, 19, "row")):
    for alias in (False, True):
      rows = 17
      chunks = (reduction + 31) // 32
      count = ((columns + 15) // 16) * chunks * 512
      storage = np.full(count + 16, 0x7FC1, dtype=np.uint16).view(dtype)
      values = (np.arange(reduction * columns).reshape(reduction, columns) % 9 - 4).astype(dtype)
      if alias:
        b = storage[: values.size].reshape(values.shape)
        b[...] = values
      else:
        b = values.copy()
      if layout == "column":
        b = b.T.copy().T if not alias else b.reshape(columns, reduction).T
      elif layout == "reverse":
        b = b[:, ::-1]
      before_b = b.copy()
      before_extra = storage.view(np.uint16)[count:].copy()
      a = (np.arange(rows * reduction).reshape(rows, reduction) % 7 - 3).astype(dtype)
      before_a = a.copy()
      output = np.full(rows * columns + 8, 0x7FC12345, dtype=np.uint32).view(np.float32)
      c = output[: rows * columns].reshape(rows, columns)
      expected = np.zeros(count, dtype=np.uint16)
      for kk in range(reduction):
        for column in range(columns):
          base = (column // 16 * chunks + kk // 32) * 512
          offset = kk % 32 // 2 * 32 + column % 16 * 2 + kk % 2
          expected[base + offset] = before_b.view(np.uint16)[kk, column]
      compiled.operator.native(a, b, storage, c, **compiled._core.specialization_values)
      np.testing.assert_array_equal(c, before_a.astype(np.float64) @ before_b.astype(np.float64))
      np.testing.assert_array_equal(storage.view(np.uint16)[:count], expected)
      np.testing.assert_array_equal(storage.view(np.uint16)[count:], before_extra)
      np.testing.assert_array_equal(output.view(np.uint32)[rows * columns :], 0x7FC12345)
      np.testing.assert_array_equal(a.view(np.uint16), before_a.view(np.uint16))
      if not alias:
        np.testing.assert_array_equal(b.view(np.uint16), before_b.view(np.uint16))
