# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Experimental AMX stream hint changes only physical B tile loads."""

import platform

import numpy as np
import pytest
import vecops.language as T
from dsl_test_utils import make_compiler
from vecops.language._compiler import ir as I
from vecops.language._compiler.backends.cpu.backend import CPUBackend
from vecops.language._compiler.ir.traversal import children
from vecops.language._library.amx_matmul import _tiles, streamed_dot_into


def _nodes(block):
  for node in block.statements:
    yield node
    for child in children(node):
      yield from _nodes(child)


@pytest.mark.parametrize("enabled", [False, True])
def test_stream_hint_never_changes_a_or_accumulator_loads(enabled):
  @T.template
  def kernel(
    A: T.Tensor((3, 2, 16, 32), dtype="bfloat16"),
    B: T.Tensor((3, 2, 16, 32), dtype="bfloat16"),
    C: T.Tensor((48, 48), dtype="float32", access="output"),
  ):
    _tiles(A, B, C, initial=True, kr=32, stream_b=enabled)

  artifact = CPUBackend().lower(kernel.ir, (), target="AVX512")
  calls = [node for node in _nodes(artifact.kernel.body) if isinstance(node, I.ArchitectureCall)]
  streamed = [call for call in calls if call.operation == "amx.stream_load"]
  assert bool(streamed) == enabled
  assert all(call.view.storage == "B" for call in streamed)
  if enabled:
    assert {call.tile for call in streamed} == {2, 6, 7}
    assert "stream_load" in artifact.source
  assert any(call.operation == "amx.load" and call.view.storage == "A" for call in calls)
  assert any(call.operation == "amx.load" and call.view.storage == "C" for call in calls)


def template(dtype):
  m, n, k = (T.dynamic(lower=0) for _ in range(3))
  ad, bd, cd = ("bfloat16", "bfloat16", "float32") if dtype == "bf16" else ("uint8", "int8", "int32")

  @T.template
  def kernel(
    A: T.Tensor((m, k), (T.dynamic(), T.dynamic()), ad),
    B: T.Tensor((k, n), (T.dynamic(), T.dynamic()), bd),
    Prior: T.Tensor((1, n), (T.dynamic(), T.dynamic()), cd),
    C: T.Tensor((m, n), (T.dynamic(), T.dynamic()), cd, access="output"),
  ):
    streamed_dot_into((T.load(A), T.load(B)), C, acc=T.load(Prior), acc_dtype=cd)

  return kernel


@pytest.fixture(scope="module", params=["bf16", "byte"])
def native(request, tmp_path_factory):
  if platform.machine().lower() not in {"x86_64", "amd64"}:
    pytest.skip("AMX stream loads require an x86 AMX host")
  root = tmp_path_factory.mktemp("amx-stream-b")
  return T.compile(
    template(request.param), compiler=make_compiler(jobs=2, cache_dir=root / "cache", build_dir=root / "build")
  ), request.param


@pytest.mark.parametrize("m,n,k", [(32, 32, 32), (48, 48, 65), (33, 49, 193), (33, 49, 0)])
def test_native_stream_b_preserves_fringe_odd_k_and_seed_alias(native, m, n, k):
  compiled, dtype = native
  bf16 = pytest.importorskip("ml_dtypes").bfloat16
  ad, bd, cd = (bf16, bf16, np.float32) if dtype == "bf16" else (np.uint8, np.int8, np.int32)
  a = np.full((m, 2 * k), np.nan if dtype == "bf16" else 255, dtype=ad)[:, ::-2]
  b = np.full((2 * k, n), np.nan if dtype == "bf16" else 127, dtype=bd)[::-2]
  a[:] = (np.arange(m * k).reshape(m, k) % 11) / (8 if dtype == "bf16" else 1)
  b[:] = (np.arange(k * n).reshape(k, n) % 13 - 6) / (8 if dtype == "bf16" else 1)
  poison = np.nan if dtype == "bf16" else 12345
  backing = np.full((m + 2, n + 2), poison, dtype=cd)
  output = backing[1:-1, 1:-1]
  prior = output[:1]
  prior[:] = np.arange(n) / 8 if dtype == "bf16" else 2147483500 + np.arange(n)
  compute = np.float64 if dtype == "bf16" else np.int64
  oracle = a.astype(compute) @ b.astype(compute) + prior.copy()
  compiled.operator.native(a, b, prior, output, **compiled._core.specialization_values)
  np.testing.assert_array_equal(output, oracle.astype(cd))
  np.testing.assert_array_equal(backing[[0, -1]], poison)
  np.testing.assert_array_equal(backing[:, [0, -1]], poison)
