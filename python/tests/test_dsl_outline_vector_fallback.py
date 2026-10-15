# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Only ordinary native-dispatch fallbacks may opt into private calls."""

import pytest
import vecops.language as T
from vecops.language._compiler import ir as I
from vecops.language._compiler.backends.cpu.backend import CPUBackend
from vecops.language._compiler.ir.traversal import children


def template(tiled=False):
  m, n, k = T.Dynamic(name="m"), T.Dynamic(name="n"), T.Dynamic(name="k")

  @T.template
  def kernel(
    A: T.Tensor((m, k), (T.dynamic(), T.dynamic()), "bfloat16"),
    B: T.Tensor((k, n), (T.dynamic(), T.dynamic()), "bfloat16"),
    C: T.Tensor((m, n), (T.dynamic(), T.dynamic()), "float32", access="output"),
  ):
    if tiled:
      value = T.matmul((A, B), C)
    else:
      value = T.dot((A, B), C)
    T.store(C, value)

  return kernel


def nodes(block):
  for node in block.statements:
    yield node
    for child in children(node):
      yield from nodes(child)


@pytest.mark.parametrize("tiled", [False, True])
@pytest.mark.parametrize("outlined", [False, True])
def test_only_vector_fallback_has_an_explicit_private_call(tiled, outlined):
  artifact = CPUBackend().lower(
    template(tiled).ir, (), target="AVX512", options={"cpu_outline_vector_fallback": outlined}
  )
  calls = [node for node in nodes(artifact.kernel.body) if isinstance(node, I.ExitRegion) and node.outlined]
  assert bool(calls) == outlined
  assert any(isinstance(node, I.HardwareRegion) and node.kind == "amx" for node in nodes(artifact.kernel.body))
  if outlined:
    assert all(not any(isinstance(node, I.HardwareRegion) for node in nodes(call.body)) for call in calls)
    assert "VECOPS_NOINLINE" in artifact.source


@pytest.mark.parametrize("value", [1, "true", None])
def test_outline_option_rejects_non_boolean(value):
  with pytest.raises(TypeError, match="cpu_outline_vector_fallback must be bool"):
    CPUBackend().lower(template().ir, (), target="AVX512", options={"cpu_outline_vector_fallback": value})


def numeric_template():
  m, n, k = T.Dynamic(name="m"), T.Dynamic(name="n"), T.Dynamic(name="k")

  @T.template
  def kernel(
    A: T.Tensor((m, k), (T.dynamic(), T.dynamic()), "bfloat16"),
    B: T.Tensor((k, n), (T.dynamic(), T.dynamic()), "bfloat16"),
    Prior: T.Tensor((1, n), (T.dynamic(), T.dynamic()), "float32"),
    C: T.Tensor((m, n), (T.dynamic(), T.dynamic()), "float32", access="output"),
  ):
    result = T.dot((A, B), C, acc=T.load(Prior))
    T.store(C, result * 2 + T.indices((m, n), axis=0) + T.indices((m, n), axis=1))

  return kernel


@pytest.fixture(scope="module")
def native(tmp_path_factory):
  import platform

  import vecops
  from dsl_test_utils import make_compiler

  if platform.machine().lower() not in {"x86_64", "amd64"}:
    pytest.skip("local AMX dispatch fallback validation")
  root = tmp_path_factory.mktemp("outlined-vector-fallback")
  compiled = T.compile(
    numeric_template(),
    compiler=make_compiler(jobs=2, cache_dir=root / "cache", build_dir=root / "build"),
    pass_options={"cpu_outline_vector_fallback": True},
  )
  args = (
    vecops.TensorMeta((1, 65), dtype="bfloat16"),
    vecops.TensorMeta((65, 8), dtype="bfloat16"),
    vecops.TensorMeta((1, 8), dtype="float32"),
    vecops.TensorMeta((1, 8), dtype="float32", access=vecops.TensorAccess.output),
  )
  with vecops.collect_compile_requests() as collector:
    compiled.operator(*args, **compiled._core.specialization_values)
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
  ],
)
def test_native_outlined_fallback_preserves_tail_seed_epilogue_and_alias(native, m, n, k, alias):
  import numpy as np

  bf16 = pytest.importorskip("ml_dtypes").bfloat16
  a = np.empty((m, k), dtype=bf16)
  b = np.empty((k, n), dtype=bf16)
  c = np.empty((m, n), dtype=np.float32)
  backing = None
  if alias in {"A", "B"}:
    shape = (m, k) if alias == "A" else (k, n)
    backing = np.full(max(m * n * 4, shape[0] * shape[1] * 2) + 64, 0xA5, dtype=np.uint8)
    source = np.ndarray(shape, dtype=bf16, buffer=backing)
    c = np.ndarray((m, n), dtype=np.float32, buffer=backing)
    if alias == "A":
      a = source
    else:
      b = source
  a[:] = (np.arange(m * k).reshape(m, k) % 7 - 3) / 8
  b[:] = (np.arange(k * n).reshape(k, n) % 9 - 4) / 8
  prior = c[:1]
  prior[:] = np.arange(n) / 8
  expected = (
    (a.astype(np.float64) @ b.astype(np.float64) + prior.copy()) * 2 + np.arange(m)[:, None] + np.arange(n)[None, :]
  )
  before = backing.copy() if backing is not None else None
  native.operator.native(a, b, prior, c, **native._core.specialization_values)
  np.testing.assert_allclose(c, expected.astype(np.float32), rtol=2e-6, atol=1e-7)
  if backing is not None:
    np.testing.assert_array_equal(backing[m * n * 4 :], before[m * n * 4 :])


@pytest.mark.parametrize("target,expected", [("SME", True), ("AVX512", False)])
def test_default_outline_policy_is_target_specific(target, expected):
  artifact = CPUBackend().lower(template().ir, (), target=target)
  calls = [node for node in nodes(artifact.kernel.body) if isinstance(node, I.ExitRegion) and node.outlined]
  assert bool(calls) == expected
