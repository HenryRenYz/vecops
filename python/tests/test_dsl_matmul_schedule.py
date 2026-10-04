"""Planned domains and bounded operand pipelines, including K=0/tails."""

import re

import numpy as np
import pytest
import torch
import vecops
import vecops.language as T


@pytest.fixture(scope="module")
def compiler(tmp_path_factory):
  root = tmp_path_factory.mktemp("dsl-matrix-schedule")
  return vecops.Compiler(
    target="Native", cc="gcc", cxx="g++", jobs=4, build_dir=root / "build", cache_dir=root / "cache"
  )


@pytest.mark.parametrize("mode", ["off", "prefetch", "staged"])
@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_native_pipeline_semantics(compiler, mode, dtype):
  m, n, k = (T.dynamic(lower=0) for _ in range(3))
  implementation = "vector" if dtype == "float32" else "auto"

  @T.func
  def affine(x):
    return x * 2 + 1

  @T.template
  def kernel(
    A: T.Tensor((m, k), dtype=dtype),
    B: T.Tensor((k, n), dtype=dtype),
    D: T.Tensor((1, n), dtype="float32"),
    C: T.Tensor((m, n), dtype="float32", access="output"),
  ):
    value = T.dot(
      (T.cast(affine(T.load(A, dtype="float32")), dtype), T.load(B)), C, acc=T.load(D), implementation=implementation
    )
    T.store(C, T.max(value * 0.5 + 1, 0))

  compiled = T.compile(kernel, compiler=compiler, pass_options={"cpu_matmul_pipeline": mode})
  plans = compiled.backend_ir.matmuls
  assert plans and all(p.schedule.pipeline == mode for p in plans.values())
  for p in plans.values():
    assert p.schedule.loop("cache.k").bound == "nonempty"
    assert p.schedule.loop("vector.k").recurrence.ordering == "sequential"
    assert "cpu-schedule-matmul" in p.pass_trace
  rng = torch.Generator().manual_seed(38)
  for M, N, K in ((0, 3, 5), (3, 0, 5), (3, 5, 0), (1, 7, 1), (7, 1, 3), (5, 19, 3), (17, 33, 65), (32, 32, 64)):
    a = torch.empty_strided((M, K), (K, 1), dtype=getattr(torch, dtype))
    b = torch.empty_strided((K, N), (N, 1), dtype=getattr(torch, dtype))
    d = torch.empty_strided((1, N), (N, 1))
    c = torch.empty_strided((M, N), (N, 1))
    a.copy_(torch.randn((M, K), generator=rng))
    b.copy_(torch.randn((K, N), generator=rng))
    d.copy_(torch.randn((1, N), generator=rng))
    expected = torch.clamp(((a.float() * 2 + 1).to(a.dtype).float() @ b.float() + d) * 0.5 + 1, min=0)
    compiled(a, b, d, c)
    torch.testing.assert_close(c, expected, rtol=3e-5, atol=3e-5)
  source = compiled.generated_source
  assert not re.search(r"(?:Vec|Mask)<[^;\n]+>\s+\w+\s*\[", source)
  if mode == "staged" and dtype == "float32":
    assert "operand pipeline prime" in source
  if mode == "prefetch" or mode == "staged" and dtype == "bfloat16":
    assert "__builtin_prefetch" in source


@pytest.mark.parametrize("mode", ["prefetch", "staged"])
def test_cache_pipeline_phased_fusion(compiler, mode):
  m, n, k = 17, 19, 65

  @T.template
  def kernel(
    A: T.Tensor((m, k), dtype="float32"),
    B: T.Tensor((k, n), dtype="float32"),
    C: T.Tensor((m, n), dtype="float32", access="inout"),
  ):
    result = T.matmul((A, B), C, acc=T.load(C), BM=8, BN=16, BK=16, implementation="vector")
    T.store(C, T.max(result * 0.5, 0))

  compiled = T.compile(kernel, compiler=compiler, pass_options={"cpu_matmul_pipeline": mode})
  rng = np.random.default_rng(39)
  a = rng.normal(size=(m, k)).astype(np.float32)
  b = rng.normal(size=(k, n)).astype(np.float32)
  c = rng.normal(size=(m, n)).astype(np.float32)
  expected = np.maximum((a @ b + c) * 0.5, 0)
  compiled(a, b, c)
  np.testing.assert_allclose(c, expected, rtol=2e-5, atol=2e-5)
