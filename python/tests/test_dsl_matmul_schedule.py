# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Planned domains and bounded operand pipelines, including K=0/tails."""

import re

import numpy as np
import pytest
import torch
import vecops.language as T
from dsl_test_utils import make_compiler
from vecops.language._compiler.backends.cpu.backend import CPUBackend


@pytest.fixture(scope="module")
def compiler(tmp_path_factory):
  root = tmp_path_factory.mktemp("dsl-matrix-schedule")
  return make_compiler(target="Native", jobs=4, build_dir=root / "build", cache_dir=root / "cache")


def _pipeline_template(dtype):
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

  return kernel


def _assert_pipeline_codegen(source, mode, dtype, architecture):
  """Check the selected backend's pipeline, not an x86-specific fallback.

  Vector/SME staging primes and rotates named operand banks. The current
  AMX BF16 path implements the staged request using bounded future prefetches.
  Both preserve the scheduled K recurrence and guard zero/tail addresses.
  """
  if mode == "prefetch":
    assert "__builtin_prefetch" in source
  elif mode == "staged":
    if dtype == "float32" or architecture == "arm":
      assert "operand pipeline prime" in source
    else:
      assert "AMX hints; no ninth tile register" in source
      assert "__builtin_prefetch" in source


def _assert_pipeline_schedule(plans, mode):
  assert plans and all(plan.schedule.pipeline == mode for plan in plans.values())
  for plan in plans.values():
    assert plan.schedule.loop("cache.k").bound == "nonempty"
    assert plan.schedule.loop("vector.k").recurrence.ordering == "sequential"
    assert "cpu-schedule-matmul" in plan.pass_trace
    if mode == "staged":
      for role in ("vector.k", "sme.k"):
        loop = plan.schedule.loop(role)
        assert loop.bound == "lookahead"
        assert tuple(action.name for action in loop.prologue) == ("operand-bank-prime",)
        assert tuple(action.name for action in loop.body) == (
          "next-operand-load",
          "multiply-accumulate",
          "rotate-operand-banks",
        )
        assert tuple(action.name for action in loop.finalization) == ("operand-pipeline-drain",)
      amx = plan.schedule.loop("amx.k")
      assert not amx.prologue and not amx.finalization
      assert amx.body[0].name == "future-panel-prefetch"


@pytest.mark.parametrize("target", ("SME", "AVX512", "generic"))
@pytest.mark.parametrize("mode", ("off", "prefetch", "staged"))
@pytest.mark.parametrize("dtype", ("float32", "bfloat16"))
def test_pipeline_codegen_contract_matches_backend_strategy(target, mode, dtype):
  artifact = CPUBackend().lower(_pipeline_template(dtype).ir, (), target=target, options={"cpu_matmul_pipeline": mode})
  architecture = {"SME": "arm", "AVX512": "x86", "generic": "generic"}[target]
  _assert_pipeline_codegen(artifact.source, mode, dtype, architecture)
  _assert_pipeline_schedule(artifact.backend_ir.matmuls, mode)


@pytest.mark.parametrize("mode", ["off", "prefetch", "staged"])
@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_native_pipeline_semantics(compiler, mode, dtype):
  kernel = _pipeline_template(dtype)
  compiled = T.compile(kernel, compiler=compiler, pass_options={"cpu_matmul_pipeline": mode})
  _assert_pipeline_schedule(compiled.backend_ir.matmuls, mode)
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
  environment = compiled.vector_ir.manager.options["_compiler_environment"]
  _assert_pipeline_codegen(source, mode, dtype, environment.codegen_target.architecture)


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
