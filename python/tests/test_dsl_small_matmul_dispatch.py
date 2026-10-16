# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Static small products must be pruned before native/cache recipe expansion."""

import pytest
import vecops.language as T
from vecops.language._compiler.backends.cpu import recipes
from vecops.language._compiler.backends.cpu.backend import CPUBackend


def template(*, m=1, n=8, k=64, tiled=True, implementation="auto", configured=False, packing="auto"):
  @T.template
  def kernel(
    A: T.Tensor((m, k), dtype="bfloat16"),
    B: T.Tensor((k, n), (1, k), "bfloat16"),
    C: T.Tensor((m, n), dtype="float32", access="output"),
  ):
    if tiled:
      T.matmul((A, B), C, implementation=implementation, BM=4 if configured else None, packing=packing)
    else:
      T.dot((A, B), C, implementation=implementation)

  return kernel


def expanded(monkeypatch, kernel, target):
  calls = []
  original = recipes.expand_helper

  def record(helper, *args, **kwargs):
    calls.append((helper.function.__name__, helper.function.__module__))
    return original(helper, *args, **kwargs)

  monkeypatch.setattr(recipes, "expand_helper", record)
  artifact = CPUBackend().lower(kernel.ir, (), target=target)
  return calls, artifact


@pytest.mark.parametrize("target", ("SME", "AVX512"))
@pytest.mark.parametrize("tiled", (False, True))
def test_static_small_auto_never_expands_native_or_cache_wrapper(monkeypatch, target, tiled):
  calls, artifact = expanded(monkeypatch, template(tiled=tiled), target)
  assert [name for name, _ in calls] == ["vector_dot_into"]
  assert "staged SME matrix" not in "\n".join(artifact.lowered.kernel.diagnostics)
  assert "staged AMX matrix" not in "\n".join(artifact.lowered.kernel.diagnostics)


@pytest.mark.parametrize("target,implementation", (("SME", "sme"), ("AVX512", "amx")))
def test_explicit_native_keeps_small_instruction_recipe(monkeypatch, target, implementation):
  calls, _ = expanded(monkeypatch, template(tiled=False, implementation=implementation), target)
  assert any(name == "dot_into" and implementation in module for name, module in calls)


def test_explicit_block_geometry_keeps_cache_wrapper(monkeypatch):
  calls, _ = expanded(monkeypatch, template(configured=True), "AVX512")
  assert [name for name, _ in calls] == ["blocked_dot_into"]


def test_one_static_small_extent_prunes_native_with_dynamic_other_extent(monkeypatch):
  calls, _ = expanded(monkeypatch, template(n=T.dynamic(lower=1)), "SME")
  assert [name for name, _ in calls] == ["blocked_dot_into"]


def test_fully_dynamic_shape_keeps_native_and_fallbacks(monkeypatch):
  calls, _ = expanded(monkeypatch, template(m=T.dynamic(lower=1), n=T.dynamic(lower=1)), "SME")
  assert any(name == "blocked_matrix" for name, _ in calls)
  assert sum(name == "blocked_dot_into" for name, _ in calls) == 2


@pytest.mark.parametrize("packing", ("A", "none"))
def test_explicit_packing_keeps_cache_wrapper(monkeypatch, packing):
  calls, _ = expanded(monkeypatch, template(packing=packing), "AVX512")
  assert [name for name, _ in calls] == ["blocked_dot_into"]


@pytest.mark.parametrize("m,n,k", ((0, 8, 64), (1, 0, 64), (1, 8, 0), (1, 8, 4096)))
def test_empty_and_long_reduction_keep_existing_wrapper(monkeypatch, m, n, k):
  calls, _ = expanded(monkeypatch, template(m=m, n=n, k=k), "AVX512")
  assert [name for name, _ in calls] == ["blocked_dot_into"]


@pytest.mark.parametrize(
  "target,tiled,static_m",
  (("AVX512", False, None), ("AVX512", True, None), ("AVX512", False, 32), ("SME", False, None)),
)
@pytest.mark.parametrize("outlined", (False, True))
def test_unknown_spatial_amx_keeps_compact_fallback_with_either_outline(monkeypatch, target, tiled, static_m, outlined):
  choices = []
  original = recipes.expand_helper

  def record(helper, *args, **kwargs):
    if helper.function.__name__ in {"vector_dot_into", "blocked_dot_into"}:
      choices.append(kwargs.get("keyword_arguments", {}).get("widening", True))
    return original(helper, *args, **kwargs)

  monkeypatch.setattr(recipes, "expand_helper", record)
  kernel = template(m=T.dynamic(lower=1) if static_m is None else static_m, n=T.dynamic(lower=1), tiled=tiled)
  artifact = CPUBackend().lower(
    kernel.ir, (), target=target, options={"cpu_outline_vector_fallback": outlined}
  )
  grouped = target == "SME"
  assert choices == [grouped, grouped]
  assert ("vecops::vec::widening_dot" in artifact.source) == grouped


def test_explicit_dynamic_vector_keeps_grouped_dot(monkeypatch):
  _, artifact = expanded(
    monkeypatch,
    template(m=T.dynamic(lower=1), n=T.dynamic(lower=1), implementation="vector"),
    "AVX512",
  )
  assert "vecops::vec::widening_dot" in artifact.source


@pytest.mark.parametrize("outlined", (False, True))
def test_dynamic_sme_byte_conversion_uses_compact_fallback(monkeypatch, outlined):
  choices = []
  original = recipes.expand_helper

  def record(helper, *args, **kwargs):
    if helper.function.__name__ in {"vector_dot_into", "blocked_dot_into"}:
      choices.append(kwargs.get("keyword_arguments", {}).get("widening", True))
    return original(helper, *args, **kwargs)

  monkeypatch.setattr(recipes, "expand_helper", record)
  m, n, k = (T.dynamic(lower=1) for _ in range(3))

  @T.template
  def kernel(A: T.Tensor((m, k), dtype="float32"), B: T.Tensor((k, n), dtype="int8"),
             C: T.Tensor((m, n), dtype="int32", access="output")):
    T.dot((T.cast(T.load(A) * 1.25 + 3, "uint8"), B), C)

  CPUBackend().lower(kernel.ir, (), target="SME", options={"cpu_outline_vector_fallback": outlined})
  assert choices == [False, False]
