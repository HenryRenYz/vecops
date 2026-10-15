# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Resident AMX B-layout evidence excludes fused and transformed policies."""

import pytest
import vecops.language as T
from vecops.language._compiler.backends.cpu import recipes
from vecops.language._compiler.backends.cpu.backend import CPUBackend


def template(
  *,
  a_memory="bfloat16",
  b_memory="bfloat16",
  seed=False,
  fused=False,
  transformed=False,
  BM=None,
  BN=None,
  BK=None,
  packing="auto",
  implementation="auto",
):
  m, n, k = (T.dynamic(lower=1) for _ in range(3))

  @T.template
  def kernel(
    A: T.Tensor((m, k), dtype=a_memory),
    B: T.Tensor((k, n), (1, k), b_memory),
    Prior: T.Tensor((1, n), dtype="float32"),
    C: T.Tensor((m, n), dtype="float32", access="output"),
  ):
    a, b = T.load(A, dtype="bfloat16"), T.load(B, dtype="bfloat16")
    if transformed:
      b = T.cast(b * 2 + 1, "bfloat16")
    product = T.matmul(
      (a, b),
      C,
      acc=T.load(Prior) if seed else None,
      BM=BM,
      BN=BN,
      BK=BK,
      packing=packing,
      implementation=implementation,
    )
    if fused:
      T.store(C, T.max(product, 0))

  return kernel


def captured(monkeypatch, **options):
  result = {}

  class Captured(Exception):
    pass

  def capture(helper, *args, **kwargs):
    assert helper.function.__name__ == "blocked_matrix"
    result.update(kwargs["keyword_arguments"])
    raise Captured

  monkeypatch.setattr(recipes, "expand_helper", capture)
  with pytest.raises(Captured):
    CPUBackend().lower(template(**options).ir, (), target="AVX512")
  return result


def test_raw_unconfigured_auto_bf16_passes_resident_b_layout(monkeypatch):
  options = captured(monkeypatch)
  assert options["raw_b"].dtype == "bfloat16"
  assert options["raw_b"] is options["cache_raw_b"]


@pytest.mark.parametrize(
  "options",
  (
    {"a_memory": "float32"},
    {"b_memory": "float32"},
    {"seed": True},
    {"fused": True},
    {"transformed": True},
    {"BM": 16},
    {"BN": 33},
    {"BK": 17},
    {"packing": "none"},
    {"implementation": "amx"},
  ),
)
def test_fused_converted_seed_and_explicit_policy_keep_previous_layout_contract(monkeypatch, options):
  arguments = captured(monkeypatch, **options)
  assert arguments.get("raw_b") is None
