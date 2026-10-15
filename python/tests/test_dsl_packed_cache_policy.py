# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Automatic native fallbacks preserve the static whole-packed policy."""

import pytest
import vecops.language as T
from vecops.language._compiler import ir as I
from vecops.language._compiler.backends.cpu import recipes
from vecops.language._compiler.backends.cpu.backend import CPUBackend
from vecops.language._compiler.ir.traversal import nodes


def template(format_, *, configured=False, packing="auto", pair=True):
  m, n, k = (T.dynamic(lower=0) for _ in range(3))

  @T.template
  def kernel(
    AP: T.Tensor((4096,), dtype="bfloat16"),
    BP: T.Tensor((4096,), dtype="bfloat16"),
    B: T.Tensor((k, n), dtype="bfloat16"),
    C: T.Tensor((m, n), dtype="float32", access="output"),
  ):
    a = T.packed(AP, (m, k), side="A", format=format_)
    b = T.packed(BP, (k, n), side="B", format=format_) if pair else T.load(B)
    T.matmul((a, b), C, BM=4 if configured else None, packing=packing)

  return kernel


@pytest.mark.parametrize("target,format_", (("SME", "sme"), ("AVX512", "amx")))
@pytest.mark.parametrize(
  "configured,packing,pair",
  ((False, "auto", True), (True, "auto", True), (False, "none", True), (False, "auto", False)),
)
def test_auto_fallback_preserves_packed_fact_and_explicit_policy(
  monkeypatch, target, format_, configured, packing, pair
):
  fallback_calls = []
  original = recipes.expand_helper

  class ExpansionCaptured(Exception):
    pass

  def record(helper, *args, **kwargs):
    result, body = original(helper, *args, **kwargs)
    if helper.function.__name__ == "blocked_dot_into":
      parameters = kwargs["keyword_arguments"]
      work = [node for node in nodes(I.Block(body)) if isinstance(node, I.Allocate)]
      fallback_calls.append((parameters["packed_pair"], bool(work)))
      if len(fallback_calls) == 2:
        raise ExpansionCaptured
    return result, body

  monkeypatch.setattr(recipes, "expand_helper", record)
  with pytest.raises(ExpansionCaptured):
    CPUBackend().lower(template(format_, configured=configured, packing=packing, pair=pair).ir, (), target=target)
  assert len(fallback_calls) == 2  # Missing compile feature and small M/N.
  whole_problem = pair and not configured and packing == "auto"
  assert fallback_calls == [(pair, not whole_problem)] * 2
