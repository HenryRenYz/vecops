# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Automatic narrow native selection respects packed cost and raw GEMV wins."""

import numpy as np
import pytest
import vecops.language as T
from dsl_test_utils import make_compiler
from vecops.language._compiler.backends.cpu.backend import CPUBackend


def template(dtype="bf16", packed=False, m=8, n=64, k=128, implementation="auto", packed_a=False):
  ad, bd, cd = ("bfloat16", "bfloat16", "float32") if dtype == "bf16" else ("uint8", "int8", "int32")

  @T.template
  def kernel(A: T.Tensor((m, k), dtype=ad), B: T.Tensor((k, n), dtype=bd),
             D: T.Tensor((1, n), dtype=cd), C: T.Tensor((m, n), dtype=cd, access="output")):
    a, b = T.load(A), T.load(B)
    if packed_a:
      a = T.pack(a, side="A")
    if packed:
      b = T.pack(b, side="B")
    value = T.dot((a, b), C, acc=T.load(D), implementation=implementation)
    T.store(C, T.max(value * 2 + 1, 0))

  return kernel


@pytest.mark.parametrize("target", ["AVX512", "SME"])
@pytest.mark.parametrize("dtype,packed,m,n,k,native", [
  ("bf16", False, 1, 8, 4097, False),
  ("bf16", True, 1, 256, 256, True),
  ("bf16", True, 15, 15, 127, True),
  ("bf16", True, 128, 1, 1024, True),
  ("byte", False, 8, 128, 128, True),
  ("byte", False, 1, 8, 64, False),
  ("byte", False, 1, 16, 255, False),
  ("byte", False, 1, 16, 256, False),
  ("byte", False, 4, 16, 255, False),
  ("byte", False, 4, 16, 256, True),
  ("byte", False, 4, 16, 512, True),
  ("byte", False, 8, 16, 256, True),
  ("byte", False, 8, 32, 128, True),
  ("byte", False, 4, 32, 256, True),
  ("byte", False, 8, 64, 128, True),
  ("bf16", True, 1, 8, 64, True),
  ("bf16", True, 1, 8, 31, False),
  ("bf16", True, 0, 64, 128, False),
])
def test_small_native_shape_policy(target, dtype, packed, m, n, k, native):
  if target == "SME" and not packed and dtype == "byte":
    native = m >= 16 and n >= 16
  artifact = CPUBackend().lower(template(dtype, packed, m, n, k).ir, (), target=target)
  diagnostics = "\n".join(artifact.lowered.kernel.diagnostics)
  assert ("matrix instruction recipe" in diagnostics) == native


@pytest.fixture(scope="module", params=[("bf16", True, False), ("byte", False, False),
                                       ("byte", True, False), ("bf16", True, True)])
def compiled(request, tmp_path_factory):
  dtype, packed, packed_a = request.param
  m, n, k = (T.dynamic(lower=0) for _ in range(3))
  directory = tmp_path_factory.mktemp("small-native-policy")
  return T.compile(template(dtype, packed, m, n, k, packed_a=packed_a),
                   compiler=make_compiler(jobs=2, cache_dir=directory / "cache", build_dir=directory / "build")), dtype


@pytest.mark.parametrize("m,n,k", [
  (1, 8, 31), (1, 16, 255), (1, 16, 256), (8, 64, 128),
  (4, 16, 256), (4, 16, 512), (8, 16, 256), (8, 32, 128), (4, 32, 256),
  (15, 15, 127), (16, 15, 33), (15, 16, 33), (17, 1, 1024),
  (1, 17, 65), (0, 17, 65), (17, 0, 65), (8, 17, 0),
])
def test_small_native_dynamic_boundaries(compiled, m, n, k):
  compiled, dtype = compiled
  bf16 = pytest.importorskip("ml_dtypes").bfloat16
  ad, bd, cd = (bf16, bf16, np.float32) if dtype == "bf16" else (np.uint8, np.int8, np.int32)
  a = (np.arange(m * k).reshape(m, k) % 7).astype(ad)
  b = (np.arange(k * n).reshape(k, n) % 9 - 4).astype(bd)
  d = np.empty((1, n), dtype=cd)
  d[0] = np.arange(n, dtype=cd)
  c = np.full((m, n), np.nan if dtype == "bf16" else -3456, dtype=cd)
  compiled.operator.native(a, b, d, c, **compiled._core.specialization_values)
  compute = np.float64 if dtype == "bf16" else np.int64
  expected = np.maximum((a.astype(compute) @ b.astype(compute) + d) * 2 + 1, 0)
  np.testing.assert_array_equal(c, expected.astype(cd))


@pytest.mark.parametrize("target", ["AVX512", "SME"])
@pytest.mark.parametrize("m,n,k,native", [(1, 8, 64, True), (15, 15, 127, True), (128, 1, 1024, True)])
def test_paired_packed_small_policy(target, m, n, k, native):
  artifact = CPUBackend().lower(template("bf16", True, m, n, k, packed_a=True).ir, (), target=target)
  assert ("matrix instruction recipe" in "\n".join(artifact.lowered.kernel.diagnostics)) == native


@pytest.mark.parametrize("fresh", [False, True])
def test_small_packed_selection_requires_current_physical_capture(fresh):
  @T.template
  def kernel(A: T.Tensor((1, 64), dtype="bfloat16"), B: T.Tensor((64, 8), dtype="bfloat16"),
             C: T.Tensor((1, 8), dtype="float32", access="output")):
    AP = T.alloc((1024,), dtype="bfloat16")
    BP = T.alloc((1024,), dtype="bfloat16")
    T.pack(A, AP, side="A", format="amx")
    T.pack(B, BP, side="B", format="amx")
    if not fresh:
      a = T.packed(AP, (1, 64), side="A", format="amx")
      b = T.packed(BP, (64, 8), side="B", format="amx")
    for repeat in T.serial(2):
      if fresh:
        a = T.packed(AP, (1, 64), side="A", format="amx")
        b = T.packed(BP, (64, 8), side="B", format="amx")
      T.dot((a, b), C)
      T.compiler_fence()

  artifact = CPUBackend().lower(kernel.ir, (), target="AVX512")
  assert ("matrix instruction recipe" in "\n".join(artifact.lowered.kernel.diagnostics)) == fresh


@pytest.mark.parametrize("target", ["AVX512", "SME"])
def test_static_native_choice_expands_only_one_feature_fallback(monkeypatch, target):
  from vecops.language._compiler.backends.cpu import recipes

  calls = []
  expand = recipes.expand_helper

  def record(helper, *args, **kwargs):
    calls.append(helper.function.__name__)
    return expand(helper, *args, **kwargs)

  monkeypatch.setattr(recipes, "expand_helper", record)
  CPUBackend().lower(template("bf16", True, 1, 256, 256).ir, (), target=target)
  assert calls.count("vector_dot_into") == 1


@pytest.mark.parametrize("target", ["AVX512", "SME"])
@pytest.mark.parametrize("small_axis", [0, 1])
def test_raw_bf16_known_small_axis_prunes_native_symmetrically(target, small_axis):
  shape = [T.dynamic(lower=1), T.dynamic(lower=1)]
  shape[small_axis] = 1
  artifact = CPUBackend().lower(template("bf16", False, *shape, 64).ir, (), target=target)
  assert "matrix instruction recipe" not in "\n".join(artifact.lowered.kernel.diagnostics)
  assert "vecops::vec::widening_dot" in artifact.source


@pytest.mark.parametrize("target", ["AVX512", "SME"])
@pytest.mark.parametrize("dtype,packed,m,n,k,native", [
  ("bf16", False, None, 8, 64, False),
  ("bf16", False, 8, None, 64, False),
  ("bf16", False, None, 16, 64, True),
  ("bf16", False, 16, None, 64, True),
  ("bf16", False, None, None, 64, True),
  ("bf16", False, 16, 16, 64, True),
  ("bf16", False, 8, 8, 64, False),
  ("bf16", True, 1, None, 64, True),
  ("bf16", True, None, 1, 64, True),
  ("bf16", True, 1, None, 31, False),
  ("bf16", True, None, 1, 31, False),
  ("byte", False, None, 8, 256, False),
  ("byte", False, 1, None, 256, False),
  ("byte", False, 4, None, 256, True),
  ("byte", False, None, 16, 256, True),
])
def test_partial_metadata_selects_possible_owner_without_dead_expansion(monkeypatch, target, dtype, packed, m, n, k, native):
  from vecops.language._compiler.backends.cpu import recipes

  if target == "SME" and not packed and dtype == "byte":
    native = (m is None or m >= 16) and (n is None or n >= 16)
  selected = []

  class RecipeSelected(Exception):
    pass

  def record(helper, *args, **kwargs):
    selected.append(helper.function.__module__)
    raise RecipeSelected

  monkeypatch.setattr(recipes, "expand_helper", record)
  with pytest.raises(RecipeSelected):
    CPUBackend().lower(template(dtype, packed,
                               T.dynamic(lower=0) if m is None else m,
                               T.dynamic(lower=0) if n is None else n, k).ir, (), target=target)
  assert any(module.endswith(("amx_matmul", "sme_matmul")) for module in selected) == native


@pytest.mark.parametrize("target,format_", [("AVX512", "amx"), ("SME", "sme")])
@pytest.mark.parametrize("fresh,multiple,native", [(False, False, False), (True, False, True), (True, True, True)])
def test_small_shared_packed_product_current_and_historical(monkeypatch, target, format_, fresh, multiple, native):
  from vecops.language._compiler.backends.cpu import recipes

  @T.template
  def kernel(AP: T.Tensor((1024,), dtype="bfloat16"), BP: T.Tensor((1024,), dtype="bfloat16"),
             C: T.Tensor((1, 8), dtype="float32", access="output"),
             E: T.Tensor((1, 8), dtype="float32", access="output")):
    if not fresh:
      a = T.packed(AP, (1, 64), side="A", format=format_)
      b = T.packed(BP, (64, 8), side="B", format=format_)
    for repeat in T.serial(2):
      if fresh:
        a = T.packed(AP, (1, 64), side="A", format=format_)
        b = T.packed(BP, (64, 8), side="B", format=format_)
      value = T.dot((a, b), C)
      if multiple:
        # The product is captured once for sibling output sinks; the second
        # consumer reads its result rather than rereading physical operands.
        T.store(E, value)

  selected = []

  class RecipeSelected(Exception):
    pass

  def record(helper, *args, **kwargs):
    selected.append(helper.function.__module__)
    raise RecipeSelected

  monkeypatch.setattr(recipes, "expand_helper", record)
  with pytest.raises(RecipeSelected):
    CPUBackend().lower(kernel.ir, (), target=target)
  assert any(module.endswith(("amx_matmul", "sme_matmul")) for module in selected) == native
