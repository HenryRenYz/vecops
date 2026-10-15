# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Algorithm migration contracts and native coverage independent of old printers."""

from pathlib import Path

import numpy as np
import pytest
import vecops.language as T
from dsl_test_utils import assert_no_vec_mask_arrays, assert_source_algorithms_lowered, make_compiler
from vecops.language._compiler import ir as I
from vecops.language._compiler.backends.cpu.backend import CPUBackend
from vecops.language._compiler.ir.traversal import nodes


def transpose_template(dtype, output_dtype=None):
  m, n = T.dynamic(lower=0), T.dynamic(lower=0)

  @T.template
  def kernel(
    A: T.Tensor((m, n), (T.dynamic(), T.dynamic()), dtype),
    C: T.Tensor((n, m), (T.dynamic(), T.dynamic()), output_dtype or dtype, access="output"),
  ):
    T.transpose_copy(A, C, implementation="vector")

  return kernel


@pytest.mark.parametrize("dtype", ["int8", "float16", "float32", "float64"])
def test_register_transpose_has_dsl_traversal_and_no_native_loop_printer(dtype):
  artifact = CPUBackend().lower(transpose_template(dtype).ir, (), target="AVX512")
  assert "staged 1 register transpose traversal" in "\n".join(artifact.kernel.diagnostics)
  assert "dsl_layout_" in artifact.source
  assert artifact.backend_ir.register_programs
  if dtype != "float64":
    assert "local_interleave" in artifact.source
  assert_no_vec_mask_arrays(artifact.source)


def blocked_template(dtype="float32", *, packing="none", seed=True):
  m, n, k = (T.dynamic(lower=0) for _ in range(3))
  acc_dtype = "int32" if dtype in {"int8", "uint8", "int32"} else "float32"

  @T.template
  def kernel(
    A: T.Tensor((m, k), (T.dynamic(), T.dynamic()), dtype),
    B: T.Tensor((k, n), (T.dynamic(), T.dynamic()), dtype),
    D: T.Tensor((1, n), dtype=acc_dtype),
    C: T.Tensor((m, n), (T.dynamic(), T.dynamic()), acc_dtype, access="output"),
  ):
    value = T.library.matmul((A, B), C, acc=T.load(D) if seed else None, BM=8, BN=16, BK=7, packing=packing)
    T.store(C, T.max(value, 0))

  return kernel


@pytest.mark.parametrize("target", ["AVX512", "SME"])
@pytest.mark.parametrize("packing", ["none", "auto", "A", "B", "AB"])
def test_cache_and_packing_control_expand_before_vector_lowering(target, packing):
  artifact = CPUBackend().lower(blocked_template(packing=packing).ir, (), target=target)
  assert not any(
    isinstance(node, I.Pack) or isinstance(node, I.Expr) and node.op == "dot" for node in nodes(artifact.kernel.body)
  )
  assert "dsl_recipe_dot_" in artifact.source
  assert_no_vec_mask_arrays(artifact.source)


@pytest.mark.parametrize("packing", ["none", "auto", "AB"])
@pytest.mark.parametrize("dtype", ["float32", "int8"])
def test_native_source_blocking_preserves_seed_epilogue_tails_and_alias(dtype, packing, tmp_path):
  compiler = make_compiler(jobs=4, cache_dir=tmp_path / "cache", build_dir=tmp_path / "build")
  kernel = T.compile(blocked_template(dtype, packing=packing), compiler=compiler)
  acc_dtype = np.float32 if dtype == "float32" else np.int32
  for m, n, k in [(0, 19, 11), (19, 0, 11), (9, 19, 0), (1, 19, 11), (19, 1, 11), (9, 19, 17)]:
    a = (np.arange(m * k).reshape(m, k) % 7 - 3).astype(dtype)
    b = (np.arange(k * n).reshape(k, n) % 5 - 2).astype(dtype)
    d = np.arange(n, dtype=acc_dtype).reshape(1, n) - 5
    for reverse in (False, True):
      aa, bb = (a[::-1, ::-1], b[::-1, ::-1]) if reverse else (a, b)
      storage = np.full((m + 2, n + 3), -17, dtype=acc_dtype)
      out = storage[1 : m + 1, 1 : n + 1]
      kernel.operator.native(aa, bb, d, out, **kernel._core.specialization_values)
      np.testing.assert_array_equal(out, np.maximum(aa.astype(acc_dtype) @ bb.astype(acc_dtype) + d, 0))
      np.testing.assert_array_equal(storage[[0, -1]], -17)
      np.testing.assert_array_equal(storage[:, [0, -1]], -17)
  if dtype == "float32":
    a = (np.arange(9 * 19).reshape(9, 19) % 7 - 3).astype(dtype)
    b = (np.arange(19 * 19).reshape(19, 19) % 5 - 2).astype(dtype)
    d = np.arange(19, dtype=acc_dtype).reshape(1, 19) - 5
    expected = np.maximum(a.copy() @ b + d, 0)
    kernel.operator.native(a, b, d, a, **kernel._core.specialization_values)
    np.testing.assert_array_equal(a, expected)


@pytest.mark.parametrize(
  "dtype,output_dtype",
  [("int8", "int8"), ("float16", "float32"), ("float32", "float16"), ("float32", "float32"), ("float64", "float64")],
)
def test_native_source_register_transpose_boundaries_and_payload(dtype, output_dtype, tmp_path):
  compiler = make_compiler(jobs=4, cache_dir=tmp_path / "cache", build_dir=tmp_path / "build")
  kernel = T.compile(transpose_template(dtype, output_dtype), compiler=compiler)
  for m, n in [(0, 19), (19, 0), (1, 19), (19, 1), (3, 5), (17, 19), (33, 65), (129, 131)]:
    a = (np.arange(m * n).reshape(m, n) % 29 - 14).astype(dtype)
    for reverse in (False, True):
      source = a[::-1, ::-1] if reverse else a
      c = np.full((n + 2, m + 3), -17, dtype=output_dtype)
      kernel.operator.native(source, c[1 : n + 1, 1 : m + 1], **kernel._core.specialization_values)
      np.testing.assert_array_equal(c[1 : n + 1, 1 : m + 1], source.T.astype(output_dtype))
      np.testing.assert_array_equal(c[[0, -1]], -17)
      np.testing.assert_array_equal(c[:, [0, -1]], -17)


def test_retired_operator_algorithm_printers_are_removed():
  root = Path(__file__).parents[1] / "vecops/language/_compiler/backends/cpu"
  for path in ["matmul.py", "schedule.py", "cpp/matmul.py", "cpp/sme_packing.py", "cpp/matmul_support.py"]:
    assert not (root / path).exists(), path
  transpose = (root / "cpp/transpose.py").read_text()
  assert "def register_transpose_variant" not in transpose
  assert "def register_tile" not in transpose


def test_native_source_packing_uses_saturating_scalar_conversion(tmp_path):
  """Physical padding is zero; scalar pack tails match vector conversion."""
  import numpy as np
  from dsl_test_utils import make_compiler

  @T.template
  def kernel(A: T.Tensor((3, 5), dtype="float32"), C: T.Tensor((3, 5), dtype="int8", access="output")):
    packed = T.pack(T.cast(T.load(A), "int8"), side="A", format="auto")
    T.store(C, packed)

  compiler = make_compiler(jobs=4, cache_dir=tmp_path / "cache", build_dir=tmp_path / "build")
  compiled = T.compile(kernel, compiler=compiler)
  a = np.array(
    [-300, -129, -128, -1, 0, 127, 128, 300, np.nan, np.inf, -np.inf, 2, -2, 17, -17], dtype=np.float32
  ).reshape(3, 5)
  output = np.empty((3, 5), dtype=np.int8)
  compiled.operator.native(a, output)
  expected = np.clip(np.nan_to_num(a, nan=0, posinf=127, neginf=-128), -128, 127).astype(np.int8)
  np.testing.assert_array_equal(output, expected)
  assert_source_algorithms_lowered(compiled.backend_ir)


@T.func
def _captured_row(a, row, dtype):
  if row < a.shape[0]:
    result = T.load(a[row, :], vector=T.Vector(dtype, N=4))
  else:
    result = T.zeros(T.Vector(dtype, N=4))
  return result


@pytest.mark.parametrize("dtype", ["float16", "float32"])
def test_native_unrolled_helper_captures_do_not_share_one_mutable_slot(dtype, tmp_path):
  @T.template
  def kernel(A: T.Tensor((3, 4), dtype=dtype), C: T.Tensor((12,), dtype=dtype, access="output")):
    for iteration in T.serial(1):
      pieces = ()
      for row in T.unroll(3):
        part = _captured_row(A, row, dtype)
        pieces = pieces + (part,)
      for row in T.unroll(3):
        T.store(C[row * 4 : (row + 1) * 4], pieces[row])

  compiler = make_compiler(jobs=4, cache_dir=tmp_path / "cache", build_dir=tmp_path / "build")
  compiled = T.compile(kernel, compiler=compiler)
  a = np.arange(12, dtype=dtype).reshape(3, 4)
  output = np.empty(12, dtype=dtype)
  compiled.operator.native(a, output)
  np.testing.assert_array_equal(output, a.reshape(-1))
