# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Typed ordinary-SVE segment layout and instruction-source regressions."""

import os

import numpy as np
import pytest
import vecops.language as T
from dsl_test_utils import make_compiler
from vecops._arm_vector_state import get_vector_state
from vecops.language import sve
from vecops.language._compiler import ir as I
from vecops.language._compiler.backends.cpu.backend import CPUBackend
from vecops.language._debug.interpreter import Interpreter


@T.template
def primitive_kernel(A: T.Tensor((128,), dtype="bfloat16"), B: T.Tensor((128,), dtype="bfloat16"),
                     C: T.Tensor((128,), dtype="float32", access="output")):
  count = T.lanes(T.Vector("bfloat16"))
  index = T.arange(count)
  a = T.as_vector(T.gather(A, index), T.Vector("bfloat16"))
  b = T.as_vector(T.gather(B, index), T.Vector("bfloat16"))
  joined = sve.join_k_groups(a, b)
  pin = sve.broadcast_segment(joined, 0)
  acc = T.as_vector(T.zeros((T.lanes(T.Vector("float32")),), dtype="float32"), T.Vector("float32"))
  result = sve.mmla(pin, joined, acc)
  T.store(C[:T.lanes(T.Vector("float32"))], sve.row_pair(result, 1))


def test_ordinary_sve_segment_primitives_emit_public_calls():
  source = CPUBackend().lower(primitive_kernel.ir, (), target="SME").source
  for operation in ("join_k_groups", "broadcast_segment<0>", "mmla", "row_pair<1>"):
    assert f"vecops::vec::sve::{operation}" in source
  assert "SVE segment index exceeds ordinary VL" in source


def test_ordinary_sve_segment_primitives_reject_x86():
  with pytest.raises(TypeError, match="SVE compilation target"):
    CPUBackend().lower(primitive_kernel.ir, (), target="AVX512")


@pytest.mark.parametrize("budget", [0, 12])
def test_sve_word_contract_survives_register_and_scratch_consumers(budget):
  source = CPUBackend().lower(primitive_kernel.ir, (), target="SME",
                              options={"cpu_state_register_budget": budget}).source
  assert "vecops::vec::lanes" not in source
  assert "sve::mmla(vecops::vec::ScalableTag<float, 0>{}" in source
  if budget == 0:
    assert "sve_projected" in source


@pytest.mark.skipif(os.uname().machine.lower() not in {"aarch64", "arm64"}, reason="requires native ordinary SVE BF16")
@pytest.mark.parametrize("budget", [0, 12])
def test_native_sve_word_consumers_have_identical_segment_results(budget, tmp_path):
  pytest.importorskip("ml_dtypes")
  dt = np.dtype("bfloat16")
  lanes = get_vector_state().sve_bytes // 2
  if lanes > 128:
    pytest.skip("fixture stores 128 BF16 lanes")
  a = (np.arange(128) % 7).astype(dt)
  b = (np.arange(128) % 5).astype(dt)
  joined = np.empty(lanes, dtype=dt)
  words = joined.view(np.uint32)
  words[0::2] = a[:lanes].view(np.uint32)[:lanes // 4]
  words[1::2] = b[:lanes].view(np.uint32)[:lanes // 4]
  rhs = joined.astype(np.float32).reshape(-1, 2, 4)
  lhs = np.broadcast_to(rhs[:1], rhs.shape)
  sums = np.matmul(lhs, np.transpose(rhs, (0, 2, 1)))
  expected = np.tile(sums[:, 1, :].reshape(-1), 2)
  output = np.full(128, -1234, np.float32)
  compiler = make_compiler(target="Native", jobs=2, cache_dir=tmp_path / "cache", build_dir=tmp_path / "build")
  compiled = T.compile(primitive_kernel, compiler=compiler, pass_options={"cpu_state_register_budget": budget})
  compiled.operator.native(a, b, output, **compiled._core.specialization_values)
  np.testing.assert_array_equal(output[:expected.size], expected)
  np.testing.assert_array_equal(output[expected.size:], -1234)


def test_debug_segment_layout_and_byte_modular_accumulation():
  kernel = type("Kernel", (), {"bindings": {}})()
  interpreter = Interpreter(kernel, {})
  def value(array, dtype):
    return I.Expr(I.TileType(dtype, (I.literal(array.size),)), "literal", data=array)
  a = np.arange(16, dtype=np.uint8)
  b = np.arange(16, dtype=np.uint8)[::-1].copy()
  acc = np.full(4, 0x7FFFFFFF, dtype=np.int32)
  expression = I.Expr(I.TileType("int32", (I.literal(4),)), "sve.mmla",
                      (value(a, "uint8"), value(b, "uint8"), value(acc, "int32")))
  expected = (a.astype(np.int64).reshape(2, 8) @ b.astype(np.int64).reshape(2, 8).T).reshape(-1) + acc
  np.testing.assert_array_equal(interpreter.value(expression), (expected & 0xFFFFFFFF).astype(np.uint32).view(np.int32))
  rows = I.Expr(expression.type, "sve.row_pair", (expression,), 1)
  result = interpreter.value(expression)
  np.testing.assert_array_equal(interpreter.value(rows), np.tile(result[2:], 2))


@T.template
def index_kernel(C: T.Tensor((128,), dtype="uint16", access="output")):
  index = sve.iota("uint16")
  T.store(C[:T.lanes(T.Vector("uint16"))], index)


def test_sve_iota_is_one_typed_native_word():
  source = CPUBackend().lower(index_kernel.ir, (), target="SVE").source
  assert "vecops::vec::iota(" in source
  assert "static_cast<std::uint16_t>" in source
  interpreter = Interpreter(type("Kernel", (), {"bindings": {}})(), {})
  for dtype in ("uint8", "uint16", "uint32", "uint64"):
    expression = I.Expr(I.VectorType(dtype, (I.literal(64 // np.dtype(dtype).itemsize),),
                                   tag=I.VectorTag(basis_dtype=dtype)), "sve.iota")
    np.testing.assert_array_equal(interpreter.value(expression),
                                  np.arange(64 // np.dtype(dtype).itemsize, dtype=dtype))
