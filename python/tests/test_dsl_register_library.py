# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Library permutation algorithms and physical-register legalization."""

import re

import numpy as np
import pytest
import vecops.language as T
from dsl_test_utils import make_compiler
from vecops.language._compiler import ir as I
from vecops.language._compiler.backends.cpu.backend import CPUBackend
from vecops.language._compiler.backends.cpu.registers import register_transpose, x86_dword_transpose
from vecops.language._compiler.frontend.helpers import expand_register_helper
from vecops.language._compiler.operations import snapshot
from vecops.language._debug.interpreter import Interpreter


@pytest.mark.parametrize("rows", [1, 2, 4, 8, 16])
@pytest.mark.parametrize("element_bytes", [1, 2, 4, 8])
@pytest.mark.parametrize("ratio", [1, 2])
def test_library_register_transpose(rows, element_bytes, ratio):
  columns = rows * ratio
  data = np.arange(rows * columns).astype(f"uint{element_bytes * 8}").reshape(rows, columns)
  program = register_transpose(rows, columns, element_bytes)
  interpreter = Interpreter(I.KernelIR(I.Block(), {}, {}), {"__register_inputs": tuple(data)})
  output = np.stack([interpreter.value(value) for value in program.outputs])
  np.testing.assert_array_equal(output, data.T)
  assert all(isinstance(value, I.Expr) for value in program.operations)


def test_library_512_dword_transpose():
  data = np.arange(256, dtype=np.uint32).reshape(16, 16)
  program = x86_dword_transpose()
  interpreter = Interpreter(I.KernelIR(I.Block(), {}, {}), {"__register_inputs": tuple(data)})
  np.testing.assert_array_equal(np.stack([interpreter.value(value) for value in program.outputs]), data.T)


def test_register_bitcast_types_are_not_retagged():
  program = register_transpose(8, 16, 4, dtype="float32")
  assert all(value.type.dtype == "float32" for value in program.inputs)
  assert any(
    value.op == "bitcast" and value.type.dtype == "uint32" and value.args[0].type.dtype == "float32"
    for value in program.operations
  )
  assert all(value.type.dtype == "float32" for value in program.outputs)


@pytest.mark.parametrize("vl", [16, 32, 64, 128])
def test_sve_public_pack_placement_uses_actual_vl(vl):
  from vecops._arm_vector_state import VectorState

  @T.template
  def kernel(A: T.Tensor((4, 16), dtype="uint8"), C: T.Tensor((64,), dtype="uint8", access="output")):
    values = ()
    for row in T.unroll(4):
      values = values + (T.load(A[row, :], vector=T.Vector("uint8", N=16)),)
    T.store(C, T.library.pack_interleave(values)[0])

  artifact = CPUBackend().lower(kernel.ir, (), target="SME", options={"_vector_state": VectorState(vl, 64)})
  if vl >= 64:
    assert artifact.backend_ir.register_programs
    assert "vecops::vec::interleave" in artifact.source
    assert "FixedTag<std::uint8_t, 16>" in artifact.source
    power = 6 - vl.bit_length() + 1
    assert f"ScalableTag<std::uint8_t, {power}>" in artifact.source
  else:
    assert not artifact.backend_ir.register_programs


def test_debug_lower_name_is_preserved():
  assert T.lower is not T.lower_half
  assert callable(T.lower)


@pytest.mark.parametrize("reduce", [False, True])
def test_nonmatrix_kernels_do_not_prepare_matrix_packing_recipes(monkeypatch, reduce):
  from vecops.language._compiler.backends.cpu import matmul

  def unexpected(*args, **kwargs):
    pytest.fail("a nonmatrix kernel staged an unrelated matrix packing algorithm")

  monkeypatch.setattr(matmul, "register_transpose", unexpected)
  monkeypatch.setattr(matmul, "x86_dword_transpose", unexpected)
  extent = 1 if reduce else 16

  @T.template
  def kernel(A: T.Tensor((16,), dtype="float32"), C: T.Tensor((extent,), dtype="float32", access="output")):
    value = T.load(A) * 2 + 1
    if reduce:
      C[0] = T.reduce_add(value)
    else:
      T.store(C, value)

  prepared = CPUBackend().lower(kernel.ir, (), target="AVX512").backend_ir
  assert prepared.packing_permutation is None
  assert prepared.packing_permutation_x86 is None


def test_products_share_one_prepared_packing_network(monkeypatch):
  from vecops.language._compiler.backends.cpu import matmul

  calls = []
  original = matmul.register_transpose

  def tracked(*args, **kwargs):
    calls.append(args)
    return original(*args, **kwargs)

  monkeypatch.setattr(matmul, "register_transpose", tracked)

  @T.template
  def kernel(
    A: T.Tensor((4, 4), dtype="float32"),
    B: T.Tensor((4, 4), dtype="float32"),
    C: T.Tensor((4, 4), dtype="float32", access="output"),
    D: T.Tensor((4, 4), dtype="float32", access="output"),
  ):
    T.dot((T.load(A), T.load(B)), C, implementation="vector")
    T.dot((T.load(B), T.load(A)), D, implementation="vector")

  prepared = CPUBackend().lower(kernel.ir, (), target="AVX512").backend_ir
  assert len(prepared.matmuls) == 2
  assert calls == [(16, 16, 4)]
  assert prepared.packing_permutation is not None
  assert prepared.packing_permutation_x86 is not None


@pytest.mark.parametrize("dtype,count", [("uint8", 4), ("int8", 4), ("float16", 2), ("bfloat16", 2)])
@pytest.mark.parametrize("lanes", [None, 4, 16])
def test_library_grouped_k_interleave(dtype, count, lanes):
  from vecops.language._library.registers import pack_interleave

  descriptor = T.Vector(dtype, N=lanes, pow=0 if lanes is not None else -(count.bit_length() - 1))
  program = expand_register_helper(pack_interleave, (descriptor,) * count, catalog=snapshot())
  actual = lanes or 8
  numpy_dtype = np.dtype(pytest.importorskip("ml_dtypes").bfloat16) if dtype == "bfloat16" else np.dtype(dtype)
  data = tuple((np.arange(actual) + i * 13).astype(numpy_dtype) for i in range(count))
  interpreter = Interpreter(I.KernelIR(I.Block(), {}, {}), {"__register_inputs": data})
  np.testing.assert_array_equal(interpreter.value(program.outputs[0]), np.stack(data, axis=-1).reshape(-1))
  assert (
    all(value.type.tag.N is None for value in program.operations)
    if lanes is None
    else all(value.type.tag.N is not None for value in program.operations)
  )


@pytest.mark.parametrize("dtype,count", [("uint8", 4), ("int8", 4), ("float16", 2), ("bfloat16", 2)])
def test_native_public_pack_interleave(compiler, dtype, count):
  lanes = 16

  @T.template
  def kernel(A: T.Tensor((count, lanes), dtype=dtype), C: T.Tensor((count * lanes,), dtype=dtype, access="output")):
    values = ()
    for group in T.unroll(count):
      values = values + (T.load(A[group, :], vector=T.Vector(dtype, N=lanes)),)
    result = T.library.pack_interleave(values)
    T.store(C, result[0])

  numpy_dtype = np.dtype(pytest.importorskip("ml_dtypes").bfloat16) if dtype == "bfloat16" else np.dtype(dtype)
  a = np.arange(count * lanes).astype(numpy_dtype).reshape(count, lanes)
  c = np.empty(count * lanes, dtype=numpy_dtype)
  compiled = T.compile(kernel, compiler=compiler)
  compiled.operator.native(a, c)
  np.testing.assert_array_equal(c, a.T.reshape(-1))
  assert "vecops::vec::interleave" in compiled.generated_source
  assert "std::array<vecops::vec::" not in compiled.generated_source


@T.func
def _static_sum(n):
  if n == 0:
    return 0
  return n + _static_sum(n - 1)


def test_static_recursion_is_expanded_and_frozen(monkeypatch):
  @T.template
  def kernel(C: T.Tensor((1,), dtype="int64", access="output")):
    T.static_assert(_static_sum(8) == 36)
    C[0] = _static_sum(8)

  # The captured self-reference is lexical, not a lookup of a changed global.
  monkeypatch.setitem(globals(), "_static_sum", T.func(lambda n: 999))
  source = CPUBackend().lower(kernel.ir, ()).source
  assert "std::int64_t{36}" in source
  assert "_static_sum(" not in source


def test_runtime_recursion_is_diagnosed():
  @T.template
  def kernel(C: T.Tensor((1,), dtype="int64", access="output"), n: int):
    C[0] = _static_sum(n)

  with pytest.raises(TypeError, match="recursive T.func calls require compile-time"):
    CPUBackend().lower(kernel.ir, ())


@pytest.fixture(scope="module")
def compiler(tmp_path_factory):
  root = tmp_path_factory.mktemp("dsl-register-library")
  return make_compiler(target="Native", jobs=4, cache_dir=root / "cache", build_dir=root / "build")


@pytest.mark.parametrize("size", [4, 12, 16, 128])
def test_native_register_geometry(compiler, size):
  @T.template
  def kernel(
    A: T.Tensor((size,), dtype="float32"),
    B: T.Tensor((size,), dtype="float32"),
    C: T.Tensor((size,), dtype="float32", access="output"),
  ):
    a = T.load(A, vector=T.Vector("float32", N=size))
    b = T.load(B, vector=T.Vector("float32", N=size))
    bits = T.bitcast(a, "uint16")
    restored = T.bitcast(bits, "float32")
    x = T.local_interleave_lower(restored, b)
    y = T.local_interleave_upper(restored, b)
    T.store(C, T.concat(T.upper_half(x), T.lower_half(y)))

  compiled = T.compile(kernel, compiler=compiler)
  a = np.arange(size, dtype=np.float32) - 3.25
  b = np.arange(size, dtype=np.float32) + 20.5
  c = np.empty_like(a)
  compiled.operator.native(a, b, c)
  lo = np.stack((a.reshape(-1, 4)[:, :2], b.reshape(-1, 4)[:, :2]), axis=-1).reshape(-1)
  hi = np.stack((a.reshape(-1, 4)[:, 2:], b.reshape(-1, 4)[:, 2:]), axis=-1).reshape(-1)
  np.testing.assert_array_equal(c, np.concatenate((lo[size // 2 :], hi[: size // 2])))
  expected = c.copy()
  compiled.operator.native(a, b, a)
  np.testing.assert_array_equal(a, expected)
  assert not re.search(r"(?:Vec|Mask)<[^;\n]+>\s+\w+\s*\[", compiled.generated_source)
  assert "std::array<vecops::vec::" not in compiled.generated_source


def test_register_diagnostics():
  size = T.dynamic(lower=0)

  @T.template
  def incomplete(A: T.Tensor((size,), dtype="float32"), C: T.Tensor((size,), dtype="float32", access="output")):
    T.store(C, T.local_interleave_lower(T.load(A), T.load(A)))

  with pytest.raises(TypeError, match="complete fixed extent"):
    CPUBackend().lower(incomplete.ir, ())


@pytest.mark.parametrize("rows,columns,dtype", [(4, 4, "float16"), (4, 4, "float32"), (8, 16, "uint8")])
def test_native_public_register_helper(compiler, rows, columns, dtype):
  @T.template
  def kernel(
    A: T.Tensor((rows, columns), dtype=dtype),
    C: T.Tensor((columns, rows), dtype=dtype, access="output"),
  ):
    values = ()
    for row in T.unroll(rows):
      values = values + (T.load(A[row, :], vector=T.Vector(dtype, N=columns)),)
    outputs = T.library.transpose_registers(values)
    for column in T.unroll(columns):
      T.store(C[column, :], outputs[column])

  compiled = T.compile(kernel, compiler=compiler)
  a = np.arange(rows * columns, dtype=dtype).reshape(rows, columns)
  c = np.empty((columns, rows), dtype=dtype)
  compiled.operator.native(a, c)
  np.testing.assert_array_equal(c, a.T)
  assert "std::array<vecops::vec::" not in compiled.generated_source
