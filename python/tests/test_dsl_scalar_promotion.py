# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Weak Python floats must not truncate in integer scalar/vector epilogues.

Integer-only arithmetic is intentionally outside this change. Typed operands
are strong; an existing floating tile retains its width for a Python literal.
The independent oracle rounds at each selected F32 boundary, not at the final
store only. Cross codegen is evidence, never a target performance claim.
"""

import ast
import ctypes
import ctypes.util
import math
import re
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest
import vecops.language as T
from dsl_test_utils import assert_no_vec_mask_arrays, make_compiler
from test_dsl_matmul_sve_optimized import _cross
from test_dsl_sme_family_owners import _assert_inlined_owners, _owners
from vecops._arm_vector_state import VectorState
from vecops.language._compiler import ir as I
from vecops.language._compiler.backends.cpu.backend import CPUBackend
from vecops.language._compiler.backends.cpu.cpp.expressions import ExpressionsEmitter
from vecops.language._compiler.frontend.parser import Frontend, lower
from vecops.language._compiler.ir.promotion import F32_MAX, F32_MIN_SUBNORMAL
from vecops.language._compiler.ir.traversal import nodes, rewrite_expressions, walk_unique
from vecops.language._compiler.operations import current


def _value(dtype="int32", *, vector=False):
  shape = (I.literal(19),)
  typ = I.VectorType(dtype, shape, tag=I.VectorTag(pow=1)) if vector else I.TileType(dtype, shape)
  return I.Expr(typ, "raw", data="value")


@pytest.mark.parametrize(
  "scalar,expected",
  (
    (0.5, "float32"),
    (-0.0, "float32"),
    (math.inf, "float32"),
    (math.nan, "float32"),
    (F32_MIN_SUBNORMAL, "float32"),
    (F32_MIN_SUBNORMAL / 2, "float64"),
    (F32_MAX, "float32"),
    (F32_MAX * 2, "float64"),
  ),
)
def test_integer_plus_weak_float_uses_minimum_nonzero_range(scalar, expected):
  result = I.binary("add", _value(), I.literal(scalar))
  assert result.type.dtype == expected
  assert result.args[0].op == "cast" and result.args[0].type.dtype == expected
  assert result.args[1].type.weak  # Fill already converts to the selected tag.


@pytest.mark.parametrize("dtype", ("float16", "bfloat16", "float32", "float64"))
def test_floating_vector_retains_dtype_tag_and_weak_literal_source(dtype):
  value, scalar = _value(dtype, vector=True), I.literal(0.5)
  result = I.binary("mul", value, scalar)
  assert result.type.dtype == dtype and isinstance(result.type, I.VectorType)
  assert result.type.tag == value.type.tag
  assert result.args == (value, scalar)
  comparison = I.binary("gt", value, scalar)
  assert comparison.type.mask and comparison.type.dtype == dtype
  assert comparison.type.tag == value.type.tag


@pytest.mark.parametrize("left,right,expected", (("bfloat16", "float16", "float32"), ("float32", "float64", "float64")))
def test_strong_floating_operands_select_shared_computation_type(left, right, expected):
  result = I.binary("add", _value(left), I.Expr(I.ScalarType(right), "raw", data="scale"))
  assert result.type.dtype == expected
  assert all(a.type.dtype == expected for a in result.args)


def test_strong_weak_identity_and_constant_folding_keep_dtype():
  weak, strong = I.literal(0.5), I.literal(0.5, "float64")
  assert not I.same_expr(weak, strong)
  assert asdict(weak.type) != asdict(strong.type)
  assert repr(weak) != repr(strong)
  result = I.binary("min", weak, strong)
  assert result.type == I.ScalarType("float64")
  assert I.binary("mul", _value("float32"), weak).type.dtype == "float32"
  assert I.binary("mul", _value("float32"), strong).type.dtype == "float64"
  folded = I.binary("add", I.literal(16777216.0, "float32"), I.literal(1.0))
  assert folded.type == I.ScalarType("float32") and folded.data == 16777216.0
  assert I.binary("add", I.literal(1.0), I.literal(2.0)).type.weak


def test_integer_only_width_and_division_keep_existing_policy():
  for dtype in ("int8", "uint8", "int32", "uint32", "int64"):
    tile = _value(dtype)
    assert I.binary("add", tile, I.literal(2)).type.dtype == dtype
    assert I.binary("div", tile, I.literal(2)).type.dtype == "float64"
    folded = I.binary("add", I.literal(1, dtype), I.literal(2, dtype))
    assert folded.type == I.ScalarType("int64") and folded.data == 3
    assert I.binary("div", I.literal(1, dtype), I.literal(2, dtype)).type == I.ScalarType("float64")


def test_strong_literal_cpp_type_and_scalar_operand_coercion():
  emitter = ExpressionsEmitter()
  emitter.environment = SimpleNamespace(operations=current())
  assert emitter.scalar(I.literal(0.1, "float32")).startswith("static_cast<float>(")
  for dtype in ("float16", "bfloat16"):
    assert emitter.scalar(I.literal(0.1, dtype)).startswith("static_cast<vecops::")
  result = I.binary("mul", I.Expr(I.ScalarType("float32"), "raw", data="x"), I.literal(0.1))
  assert result.type.dtype == "float32" and result.args[1].op == "cast"
  assert result.args[1].type == I.ScalarType("float32")


@pytest.mark.parametrize("dtype,value", (("float16", 1.0001), ("bfloat16", 1.001)))
def test_strong_half_literals_round_before_constant_folding(dtype, value):
  strong = I.literal(value, dtype)
  assert strong.data == 1.0
  result = I.binary("add", strong, I.literal(0.25, "float32"))
  assert result.type == I.ScalarType("float32") and result.data == 1.25


def _pointwise(dtype="int32"):
  N, S, CS = T.dynamic(lower=0), T.dynamic(), T.dynamic()
  output = "float64" if dtype == "float64" else "float32"

  @T.template
  def kernel(A: T.Tensor((N,), (S,), dtype), C: T.Tensor((5, N), (CS, 1), output, access="output")):
    x = T.load(A)
    T.store(C[0, :], x * 0.5 + 0.125)
    T.store(C[1, :], T.where(x > 0.5, x * 0.5 + 0.125, -0.25))
    T.store(C[2, :], T.fma(x, 0.5, 0.125))
    T.store(C[3, :], T.where(x >= 0.5, 0.25, 0.5))
    T.store(C[4, :], T.max(x * 0.5 + 0.125, 0))

  return kernel


def _scalar_precision():
  @T.template
  def kernel(A: T.Tensor((1,), dtype="float32"), C: T.Tensor((5,), dtype="float32", access="output")):
    x = T.load(A[0])
    T.store(C[0], x * 0.1 + 0.2)
    T.store(C[1], T.fma(x, 0.1, 0.2))
    T.store(C[2], T.where(x > 0, x * 0.1 + 0.2, x * 0.3 + 0.4))
    a = T.cast(16777216.0, "float32")
    T.store(C[3], -(a + 1.0))
    T.store(C[4], T.cast(x, "float16") * 0.1 + 0.2)

  return kernel


def test_frontend_where_fma_and_mask_basis_use_promoted_dtype():
  ir = lower(_pointwise().ir)
  stores = [s for s in nodes(ir.body) if isinstance(s, I.Store)]
  assert len(stores) == 5
  for store in stores:
    for value in walk_unique(store.value):
      if value.op in {"mul", "add", "fma", "where", "max"}:
        assert value.type.dtype == "float32"
      if value.op in {"gt", "ge"}:
        assert value.type.mask and value.type.dtype == "float32"
  source = CPUBackend().lower(_pointwise().ir, (), target="AVX512").source
  assert "ScalableTag<float, 0>" in source
  assert "vecops::vec::fmadd(" in source
  assert "static_cast<vecops::vec::ElementOf<" in source
  assert_no_vec_mask_arrays(source)


def test_frontend_scalar_control_local_is_strong_and_unary_keeps_type():
  @T.template
  def kernel(C: T.Tensor((1,), dtype="float32", access="output"), *, count: int = 2):
    scale = 0.5
    for i in T.serial(count):
      scale = scale + 0.25
    T.store(C[0], -scale)

  ir = lower(kernel.ir)
  store = next(s for s in nodes(ir.body) if isinstance(s, I.Store))
  local = next(v for v in walk_unique(store.value) if v.op == "frozen")
  assert isinstance(local.type, I.ScalarType) and not local.type.weak
  frontend = Frontend(kernel.ir)
  frontend.env["typed"] = I.binary("add", I.literal(16777216.0, "float32"), I.literal(1.0))
  unary = frontend.expression(ast.parse("-typed", mode="eval").body)
  assert unary.type == I.ScalarType("float32") and unary.data == -16777216.0
  scalar = lower(_scalar_precision().ir)
  store = [s for s in nodes(scalar.body) if isinstance(s, I.Store)][3]
  assert store.value.type == I.ScalarType("float32")
  assignment = next(s for s in nodes(scalar.body) if isinstance(s, I.Assign) and s.target == store.value.data)
  assert assignment.value.op == "neg" and assignment.value.type == I.ScalarType("float32")


def _fmaf(a, b, c):
  library = ctypes.CDLL(ctypes.util.find_library("m"))
  fn = library.fmaf
  fn.argtypes = (ctypes.c_float,) * 3
  fn.restype = ctypes.c_float
  return np.float32(fn(float(a), float(b), float(c)))


def _pointwise_expected(a, dtype):
  computation = np.dtype(dtype) if dtype in {"float16", "bfloat16", "float64"} else np.dtype("float32")
  x = np.asarray(a, dtype=computation)
  multiply = (x * computation.type(0.5)).astype(computation)
  arithmetic = (multiply + 0.125).astype(computation)
  fused = (x.astype(np.float64) * 0.5 + 0.125).astype(computation)
  return np.stack(
    (arithmetic, np.where(x > 0.5, arithmetic, -0.25), fused, np.where(x >= 0.5, 0.25, 0.5), np.maximum(arithmetic, 0))
  )


@pytest.fixture(scope="module")
def compiler(tmp_path_factory):
  root = tmp_path_factory.mktemp("weak-scalar-native")
  return make_compiler(jobs=4, cache_dir=root / "cache", build_dir=root / "build")


@pytest.mark.parametrize("dtype", ("int32", "float16", "bfloat16", "float32", "float64"))
def test_native_weak_scalar_pointwise_where_fma_and_signed_layouts(dtype, compiler):
  if dtype == "bfloat16":
    pytest.importorskip("ml_dtypes")
  compiled = T.compile(_pointwise(dtype), compiler=compiler)
  for size in (1, 19, 37):
    backing = np.arange(2 * size, dtype=np.int32) - size
    a = backing.astype(dtype)[::-2]
    c = np.full((5, size), -123, dtype="float64" if dtype == "float64" else "float32")
    compiled(a, c)
    np.testing.assert_array_equal(c, _pointwise_expected(a, dtype))


def test_native_scalar_f32_precision_not_cpp_double_and_explicit_fma(compiler):
  compiled = T.compile(_scalar_precision(), compiler=compiler)
  for value in (8388609.0, 1.234567, -1.234567):
    a, c = np.array((value,), dtype=np.float32), np.empty(5, dtype=np.float32)
    compiled(a, c)
    x = a[0]
    ordinary = np.float32(np.float32(x * np.float32(0.1)) + np.float32(0.2))
    alternate = np.float32(np.float32(x * np.float32(0.3)) + np.float32(0.4))
    half = np.float32(np.float16(x))
    expected = np.array(
      (
        ordinary,
        _fmaf(x, 0.1, 0.2),
        ordinary if x > 0 else alternate,
        -16777216.0,
        np.float32(np.float32(half * np.float32(0.1)) + np.float32(0.2)),
      ),
      dtype=np.float32,
    )
    np.testing.assert_array_equal(c, expected)


@pytest.mark.parametrize("dtype", ("float32", "float64"))
def test_native_strong_runtime_scalar_selects_compute_width_for_vector_where_fma(dtype, compiler):
  scalar_type = T.DType(dtype)

  @T.template
  def kernel(
    A: T.Tensor((19,), dtype="float32"),
    C: T.Tensor((3, 19), dtype="float64", access="output"),
    *,
    scale: scalar_type = 0.1,
  ):
    x = T.load(A)
    T.store(C[0, :], x * scale + 0.2)
    T.store(C[1, :], T.where(x > scale, x * scale + 0.2, x * scale - 0.2))
    T.store(C[2, :], T.fma(x, scale, 0.2))

  compiled = T.compile(kernel, compiler=compiler)
  computation = np.dtype(dtype)
  libm = ctypes.CDLL(ctypes.util.find_library("m"))
  fma = libm.fmaf if dtype == "float32" else libm.fma
  fma.argtypes = (ctypes.c_float if dtype == "float32" else ctypes.c_double,) * 3
  fma.restype = ctypes.c_float if dtype == "float32" else ctypes.c_double
  for scale in (0.1, 1.00000001):
    a = np.linspace(-1.234567, 3.123456, 19, dtype=np.float32)
    c = np.empty((3, 19), dtype=np.float64)
    compiled(a, c, scale=scale)
    x, factor, offset = a.astype(computation), computation.type(scale), computation.type(0.2)
    product = (x * factor).astype(computation)
    expected = np.stack(
      (
        (product + offset).astype(computation),
        np.where(x > factor, (product + offset).astype(computation), (product - offset).astype(computation)),
        np.array([fma(float(v), float(factor), float(offset)) for v in x], dtype=computation),
      )
    )
    np.testing.assert_array_equal(c, expected)


def test_native_strong_f32_literal_codegen_not_late_double_rounding(compiler):
  @T.template
  def typed_literal_gate(A: T.Tensor((1,), dtype="float32"), C: T.Tensor((1,), dtype="float32", access="output")):
    T.store(C[0], T.load(A[0]) * 0.1 + 0.2)

  # Compiler-created typed literals are a supported IR entry, not public
  # Python captures. Exercise that boundary with one scoped registered pass.
  name = "unit-promotion-typed-literal"

  @T.passes.register_pass(name)
  def explicit_literals(ir, context):
    assert ir.name == typed_literal_gate.ir.name

    def typed(value):
      if value.op == "cast" and value.type.dtype == "float32" and value.args[0].op == "literal":
        return I.literal(value.args[0].data, "float32")
      return value

    rewrite_expressions(ir, typed)
    return ir

  try:
    compiled = T.compile(typed_literal_gate, compiler=compiler, pipeline=[name])
    assert "static_cast<float>(0.10000000149011612)" in compiled.generated_source
    for value in (1.234567, -1.234567, 8388609.0):
      a, c = np.array((value,), np.float32), np.empty(1, np.float32)
      compiled(a, c)
      expected = np.float32(np.float32(a[0] * np.float32(0.1)) + np.float32(0.2))
      np.testing.assert_array_equal(c, (expected,))
  finally:
    T.passes.unregister_pass(name)


def _matmul(level="dot", *, strong=False):
  M, N, K = (T.dynamic(lower=0) for _ in range(3))
  D0, D1, C0, C1 = (T.dynamic() for _ in range(4))
  output = "float64" if strong else "int32"

  @T.template
  def kernel(
    A: T.Tensor((M, K), dtype="int8"),
    B: T.Tensor((K, N), dtype="int8"),
    D: T.Tensor((M, N), (D0, D1), "int32"),
    C: T.Tensor((M, N), (C0, C1), output, access="output"),
    *,
    scale: float = 0.5,
  ):
    value = (
      T.dot((T.load(A), T.load(B)), C, acc=T.load(D))
      if level == "dot"
      else T.library.matmul(
        (T.load(A), T.load(B)), C, acc=T.load(D), BK=3 if level == "blocked" else None, packing="none", swap=False
      )
    )
    if strong:
      T.store(C, T.max(value * scale + 0.125, 0))
    else:
      T.store(C, T.max(value * 0.5 + 0.125, 0))

  return kernel


@pytest.mark.parametrize("level", ("dot", "matmul", "blocked"))
def test_native_integer_dot_epilogue_weak_float_prior_alias_and_split_k(level, compiler):
  compiled = T.compile(
    _matmul(level), compiler=compiler, pass_options={"cpu_matmul_fusion": "always", "cpu_matmul_pipeline": "off"}
  )
  for k in (0, 5, 9):
    a, b = (
      (np.arange(8 * k, dtype=np.int8).reshape(8, k) % 5 - 2),
      (np.arange(k * 3, dtype=np.int8).reshape(k, 3) % 7 - 3),
    )
    for alias in (False, True):
      d = (np.arange(8 * 3, dtype=np.int32).reshape(8, 3) % 5 - 2)[:, ::-1]
      original = d.copy()
      c = d if alias else np.full((8, 3), -123, dtype=np.int32)[:, ::-1]
      expected = np.maximum(
        (a.astype(np.int32) @ b.astype(np.int32) + original).astype(np.float32) * np.float32(0.5) + np.float32(0.125), 0
      ).astype(np.int32)
      compiled(a, b, d, c)
      np.testing.assert_array_equal(c, expected)


def _wrapper(source, strong=False):
  output = "double" if strong else "int32_t"
  return "\n".join(
    (
      '#include "vecops/Kernel.h"',
      "namespace vecops::spec { inline constexpr auto Parallelism=vecops::meta::cint<1>; }",
      source,
      'extern "C" void promoted(void* ap,void* bp,void* dp,void* cp,vecops::nint_t m,vecops::nint_t n,vecops::nint_t k) {',
      "auto A=vecops::tensor::make_tensor(static_cast<int8_t*>(ap),vecops::tensor::make_shape(vecops::meta::Any{m},vecops::meta::Any{k}));",
      "auto B=vecops::tensor::make_tensor(static_cast<int8_t*>(bp),vecops::tensor::make_shape(vecops::meta::Any{k},vecops::meta::Any{n}));",
      "auto D=vecops::tensor::make_tensor(static_cast<int32_t*>(dp),vecops::tensor::make_shape(vecops::meta::Any{m},vecops::meta::Any{n}));",
      f"auto C=vecops::tensor::make_tensor(static_cast<{output}*>(cp),vecops::tensor::make_shape(vecops::meta::Any{{m}},vecops::meta::Any{{n}}));",
      "__kernel__(A,B,D,C,0.5);",
      "}",
    )
  )


def test_native_integer_scalar_minmax_signed_extremes(compiler):
  @T.template
  def kernel(C: T.Tensor((3,), dtype="int64", access="output"), left: int, right: int):
    T.store(C[0], T.min(left, right))
    T.store(C[1], T.max(left, right))
    T.store(C[2], T.min(T.max(left, -7), 13))

  compiled = T.compile(kernel, compiler=compiler)
  out = np.empty(3, np.int64)
  minimum, maximum = -(1 << 63), (1 << 63) - 1
  for left, right in ((minimum, maximum), (maximum, minimum), (minimum, minimum),
                      (maximum, maximum), (-1, 0), (0, -1)):
    compiled(out, left, right)
    np.testing.assert_array_equal(out, (min(left, right), max(left, right), min(max(left, -7), 13)))


@pytest.mark.parametrize("divisor", [1, 2, 3, 32, (1 << 63) - 1])
def test_native_positive_constant_division_signed_extremes(compiler, divisor):
  @T.template
  def kernel(C: T.Tensor((3,), dtype="int64", access="output"), value: int):
    T.store(C[0], value // divisor)
    T.store(C[1], value % divisor)
    T.store(C[2], T.ceildiv(value, divisor))

  compiled = T.compile(kernel, compiler=compiler)
  out = np.empty(3, np.int64)
  for value in (-(1 << 63), -(1 << 63) + 1, -33, -32, -31, -3, -1, 0, 1, 3, 31, 32, 33,
                (1 << 63) - 2, (1 << 63) - 1):
    compiled(out, value)
    # Avoid negating INT64_MIN in generated code; Python has unbounded ints.
    np.testing.assert_array_equal(out, (value // divisor, value % divisor, -((-value) // divisor)))


def test_native_runtime_divisor_keeps_signed_semantics_and_errors(compiler):
  @T.template
  def kernel(C: T.Tensor((3,), dtype="int64", access="output"), value: int, divisor: int):
    T.store(C[0], value // divisor)
    T.store(C[1], value % divisor)
    T.store(C[2], T.ceildiv(value, divisor))

  compiled = T.compile(kernel, compiler=compiler)
  out = np.empty(3, np.int64)
  for value, divisor in [(-33, 32), (33, -32), (-33, -32), (-(1 << 63), 3), ((1 << 63) - 1, -3)]:
    compiled(out, value, divisor)
    np.testing.assert_array_equal(out, (value // divisor, value % divisor, -((-value) // divisor)))
  for value, divisor in [(1, 0), (-(1 << 63), -1)]:
    with pytest.raises(Exception, match="division by zero or overflow"):
      compiled(out, value, divisor)


@pytest.mark.parametrize("dtype", ["uint8", "uint16", "uint32"])
def test_native_unsigned_sequence_cast_wraps_at_storage_width(dtype, compiler):
  @T.template
  def kernel(C: T.Tensor((131,), dtype=dtype, access="output"), start: int):
    T.store(C, T.cast(T.arange(start, start + 131), dtype))

  compiled = T.compile(kernel, compiler=compiler)
  out = np.empty(131, dtype=dtype)
  modulus = 1 << (out.itemsize * 8)
  for start in (-256, -3, 0, 250, 65530, (1 << 32) - 5):
    compiled(out, start)
    expected = np.array([(start + index) % modulus for index in range(131)], dtype=dtype)
    np.testing.assert_array_equal(out, expected)


@pytest.mark.parametrize("vl,svl,strong", ((16, 64, False), (64, 16, False), (64, 64, True)))
def test_real_cross_promoted_integer_epilogue_inline_owners_no_register_arrays(vl, svl, strong, tmp_path):
  source = (
    CPUBackend()
    .lower(
      _matmul(strong=strong).ir,
      (),
      target="SME",
      options={"cpu_matmul_fusion": "always", "cpu_matmul_pipeline": "off", "_vector_state": VectorState(vl, svl)},
    )
    .source
  )
  assert re.search(r"vecops::vec::Rebind<" + ("double" if strong else "float") + r", \w+>", source)
  assert_no_vec_mask_arrays(source)
  assembly = _cross(tmp_path, source, lambda text: _wrapper(text, strong), vl=vl, svl=svl)
  owners = _owners(assembly, vl=vl, svl=svl)
  assert owners
  _assert_inlined_owners(assembly, len(owners), vl=vl, svl=svl)
  # Full dynamic 4x1 integer owners currently spill two predicate values;
  # this is independently recorded, not a proof of a no-P-spill backend.
  # No Z-register spill or outlined call is introduced by numeric promotion.
  assert not any(re.search(r"^\s*(?:str|ldr|stp|ldp)\s+z\d", owner, re.MULTILINE) for owner in owners)
