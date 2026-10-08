# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Independent anonymous symbols survive a TensorType-only factory boundary."""

import re

import numpy as np
import pytest
import vecops
import vecops.language as T
from dsl_test_utils import make_compiler
from vecops.language._compiler.backends.cpu.backend import CPUBackend
from vecops.language._compiler.backends.cpu.source import _build_definition
from vecops.language._compiler.frontend.parser import lower


def _tensor_factory(AT, BT, CT, PT):
  @T.template
  def kernel(A: AT, B: BT, C: CT, P: PT):
    T.pack(T.load(A), P, side="A", format="dense")
    T.store(C, T.dot((A, B), C))

  return kernel


def anonymous_template():
  # Every object is distinct, but none of the original names is in the
  # decorator frame or the body closure: only TensorTypes cross the factory.
  M, N, K, PS = (T.dynamic(lower=1) for _ in range(4))
  return _tensor_factory(
    T.Tensor((M, K), (K, 1), "float32"),
    T.Tensor((K, N), (N, 1), "float32"),
    T.Tensor((M, N), (N, 1), "float32", access="output"),
    T.Tensor((PS,), (1,), "float32", access="output"),
  )


def test_anonymous_factory_codegen_preserves_each_runtime_dimension():
  template = anonymous_template()
  source = CPUBackend().lower(template.ir, (), target="AVX512").source
  entry = {
    expression: local
    for local, expression in re.findall(
      r"const auto (dsl_c_parameter_\w+) = (\w+\.(?:data|size|stride)\([^)]*\));", source
    )
  }
  assert "pack_size<float, 0>(" + entry["A.size(0)"] + ", " + entry["A.size(1)"] + ")" in source
  assert "((" + entry["P.size(0)"] + "))" in source
  assert len(entry.values()) == len(set(entry.values()))
  assert all(source.count(expression) == 1 for expression in entry)
  # Fresh factory/capture identity must preserve both the anonymous-symbol
  # bindings and their deterministic entry-local code-generation mapping.
  second = anonymous_template()
  assert template.ir.canonical() == second.ir.canonical()
  assert source == CPUBackend().lower(second.ir, (), target="AVX512").source
  bindings = lower(template.ir).bindings
  assert len(bindings) == 4 and None not in bindings
  assert {value.data for value in bindings.values()} == {"A.size(0)", "A.size(1)", "B.size(1)", "P.size(0)"}


def test_anonymous_fresh_factories_have_stable_identity_and_preserve_aliases():
  first, second = anonymous_template(), anonymous_template()
  assert first.ir.canonical() == second.ir.canonical()
  a, b, c = (first.ir.annotations[name] for name in ("A", "B", "C"))
  assert a.shape[1] is b.shape[0]
  assert a.shape[0].name == c.shape[0].name
  assert a.shape[0].name != a.shape[1].name != b.shape[1].name
  definition, _, _ = _build_definition(first.ir, ())
  native = definition._native()
  operands = [(2, 5), (5, 3), (2, 3), (10,)]

  def metadata(shapes):
    return [
      vecops.TensorMeta(
        shape, dtype="float32", access=vecops.TensorAccess.output if index >= 2 else vecops.TensorAccess.input
      )._native()
      for index, shape in enumerate(shapes)
    ]

  vecops._C.bind_kernel_call(native, vecops._C.KernelCall(metadata(operands)))
  with pytest.raises(ValueError):
    vecops._C.bind_kernel_call(
      native,
      vecops._C.KernelCall(metadata([(2, 5), (4, 3), (2, 3), (10,)])),
    )


def _dense_copy_factory(AT, CT):
  @T.template
  def kernel(A: AT, C: CT):
    T.store(C, T.load(A) + 1)

  return kernel


def _anonymous_dense_copy():
  m, n = T.dynamic(lower=0), T.dynamic(lower=0)
  dtype = T.DType("float32")
  return _dense_copy_factory(T.Tensor((m, n), dtype=dtype), T.Tensor((m, n), dtype=dtype, access="output"))


def test_anonymous_dense_strides_and_dtype_cross_tensor_factory_boundary():
  first, second = _anonymous_dense_copy(), _anonymous_dense_copy()
  assert first.ir.canonical() == second.ir.canonical()
  definition, values, _ = _build_definition(first.ir, ())
  assert tuple(values.values()) == (vecops.float32,)
  definition._native()  # Derived dense strides cannot contain anonymous symbols.
  source = CPUBackend().lower(first.ir, (), target="Scalar").source
  assert "A.size(1)" in source


def test_generated_annotation_names_do_not_shadow_explicit_specializations():
  dimensions = (T.dynamic(), T.Dynamic(name="__dsl_dynamic_0"))
  template = _dense_copy_factory(T.Tensor(dimensions, dtype="float32"), T.Tensor(dimensions, dtype="float32"))
  first, second = template.ir.annotations["A"].shape
  assert first.name != second.name
  assert first.name == "__dsl_dynamic_0_1"


def test_anonymous_factory_native_non_square_shape_and_packed_extent(tmp_path):
  compiler = make_compiler(
    target="Native", jobs=2, output="quiet", cache_dir=tmp_path / "cache", build_dir=tmp_path / "build"
  )
  compiled = T.compile(anonymous_template(), compiler=compiler)
  for m, n, k in [(2, 3, 5), (3, 2, 7)]:
    a, b = np.arange(m * k, dtype=np.float32).reshape(m, k), np.arange(k * n, dtype=np.float32).reshape(k, n)
    c, p = np.empty((m, n), dtype=np.float32), np.empty(m * k, dtype=np.float32)
    compiled(a, b, c, p)
    np.testing.assert_array_equal(p, a.ravel())
    np.testing.assert_allclose(c, a @ b)


def test_anonymous_dense_copy_native_and_automatic_output(tmp_path):
  compiled = T.compile(
    _anonymous_dense_copy(),
    out_idx=-1,
    compiler=make_compiler(output="quiet", jobs=2, cache_dir=tmp_path / "cache", build_dir=tmp_path / "build"),
  )
  for shape in [(2, 5), (3, 7)]:
    a = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    np.testing.assert_array_equal(compiled(a), a + 1)
