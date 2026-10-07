"""Logical producer windows preserve captures and ordinary vector lowering."""

import numpy as np
import pytest
import vecops.language as T
from dsl_test_utils import assert_no_vec_mask_arrays, make_compiler
from vecops.language._compiler.backends.cpu.backend import CPUBackend
from vecops.language._compiler.frontend.parser import lower
from vecops.language._debug.interpreter import Interpreter


@T.func
def activation(source):
  return T.load(source) * 2 + 1


def rectangle_template():
  @T.template
  def kernel(A: T.Tensor((13, 19), dtype="float32"), C: T.Tensor((9, 12), dtype="float16", access="output")):
    T.store(C, activation(A)[2:11, 3:15])

  return kernel


def captured_template():
  @T.template
  def kernel(
    A: T.Tensor((13, 19), dtype="float32"),
    C: T.Tensor((4, 12), dtype="float32", access="output"),
    begin: int,
  ):
    row = begin
    part = activation(A)[row : row + 4, 3:15]
    row = row + 5
    T.store(C, part)

  return kernel


def row_template(*, strided_output=False):
  @T.template
  def kernel(
    A: T.Tensor((13, 19), dtype="float32"),
    C: T.Tensor((13, 12), (19, 1) if strided_output else None, "float32", access="output"),
  ):
    value = activation(A)
    for row in T.serial(13):
      T.store(C[row, :], value[row, 3:15])

  return kernel


@pytest.mark.parametrize("factory", [rectangle_template, captured_template, row_template])
def test_value_windows_interpret_and_vectorize(factory):
  kernel = factory()
  data = np.arange(13 * 19, dtype=np.float32).reshape(13, 19) / 16
  if factory is rectangle_template:
    output = np.empty((9, 12), np.float16)
    expected = (data * 2 + 1)[2:11, 3:15].astype(np.float16)
  elif factory is captured_template:
    output = np.empty((4, 12), np.float32)
    expected = (data * 2 + 1)[2:6, 3:15]
  else:
    output = np.empty((13, 12), np.float32)
    expected = (data * 2 + 1)[:, 3:15]
  ir = lower(kernel.ir)
  Interpreter(ir, {"A": data, "C": output, "begin": 2}).block(ir.body)
  np.testing.assert_array_equal(output, expected)
  source = CPUBackend().lower(kernel.ir, (), target="AVX512").source
  assert "vecops::vec::load_convert" in source
  assert "vecops::vec::store_convert" in source
  assert_no_vec_mask_arrays(source)


def test_value_window_rejects_stepped_or_scalar_only_geometry():
  @T.template
  def stepped(A: T.Tensor((8,), dtype="float32"), C: T.Tensor((4,), dtype="float32", access="output")):
    T.store(C, T.load(A)[::2])

  @T.template
  def scalar(A: T.Tensor((8,), dtype="float32")):
    T.load(A)[0]

  with pytest.raises(TypeError, match="unit slice"):
    lower(stepped.ir)
  with pytest.raises(TypeError, match="element_at"):
    lower(scalar.ir)


@pytest.mark.parametrize("factory", [rectangle_template, captured_template, row_template])
def test_native_value_windows(factory, tmp_path):
  kernel = factory()
  compiler = make_compiler(jobs=4, cache_dir=tmp_path / "cache", build_dir=tmp_path / "build")
  compiled = T.compile(kernel, compiler=compiler)
  data = np.arange(13 * 19, dtype=np.float32).reshape(13, 19) / 16
  if factory is rectangle_template:
    output = np.empty((9, 12), np.float16)
    expected = (data * 2 + 1)[2:11, 3:15].astype(np.float16)
    arguments = (data, output)
  elif factory is captured_template:
    output = np.empty((4, 12), np.float32)
    expected = (data * 2 + 1)[2:6, 3:15]
    arguments = (data, output, 2)
  else:
    output = np.empty((13, 12), np.float32)
    expected = (data * 2 + 1)[:, 3:15]
    arguments = (data, output)
  compiled.operator.native(*arguments, **compiled._core.specialization_values)
  np.testing.assert_array_equal(output, expected)


def test_native_value_window_keeps_read_before_write_and_output_alias(tmp_path):
  @T.template
  def historical(
    A: T.Tensor((13, 19), dtype="float32", access="inout"),
    C: T.Tensor((9, 12), dtype="float32", access="output"),
  ):
    part = activation(A)[2:11, 3:15]
    T.store(A, T.full((13, 19), 99, dtype="float32"))
    T.store(C, part)

  compiler = make_compiler(jobs=4, cache_dir=tmp_path / "cache", build_dir=tmp_path / "build")
  compiled = T.compile(historical, compiler=compiler)
  data = np.arange(13 * 19, dtype=np.float32).reshape(13, 19) / 16
  expected = (data * 2 + 1)[2:11, 3:15].copy()
  output = np.empty_like(expected)
  compiled.operator.native(data, output, **compiled._core.specialization_values)
  np.testing.assert_array_equal(output, expected)
  np.testing.assert_array_equal(data, np.full(data.shape, 99, np.float32))

  source = np.arange(13 * 19, dtype=np.float32).reshape(13, 19) / 16
  # A current window consumer also needs an overlap-safe ordinary load path.
  rectangle = T.compile(row_template(strided_output=True), compiler=compiler)
  expected = (source * 2 + 1)[:, 3:15].copy()
  rectangle.operator.native(source, source[:, 3:15], **rectangle._core.specialization_values)
  np.testing.assert_array_equal(source[:, 3:15], expected)
