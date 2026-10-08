# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Python aliases retain source names without splitting annotation identities."""

from pathlib import Path

import numpy as np
import pytest
import vecops
import vecops.language as T
from dsl_test_utils import make_compiler
from vecops.language._compiler.backends.cpu.source import _specialization_environment

SOURCE = Path(__file__).with_name("dsl_source_capture_alias.cpp")
GLOBAL_A = GLOBAL_B = T.Const(5)


def template(dtype="float32", factor=2, enabled=True, *, explicit=False):
  NativeA = NativeB = T.DType(dtype, name="PrimaryNative" if explicit else None)
  ScaleA = ScaleB = T.Const(factor, name="PrimaryScale" if explicit else None)
  EnabledA = EnabledB = T.Const(enabled, name="PrimaryEnabled" if explicit else None)
  N, S = T.dynamic(lower=0), T.dynamic()

  @T.template(source=SOURCE)
  def kernel(
    input: T.Tensor((N,), (S,), NativeA),
    output: T.Tensor((N,), (1,), NativeB, access="output"),
  ):
    NativeA, NativeB, ScaleA, ScaleB, EnabledA, EnabledB  # noqa: B018

  return kernel


@pytest.mark.parametrize("explicit", (False, True))
def test_source_aliases_export_both_names_and_keep_shared_annotation_identity(explicit):
  kernel = template(explicit=explicit)
  kinds, values, _ = _specialization_environment(kernel.ir)
  assert values["NativeA"] == values["NativeB"] == vecops.float32
  assert values["ScaleA"] == values["ScaleB"] == 2
  assert values["EnabledA"] == values["EnabledB"] == 1
  assert kinds["ScaleA"] is kinds["ScaleB"] is vecops.Const
  assert kinds["NativeA"] is kinds["NativeB"] is vecops.DType
  a, c = (kernel.ir.annotations[name] for name in ("input", "output"))
  assert a.dtype is c.dtype
  assert a.shape[0] is c.shape[0]
  assert a.dtype.name == ("PrimaryNative" if explicit else "NativeA")
  assert kernel.ir.canonical() == template(explicit=explicit).ir.canonical()


def test_source_alias_captures_keep_reference_not_an_early_value_copy():
  kernel = template()
  primary = next(value for value in kernel.ir.captures if isinstance(value, T.Const) and value.name == "ScaleA")
  primary.value = 7
  _, values, _ = _specialization_environment(kernel.ir)
  assert values["ScaleA"] == values["ScaleB"] == 7


def test_source_aliases_support_globals_and_nested_immutable_names():
  value = T.Const(3)
  pair = (value, (value,))

  @T.template(source=SOURCE)
  def kernel(input: T.Tensor((1,), dtype="float32")):
    GLOBAL_A, GLOBAL_B, pair  # noqa: B018

  _, values, _ = _specialization_environment(kernel.ir)
  assert values["GLOBAL_A"] == values["GLOBAL_B"] == 5
  assert values["pair_0"] == values["pair_1_0"] == 3


def test_source_less_aliases_do_not_add_cpp_only_specialization_names():
  A = B = T.Const(2)

  @T.template
  def kernel(input: T.Tensor((1,), dtype="float32"), output: T.Tensor((1,), dtype="float32", access="output")):
    T.store(output, T.load(input) * A + B)

  assert {value.name for value in kernel.ir.captures if getattr(value, "name", None)} == {"A"}
  assert set(kernel.ir.body.environment) >= {"A", "B"}


@pytest.mark.parametrize("dtype", ("float16", "bfloat16", "float32"))
def test_native_source_aliases_and_batch_capture_isolation(dtype, tmp_path):
  memory_dtype = np.dtype(dtype) if dtype != "bfloat16" else np.dtype(pytest.importorskip("ml_dtypes").bfloat16)
  compiler = make_compiler(build_dir=tmp_path / "build", cache_dir=tmp_path / "cache", jobs=4)
  kernels = []
  with vecops.collect_compile_requests() as collector:
    for factor, enabled, explicit in ((2, True, False), (-3, False, True)):
      compiled = T.compile(template(dtype, factor, enabled, explicit=explicit), compiler=compiler)
      compiled.operator(
        vecops.TensorMeta((17,), strides=(1,), dtype=dtype),
        vecops.TensorMeta((17,), strides=(1,), dtype=dtype),
        **compiled._core.specialization_values,
      )
      kernels.append((compiled, factor, enabled))
  assert len(collector.requests) == 2
  vecops.compile_batch(collector.requests, parallelism=4, execution_parallelism=1)
  assert kernels[0][0].get_cache_info()["key"] != kernels[1][0].get_cache_info()["key"]
  for compiled, factor, enabled in kernels:
    for size in (0, 1, 17):
      storage = (np.arange(max(1, 2 * size)) % 13 - 6).astype(memory_dtype)
      for input in (storage[:size], storage[: 2 * size : 2][::-1]):
        output_storage = np.full(size + 8, -100, dtype=memory_dtype)
        output = output_storage[4:-4]
        compiled.operator.native(input, output, **compiled._core.specialization_values)
        expected = (input.astype(np.float64) * factor + factor + (3 if enabled else 0)).astype(memory_dtype)
        np.testing.assert_array_equal(output, expected)
        np.testing.assert_array_equal(output_storage[:4], -100)
        np.testing.assert_array_equal(output_storage[-4:], -100)
