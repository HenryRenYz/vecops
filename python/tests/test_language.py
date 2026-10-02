"""Tests for the expression-based vecops.language frontend."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np
import pytest
import vecops
import vecops.language as T

VECOPS_ROOT = Path(__file__).resolve().parents[2]
_TEST_CC = os.environ.get("CC", "gcc")
_TEST_CXX = os.environ.get("CXX", "g++")


def _compiler(tmp_path: Path) -> vecops.Compiler:
  return vecops.Compiler(
    cache_mode="compile-only",
    build_dir=tmp_path / "build",
    target="Scalar",
    cc=_TEST_CC,
    cxx=_TEST_CXX,
    jobs=2,
    cache_namespace="language-tests-v1",
  )


def test_jit_ignores_hints_and_normalizes_actual_call_values() -> None:
  calls = []

  @T.jit
  def factory(value: T.Any, *extra: object, **options: object):
    calls.append((value, extra, options))
    return value, extra, options

  first = factory(3, True, (4, None), dtype="float32")
  second = factory(3, True, (4, None), dtype=np.float32)
  assert isinstance(first[0], T.Const)
  assert isinstance(first[1][0], T.Const)
  assert isinstance(first[1][1][0], T.Const)
  assert isinstance(first[2]["dtype"], T.DType)
  assert second is first
  assert len(calls) == 1

  with pytest.raises(TypeError, match="mutable argument type list"):
    factory([1, 2])
  with pytest.raises(TypeError, match="mutable argument type dict"):
    factory({"x": 1})


def test_jit_forwarding_rebinds_specialization_names_to_the_callee() -> None:
  seen = []

  @T.jit
  def inner(callee_name):
    seen.append(callee_name)
    return callee_name

  @T.jit
  def outer(caller_name):
    return inner(caller_name)

  result = outer(7)
  assert result.name == "callee_name"
  assert seen[0] is result


def test_dimension_expression_binds_runtime_symbols_without_specializing() -> None:
  m = T.dynamic(lower=0)
  m.bind_name("M")
  definition = vecops.KernelDef(
    "test::language_expression",
    [
      vecops.TensorDef("input", (m,), (1,), dtype=vecops.float32),
      vecops.TensorDef(
        "output",
        (2 * m,),
        (1,),
        dtype=vecops.float32,
        access=vecops.TensorAccess.output,
      ),
    ],
  )
  native = definition._native()
  first = vecops._C.bind_kernel_call(
    native,
    vecops._C.KernelCall(
      [
        vecops.TensorMeta((3,), dtype=vecops.float32)._native(),
        vecops.TensorMeta((6,), dtype=vecops.float32, access=vecops.TensorAccess.output)._native(),
      ]
    ),
  )
  second = vecops._C.bind_kernel_call(
    native,
    vecops._C.KernelCall(
      [
        vecops.TensorMeta((5,), dtype=vecops.float32)._native(),
        vecops.TensorMeta((10,), dtype=vecops.float32, access=vecops.TensorAccess.output)._native(),
      ]
    ),
  )
  assert first["specialization_key"] == second["specialization_key"]
  with pytest.raises(ValueError, match="must equal 6"):
    vecops._C.bind_kernel_call(
      native,
      vecops._C.KernelCall(
        [
          vecops.TensorMeta((3,), dtype=vecops.float32)._native(),
          vecops.TensorMeta((5,), dtype=vecops.float32, access=vecops.TensorAccess.output)._native(),
        ]
      ),
    )


def test_dimension_expression_uses_python_integer_semantics_and_checks_overflow() -> None:
  value = T.Const(0, name="C")
  floor_definition = vecops.KernelDef(
    "test::language_floor",
    [vecops.TensorDef("input", (1,), (value // 2,), dtype=vecops.float32)],
  )._native()
  vecops._C.bind_kernel_call(
    floor_definition,
    vecops._C.KernelCall(
      [vecops.TensorMeta((1,), strides=(-2,), dtype=vecops.float32)._native()],
      {"C": -3},
    ),
  )

  modulo_definition = vecops.KernelDef(
    "test::language_modulo",
    [vecops.TensorDef("input", (value % -1,), (1,), dtype=vecops.float32)],
  )._native()
  vecops._C.bind_kernel_call(
    modulo_definition,
    vecops._C.KernelCall(
      [vecops.TensorMeta((0,), dtype=vecops.float32)._native()],
      {"C": -(2**63)},
    ),
  )

  overflow_definition = vecops.KernelDef(
    "test::language_overflow",
    [vecops.TensorDef("input", (value + 1,), (1,), dtype=vecops.float32)],
  )._native()
  with pytest.raises(ValueError, match="overflows int64"):
    vecops._C.bind_kernel_call(
      overflow_definition,
      vecops._C.KernelCall(
        [vecops.TensorMeta((0,), dtype=vecops.float32)._native()],
        {"C": 2**63 - 1},
      ),
    )


def test_dynamic_constraint_captures_const_dependency(tmp_path: Path) -> None:
  @T.jit(compiler=_compiler(tmp_path))
  def factory(alignment):
    size = T.dynamic(alignment=alignment, lower=0)

    @T.template(source="../../tests/runtime/TestKernel.cpp")
    def kernel(input: T.Tensor((size,), (1,), vecops.float32)):
      pass

    return kernel

  kernel = factory(8)
  assert kernel.kernel_def.values == {"alignment": vecops.Const}


def _make_scale(compiler: vecops.Compiler):
  @T.jit(out_idx=[-1], compiler=compiler)
  def scale(B, D, stride_D, ComputeType):
    @T.template(source="../../tests/runtime/TestKernel.cpp")
    def kernel(
      input: T.Tensor((B, D), (stride_D, 1), vecops.float32),
      output: T.Tensor((B, D), (D, 1), vecops.float32),
      *,
      factor: float = 2.0,
      add_one: bool = False,
    ):
      ComputeType  # noqa: B018 - a bare reference is the explicit template capture marker

    return kernel

  return scale


def test_source_template_allocates_numpy_output_and_executes(tmp_path: Path) -> None:
  scale = _make_scale(_compiler(tmp_path))
  kernel = scale(2, 4, 4, np.float32)
  assert isinstance(kernel, T.CompiledKernel)
  input = np.arange(1, 9, dtype=np.float32).reshape(2, 4)
  output = kernel(input, factor=3.0, add_one=True)
  assert output.shape == (2, 4)
  assert output.strides == (16, 4)
  np.testing.assert_array_equal(output, input * 3 + 1)


def test_source_template_allocates_overlapping_numpy_storage(tmp_path: Path) -> None:
  compiler = _compiler(tmp_path)

  @T.jit(out_idx=[-1], compiler=compiler)
  def scale(B, D, stride_D, output_stride, ComputeType):
    @T.template(source="../../tests/runtime/TestKernel.cpp")
    def kernel(
      input: T.Tensor((B, D), (stride_D, 1), vecops.float32),
      output: T.Tensor((B, D), (output_stride, 1), vecops.float32),
      *,
      factor: float = 1.0,
      add_one: bool = False,
    ):
      ComputeType  # noqa: B018 - a bare reference is the explicit template capture marker

    return kernel

  kernel = scale(2, 2, 2, 1, np.float32)
  input = np.arange(1, 5, dtype=np.float32).reshape(2, 2)
  output = kernel(input)
  assert output.strides == (4, 4)
  assert output.base.nbytes == 3 * output.itemsize
  np.testing.assert_array_equal(output, np.array([[1, 3], [3, 4]], dtype=np.float32))


def test_source_template_allocates_torch_output_and_executes(tmp_path: Path) -> None:
  torch = pytest.importorskip("torch")
  scale = _make_scale(_compiler(tmp_path))
  kernel = scale(2, 4, 4, torch.float32)
  input = torch.arange(1, 9, dtype=torch.float32).reshape(2, 4)
  output = kernel(input, factor=3.0, add_one=True)
  assert output.shape == (2, 4)
  assert output.stride() == (4, 1)
  torch.testing.assert_close(output, input * 3 + 1)


def test_template_compile_collection_preserves_distinct_captures(tmp_path: Path) -> None:
  compiler = _compiler(tmp_path)
  scale = _make_scale(compiler)
  first = scale(2, 4, 4, np.float32)
  second = scale(3, 5, 5, np.float32)
  with vecops.collect_compile_requests() as collector:
    first(np.zeros((2, 4), dtype=np.float32))
    second(np.zeros((3, 5), dtype=np.float32))
  assert len(collector.requests) == 2
  result = vecops.compile_batch(
    collector.requests,
    parallelism=2,
    execution_parallelism=1,
  )
  assert result.prepared == 2


def test_source_less_template_fails_only_at_compile() -> None:
  @T.template
  def kernel(value: int):
    pass

  with pytest.raises(NotImplementedError, match="Python DSL"):
    T.compile(kernel)


def test_typing_optional_marks_a_template_tensor_optional(tmp_path: Path) -> None:
  tensor = T.Tensor((2,), dtype=vecops.float32)

  @T.template(source="../../tests/runtime/TestKernel.cpp", compiler=_compiler(tmp_path))
  def kernel(input: Optional[tensor] = None):  # noqa: UP045 - exercise typing.Optional support
    pass

  compiled = T.compile(kernel)
  assert compiled.kernel_def.parameters[0].optional
