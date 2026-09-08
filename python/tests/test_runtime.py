"""Tests for the Pythonic schema, JIT, and optional-framework surface."""

import sys
from pathlib import Path

import numpy as np
import pytest
import vecops
import vecops.typing as vt

VECOPS_ROOT = Path(__file__).resolve().parents[2]
_DYNAMIC_OUTPUT_STRIDES = "Dynamic<1,D,1048576> 1"


def test_dtype_normalization_has_no_required_torch_dependency() -> None:
  assert vecops.normalize_dtype(np.float32) == vecops.float32
  assert vecops.normalize_dtype(np.dtype("float64")) == vecops.float64
  assert vecops.normalize_dtype(vecops.bfloat16) == vecops.bfloat16
  assert "torch" not in sys.modules
  with pytest.raises(TypeError, match="unsupported dtype"):
    vecops.normalize_dtype(object())


def test_string_dimension_dsl_and_symbol_inference() -> None:
  annotation = vt.In(np.float32)["Dynamic<8,1,MaxB> N", "Const 1"]
  definition = vecops.KernelDef(
    "test::dsl",
    [vecops.TensorDef("input", annotation.shape, annotation.strides, dtype=annotation.dtype)],
  )
  assert definition.values == {"MaxB": vecops.Const, "N": vecops.Const}
  native = definition._native()
  assert native.inputs[0].shape[0].kind == vecops._C.DimensionKind.dynamic
  assert native.inputs[0].strides[0].kind == vecops._C.DimensionKind.const


def test_omitted_strides_are_anonymous_constants() -> None:
  annotation = vt.In()["B N"]
  assert annotation.strides == (vecops.Const, vecops.Const)


def test_named_dynamic_symbol_checks_equality_without_specializing() -> None:
  definition = vecops.KernelDef(
    "test::named_dynamic",
    [
      vecops.TensorDef("input", ["B", "D"], ["D", 1], dtype=vecops.float32),
      vecops.TensorDef(
        "output",
        ["B", "D"],
        ["D", 1],
        dtype=vecops.float32,
        access=vecops.TensorAccess.output,
      ),
    ],
    symbols={"B": vt.Dynamic(lower=0)},
  )
  assert definition.values == {"D": vecops.Const}
  native = definition._native()
  assert native.inputs[0].shape[0].kind == vecops._C.DimensionKind.dynamic
  assert native.inputs[0].shape[0].symbol == "B"

  def bind(batch: int, output_batch: int):
    return vecops._C.bind_kernel_call(
      native,
      vecops._C.KernelCall(
        [
          vecops.TensorMeta((batch, 4), dtype=vecops.float32)._native(),
          vecops.TensorMeta(
            (output_batch, 4),
            dtype=vecops.float32,
            access=vecops.TensorAccess.output,
          )._native(),
        ]
      ),
    )

  first = bind(2, 2)
  second = bind(7, 7)
  assert first["specialization_key"] == second["specialization_key"]
  with pytest.raises(ValueError, match="runtime dimension 'B'"):
    bind(2, 3)


def test_runtime_tensor_signature_supports_numpy_variadic_and_product() -> None:
  @vt.check_tensors(symbols={"B": vt.Dynamic(lower=0)})
  def checked(
    act: vt.Tensor(np.float32)["*B N C"],
    transition1: vt.Tensor(np.float32)["C 2*M"],
    transition2: vt.Tensor(np.float32)["M C"],
  ):
    return act

  act = np.zeros((2, 3, 5, 7), dtype=np.float32)
  transition1 = np.zeros((7, 22), dtype=np.float32)
  transition2 = np.zeros((11, 7), dtype=np.float32)
  assert checked(act, transition1, transition2) is act
  with pytest.raises(ValueError, match="expected 22"):
    checked(act, np.zeros((7, 21), dtype=np.float32), transition2)
  with pytest.raises(TypeError, match="expected"):
    checked(act.astype(np.float64), transition1, transition2)


def test_runtime_tensor_pattern_accepts_storage_free_metadata() -> None:
  pattern = vt.Tensor(vecops.float32)["B C", "C 1"]
  metadata = vecops.TensorMeta((3, 5), dtype=vecops.float32)
  assert pattern.check(metadata) is metadata


@pytest.mark.parametrize(
  "declaration",
  ["Dynamic<0,1,8> N", "Dynamic<8,broken-name,8> N", "Dynamic<8,1 N"],
)
def test_invalid_dimension_dsl_is_rejected(declaration: str) -> None:
  with pytest.raises((TypeError, ValueError)):
    annotation = vt.In()[declaration]
    vecops.KernelDef("bad", [vecops.TensorDef("input", annotation.shape, annotation.strides)])


def test_access_modes_lower_to_native_flags() -> None:
  output = vecops.TensorDef("out", ["N"], [1], access=vecops.TensorAccess.output)
  inout = vecops.TensorDef("state", ["N"], [1], access=vecops.TensorAccess.inout)
  assert output.output
  assert inout.output
  assert output._native().access == vecops._C.TensorAccess.output
  assert inout._native().access == vecops._C.TensorAccess.inout
  with pytest.raises(ValueError, match="writable tensors cannot be optional"):
    vecops.TensorDef("out", ["N"], [1], optional=True, access=vecops.TensorAccess.output)


def make_test_kernel(compiler: vecops.Compiler):
  @vecops.jit(source="../../tests/runtime/TestKernel.cpp", compiler=compiler)
  def kernel(
    input: vt.In(vecops.float32)["B D", "stride_D 1"],
    output: vt.Out(vecops.float32)["B D", _DYNAMIC_OUTPUT_STRIDES],
    factor: float = 2.0,
    add_one: bool = False,
    *,
    ComputeType: vt.DType = np.float32,
  ) -> None: ...

  return kernel


def test_jit_derives_complete_kernel_def_and_resolves_source(tmp_path: Path) -> None:
  compiler = vecops.Compiler(
    cache_mode="compile-only",
    build_dir=tmp_path / "build",
    target="Scalar",
    cc="/usr/bin/gcc",
    cxx="/usr/bin/g++",
  )
  kernel = make_test_kernel(compiler)
  assert kernel.source == VECOPS_ROOT / "tests/runtime/TestKernel.cpp"
  assert kernel.kernel_def.values == {
    "B": vecops.Const,
    "D": vecops.Const,
    "stride_D": vecops.Const,
    "ComputeType": vecops.DType,
  }


def test_jit_numpy_execution(tmp_path: Path) -> None:
  kernel = make_test_kernel(
    vecops.Compiler(
      cache_mode="compile-only",
      build_dir=tmp_path / "build",
      target="Scalar",
      cc="/usr/bin/gcc",
      cxx="/usr/bin/g++",
      jobs=2,
    )
  )
  input = np.arange(1, 9, dtype=np.float32).reshape(2, 4)
  output = np.zeros_like(input)
  result = kernel(input, output, 3.0, True)
  assert result is output
  np.testing.assert_array_equal(output, input * 3 + 1)


def test_compile_for_accepts_storage_free_tensor_meta(tmp_path: Path) -> None:
  kernel = make_test_kernel(
    vecops.Compiler(
      cache_mode="compile-only",
      build_dir=tmp_path / "build",
      target="Scalar",
      cc="/usr/bin/gcc",
      cxx="/usr/bin/g++",
      jobs=2,
    )
  )
  kernel.compile_for(
    vecops.TensorMeta((2, 4), dtype=vecops.float32),
    vecops.TensorMeta((2, 4), dtype=vecops.float32, access=vecops.TensorAccess.output),
  )


def test_cache_only_does_not_discover_tools_or_create_directories(tmp_path: Path) -> None:
  cache = tmp_path / "absent-cache"
  build = tmp_path / "absent-build"
  compiler = vecops.Compiler(cache_mode="cache-only", cache_dir=cache, build_dir=build, cc="missing", cxx="missing")
  kernel = make_test_kernel(compiler)
  assert compiler.native is None
  input = np.ones((2, 4), dtype=np.float32)
  output = np.zeros_like(input)
  with pytest.raises(ImportError, match="cache-only"):
    kernel(input, output)
  assert not cache.exists()
  assert not build.exists()


def test_generated_torch_bridge_uses_mutable_out_schema() -> None:
  from vecops._schema_bridge import _generate_torch_source, _torch_schema

  definition = make_test_kernel(vecops.Compiler(cache_mode="cache-only")).kernel_def._native()
  schema = _torch_schema("run", definition)
  assert "Tensor(a!) output" in schema
  assert schema.endswith("-> ()")
  source = _generate_torch_source("vecops_test", "run", definition)
  assert "TORCH_LIBRARY_FRAGMENT(vecops_test" in source
  assert "vecops_operator_bridge_invoke_v1" in source
