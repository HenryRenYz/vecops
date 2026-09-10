"""Tests for the Pythonic schema, JIT, and optional-framework surface."""

import ctypes
import sys
import threading
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


def test_framework_bridge_forwards_external_workspace(tmp_path: Path) -> None:
  """The process-local C bridge must preserve the complete VecopsCall frame."""

  class TensorView(ctypes.Structure):
    _fields_ = [
      ("struct_size", ctypes.c_uint32),
      ("data", ctypes.c_void_p),
      ("byte_offset", ctypes.c_uint64),
      ("device_type", ctypes.c_uint32),
      ("device_index", ctypes.c_int32),
      ("dtype", ctypes.c_uint32),
      ("rank", ctypes.c_uint32),
      ("sizes", ctypes.POINTER(ctypes.c_int64)),
      ("strides", ctypes.POINTER(ctypes.c_int64)),
      ("flags", ctypes.c_uint64),
    ]

  class ScalarStorage(ctypes.Union):
    _fields_ = [
      ("i64", ctypes.c_int64),
      ("u64", ctypes.c_uint64),
      ("f64", ctypes.c_double),
    ]

  class Scalar(ctypes.Structure):
    _fields_ = [
      ("struct_size", ctypes.c_uint32),
      ("dtype", ctypes.c_uint32),
      ("value", ScalarStorage),
    ]

  class ValuePayload(ctypes.Union):
    _fields_ = [("tensor", TensorView), ("scalar", Scalar)]

  class Value(ctypes.Structure):
    _fields_ = [
      ("struct_size", ctypes.c_uint32),
      ("kind", ctypes.c_uint32),
      ("value", ValuePayload),
    ]

  class ExecutionContext(ctypes.Structure):
    _fields_ = [
      ("struct_size", ctypes.c_uint32),
      ("requested_threads", ctypes.c_uint32),
      ("stream", ctypes.c_void_p),
      ("user_data", ctypes.c_void_p),
      ("flags", ctypes.c_uint64),
    ]

  class Call(ctypes.Structure):
    _fields_ = [
      ("struct_size", ctypes.c_uint32),
      ("num_values", ctypes.c_uint32),
      ("values", ctypes.POINTER(Value)),
      ("workspace", ctypes.c_void_p),
      ("workspace_size", ctypes.c_uint64),
      ("context", ctypes.POINTER(ExecutionContext)),
    ]

  class Error(ctypes.Structure):
    _fields_ = [
      ("struct_size", ctypes.c_uint32),
      ("code", ctypes.c_int32),
      ("message", ctypes.c_char_p),
      ("message_capacity", ctypes.c_size_t),
      ("message_required", ctypes.c_size_t),
    ]

  class SpecializationPayload(ctypes.Union):
    _fields_ = [("integer", ctypes.c_int64), ("dtype", ctypes.c_uint32)]

  class SpecializationArgument(ctypes.Structure):
    _fields_ = [
      ("struct_size", ctypes.c_uint32),
      ("kind", ctypes.c_uint32),
      ("name", ctypes.c_char_p),
      ("value", SpecializationPayload),
    ]

  compiler = vecops.Compiler(
    cache_mode="compile-only",
    build_dir=tmp_path / "build",
    target="Scalar",
    cc="/usr/bin/gcc",
    cxx="/usr/bin/g++",
    jobs=2,
  )
  kernel = make_test_kernel(compiler)
  input = np.arange(1, 9, dtype=np.float32).reshape(2, 4)
  output = np.zeros_like(input)
  kernel.compile_for(
    vecops.TensorMeta(input.shape, dtype=vecops.float32),
    vecops.TensorMeta(
      output.shape, dtype=vecops.float32,
      access=vecops.TensorAccess.output,
    ),
  )

  sizes = (ctypes.c_int64 * 2)(2, 4)
  strides = (ctypes.c_int64 * 2)(4, 1)

  def tensor_value(array: np.ndarray, flags: int) -> Value:
    tensor = TensorView(
      ctypes.sizeof(TensorView), array.ctypes.data, 0, 1, 0, 12, 2,
      sizes, strides, flags,
    )
    return Value(ctypes.sizeof(Value), 1, ValuePayload(tensor=tensor))

  factor = Scalar(
    ctypes.sizeof(Scalar), 13, ScalarStorage(f64=3.0),
  )
  add_one = Scalar(
    ctypes.sizeof(Scalar), 1, ScalarStorage(u64=1),
  )
  values = (Value * 4)(
    tensor_value(input, 1), tensor_value(output, 2),
    Value(ctypes.sizeof(Value), 2, ValuePayload(scalar=factor)),
    Value(ctypes.sizeof(Value), 2, ValuePayload(scalar=add_one)),
  )

  raw_workspace = (ctypes.c_ubyte * 128)()
  workspace_address = (
    ctypes.addressof(raw_workspace) + 63
  ) & ~63
  ctypes.memset(workspace_address, 0xcc, 64)
  context = ExecutionContext(
    ctypes.sizeof(ExecutionContext), 3, None, None, 0x55,
  )
  call = Call(
    ctypes.sizeof(Call), len(values), values, workspace_address, 64,
    ctypes.pointer(context),
  )
  specialization = (SpecializationArgument * 1)(
    SpecializationArgument(
      ctypes.sizeof(SpecializationArgument), 2, b"ComputeType",
      SpecializationPayload(dtype=12),
    )
  )
  message = ctypes.create_string_buffer(1024)
  error = Error(
    ctypes.sizeof(Error), 0, ctypes.cast(message, ctypes.c_char_p),
    len(message), 0,
  )

  library = ctypes.CDLL(vecops._C.__file__)
  bridge = library.vecops_operator_bridge_invoke_v1
  bridge.argtypes = [
    ctypes.c_uint64, ctypes.POINTER(Call), ctypes.c_uint32,
    ctypes.POINTER(SpecializationArgument), ctypes.POINTER(Error),
  ]
  bridge.restype = ctypes.c_int32
  result = bridge(
    kernel.operator.native._handle, ctypes.byref(call), len(specialization),
    specialization, ctypes.byref(error),
  )
  assert result == 0, message.value.decode()
  np.testing.assert_array_equal(output, input * 3 + 1)
  assert ctypes.c_float.from_address(workspace_address).value == 25.0


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
  returning_schema = _torch_schema("run", definition, return_outputs=True)
  assert returning_schema.endswith("-> Tensor")
  returning_source = _generate_torch_source(
    "vecops_test", "run", definition, return_outputs=True
  )
  assert "at::Tensor wrapper(" in returning_source
  assert "return p1;" in returning_source


def test_compile_batch_validates_parallelism() -> None:
  assert vecops.compile_batch([], parallelism=3).parallelism == 0
  with pytest.raises(ValueError, match="positive integer"):
    vecops.compile_batch([], parallelism=0)
  with pytest.raises(ValueError, match="positive integer"):
    vecops.compile_batch([], parallelism=True)


def test_compile_batch_submits_requests_concurrently() -> None:
  rendezvous = threading.Barrier(2)

  class PreparingOperator:
    def prepare(self, call) -> None:
      del call
      rendezvous.wait(timeout=5)

  requests = [
    vecops.CompileRequest(
      operator=PreparingOperator(),
      call=vecops._C.KernelCall([]),
      key=(index,),
      name=f"request-{index}",
    )
    for index in range(2)
  ]
  result = vecops.compile_batch(requests, parallelism=2)
  assert result.prepared == 2
  assert result.parallelism == 2
