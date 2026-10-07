"""Tests for the low-level schema, operators, and optional-framework surface."""

import ctypes
import os
import sys
import threading
from pathlib import Path

import numpy as np
import pytest
import vecops

VECOPS_ROOT = Path(__file__).resolve().parents[2]
_TEST_CC = os.environ.get("CC", "gcc")
_TEST_CXX = os.environ.get("CXX", "g++")


def test_flush_subnormals_sets_current_and_new_threads() -> None:
  if not vecops._C.flush_subnormals_supported():
    pytest.skip("hardware subnormal mode is unavailable")
  original = vecops.current_thread_flush_subnormals()
  old_env = os.environ.get("VECOPS_FLUSH_SUBNORMALS")
  try:
    vecops.set_flush_subnormals(True)
    assert vecops.current_thread_flush_subnormals()
    inherited: list[bool] = []
    thread = threading.Thread(target=lambda: inherited.append(vecops.current_thread_flush_subnormals()))
    thread.start()
    thread.join()
    assert inherited == [True]
    vecops.set_flush_subnormals(False)
    assert not vecops.current_thread_flush_subnormals()
  finally:
    vecops.set_flush_subnormals(original)
    if old_env is None:
      os.environ.pop("VECOPS_FLUSH_SUBNORMALS", None)
    else:
      os.environ["VECOPS_FLUSH_SUBNORMALS"] = old_env


def test_dtype_normalization_has_no_required_torch_dependency() -> None:
  torch_loaded = "torch" in sys.modules
  assert vecops.normalize_dtype(np.float32) == vecops.float32
  assert vecops.normalize_dtype(np.dtype("float64")) == vecops.float64
  assert vecops.normalize_dtype(vecops.bfloat16) == vecops.bfloat16
  assert ("torch" in sys.modules) == torch_loaded
  with pytest.raises(TypeError, match="unsupported dtype"):
    vecops.normalize_dtype(object())


def test_native_compiler_target_allows_process_override(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.setenv("VECOPS_TARGET_ARCH", "NativeFixedSVE")
  defaulted = vecops.Compiler(cache_mode="cache-only")
  explicit = vecops.Compiler(cache_mode="cache-only", target="Scalar")
  assert defaulted.target == "NativeFixedSVE"
  assert explicit.target == "Scalar"
  assert defaulted.cache_namespace == "python-sdk-v12;target=NativeFixedSVE"


def test_schema_dimension_constraints_and_symbol_inference() -> None:
  definition = vecops.KernelDef(
    "test::dsl",
    [vecops.TensorDef("input", [vecops.Dynamic(8, 1, "MaxB"), "N"], [vecops.Const, 1], dtype=np.float32)],
  )
  assert definition.values == {"MaxB": vecops.Const, "N": vecops.Const}
  native = definition._native()
  assert native.inputs[0].shape[0].kind == vecops._C.DimensionKind.dynamic
  assert native.inputs[0].strides[0].kind == vecops._C.DimensionKind.const


def test_omitted_strides_are_anonymous_constants() -> None:
  tensor = vecops.TensorDef("input", ["B", "N"])
  assert tensor.strides == (vecops.Const, vecops.Const)


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
    symbols={"B": vecops.Dynamic(lower=0)},
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


@pytest.mark.parametrize(
  "declaration",
  ["Dynamic<0,1,8> N", "Dynamic<8,broken-name,8> N", "Dynamic<8,1 N"],
)
def test_invalid_dimension_dsl_is_rejected(declaration: str) -> None:
  with pytest.raises((TypeError, ValueError)):
    vecops.KernelDef("bad", [vecops.TensorDef("input", [declaration])])._native()


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
  # Exercise the runtime ABI directly; frontend behavior belongs in test_language.
  definition = vecops.KernelDef(
    "test::scale",
    [
      vecops.TensorDef("input", ["B", "D"], ["stride_D", 1], dtype="ComputeType"),
      vecops.TensorDef(
        "output",
        ["B", "D"],
        [vecops.Dynamic(1, "D", 1048576), 1],
        dtype="ComputeType",
        access=vecops.TensorAccess.output,
      ),
      vecops.ValueDef("factor", float, 2.0),
      vecops.ValueDef("add_one", bool, False),
    ],
  )
  return vecops.Operator(VECOPS_ROOT / "tests/runtime/TestKernel.cpp", definition, compiler)


def test_operator_preserves_complete_kernel_def_and_source(tmp_path: Path) -> None:
  compiler = vecops.Compiler(
    cache_mode="compile-only",
    build_dir=tmp_path / "build",
    target="Scalar",
    cc=_TEST_CC,
    cxx=_TEST_CXX,
  )
  kernel = make_test_kernel(compiler)
  assert kernel.source == VECOPS_ROOT / "tests/runtime/TestKernel.cpp"
  assert kernel.kernel_def.values == {
    "B": vecops.Const,
    "D": vecops.Const,
    "stride_D": vecops.Const,
    "ComputeType": vecops.DType,
  }


def test_operator_numpy_execution(tmp_path: Path) -> None:
  kernel = make_test_kernel(
    vecops.Compiler(
      cache_mode="compile-only",
      build_dir=tmp_path / "build",
      target="Scalar",
      cc=_TEST_CC,
      cxx=_TEST_CXX,
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
      cc=_TEST_CC,
      cxx=_TEST_CXX,
      jobs=2,
    )
  )
  kernel.compile_for(
    vecops.TensorMeta((2, 4), dtype=vecops.float32),
    vecops.TensorMeta((2, 4), dtype=vecops.float32, access=vecops.TensorAccess.output),
  )


def test_framework_bridge_forwards_workspace_context_and_memory_provider(tmp_path: Path) -> None:
  """The process-local C bridge must preserve or enrich the complete call."""

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
      ("workspace_provider", ctypes.c_void_p),
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
    cc=_TEST_CC,
    cxx=_TEST_CXX,
    jobs=2,
  )
  kernel = make_test_kernel(compiler)
  input = np.arange(1, 9, dtype=np.float32).reshape(2, 4)
  output = np.zeros_like(input)
  kernel.compile_for(
    vecops.TensorMeta(input.shape, dtype=vecops.float32),
    vecops.TensorMeta(
      output.shape,
      dtype=vecops.float32,
      access=vecops.TensorAccess.output,
    ),
  )

  direct_memory = vecops.memory.System()
  with direct_memory.workspace_session(fast_capacity=4096, slow_capacity=4096):
    kernel(input, output)
    assert sum(item["managed_bytes"] for item in direct_memory.stats()) == 8192
  np.testing.assert_array_equal(output, input * 2)
  assert sum(item["managed_bytes"] for item in direct_memory.stats()) == 0
  output.fill(0)

  sizes = (ctypes.c_int64 * 2)(2, 4)
  strides = (ctypes.c_int64 * 2)(4, 1)

  def tensor_value(array: np.ndarray, flags: int) -> Value:
    tensor = TensorView(
      ctypes.sizeof(TensorView),
      array.ctypes.data,
      0,
      1,
      0,
      12,
      2,
      sizes,
      strides,
      flags,
    )
    return Value(ctypes.sizeof(Value), 1, ValuePayload(tensor=tensor))

  factor = Scalar(
    ctypes.sizeof(Scalar),
    13,
    ScalarStorage(f64=3.0),
  )
  add_one = Scalar(
    ctypes.sizeof(Scalar),
    1,
    ScalarStorage(u64=1),
  )
  values = (Value * 4)(
    tensor_value(input, 1),
    tensor_value(output, 2),
    Value(ctypes.sizeof(Value), 2, ValuePayload(scalar=factor)),
    Value(ctypes.sizeof(Value), 2, ValuePayload(scalar=add_one)),
  )

  raw_workspace = (ctypes.c_ubyte * 128)()
  workspace_address = (ctypes.addressof(raw_workspace) + 63) & ~63
  ctypes.memset(workspace_address, 0xCC, 64)
  context = ExecutionContext(
    ctypes.sizeof(ExecutionContext),
    3,
    None,
    None,
    0x55,
  )
  call = Call(
    ctypes.sizeof(Call),
    len(values),
    values,
    workspace_address,
    64,
    ctypes.pointer(context),
  )
  specialization = (SpecializationArgument * 1)(
    SpecializationArgument(
      ctypes.sizeof(SpecializationArgument),
      2,
      b"ComputeType",
      SpecializationPayload(dtype=12),
    )
  )
  message = ctypes.create_string_buffer(1024)
  error = Error(
    ctypes.sizeof(Error),
    0,
    ctypes.cast(message, ctypes.c_char_p),
    len(message),
    0,
  )

  library = ctypes.CDLL(vecops._C.__file__)
  bridge = library.vecops_operator_bridge_invoke_v1
  bridge.argtypes = [
    ctypes.c_uint64,
    ctypes.POINTER(Call),
    ctypes.c_uint32,
    ctypes.POINTER(SpecializationArgument),
    ctypes.POINTER(Error),
  ]
  bridge.restype = ctypes.c_int32
  result = bridge(
    kernel.native._handle,
    ctypes.byref(call),
    len(specialization),
    specialization,
    ctypes.byref(error),
  )
  assert result == 0, message.value.decode()
  np.testing.assert_array_equal(output, input * 3 + 1)
  assert ctypes.c_float.from_address(workspace_address).value == 25.0

  memory_system = vecops.memory.System()
  output.fill(0)
  with memory_system.workspace_session(fast_capacity=4096, slow_capacity=4096):
    result = bridge(
      kernel.native._handle,
      ctypes.byref(call),
      len(specialization),
      specialization,
      ctypes.byref(error),
    )
  assert result == 0, message.value.decode()
  assert sum(item["managed_bytes"] for item in memory_system.stats()) == 0

  output.fill(0)
  managed_context = ExecutionContext(
    ctypes.sizeof(ExecutionContext),
    3,
    None,
    ctypes.c_void_p(0x1234),
    0x55,
    None,
  )
  managed_call = Call(
    ctypes.sizeof(Call),
    len(values),
    values,
    None,
    0,
    ctypes.pointer(managed_context),
  )
  with memory_system.workspace_session(fast_capacity=4096, slow_capacity=4096):
    result = bridge(
      kernel.native._handle,
      ctypes.byref(managed_call),
      len(specialization),
      specialization,
      ctypes.byref(error),
    )
    assert sum(item["managed_bytes"] for item in memory_system.stats()) == 8192
  assert result == 0, message.value.decode()
  np.testing.assert_array_equal(output, input * 3 + 1)
  assert sum(item["managed_bytes"] for item in memory_system.stats()) == 0


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
  assert "TORCH_LIBRARY_FRAGMENT(vecops_test" not in source
  assert "vecops_operator_bridge_invoke_v1" in source
  assert "vecops_torch_bridge_register_v2" in source
  assert "struct CpuFunctor" in source and "struct MetaFunctor" in source
  assert "std::shared_ptr<State> state" in source
  assert "at::parallel_for(0, count, 1" in source
  assert "libgomp.so.1" not in source
  assert "GOMP_parallel" not in source
  assert "dlsym" not in source
  assert "RTLD_DEFAULT" not in source
  assert "state->pool ? state->pool : &state->torch_pool" in source
  assert "torch/extension.h" not in source
  assert "PYBIND11_MODULE" not in source
  returning_schema = _torch_schema("run", definition, return_outputs=True)
  assert returning_schema.endswith("-> Tensor")
  returning_source = _generate_torch_source("vecops_test", "run", definition, return_outputs=True)
  assert "at::Tensor operator()(" in returning_source
  assert "return p1;" in returning_source


def test_gnu_torch_bridge_filters_only_clang_gcc_install_selector(monkeypatch):
  from vecops import _schema_bridge

  toolchain = vecops._C.ToolchainSpec()
  toolchain.c_compiler = "/kernel/clang"
  toolchain.cxx_compiler = "/kernel/clang++"
  toolchain.environment = {
    "CXXFLAGS": (
      "--gcc-install-dir=/opt/gcc/14 --gcc-toolchain=/opt/gcc -nostdinc++ "
      "-isystem /private/include -L/private/lib -Wl,-rpath,/private/lib"
    ),
    "KEEP_ME": "unchanged",
  }
  config = vecops._C.KernelCompilerConfig()
  config.toolchain = toolchain
  monkeypatch.setenv("VECOPS_TORCH_BRIDGE_CC", "gcc")
  monkeypatch.setenv("VECOPS_TORCH_BRIDGE_CXX", "g++")
  monkeypatch.setattr(_schema_bridge, "_torch_openmp_runtime", lambda *_args: "gomp")
  monkeypatch.setattr(
    _schema_bridge.shutil,
    "which",
    lambda name: {"g++": "/opt/gcc/bin/g++", "gcc": "/opt/gcc/bin/gcc"}[name],
  )
  policy = _schema_bridge._torch_bridge_build_policy(config, (), "")
  bridge = policy.toolchain
  assert policy.compile_flags == ("-fopenmp",)
  assert str(bridge.cxx_compiler) == "/opt/gcc/bin/g++"
  normalized = bridge.environment["CXXFLAGS"]
  assert "--gcc-install-dir" not in normalized and "--gcc-toolchain" not in normalized
  for retained in (
    "-nostdinc++",
    "-isystem /private/include",
    "-L/private/lib",
    "-Wl,-rpath,/private/lib",
  ):
    assert retained in normalized
  assert bridge.environment["KEEP_ME"] == "unchanged"
  assert toolchain.environment["CXXFLAGS"].startswith("--gcc-install-dir=")


def test_gnu_torch_bridge_normalizes_split_clang_selector(monkeypatch):
  from vecops._schema_bridge import _gnu_compatible_cxxflags

  flags = "--gcc-install-dir /opt/gcc/14 --gcc-toolchain /opt/gcc -isystem /private/include -L /private/lib"
  normalized = _gnu_compatible_cxxflags(flags)
  assert "--gcc-install-dir" not in normalized
  assert "-isystem /private/include" in normalized
  assert "-L /private/lib" in normalized
  with pytest.raises(ValueError, match="without a path"):
    _gnu_compatible_cxxflags("-O2 --gcc-install-dir")


def test_cache_only_torch_bridge_loads_existing_artifact_without_writes(monkeypatch, tmp_path):
  from vecops import _schema_bridge

  class TorchStub:
    pass

  class CompilerStub:
    cache_mode = vecops._C.ArtifactCacheMode.cache_only

    def _native_config(self):
      pytest.fail("cache-only bridge lookup must not resolve compiler tools")

  base = tmp_path / "bridge-cache"
  source = "generated bridge source"
  registration = _schema_bridge._BridgeRegistration(
    "lib", "op", object(), object(), CompilerStub(), None, False, (), False
  )
  registration.ready[0] = False
  loaded = []
  monkeypatch.setenv("VECOPS_TORCH_BRIDGE_DIR", str(base))
  monkeypatch.setattr(
    _schema_bridge,
    "_torch_bridge_inputs",
    lambda: (TorchStub(), None, (), (), "", tmp_path / "native.so", ("torch-identity",)),
  )
  monkeypatch.setattr(_schema_bridge, "_generate_torch_source", lambda *_args, **_kwargs: source)
  monkeypatch.setattr(_schema_bridge, "_bridge_signature_identity", lambda _registration: "signature")

  def load_bridge(reg, path, *_args):
    loaded.append(path)
    reg.materialized = True

  monkeypatch.setattr(_schema_bridge, "_load_bridge", load_bridge)
  import vecops._identity as identity_module

  monkeypatch.setattr(identity_module, "compiler_identity", lambda _compiler: "compiler-attestation")
  monkeypatch.setattr(identity_module, "file_identity", lambda path: (str(path), f"{path}-content"))
  key = _schema_bridge._bridge_artifact_key(source, ("torch-identity",), "compiler-attestation", (), "signature")
  directory = base / key
  directory.mkdir(parents=True)
  output_name = f"vecops_bridge_{key}"
  (directory / f"{output_name}.so").write_bytes(b"bridge")
  manifest_identity = (
    source,
    ("torch-identity",),
    "signature",
    ("torch-bridge-v7", "compiler-attestation", (), None, None, "aten"),
  )
  (directory / f"{output_name}.identity").write_text(repr(manifest_identity))
  before = sorted((path.relative_to(tmp_path), path.stat().st_mtime_ns) for path in tmp_path.rglob("*"))
  monkeypatch.setattr(_schema_bridge, "_bridge_cache_base", lambda _registration: base)
  monkeypatch.setattr(
    _schema_bridge,
    "_compile_pending_torch_bridges_impl",
    lambda **_kwargs: pytest.fail("cache-only must never call the bridge compiler"),
  )
  monkeypatch.setattr(_schema_bridge, "_registrations", {("lib", "op"): registration})

  assert _schema_bridge.compile_pending_torch_bridges() == 0
  assert loaded == [directory / f"{output_name}.so"]
  assert registration.materialized
  assert sorted((path.relative_to(tmp_path), path.stat().st_mtime_ns) for path in tmp_path.rglob("*")) == before


def test_cache_only_torch_bridge_miss_is_import_error_without_creation(monkeypatch, tmp_path):
  from vecops import _schema_bridge

  class TorchStub:
    pass

  class CompilerStub:
    cache_mode = vecops._C.ArtifactCacheMode.cache_only

    def _native_config(self):
      pytest.fail("cache-only bridge miss must not resolve compiler tools")

  cache = tmp_path / "not-created"
  registration = _schema_bridge._BridgeRegistration(
    "lib", "missing", object(), object(), CompilerStub(), None, False, (), False
  )
  monkeypatch.setenv("VECOPS_TORCH_BRIDGE_DIR", str(cache))
  monkeypatch.setattr(
    _schema_bridge,
    "_torch_bridge_inputs",
    lambda: (TorchStub(), None, (), (), "", tmp_path / "native.so", ("torch-identity",)),
  )
  monkeypatch.setattr(_schema_bridge, "_generate_torch_source", lambda *_args, **_kwargs: "source")
  monkeypatch.setattr(_schema_bridge, "_bridge_signature_identity", lambda _registration: "signature")
  import vecops._identity as identity_module

  monkeypatch.setattr(identity_module, "compiler_identity", lambda _compiler: "compiler-attestation")
  monkeypatch.setattr(
    _schema_bridge,
    "_torch_bridge_build_policy",
    lambda config, *_args: _schema_bridge._TorchBridgeBuildPolicy(config.toolchain),
  )
  monkeypatch.setattr(_schema_bridge, "_registrations", {("lib", "missing"): registration})
  monkeypatch.setattr(
    _schema_bridge,
    "_compile_pending_torch_bridges_impl",
    lambda **_kwargs: pytest.fail("cache-only must never compile a missing bridge"),
  )
  with pytest.raises(ImportError, match="cache-only Torch bridge.*not found"):
    _schema_bridge.compile_pending_torch_bridges()
  assert not cache.exists()


def test_native_thread_pool_operator_filter(monkeypatch) -> None:
  from vecops import _schema_bridge

  class TorchStub:
    @staticmethod
    def get_num_threads():
      return 7

  monkeypatch.setattr(
    _schema_bridge.native,
    "native_thread_pool_address",
    lambda threads: 1234,
    raising=False,
  )
  monkeypatch.setattr(_schema_bridge, "_native_thread_pool", None)
  monkeypatch.setenv("VECOPS_THREAD_POOL", "native")
  monkeypatch.setenv(
    "VECOPS_NATIVE_THREAD_POOL_OPS",
    "alphafold3_native::diffusion_*,other::exact",
  )
  assert not _schema_bridge._thread_pool_pointer(TorchStub(), "alphafold3_native::linear_packed").value
  assert _schema_bridge._thread_pool_pointer(TorchStub(), "alphafold3_native::diffusion_self_attention").value == 1234
  assert _schema_bridge._thread_pool_pointer(TorchStub(), "other::exact").value == 1234


def test_native_thread_pool_region_is_scoped_and_nestable(monkeypatch) -> None:
  events = []
  monkeypatch.setattr(
    vecops._C,
    "native_thread_pool_begin_active",
    lambda threads: events.append(("begin", threads)),
    raising=False,
  )
  monkeypatch.setattr(
    vecops._C,
    "native_thread_pool_end_active",
    lambda: events.append(("end", None)),
    raising=False,
  )
  with vecops.native_thread_pool_region(7):
    with vecops.native_thread_pool_region(7):
      events.append(("body", None))
  assert events == [
    ("begin", 7),
    ("begin", 7),
    ("body", None),
    ("end", None),
    ("end", None),
  ]


def test_native_thread_pool_region_can_be_disabled(monkeypatch) -> None:
  monkeypatch.setattr(
    vecops._C,
    "native_thread_pool_begin_active",
    lambda threads: pytest.fail(f"unexpected begin for {threads}"),
    raising=False,
  )
  with vecops.native_thread_pool_region(0, enabled=False):
    pass


def test_python_sdk_core_target_does_not_propagate_openmp() -> None:
  sdk_targets = (VECOPS_ROOT / "cmake" / "VecopsPythonSdkTargets.cmake.in").read_text()
  assert "OpenMP::OpenMP_CXX" not in sdk_targets
  assert 'INTERFACE_LINK_LIBRARIES "${CMAKE_DL_LIBS}"' in sdk_targets


def test_compile_batch_validates_parallelism() -> None:
  assert vecops.compile_batch([], parallelism=3).parallelism == 0
  with pytest.raises(ValueError, match="positive integer"):
    vecops.compile_batch([], parallelism=0)
  with pytest.raises(ValueError, match="positive integer"):
    vecops.compile_batch([], parallelism=True)


def test_native_build_api_mismatch_has_repair_command(monkeypatch) -> None:
  from vecops._compiler import require_native_build_api

  monkeypatch.setattr(vecops._C, "build_api_version", 0)
  with pytest.raises(RuntimeError, match=r"pip install -e .* --no-build-isolation"):
    require_native_build_api()


def test_compile_batch_delegates_one_native_build_graph(monkeypatch) -> None:
  submitted = []

  def prepare_batch(requests, *, parallel_jobs, execution_parallelism):
    submitted.append((requests, parallel_jobs, execution_parallelism))

  monkeypatch.setattr(vecops._C, "prepare_batch", prepare_batch)

  requests = [
    vecops.CompileRequest(
      operator=object(),
      call=vecops._C.KernelCall([]),
      key=(index,),
      name=f"request-{index}",
    )
    for index in range(2)
  ]
  result = vecops.compile_batch(requests, parallelism=2, execution_parallelism=7)
  assert result.prepared == 2
  assert result.parallelism == 2
  assert len(submitted) == 1
  assert submitted[0][1] == 2
  assert submitted[0][2] == 7
  assert [item[0] for item in submitted[0][0]] == [request.operator for request in requests]
