"""The generated DSL target, not source C++ or global flags, owns FP policy."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import vecops
import vecops.language as T
from dsl_test_utils import make_compiler
from vecops._arm_vector_state import VectorState
from vecops._compiler import Compiler
from vecops.language._compiler import driver
from vecops.language._compiler.cache import _compiler_key
from vecops.language._debug.report import DebugInfo


@pytest.mark.parametrize("state", (VectorState(None, None), VectorState(32, 64)))
def test_strict_dsl_target_is_an_independent_frozen_compiler(monkeypatch, state):
  monkeypatch.setattr("vecops._arm_vector_state.get_vector_state", lambda: state)
  original = Compiler(cflags=("-O3", "-ffp-contract=fast"), environment={"CXXFLAGS": "-fno-math-errno"})
  frozen = original.snapshot()
  handle = object()
  frozen._native_compiler = handle
  before = _compiler_key(frozen)
  strict = driver._strict_dsl_compiler(frozen)
  assert strict is not frozen and strict.snapshot() is strict
  assert strict.cflags == ("-O3", "-ffp-contract=fast", "-ffp-contract=off")
  assert original.cflags == frozen.cflags == ("-O3", "-ffp-contract=fast")
  assert strict.environment == frozen.environment
  assert strict.environment is not frozen.environment
  assert strict.vector_state == frozen.vector_state == state
  assert strict.build_events is frozen.build_events is original.build_events
  assert strict._native_compiler is None and frozen._native_compiler is handle
  assert _compiler_key(strict) != before == _compiler_key(frozen)


@pytest.mark.parametrize("generated", (False, True))
def test_compile_policy_only_applies_to_generated_dsl(monkeypatch, tmp_path, generated):
  """Exercise the driver branch while keeping this policy test build-free."""
  from vecops.language._debug import report

  source = tmp_path / "source.cpp"
  source.write_text("// C++ reference, untouched FP policy\n")

  def kernel():
    pass

  template = T.template(kernel) if generated else T.template(source=source)(kernel)
  compiler = Compiler(cflags=("-ffp-contract=on",), cache_dir=tmp_path / "cache")
  monkeypatch.setattr(driver, "_source_identity", lambda _ir: "fp-policy-source")
  monkeypatch.setattr(driver, "_build_definition", lambda *_args, **_kwargs: (object(), {}, ()))
  monkeypatch.setattr(driver, "_CompiledCore", type("FakeCore", (), {"__init__": lambda self, *args: None}))
  monkeypatch.setattr(driver, "CompiledKernel", lambda _core, _template: SimpleNamespace())
  monkeypatch.setattr(driver, "Operator", lambda *_args, **_kwargs: SimpleNamespace())
  monkeypatch.setattr(
    report,
    "lower",
    lambda *_args, **_kwargs: SimpleNamespace(
      artifact=SimpleNamespace(
        kernel=SimpleNamespace(diagnostics=()), lowered=None, backend_ir=None, source="// generated DSL\n"
      ),
      debug=DebugInfo(),
    ),
  )
  import vecops._identity as identity

  monkeypatch.setattr(identity, "compiler_identity", lambda selected: repr(selected.cflags))
  monkeypatch.setattr(identity, "implementation_identity", lambda: "fp-policy-implementation")
  monkeypatch.setattr(identity, "include_roots", lambda _compiler: ())
  monkeypatch.setattr(identity, "source_inputs", lambda *_args, **_kwargs: (("source", "same"),))
  driver._compile_cache.clear()
  result = driver.compile(template, compiler=compiler)
  assert result.debug.compiler.cflags == (
    ("-ffp-contract=on", "-ffp-contract=off") if generated else ("-ffp-contract=on",)
  )
  assert compiler.cflags == ("-ffp-contract=on",)
  assert result.debug._validate_cache()
  compiler.cflags = ("-O2",)
  assert not result.debug._validate_cache()


def _pointwise(*, path, fma, output):
  descriptor = T.Vector("float32", N=37)

  @T.template
  def kernel(
    A: T.Tensor((37,), dtype="float64"),
    Scale: T.Tensor((37,), dtype="float32"),
    Bias: T.Tensor((37,), dtype="float32"),
    C: T.Tensor((37,), dtype=output, access="output"),
  ):
    if path == "scalar":
      for i in T.serial(37, unroll=2):
        x = T.load(A[i], dtype="float32")
        scale, bias = T.load(Scale[i]), T.load(Bias[i])
        C[i] = T.fma(x, scale, bias) if fma else x * scale + bias
    elif path == "vector":
      x = T.load(A, dtype="float32", vector=descriptor)
      scale = T.load(Scale, vector=descriptor)
      bias = T.load(Bias, vector=descriptor)
      T.store(C, T.fma(x, scale, bias) if fma else x * scale + bias)
    else:
      x, scale, bias = T.load(A, dtype="float32"), T.load(Scale), T.load(Bias)
      T.store(C, T.fma(x, scale, bias) if fma else x * scale + bias)

  return kernel


@pytest.fixture(scope="module")
def pointwise_kernels(tmp_path_factory):
  root = tmp_path_factory.mktemp("dsl-fp-contract")
  compiler = make_compiler(
    target="Native", jobs=4, cache_dir=root / "cache", build_dir=root / "build", cflags=("-ffp-contract=fast",)
  )
  result = {}
  with vecops.collect_compile_requests() as collector:
    for path in ("scalar", "vector", "tile"):
      for fma in (False, True):
        for output in ("float32", "uint8"):
          compiled = T.compile(_pointwise(path=path, fma=fma, output=output), compiler=compiler)
          result[path, fma, output] = compiled
          compiled.operator(
            vecops.TensorMeta((37,), dtype=vecops.float64),
            vecops.TensorMeta((37,), dtype=vecops.float32),
            vecops.TensorMeta((37,), dtype=vecops.float32),
            vecops.TensorMeta((37,), dtype=getattr(vecops, output)),
            **compiled._core.specialization_values,
          )
  vecops.compile_batch(collector.requests, parallelism=4, execution_parallelism=1)
  return result


@pytest.mark.parametrize("path", ("scalar", "vector", "tile"))
@pytest.mark.parametrize("fma", (False, True))
@pytest.mark.parametrize("output", ("float32", "uint8"))
def test_pointwise_native_strict_and_explicit_fma(pointwise_kernels, path, fma, output):
  compiled = pointwise_kernels[path, fma, output]
  # Exactly representable F32 inputs, including a separating rounding case.
  a = np.resize(np.array([-1.0000001192092896, np.nan, np.inf, -np.inf, 0, 126, 127, 128, -4], dtype="float64"), 37)
  scale = np.full(37, 1.9999998807907104, dtype="float32")
  bias = np.full(37, 3, dtype="float32")
  if output == "uint8":
    # Keep this FP-contraction oracle independent of backend nonfinite float-
    # to-integer conventions. F32 output still probes NaN and both infinities.
    # Scalar C++ casts additionally require their result in the integer range;
    # vector/tile output exercises finite saturating narrowing as well.
    values = (
      [-1.0000001192092896, 0, 1, 2, 8, 16, 63, 64, 126]
      if path == "scalar"
      else [-1.0000001192092896, -1000, -4, 0, 126, 127, 128, 255, 1000]
    )
    a = np.resize(np.array(values, dtype="float64"), 37)
  with np.errstate(invalid="ignore", over="ignore"):
    x = a.astype("float32")
    expected = (
      np.float32(x.astype("float64") * scale.astype("float64") + bias.astype("float64"))
      if fma
      else np.float32(np.float32(x * scale) + bias)
    )
    assert int(expected[0]) == (0 if fma else 1)
    if output == "uint8":
      expected = np.clip(np.trunc(np.nan_to_num(expected, nan=0, posinf=255, neginf=0)), 0, 255).astype("uint8")
  c = np.empty(37, dtype=output)
  compiled.operator.native(a, scale, bias, c, **compiled._core.specialization_values)
  np.testing.assert_array_equal(c, expected)
  commands = compiled.debug.artifacts()
  assert commands, "retain the actual per-kernel compile command, not a source-only assertion"
  assert all("-ffp-contract=off" in command["command"]["command"] for command in commands)
  assert compiled.operator.compiler.cflags[-1] == "-ffp-contract=off"
