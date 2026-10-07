"""Compiler snapshots isolate prepared kernels from mutable user policy."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import vecops.language as T
from vecops._arm_vector_state import VectorState
from vecops._compiler import Compiler
from vecops.language._compiler import driver
from vecops.language._compiler.cache import _compiler_key, _compiler_policy_key
from vecops.language._debug.report import DebugInfo


@pytest.mark.parametrize("state", [VectorState(None, None), VectorState(32, 64)])
def test_snapshot_is_independent_of_mutable_compiler_policy(monkeypatch, state):
  monkeypatch.setattr("vecops._arm_vector_state.get_vector_state", lambda: state)
  compiler = Compiler(
    target="Native",
    cflags=("-O2",),
    cache_mode="cache-only",
    environment={"VECOPS_TEST_POLICY": "original"},
  )
  original_backend = object()
  compiler._native_compiler = original_backend

  frozen = compiler.snapshot()
  frozen_key = _compiler_key(frozen)
  original_policy = _compiler_policy_key(compiler)
  frozen_vector_state = frozen.vector_state
  assert frozen is not compiler
  assert frozen.snapshot() is frozen
  assert frozen._native_compiler is None
  assert frozen.environment is not compiler.environment

  compiler.cflags = ("-O3",)
  compiler.environment["VECOPS_TEST_POLICY"] = "mutated-original"
  assert frozen.cflags == ("-O2",)
  assert frozen.environment["VECOPS_TEST_POLICY"] == "original"
  assert _compiler_key(frozen) == frozen_key
  assert _compiler_policy_key(compiler) != original_policy
  assert compiler._native_compiler is original_backend

  changed_state = VectorState(64, 32)
  monkeypatch.setattr("vecops._arm_vector_state.get_vector_state", lambda: changed_state)
  assert frozen.vector_state == frozen_vector_state
  assert _compiler_key(frozen) == frozen_key
  assert not frozen.is_environment_current()


@pytest.mark.parametrize("state", [VectorState(None, None), VectorState(64, 32)])
def test_jit_cache_validator_tracks_source_compiler_policy(monkeypatch, tmp_path, state):
  """The cache guard follows the caller's policy; the built core owns a copy."""
  monkeypatch.setattr("vecops._arm_vector_state.get_vector_state", lambda: state)
  monkeypatch.setattr(driver, "_source_identity", lambda _ir: "source-id")
  monkeypatch.setattr(driver, "_build_definition", lambda *_args, **_kwargs: (object(), {}, ()))
  monkeypatch.setattr(driver, "_CompiledCore", type("FakeCore", (), {"__init__": lambda self, *args: None}))
  monkeypatch.setattr(driver, "CompiledKernel", lambda _core, _template: SimpleNamespace())
  monkeypatch.setattr(driver, "Operator", lambda *_args, **_kwargs: SimpleNamespace())

  from vecops.language._debug import report

  monkeypatch.setattr(
    report,
    "lower",
    lambda *_args, **_kwargs: SimpleNamespace(artifact=None, debug=DebugInfo()),
  )

  import vecops._identity as identity

  monkeypatch.setattr(identity, "compiler_identity", lambda _compiler: "build-id")
  monkeypatch.setattr(identity, "implementation_identity", lambda: "implementation-id")
  monkeypatch.setattr(identity, "include_roots", lambda _compiler: ())
  monkeypatch.setattr(identity, "source_inputs", lambda *_args, **_kwargs: (("source", "same"),))

  source = tmp_path / "kernel.cpp"
  source.write_text("// mocked source input\n")
  template = T.template(source=source)(lambda: None)
  compiler = Compiler(target="Native", cflags=("-ffp-contract=off",), cache_dir=tmp_path / "cache")
  driver._compile_cache.clear()
  result = driver.compile(template, compiler=compiler)

  assert result.debug._validate_cache()
  frozen = result.debug.compiler
  assert frozen is not compiler
  compiler.cflags = ("-fno-math-errno",)
  assert frozen.cflags == ("-ffp-contract=off",)
  assert not result.debug._validate_cache()
