import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import vecops
from vecops import _identity, _operator


def test_trace_writer_is_opt_in_and_appends_jsonl(tmp_path, monkeypatch):
  record = {"event": "probe", "digest": "abc123"}
  path = tmp_path / "trace.jsonl"

  monkeypatch.delenv("VECOPS_BUILD_INPUT_TRACE", raising=False)
  _identity._write_build_input_trace(record)
  assert not path.exists()

  monkeypatch.setenv("VECOPS_BUILD_INPUT_TRACE", str(path))
  _identity._write_build_input_trace(record)
  _identity._write_build_input_trace({"event": "second"})
  assert [json.loads(line) for line in path.read_text().splitlines()] == [record, {"event": "second"}]


def test_trace_path_must_be_absolute_and_parent_must_exist(tmp_path, monkeypatch):
  monkeypatch.setenv("VECOPS_BUILD_INPUT_TRACE", "relative.jsonl")
  with pytest.raises(ValueError, match="absolute"):
    _identity._write_build_input_trace({"event": "probe"})

  missing_parent = tmp_path / "missing" / "trace.jsonl"
  monkeypatch.setenv("VECOPS_BUILD_INPUT_TRACE", str(missing_parent))
  with pytest.raises(ValueError, match="parent directory"):
    _identity._write_build_input_trace({"event": "probe"})


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires POSIX fork")
def test_trace_lock_is_reset_after_fork(tmp_path, monkeypatch):
  path = tmp_path / "fork-trace.jsonl"
  monkeypatch.setenv("VECOPS_BUILD_INPUT_TRACE", str(path))
  _identity._trace_lock.acquire()
  try:
    child = os.fork()
    if child == 0:
      _identity._write_build_input_trace({"event": "child"})
      os._exit(0)
    _, status = os.waitpid(child, 0)
    assert os.waitstatus_to_exitcode(status) == 0
  finally:
    _identity._trace_lock.release()
  assert json.loads(path.read_text()) == {"event": "child"}


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires POSIX fork")
def test_trace_records_are_not_interleaved_between_processes(tmp_path, monkeypatch):
  path = tmp_path / "multiprocess-trace.jsonl"
  monkeypatch.setenv("VECOPS_BUILD_INPUT_TRACE", str(path))
  children = []
  for process_index in range(3):
    child = os.fork()
    if child == 0:
      for record_index in range(24):
        _identity._write_build_input_trace({"process": process_index, "record": record_index})
      os._exit(0)
    children.append(child)
  for child in children:
    _, status = os.waitpid(child, 0)
    assert os.waitstatus_to_exitcode(status) == 0

  records = [json.loads(line) for line in path.read_text().splitlines()]
  assert len(records) == 72
  assert len({(record["process"], record["record"]) for record in records}) == 72


def test_source_mismatch_trace_preserves_compiler_identity_short_circuit(tmp_path, monkeypatch):
  source = tmp_path / "kernel.cpp"
  source.write_text("void kernel() {}\n")
  compiler = vecops.Compiler(target="Scalar", cache_dir=tmp_path / "cache")
  kernel_def = vecops.KernelDef("trace_test", [])
  old = (((str(source), "before"),), ())
  new = (((str(source), "after"),), ())
  current = {"inputs": old}
  calls = []
  native_operator_args = {}

  monkeypatch.setattr(_identity, "include_roots", lambda _compiler: ())
  monkeypatch.setattr(_identity, "source_inputs", lambda *_args, **_kwargs: current["inputs"])
  monkeypatch.setattr(_identity, "compiler_identity", lambda _compiler: calls.append("identity") or "id")

  class FakeNativeOperator:
    def __init__(self, *args):
      native_operator_args["validate"] = args[-1]

  monkeypatch.setattr(_operator._C, "Operator", FakeNativeOperator)
  trace = tmp_path / "trace.jsonl"
  monkeypatch.setenv("VECOPS_BUILD_INPUT_TRACE", str(trace))
  _operator.Operator(source, kernel_def, compiler, build_identity="expected")

  current["inputs"] = new
  assert native_operator_args["validate"]() is False
  assert calls == []
  record = json.loads(trace.read_text().splitlines()[-1])
  assert record["event"] == "operator_input_validation"
  assert record["source_inputs_equal"] is False
  assert record["compiler_identity_equal"] is None
  assert record["source_diff"] == {
    "changed_count": 1,
    "paths": [{"path": str(source), "before": "before", "after": "after"}],
  }


def test_cache_only_trace_does_not_write_or_resolve_tools(tmp_path, monkeypatch):
  compiler = vecops.Compiler(target="Scalar", cache_dir=tmp_path / "cache", cache_mode="cache-only")
  monkeypatch.setattr(_identity, "implementation_identity", lambda: "sdk-id")
  monkeypatch.setattr(_identity, "file_identity", lambda path: (str(path), "file-id"))
  trace = tmp_path / "trace.jsonl"
  monkeypatch.setenv("VECOPS_BUILD_INPUT_TRACE", str(trace))

  def unexpected(*_args, **_kwargs):
    pytest.fail("cache-only identity lookup must not discover tools")

  monkeypatch.setattr("vecops._compiler._tool", unexpected)
  with pytest.raises(ImportError, match="no matching build-input attestation"):
    _identity.compiler_identity(compiler)
  assert not trace.exists()
  assert not (compiler.cache_dir / "identities").exists()


def test_compiler_trace_preserves_identity_and_enriches_legacy_attestation(tmp_path, monkeypatch):
  compiler = vecops.Compiler(target="Scalar", cache_dir=tmp_path / "cache")
  monkeypatch.setattr(_identity, "implementation_identity", lambda: "sdk-id")
  monkeypatch.setattr(_identity, "file_identity", lambda path: (str(path), "file-id"))
  monkeypatch.setattr(_identity, "tree_identity", lambda _path: ())
  monkeypatch.setattr("vecops._compiler._tool", lambda configured, _var, _choices: Path(configured or "/mock/tool"))
  monkeypatch.setattr(
    subprocess,
    "run",
    lambda command, **_kwargs: SimpleNamespace(
      stderr="#include <...> search starts here:\nEnd of search list\n",
      stdout="/missing/runtime.so" if "-print-file-name=" in command[-1] else "",
    ),
  )
  monkeypatch.setattr("shutil.which", lambda _name: None)

  trace = tmp_path / "trace.jsonl"
  monkeypatch.delenv("VECOPS_BUILD_INPUT_TRACE", raising=False)
  identity_without_trace = _identity.compiler_identity(compiler)
  monkeypatch.setenv("VECOPS_BUILD_INPUT_TRACE", str(trace))
  monkeypatch.setenv("PRIVATE_BUILD_SECRET", "must-not-be-logged")
  identity_with_trace = _identity.compiler_identity(compiler)
  assert identity_with_trace == identity_without_trace

  record = json.loads(trace.read_text().splitlines()[-1])
  assert record["event"] == "compiler_identity"
  assert record["identity"] == identity_without_trace
  assert {"config_key", "tools", "system_roots", "linked_runtime", "hardware_digest", "generator"} <= set(
    record["components"]
  )
  assert "must-not-be-logged" not in trace.read_text()

  attestation = next((compiler.cache_dir / "identities").glob("*.json"))
  records = json.loads(attestation.read_text())
  records[0].pop("components")
  attestation.write_text(json.dumps(records))
  trace.unlink()
  cache_only = vecops.Compiler(
    target="Scalar", cache_dir=compiler.cache_dir, cache_mode="cache-only", cc=compiler.cc, cxx=compiler.cxx
  )
  assert _identity.compiler_identity(cache_only) == identity_without_trace
  assert not trace.exists()
  assert "components" not in json.loads(attestation.read_text())[0]
