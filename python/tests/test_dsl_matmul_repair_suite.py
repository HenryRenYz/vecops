# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Reuse never follows an execution ID without source and ABI evidence."""

import importlib.util
import io
import json
import sys
import tarfile
from pathlib import Path

import pytest

BENCH = Path(__file__).resolve().parents[2] / "benchmarks"
sys.path.insert(0, str(BENCH))
spec = importlib.util.spec_from_file_location("repair_test", BENCH / "dsl_matmul_repair_suite.py")
repair = importlib.util.module_from_spec(spec)
spec.loader.exec_module(repair)


def test_only_leading_compiler_environment_comment_is_normalized():
  source = b"// compiler environment: abc123\nint x;\n"
  assert repair.normalized(source) == b"int x;\n"
  assert repair.normalized(b"int x;\n" + source) == b"int x;\n" + source
  assert repair.normalized(b"// compiler environment: invalid!\nint x;\n").startswith(b"//")


def test_reuse_requires_exact_remaining_body_and_full_abi(monkeypatch):
  import dsl_matmul_artifact_ab as ab

  monkeypatch.setattr(ab, "adapter_contract", lambda _: (("dtype", "rank", "inout", "bounds", "equality"),))
  contract = (("dtype", "rank", "inout", "bounds", "equality"),)
  candidate = {
    "binary": "old.so",
    "binary_sha256": "sha",
    "source": b"// compiler environment: aa\nbody\n",
    "adapter": "schema",
    "original_binary": "task.so",
    "archive": "evidence.gz",
    "archive_sha256": "sha",
    "execution_ids": ["row"],
  }
  proof = repair.prove_reuse(b"// compiler environment: bb\nbody\n", candidate, contract)
  assert proof["old_raw_sha256"] != proof["new_raw_sha256"]
  assert repair.prove_reuse(b"changed\n", candidate, contract) is None
  assert repair.prove_reuse(b"body\n", candidate, (("dtype", "rank", "readonly", "bounds", "equality"),)) is None


def test_exact_archive_path_does_not_guess_by_basename():
  members = {"work/a/kernel.cpp": b"a", "work/b/kernel.cpp": b"b"}
  assert repair.archived_path(members, "/old/work/a/kernel.cpp") == b"a"
  with pytest.raises(ValueError, match="exact original"):
    repair.archived_path(members, "/different/kernel.cpp")


def test_failed_group_orphan_binary_can_be_reused_only_with_task_mapping(tmp_path):
  group = tmp_path / "groups/g"
  group.mkdir(parents=True)
  binary = group / "retained.so"
  binary.write_bytes(b"elf")
  original = str(group / "work/build/kernel-batch-x/artifacts/0/lib.so")
  source = str(group / "work/dsl-sources/hash/kernel.cpp")
  adapter = b'#include "' + source.encode() + b'"\n'
  archive = group / "evidence.tar.gz"
  with tarfile.open(archive, "w:gz") as stream:
    for name, data in {
      "work/build/kernel-batch-x/tasks/0/input/vecops_adapter.cpp": adapter,
      "work/dsl-sources/hash/kernel.cpp": b"body",
    }.items():
      info = tarfile.TarInfo(name)
      info.size = len(data)
      stream.addfile(info, io.BytesIO(data))
  (tmp_path / "plan.json").write_text(json.dumps({"groups": [{"id": "g", "execution_ids": ["row"]}]}))
  evidence = {
    "evidence_archive": str(archive),
    "evidence_sha256": repair.digest(archive.read_bytes()),
    "retained_binary_mapping": {original: str(binary)},
    "retained_binary_sha256": {str(binary): repair.digest(binary.read_bytes())},
  }
  (group / "failure-evidence.json").write_text(json.dumps(evidence))
  found = list(repair.candidates(tmp_path, ["row"]))
  assert len(found) == 1
  assert found[0]["source"] == b"body"
  binary.write_bytes(b"changed")
  with pytest.raises(ValueError, match="binary changed"):
    list(repair.candidates(tmp_path, ["row"]))


def test_archive_integrity_is_checked_before_evidence_read(tmp_path):
  archive = tmp_path / "bad.gz"
  archive.write_bytes(b"changed")
  with pytest.raises(ValueError, match="archive changed"):
    repair.archive_members({"evidence_archive": str(archive), "evidence_sha256": "wrong"})


def test_runtime_identity_covers_python_sdk_and_extension_not_harness(tmp_path):
  for name in ("python/vecops/a.py", "python/vecops/_C.so", "include/api.h", "benchmarks/ref.cpp"):
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"source")
  first = repair.runtime_identity(tmp_path)
  assert set(first) == {"python/vecops/a.py", "python/vecops/_C.so", "include/api.h"}
  (tmp_path / "include/api.h").write_bytes(b"different")
  assert repair.runtime_identity(tmp_path) != first


def test_partial_chunk_retains_successful_roles_without_missing_none_poison(tmp_path, monkeypatch):
  from types import SimpleNamespace

  group = {"id": "g", "execution_ids": ["row"]}
  directory = tmp_path / "groups/g"
  directory.mkdir(parents=True)
  work = tmp_path / "chunks/0/work"
  work.mkdir(parents=True)
  (work / ".full-suite-owned-work").write_text("repair")
  binary = work / "good.so"
  binary.write_bytes(b"successful module")

  def index_partial(*_):
    repair.suite.save(
      directory / "index.json", {"artifacts": {"row": {"dsl": str(binary), "cpp": None}}, "schema_audit": {"row": {}}}
    )

  monkeypatch.setattr(repair, "index_partial", index_partial)
  repair.finalize_chunk(
    SimpleNamespace(suite_dir=tmp_path),
    None,
    {"runner_hashes": {"ref": "new"}},
    [group],
    work,
    [{}],
    "one module failed",
  )
  index = repair.suite.read(directory / "index.json")
  assert Path(index["artifacts"]["row"]["dsl"]).read_bytes() == b"successful module"
  assert index["artifacts"]["row"]["cpp"] is None
  assert repair.suite.read(directory / "build.json")["status"] == "failed"
  assert set(repair.suite.read(directory / "completed-roles.json")) == {"dsl"}
  assert not work.exists()


def test_prelowered_operator_preserves_runtime_abi_flags_without_lowering(tmp_path, monkeypatch):
  import dsl_matmul_artifact_ab as ab
  import vecops
  import vecops._operator as operator_module
  import vecops.language as T

  @T.template
  def fixture(A: T.Tensor((4,), dtype=T.float32), C: T.Tensor((4,), dtype=T.float32, access="output")):
    T.store(C, T.load(A))

  compiler = vecops.Compiler(cache_dir=tmp_path / "cache", build_dir=tmp_path / "build", cflags=["-DTEST"])
  source = tmp_path / "kernel.cpp"
  source.write_bytes(b"proved generated source")
  contract, _ = ab.template_contract(fixture)
  proof = {
    "source": str(source),
    "raw_sha256": repair.digest(source.read_bytes()),
    "contract": contract,
    "effective_cflags": ["-DTEST", "-ffp-contract=off", "-mllvm=-simplifycfg-sink-common=false"],
    "vector_state_token": compiler.snapshot().vector_state.env_token,
  }
  observed = {}

  def operator(source, definition, effective, **kwargs):
    observed.update(source=source, definition=definition, flags=effective.cflags, kwargs=kwargs)
    return "source operator"

  monkeypatch.setattr(operator_module, "Operator", operator)
  monkeypatch.setattr(T, "lower", lambda *_args, **_kwargs: pytest.fail("must not lower twice"))
  result = repair.compile_prelowered(fixture, proof, compiler)
  assert result.operator == "source operator"
  assert result.kernel_def.runtime_parallelism is True
  assert observed["flags"] == tuple(proof["effective_cflags"])
  assert observed["kwargs"]["dependencies"] == fixture.ir.dependencies
  assert compiler.cflags == ["-DTEST"] or compiler.cflags == ("-DTEST",)
  bad = proof | {"vector_state_token": "different"}
  with pytest.raises(ValueError, match="vector state"):
    repair.compile_prelowered(fixture, bad, compiler)
  bad = proof | {"contract": ["different ABI"]}
  with pytest.raises(ValueError, match="ABI"):
    repair.compile_prelowered(fixture, bad, compiler)
  source.write_bytes(b"changed")
  with pytest.raises(ValueError, match="source changed"):
    repair.compile_prelowered(fixture, proof, compiler)


@pytest.mark.parametrize("compress", (False, True))
@pytest.mark.parametrize("drift", (False, True))
def test_pure_lower_worker_does_not_attest_toolchain_or_watch_sdk(tmp_path, monkeypatch, compress, drift):
  from types import SimpleNamespace

  import dsl_matmul_artifact_ab as ab
  import vecops._identity as identity
  import vecops.language as T

  @T.template
  def fixture(A: T.Tensor((4,), dtype=T.float32), C: T.Tensor((4,), dtype=T.float32, access="output")):
    T.store(C, T.load(A))

  def forbidden(*_args, **_kwargs):
    pytest.fail("pure lowering must not discover tools or attest/watch SDK inputs")

  monkeypatch.setattr(identity, "compiler_identity", forbidden)
  monkeypatch.setattr(identity, "source_inputs", forbidden)
  monkeypatch.setattr(ab, "source_scenarios", lambda _: [SimpleNamespace(execution_id="row")])
  plan = {
    "groups": [{"id": "g", "execution_ids": ["row"]}],
    "executions": [{"execution_id": "row"}],
    "runner_hashes": {},
    "pass_options": {},
    "target": "Generic",
    "toolchain": {"cc": None, "cxx": None},
    "compiler_cflags": [],
  }
  source = SimpleNamespace(source_templates=lambda _: ((fixture, None), None))
  guards = iter(({"new": {}}, {"new": {"changed.py": "sha"}} if drift else {"new": {}}))
  monkeypatch.setattr(repair, "guard_runtime", lambda _: next(guards))
  if drift:
    with pytest.raises(ValueError, match="changed during lowering"):
      repair.lower_worker(SimpleNamespace(group_index=0, suite_dir=tmp_path, compress_prelower=compress), source, plan)
    assert not (tmp_path / "groups/g/lower-proof.json").exists()
    return
  repair.lower_worker(SimpleNamespace(group_index=0, suite_dir=tmp_path, compress_prelower=compress), source, plan)
  proof = repair.suite.read(tmp_path / "groups/g/lower-proof.json")["dsl"]
  assert proof["effective_cflags"] == ["-ffp-contract=off"]
  assert proof["runtime_parallelism"] is True


def test_compressed_and_plain_source_materialize_exact_same_native_input(tmp_path):
  import gzip

  data = b"// compiler environment: aa\nexact kernel body\n"
  plain = tmp_path / "plain.cpp"
  plain.write_bytes(data)
  compressed = tmp_path / "kernel.cpp.gz"
  compressed.write_bytes(gzip.compress(data, mtime=0))
  proof = {"source": str(plain), "raw_sha256": repair.digest(data)}
  zipped = proof | {
    "source": str(compressed),
    "compression": "gzip",
    "compressed_sha256": repair.digest(compressed.read_bytes()),
  }
  assert repair.source_bytes(proof) == repair.source_bytes(zipped) == data
  work = tmp_path / "work"
  first = repair.materialize_proof(proof, work)
  second = repair.materialize_proof(zipped, work)
  assert first == second
  assert Path(second["source"]).read_bytes() == data
  assert Path(second["source"]).parent.name == repair.digest(data)
  compressed.write_bytes(b"tampered gzip")
  with pytest.raises(ValueError, match="compressed prelowered source changed"):
    repair.source_bytes(zipped)


def test_cpp_source_operator_matches_normal_definition_without_cache_identity(tmp_path, monkeypatch):
  from dataclasses import replace

  import vecops
  import vecops._operator as operator_module
  import vecops.language as T
  from vecops.language._compiler.backends.cpu.source import _build_definition
  from vecops.language.template import TemplateFunction

  @T.template
  def fixture(A: T.Tensor((4,), dtype=T.float32), C: T.Tensor((4,), dtype=T.float32, access="output")):
    T.store(C, T.load(A))

  path = tmp_path / "reference.cpp"
  path.write_text("original reference body")
  dependency = tmp_path / "dependency.h"
  dependency.write_text("original dependent header")
  template = TemplateFunction(fixture.function, replace(fixture.ir, source=path, dependencies=(dependency,)))
  compiler = vecops.Compiler(cache_dir=tmp_path / "cache", build_dir=tmp_path / "build", cflags=["-DTEST"])
  expected, values, order = _build_definition(template.ir, (), runtime_parallelism=False)
  observed = {}

  def operator(source, definition, selected, **kwargs):
    observed.update(source=source, definition=definition, flags=selected.cflags, kwargs=kwargs)
    return "normal attesting operator"

  monkeypatch.setattr(operator_module, "Operator", operator)
  monkeypatch.setattr(T, "compile", lambda *_args, **_kwargs: pytest.fail("redundant T.compile cache must be skipped"))
  result = repair.compile_cpp_source(template, compiler)
  assert result.operator == "normal attesting operator"
  assert observed["source"] == path
  assert result.kernel_def.runtime_parallelism is False
  assert [p.name for p in result.kernel_def.parameters] == [p.name for p in expected.parameters]
  assert result._core.specialization_values == values
  assert result._core.specialization_order == order
  assert observed["flags"] == compiler.snapshot().cflags
  assert observed["kwargs"]["dependencies"] == (dependency,)
  assert "build_identity" not in observed["kwargs"]
  with pytest.raises(ValueError, match="explicit source"):
    repair.compile_cpp_source(fixture, compiler)


def test_batch_policies_share_only_real_full_identity_and_guard_latches_failure(tmp_path, monkeypatch):
  import vecops
  import vecops._identity as identity
  import vecops.language as T
  from vecops.language._compiler.backends.cpu.source import _build_definition

  @T.template
  def fixture(A: T.Tensor((4,), dtype=T.float32), C: T.Tensor((4,), dtype=T.float32, access="output")):
    T.store(C, T.load(A))

  work = tmp_path / "work"
  work.mkdir()
  compiler = vecops.Compiler(cache_dir=work / "cache", build_dir=work / "build")
  source = work / "kernel.cpp"
  source.write_text("source")
  definition, _, _ = _build_definition(fixture.ir, (), runtime_parallelism=True)
  state = {"identity": "real_test_attestation", "calls": 0}

  def attest(_):
    state["calls"] += 1
    return state["identity"]

  native_calls = []
  monkeypatch.setattr(identity, "compiler_identity", attest)
  monkeypatch.setattr(vecops._C, "Operator", lambda *args: native_calls.append(args) or "native")
  batch = repair.BatchInputAttestation(work)
  first = batch.operator(source, definition, compiler, recipe_id="exact recipe")
  second = batch.operator(source, definition, compiler, recipe_id="exact recipe")
  assert state["calls"] == 1
  assert first.native == second.native == "native"
  assert native_calls[0][5] == compiler.cache_namespace + ";inputs=real_test_attestation"
  assert native_calls[0][6] == "exact recipe"
  assert native_calls[0][7] == identity.digest(batch.sources[0]["inputs"])
  assert native_calls[0][8]()
  batch.guard("pre_build")
  assert state["calls"] == 2
  state["identity"] = "changed SDK"
  with pytest.raises(repair.BatchInputAttestationError, match="compiler/SDK"):
    batch.guard("post_build")
  state["identity"] = "real_test_attestation"
  with pytest.raises(repair.BatchInputAttestationError):
    batch.guard("must not wash away observed mutation")
  restored = repair.BatchInputAttestation.restore(work)
  with pytest.raises(repair.BatchInputAttestationError):
    restored.guard("recovery")


def test_batch_guard_transient_attestation_error_cannot_publish_later(tmp_path, monkeypatch):
  import vecops
  import vecops._identity as identity

  work = tmp_path / "work"
  work.mkdir()
  compiler = vecops.Compiler(cache_dir=work / "cache", build_dir=work / "build").snapshot()
  batch = repair.BatchInputAttestation(work)
  key = batch._policy_key(compiler)
  batch.policies[key] = {"compiler": compiler, "identity": "unchanged"}
  monkeypatch.setattr(identity, "compiler_identity", lambda _: (_ for _ in ()).throw(OSError("temporary read failure")))
  with pytest.raises(repair.BatchInputAttestationError, match="temporary read failure"):
    batch.guard("pre_build")
  monkeypatch.setattr(identity, "compiler_identity", lambda _: "unchanged")
  with pytest.raises(repair.BatchInputAttestationError):
    batch.guard("post_build")


def test_batch_policy_separates_effective_flags_vector_lengths_and_cwd(tmp_path):
  from copy import copy

  import vecops
  from vecops._arm_vector_state import VectorState

  compiler = vecops.Compiler(cache_dir=tmp_path / "cache", build_dir=tmp_path / "build").snapshot()
  first = repair.BatchInputAttestation._policy_key(compiler)
  changed = copy(compiler)
  changed.cflags = (*compiler.cflags, "-ffp-contract=off")
  assert repair.BatchInputAttestation._policy_key(changed) != first
  changed = copy(compiler)
  changed._vector_state = VectorState(32, 64)
  assert repair.BatchInputAttestation._policy_key(changed) != first
  changed = copy(compiler)
  changed.definition_cwd = tmp_path / "other"
  assert repair.BatchInputAttestation._policy_key(changed) != first


def test_posthash_mismatch_archives_untrusted_so_and_never_promotes(tmp_path, monkeypatch):
  from types import SimpleNamespace

  import dsl_matmul_artifact_ab as ab
  import vecops

  group = {"index": 0, "id": "g", "execution_ids": ["row"]}
  directory = tmp_path / "groups/g"
  directory.mkdir(parents=True)
  prior = tmp_path / "prior_valid.so"
  prior.write_bytes(b"prior independently attested module")
  old_index = {"artifacts": {"row": {"cpp": str(prior)}}}
  completed = {
    "cpp": {
      "binary": str(prior),
      "binary_sha256": repair.digest(prior.read_bytes()),
      "corrected_plan_runner_hashes": {},
    }
  }
  repair.suite.save(directory / "index.json", old_index)
  repair.suite.save(directory / "completed-roles.json", completed)
  source_path = tmp_path / "prelowered.cpp"
  source_path.write_bytes(b"body")
  repair.suite.save(
    directory / "lower-proof.json", {"dsl": {"source": str(source_path), "raw_sha256": repair.digest(b"body")}}
  )
  plan = {
    "groups": [group],
    "executions": [{"execution_id": "row"}],
    "target": "Generic",
    "toolchain": {"cc": None, "cxx": None},
    "compiler_cflags": [],
    "runner_hashes": {},
    "pass_options": {},
  }
  scenario = SimpleNamespace(execution_id="row")
  monkeypatch.setattr(ab, "source_scenarios", lambda _: [scenario])
  source = SimpleNamespace(source_templates=lambda _: ((object(), object()), None), source_metadata=lambda _: ())
  kernel = SimpleNamespace(
    source=source_path, operator=lambda *_args, **_kwargs: None, _core=SimpleNamespace(specialization_values={})
  )
  monkeypatch.setattr(repair, "compile_prelowered", lambda *_args, **_kwargs: kernel)

  class InvalidPostGuard:
    def __init__(self, work):
      self.work = work

    def guard(self, phase):
      if phase == "post_build":
        raise repair.BatchInputAttestationError("SDK changed after compilation")

  monkeypatch.setattr(repair, "BatchInputAttestation", InvalidPostGuard)
  work = tmp_path / "chunks/0/work"

  def native_build(*_args, **_kwargs):
    (work / "new.so").write_bytes(b"new untrusted module")

  monkeypatch.setattr(vecops, "compile_batch", native_build)
  monkeypatch.setattr(repair, "index_partial", lambda *_args: pytest.fail("must not index new modules"))
  monkeypatch.setattr(repair.suite, "retain", lambda *_args: pytest.fail("must not promote standalone new binaries"))
  args = SimpleNamespace(chunk="0", suite_dir=tmp_path, jobs=1, batch_input_attestation=True)
  repair.compile_worker(args, source, plan)
  assert repair.suite.read(directory / "index.json") == old_index
  assert repair.suite.read(directory / "completed-roles.json") == completed
  assert repair.suite.read(directory / "build.json")["status"] == "batch_attestation_failed"
  assert not work.exists()
  assert not list(tmp_path.rglob("retained/*.so"))
  archive = tmp_path / "chunks/0/invalid-attestation-evidence.tar.gz"
  with tarfile.open(archive) as stream:
    assert stream.extractfile("untrusted-work/new.so").read() == b"new untrusted module"


def test_verified_batch_evidence_survives_cleanup_and_is_linked_from_index(tmp_path, monkeypatch):
  from types import SimpleNamespace

  group = {"id": "g", "execution_ids": ["row"]}
  directory = tmp_path / "groups/g"
  directory.mkdir(parents=True)
  work = tmp_path / "chunks/0/work"
  work.mkdir(parents=True)
  (work / ".full-suite-owned-work").write_text("repair")
  (work / ".batch-input-attestation-required").write_text("1")
  document = {
    "invalid_reason": None,
    "policies": [{"identity": "actual full identity"}],
    "sources": [{"inputs": ["full source closure"]}],
    "events": [{"phase": "pre_build", "verified": True}, {"phase": "post_build", "verified": True}],
  }
  repair.suite.save(work / "batch-input-attestation.json", document)
  binary = work / "module.so"
  binary.write_bytes(b"actual new module")

  def index_partial(*_):
    repair.suite.save(
      directory / "index.json", {"artifacts": {"row": {"cpp": str(binary)}}, "schema_audit": {"row": {}}}
    )

  monkeypatch.setattr(repair, "index_partial", index_partial)
  repair.finalize_chunk(SimpleNamespace(suite_dir=tmp_path), None, {"runner_hashes": {}}, [group], work, [{}], None)
  assert not work.exists()
  proof = repair.suite.read(directory / "index.json")["batch_input_attestation"]
  assert repair.digest(Path(proof["path"]).read_bytes()) == proof["sha256"]
  assert repair.suite.read(Path(proof["path"])) == document


@pytest.mark.parametrize(
  "events,invalid",
  [
    ([], None),
    ([{"phase": "pre_build", "verified": True}], None),
    ([{"phase": "post_build", "verified": True}], None),
    ([{"phase": "pre_build", "verified": True}, {"phase": "post_build", "verified": True}], "changed SDK"),
  ],
)
def test_incomplete_or_rejected_checkpoint_cannot_enter_finalizer(tmp_path, monkeypatch, events, invalid):
  work = tmp_path / "work"
  work.mkdir()
  (work / ".batch-input-attestation-required").write_text("1")
  repair.suite.save(work / "batch-input-attestation.json", {"events": events, "invalid_reason": invalid})
  monkeypatch.setattr(repair, "index_partial", lambda *_args: pytest.fail("must not enter index promotion"))
  with pytest.raises(repair.BatchInputAttestationError, match="complete successful"):
    repair.finalize_chunk(None, None, None, [], work, [], None)


def test_keepgoing_attempts_other_policy_and_preserves_original_graph_flags(tmp_path, monkeypatch):
  from types import SimpleNamespace

  import vecops
  import vecops._compiler as compiler_module

  work = tmp_path / "work"
  work.mkdir()
  order = []
  first_error = RuntimeError("original LLVM fatal")
  compiler = SimpleNamespace(
    cmake="real configured cmake", definition_cwd=tmp_path, environment={"CXXFLAGS": "unchanged source flags"}
  )
  policies = {
    "dsl": {"key": "dsl", "compiler": compiler, "requests": ["bad", "good"]},
    "cpp": {"key": "cpp", "compiler": compiler, "requests": ["other-policy"]},
  }

  def build(requests, **kwargs):
    order.append(tuple(requests))
    assert kwargs == {"parallelism": 2, "execution_parallelism": 1}
    if requests == ["bad", "good"]:
      graph = work / "build/kernel-batch-failed/build"
      graph.mkdir(parents=True)
      (graph / "CMakeCache.txt").write_text("CMAKE_GENERATOR:INTERNAL=Ninja\nCMAKE_BUILD_TYPE:STRING=Release\n")
      (graph / "build.ninja").write_text("original configured flags")
      raise first_error

  def tool(selected, variable, defaults):
    assert selected == compiler.cmake and variable == "CMAKE" and defaults == ("cmake",)
    return "/actual/cmake"

  phases = []
  attestation = SimpleNamespace(invalid_reason=None, guard=phases.append)

  def run(command, **kwargs):
    order.append("continue")
    assert phases == ["pre_continue_build"]
    assert command[0] == "/actual/cmake"
    assert command[-3:] == ["--", "-k", "0"]
    assert command[command.index("--target") + 1] == "vecops_batch_all"
    assert command[command.index("--parallel") + 1] == "2"
    assert kwargs["env"]["CXXFLAGS"] == "unchanged source flags"
    assert kwargs["cwd"] == tmp_path and kwargs["check"] is False
    (work / "other-good.so").write_bytes(b"survived bad module")
    return SimpleNamespace(returncode=1)

  monkeypatch.setattr(vecops, "compile_batch", build)
  monkeypatch.setattr(compiler_module, "_tool", tool)
  monkeypatch.setattr(repair.subprocess, "run", run)
  error = repair.compile_policies_keep_going(SimpleNamespace(jobs=2), policies, attestation, work)
  assert "original LLVM fatal" in error
  assert order == [("bad", "good"), "continue", ("other-policy",)]
  assert (work / "build/kernel-batch-failed/build/build.ninja").read_text() == "original configured flags"
  assert "original LLVM fatal" in repair.suite.read(next(work.parent.glob("policy-0-first-error-*.json")))["error"]
  assert len(repair.suite.read(work.parent / "keepgoing-policy-results.json")) == 3


def test_keepgoing_never_runs_unconfigured_or_non_ninja_graph(tmp_path, monkeypatch):
  from types import SimpleNamespace

  import vecops

  work = tmp_path / "work"
  work.mkdir()
  compiler = SimpleNamespace()
  policies = {"cpp": {"key": "cpp", "compiler": compiler, "requests": ["unsupported-config"]}}

  def build(*_args, **_kwargs):
    graph = work / "build/kernel-batch-failed/build"
    graph.mkdir(parents=True)
    (graph / "CMakeCache.txt").write_text("CMAKE_GENERATOR:INTERNAL=Unix Makefiles\n")
    raise RuntimeError("not Ninja")

  monkeypatch.setattr(vecops, "compile_batch", build)
  monkeypatch.setattr(
    repair.subprocess, "run", lambda *_args, **_kwargs: pytest.fail("must not change another generator")
  )
  attestation = SimpleNamespace(invalid_reason=None, guard=lambda *_args: pytest.fail("no continuation"))
  assert "not Ninja" in repair.compile_policies_keep_going(SimpleNamespace(jobs=1), policies, attestation, work)
  with pytest.raises(ValueError, match="requires batch"):
    repair.compile_policies_keep_going(SimpleNamespace(jobs=1), policies, None, work)


def test_keepgoing_marker_only_enables_guarded_future_compile_workers(tmp_path):
  from types import SimpleNamespace

  marker = tmp_path / ".repair-keepgoing-opt-in.json"
  policy = {
    "version": 1,
    "driver_sha256": repair.digest(Path(repair.__file__).read_bytes()),
    "scope": "compile_workers_with_batch_input_attestation",
    "keep_going": True,
  }
  repair.suite.save(marker, policy)
  marker.chmod(0o600)
  for worker, guard in ((None, True), ("lower", True), ("recover", True), ("compile", False)):
    args = SimpleNamespace(worker=worker, keep_going=False, batch_input_attestation=guard, suite_dir=tmp_path)
    repair.apply_keepgoing_marker(args)
    assert not args.keep_going
  args = SimpleNamespace(worker="compile", keep_going=False, batch_input_attestation=True, suite_dir=tmp_path)
  repair.apply_keepgoing_marker(args)
  assert args.keep_going and args.keep_going_marker["sha256"] == repair.digest(marker.read_bytes())
  marker.chmod(0o644)
  args.keep_going = False
  with pytest.raises(ValueError, match="private"):
    repair.apply_keepgoing_marker(args)
  marker.chmod(0o600)
  policy["driver_sha256"] = "different-driver"
  repair.suite.save(marker, policy)
  with pytest.raises(ValueError, match="does not match"):
    repair.apply_keepgoing_marker(args)


def test_keepgoing_marker_rejects_nonregular_file_without_reading(tmp_path):
  from types import SimpleNamespace

  marker = tmp_path / ".repair-keepgoing-opt-in.json"
  import os

  os.mkfifo(marker, 0o600)
  args = SimpleNamespace(worker="compile", keep_going=False, batch_input_attestation=True, suite_dir=tmp_path)
  with pytest.raises(ValueError, match="regular"):
    repair.apply_keepgoing_marker(args)


def test_interrupted_recovery_marker_exact_driver_and_chunk(tmp_path):
  from types import SimpleNamespace

  marker = tmp_path / ".repair-interrupted-recovery-opt-in.json"
  policy = {"version": 1, "driver_sha256": repair.digest(Path(repair.__file__).read_bytes()), "chunks": ["1,2"]}
  repair.suite.save(marker, policy)
  marker.chmod(0o600)
  args = SimpleNamespace(suite_dir=tmp_path, worker="recover", chunk="1,2")
  assert repair.interrupted_recovery_policy(args) == policy
  args.chunk = "3"
  assert repair.interrupted_recovery_policy(args) is None
  args.chunk = "1,2"
  marker.chmod(0o644)
  with pytest.raises(ValueError, match="private regular"):
    repair.interrupted_recovery_policy(args)


def test_interrupted_recovery_refuses_live_graph_before_guard(tmp_path, monkeypatch):
  from types import SimpleNamespace
  from unittest.mock import Mock

  work = tmp_path / "work"
  work.mkdir()
  attestation = SimpleNamespace(guard=Mock())
  monkeypatch.setattr(repair, "work_readers", lambda _: [{"pid": 123}])
  with pytest.raises(ValueError, match="live readers"):
    repair.continue_interrupted_graphs(SimpleNamespace(), work, attestation)
  attestation.guard.assert_not_called()
  assert work.exists()


def test_configured_graph_flags_must_match_restored_policy(tmp_path):
  from types import SimpleNamespace

  batch = tmp_path / "batch"
  build = batch / "build"
  build.mkdir(parents=True)
  adapter = batch / "tasks/0/input/vecops_adapter.cpp"
  adapter.parent.mkdir(parents=True)
  adapter.write_text("")
  project = batch / "project/tasks/0/CMakeLists.txt"
  project.parent.mkdir(parents=True)
  project.write_text(
    'TARGET_ARCH "Native"\n    COMPILE_OPTIONS\n        "-fstrict"\n    COMPILE_DEFINITIONS\n        "VL=64"\n'
  )
  (build / "CMakeCache.txt").write_text(
    "CMAKE_C_COMPILER:FILEPATH=/cc\nCMAKE_CXX_COMPILER:FILEPATH=/cxx\nCMAKE_CXX_FLAGS:STRING=\n"
  )
  config = SimpleNamespace(
    toolchain=SimpleNamespace(c_compiler="/cc", cxx_compiler="/cxx"),
    target_arch="Native",
    compile_options=["-fstrict"],
    compile_definitions=["VL=64"],
    link_options=[],
    link_libraries=[],
  )
  compiler = SimpleNamespace(_native_config=lambda: config, environment={"CXXFLAGS": ""})
  assert repair.validate_configured_policy(compiler, build, [adapter])
  config.compile_options = ["-frelaxed"]
  with pytest.raises(ValueError, match="COMPILE_OPTIONS"):
    repair.validate_configured_policy(compiler, build, [adapter])


@pytest.mark.parametrize("path", sorted(repair.CODEGEN_CHANGE_PATHS))
def test_codegen_change_requires_explicit_existing_file(path):
  old = {path: "old", "python/vecops/_C.so": "extension"}
  new = {**old, path: "new"}
  assert repair.validate_runtime_change(old, new, [path]) == {path: {"old_sha256": "old", "new_sha256": "new"}}
  with pytest.raises(ValueError, match="undeclared"):
    repair.validate_runtime_change(old, new, [])
  for invalid in ({path: "new"}, {**new, "extra.py": "extra"}):
    with pytest.raises(ValueError, match="inventory"):
      repair.validate_runtime_change(old, invalid, [path])
  with pytest.raises(ValueError, match="undeclared"):
    repair.validate_runtime_change(old, {**new, "python/vecops/_C.so": "different"}, [path])


@pytest.mark.parametrize(
  "path",
  [
    "python/vecops/language/_compiler/driver.py",
    "python/vecops/language/_compiler/frontend/primitives.py",
    "include/vecops/Vec.h",
    "python/vecops/language/library.py",
    "python/vecops/language/_library/__init__.py",
    "/python/vecops/language/_compiler/analysis/affine.py",
    "python/vecops/language/_compiler/analysis/../analysis/affine.py",
  ],
)
def test_codegen_change_whitelist_rejects_other_paths_even_if_explicit(path):
  with pytest.raises(ValueError, match="unsupported"):
    repair.validate_runtime_change({path: "old"}, {path: "new"}, [path])


def test_runtime_identity_includes_installed_sdk_all_files(tmp_path):
  path = tmp_path / "python/vecops/_sdk/lib/static.a"
  path.parent.mkdir(parents=True)
  path.write_bytes(b"sdk")
  assert "python/vecops/_sdk/lib/static.a" in repair.runtime_identity(tmp_path)


def test_lower_proof_identity_binds_current_runtime():
  plan = {key: {} for key in ("runner_hashes", "pass_options", "target", "toolchain", "compiler_cflags")}
  assert repair.proof_identity(plan, {"codegen.py": "old"}) != repair.proof_identity(plan, {"codegen.py": "new"})


def test_codegen_role_evidence_requires_exact_spec_policy_and_task_command():
  spec = b"namespace vecops::spec {\ninline constexpr auto Parallelism = ::vecops::meta::cint<1>;\nusing PSA = ::vecops::meta::Dynamic<1, 1, ::vecops::meta::kHiInf>;\n} // namespace vecops::spec\n"
  components = {"config_key": "key", "tools": [{"path": "/gcc", "content": "hash"}]}
  fresh = {
    "captures": {"Parallelism": 1},
    "old_build_components": components,
    "effective_cflags": ["-ffp-contract=off"],
  }
  candidate = {
    "spec": spec,
    "build_evidence": {
      "identities": [{"components": components}],
      "commands": [{"file": "adapter.cpp", "command": "/gcc -ffp-contract=off -o vecops_adapter.cpp.o -c adapter.cpp"}],
    },
  }
  assert repair.codegen_role_evidence(fresh, candidate)
  assert repair.codegen_role_evidence(fresh, {**candidate, "spec": None}) is None
  assert repair.codegen_role_evidence({**fresh, "captures": {"Parallelism": 2}}, candidate) is None
  assert (
    repair.codegen_role_evidence({**fresh, "old_build_components": {**components, "config_key": "changed"}}, candidate)
    is None
  )
  assert repair.codegen_role_evidence(fresh, {**candidate, "build_evidence": {}}) is None
  assert repair.codegen_role_evidence(fresh, {**candidate, "spec": spec + b"unknown declaration;"}) is None


def test_archive_build_evidence_requires_exact_adapter_path():
  data = json.dumps([{"file": "/old/work/task/adapter.cpp", "command": "gcc"}]).encode()
  members = {
    "work/build/compile_commands.json": data,
    "work/cache/identities/key.json": b'[{"components": {"config_key": "key"}}]',
  }
  assert len(repair.archived_build_evidence(members, "/old/work/task/adapter.cpp")["commands"]) == 1
  assert not repair.archived_build_evidence(members, "/new/work/task/adapter.cpp")["commands"]


def test_codegen_task_flags_reject_unattested_extras_and_wrong_includes():
  fresh = {
    "old_build_components": {"tools": [{"path": "/gcc"}]},
    "effective_cflags": ["-ffp-contract=off"],
    "compile_include_roots": ["/sdk/include"],
  }
  command = {
    "file": "/task/vecops_adapter.cpp",
    "command": "/gcc -O3 -I/sdk/include -I/task -ffp-contract=off -o vecops_adapter.cpp.o -c /task/vecops_adapter.cpp",
  }
  assert repair.task_command_matches(fresh, {}, command)
  for flag in ("-ffast-math", "-ffp-contract=fast", "-I/other-sdk", "-O0"):
    assert not repair.task_command_matches(fresh, {}, {**command, "command": command["command"] + " " + flag})


def test_codegen_paths_propagate_to_every_child(tmp_path, monkeypatch):
  from types import SimpleNamespace

  commands = []

  class Process:
    pid = 1

    def __init__(self, command, **kwargs):
      commands.append(command)

    def wait(self, **kwargs):
      return 0

  monkeypatch.setattr(repair.subprocess, "Popen", Process)
  monkeypatch.setattr(repair.suite, "process_identity", lambda _: {})
  paths = sorted(repair.CODEGEN_CHANGE_PATHS)
  args = SimpleNamespace(
    snapshot=tmp_path,
    suite_dir=tmp_path,
    old_suite=tmp_path,
    jobs=1,
    min_free_gb=0,
    compress_prelower=False,
    batch_input_attestation=True,
    keep_going=True,
    allow_codegen_change=paths,
  )
  for worker in ("lower", "compile", "recover"):
    assert repair.child(args, ["--worker", worker], tmp_path / "child.log", 1) == 0
  assert all(
    command.count("--allow-codegen-change") == len(paths) and all(path in command for path in paths) for command in commands
  )
