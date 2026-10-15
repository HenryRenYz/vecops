# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Recovery, exact coverage, and evidence retention without native builds."""

import importlib.util
import signal
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest

MODULE = Path(__file__).resolve().parents[2] / "benchmarks/dsl_matmul_full_suite.py"
spec = importlib.util.spec_from_file_location("full_suite_test", MODULE)
suite = importlib.util.module_from_spec(spec)
spec.loader.exec_module(suite)


def args(tmp_path, **changes):
  return SimpleNamespace(
    **(
      {
        "suite_dir": tmp_path,
        "build_workers": 2,
        "build_timeout": 10,
        "timing_timeout": 10,
        "retry_failed": False,
        "stages": ["compute"],
        "min_free_gb": 0,
        "runner_arguments": [],
        "keep_work": False,
      }
      | changes
    )
  )


def test_grouping_preserves_execution_ids_with_one_compilation_key():
  scenarios = [
    SimpleNamespace(compilation_key=("shared",), execution_id="a"),
    SimpleNamespace(compilation_key=("shared",), execution_id="b"),
    SimpleNamespace(compilation_key=("other",), execution_id="c"),
  ]
  groups = suite.grouped(scenarios)
  assert [g["execution_ids"] for g in groups] == [["a", "b"], ["c"]]
  assert len({g["id"] for g in groups}) == 2


def test_subset_manifest_does_not_duplicate_full_plan():
  plan = {
    "executions": [{"execution_id": "a"}, {"execution_id": "b"}],
    "groups": ["large"],
    "declaration_indexes": ["large"],
    "source_hashes": {"source": "hash"},
    "pass_options": {},
  }
  subset = suite.subset_manifest(plan, ["b"])
  assert subset["executions"] == [{"execution_id": "b"}]
  assert subset["selected_execution_ids"] == ["b"]
  assert "groups" not in subset and "declaration_indexes" not in subset


def test_failed_index_retains_orphan_binary_source_and_mapping(tmp_path):
  work = tmp_path / "work"
  (work / "dsl-sources/x").mkdir(parents=True)
  (work / ".full-suite-owned-work").write_text("owned")
  (work / "dsl-sources/x/kernel.cpp").write_text("source")
  binary = work / "unindexed.so"
  binary.write_bytes(b"ELF evidence")
  index = suite.retain(tmp_path, {}, work)
  destination = Path(index["retained_binary_mapping"][str(binary)])
  assert destination.read_bytes() == b"ELF evidence"
  with tarfile.open(index["evidence_archive"]) as archive:
    assert "work/dsl-sources/x/kernel.cpp" in archive.getnames()
  suite.clean_work(work)
  assert not work.exists() and destination.exists()


def test_cleanup_refuses_unmarked_directory(tmp_path):
  work = tmp_path / "work"
  work.mkdir()
  with pytest.raises(ValueError, match="unmarked"):
    suite.clean_work(work)
  assert work.exists()


@pytest.mark.parametrize(
  "code,text,expected",
  [(-signal.SIGKILL, "", "signal_killed"), (1, "MemoryError", "oom"), (1, "unsupported type", "unsupported")],
)
def test_failure_classification_does_not_infer_oom_from_sigkill(code, text, expected):
  assert suite.failure(code, text) == expected
  assert suite.failure(code, text, timed_out=True) == "timeout"


def test_build_failure_isolated_from_next_group(monkeypatch, tmp_path):
  groups = [{"id": "bad", "index": 0, "execution_ids": ["a"]}, {"id": "good", "index": 1, "execution_ids": ["b"]}]

  class Process:
    def __init__(self, command, **kwargs):
      self.group = command[command.index("--worker-group") + 1]
      self.pid = 101 if self.group == "bad" else 102

    def poll(self):
      if self.group == "good":
        suite.save(tmp_path / "groups/good/build.json", {"status": "built"})
        return 0
      return 1

  monkeypatch.setattr(suite.subprocess, "Popen", Process)
  suite.dispatch(args(tmp_path), {}, groups, "build")
  assert suite.read(tmp_path / "groups/bad/build.json")["status"] == "failed"
  assert suite.read(tmp_path / "groups/good/build.json")["status"] == "built"


def test_signal_marks_one_row_and_recovers_remaining_without_retry_loop(monkeypatch, tmp_path):
  group = {"id": "shared", "index": 0, "execution_ids": ["a", "b"]}
  directory = tmp_path / "groups/shared"
  suite.save(directory / "build.json", {"status": "built"})
  calls = []

  class Process:
    def __init__(self, command, **kwargs):
      calls.append(command)
      self.pid = 100 + len(calls)
      self.recovered = "--resume-after-row-failure" in command

    def poll(self):
      records = suite.read(directory / "timing.json", {})
      if self.recovered:
        records["b/compute"] = {"status": "passed"}
        suite.save(directory / "timing.json", records)
        suite.save(directory / "time.json", {"status": "timed", "failures": 1})
        return 0
      suite.save(directory / "timing.json", {"a/compute": {"status": "running"}})
      return -signal.SIGILL

  monkeypatch.setattr(suite.subprocess, "Popen", Process)
  suite.dispatch(args(tmp_path, retry_failed=True), {}, [group], "time")
  assert len(calls) == 2
  records = suite.read(directory / "timing.json")
  assert records["a/compute"]["status"] == "signal_killed"
  assert records["b/compute"]["status"] == "passed"


def test_worker_setup_failure_is_terminal_for_each_id(monkeypatch, tmp_path):
  group = {"id": "shared", "index": 0, "execution_ids": ["a", "b"]}
  directory = tmp_path / "groups/shared"
  suite.save(directory / "build.json", {"status": "built"})
  calls = []

  class Process:
    def __init__(self, command, **kwargs):
      calls.append(command)
      self.pid = 100

    def poll(self):
      return 1

  monkeypatch.setattr(suite.subprocess, "Popen", Process)
  suite.dispatch(args(tmp_path), {}, [group], "time")
  assert len(calls) == 1
  assert all(r["status"] == "group_setup_failed" for r in suite.read(directory / "timing.json").values())


def test_source_hash_guard_rejects_same_path_content_drift(tmp_path):
  import hashlib

  (tmp_path / "benchmarks").mkdir()
  source = tmp_path / "case.cpp"
  harness = tmp_path / "benchmarks/templates.py"
  source.write_text("frozen source")
  harness.write_text("frozen templates")
  plan = {
    "source_hashes": {"case.cpp": hashlib.sha256(source.read_bytes()).hexdigest()},
    "runner_hashes": {"templates.py": hashlib.sha256(harness.read_bytes()).hexdigest()},
  }
  suite.verify_plan_sources(tmp_path, plan)
  harness.write_text("changed templates")
  with pytest.raises(ValueError, match="immutable plan source changed"):
    suite.verify_plan_sources(tmp_path, plan)


def test_failed_build_has_terminal_unrun_stage_not_pending():
  stage = suite.unrun_stage({"status": "timeout", "log": "build.log"}, {"status": "pending"})
  assert stage == {"status": "not_run", "reason": "build_timeout", "build_log": "build.log"}
  assert suite.unrun_stage({"status": "running"}, {"status": "pending"}) == {"status": "pending"}


def test_per_row_deadline_survives_long_valid_group(monkeypatch, tmp_path):
  monkeypatch.setattr(suite.time, "monotonic_ns", lambda: 1000_000_000_000)
  suite.save(tmp_path / "timing.json", {"row/compute": {"worker_pid": 42, "started_monotonic_ns": 990_000_000_000}})
  assert suite.timing_deadline_base(tmp_path, 42, 100) == 990
  assert suite.timing_deadline_base(tmp_path, 99, 100) == 100
  suite.save(tmp_path / "timing.json", {"row/compute": {"worker_pid": 42, "finished_monotonic_ns": 995_000_000_000}})
  assert suite.timing_deadline_base(tmp_path, 42, 100) == 995


def test_deadline_ignores_historical_future_clock_after_reboot(monkeypatch, tmp_path):
  monkeypatch.setattr(suite.time, "monotonic_ns", lambda: 1000_000_000_000)
  suite.save(tmp_path / "timing.json", {"row/compute": {"worker_pid": 42, "started_monotonic_ns": 2000_000_000_000}})
  assert suite.timing_deadline_base(tmp_path, 42, 100) == 100


def test_failed_group_runs_available_packing_only_and_keeps_build_failure(tmp_path, monkeypatch):
  monkeypatch.syspath_prepend(str(MODULE.parent))
  import dsl_matmul_artifact_ab as artifact

  directory = tmp_path / "groups/partial"
  group = {"id": "partial", "index": 0, "execution_ids": ["id"]}
  plan = {"executions": [{"execution_id": "id", "packing": "onlinepackedb"}]}
  suite.save(directory / "build.json", {"status": "failed", "error": "LLVM compile failure"})
  paths = {"dsl": None, "cpp": "compute-cpp", "pack_dsl": "pack-dsl", "pack_cpp": "pack-cpp"}
  suite.save(directory / "index.json", {"artifacts": {"id": paths}})
  calls = []

  def compare(options):
    calls.append(options.packing_only)
    assert options.packing_only
    suite.save(options.output_dir / "results.json", {"rows": [{"execution_id": "id", "stage": "packing"}]})

  monkeypatch.setattr(artifact, "compare", compare)
  worker = args(tmp_path, stages=["compute", "packing"], cpu=53, rounds=12, resume_after_row_failure=False)
  suite.timing_worker(worker, None, plan, group, directory)
  records = suite.read(directory / "timing.json")
  assert calls == [True]
  assert records["id/compute"] == {
    "execution_id": "id",
    "stage": "compute",
    "status": "not_run",
    "reason": "missing_artifacts",
    "missing_roles": ["candidate.dsl"],
  }
  assert records["id/packing"]["status"] == "passed"
  assert suite.read(directory / "build.json")["status"] == "failed"


def test_failed_group_missing_packers_records_not_run_without_starting_worker(tmp_path, monkeypatch):
  monkeypatch.syspath_prepend(str(MODULE.parent))
  group = {"id": "missing", "index": 0, "execution_ids": ["id"]}
  plan = {"executions": [{"execution_id": "id", "packing": "packedab"}]}
  directory = tmp_path / "groups/missing"
  suite.save(directory / "build.json", {"status": "failed"})
  suite.save(directory / "index.json", {"artifacts": {"id": {"dsl": None, "cpp": None}}})
  monkeypatch.setattr(suite.subprocess, "Popen", lambda *a, **k: pytest.fail("no available stage"))
  suite.dispatch(args(tmp_path, stages=["packing", "compute"]), plan, [group], "time")
  records = suite.read(directory / "timing.json")
  assert records["id/packing"]["status"] == "not_run"
  assert records["id/compute"]["status"] == "not_run"
  assert suite.read(directory / "build.json")["status"] == "failed"


def test_dispatch_failed_group_with_available_packing_stage(tmp_path, monkeypatch):
  monkeypatch.syspath_prepend(str(MODULE.parent))
  group = {"id": "partial", "index": 0, "execution_ids": ["id"]}
  plan = {"executions": [{"execution_id": "id", "packing": "packeda"}]}
  directory = tmp_path / "groups/partial"
  suite.save(directory / "build.json", {"status": "failed", "error": "compile crash"})
  suite.save(
    directory / "index.json", {"artifacts": {"id": {"dsl": None, "cpp": None, "pack_dsl": "dp", "pack_cpp": "cp"}}}
  )
  calls = []

  class Process:
    def __init__(self, command, **kwargs):
      self.pid = 1234
      calls.append(command)

    def poll(self):
      records = suite.read(directory / "timing.json")
      records["id/packing"] = {"execution_id": "id", "stage": "packing", "status": "passed"}
      suite.save(directory / "timing.json", records)
      suite.save(directory / "time.json", {"status": "timed"})
      return 0

  monkeypatch.setattr(suite.subprocess, "Popen", Process)
  suite.dispatch(args(tmp_path, stages=["compute", "packing"]), plan, [group], "time")
  assert len(calls) == 1
  assert suite.read(directory / "timing.json")["id/packing"]["status"] == "passed"
  assert suite.read(directory / "timing.json")["id/compute"]["status"] == "not_run"
  assert suite.read(directory / "build.json")["error"] == "compile crash"


def test_newly_available_stage_retries_not_run_without_retrying_runtime_failures(tmp_path, monkeypatch):
  monkeypatch.syspath_prepend(str(MODULE.parent))
  import dsl_matmul_artifact_ab as artifact

  group = {"id": "recovered", "index": 0, "execution_ids": ["id"]}
  plan = {"executions": [{"execution_id": "id", "packing": "raw"}]}
  directory = tmp_path / "groups/recovered"
  suite.save(directory / "index.json", {"artifacts": {"id": {"dsl": "dsl", "cpp": "cpp"}}})
  suite.save(directory / "timing.json", {"id/compute": {"status": "not_run", "reason": "missing_artifacts"}})
  calls = []

  def compare(options):
    calls.append(options)
    suite.save(options.output_dir / "results.json", {"rows": [{"execution_id": "id", "stage": "compute"}]})

  monkeypatch.setattr(artifact, "compare", compare)
  worker = args(tmp_path, cpu=53, rounds=12, resume_after_row_failure=False)
  suite.timing_worker(worker, None, plan, group, directory)
  assert len(calls) == 1 and suite.read(directory / "timing.json")["id/compute"]["status"] == "passed"
  suite.save(directory / "timing.json", {"id/compute": {"status": "validation_failed"}})
  suite.timing_worker(worker, None, plan, group, directory)
  assert len(calls) == 1


def test_stop_terminates_separate_descendant_group_but_not_unrelated_process(tmp_path):
  import json
  import subprocess
  import sys
  import time

  info = tmp_path / "descendant.json"
  code = "import subprocess,sys,json,time; p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'],process_group=0); open(sys.argv[1],'w').write(json.dumps({'pid':p.pid})); time.sleep(60)"
  unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True)
  root = subprocess.Popen([sys.executable, "-c", code, str(info)], start_new_session=True)
  root._full_suite_identity = suite.process_identity(root.pid)
  leaf = None
  try:
    deadline = time.monotonic() + 3
    while not info.exists() and time.monotonic() < deadline:
      time.sleep(0.01)
    assert info.exists()
    leaf = json.loads(info.read_text())["pid"]
    assert suite.process_identity(leaf)["pgid"] == leaf != root.pid
    suite.stop(root, grace=0.2, audit_path=tmp_path / "cleanup.json")
    current = suite.process_identity(leaf)
    assert current is None or current["state"] == "Z"
    assert unrelated.poll() is None
    audit = suite.read(tmp_path / "cleanup.json")
    assert leaf in {r["pid"] for r in audit["captured"]}
    assert not audit["remaining"]
    assert {event["method"] for event in audit["signals"]} == {"pidfd"}
  finally:
    if root.poll() is None:
      root.kill()
      root.wait()
    if leaf is not None and (state := suite.process_identity(leaf)) and state["state"] != "Z":
      import os

      os.kill(leaf, signal.SIGKILL)
    unrelated.kill()
    unrelated.wait()


def test_reused_process_identity_is_not_signalled(monkeypatch):
  record = {"pid": 1234567, "uid": 1000, "start_ticks": 10}
  assert not suite.same_process(record, {**record, "start_ticks": 11})
  assert not suite.same_process(record, {**record, "uid": 1001})
  monkeypatch.setattr(suite, "process_identity", lambda pid: {**record, "start_ticks": 11})
  assert suite.owned_process_tree(record) == {}


def test_pidfd_rechecks_identity_before_signal_and_closes_descriptor(monkeypatch):
  record = {"pid": 1234567, "uid": 1000, "start_ticks": 10}
  opened, closed, signals = [], [], []
  monkeypatch.setattr(suite, "open_process_descriptor", lambda pid: opened.append(pid) or 42)
  monkeypatch.setattr(suite.os, "close", closed.append)
  monkeypatch.setattr(suite, "send_process_descriptor", lambda *args: signals.append(args))
  monkeypatch.setattr(suite, "process_identity", lambda pid: {**record, "start_ticks": 11})
  assert suite.signal_owned_processes({record["pid"]: record}, signal.SIGTERM) == []
  assert opened == [record["pid"]]
  assert closed == [42]
  assert signals == []


def test_retain_already_retained_same_binary_preserves_hash(tmp_path):
  import hashlib

  work = tmp_path / "work"
  work.mkdir()
  retained = tmp_path / "retained"
  retained.mkdir()
  content = b"previously verified native binary"
  sha = hashlib.sha256(content).hexdigest()
  binary = retained / f"{sha}.so"
  binary.write_bytes(content)
  original_mtime = binary.stat().st_mtime_ns
  index = {"artifacts": {"execution": {"dsl": str(binary)}}}
  result = suite.retain(tmp_path, index, work)
  assert result["artifacts"]["execution"]["dsl"] == str(binary)
  assert binary.read_bytes() == content
  assert binary.stat().st_mtime_ns == original_mtime
  assert result["retained_binary_sha256"][str(binary)] == sha


def test_batch_worker_reuses_process_and_persists_each_group(monkeypatch, tmp_path):
  groups = [{"id": name, "index": i, "execution_ids": [name]} for i, name in enumerate(("a", "b"))]
  request = tmp_path / "batch.json"
  suite.save(request, {"groups": [{"id": "a"}, {"id": "b", "recovering": True}]})
  observed = []

  def worker(options, source, plan, group, directory):
    marker = suite.read(request.with_suffix(".progress.json"))
    assert marker["group_id"] == group["id"] and not marker.get("completed")
    if group["id"] == "b":
      assert suite.read(tmp_path / "groups/a/time.json")["status"] == "timed"
    observed.append((group["id"], options.resume_after_row_failure))
    return {"status": "timed"}

  monkeypatch.setattr(suite, "timing_worker", worker)
  assert suite.timing_batch_worker(args(tmp_path), None, {"groups": groups}, request) == 0
  assert observed == [("a", False), ("b", True)]
  assert suite.read(request.with_suffix(".progress.json"))["completed"]


@pytest.mark.parametrize("timed_out", [False, True])
def test_batch_crash_or_timeout_preserves_completed_and_restarts_unstarted(monkeypatch, tmp_path, timed_out):
  groups = [{"id": name, "index": i, "execution_ids": [name + "1", name + "2"]}
            for i, name in enumerate(("a", "b", "c"))]
  for group in groups:
    suite.save(tmp_path / "groups" / group["id"] / "build.json", {"status": "built"})
  calls = []
  clock = [100.0]
  monkeypatch.setattr(suite.time, "monotonic", lambda: clock[0])
  monkeypatch.setattr(suite.time, "monotonic_ns", lambda: int(clock[0] * 1e9))
  monkeypatch.setattr(suite.time, "sleep", lambda _: None)
  monkeypatch.setattr(suite, "stop", lambda process: setattr(process, "stopped", True))

  class Process:
    def __init__(self, command, **kwargs):
      self.request = Path(command[command.index("--worker-batch-file") + 1])
      self.entries = suite.read(self.request)["groups"]
      calls.append(self.entries)
      self.pid = 100 + len(calls)
      self.stopped = False
      self.polls = 0
      if len(calls) == 1:
        # Group A was fully committed, group B crashes in its first row.
        suite.save(tmp_path / "groups/a/time.json", {"status": "timed"})
        suite.save(self.request.with_suffix(".progress.json"),
                   {"group_id": "b", "worker_pid": self.pid, "started_monotonic_ns": int(clock[0] * 1e9)})
        suite.save(tmp_path / "groups/b/timing.json",
                   {"b1/compute": {"status": "running", "worker_pid": self.pid}})
      else:
        for entry in self.entries:
          directory = tmp_path / "groups" / entry["id"]
          records = suite.read(directory / "timing.json", {})
          for eid in next(g for g in groups if g["id"] == entry["id"])["execution_ids"]:
            records.setdefault(eid + "/compute", {"status": "passed"})
          suite.save(directory / "timing.json", records)
          suite.save(directory / "time.json", {"status": "timed"})

    def poll(self):
      self.polls += 1
      if len(calls) == 1 and timed_out and not self.stopped:
        clock[0] += 11
        return None
      return -signal.SIGILL if len(calls) == 1 else 0

  monkeypatch.setattr(suite.subprocess, "Popen", Process)
  suite.dispatch(args(tmp_path, timing_batch_size=3, retry_failed=True), {}, groups, "time")
  assert [[g["id"] for g in batch] for batch in calls] == [["a", "b", "c"], ["b", "c"]]
  assert calls[1][0]["recovering"]
  assert suite.read(tmp_path / "groups/a/time.json")["status"] == "timed"
  rows = suite.read(tmp_path / "groups/b/timing.json")
  assert rows["b1/compute"]["status"] == ("timeout" if timed_out else "signal_killed")
  assert rows["b2/compute"]["status"] == "passed"
  assert suite.read(tmp_path / "groups/c/time.json")["status"] == "timed"


def test_batch_setup_exception_does_not_drop_following_groups(monkeypatch, tmp_path):
  groups = [{"id": name, "index": i, "execution_ids": [name]} for i, name in enumerate(("a", "b"))]
  request = tmp_path / "batch.json"
  suite.save(request, {"groups": [{"id": "a"}, {"id": "b"}]})

  def worker(options, source, plan, group, directory):
    if group["id"] == "a":
      raise ValueError("group-specific setup failure")
    return {"status": "timed"}

  monkeypatch.setattr(suite, "timing_worker", worker)
  suite.timing_batch_worker(args(tmp_path), None, {"groups": groups}, request)
  assert suite.read(tmp_path / "groups/a/timing.json")["a/compute"]["status"] == "group_setup_failed"
  assert suite.read(tmp_path / "groups/b/time.json")["status"] == "timed"


def test_retain_build_identities_and_scalable_sink_attestation(tmp_path):
  work = tmp_path / "work"
  (work / "identities").mkdir(parents=True)
  (work / "CMakeFiles/scalable-sink").mkdir(parents=True)
  (work / "identities/compiler.json").write_text('{}')
  (work / "CMakeFiles/scalable-sink/attestation.txt").write_text('probe evidence')
  index = suite.retain(tmp_path, {}, work)
  with tarfile.open(index["evidence_archive"]) as archive:
    assert "work/identities/compiler.json" in archive.getnames()
    assert "work/CMakeFiles/scalable-sink/attestation.txt" in archive.getnames()


def test_batch_setup_exception_terminalizes_own_running_row_preserves_passed(monkeypatch, tmp_path):
  group = {"id": "a", "index": 0, "execution_ids": ["passed", "running", "unstarted"]}
  request = tmp_path / "batch.json"
  suite.save(request, {"groups": [{"id": "a"}]})

  def worker(options, source, plan, group, directory):
    suite.save(directory / "timing.json", {
      "passed/compute": {"status": "passed"},
      "running/compute": {"status": "running", "worker_pid": suite.os.getpid()},
    })
    raise OSError("row ledger persistence failed")

  monkeypatch.setattr(suite, "timing_worker", worker)
  suite.timing_batch_worker(args(tmp_path), None, {"groups": [group]}, request)
  records = suite.read(tmp_path / "groups/a/timing.json")
  assert records["passed/compute"]["status"] == "passed"
  assert records["running/compute"]["status"] == "group_setup_failed"
  assert "row ledger persistence failed" in records["running/compute"]["error"]
  assert records["running/compute"]["finished_monotonic_ns"] > 0
  assert records["unstarted/compute"]["status"] == "group_setup_failed"
