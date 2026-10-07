"""Local fake-tree tests for the distributable ARM job-environment helper."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import io
import json
import sys
import tarfile
from pathlib import Path

import pytest

HELPER_PATH = Path(__file__).parents[2] / "benchmarks/arm_job_env_helper.py"
SPEC = importlib.util.spec_from_file_location("arm_job_env_helper", HELPER_PATH)
assert SPEC and SPEC.loader
helper = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = helper
SPEC.loader.exec_module(helper)


def _write_tar(path: Path, entries: dict[str, bytes], *, symlinks=()) -> str:
  with tarfile.open(path, "w:gz") as archive:
    for name, payload in entries.items():
      item = tarfile.TarInfo(name)
      item.size = len(payload)
      archive.addfile(item, io.BytesIO(payload))
    for name, target in symlinks:
      item = tarfile.TarInfo(name)
      item.type = tarfile.SYMTYPE
      item.linkname = target
      archive.addfile(item)
  return hashlib.sha256(path.read_bytes()).hexdigest()


def test_helper_imports_only_python_standard_library():
  tree = ast.parse(HELPER_PATH.read_text())
  imported = set()
  for node in ast.walk(tree):
    if isinstance(node, ast.Import):
      imported.update(alias.name.split(".", 1)[0] for alias in node.names)
    elif isinstance(node, ast.ImportFrom) and node.module:
      imported.add(node.module.split(".", 1)[0])
  assert not imported & {"numpy", "torch", "vecops"}
  assert imported <= {
    "__future__",
    "argparse",
    "ast",
    "collections",
    "hashlib",
    "json",
    "os",
    "pathlib",
    "re",
    "shlex",
    "shutil",
    "stat",
    "subprocess",
    "sys",
    "tarfile",
    "tempfile",
  }


def test_prepare_checks_readable_sourceable_shell_and_probe_before_mkdir(tmp_path, monkeypatch):
  scratch = tmp_path / "future-root"
  monkeypatch.setattr(helper, "_private_root", lambda: pytest.fail("must fail before creating task root"))
  clang = tmp_path / "clang"
  clangxx = tmp_path / "clang++"
  for compiler in (clang, clangxx):
    compiler.write_text("fake compiler")
    compiler.chmod(0o700)
  runtime = tmp_path / "runtime"
  runtime.mkdir()
  libgcc = runtime / "libgcc_s.so.1"
  libstdcxx = runtime / "libstdc++.so.6"
  libgcc.write_bytes(b"gcc")
  libstdcxx.write_bytes(b"cxx")
  bundle = tmp_path / "bundle.tar.gz"
  bundle_sha = _write_tar(bundle, {"x": b"unused"})
  shell = tmp_path / "arm_job_env.sh"
  probe = tmp_path / "arm_toolchain_probe.cpp"
  shell.write_text("# sourceable\n")
  probe.write_text("// probe\n")
  monkeypatch.setattr(helper, "__file__", str(tmp_path / "missing_helper.py"))
  with pytest.raises(FileNotFoundError):
    helper.prepare(
      python=sys.executable,
      conda_prefix=sys.prefix,
      clang=clang,
      clangxx=clangxx,
      bundle=bundle,
      bundle_sha256=bundle_sha,
      libgcc=libgcc,
      libgcc_sha256=helper.sha256_file(libgcc),
      libstdcxx=libstdcxx,
      libstdcxx_sha256=helper.sha256_file(libstdcxx),
      shell_path=shell,
      probe_path=probe,
    )
  assert not scratch.exists()


def test_prepare_materializes_private_headers_and_preloads_runtime_before_python(tmp_path, monkeypatch):
  # This fixture describes a fresh job, not a second nested preparation inside
  # the real ARM test runner. The real helper must still reject stale selectors.
  for name in ("CXXFLAGS", "LDFLAGS", "LD_PRELOAD", "LD_LIBRARY_PATH"):
    monkeypatch.delenv(name, raising=False)
  install = "installation/lib/gcc/aarch64-conda-linux-gnu/14.4.0"
  runtime = tmp_path / "runtime"
  runtime.mkdir()
  libgcc, libstdcxx = runtime / "libgcc_s.so.1", runtime / "libstdc++.so.6"
  libgcc.write_bytes(b"libgcc")
  libstdcxx.write_bytes(b"libstdc++")
  bundle = tmp_path / "gcc14.tar.gz"
  bundle_sha = _write_tar(
    bundle,
    {
      f"{install}/libgcc.a": b"selector",
      f"{install}/include/c++/format": b"format",
      f"{install}/include/c++/aarch64-conda-linux-gnu/bits/c++config.h": b"target",
      f"{install}/include/c++/backward/strstream": b"backward",
    },
    symlinks=[
      ("installation/lib/libgcc_s.so", str(libgcc)),
      ("installation/lib/libgcc_s.so.1", str(libgcc)),
      ("installation/lib/libstdc++.so", str(libstdcxx)),
      ("installation/lib/libstdc++.so.6", str(libstdcxx)),
    ],
  )
  clang, clangxx = tmp_path / "clang-19", tmp_path / "clang++"
  clang.write_text("fake clang")
  clang.chmod(0o700)
  clangxx.symlink_to(clang)
  project = tmp_path / "project"
  shell = project / "benchmarks/arm_job_env.sh"
  probe = project / "benchmarks/arm_toolchain_probe.cpp"
  shell.parent.mkdir(parents=True)
  shell.write_text("# sourceable\n")
  probe.write_text("// compiler probe\n")
  header = project / "include/vecops/execution/Print.h"
  header.parent.mkdir(parents=True)
  header.write_text("// print header\n")
  root = tmp_path / "private-root"
  root.mkdir(mode=0o700)
  root.chmod(0o700)
  monkeypatch.setattr(helper, "_private_root", lambda: root)
  monkeypatch.setattr(
    helper, "_validate_af3", lambda *_args: {"executable": "/fake/af3/bin/python", "prefix": "/fake/af3"}
  )
  commands = []

  def fake_run(command, **_kwargs):
    commands.append(command)
    output = "clang version 19.1" if command[-1] == "--version" else "probe ok"
    if "-o" in command:
      Path(command[command.index("-o") + 1]).write_bytes(b"fake ELF probe")
    return type("Completed", (), {"stdout": output, "stderr": "", "returncode": 0})()

  monkeypatch.setattr(
    helper.subprocess,
    "run",
    fake_run,
  )
  result = helper.prepare(
    python="/fake/af3/bin/python",
    conda_prefix="/fake/af3",
    clang=clang,
    clangxx=clangxx,
    bundle=bundle,
    bundle_sha256=bundle_sha,
    libgcc=libgcc,
    libgcc_sha256=helper.sha256_file(libgcc),
    libstdcxx=libstdcxx,
    libstdcxx_sha256=helper.sha256_file(libstdcxx),
    shell_path=shell,
    probe_path=probe,
  )
  env_text = Path(result["job_env"]).read_text()
  manifest = json.loads(Path(result["manifest"]).read_text())
  assert f"export LD_PRELOAD={libgcc}:{libstdcxx}" in env_text
  assert manifest["gcc_install"].endswith("aarch64-conda-linux-gnu/14.4.0")
  assert manifest["clang"] == str(clang.resolve())
  assert manifest["clangxx"] == str(clangxx.absolute())
  assert manifest["compiler_identity"]["clangxx"]["resolved_path"] == str(clang.resolve())
  assert manifest["compiler_identity"]["clangxx"]["path"].endswith("/clang++")
  assert manifest["compiler_identity"]["clangxx"]["path"] != manifest["compiler_identity"]["clangxx"]["resolved_path"]
  assert any(command[0] == str(clangxx.absolute()) for command in commands)
  assert manifest["source_shell_sha256"] == helper.sha256_file(shell)
  assert manifest["probe_sha256"] == helper.sha256_file(probe)
  assert manifest["print_header_sha256"] == helper.sha256_file(header)
  assert manifest["probe"] == {"syntax": "passed", "link": "passed", "run": "passed"}
  assert Path(result["job_env"]).stat().st_mode & 0o777 == 0o600
  syntax = next(command for command in commands if "-fsyntax-only" in command)
  link = next(command for command in commands if "-o" in command)
  assert f"-I{project / 'include'}" in syntax and "-pthread" in syntax
  assert f"-I{project / 'include'}" in link and "-pthread" in link and "-O2" in link


@pytest.mark.parametrize(
  "flags", ("--gcc-install-dir=/tmp/vecops-existing-private-tree", "-I/dev/shm/vecops-old/include")
)
def test_fake_prepare_does_not_inherit_active_job_toolchain(tmp_path, monkeypatch, flags):
  monkeypatch.setenv("CXXFLAGS", flags)
  monkeypatch.setenv("LDFLAGS", "-L/tmp/vecops-existing-private-tree/lib")
  test_prepare_materializes_private_headers_and_preloads_runtime_before_python(tmp_path, monkeypatch)


def test_bundle_rejects_symlink_member_and_runtime_hash_mismatch(tmp_path):
  bundle = tmp_path / "unsafe.tar.gz"
  with tarfile.open(bundle, "w:gz") as archive:
    item = tarfile.TarInfo("include/c++/link")
    item.type = tarfile.SYMTYPE
    item.linkname = "../../outside"
    archive.addfile(item)
  digest = hashlib.sha256(bundle.read_bytes()).hexdigest()
  with pytest.raises(ValueError, match="links/special files"):
    helper._extract_bundle(bundle, digest, tmp_path / "extract")
  with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
    helper._extract_bundle(bundle, "0" * 64, tmp_path / "wrong")


def _write_runtime_link_bundle(path: Path, link_members: list[tuple[str, str]], extra_files=()) -> str:
  with tarfile.open(path, "w:gz") as archive:
    for name, payload in extra_files:
      item = tarfile.TarInfo(name)
      item.size = len(payload)
      archive.addfile(item, io.BytesIO(payload))
    for name, target in link_members:
      item = tarfile.TarInfo(name)
      item.type = tarfile.SYMTYPE
      item.linkname = target
      archive.addfile(item)
  return hashlib.sha256(path.read_bytes()).hexdigest()


def test_bundle_allows_only_exact_hashed_durable_runtime_links(tmp_path):
  runtime = tmp_path / "durable-runtime"
  runtime.mkdir()
  libgcc = runtime / "libgcc_s.so.1"
  libstdcxx = runtime / "libstdc++.so.6"
  libgcc.write_bytes(b"verified gcc runtime")
  libstdcxx.write_bytes(b"verified cxx runtime")
  gcc_hash = helper.sha256_file(libgcc)
  cxx_hash = helper.sha256_file(libstdcxx)
  links = [
    ("installation/lib/libgcc_s.so", str(libgcc)),
    ("installation/lib/libgcc_s.so.1", str(libgcc)),
    ("installation/lib/libstdc++.so", str(libstdcxx)),
    ("installation/lib/libstdc++.so.6", str(libstdcxx)),
  ]
  bundle = tmp_path / "runtime-links.tar.gz"
  digest = _write_runtime_link_bundle(bundle, links, [("installation/include/c++/format", b"format")])
  allowed = {
    "installation/lib/libgcc_s.so": (libgcc, gcc_hash),
    "installation/lib/libgcc_s.so.1": (libgcc, gcc_hash),
    "installation/lib/libstdc++.so": (libstdcxx, cxx_hash),
    "installation/lib/libstdc++.so.6": (libstdcxx, cxx_hash),
  }
  extracted = helper._extract_bundle(bundle, digest, tmp_path / "good", allowed_runtime_links=allowed)
  assert len(extracted) == 5
  for relative, target in links:
    link = tmp_path / "good" / relative
    assert link.is_symlink() and link.resolve(strict=True) == Path(target)
    assert helper.sha256_file(link) == (gcc_hash if "libgcc" in relative else cxx_hash)


def test_bundle_requires_complete_pinned_runtime_link_set(tmp_path):
  libgcc = tmp_path / "libgcc_s.so.1"
  libstdcxx = tmp_path / "libstdc++.so.6"
  libgcc.write_bytes(b"gcc")
  libstdcxx.write_bytes(b"cxx")
  links = {
    "installation/lib/libgcc_s.so": (libgcc, helper.sha256_file(libgcc)),
    "installation/lib/libgcc_s.so.1": (libgcc, helper.sha256_file(libgcc)),
    "installation/lib/libstdc++.so": (libstdcxx, helper.sha256_file(libstdcxx)),
    "installation/lib/libstdc++.so.6": (libstdcxx, helper.sha256_file(libstdcxx)),
  }
  partial = tmp_path / "partial.tar.gz"
  digest = _write_runtime_link_bundle(
    partial,
    [(name, str(target)) for name, (target, _) in list(links.items())[:-1]],
  )
  with pytest.raises(ValueError, match="runtime symlink set differs"):
    helper._extract_bundle(partial, digest, tmp_path / "partial", allowed_runtime_links=links)


@pytest.mark.parametrize(
  ("member_name", "link_target", "extra_files", "allowed_name", "expected_error"),
  [
    (
      "installation/lib/libgcc_s.so.2",
      "RUNTIME_GCC",
      (),
      "installation/lib/libgcc_s.so",
      "unapproved runtime symlink",
    ),
    (
      "installation/lib/libgcc_s.so",
      "/tmp/attacker-libgcc.so",
      (),
      "installation/lib/libgcc_s.so",
      "unapproved runtime symlink",
    ),
    (
      "installation/lib/libgcc_s.so",
      "installation/lib/libgcc_s.so.1",
      (),
      "installation/lib/libgcc_s.so",
      "unapproved runtime symlink",
    ),
    (
      "installation/lib/other.so",
      "/tmp/attacker.so",
      (),
      "installation/lib/libgcc_s.so",
      "unapproved runtime symlink",
    ),
    (
      "installation/lib/libgcc_s.so",
      "RUNTIME_GCC",
      (("installation/lib/libgcc_s.so/payload", b"escape"),),
      "installation/lib/libgcc_s.so",
      "nested below a runtime symlink",
    ),
  ],
)
def test_bundle_rejects_wrong_runtime_link_name_target_and_chain(
  tmp_path, member_name, link_target, extra_files, allowed_name, expected_error
):
  runtime = tmp_path / "runtime.so"
  runtime.write_bytes(b"runtime")
  bundle = tmp_path / "candidate.tar.gz"
  if link_target == "RUNTIME_GCC":
    link_target = str(runtime)
  digest = _write_runtime_link_bundle(bundle, [(member_name, link_target)], extra_files)
  allowed = {allowed_name: (runtime, helper.sha256_file(runtime))}
  with pytest.raises(ValueError, match=expected_error):
    helper._extract_bundle(bundle, digest, tmp_path / "rejected", allowed_runtime_links=allowed)


def test_bundle_rejects_runtime_target_hash_change(tmp_path):
  runtime = tmp_path / "runtime.so"
  runtime.write_bytes(b"changed runtime")
  bundle = tmp_path / "candidate.tar.gz"
  digest = _write_runtime_link_bundle(bundle, [("installation/lib/libgcc_s.so", str(runtime))])
  with pytest.raises(RuntimeError, match="allowed runtime link target SHA-256 mismatch"):
    helper._extract_bundle(
      bundle,
      digest,
      tmp_path / "rejected",
      allowed_runtime_links={"installation/lib/libgcc_s.so": (runtime, "0" * 64)},
    )
