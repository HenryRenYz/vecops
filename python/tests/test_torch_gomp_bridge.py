# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Focused policy and source tests for the Clang-to-Torch-GOMP adapter."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import vecops


def _config(c_compiler: str = "/toolchain/bin/clang", cxx_compiler: str = "/toolchain/bin/clang++"):
  toolchain = vecops._C.ToolchainSpec()
  toolchain.c_compiler = c_compiler
  toolchain.cxx_compiler = cxx_compiler
  toolchain.environment = {"CXXFLAGS": "-nostdinc++ -isystem /private/cxx/include -L/private/cxx/lib"}
  config = vecops._C.KernelCompilerConfig()
  config.toolchain = toolchain
  return config


def test_clang_gomp_policy_binds_exact_runtime_without_openmp_compiler_flags(monkeypatch, tmp_path: Path):
  from vecops import _schema_bridge

  packaged = tmp_path / "torch" / "lib"
  packaged.mkdir(parents=True)
  runtime = packaged / "libgomp.so.1"
  runtime.write_bytes(b"runtime-fixture")
  monkeypatch.delenv("VECOPS_TORCH_BRIDGE_CC", raising=False)
  monkeypatch.delenv("VECOPS_TORCH_BRIDGE_CXX", raising=False)
  monkeypatch.setattr(_schema_bridge, "_verify_gomp_runtime", lambda path: (str(path), "sha256:fixture"))
  monkeypatch.setattr(_schema_bridge.shutil, "which", lambda _name: pytest.fail("must not probe or switch compilers"))

  policy = _schema_bridge._torch_bridge_build_policy(_config(), (packaged,), "ATen parallel backend: OpenMP")

  assert policy.parallel_backend == "gomp"
  assert policy.compile_flags == ()
  assert policy.link_libraries == (str(runtime.resolve()),)
  assert policy.link_options == ("-Wl,--no-as-needed",)
  assert policy.runtime_path == runtime.resolve()
  assert policy.runtime_identity == (str(runtime.resolve()), "sha256:fixture")
  assert str(policy.toolchain.cxx_compiler) == "/toolchain/bin/clang++"
  assert policy.toolchain.environment["CXXFLAGS"].endswith("-L/private/cxx/lib")


def test_clang_gomp_policy_rejects_missing_or_ambiguous_runtime(monkeypatch, tmp_path: Path):
  from vecops import _schema_bridge

  monkeypatch.delenv("VECOPS_TORCH_BRIDGE_CC", raising=False)
  monkeypatch.delenv("VECOPS_TORCH_BRIDGE_CXX", raising=False)
  with pytest.raises(RuntimeError, match="could not be identified unambiguously"):
    _schema_bridge._torch_bridge_build_policy(_config(), (tmp_path,), "ATen parallel backend: OpenMP")

  library_dir = tmp_path / "libs"
  library_dir.mkdir()
  (library_dir / "libgomp.so.1").write_bytes(b"one")
  (library_dir / "libomp.so").write_bytes(b"other-runtime")
  with pytest.raises(RuntimeError, match="unambiguously"):
    _schema_bridge._torch_bridge_build_policy(_config(), (library_dir,), "ATen parallel backend: OpenMP")


def test_unknown_gomp_compiler_fails_without_implicit_compiler_switch(monkeypatch, tmp_path: Path):
  from vecops import _schema_bridge

  library_dir = tmp_path / "torch-libs"
  library_dir.mkdir()
  (library_dir / "libgomp.so.1").write_bytes(b"runtime")
  monkeypatch.delenv("VECOPS_TORCH_BRIDGE_CC", raising=False)
  monkeypatch.delenv("VECOPS_TORCH_BRIDGE_CXX", raising=False)
  monkeypatch.setattr(_schema_bridge, "_torch_openmp_runtime", lambda *_args: "gomp")
  monkeypatch.setattr(_schema_bridge.shutil, "which", lambda _name: pytest.fail("implicit compiler switch"))

  with pytest.raises(RuntimeError, match="neither Clang nor GNU"):
    _schema_bridge._torch_bridge_build_policy(_config("/toolchain/c", "/toolchain/custom-cxx"), (library_dir,), "")


def test_gomp_policy_rejects_flags_that_would_link_another_openmp_runtime(monkeypatch, tmp_path: Path):
  from vecops import _schema_bridge

  library_dir = tmp_path / "torch-libs"
  library_dir.mkdir()
  (library_dir / "libgomp.so.1").write_bytes(b"runtime")
  config = _config()
  toolchain = config.toolchain
  environment = dict(toolchain.environment)
  environment["CXXFLAGS"] += " -fopenmp=libomp"
  toolchain.environment = environment
  config.toolchain = toolchain
  monkeypatch.delenv("VECOPS_TORCH_BRIDGE_CC", raising=False)
  monkeypatch.delenv("VECOPS_TORCH_BRIDGE_CXX", raising=False)
  monkeypatch.setattr(_schema_bridge, "_verify_gomp_runtime", lambda path: (str(path), "sha256:fixture"))

  with pytest.raises(RuntimeError, match="cannot use compiler OpenMP flags"):
    _schema_bridge._torch_bridge_build_policy(config, (library_dir,), "ATen parallel backend: OpenMP")


def test_gomp_adapter_mirrors_aten_team_chunk_guards_and_errors():
  from vecops import _schema_bridge

  definition = vecops.KernelDef(
    "bridge::gomp_test",
    [vecops.TensorDef("input", [1], [1], dtype=vecops.float32)],
  )._native()
  source = _schema_bridge._generate_torch_source("vecops_test", "run_gomp", definition, parallel_backend="gomp")

  assert "using GompParallel =" in source
  assert "OmpInt get_max, get_num, get_id, is_nested" in source
  assert "count <= 1 || in_parallel(p) || max_threads(p) <= 1" in source
  assert "at::internal::ThreadIdGuard id_guard(tid);" in source
  assert "c10::ParallelGuard parallel_guard(true);" in source
  assert "s->gomp(&work_gomp, &w, 0, 0)" in source
  assert "vecops_torch_bridge_register_v2" in source
  assert "std::atomic_flag failed" in source
  assert "std::rethrow_exception(w.exception)" in source
  assert "at::parallel_for(" in source


def test_default_torch_bridge_remains_on_aten_adapter():
  from vecops import _schema_bridge

  definition = vecops.KernelDef(
    "bridge::aten_test",
    [vecops.TensorDef("input", [1], [1], dtype=vecops.float32)],
  )._native()
  source = _schema_bridge._generate_torch_source("vecops_test", "run_aten", definition)
  assert "at::parallel_for(0, count, 1" in source
  assert "at::get_num_threads()" in source
  assert "GOMP_parallel" not in source


def test_gomp_symbol_verification_requires_already_loaded_runtime(monkeypatch, tmp_path: Path):
  from vecops import _schema_bridge

  runtime = tmp_path / "libgomp.so.1"
  runtime.write_bytes(b"runtime")
  symbols = {
    name: object()
    for name in (
      "GOMP_parallel",
      "omp_get_max_threads",
      "omp_get_num_threads",
      "omp_get_thread_num",
      "omp_in_parallel",
    )
  }
  symbols["omp_get_max_threads"] = lambda: 3
  monkeypatch.setattr(_schema_bridge.os, "RTLD_NOLOAD", 4, raising=False)
  monkeypatch.setattr(_schema_bridge.os, "RTLD_LOCAL", 0, raising=False)
  monkeypatch.setattr(_schema_bridge.os, "RTLD_NOW", 2, raising=False)
  monkeypatch.setattr(_schema_bridge, "_mapped_openmp_runtime_paths", lambda: {runtime.resolve()})
  monkeypatch.setattr(_schema_bridge.ctypes, "CDLL", lambda *_args, **_kwargs: SimpleNamespace(**symbols))
  addresses = {symbol: index + 100 for index, symbol in enumerate(symbols.values())}
  monkeypatch.setattr(_schema_bridge.ctypes, "cast", lambda symbol, _type: SimpleNamespace(value=addresses[symbol]))
  monkeypatch.setattr(_schema_bridge, "_dladdr_library", lambda _address: runtime.resolve())
  monkeypatch.setattr(_schema_bridge, "_verified_gomp_handles", {})
  monkeypatch.setitem(__import__("sys").modules, "torch", SimpleNamespace(get_num_threads=lambda: 3))

  assert _schema_bridge._verify_gomp_runtime(runtime)[0] == str(runtime)


def test_openmp_maps_recognizes_hashed_torch_runtimes_and_deleted_mappings(tmp_path: Path):
  from vecops import _schema_bridge

  torch_gomp = tmp_path / "libgomp-a49a47f9.so.1.0.0"
  torch_omp = tmp_path / "libomp-1234abcd.so.1"
  deleted = tmp_path / "libiomp5.so.5 (deleted)"
  lines = [
    f"7f000000-7f001000 r-xp 00000000 08:01 123 {torch_gomp}",
    f"7f001000-7f002000 r-xp 00000000 08:01 124 {torch_omp}",
    f"7f002000-7f003000 r-xp 00000000 08:01 125 {deleted}",
    f"7f003000-7f004000 r-xp 00000000 08:01 126 {tmp_path / 'libgomp_helper.so'}",
    f"7f004000-7f005000 r-xp 00000000 08:01 127 {tmp_path / 'libgompper.so.1'}",
    f"7f005000-7f006000 r-xp 00000000 08:01 128 {tmp_path / 'libgomp.a'}",
    f"7f006000-7f007000 r-xp 00000000 08:01 129 {tmp_path / 'libgomp_runtime.txt'}",
  ]

  paths = {_schema_bridge._openmp_runtime_path_from_maps_line(line) for line in lines}

  assert paths == {torch_gomp.resolve(), torch_omp.resolve(), (tmp_path / "libiomp5.so.5").resolve(), None}


def test_torch_gomp_discovery_accepts_hash_only_wheel_name_and_deduplicates_symlink(tmp_path: Path):
  from vecops import _schema_bridge

  torch_libs = tmp_path / "torch.libs"
  torch_libs.mkdir()
  runtime = torch_libs / "libgomp-a49a47f9.so.1.0.0"
  runtime.write_bytes(b"runtime")
  (torch_libs / "libgomp.so.1").symlink_to(runtime.name)
  (torch_libs / "libgomp_helper.so").write_bytes(b"not an OpenMP runtime")
  (torch_libs / "libgomp.a").write_bytes(b"not a shared runtime")

  assert _schema_bridge._torch_gomp_library((torch_libs,)) == runtime.resolve()
  assert _schema_bridge._torch_openmp_runtime((torch_libs,), "ATen parallel backend: OpenMP") == "gomp"


def test_read_write_bridge_publishes_artifact_after_unchanged_validation(monkeypatch, tmp_path: Path):
  import vecops._identity as identity
  from vecops import _schema_bridge

  class TorchStub:
    @staticmethod
    def compiled_with_cxx11_abi():
      return False

  compiler_config = vecops.Compiler(
    target="Scalar", cache_dir=tmp_path / "compiler-cache", build_dir=tmp_path / "compiler-build"
  )._native_config()
  build_dir = tmp_path / "bridge-build"
  build_dir.mkdir()

  class BackendStub:
    calls = 0

    def compile_batch(self, batch):
      self.calls += 1
      task = batch.tasks[0]
      artifact = Path(task.artifact_directory) / f"{task.output_name}.so"
      artifact.parent.mkdir(parents=True, exist_ok=True)
      artifact.write_bytes(b"native-bridge-fixture")
      return SimpleNamespace(
        success=True,
        error="",
        configure=None,
        build=None,
        tasks=[SimpleNamespace(success=True, kernel_library=artifact, error="")],
      )

  class CompilerStub:
    cache_mode = vecops._C.ArtifactCacheMode.read_write
    build_backend = BackendStub()

    @staticmethod
    def _native_config():
      return compiler_config

  CompilerStub.build_dir = build_dir

  registration = _schema_bridge._BridgeRegistration(
    "test_lib",
    "test_op",
    object(),
    vecops.KernelDef("bridge::publication", [vecops.TensorDef("input", [1], [1], dtype=vecops.float32)])._native(),
    CompilerStub(),
    tmp_path / "published",
    False,
    (),
    False,
  )
  include_root = tmp_path / "include"
  include_root.mkdir()
  native_library = tmp_path / "native.so"
  native_library.write_bytes(b"native")
  torch_identity = (
    "torch",
    False,
    (),
    (),
    "ATen parallel backend: native",
    f"tree:{include_root}",
    (),
    (),
    (str(native_library), "stable-file"),
  )
  monkeypatch.setattr(
    _schema_bridge,
    "_torch_bridge_inputs",
    lambda: (TorchStub(), include_root, (), (), "ATen parallel backend: native", native_library, torch_identity),
  )
  monkeypatch.setattr(identity, "compiler_identity", lambda _compiler: "compiler-id")
  monkeypatch.setattr(identity, "file_identity", lambda path: (str(path), "stable-file"))
  monkeypatch.setattr(identity, "tree_identity", lambda path, *_args: f"tree:{path}")
  monkeypatch.setattr(_schema_bridge, "_load_bridge", lambda reg, *_args: setattr(reg, "materialized", True))

  assert _schema_bridge._compile_pending_torch_bridges_impl(registrations=[registration]) == 1
  assert registration.materialized
  published = list((tmp_path / "published").glob("*"))
  assert len(published) == 2
  assert any(path.suffix == ".identity" for path in published)
  assert any(path.suffix == ".so" for path in published)
  assert CompilerStub.build_backend.calls == 1

  registration.materialized = False
  assert _schema_bridge._compile_pending_torch_bridges_impl(registrations=[registration]) == 0
  assert CompilerStub.build_backend.calls == 1
  assert registration.materialized
