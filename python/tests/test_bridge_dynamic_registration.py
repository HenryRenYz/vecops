# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Typed Torch bridge artifacts are independent of dispatcher registration names."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
import vecops
import vecops.language as T


def _copy_factory(compiler):
  @T.jit(out_idx=-1, compiler=compiler)
  def factory():
    @T.template
    def copy(A: T.Tensor((4,), dtype="float32"), C: T.Tensor((4,), dtype="float32")):
      for index in T.grid(4, parallel="require"):
        T.store(C[index], T.load(A[index]))

    return copy

  return factory


def test_bridge_source_and_artifact_key_ignore_op_name_and_library():
  from vecops import _schema_bridge

  definition = vecops.KernelDef(
    "first::copy_a",
    [
      vecops.TensorDef("A", [16], [1], dtype=vecops.float32),
      vecops.TensorDef("C", [16], [1], dtype=vecops.float32, access=vecops.TensorAccess.output),
    ],
  )._native()
  source_a = _schema_bridge._generate_torch_source("first", "copy_a", definition)
  source_b = _schema_bridge._generate_torch_source("second", "copy_b", definition)
  assert source_a == source_b
  assert '"first"' not in source_a and '"second"' not in source_a
  assert "copy_a" not in source_a and "copy_b" not in source_a
  assert "TORCH_LIBRARY" not in source_a
  assert "struct CpuFunctor" in source_a and "struct MetaFunctor" in source_a
  assert "vecops_torch_bridge_register_v2" in source_a
  signature = _schema_bridge._torch_schema("vecops_signature", definition)
  assert _schema_bridge._bridge_artifact_key(source_a, ("torch",), "compiler", (), signature) == (
    _schema_bridge._bridge_artifact_key(source_b, ("torch",), "compiler", (), signature)
  )

  symbolic = vecops.KernelDef("first::symbols", [vecops.TensorDef("A", ["M", "N"], dtype=vecops.float32)])._native()
  ordered = list(symbolic.values)
  reversed_schema = _schema_bridge._torch_schema("vecops_signature", symbolic, spec_order=tuple(reversed(ordered)))
  normal_schema = _schema_bridge._torch_schema("vecops_signature", symbolic, spec_order=tuple(ordered))
  signature_source = _schema_bridge._generate_torch_source("first", "symbols", symbolic)
  assert signature_source == _schema_bridge._generate_torch_source("second", "other", symbolic)
  assert _schema_bridge._bridge_artifact_key(signature_source, (), "compiler", (), normal_schema) != (
    _schema_bridge._bridge_artifact_key(signature_source, (), "compiler", (), reversed_schema)
  )


@pytest.mark.parametrize("nthreads", [2, 4])
def test_same_artifact_can_register_independent_rw_and_co_names(tmp_path, monkeypatch, nthreads):
  torch = pytest.importorskip("torch")
  from dsl_test_utils import make_compiler
  from vecops import _schema_bridge

  bridge_cache = tmp_path / "bridge-cache"
  monkeypatch.setenv("VECOPS_TORCH_BRIDGE_DIR", str(bridge_cache))
  rw = make_compiler(target="Native", jobs=2, cache_dir=tmp_path / "kernels", build_dir=tmp_path / "rw-build")
  torch.set_num_threads(nthreads)
  keys_before = set(_schema_bridge._registrations)
  first = _copy_factory(rw)()
  data = torch.arange(4, dtype=torch.float32)
  result = first(data)
  torch.testing.assert_close(result, data)
  with pytest.raises(RuntimeError):
    first(data.to(torch.int32))
  meta_input = torch.empty((4,), dtype=torch.float32, device="meta")
  meta_output = torch.empty_like(meta_input)
  assert first._core.torch()(meta_input, meta_output) is meta_output
  co = vecops.Compiler(
    target="Native",
    cache_mode="cache-only",
    cache_dir=tmp_path / "kernels",
    build_dir=tmp_path / "co-build",
    cc=rw.cc,
    cxx=rw.cxx,
    jobs=2,
  )
  second = _copy_factory(co)()
  result2 = second(data)
  torch.testing.assert_close(result2, data)
  registrations = [
    r
    for key, r in _schema_bridge._registrations.items()
    if key not in keys_before and r.materialized and r.library_path
  ]
  assert len({r.library_path for r in registrations if r.library == "vecops_language"}) == 1
  assert len({r.name for r in registrations if r.library == "vecops_language"}) >= 2
  assert len({r.operator._handle for r in registrations if r.library == "vecops_language"}) >= 2
  assert list(bridge_cache.rglob("*vecops_bridge_*.so"))
  first_registration, second_registration = [r for r in registrations if r.library == "vecops_language"]
  # The registrations share a DSO, but every typed functor must retain its own
  # operator handle. Exercise both contexts concurrently to catch a static
  # handle being overwritten by the most recent registration.
  with ThreadPoolExecutor(max_workers=2) as workers:
    concurrent = list(workers.map(lambda kernel: kernel(data), (first, second) * 8))
  for output in concurrent:
    torch.testing.assert_close(output, data)
  isolated = _schema_bridge._BridgeRegistration(
    f"vecops_isolated_bridge_test_{nthreads}",
    "copy",
    second_registration.operator,
    second_registration.kernel_def,
    second_registration.compiler,
    second_registration.build_directory,
    second_registration.verbose,
    second_registration.extra_cflags,
    second_registration.return_outputs,
    spec_order=second_registration.spec_order,
  )
  _schema_bridge._load_bridge(isolated, first_registration.library_path, Path(vecops._C.__file__), torch)
  isolated_output = torch.empty_like(data)
  getattr(torch.ops, isolated.library).copy.default(data, isolated_output)
  torch.testing.assert_close(isolated_output, data)
  conflicting = _schema_bridge._BridgeRegistration(
    first_registration.library,
    first_registration.name,
    second_registration.operator,
    first_registration.kernel_def,
    first_registration.compiler,
    first_registration.build_directory,
    first_registration.verbose,
    first_registration.extra_cflags,
    first_registration.return_outputs,
    spec_order=first_registration.spec_order,
  )
  with pytest.raises(RuntimeError, match="different operator handle"):
    _schema_bridge._load_bridge(
      conflicting,
      first_registration.library_path,
      Path(vecops._C.__file__),
      torch,
    )
  conflicting_context = _schema_bridge._BridgeRegistration(
    first_registration.library,
    first_registration.name,
    first_registration.operator,
    first_registration.kernel_def,
    first_registration.compiler,
    first_registration.build_directory,
    first_registration.verbose,
    first_registration.extra_cflags,
    not first_registration.return_outputs,
    spec_order=first_registration.spec_order,
  )
  with pytest.raises(RuntimeError, match="different operator handle or bridge context"):
    _schema_bridge._load_bridge(
      conflicting_context,
      first_registration.library_path,
      Path(vecops._C.__file__),
      torch,
    )


def test_cache_only_bridge_reuses_rw_artifact_in_a_fresh_process(tmp_path):
  from vecops._compiler import _sdk

  # The test's checkout is not necessarily the selected distribution. In an
  # editable install the extension is elsewhere; in a wheel even the Python
  # package is elsewhere. Preserve and verify the parent's actual selection.
  root = Path(__file__).resolve().parents[2]
  sdk, layout = _sdk()
  bridge_cache = tmp_path / "bridges"
  env = os.environ.copy()
  env["VECOPS_BRIDGE_TEST_ROOT"] = str(root)
  env["VECOPS_BRIDGE_TEST_DISTRIBUTION"] = json.dumps(
    {
      "package": str(Path(vecops.__file__).resolve()),
      "extension": str(Path(vecops._C.__file__).resolve()),
      "sdk": str(sdk.resolve()),
      "sdk_layout": str(layout),
    }
  )
  env["VECOPS_BRIDGE_CACHE"] = str(tmp_path / "kernels")
  env["VECOPS_TORCH_BRIDGE_DIR"] = str(bridge_cache)
  names = []
  for mode in ("read-write", "cache-only", "read-write"):
    env["VECOPS_BRIDGE_MODE"] = mode
    env["VECOPS_BRIDGE_BUILD"] = str(tmp_path / f"{mode}-build")
    before = (
      sorted(
        (path.relative_to(bridge_cache), path.stat().st_size, path.stat().st_mtime_ns)
        for path in bridge_cache.rglob("*")
        if path.is_file()
      )
      if bridge_cache.exists()
      else []
    )
    process = subprocess.run(
      [sys.executable, str(Path(__file__).with_name("bridge_dynamic_registration_child.py"))],
      cwd=root,
      env=env,
      capture_output=True,
      text=True,
      check=False,
    )
    assert process.returncode == 0, process.stdout + process.stderr
    names.append(process.stdout.strip().splitlines()[-1])
    if mode == "cache-only":
      after = sorted(
        (path.relative_to(bridge_cache), path.stat().st_size, path.stat().st_mtime_ns)
        for path in bridge_cache.rglob("*")
        if path.is_file()
      )
      assert after == before
      assert not (tmp_path / "cache-only-build").exists()
  # The read-write caller after cache-only must reuse the same artifact without
  # replacing the cache-only registration or compiling a mode-specific DSO.
  assert names[0] == names[2]
  assert names[0] != names[1]
  assert len(list(bridge_cache.rglob("*.identity"))) == 1
