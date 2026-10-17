# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Native cold-batch deduplication without weakening per-call validation."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import numpy as np
import pytest
import vecops

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "tests/runtime/NamedDynamicKernel.cpp"


def _definition():
  return vecops.KernelDef(
    "test::batch-artifact-dynamic",
    [
      vecops.TensorDef("input", ("B", "D"), ("D", 1), dtype=vecops.float32),
      vecops.TensorDef(
        "output",
        ("B", "D"),
        ("D", 1),
        dtype=vecops.float32,
        access=vecops.TensorAccess.output,
      ),
    ],
    symbols={"B": vecops.Dynamic(lower=0, upper=1024)},
    # This source never specializes on worker count. Match cold preparation
    # and ordinary invocation despite the latter attaching the process pool.
    runtime_parallelism=True,
  )._native()


def _compiler(directory):
  # Use the installed SDK when available, including when a fresh extension is
  # loaded from a separate build directory for isolated regression testing.
  sdk_path = Path(os.environ.get("VECOPS_TEST_BATCH_SDK", Path(vecops._C.__file__).parent / "_sdk"))
  sdk = vecops._C.SdkSpec()
  if (sdk_path / "lib/cmake/Vecops/VecopsConfig.cmake").is_file():
    sdk.path = sdk_path
    sdk.layout = vecops._C.SdkLayout.package
  else:
    from vecops._compiler import _sdk

    sdk.path, sdk.layout = _sdk()
  tools = vecops._C.ToolchainSpec()
  tools.cmake_program = shutil.which("cmake")
  tools.c_compiler = shutil.which(os.environ.get("CC", "gcc"))
  tools.cxx_compiler = shutil.which(os.environ.get("CXX", "g++"))
  tools.parallel_jobs = 2
  tools.generator = "Ninja" if shutil.which("ninja") else "Unix Makefiles"
  config = vecops._C.KernelCompilerConfig()
  config.sdk = sdk
  config.toolchain = tools
  config.work_directory = directory / "build"
  config.target_arch = "Scalar"
  return vecops._C.Compiler(config)


def _operator(compiler, cache, namespace="batch", validate=None, mode=None):
  kwargs = {} if validate is None else {"validate_inputs": validate}
  return vecops._C.Operator(
    SOURCE,
    _definition(),
    compiler=compiler,
    cache_mode=mode or vecops._C.ArtifactCacheMode.read_write,
    cache_directory=cache,
    namespace_key=namespace,
    recipe_id="batch-artifact-test",
    **kwargs,
  )


def _metadata_call(rows, columns=4, *, output_rows=None):
  return vecops._C.KernelCall(
    [
      vecops.TensorMeta((rows, columns), dtype=vecops.float32)._native(),
      vecops.TensorMeta(
        (rows if output_rows is None else output_rows, columns),
        dtype=vecops.float32,
        access=vecops.TensorAccess.output,
      )._native(),
    ]
  )


def _targets(directory):
  return tuple((directory / "build").glob("kernel-batch-*/tasks/*/input/vecops_adapter.cpp"))


def _assert_executes(operator, rows, columns=4):
  source = np.arange(rows * columns, dtype=np.float32).reshape(rows, columns)
  output = np.empty_like(source)
  operator(source, output)
  np.testing.assert_array_equal(output, source + 1)


def test_dynamic_observations_have_one_bound_artifact_identity():
  definition = _definition()
  first = vecops._C.bind_kernel_call(definition, _metadata_call(2))
  second = vecops._C.bind_kernel_call(definition, _metadata_call(19))
  other = vecops._C.bind_kernel_call(definition, _metadata_call(19, 8))
  assert first["specialization_key"] == second["specialization_key"]
  assert first["specialization_key"] != other["specialization_key"]


def test_native_batch_compiles_dynamic_aliases_only_once(tmp_path):
  operator = _operator(_compiler(tmp_path), tmp_path / "cache")
  calls = [_metadata_call(rows) for rows in (2, 19, 23)]
  requests = [
    vecops.CompileRequest(operator, call, (rows,), "test::batch-artifact-dynamic")
    for rows, call in zip((2, 19, 23), calls, strict=True)
  ]
  result = vecops.compile_batch(requests, parallelism=2, execution_parallelism=1)
  assert result.collected == result.prepared == 3
  assert len(_targets(tmp_path)) == 1
  for rows in (2, 19, 23):
    _assert_executes(operator, rows)
  assert vecops.compile_batch(requests, parallelism=2, execution_parallelism=1).prepared == 3
  assert len(_targets(tmp_path)) == 1  # Warm calls perform no new compilation.


def test_native_batch_checks_one_operator_once_per_boundary_and_fresh_batch(tmp_path):
  checks = 0

  def validate():
    nonlocal checks
    checks += 1
    return True

  compiler = _compiler(tmp_path)

  def requests_for(operator):
    return [
      vecops.CompileRequest(operator, _metadata_call(rows), (rows,), "test::batch-artifact-dynamic")
      for rows in range(1, 43)
    ]

  operator = _operator(compiler, tmp_path / "cache", validate=validate)
  requests = requests_for(operator)
  result = vecops.compile_batch(requests, parallelism=2, execution_parallelism=1)
  assert result.collected == result.prepared == 42
  assert len(_targets(tmp_path)) == 1
  assert checks == 2  # One fresh check before building and one before publication.
  for rows in range(1, 43):
    _assert_executes(operator, rows)
  assert vecops.compile_batch(requests, parallelism=2, execution_parallelism=1).prepared == 42
  assert checks == 2  # Warm preparation and invocation do not run cold validators.
  assert len(_targets(tmp_path)) == 1

  # A separate cache forces a genuinely new batch; no previous boundary's
  # validation result may survive, even with the same Python callback/compiler.
  independent = _operator(compiler, tmp_path / "fresh-cache", validate=validate)
  fresh = vecops.compile_batch(requests_for(independent), parallelism=2, execution_parallelism=1)
  assert fresh.collected == fresh.prepared == 42
  assert len(_targets(tmp_path)) == 2
  assert checks == 4
  for rows in range(1, 43):
    _assert_executes(independent, rows)
  assert checks == 4


def test_native_batch_keeps_distinct_constant_specializations(tmp_path):
  operator = _operator(_compiler(tmp_path), tmp_path / "cache")
  calls = [_metadata_call(2), _metadata_call(19, 8), _metadata_call(23)]
  vecops._C.prepare_batch([(operator, call) for call in calls], parallel_jobs=2)
  assert len(_targets(tmp_path)) == 2
  _assert_executes(operator, 2)
  _assert_executes(operator, 19, 8)
  _assert_executes(operator, 23)


@pytest.mark.parametrize("independent_roots", [False, True])
def test_native_batch_adopts_all_providers_and_validates_each_alias(tmp_path, independent_roots):
  compiler = _compiler(tmp_path)
  caches = [tmp_path / "cache", tmp_path / ("other-cache" if independent_roots else "cache")]
  counts = [0, 0]

  def validator(index):
    def validate():
      counts[index] += 1
      return True

    return validate

  operators = [_operator(compiler, caches[index], f"namespace-{index}", validator(index)) for index in range(2)]
  requests = [(operator, _metadata_call(rows)) for operator, rows in zip(operators, (2, 19), strict=True)]
  vecops._C.prepare_batch(requests, parallel_jobs=2)
  assert len(_targets(tmp_path)) == 1
  assert counts == [2, 2]  # Each alias is validated before build and publication.
  manifests = {path for cache in caches for path in cache.rglob("vecops-cache-identity.txt")}
  assert len(manifests) == 2
  for index, rows in enumerate((2, 19)):
    _assert_executes(operators[index], rows)
    # An independent cache-only provider must find the published alias too.
    cached = _operator(None, caches[index], f"namespace-{index}", mode=vecops._C.ArtifactCacheMode.cache_only)
    _assert_executes(cached, rows)
  vecops._C.prepare_batch(requests, parallel_jobs=2)
  assert counts == [2, 2]
  assert len(_targets(tmp_path)) == 1


@pytest.mark.parametrize("failure_phase", ["before", "after"])
def test_native_batch_rejects_changed_alias_and_never_publishes(tmp_path, failure_phase):
  compiler = _compiler(tmp_path)
  counts = [0, 0]

  def valid():
    counts[0] += 1
    return True

  def changed():
    counts[1] += 1
    return failure_phase == "after" and counts[1] == 1

  cache = tmp_path / "cache"
  operators = [_operator(compiler, cache, "first", valid), _operator(compiler, cache, "second", changed)]
  requests = [(operator, _metadata_call(rows)) for operator, rows in zip(operators, (2, 19), strict=True)]
  with pytest.raises(ValueError, match="kernel build inputs changed"):
    vecops._C.prepare_batch(requests, parallel_jobs=2)
  expected = 1 if failure_phase == "before" else 2
  assert counts == [expected, expected]
  assert len(_targets(tmp_path)) == (0 if failure_phase == "before" else 1)
  if failure_phase == "after":
    assert tuple((tmp_path / "build").glob("kernel-batch-*/artifacts/*/*"))
  assert not tuple(cache.rglob("vecops-cache-identity.txt"))


def test_native_batch_still_checks_every_dynamic_relation_before_cache_lookup(tmp_path):
  operator = _operator(_compiler(tmp_path), tmp_path / "cache")
  valid = _metadata_call(2)
  invalid = _metadata_call(19, output_rows=23)
  with pytest.raises(ValueError, match="runtime dimension 'B' is 19.*observed 23"):
    vecops._C.prepare_batch([(operator, valid), (operator, invalid)], parallel_jobs=2)
  assert not _targets(tmp_path)
  vecops._C.prepare_batch([(operator, valid)], parallel_jobs=2)
  with pytest.raises(ValueError, match="runtime dimension 'B' is 19.*observed 23"):
    vecops._C.prepare_batch([(operator, valid), (operator, invalid)], parallel_jobs=2)
  assert len(_targets(tmp_path)) == 1
