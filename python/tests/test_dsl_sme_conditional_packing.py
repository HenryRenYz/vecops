"""Inactive conditional caches may use ZA; active/history paths remain intact.

The Native library oracle checks strict conversion before the dot, an
independent prior snapshot, split-K and aliasing. ARM-only physical pack
checks reuse the independent existing SME pack oracle and runtime geometry.
No native packer or mathematical tolerance is substituted for the fallback.
"""

import re

import numpy as np
import pytest
import vecops
import vecops.language as T
from dsl_test_utils import assert_no_vec_mask_arrays, make_compiler
from test_dsl_sme_transformed_packing import _template
from vecops.language._compiler import ir as I
from vecops.language._compiler.backends.cpu.backend import CPUBackend
from vecops.language._compiler.backends.cpu.cpp.emitter import CPUEmitter
from vecops.language._compiler.ir.traversal import nodes, walk_unique


def _pack_emitter(dtype="bfloat16", side="A", pipeline="affine", *, masked=False):
  artifact = CPUBackend().lower(_template(dtype, side, pipeline, masked=masked).ir, (), target="SME")
  pack = next(value for value in nodes(artifact.kernel.body) if isinstance(value, I.Pack))
  emitter = CPUEmitter(artifact.backend_ir)
  emitter.materialized[id(pack.source)] = (pack.destination.view, "cached_input")
  return emitter, pack


@pytest.mark.parametrize("dtype", ("bfloat16", "float16", "int8", "uint8"))
@pytest.mark.parametrize("side", ("A", "B"))
def test_inactive_materialization_is_selected_only_under_its_original_guard(dtype, side):
  emitter, pack = _pack_emitter(dtype, side)
  original = emitter.materialized
  selected, inactive, guards = emitter.sme_inactive_pack_transform(pack.source, pack.destination)
  assert selected is not None
  assert inactive == (id(pack.source),)
  assert guards == ("(cached_input)",)
  assert emitter.materialized is original
  assert emitter.sme_pack_transform(pack.source, pack.destination) is None


@pytest.mark.parametrize("masked", (False, True))
def test_unknown_unconditional_and_masked_boundaries_do_not_gain_a_route(masked):
  emitter, pack = _pack_emitter(masked=masked)
  for entry in (object(), (pack.destination.view, None), (pack.destination.view, True), (pack.destination.view, "")):
    emitter.materialized[id(pack.source)] = entry
    assert emitter.sme_inactive_pack_transform(pack.source, pack.destination) == (None, (), ())
  emitter.materialized[id(pack.source)] = (pack.destination.view, "cached_input")
  if masked:
    assert emitter.sme_inactive_pack_transform(pack.source, pack.destination) == (None, (), ())


def test_all_dag_cache_and_historical_snapshot_guards_are_checked_without_identity_loss():
  emitter, pack = _pack_emitter(pipeline="params")
  loads = [value for value in walk_unique(pack.source) if value.op == "load"]
  assert len(loads) >= 3
  extra = loads[-1]
  unrelated = I.literal(91)
  emitter.materialized[id(extra)] = (extra.data, "cached_parameter")
  emitter.materialized[id(unrelated)] = (pack.destination.view, "unrelated_cache")
  emitter.snapshots[id(loads[0])] = (loads[0].data, "historical_alias")
  old_materialized, old_snapshots = emitter.materialized, emitter.snapshots
  transform, inactive, guards = emitter.sme_inactive_pack_transform(pack.source, pack.destination)
  assert transform is not None
  assert set(inactive) == {id(pack.source), id(extra)}
  assert set(guards) == {"(cached_input)", "(cached_parameter)", "(historical_alias)"}
  assert emitter.materialized is old_materialized and emitter.snapshots is old_snapshots
  emitter.snapshots[id(extra)] = (extra.data, None)
  assert emitter.sme_inactive_pack_transform(pack.source, pack.destination) == (None, (), ())


@pytest.mark.parametrize("fail", (False, True))
@pytest.mark.parametrize("scope", (None, ("sme", "existing_scope")))
def test_fast_emission_restores_guarded_cache_identity_on_success_and_exception(fail, scope):
  emitter, pack = _pack_emitter()
  emitter.hardware_scope = scope
  load = next(value for value in walk_unique(pack.source) if value.op == "load")
  emitter.snapshots[id(load)] = (load.data, "historical_alias")
  emitter.chunk_cache[("previous",)] = "prior_cpp_value"
  original = (emitter.materialized, emitter.snapshots, emitter.chunk_cache, emitter.hardware_scope)
  body = emitter.sme_transform_pack_body

  def checked(*args):
    assert id(pack.source) not in emitter.materialized
    assert id(load) in emitter.snapshots
    assert not emitter.chunk_cache
    if fail:
      raise RuntimeError("test emission failure")
    return body(*args)

  emitter.sme_transform_pack_body = checked
  spatial, k = (emitter.scalar(extent) for extent in pack.destination.shape)
  if fail:
    with pytest.raises(RuntimeError, match="test emission failure"):
      emitter.sme_raw_pack(pack.source, pack.destination, spatial, k, 2)
  else:
    assert emitter.sme_raw_pack(pack.source, pack.destination, spatial, k, 2)
    emitter.close_sme_raw_pack()
    source = "\n".join(emitter.lines)
    assert "&& !((cached_input) || (historical_alias))" in source
    assert "VECOPS_INLINE_LAMBDA_NOEXCEPT" in source
    assert "do not evaluate padded K DAG" in source
  assert emitter.materialized is original[0]
  assert emitter.snapshots is original[1]
  assert emitter.chunk_cache is original[2]
  assert emitter.hardware_scope is original[3]


def library_template(dtype="bfloat16", *, captured=False):
  M, N, K = (T.dynamic(lower=0) for _ in range(3))
  AS, AK, BKs, BS, CS, CN, DS, DN = (T.dynamic() for _ in range(8))

  @T.template
  def kernel(
    A: T.Tensor((M, K), (AS, AK), "float32"),
    B: T.Tensor((K, N), (BKs, BS), "float32"),
    C: T.Tensor((M, N), (CS, CN), "float32", access="output"),
    D: T.Tensor((M, N), (DS, DN), "float32"),
  ):
    a = T.load(A)
    if captured:
      T.store(C, T.broadcast_to(T.expand_dims(T.reduce_add(a, axis=1), 1), (M, N)))
    left = T.cast(a * 0.5 + 0.125, dtype)
    right = T.cast(T.load(B), dtype)
    result = T.library.matmul((left, right), C, acc=T.load(D), BM=16, BN=4, BK=3, packing="AB", swap=False)
    T.store(C, T.max(result * 0.5, 0.0))

  return kernel


def test_dynamic_library_guards_fast_pack_once_and_keeps_one_original_fallback():
  artifact = CPUBackend().lower(library_template().ir, (), target="SME")
  source = artifact.source
  assert_no_vec_mask_arrays(source)
  guards = re.findall(r"const bool (\w+input_cache\w*)=", source)
  assert len(guards) == 2
  fast = [line for line in source.splitlines() if "FP32 DAG K-contiguous SME ZA packing" in line]
  assert fast and any(guards[0] in line for line in fast)
  assert any(guards[1] in line for line in fast)
  assert "non-contiguous K preserves vector gather packing" in source
  # The old producer-cache branch remains available to every ordinary
  # fallback load, not consumed or erased by the new fast callback emission.
  assert source.count(f"if ({guards[0]})") > 2
  assert source.count(f"if ({guards[1]})") > 2
  assert (
    "vecops::vec::fmadd("
    not in source.split("FP32 DAG K-contiguous SME ZA packing", 1)[1].split(
      "non-contiguous K preserves vector gather packing", 1
    )[0]
  )


def test_historical_write_capture_and_empty_regions_keep_their_original_guards():
  source = CPUBackend().lower(library_template(captured=True).ir, (), target="SME").source
  guards = [line for line in source.splitlines() if "FP32 DAG K-contiguous SME ZA packing" in line]
  assert guards
  first_cache = source.index("/* input fusion:")
  aliases = re.findall(r"const bool (\w+alias_\d+)\s*=", source[:first_cache])
  assert aliases and any(any(alias in guard for alias in aliases) for guard in guards)
  # Optional DSL tensors are rejected by this frontend. The new eligibility
  # must not inject an optional accessor, nor evaluate scratch.data before
  # its materialization guard; input bases here are immutable ABI pointers.
  assert ".value()" not in source
  assert "empty products need no resources" in source
  for line in source.splitlines():
    if "sme_pack_input" in line and "const auto* const" in line:
      assert "dot_input" not in line


@pytest.mark.parametrize("fusion", ("auto", "never"))
@pytest.mark.parametrize("dtype", ("bfloat16", "float16"))
def test_native_conditional_pack_library_preserves_conversion_prior_and_aliases(tmp_path, fusion, dtype):
  ml_dtypes = pytest.importorskip("ml_dtypes")
  compiler = make_compiler(target="Native", jobs=4, cache_dir=tmp_path / "cache", build_dir=tmp_path / "build")
  compiled = T.compile(library_template(dtype), compiler=compiler, pass_options={"cpu_matmul_fusion": fusion})
  specs = []
  for m, n, k in ((3, 8, 5), (17, 80, 9), (0, 8, 5), (3, 8, 0)):
    specs.append(
      (
        vecops.TensorMeta((m, k), strides=(max(k, 1), 1), dtype="float32"),
        vecops.TensorMeta((k, n), strides=(1, max(k, 1)), dtype="float32"),
        vecops.TensorMeta((m, n), strides=(max(n, 1), 1), dtype="float32"),
        vecops.TensorMeta((m, n), strides=(max(n, 1), 1), dtype="float32"),
      )
    )
  with vecops.collect_compile_requests() as collector:
    for spec in specs:
      compiled.operator(*spec)
  assert len(collector.requests) == 4
  vecops.compile_batch(collector.requests, parallelism=compiler.jobs, execution_parallelism=1)
  native = ml_dtypes.bfloat16 if dtype == "bfloat16" else np.float16
  rng = np.random.default_rng(6207)
  calls = 0
  for m, n, k in ((3, 8, 5), (17, 80, 9), (3, 8, 0)):
    for layout in ("dense", "reverse", "k_stride", "alias_input", "alias_prior"):
      size = max(m * max(k, 1), m * n, 1)
      storage = rng.integers(-8, 9, size * 2).astype(np.float32) / np.float32(8)
      a = np.ndarray((m, k), np.float32, buffer=storage, strides=(max(k, 1) * 4, 4))
      b_backing = rng.integers(-8, 9, max(n * max(k, 1), 1)).astype(np.float32) / np.float32(8)
      b = np.ndarray((k, n), np.float32, buffer=b_backing, strides=(4, max(k, 1) * 4))
      c = np.full((m, n), np.nan, np.float32)
      prior = rng.integers(-4, 5, (m, n)).astype(np.float32) / np.float32(16)
      if layout == "reverse":
        a, b, c, prior = a[::-1], b[:, ::-1], c[:, ::-1], prior[::-1, ::-1]
      elif layout == "k_stride":
        expanded = np.zeros((m, max(k, 1) * 2), np.float32)
        expanded[:, : k * 2 : 2] = a
        a = expanded[:, : k * 2 : 2]
      elif layout == "alias_input":
        c = np.ndarray((m, n), np.float32, buffer=storage, strides=(n * 4, 4))
      elif layout == "alias_prior":
        c[...] = prior
        prior = c
      before_a, before_b, before_prior = a.copy(), b.copy(), prior.copy()
      converted_a = (before_a * np.float32(0.5) + np.float32(0.125)).astype(native).astype(np.float32)
      converted_b = before_b.astype(native).astype(np.float32)
      expected = np.maximum((converted_a @ converted_b + before_prior) * np.float32(0.5), np.float32(0))
      compiled.operator.native(a, b, c, prior, **compiled._core.specialization_values)
      # Dyadic fixture values keep multiply/add and split-K exact; no
      # approximate math or relaxed reduction tolerance can hide an alias bug.
      np.testing.assert_array_equal(c, expected)
      calls += 1
  assert calls == 15


@pytest.mark.parametrize("dtype", ("bfloat16", "float16"))
def test_native_history_survives_an_intervening_alias_write_before_packing(tmp_path, dtype):
  ml_dtypes = pytest.importorskip("ml_dtypes")
  compiler = make_compiler(target="Native", jobs=4, cache_dir=tmp_path / "cache", build_dir=tmp_path / "build")
  compiled = T.compile(library_template(dtype, captured=True), compiler=compiler)
  with vecops.collect_compile_requests() as collector:
    compiled.operator(
      vecops.TensorMeta((3, 5), dtype="float32"),
      vecops.TensorMeta((5, 8), strides=(1, 5), dtype="float32"),
      vecops.TensorMeta((3, 8), dtype="float32"),
      vecops.TensorMeta((3, 8), dtype="float32"),
    )
  assert len(collector.requests) == 1
  vecops.compile_batch(collector.requests, parallelism=compiler.jobs, execution_parallelism=1)
  narrow = ml_dtypes.bfloat16 if dtype == "bfloat16" else np.float16
  for m, n, k in ((3, 8, 5), (17, 80, 9)):
    for aliases in (False, True):
      storage = np.arange(max(m * k, m * n), dtype=np.float32) % 9 / np.float32(8)
      a = np.ndarray((m, k), np.float32, buffer=storage, strides=(k * 4, 4))
      b = (np.arange(n * k, dtype=np.float32).reshape(n, k) % 7 / np.float32(8)).T
      c = (
        np.ndarray((m, n), np.float32, buffer=storage, strides=(n * 4, 4)) if aliases else np.empty((m, n), np.float32)
      )
      prior = np.full((m, n), np.float32(0.25))
      original = a.copy()
      left = (original * np.float32(0.5) + np.float32(0.125)).astype(narrow).astype(np.float32)
      right = b.astype(narrow).astype(np.float32)
      expected = np.maximum((left @ right + prior.copy()) * np.float32(0.5), np.float32(0))
      compiled.operator.native(a, b, c, prior, **compiled._core.specialization_values)
      np.testing.assert_array_equal(c, expected)
