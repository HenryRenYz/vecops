"""Private lazy read captures are dominated by normalization's alias guard."""

import re

import numpy as np
import pytest
import vecops
import vecops.language as T
from dsl_test_utils import assert_no_vec_mask_arrays, make_compiler
from test_dsl_normalization_streaming import reference, streaming_guard
from vecops.language._compiler import ir as I
from vecops.language._compiler.backends.cpu.backend import CPUBackend
from vecops.language._compiler.ir.traversal import nodes


def capture_template(kind="moments", boundary="sole"):
  M, N = T.dynamic(lower=0), T.dynamic(lower=0)
  AS0, AS1, CS0, CS1 = (T.dynamic() for _ in range(4))

  @T.template
  def kernel(
    A: T.Tensor((M, N), (AS0, AS1), T.float32, access="inout"),
    G: T.Tensor((N,), dtype=T.float32),
    B: T.Tensor((N,), dtype=T.float32),
    C: T.Tensor((M, N), (CS0, CS1), T.float32, access="output"),
    D: T.Tensor((M, N), dtype=T.float32, access="output"),
  ):
    x = T.load(A)
    if boundary == "write_before":
      T.store(A, 42)
    if boundary == "scalar":
      factor = T.cast(1, T.float32)
      x = x * factor
    if kind == "moments":
      y = T.library.layernorm(x, variance="moments")
    else:
      y = T.library.softmax(x, accuracy="strict")
    y = T.fma(y, T.load(G), T.load(B))
    T.store(C, y)
    if boundary == "other_after":
      T.store(A, 42)
      T.store(D, x + 5)
    elif boundary == "external":
      T.store(A, 42)
      for i in T.serial(1, unroll=1):
        T.store(D, x + 5)

  return kernel


def source_block(source, opening):
  """Extract a generated scope, ignoring initializer braces on its header."""
  start = source.rfind("{", opening, source.index("\n", opening))
  depth = 1
  for stop in range(start + 1, len(source)):
    depth += (source[stop] == "{") - (source[stop] == "}")
    if depth == 0:
      return source[start + 1 : stop], stop
  raise AssertionError("unterminated generated C++ scope")


def streamed_sources(source):
  row = source.index("for (vecops::nint_t dsl_o_map_stream_")
  fast, stop = source_block(source, row)
  fallback = source.index("else", stop)
  slow, _ = source_block(source, fallback)
  return fast, slow


@pytest.mark.parametrize("kind", ("moments", "strict"))
def test_private_read_captures_do_not_select_snapshots_in_disjoint_rows(kind):
  template = capture_template(kind)
  lowered = CPUBackend().lower(template.ir, (), target="SVE")
  guard = streaming_guard(lowered.kernel)
  assert guard is not None
  assert not any(isinstance(node, I.BindTile) for node in lowered.kernel.body.statements)
  fast, fallback = streamed_sources(lowered.source)
  assert "dsl_c_alias_" not in fast
  assert "dsl_c_snapshot_" not in fast
  assert "dsl_c_alias_" in fallback
  assert "dsl_c_snapshot_" in fallback
  assert "vecops::vec::load_convert" in fast
  assert "vecops::vec::store_convert" in fast
  assert_no_vec_mask_arrays(lowered.source)
  old = CPUBackend().lower(template.ir, (), target="SVE", options={"row_normalization_release_captures": False})
  old_fast, _ = streamed_sources(old.source)
  assert "dsl_c_alias_" in old_fast
  assert "dsl_c_snapshot_" in old_fast


@pytest.mark.parametrize("boundary", ("write_before", "scalar", "other_after", "external"))
def test_effects_or_other_read_consumers_keep_capture_protection(boundary):
  lowered = CPUBackend().lower(capture_template(boundary=boundary).ir, (), target="SVE")
  assert any(isinstance(node, I.BindTile) for node in lowered.kernel.body.statements)
  guard = streaming_guard(lowered.kernel)
  if boundary in {"write_before", "scalar"}:
    assert guard is None
  else:
    assert guard is not None
    fast, _ = streamed_sources(lowered.source)
    assert "dsl_c_alias_" in fast
    assert "dsl_c_snapshot_" in fast
  assert "dsl_c_alias_" in lowered.source
  assert "dsl_c_snapshot_" in lowered.source
  assert_no_vec_mask_arrays(lowered.source)


@pytest.mark.parametrize("kind", ("moments", "strict"))
@pytest.mark.parametrize("target", (pytest.param("Native", id="local"), pytest.param("SVE", id="arm")))
def test_statistics_and_epilogue_read_the_capture_saved_before_a_write(kind, target):
  lowered = CPUBackend().lower(capture_template(kind, "write_before").ir, (), target=target)
  assert streaming_guard(lowered.kernel) is None
  source = lowered.source
  snapshot = re.search(r"Buffer<float> (dsl_c_snapshot_\d+);", source).group(1)
  alias = re.search(r"const bool (dsl_c_alias_\d+) =", source).group(1)
  row = source.index("for (vecops::nint_t dsl_o_map_1_row")
  statistics, stop = source_block(source, row)
  loads = [line for line in statistics.splitlines() if "load_convert" in line]
  assert loads and all(snapshot + ".data()" in load and alias + " ?" in load for load in loads)
  # The epilogue may renew its alias snapshot before writing C. Its renewal
  # must read the already-preserved value, not the overwritten ABI tensor.
  renewal = source.index("const bool dsl_c_alias_", stop)
  renewed_snapshot = re.search(r"Buffer<float> (dsl_c_snapshot_\d+);", source[renewal:]).group(1)
  renewal_body, _ = source_block(source, source.index("if (dsl_c_alias_", renewal))
  assert snapshot + ".data()" in renewal_body and alias + " ?" in renewal_body
  epilogue = source[source.rindex("vecops_dsl::check_writable<2>") :]
  assert renewed_snapshot + ".data()" in epilogue
  assert_no_vec_mask_arrays(source)


@pytest.mark.parametrize("kind", ("moments", "strict"))
def test_effect_free_materialized_statistics_keep_direct_sliced_loads(kind):
  lowered = CPUBackend().lower(
    capture_template(kind).ir, (), target="SVE", options={"row_normalization_streaming": False}
  )
  transforms = [
    node for node in nodes(lowered.kernel.body) if isinstance(node, I.Expr) and node.op == "value_transform"
  ]
  assert not transforms
  assert_no_vec_mask_arrays(lowered.source)


@pytest.mark.parametrize("operation", ("dot", "atomic_add", "state_ref", "team_reduce", "unregistered_capture"))
def test_unused_boundary_capture_backed_by_private_reads_is_not_discarded(operation):
  from types import SimpleNamespace

  from vecops.language._compiler.environment import freeze
  from vecops.language._compiler.frontend.parser import lower
  from vecops.language._compiler.passes.kernel.regions import stream_row_maps

  frontend = lower(capture_template().ir)
  source = next(node for node in nodes(frontend.body) if isinstance(node, I.Expr) and node.op == "load")
  captured = I.Expr(source.type, operation, (source,))
  statement = I.BindTile(captured)
  frontend.body.statements.insert(1, statement)
  context = SimpleNamespace(options={}, environment=freeze(target="SVE"))
  rewritten = stream_row_maps(frontend, context)
  retained = [node for node in rewritten.body.statements if isinstance(node, I.BindTile)]
  assert any(node.value is captured for node in retained)
  # Never turn private-load membership into a license to erase an unrelated
  # state/matrix/collective boundary, including unregistered plugin operations.
  assert streaming_guard(rewritten) is not None


def test_row_substitution_preserves_shared_affine_dag_edges():
  from dataclasses import replace
  from types import SimpleNamespace

  from vecops.language._compiler.environment import freeze
  from vecops.language._compiler.frontend.parser import lower
  from vecops.language._compiler.passes.kernel.regions import stream_row_maps

  frontend = lower(capture_template().ir)
  sink = frontend.body.statements[-1]
  value = sink.value
  for _ in range(18):
    value = I.binary("add", value, value)
  frontend.body.statements[-1] = replace(sink, value=value)
  context = SimpleNamespace(options={}, environment=freeze(target="SVE"))
  rewritten = stream_row_maps(frontend, context)
  guard = streaming_guard(rewritten)
  assert guard is not None
  value = next(node.value for node in nodes(guard.body) if isinstance(node, I.Store))
  for _ in range(18):
    assert value.op == "add" and value.args[0] is value.args[1]
    value = value.args[0]


@pytest.fixture(scope="module")
def native_capture_kernels(tmp_path_factory):
  root = tmp_path_factory.mktemp("normalization-capture-release")
  compiler = make_compiler(jobs=4, cache_dir=root / "cache", build_dir=root / "build")
  kernels = {}
  with vecops.collect_compile_requests() as collector:
    for kind in ("moments", "strict"):
      for boundary in ("sole", "write_before", "other_after"):
        compiled = T.compile(capture_template(kind, boundary), compiler=compiler)
        metadata = (
          vecops.TensorMeta((3, 65), dtype="float32"),
          vecops.TensorMeta((65,), dtype="float32"),
          vecops.TensorMeta((65,), dtype="float32"),
          vecops.TensorMeta((3, 65), dtype="float32"),
          vecops.TensorMeta((3, 65), dtype="float32"),
        )
        compiled.operator(*metadata, **compiled._core.specialization_values)
        kernels[kind, boundary] = compiled
  vecops.compile_batch(collector.requests, parallelism=4, execution_parallelism=1)
  return kernels


@pytest.mark.parametrize("kind", ("moments", "strict"))
@pytest.mark.parametrize("boundary", ("sole", "write_before", "other_after"))
def test_native_private_and_retained_capture_lifetimes(native_capture_kernels, kind, boundary):
  compiled = native_capture_kernels[kind, boundary]
  rng = np.random.default_rng(693)
  for width in (0, 1, 17, 65, 128):
    for alias in ("none", "inplace", "shifted", "gamma", "beta"):
      storage = rng.normal(size=3 * width + 1).astype(np.float32)
      inputs = storage[:-1].reshape(3, width)
      output = np.zeros_like(inputs)
      gamma, beta = rng.normal(size=(2, width)).astype(np.float32)
      if alias == "inplace":
        output = inputs
      elif alias == "shifted":
        output = storage[1:].reshape(3, width)
      elif alias == "gamma":
        gamma = output[0]
        gamma[:] = rng.normal(size=width)
      elif alias == "beta":
        beta = output[0]
        beta[:] = rng.normal(size=width)
      before = inputs.copy()
      normalized = reference(before, kind) if width else np.empty_like(before)
      result = normalized * gamma.astype(np.float64) + beta.astype(np.float64)
      expected_storage = storage.copy()
      expected_inputs = expected_storage[:-1].reshape(3, width)
      expected_output = np.empty_like(output)
      if alias == "inplace":
        expected_output = expected_inputs
      elif alias == "shifted":
        expected_output = expected_storage[1:].reshape(3, width)
      if boundary == "write_before":
        expected_inputs[:] = 42
      expected_output[:] = result
      if boundary == "other_after":
        expected_inputs[:] = 42
      other = np.full_like(before, -73)
      compiled(inputs, gamma, beta, output, other)
      label = f"{kind}/{boundary}/{width}/{alias}"
      np.testing.assert_allclose(output, expected_output, rtol=3e-5, atol=3e-5, err_msg=label)
      np.testing.assert_allclose(inputs, expected_inputs, rtol=3e-5, atol=3e-5, err_msg=label)
      if boundary == "other_after":
        np.testing.assert_array_equal(other, before + 5)
      else:
        np.testing.assert_array_equal(other, -73)
