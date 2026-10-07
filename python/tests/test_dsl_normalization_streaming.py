"""Automatic row-statistic/epilogue fusion keeps layout and alias boundaries."""

import numpy as np
import pytest
import vecops
import vecops.language as T
from dsl_test_utils import assert_no_vec_mask_arrays, make_compiler
from vecops.language._compiler import ir as I
from vecops.language._compiler.backends.cpu.backend import CPUBackend
from vecops.language._compiler.ir.traversal import nodes
from vecops.language._compiler.passes.manager import registered_passes

KINDS = ("centered", "moments", "strict", "fast", "estimate", "rmsnorm")


def streaming_template(kind="moments", *, affine=False, boundary="none"):
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
    if kind == "centered" or kind == "moments":
      y = T.library.layernorm(x, variance=kind)
    elif kind == "rmsnorm":
      y = T.library.rmsnorm(x)
    else:
      y = T.library.softmax(x, accuracy=kind)
    if affine:
      y = T.fma(y, T.load(G), T.load(B))
    if boundary == "multiple":
      T.store(D, y)
    elif boundary == "write":
      T.store(A, 42)
    elif boundary == "transpose":
      y = T.transpose(y)
    T.store(C, y)

  return kernel


def streaming_guard(kernel):
  return next(
    (
      node
      for node in nodes(kernel.body)
      if isinstance(node, I.If)
      and any(isinstance(child, I.For) and child.target.startswith("dsl_o_map_stream_") for child in nodes(node.body))
    ),
    None,
  )


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("affine", (False, True))
def test_streamed_row_statistics_and_epilogue_share_a_guarded_ir_region(kind, affine):
  lowered = CPUBackend().lower(streaming_template(kind, affine=affine).ir, (), target="SVE")
  guard = streaming_guard(lowered.kernel)
  assert guard is not None
  assert not any(isinstance(node, I.Allocate) for node in nodes(guard.body))
  assert any(isinstance(node, I.Allocate) for node in nodes(guard.otherwise))
  proofs = [node for node in nodes(guard.condition) if isinstance(node, I.Expr) and node.op == "view_disjoint"]
  assert {proof.data[0].storage for proof in proofs} == ({"A", "G", "B"} if affine else {"A"})
  assert all(len(proof.data[1].shape) == 2 for proof in proofs)
  assert next(node for node in nodes(guard.condition) if isinstance(node, I.Expr) and node.op == "view_nonoverlap")
  assert_no_vec_mask_arrays(lowered.source)
  assert "vecops::ops::" not in lowered.source


@pytest.mark.parametrize("boundary", ("multiple", "write", "transpose"))
def test_shared_effectful_or_layout_changing_consumers_keep_original_boundary(boundary):
  # Square shape makes the transpose store legal while retaining its layout
  # transform boundary; a shape error must not masquerade as a fusion test.
  template = streaming_template(boundary=boundary)
  if boundary == "transpose":
    from dataclasses import replace

    annotations = dict(template.ir.annotations)
    annotations["C"] = replace(annotations["C"], shape=annotations["A"].shape[::-1])
    ir = replace(template.ir, annotations=annotations)
  else:
    ir = template.ir
  lowered = CPUBackend().lower(ir, (), target="SVE")
  assert streaming_guard(lowered.kernel) is None
  assert any(isinstance(node, I.Allocate) and "map" in node.view.storage for node in nodes(lowered.kernel.body))


def test_streaming_pass_is_registered_and_independently_disabled():
  lowered = CPUBackend().lower(
    streaming_template().ir, (), target="SVE", options={"row_normalization_streaming": False}
  )
  assert streaming_guard(lowered.kernel) is None
  assert "stream-row-maps" in {spec.name for spec in registered_passes()}
  assert (
    lowered.kernel.pass_trace.index("normalize-reduction-layout")
    < lowered.kernel.pass_trace.index("stream-row-maps")
    < lowered.kernel.pass_trace.index("decompose-row-maps")
  )


@pytest.mark.parametrize("packed", (False, True))
def test_normalization_to_dot_and_pack_remains_a_materialized_statistics_boundary(packed):
  @T.template
  def dot_kernel(
    A: T.Tensor((8, 128), dtype=T.float32),
    B: T.Tensor((128, 16), dtype=T.float32),
    C: T.Tensor((8, 16), dtype=T.float32, access="output"),
  ):
    y = T.library.layernorm(T.load(A), variance="moments")
    if packed:
      y = T.pack(y, side="A", format="dense")
    T.store(C, T.dot(y, T.load(B)))

  lowered = CPUBackend().lower(dot_kernel.ir, (), target="SVE")
  assert streaming_guard(lowered.kernel) is None
  assert any(isinstance(node, I.Allocate) and "map" in node.view.storage for node in nodes(lowered.kernel.body))


@pytest.fixture(scope="module")
def native_streaming_kernels(tmp_path_factory):
  path = tmp_path_factory.mktemp("row-normalization-streaming")
  compiler = make_compiler(jobs=4, cache_dir=path / "cache", build_dir=path / "build")
  kernels = {}
  with vecops.collect_compile_requests() as collector:
    for kind in KINDS:
      for affine in (False, True):
        compiled = T.compile(
          streaming_template(kind, affine=affine), compiler=compiler, debug=T.DebugOptions(memory=True)
        )
        operands = (
          vecops.TensorMeta((3, 37), dtype="float32"),
          vecops.TensorMeta((37,), dtype="float32"),
          vecops.TensorMeta((37,), dtype="float32"),
          vecops.TensorMeta((3, 37), dtype="float32"),
          vecops.TensorMeta((3, 37), dtype="float32"),
        )
        compiled.operator(*operands, **compiled._core.specialization_values)
        kernels[kind, affine] = compiled
  vecops.compile_batch(collector.requests, parallelism=4, execution_parallelism=1)
  return kernels


@T.func
def row_standardize_range(x, *, scale=1, unroll=2):
  low = T.reduce_min(x, unroll=unroll)
  high = T.reduce_max(x, unroll=unroll)
  extent = T.max(high - low, T.cast(1e-5, x.dtype))
  return (x - low) / extent * scale


@pytest.mark.parametrize("streaming", (False, True))
def test_user_row_program_uses_the_same_generic_region_scheduler(streaming):
  @T.template
  def kernel(A: T.Tensor((3, 37), dtype="float32"), C: T.Tensor((3, 37), dtype="float32", access="output")):
    T.store(C, T.map_rows(row_standardize_range, T.load(A), scale=2))

  lowered = CPUBackend().lower(kernel.ir, (), target="SVE", options={"row_map_streaming": streaming})
  assert (streaming_guard(lowered.kernel) is not None) is streaming
  assert not any(
    node.op in {"map_rows", "region_argument"} for node in nodes(lowered.kernel.body) if isinstance(node, I.Expr)
  )
  assert "vecops::vec::reduce_min" in lowered.source
  assert "vecops::vec::reduce_max" in lowered.source
  assert_no_vec_mask_arrays(lowered.source)


@T.func
def effectful_row(x):
  T.print("not a pure row program")
  return x


def test_effectful_row_program_is_rejected_at_its_source_site():
  @T.template
  def kernel(A: T.Tensor((3, 37), dtype="float32"), C: T.Tensor((3, 37), dtype="float32", access="output")):
    T.store(C, T.map_rows(effectful_row, T.load(A)))

  with pytest.raises(TypeError, match="pure straight-line"):
    CPUBackend().lower(kernel.ir, (), target="Native")


def reference(inputs, kind):
  x = inputs.astype(np.float64)
  if kind in {"centered", "moments"}:
    centered = x - x.mean(axis=-1, keepdims=True)
    return centered / np.sqrt((centered * centered).mean(axis=-1, keepdims=True) + 1e-5)
  if kind == "rmsnorm":
    return x / np.sqrt((x * x).mean(axis=-1, keepdims=True) + 1e-5)
  exponential = np.exp(x - x.max(axis=-1, keepdims=True))
  return exponential / exponential.sum(axis=-1, keepdims=True)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("affine", (False, True))
def test_native_streaming_fast_path_and_full_alias_fallback(native_streaming_kernels, kind, affine):
  compiled = native_streaming_kernels[kind, affine]
  rng = np.random.default_rng(91)
  for width in (1, 7, 17, 65, 257):
    for alias in ("none", "inplace", "shifted", "negative", "gamma", "beta"):
      storage = rng.normal(size=3 * width + 1).astype(np.float32)
      inputs = storage[:-1].reshape(3, width)
      output = np.empty_like(inputs)
      gamma, beta = rng.normal(size=(2, width)).astype(np.float32)
      if alias == "inplace":
        output = inputs
      elif alias == "shifted":
        output = storage[1:].reshape(3, width)
      elif alias == "negative":
        output = inputs[:, ::-1]
      elif alias == "gamma":
        gamma = output[0]
        gamma[:] = rng.normal(size=width)
      elif alias == "beta":
        beta = output[0]
        beta[:] = rng.normal(size=width)
      expected = reference(inputs, kind)
      if affine:
        expected = expected * gamma.astype(np.float64) + beta.astype(np.float64)
      compiled(inputs, gamma, beta, output, np.empty_like(inputs))
      tolerance = 3e-3 if kind == "estimate" else 3e-5
      np.testing.assert_allclose(
        output, expected, rtol=tolerance, atol=tolerance, err_msg=f"{kind}/{affine}/{width}/{alias}"
      )
  # Row-local writable checks alone would miss this inter-row overlap.
  inputs = rng.normal(size=(3, 37)).astype(np.float32)
  output = np.lib.stride_tricks.as_strided(np.zeros(37, np.float32), shape=(3, 37), strides=(0, 4))
  with pytest.raises((ValueError, RuntimeError), match="overlap|writable"):
    compiled(inputs, np.ones(37, np.float32), np.zeros(37, np.float32), output, np.empty_like(inputs))
  # The common non-alias moments/LN/RMSNorm path has no statistics allocation.
  # Softmax may legitimately cache its expensive per-element exponential.
  report = compiled.profile(
    inputs,
    np.ones(37, np.float32),
    np.zeros(37, np.float32),
    np.empty_like(inputs),
    np.empty_like(inputs),
    repeat=1,
    warmup=0,
    memory=True,
  )
  assert not any("map_" in event.get("site", "") and event.get("action", 0) == 0 for event in report.memory_events)
