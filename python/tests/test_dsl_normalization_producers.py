# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Same-row normalization helper consumers share one logical producer."""

import numpy as np
import pytest
import vecops
import vecops.language as T
from dsl_test_utils import assert_no_vec_mask_arrays, make_compiler
from test_dsl_normalization_streaming import streaming_guard
from vecops.language._compiler import ir as I
from vecops.language._compiler.backends.cpu.backend import CPUBackend
from vecops.language._compiler.ir.traversal import nodes
from vecops.language._library.math import masked_softmax_row


def producer_template(dtype="float32", *, masked=False, accuracy="estimate", boundary="none"):
  M, N = T.dynamic(lower=0), T.dynamic(lower=0)
  AS0, AS1, MS0, MS1, CS0, CS1 = (T.dynamic() for _ in range(6))
  DType = T.DType(dtype)
  Compute = T.float64 if dtype == "float64" else T.float32
  NegativeInfinity = float("-inf")

  @T.template
  def kernel(
    A: T.Tensor((M, N), (AS0, AS1), DType, access="inout"),
    Mask: T.Tensor((M, N), (MS0, MS1), T.float32),
    C: T.Tensor((M, N), (CS0, CS1), DType, access="output"),
    D: T.Tensor((M, N), dtype=DType, access="output"),
  ):
    x = T.load(A, dtype=Compute)
    if boundary == "helper":
      y = T.library.masked_softmax(x, T.load(Mask), accuracy=accuracy)
    elif masked:
      scores = T.where(T.load(Mask), x, T.cast(NegativeInfinity, Compute))
      y = T.map_rows(masked_softmax_row, scores, accuracy=accuracy)
    else:
      y = T.library.softmax(x, accuracy=accuracy)
    if boundary == "multiple":
      T.store(D, y)
    elif boundary == "write":
      T.store(A, 42)
    elif boundary == "transpose":
      y = T.transpose(y)
    T.store(C, y)

  return kernel


def expressions(value):
  return [node for node in nodes(value) if isinstance(node, I.Expr)]


@pytest.mark.parametrize("dtype", ("float16", "float32", "float64"))
@pytest.mark.parametrize("masked", (False, True))
@pytest.mark.parametrize("accuracy", ("strict", "fast", "estimate"))
def test_same_row_producer_identity_is_visible_to_generic_reuse(dtype, masked, accuracy):
  lowered = CPUBackend().lower(producer_template(dtype, masked=masked, accuracy=accuracy).ir, (), target="SVE")
  guard = streaming_guard(lowered.kernel)
  assert guard is not None
  reductions = [node for node in expressions(guard.body) if node.op == "reduce_add"]
  assert len(reductions) == 1
  producer = reductions[0].args[0]
  stores = [node for node in nodes(guard.body) if isinstance(node, I.Store)]
  assert len(stores) == 1
  assert any(node is producer for node in expressions(stores[0].value))
  cache = lowered.kernel.materializations[id(producer)]
  assert cache.dtype == ("float64" if dtype == "float64" else "float32")
  assert len(cache.shape) == 1
  assert "norm_stream_1_tail" not in lowered.source
  assert_no_vec_mask_arrays(lowered.source)


@pytest.mark.parametrize("boundary", ("multiple", "write", "transpose"))
def test_nonstreaming_boundary_does_not_export_a_row_local_producer(boundary):
  template = producer_template(masked=True, boundary=boundary)
  if boundary == "transpose":
    from dataclasses import replace

    annotations = dict(template.ir.annotations)
    annotations["C"] = replace(annotations["C"], shape=annotations["A"].shape[::-1])
    kernel = replace(template.ir, annotations=annotations)
  else:
    kernel = template.ir
  lowered = CPUBackend().lower(kernel, (), target="SVE")
  assert streaming_guard(lowered.kernel) is None
  rows = [node for node in nodes(lowered.kernel.body) if isinstance(node, I.For) and "map" in node.target]
  assert rows
  producers = {id(node.args[0]) for row in rows for node in expressions(row.body) if node.op == "reduce_add"}
  sinks = [
    node for node in nodes(lowered.kernel.body) if isinstance(node, I.Store) and node.destination.storage in {"C", "D"}
  ]
  assert {sink.destination.storage for sink in sinks} == ({"C", "D"} if boundary == "multiple" else {"C"})
  assert all(id(node) not in producers for sink in sinks for node in expressions(sink.value))
  assert all(any(node.op in {"exp_neg", "fma"} for node in expressions(sink.value)) for sink in sinks)
  assert_no_vec_mask_arrays(lowered.source)


def test_public_helper_scalar_capture_boundary_keeps_existing_nonstreaming_schedule():
  from vecops.language._compiler.frontend.parser import lower

  template = producer_template(masked=True, boundary="helper")
  frontend = lower(template.ir)
  assignments = [node for node in frontend.body.statements if isinstance(node, I.Assign)]
  negative = next(
    node
    for node in assignments
    if node.value.op == "cast" and node.value.args[0].op == "literal" and node.value.args[0].data == float("-inf")
  )
  captured = next(node for node in assignments if node.value.op == "local" and node.value.data == negative.target)
  normalization = next(node for node in expressions(frontend.body) if node.op == "map_rows")
  assert any(node.op == "frozen" and node.data == captured.target for node in expressions(normalization.args[0]))
  lowered = CPUBackend().lower(template.ir, (), target="SVE")
  # The helper captures a scalar after its lazy input bindings. Keep the
  # existing conservative barrier rather than move producers across it.
  assert streaming_guard(lowered.kernel) is None
  assert any(isinstance(node, I.Allocate) and "map" in node.view.storage for node in nodes(lowered.kernel.body))


def test_shared_identity_does_not_override_the_generic_cost_policy():
  from dataclasses import replace

  from vecops.language._compiler.cost import CostProfile
  from vecops.language._compiler.environment import freeze

  environment = freeze(target="SVE")
  profile = CostProfile("test-no-normalization-cache", policies=(("reuse_compute_budget", 1e9),))
  environment = replace(environment, costs=replace(environment.costs, profiles=(*environment.costs.profiles, profile)))
  lowered = CPUBackend().lower(
    producer_template(accuracy="strict").ir,
    (),
    target="SVE",
    options={"_compiler_environment": environment},
  )
  guard = streaming_guard(lowered.kernel)
  producer = next(node.args[0] for node in expressions(guard.body) if node.op == "reduce_add")
  sink = next(node for node in nodes(guard.body) if isinstance(node, I.Store))
  assert any(node is producer for node in expressions(sink.value))
  assert not lowered.kernel.materializations


@pytest.fixture(scope="module")
def native_shared_producers(tmp_path_factory):
  path = tmp_path_factory.mktemp("normalization-producers")
  compiler = make_compiler(jobs=4, cache_dir=path / "cache", build_dir=path / "build")
  kernels = {}
  with vecops.collect_compile_requests() as collector:
    for dtype in ("float16", "float32", "float64"):
      for masked in (False, True):
        compiled = T.compile(producer_template(dtype, masked=masked), compiler=compiler)
        inputs = (
          vecops.TensorMeta((3, 129), dtype=dtype),
          vecops.TensorMeta((3, 129), dtype="float32"),
          vecops.TensorMeta((3, 129), dtype=dtype),
          vecops.TensorMeta((3, 129), dtype=dtype),
        )
        compiled.operator(*inputs, **compiled._core.specialization_values)
        kernels[dtype, masked] = compiled
  vecops.compile_batch(collector.requests, parallelism=4, execution_parallelism=1)
  return kernels


def reference(source, mask, masked):
  values = source.astype(np.float64)
  if not values.shape[-1]:
    return values
  scores = np.where(mask != 0, values, -np.inf) if masked else values
  maximum = scores.max(axis=-1, keepdims=True)
  maximum = np.where(maximum == -np.inf, 0, maximum) if masked else maximum
  probability = np.exp(scores - maximum)
  total = probability.sum(axis=-1, keepdims=True)
  return probability / np.where(total > 0, total, 1)


@pytest.mark.parametrize("dtype", ("float16", "float32", "float64"))
@pytest.mark.parametrize("masked", (False, True))
def test_native_shared_producer_tail_layouts_aliases_and_row_refresh(native_shared_producers, dtype, masked):
  compiled = native_shared_producers[dtype, masked]
  rng = np.random.default_rng(413)
  for width in (0, 1, 7, 17, 65, 128, 129, 257):
    layouts = ("dense", "strided", "negative", "inplace", "shifted")
    if masked and dtype == "float32":
      layouts += ("maskalias",)
    for layout in layouts:
      storage = np.empty(3 * width + 1, dtype=dtype)
      source = storage[:-1].reshape(3, width)
      if layout == "strided":
        source = np.empty((3, 2 * width), dtype=dtype)[:, ::2]
      elif layout == "negative":
        source = source[:, ::-1]
      output_storage = np.full((3, 2 * width + 2), -73, dtype=dtype)
      output = output_storage[:, 1 : width + 1]
      if layout == "strided":
        output = output_storage[:, 1 : 2 * width + 1 : 2]
      elif layout == "negative":
        output = output[:, ::-1]
      elif layout == "inplace":
        output = source
      elif layout == "shifted":
        output = storage[1:].reshape(3, width)
      mask = (rng.random((3, width)) > 0.3).astype(np.float32)
      if masked:
        mask[1] = 0  # All-masked row must remain exactly zero.
      if layout == "maskalias":
        output = mask
      other = np.empty((3, width), dtype=dtype)
      for repeat in range(2):
        if layout == "maskalias":
          mask[...] = rng.random((3, width)) > 0.3
          mask[1] = 0
        source[...] = rng.normal(size=(3, width)) * 4 + repeat
        if width:
          source[2, 0] = 100  # Exercise the <-80 estimate cutoff and tail masks.
        before = source.copy()
        expected = reference(before, mask, masked)
        compiled(source, mask, output, other)
        # Match the published Estimate contract, including unrefined SVE
        # reciprocal estimates; do not silently switch to strict math here.
        np.testing.assert_allclose(output, expected, rtol=8e-3, atol=2e-6)
        if masked:
          np.testing.assert_array_equal(output[1], 0)
      if layout not in {"inplace", "shifted", "maskalias"}:
        np.testing.assert_array_equal(output_storage[:, 0], -73)
        np.testing.assert_array_equal(output_storage[:, -1], -73)
        if layout == "strided":
          np.testing.assert_array_equal(output_storage[:, 2 : 2 * width + 1 : 2], -73)
  # The cached producer must match the existing materialized schedule for
  # special values. Masked softmax specifies finite valid logits, and native
  # exp_neg keeps backend NaN behavior; neither promises a NumPy-style whole
  # NaN row. FP32-compute Estimate's visible polynomial does preserve NaNs.
  special = np.zeros((3, 129), dtype=dtype)
  special[0, 127] = np.nan
  special[1, 1:] = -1000
  mask = np.ones(special.shape, dtype=np.float32)
  if masked:
    mask[2] = 0
  output = np.empty_like(special)
  with np.errstate(invalid="ignore", over="ignore"):
    expected = reference(special, mask, masked)
  compiled(special, mask, output, np.empty_like(special))
  baseline = special.copy()
  compiled(baseline, mask, baseline, np.empty_like(special))
  np.testing.assert_array_equal(np.isnan(output), np.isnan(baseline))
  # Aliasing selects the whole-region fallback, which does not share a
  # row-local producer. Keep exact finite-bit parity, including signed zeros;
  # NaN payloads are deliberately not part of the math contract.
  bits = {"float16": np.uint16, "float32": np.uint32, "float64": np.uint64}[dtype]
  present = ~np.isnan(output)
  np.testing.assert_array_equal(output[present].view(bits), baseline[present].view(bits))
  np.testing.assert_allclose(output[1:], expected[1:], rtol=8e-3, atol=2e-6)
  if not masked and dtype != "float64":
    assert np.isnan(output[0]).all()
  np.testing.assert_array_equal(output[1, 1:], 0)
  if masked:
    np.testing.assert_array_equal(output[2], 0)
