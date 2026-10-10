# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""FP32 pointwise packing through ZA, including independently broadcast DAGs.

The ISA-neutral checks inspect both the ZA specialization and its ordinary
gather fallback. Native execution is deliberately ARM-only: a compile-time
route is not evidence that streaming loads, conversions or tails are correct.
"""

from pathlib import Path

import numpy as np
import pytest
import vecops
import vecops.language as T
from dsl_test_utils import assert_no_vec_mask_arrays, make_compiler
from test_dsl_sme_packing import _available, _dtype, _expected
from vecops._arm_vector_state import get_vector_state
from vecops.language._compiler.backends.cpu.backend import CPUBackend

DTYPES = ("bfloat16", "float16", "int8", "uint8")
PIPELINES = ("convert", "affine", "rowstats", "params")


def _template(dtype, side, pipeline, parameter_dtype="float32", masked=False, reference=False):
  S, K, PS = (T.dynamic(lower=0) for _ in range(3))
  SS, KS = T.dynamic(), T.dynamic()
  shape, strides = ((S, K), (SS, KS)) if side == "A" else ((K, S), (KS, SS))
  row_shape = (S, 1) if side == "A" else (1, S)
  column_shape = (1, K) if side == "A" else (K, 1)

  if reference:
    Pipeline, SideB = T.Const(PIPELINES.index(pipeline)), T.Const(side == "B")
    Native = T.DType(dtype)

    @T.template(source=Path(__file__).with_name("dsl_sme_transformed_pack_reference.cpp"))
    def cpp(
      X: T.Tensor(shape, strides, "float32"),
      Mean: T.Tensor(row_shape, dtype=parameter_dtype),
      Rstd: T.Tensor(row_shape, dtype="float32"),
      Weight: T.Tensor(column_shape, dtype="float32"),
      Bias: T.Tensor(column_shape, dtype="float32"),
      P: T.Tensor((PS,), dtype=dtype, access="output"),
    ):
      Pipeline, SideB, Native  # noqa: B018 -- C++ specialization captures

    return cpp

  @T.template
  def kernel(
    X: T.Tensor(shape, strides, "float32"),
    Mean: T.Tensor(row_shape, dtype=parameter_dtype),
    Rstd: T.Tensor(row_shape, dtype="float32"),
    Weight: T.Tensor(column_shape, dtype="float32"),
    Bias: T.Tensor(column_shape, dtype="float32"),
    P: T.Tensor((PS,), dtype=dtype, access="output"),
  ):
    x = T.load(X)
    if masked:
      x = T.load(X, mask=x > 0, other=0.0)
    if pipeline == "affine":
      value = x * 0.5 + 7.0
    elif pipeline == "rowstats":
      value = (x - T.load(Mean, dtype="float32")) * T.load(Rstd) + 7.0
    elif pipeline == "params":
      normalized = (x - T.load(Mean, dtype="float32")) * T.load(Rstd)
      value = normalized * T.load(Weight) + T.load(Bias)
    else:
      value = x
    T.pack(value, P, dtype=dtype, side=side, format="sme")

  return kernel


def _za_body(source):
  marker = source.index("/* FP32 DAG K-contiguous SME ZA packing */")
  return source[marker : source.index("/* non-contiguous K preserves vector gather packing */", marker)]


def _uncaptured_convert_template(dtype, side):
  S, K, PS = (T.dynamic(lower=0) for _ in range(3))
  SS, KS = T.dynamic(), T.dynamic()
  shape, strides = ((S, K), (SS, KS)) if side == "A" else ((K, S), (KS, SS))

  @T.template
  def kernel(
    X: T.Tensor(shape, strides, "float32"),
    P: T.Tensor((PS,), dtype=dtype, access="output"),
  ):
    # Unlike _template's lexical x binding, this load has no historical
    # capture edge; canonicalization may fold its final storage conversion.
    T.pack(T.load(X), P, dtype=dtype, side=side, format="sme")

  return kernel


@pytest.mark.parametrize("side", ("A", "B"))
def test_widening_parameter_loads_and_masked_primary_preserve_ordinary_fallback(side):
  for parameter_dtype, masked in (("bfloat16", False), ("float32", True)):
    source = (
      CPUBackend().lower(_template("bfloat16", side, "params", parameter_dtype, masked).ir, (), target="SME").source
    )
    assert "FP32 DAG K-contiguous SME ZA packing" not in source
    assert "raw K-contiguous SME ZA packing" not in source
    assert "pack_group" in source
    assert_no_vec_mask_arrays(source)


def test_intermediate_fp16_rounding_is_not_removed_from_a_bf16_pack():
  @T.template
  def kernel(X: T.Tensor((19, 17), dtype="float32"), P: T.Tensor((4096,), dtype="bfloat16", access="output")):
    half = T.cast(T.load(X), dtype="float16")
    T.pack(half, P, dtype="bfloat16", format="sme")

  source = CPUBackend().lower(kernel.ir, (), target="SME").source
  assert "FP32 DAG K-contiguous SME ZA packing" not in source
  assert "pack_group" in source


@pytest.mark.parametrize("shape", ((0, 1), (1, 0), (1, 1), (19, 1), (1, 17)))
def test_broadcast_fixture_keeps_the_annotations_exact_dense_strides(shape):
  values = np.empty(shape, dtype=np.float32)
  for source in (values, values.T):
    result = _dense(source)
    assert tuple(stride // result.dtype.itemsize for stride in result.strides) == (result.shape[1], 1)


def _values(spatial, k):
  rows = np.arange(spatial, dtype=np.float32).reshape(-1, 1)
  columns = np.arange(k, dtype=np.float32).reshape(1, -1)
  table = np.array(
    (-200, -128.75, -128, -7.5, -1.75, -0.0, 0, 0.25, 1.5, 7.75, 127, 127.75, 128, 255, 300),
    dtype=np.float32,
  )
  x = table[((rows * 3 + columns * 5) % len(table)).astype(np.int32)]
  mean = (rows % 3) - 1
  rstd = (rows % 2) + 1
  weight = ((columns % 2) + 1) * 0.5
  bias = (columns % 3) + 32
  return x, mean, rstd, weight, bias


def _transform(inputs, pipeline):
  x, mean, rstd, weight, bias = inputs
  if pipeline == "affine":
    return x * np.float32(0.5) + np.float32(7)
  if pipeline == "rowstats":
    return (x - mean) * rstd + np.float32(7)
  if pipeline == "params":
    return ((x - mean) * rstd) * weight + bias
  return x


def _dense(values):
  """Honor exact annotation strides, including singleton and empty axes."""
  storage = np.empty(max(1, values.size), dtype=values.dtype)
  result = np.ndarray(
    values.shape,
    dtype=values.dtype,
    buffer=storage,
    strides=(values.shape[1] * values.dtype.itemsize, values.dtype.itemsize),
  )
  result[...] = values
  return result


@pytest.mark.skipif(not _available(), reason="SME+FA64 conversion packing requires a target ARM CPU")
@pytest.mark.parametrize("dtype", DTYPES)
def test_uncaptured_folded_za_pack_preserves_narrow_alias_and_zero_padding(dtype, tmp_path):
  state = get_vector_state()
  assert state.sme_bytes is not None
  panel = state.sme_bytes // 2
  kp = 2 if dtype in {"bfloat16", "float16"} else 4
  compiler = make_compiler(target="Native", jobs=4, cache_dir=tmp_path / "cache", build_dir=tmp_path / "build")
  compiled_kernels = {}
  with vecops.collect_compile_requests() as collector:
    for side in ("A", "B"):
      compiled = T.compile(_uncaptured_convert_template(dtype, side), compiler=compiler)
      compiled.operator(
        vecops.TensorMeta((19, 17), dtype="float32"),
        vecops.TensorMeta((8192,), dtype=dtype),
        **compiled._core.specialization_values,
      )
      assert "FP32 DAG K-contiguous SME ZA packing" in compiled.generated_source
      compiled_kernels[side] = compiled
  assert len(collector.requests) == 2
  vecops.compile_batch(collector.requests, parallelism=compiler.jobs, execution_parallelism=1)
  dtype_np = _dtype(dtype)
  bits_dtype = np.dtype(f"uint{dtype_np.itemsize * 8}")
  shapes = ((0, 5), (5, 0), (1, 1), (panel - 1, kp + 1), (panel, kp), (panel + 3, 2 * kp + 1))
  for spatial, k in shapes:
    x = _values(spatial, k)[0]
    backing = np.empty((spatial, max(1, 2 * k)), dtype=np.float32)
    backing[:, : 2 * k : 2] = x
    for values in (x, x[::-1], backing[:, : 2 * k : 2]):
      converted = values
      if dtype in {"int8", "uint8"}:
        limits = np.iinfo(dtype_np)
        converted = np.clip(np.trunc(converted), limits.min, limits.max)
      expected = _expected(converted.astype(dtype_np), panel, kp).view(bits_dtype)
      for side, compiled in compiled_kernels.items():
        output = np.full(expected.size + 14, 0x55, dtype=bits_dtype)
        compiled.operator.native(
          values if side == "A" else values.T,
          output[7:-7].view(dtype_np),
          **compiled._core.specialization_values,
        )
        np.testing.assert_array_equal(output[7:-7], expected)
        np.testing.assert_array_equal(output[:7], 0x55)
        np.testing.assert_array_equal(output[-7:], 0x55)

  # The folded read's historical snapshot is narrow. An overlapping output
  # must use that snapshot's normal vector path, not load it into ZA as F32.
  spatial, k = panel + 1, kp + 1
  values = _values(spatial, k)[0]
  converted = values
  if dtype in {"int8", "uint8"}:
    limits = np.iinfo(dtype_np)
    converted = np.clip(np.trunc(converted), limits.min, limits.max)
  expected = _expected(converted.astype(dtype_np), panel, kp).view(bits_dtype)
  storage_bytes = max(values.nbytes, expected.nbytes)
  for side, compiled in compiled_kernels.items():
    storage = np.full(storage_bytes + 14, 0x55, dtype=np.uint8)
    logical = np.ndarray((spatial, k), dtype=np.float32, buffer=storage)
    logical[...] = values
    compiled.operator.native(
      logical if side == "A" else logical.T,
      storage[: expected.nbytes].view(dtype_np),
      **compiled._core.specialization_values,
    )
    np.testing.assert_array_equal(storage[: expected.nbytes].view(bits_dtype), expected)
    np.testing.assert_array_equal(storage[storage_bytes:], 0x55)


@pytest.mark.skipif(not _available(), reason="SME+FA64 conversion packing requires a target ARM CPU")
@pytest.mark.parametrize("dtype", DTYPES)
def test_transformed_za_pack_conversions_broadcast_tails_and_noncontiguous_fallback(dtype, tmp_path):
  state = get_vector_state()
  assert state.sme_bytes is not None
  panel, chunk = state.sme_bytes // 2, state.sme_bytes // 4
  kp = 2 if dtype in {"bfloat16", "float16"} else 4
  compiler = make_compiler(target="Native", jobs=4, cache_dir=tmp_path / "cache", build_dir=tmp_path / "build")
  compiled_kernels = {}
  with vecops.collect_compile_requests() as collector:
    for side in ("A", "B"):
      for pipeline in PIPELINES:
        shapes = ((19, 17), (19, 1), (19, 1), (1, 17), (1, 17))
        shapes = shapes if side == "A" else tuple(shape[::-1] for shape in shapes)
        arguments = [vecops.TensorMeta(shape, dtype="float32") for shape in shapes]
        pair = tuple(
          T.compile(_template(dtype, side, pipeline, reference=ref), compiler=compiler) for ref in (False, True)
        )
        for compiled in pair:
          compiled.operator(*arguments, vecops.TensorMeta((8192,), dtype=dtype), **compiled._core.specialization_values)
        assert_no_vec_mask_arrays(pair[0].generated_source)
        assert "FP32 DAG K-contiguous SME ZA packing" in pair[0].generated_source
        compiled_kernels[side, pipeline] = pair
  vecops.compile_batch(collector.requests, parallelism=compiler.jobs, execution_parallelism=1)
  shapes = ((0, 5), (5, 0), (1, 1), (panel - 1, kp + 1), (panel, chunk), (2 * panel + 3, 2 * chunk + 1))
  dtype_np = _dtype(dtype)
  bits_dtype = np.dtype(f"uint{dtype_np.itemsize * 8}")
  for spatial, k in shapes:
    inputs = _values(spatial, k)
    x, *parameters = inputs
    backing = np.empty((spatial, max(1, 2 * k)), dtype=np.float32)
    backing[:, : 2 * k : 2] = x
    for variant in ("dense", "negative", "gather"):
      logical = x if variant == "dense" else x[::-1] if variant == "negative" else backing[:, : 2 * k : 2]
      for (side, pipeline), pair in compiled_kernels.items():
        values = _transform((logical, *parameters), pipeline)
        # Integer conversion saturates and truncates; this oracle does not
        # assume NumPy's wrapping cast for negative uint8 or overflowing int8.
        if dtype in {"int8", "uint8"}:
          limits = np.iinfo(dtype_np)
          values = np.clip(np.trunc(values), limits.min, limits.max)
        expected = _expected(values.astype(dtype_np), panel, kp).view(bits_dtype)
        output = np.full(expected.size + 14, 0x55, dtype=bits_dtype)
        arguments = (logical if side == "A" else logical.T,) + tuple(
          _dense(value if side == "A" else value.T) for value in parameters
        )
        # The public C++ TransformPack baseline is not replaced by an
        # elementwise precomputation: both kernels evaluate the same graph.
        for compiled in pair:
          output.fill(0x55)
          compiled.operator.native(*arguments, output[7:-7].view(dtype_np), **compiled._core.specialization_values)
          np.testing.assert_array_equal(output[7:-7], expected)
          np.testing.assert_array_equal(output[:7], 0x55)
          np.testing.assert_array_equal(output[-7:], 0x55)
  # Validate the DSL alias contract independently: the public C++ packer is
  # not required to accept an output that overwrites its transformed input.
  spatial, k = panel + 1, kp + 1
  values = _values(spatial, k)
  for (side, pipeline), (compiled, _) in compiled_kernels.items():
    expected_values = _transform(values, pipeline)
    if dtype in {"int8", "uint8"}:
      limits = np.iinfo(dtype_np)
      expected_values = np.clip(np.trunc(expected_values), limits.min, limits.max)
    expected = _expected(expected_values.astype(dtype_np), panel, kp).view(bits_dtype)
    output_bytes = expected.nbytes
    storage_bytes = max(output_bytes, values[0].nbytes)
    storage = np.full(storage_bytes + 14, 0x55, dtype=np.uint8)
    logical = np.ndarray((spatial, k), dtype=np.float32, buffer=storage)
    logical[...] = values[0]
    arguments = (logical if side == "A" else logical.T,) + tuple(
      _dense(value if side == "A" else value.T) for value in values[1:]
    )
    compiled.operator.native(*arguments, storage[:output_bytes].view(dtype_np), **compiled._core.specialization_values)
    np.testing.assert_array_equal(storage[:output_bytes].view(bits_dtype), expected)
    np.testing.assert_array_equal(storage[storage_bytes:], 0x55)
