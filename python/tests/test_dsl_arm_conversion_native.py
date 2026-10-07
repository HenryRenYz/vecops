"""Native cross-dtype conversion and indexed-memory correctness.

This is intentionally ISA-neutral: it runs on the local target or on ARM via
``VECOPS_TEST_TARGET`` and ``make_compiler``. All generated kernels are
registered in one compile batch to keep cold-build cost bounded.
"""

from __future__ import annotations

import numpy as np
import pytest
import vecops
import vecops.language as T
from dsl_test_utils import assert_no_vec_mask_arrays, make_compiler
from vecops.language._compiler.backends.cpu.transpose import TransposePlan

FLOAT_CONVERSIONS = (
  ("bfloat16", "float16"),
  ("float16", "bfloat16"),
  ("bfloat16", "float32"),
  ("float32", "bfloat16"),
  ("float16", "float32"),
  ("float32", "float16"),
)
UNSIGNED_DTYPES = ("uint16", "uint32", "uint64")


def _np_dtype(name):
  if name == "bfloat16":
    return np.dtype(pytest.importorskip("ml_dtypes").bfloat16)
  return np.dtype(name)


def _transpose_kernel(source_dtype, destination_dtype):
  M, N = T.dynamic(lower=0), T.dynamic(lower=0)
  AS0, AS1, CS0, CS1 = (T.dynamic() for _ in range(4))

  @T.template
  def kernel(
    A: T.Tensor((M, N), (AS0, AS1), source_dtype),
    C: T.Tensor((N, M), (CS0, CS1), destination_dtype, access="output"),
  ):
    T.transpose_copy(A, C, compute_dtype="float32")

  return kernel


def _unsigned_indexed_kernel(dtype):
  N = T.dynamic(lower=0)
  AS, VS, LS, GS, DS = (T.dynamic() for _ in range(5))

  @T.template
  def kernel(
    A: T.Tensor((N,), (AS,), dtype),
    Indices: T.Tensor((N,), dtype="int64"),
    Values: T.Tensor((N,), (VS,), dtype="float32"),
    Loaded: T.Tensor((N,), (LS,), dtype="float32", access="output"),
    Gathered: T.Tensor((N,), (GS,), dtype="float32", access="output"),
    Scattered: T.Tensor((N,), (DS,), dtype=dtype, access="inout"),
  ):
    indices = T.load(Indices)
    valid = (indices >= 0) & (indices < N)

    # Exercise converted masked load/store independently from indexed gather.
    loaded = T.load(A, dtype="float32", mask=valid, other=0.0)
    T.store(Loaded, loaded, mask=valid)
    gathered = T.gather(A, indices, dtype="float32", mask=valid, other=0.0)
    T.store(Gathered, gathered, mask=valid)

    # The selected values are exact integers in float32 and all within the
    # destination range, including uint64; invalid indices must not scatter.
    values = T.cast(T.load(Values), dtype=dtype)
    T.scatter(Scattered, indices, values, mask=valid, unique=True)

  return kernel


@T.template
def _low_precision_and_unsigned_print_kernel(
  BF16: T.Tensor((3,), dtype="bfloat16"),
  U16: T.Tensor((3,), dtype="uint16"),
  U32: T.Tensor((3,), dtype="uint32"),
  U64: T.Tensor((3,), dtype="uint64"),
):
  T.print(T.load(BF16), flush=True)
  T.print(T.load(U16), flush=True)
  T.print(T.load(U32), flush=True)
  T.print(T.load(U64), flush=True)


@pytest.fixture(scope="module")
def compiler(tmp_path_factory):
  directory = tmp_path_factory.mktemp("dsl-arm-conversions")
  return make_compiler(
    target="Native",
    jobs=4,
    output="quiet",
    cache_dir=directory / "cache",
    build_dir=directory / "build",
  )


def test_small_float_conversion_transpose_tails_and_strides(compiler):
  requests = []
  with vecops.collect_compile_requests() as collector:
    for source_dtype, destination_dtype in FLOAT_CONVERSIONS:
      compiled = T.compile(_transpose_kernel(source_dtype, destination_dtype), compiler=compiler)
      compiled.operator(
        vecops.TensorMeta((5, 7), strides=(7, 1), dtype=source_dtype),
        vecops.TensorMeta((7, 5), strides=(5, 1), dtype=destination_dtype),
        **compiled._core.specialization_values,
      )
      requests.append((source_dtype, destination_dtype, compiled))
  vecops.compile_batch(collector.requests, parallelism=4, execution_parallelism=1)

  rng = np.random.default_rng(51)
  for source_dtype, destination_dtype, compiled in requests:
    assert any(isinstance(region, TransposePlan) for region in compiled.vector_ir.regions)
    assert_no_vec_mask_arrays(compiled.generated_source)
    source_np, destination_np = _np_dtype(source_dtype), _np_dtype(destination_dtype)
    for rows, columns in ((5, 7), (17, 19), (31, 33)):
      # Fractions with a power-of-two denominator make the widening path exact;
      # the final cast is checked against NumPy's destination rounding.
      values = rng.integers(-127, 128, size=(rows, columns)).astype(np.float32) / 16
      source = values.astype(source_np)
      for input_layout in ("C", "F", "negative-columns"):
        a = source if input_layout == "C" else np.asfortranarray(source) if input_layout == "F" else source[:, ::-1]
        for output_layout in ("C", "strided", "negative-rows"):
          if output_layout == "C":
            output = np.empty((columns, rows), dtype=destination_np)
          elif output_layout == "strided":
            backing = np.empty((columns, rows * 2), dtype=destination_np)
            output = backing[:, ::2]
          else:
            output = np.empty((columns, rows), dtype=destination_np)[::-1]
          compiled.operator.native(a, output, **compiled._core.specialization_values)
          expected = a.astype(np.float32).T.astype(destination_np)
          np.testing.assert_array_equal(output, expected)


def test_unsigned_cast_masked_load_gather_scatter_tails(compiler):
  requests = []
  with vecops.collect_compile_requests() as collector:
    for dtype in UNSIGNED_DTYPES:
      compiled = T.compile(_unsigned_indexed_kernel(dtype), compiler=compiler)
      compiled.operator(
        vecops.TensorMeta((19,), strides=(1,), dtype=dtype),
        vecops.TensorMeta((19,), dtype="int64"),
        vecops.TensorMeta((19,), strides=(1,), dtype="float32"),
        vecops.TensorMeta((19,), strides=(1,), dtype="float32", access=vecops.TensorAccess.output),
        vecops.TensorMeta((19,), strides=(1,), dtype="float32", access=vecops.TensorAccess.output),
        vecops.TensorMeta((19,), strides=(1,), dtype=dtype, access=vecops.TensorAccess.inout),
        **compiled._core.specialization_values,
      )
      requests.append((dtype, compiled))
  vecops.compile_batch(collector.requests, parallelism=4, execution_parallelism=1)

  for dtype, compiled in requests:
    assert_no_vec_mask_arrays(compiled.generated_source)
    native = _np_dtype(dtype)
    for size in (5, 19, 33):
      # Values remain exactly representable in float32, including for uint64;
      # array views also exercise dynamic negative/non-unit strides.
      logical = np.arange(size, dtype=np.uint64) * 512 + 7
      base = np.empty(size * 2, dtype=native)
      base[::2] = logical.astype(native)
      source = base[::2][::-1]
      indices = np.arange(size - 1, -1, -1, dtype=np.int64)
      if size >= 3:
        indices[0], indices[1] = -1, size
      float_values = (np.arange(size, dtype=np.float32) * 256 + 13).astype(np.float32)

      loaded_backing = np.full(size * 2, -11, dtype=np.float32)
      loaded = loaded_backing[::2][::-1]
      gathered_backing = np.full(size * 2, -13, dtype=np.float32)
      gathered = gathered_backing[::2]
      scattered_backing = np.zeros(size * 2, dtype=native)
      scattered = scattered_backing[1::2]
      compiled.operator.native(
        source,
        indices,
        float_values,
        loaded,
        gathered,
        scattered,
        **compiled._core.specialization_values,
      )

      valid = (indices >= 0) & (indices < size)
      expected_loaded = np.full(size, -11, dtype=np.float32)
      expected_loaded[valid] = source.astype(np.float32)[valid]
      expected_gathered = np.full(size, -13, dtype=np.float32)
      expected_gathered[valid] = source[indices[valid]].astype(np.float32)
      expected_scattered = np.zeros(size, dtype=native)
      expected_scattered[indices[valid]] = float_values[valid].astype(native)
      np.testing.assert_array_equal(loaded, expected_loaded)
      np.testing.assert_array_equal(gathered, expected_gathered)
      np.testing.assert_array_equal(scattered, expected_scattered)


def test_print_bfloat16_and_unsigned_arrays(compiler, capfd):
  dtype = _np_dtype("bfloat16")
  compiled = T.compile(_low_precision_and_unsigned_print_kernel, compiler=compiler)
  metadata = (
    vecops.TensorMeta((3,), dtype="bfloat16"),
    vecops.TensorMeta((3,), dtype="uint16"),
    vecops.TensorMeta((3,), dtype="uint32"),
    vecops.TensorMeta((3,), dtype="uint64"),
  )
  with vecops.collect_compile_requests() as collector:
    compiled.operator(*metadata, **compiled._core.specialization_values)
  vecops.compile_batch(collector.requests, parallelism=1, execution_parallelism=1)
  assert_no_vec_mask_arrays(compiled.generated_source)

  bf16 = np.array([0, 0.5, -1], dtype=dtype)
  u16 = np.array([0, 1, 255], dtype=np.uint16)
  u32 = np.array([0, 1, 255], dtype=np.uint32)
  u64 = np.array([0, 1, 255], dtype=np.uint64)
  compiled.operator.native(bf16, u16, u32, u64, **compiled._core.specialization_values)
  printed = capfd.readouterr().out
  assert "bfloat16[0, 0.5, -1]" in printed
  assert "uint16[0, 1, 255]" in printed
  assert "uint32[0, 1, 255]" in printed
  assert "uint64[0, 1, 255]" in printed
