# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Full reduction cache writes stay unmasked; partial lanes retain semantics."""

import re

import numpy as np
import pytest
import vecops
import vecops.language as T
from dsl_test_utils import assert_no_vec_mask_arrays, make_compiler
from vecops.language._compiler.backends.cpu.backend import CPUBackend


def cached_reduction_template(dtype="float32", operation="add"):
  N = T.dynamic(lower=0)
  AS, CS = T.dynamic(), T.dynamic()
  DType = T.DType(dtype)

  @T.template
  def kernel(
    A: T.Tensor((N,), (AS,), DType),
    C: T.Tensor((N,), (CS,), DType, access="output"),
  ):
    producer = T.exp(T.load(A), accuracy="strict")
    if operation == "add":
      statistic = T.reduce_add(producer)
    elif operation == "min":
      statistic = T.reduce_min(producer)
    else:
      statistic = T.reduce_max(producer)
    T.store(C, producer + statistic)

  return kernel


def full_vector_blocks(source):
  """Select lexical full groups rather than unrelated fallback/tail code."""
  lines = source.splitlines()
  for index, line in enumerate(lines):
    if "/* full vector groups */" not in line:
      continue
    balance = line.count("{") - line.count("}")
    block = [line]
    for child in lines[index + 1 :]:
      block.append(child)
      balance += child.count("{") - child.count("}")
      if not balance:
        break
    assert not balance
    yield "\n".join(block)


@pytest.mark.parametrize("dtype", ("float32", "float64"))
@pytest.mark.parametrize("operation", ("add", "min", "max"))
def test_full_multiword_reduction_cache_writes_do_not_construct_masks(dtype, operation):
  lowered = CPUBackend().lower(
    cached_reduction_template(dtype, operation).ir,
    (),
    target="SVE",
    options={"cpu_math_chunk_pow": 1},
  )
  assert len(lowered.kernel.materializations) == 1
  cache = next(iter(lowered.kernel.materializations.values())).storage
  cache_stores = re.compile(rf"vecops::vec::store_convert\([^\n]*{cache}\.data\(\)[^\n]*;")
  stores = [store for block in full_vector_blocks(lowered.source) for store in cache_stores.findall(block)]
  assert stores
  assert all("vecops::vec::opt::unmasked" in store and "opt::first" not in store for store in stores)
  for block in full_vector_blocks(lowered.source):
    if cache_stores.search(block):
      assert f"if ({cache}_enabled)" in block
      assert not re.search(rf"if \({cache}_enabled &&", block)
      assert "vecops::vec::mwhilelt" not in block
  # Full groups do not remove the bounds predicate or reduction identity from
  # masked tails, including min/max on an elementwise-transformed input.
  all_stores = cache_stores.findall(lowered.source)
  assert any("vecops::vec::opt::first(" in store for store in all_stores)
  assert "vecops::vec::mwhilelt" in lowered.source
  identity = "float{0}" if dtype == "float32" else "double{0}"
  if operation != "add":
    identity = f"{'-' if operation == 'max' else ''}std::numeric_limits<{'float' if dtype == 'float32' else 'double'}>::infinity()"
  assert identity in lowered.source
  assert_no_vec_mask_arrays(lowered.source)


@pytest.fixture(scope="module")
def native_cached_reductions(tmp_path_factory):
  path = tmp_path_factory.mktemp("reduction-cache")
  compiler = make_compiler(jobs=4, cache_dir=path / "cache", build_dir=path / "build")
  kernels = {}
  with vecops.collect_compile_requests() as collector:
    for dtype in ("float32", "float64"):
      for operation in ("add", "min", "max"):
        compiled = T.compile(
          cached_reduction_template(dtype, operation),
          compiler=compiler,
          pass_options={"cpu_math_chunk_pow": 1},
        )
        operands = (vecops.TensorMeta((129,), dtype=dtype),) * 2
        compiled.operator(*operands, **compiled._core.specialization_values)
        kernels[dtype, operation] = compiled
  vecops.compile_batch(collector.requests, parallelism=4, execution_parallelism=1)
  return kernels


@pytest.mark.parametrize("dtype", ("float32", "float64"))
@pytest.mark.parametrize("operation", ("add", "min", "max"))
def test_native_cache_full_groups_tail_and_aliases(native_cached_reductions, dtype, operation):
  compiled = native_cached_reductions[dtype, operation]
  reducer = {"add": np.sum, "min": np.min, "max": np.max}[operation]
  # Widths span native-word, multiword and masked-tail cases across x86/SVE.
  for width in (0, 1, 7, 17, 32, 64, 65, 128, 129, 257):
    for layout in ("dense", "strided", "negative", "inplace", "shifted"):
      storage = np.full(2 * width + 3, -91, dtype=dtype)
      source = storage[1 : 1 + width]
      if layout == "strided":
        source = storage[1 : 1 + 2 * width : 2]
      elif layout == "negative":
        source = source[::-1]
      output_storage = np.full(2 * width + 3, -73, dtype=dtype)
      output = output_storage[1 : 1 + width]
      if layout == "strided":
        output = output_storage[1 : 1 + 2 * width : 2]
      elif layout == "negative":
        output = output[::-1]
      elif layout == "inplace":
        output = source
      elif layout == "shifted":
        output = storage[2 : 2 + width]
      for bias in (0.0, 0.125):
        # Positive min inputs and negative max inputs expose an incorrect
        # identity after exp() turns inactive zero-filled lanes into ones.
        lower, upper = (1.0, 2.0) if operation == "min" else (-3.0, -1.0)
        source[...] = np.linspace(lower + bias, upper + bias, width, dtype=dtype)
        before = source.copy()
        exponential = np.exp(before.astype(np.float64))
        expected = exponential + reducer(exponential) if width else exponential
        compiled(source, output)
        np.testing.assert_allclose(output, expected, rtol=3e-5 if dtype == "float32" else 5e-12, atol=1e-6)
      if output is not source and layout != "shifted":
        if layout == "strided":
          np.testing.assert_array_equal(output_storage[::2], -73)
        assert output_storage[0] == -73 and output_storage[-1] == -73
