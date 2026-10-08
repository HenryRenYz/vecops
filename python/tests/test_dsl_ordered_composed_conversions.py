# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Ordered DSL casts retain intermediate rounding across grouped vector words.

This diagnoses the DSL's ordered conversion path, not the separately skipped
unordered C++ conversion round-trip. No compiler-version skip is appropriate:
the NumPy/ml_dtypes oracle must hold on every supported native compiler.
F64 inputs are exactly representable in F32: this tests ordered tuple/codegen
and observable half-precision intermediate rounding without changing the
public F64->F32->F16 backend/scalar rounding contract.
"""

import re

import numpy as np
import pytest
import vecops
import vecops.language as T
from dsl_test_utils import assert_no_vec_mask_arrays, make_compiler
from test_dsl_matmul_sve_optimized import _cross
from vecops._arm_vector_state import VectorState, get_vector_state
from vecops.language._compiler.backends.cpu.backend import CPUBackend
from vecops.language._compiler.backends.cpu.capabilities import CPP_TYPES
from vecops.language._compiler.backends.cpu.cost import resolve_target

CONVERSIONS = (("float32", "float16"), ("float64", "float16"), ("float32", "bfloat16"))
UNROLLS = (1, 2, 7)
OPTIONS = {"cpu_reduction_chunk_pow": 0, "cpu_math_chunk_pow": 0}


def _template(source_dtype, intermediate_dtype, unroll, masked):
  N = T.dynamic(lower=0)
  AS, MS, CS = (T.dynamic() for _ in range(3))

  @T.template
  def kernel(
    A: T.Tensor((N,), (AS,), source_dtype),
    Valid: T.Tensor((N,), (MS,), "int8"),
    C: T.Tensor((N,), (CS,), source_dtype, access="output"),
    Sum: T.Tensor((1,), dtype=source_dtype, access="output"),
  ):
    if masked:
      valid = T.load(Valid) != 0
      narrowed = T.load(A, dtype=intermediate_dtype, mask=valid, other=0)
    else:
      narrowed = T.load(A, dtype=intermediate_dtype)
    widened = T.cast(narrowed, source_dtype)
    total = T.reduce_add(widened, unroll=unroll)
    if masked:
      T.store(C, narrowed, mask=valid)
    else:
      T.store(C, narrowed)
    Sum[0] = total

  return kernel


def _lower(source_dtype, intermediate_dtype, unroll, masked, *, vl=64, svl=64):
  return CPUBackend().lower(
    _template(source_dtype, intermediate_dtype, unroll, masked).ir,
    (),
    target="SVE2",
    options={**OPTIONS, "_vector_state": VectorState(vl, svl)},
  )


def _assert_ordered_source(source, source_dtype, intermediate_dtype):
  assert_no_vec_mask_arrays(source)
  # Conversion defaults are ordered; an unordered token would weaken the
  # public lane-order/intermediate-rounding contract exercised by this test.
  assert "unordered" not in source
  assert "vecops::vec::load_convert(" in source
  assert "vecops::vec::store_convert(" in source
  assert "vecops::vec::convert(" in source
  assert CPP_TYPES[source_dtype] in source and CPP_TYPES[intermediate_dtype] in source


@pytest.mark.parametrize("source_dtype,intermediate_dtype", CONVERSIONS)
@pytest.mark.parametrize("unroll", UNROLLS)
@pytest.mark.parametrize("masked", (False, True))
def test_ordered_composed_casts_group_full_words_and_keep_rounding(source_dtype, intermediate_dtype, unroll, masked):
  artifact = _lower(source_dtype, intermediate_dtype, unroll, masked)
  _assert_ordered_source(artifact.source, source_dtype, intermediate_dtype)
  regions = [region for region in artifact.lowered.regions if region.unroll == unroll]
  assert regions
  counts = {1: [1], 2: [2], 7: [4, 2, 1]}[unroll]
  assert any([group.count for group in groups] == counts for region in regions for groups in region.groupings.values())
  if unroll == 7:
    assert f"ScalableTag<{CPP_TYPES[source_dtype]}, 2>" in artifact.source


def _dtype(name):
  if name == "bfloat16":
    return np.dtype(pytest.importorskip("ml_dtypes").bfloat16)
  return np.dtype(name)


def _values(size, source_dtype, intermediate_dtype):
  """Finite ties/neighbors, signed zero, and exactly summed rounded values."""
  dtype = _dtype(source_dtype)
  quantum = 2.0 ** (-7 if intermediate_dtype == "bfloat16" else -10)
  lane = np.arange(size)
  # Float16(double) and the SVE 64->16 demotion both pass through float32.
  # Use exactly-f32-representable F64 inputs so a direct NumPy oracle agrees
  # with that public backend/scalar rounding contract. Midpoint neighbors
  # still make the half-precision intermediate rounding observable.
  neighbor_dtype = np.dtype("float32") if source_dtype == "float64" else dtype
  values = (1 + ((lane % 17) + 0.5) * quantum).astype(neighbor_dtype)
  direction = np.where(lane % 3 == 0, np.inf, -np.inf).astype(neighbor_dtype)
  neighboring = np.nextafter(values, direction)
  values = np.where(lane % 3 == 2, values, neighboring).astype(dtype)
  values[lane % 2 == 1] *= -1
  values[lane % 31 == 0] = dtype.type(0)
  values[lane % 31 == 1] = dtype.type(-0.0)
  return values


def _view(values, stride):
  backing = np.empty(values.size * abs(stride), dtype=values.dtype)
  view = backing[::stride] if stride > 0 else backing[:: -abs(stride)]
  view[:] = values
  assert view.size == values.size
  return view


def _case(size, source_dtype, intermediate_dtype, masked, layout):
  strides = {"dense": (1, 1, 1), "strided": (2, 3, 2), "signed": (-2, -3, -2)}[layout]
  values = _values(size, source_dtype, intermediate_dtype)
  valid = np.ones(size, dtype="int8")
  if masked:
    valid[np.arange(size) % 5 == 2] = 0
    valid[-1] = 0
  a, mask = _view(values, strides[0]), _view(valid, strides[1])
  output = _view(np.full(size, -13, dtype=_dtype(source_dtype)), strides[2])
  total = np.empty(1, dtype=_dtype(source_dtype))
  narrowed = a.astype(_dtype(intermediate_dtype)).astype(_dtype(source_dtype))
  if source_dtype == "float64":
    np.testing.assert_array_equal(narrowed, a.astype("float32").astype(_dtype(intermediate_dtype)).astype("float64"))
  expected = output.copy()
  expected[valid != 0] = narrowed[valid != 0]
  expected_sum = np.where(valid != 0, narrowed, 0).sum(dtype=_dtype(source_dtype))
  return (a, mask, output, total), expected, expected_sum


@pytest.mark.parametrize("source_dtype,intermediate_dtype", CONVERSIONS)
@pytest.mark.parametrize("masked", (False, True))
@pytest.mark.parametrize("layout", ("dense", "strided", "signed"))
def test_ordered_composed_fixture_has_independent_rounding_and_layout_oracle(
  source_dtype, intermediate_dtype, masked, layout
):
  arguments, expected, expected_sum = _case(131, source_dtype, intermediate_dtype, masked, layout)
  a, mask, output, _ = arguments
  narrowed = a.astype(_dtype(intermediate_dtype)).astype(_dtype(source_dtype))
  valid = mask != 0
  assert np.any(a[valid] != narrowed[valid])  # Identity-folding is observably wrong.
  np.testing.assert_array_equal(expected[valid], narrowed[valid])
  np.testing.assert_array_equal(expected[~valid], output[~valid])
  assert expected_sum == np.where(valid, narrowed, 0).sum(dtype=_dtype(source_dtype))
  assert (
    tuple(argument.strides[0] // argument.itemsize for argument in arguments[:3])
    == {
      "dense": (1, 1, 1),
      "strided": (2, 3, 2),
      "signed": (-2, -3, -2),
    }[layout]
  )


def _cross_wrapper(source, source_dtype):
  definitions = (
    '#include "vecops/Kernel.h"',
    "namespace vecops::spec { inline constexpr auto Parallelism = vecops::meta::cint<1>; }",
    source,
    (
      f'extern "C" void probe({CPP_TYPES[source_dtype]}* a, std::int8_t* valid, '
      f"{CPP_TYPES[source_dtype]}* c, {CPP_TYPES[source_dtype]}* sum,vecops::nint_t n) {{"
    ),
    "auto shape=vecops::tensor::make_shape(vecops::meta::Any{n});",
    "auto strides=vecops::tensor::make_strides(vecops::meta::Any{-2});",
    "auto mask_strides=vecops::tensor::make_strides(vecops::meta::Any{-3});",
    "auto a_view=vecops::tensor::make_tensor(a, vecops::tensor::Layout{shape,strides});",
    "auto mask_view=vecops::tensor::make_tensor(valid, vecops::tensor::Layout{shape,mask_strides});",
    "auto c_view=vecops::tensor::make_tensor(c, vecops::tensor::Layout{shape,strides});",
    "auto sum_view=vecops::tensor::make_tensor(sum, vecops::tensor::Layout{",
    "vecops::tensor::make_shape(vecops::meta::cint<1>),vecops::tensor::make_strides(vecops::meta::cint<1>)});",
    "__kernel__(a_view, mask_view, c_view, sum_view);",
    "}",
  )
  return "\n".join(definitions)


@pytest.mark.parametrize("source_dtype,intermediate_dtype", CONVERSIONS)
@pytest.mark.parametrize("unroll,vl,svl", ((1, 16, 64), (2, 32, 16), (7, 64, 32)))
def test_ordered_composed_conversions_real_arm_cross(tmp_path, source_dtype, intermediate_dtype, unroll, vl, svl):
  artifact = _lower(source_dtype, intermediate_dtype, unroll, True, vl=vl, svl=svl)
  _assert_ordered_source(artifact.source, source_dtype, intermediate_dtype)
  assembly = _cross(tmp_path, artifact.source, lambda source: _cross_wrapper(source, source_dtype), vl=vl, svl=svl)
  # The wrapper has runtime data, not a compile-time known input array. The
  # narrowing must survive compilation, and neither path enters streaming SM.
  instruction = "bfcvt" if intermediate_dtype == "bfloat16" else "fcvt"
  assert re.search(rf"^\s*{instruction}\w*\s", assembly, re.MULTILINE)
  assert not re.search(r"^\s*smstart\b", assembly, re.MULTILINE)


def test_ordered_composed_conversions_native_words_masks_tails_and_signed_layouts(tmp_path):
  compiler = make_compiler(
    target="Native", jobs=4, output="quiet", cache_dir=tmp_path / "cache", build_dir=tmp_path / "build"
  )
  kernels = []
  with vecops.collect_compile_requests() as collector:
    for source_dtype, intermediate_dtype in CONVERSIONS:
      for unroll in UNROLLS:
        for masked in (False, True):
          compiled = T.compile(
            _template(source_dtype, intermediate_dtype, unroll, masked), compiler=compiler, pass_options=OPTIONS
          )
          compiled.operator(
            vecops.TensorMeta((131,), strides=(-2,), dtype=source_dtype),
            vecops.TensorMeta((131,), strides=(-3,), dtype="int8"),
            vecops.TensorMeta((131,), strides=(-2,), dtype=source_dtype, access=vecops.TensorAccess.output),
            vecops.TensorMeta((1,), dtype=source_dtype, access=vecops.TensorAccess.output),
            **compiled._core.specialization_values,
          )
          kernels.append((source_dtype, intermediate_dtype, unroll, masked, compiled))
  assert len(collector.requests) == len(kernels) == 18
  vecops.compile_batch(collector.requests, parallelism=4, execution_parallelism=1)
  vl = get_vector_state().sve_bytes
  if vl is None:
    # Non-SVE tests derive their geometry from the selected native target,
    # rather than borrowing a fixed ARM VL assumption.
    vl = (resolve_target(compiler.target).vector_bits or 128) // 8
  for source_dtype, intermediate_dtype, unroll, masked, compiled in kernels:
    _assert_ordered_source(compiled.generated_source, source_dtype, intermediate_dtype)
    lanes = vl // _dtype(source_dtype).itemsize
    for size in (1, lanes, 7 * lanes, 7 * lanes + 1, 7 * lanes + lanes - 1):
      for layout in ("dense", "strided", "signed"):
        arguments, expected, expected_sum = _case(size, source_dtype, intermediate_dtype, masked, layout)
        compiled.operator.native(*arguments, **compiled._core.specialization_values)
        label = f"{source_dtype}/{intermediate_dtype}/unroll={unroll}/masked={masked}/N={size}/{layout}"
        # Bitwise comparison retains signed zero and distinguishes even a
        # one-ULP intermediate-rounding error. All inputs here are finite.
        np.testing.assert_array_equal(
          arguments[2].view(f"u{arguments[2].itemsize}"), expected.view(f"u{expected.itemsize}"), err_msg=label
        )
        np.testing.assert_array_equal(arguments[3], [expected_sum], err_msg=label)
