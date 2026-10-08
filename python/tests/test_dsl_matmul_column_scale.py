# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Column parameters follow original logical axes, including col-GEMV/swap.

This fixes benchmark transformations, not Matmul or the DSL to match a broken
reference. Native checks each side against independent NumPy values; it never
accepts agreement between two wrong implementations. The complete old catalog
keeps its recorded two failures and must be rerun under new source hashes.
"""

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import vecops
import vecops.language as T
from dsl_test_utils import make_compiler
from vecops._arm_vector_state import get_vector_state

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmarks"))

import dsl_matmul_source_catalog as runner

PIPELINES = ("PerColumnScale", "BiasPerColumnDequantize", "BiasPerColumnRequantizeU8")


def transform_template(pipeline, axis, swap):
  M, N = T.dynamic(lower=1), T.dynamic(lower=1)
  A0, A1, C0, C1 = (T.dynamic() for _ in range(4))
  Axis, SwapCoordinates = T.Const(axis), T.Const(swap)
  OutputPipeline = T.Const(runner.OUTPUTS.index(pipeline))
  # The included reference retains its real public Matmul entry. These
  # bindings only satisfy that unused dependent function's public contract.
  NativeA = T.DType(vecops.bfloat16)
  NativeB = T.DType(vecops.bfloat16)
  InputPipelineA = T.Const(0)
  InputPipelineB = T.Const(0)
  HasPrior = T.Const(False)
  BroadcastPrior = T.Const(False)
  SwapAB = T.Const(True)
  output = "uint8" if pipeline == "BiasPerColumnRequantizeU8" else "float32"

  @T.template(source=Path(__file__).with_name("dsl_matmul_column_scale_reference.cpp"))
  def kernel(
    A: T.Tensor((M, N), (A0, A1), "float32"),
    C: T.Tensor((M, N), (C0, C1), output, access="output"),
    Scale: T.Tensor((1, N), dtype="float32"),
    Zero: T.Tensor((1, 1), dtype="int32"),
  ):
    Axis, SwapCoordinates, OutputPipeline  # noqa: B018
    NativeA, NativeB, InputPipelineA, InputPipelineB, HasPrior, BroadcastPrior, SwapAB  # noqa: B018

  return kernel


def transform_arguments(pipeline, shape=(19, 23)):
  m, n = shape
  a = ((np.arange(m * n).reshape(m, n) % 29) - 7).astype(np.float32)[:, ::-1]
  # Nonunit/reversed output addresses should not change the logical scale.
  dtype = np.uint8 if pipeline == "BiasPerColumnRequantizeU8" else np.float32
  c = np.empty((m, 2 * n), dtype=dtype)[:, ::-2]
  storage = np.zeros(n + 256, dtype=np.float32)
  scale = np.ndarray((1, n), dtype=np.float32, buffer=storage)
  scale[:] = 0.5 + (np.arange(n) % 7) * 0.0625
  zero = np.array([[7]], dtype=np.int32)
  expected = a * scale
  if pipeline == "BiasPerColumnRequantizeU8":
    expected = np.clip(expected + 7, 0, 255).astype(np.uint8)
  return (a, c, scale, zero), expected


def regression_scenarios(*, svl_bytes=64):
  (base,) = (
    case
    for case in runner.load_cases(ROOT, svl_bytes=svl_bytes)
    if case.catalog == "MatmulFusionExpanded"
    and case.batch is None
    and case.name.endswith("skinny_long_k_tail_col")
    and case.native_a == "bf16"
    and case.memory_a == "fp32"
    and case.output_pipeline == "PerColumnScale"
  )
  for swap in (False, True):
    config = None if swap else "EnableSwapAB=false"
    for batch, shape in ((None, (8, 1, 4097)), (None, (1, 19, 9)), ("2", (8, 1, 4097))):
      case = replace(
        base, m=shape[0], n=shape[1], k=shape[2], configuration=config, batch=batch, broadcast="B" if batch else None
      )
      for mode in ("Const", "Dynamic"):
        yield runner.SourceScenario(case, mode, "dot", "serial", "raw")


def test_column_transform_original_and_paired_sources_select_original_axis():
  for path in (
    ROOT / "benchmarks/ops/MatmulScenarioBenchCommon.h",
    ROOT / "benchmarks/dsl_matmul_fusion_reference.cpp",
  ):
    source = path.read_text()
    assert source.count("context.vector_axis == static_cast<int>(coordinate.size()) - 1") == 2
    assert "fill(tag, scales[" in source


def test_column_transform_fixture_has_independent_named_reference_captures():
  template = transform_template("PerColumnScale", 0, True)
  names = {value.name for value in template.ir.captures}
  assert {
    "Axis",
    "SwapCoordinates",
    "OutputPipeline",
    "NativeA",
    "NativeB",
    "InputPipelineA",
    "InputPipelineB",
    "HasPrior",
    "BroadcastPrior",
    "SwapAB",
  } <= names


@pytest.mark.parametrize("pipeline", PIPELINES)
def test_column_transform_fixture_oracle_has_nonconstant_columns_and_zero_padding(pipeline):
  for shape in ((19, 23), (8, 1), (1, 9)):
    values, expected = transform_arguments(pipeline, shape)
    assert values[0].strides[1] < 0 and values[1].strides[1] < 0
    assert values[1].shape == expected.shape
    assert not np.asarray(values[2].base)[shape[1] :].any()
    if pipeline == "PerColumnScale" and shape == (8, 1):
      np.testing.assert_array_equal(expected, values[0] * np.float32(0.5))


def test_column_scale_critical_matrix_fixture_keeps_const_dynamic_swap_and_shared_b():
  scenarios = tuple(regression_scenarios())
  assert len(scenarios) == 12
  assert {scenario.mode for scenario in scenarios} == {"Const", "Dynamic"}
  assert {scenario.swap for scenario in scenarios} == {False, True}
  assert {scenario.batch for scenario in scenarios} == {1, 2}
  for scenario in scenarios:
    scenario.validate()
    values, expected = runner.source_arguments(scenario)
    assert values[2].shape == expected.shape
    if scenario.case.n == 1:
      np.testing.assert_array_equal(expected, np.full_like(expected, -96.0234375))
    if scenario.batched:
      assert scenario.shared_b and not scenario.flatten_batch
      assert values[1].strides[0] == 0


@pytest.fixture(scope="module")
def compiled_transforms(tmp_path_factory):
  compiler = make_compiler(build_dir=tmp_path_factory.mktemp("column-transform"))
  kernels = {
    (pipeline, axis, swap): T.compile(transform_template(pipeline, axis, swap), compiler=compiler)
    for pipeline in PIPELINES
    for axis in (0, 1)
    for swap in (False, True)
  }
  with vecops.collect_compile_requests() as collector:
    for (pipeline, _, _), kernel in kernels.items():
      values, _ = transform_arguments(pipeline)
      kernel.operator(*values, **kernel._core.specialization_values)
  assert len(collector.requests) == len(kernels) == 12
  vecops.compile_batch(collector.requests, parallelism=4, execution_parallelism=1)
  return kernels


@pytest.mark.parametrize("pipeline", PIPELINES)
def test_native_column_transforms_both_axes_and_swap_mapping(compiled_transforms, pipeline):
  for axis in (0, 1):
    for swap in (False, True):
      kernel = compiled_transforms[pipeline, axis, swap]
      for shape in ((19, 23), (8, 1), (1, 9)):
        values, expected = transform_arguments(pipeline, shape)
        kernel.operator.native(*values, **kernel._core.specialization_values)
        np.testing.assert_array_equal(values[1], expected)


def test_native_column_scale_matrix_const_dynamic_swap_and_shared_b(tmp_path):
  compiler = make_compiler(build_dir=tmp_path / "column-matrix")
  prepared = []
  with vecops.collect_compile_requests() as collector:
    for scenario in regression_scenarios(svl_bytes=get_vector_state().sme_bytes):
      templates, _ = runner.source_templates(scenario)
      pair = tuple(T.compile(template, compiler=compiler) for template in templates)
      for kernel in pair:
        kernel.operator(*runner.source_metadata(scenario), 1, **kernel._core.specialization_values)
      prepared.append((scenario, pair))
  assert len(collector.requests) == 24
  vecops.compile_batch(collector.requests, parallelism=4, execution_parallelism=1)
  for scenario, pair in prepared:
    values, expected = runner.source_arguments(scenario)
    for kernel in pair:
      values[2].fill(np.nan)
      kernel.operator.native(*values, 1, **kernel._core.specialization_values)
      np.testing.assert_allclose(values[2], expected, rtol=8e-4, atol=8e-4)
