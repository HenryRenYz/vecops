# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Normalization benchmark metadata and accuracy propagation contracts."""

import importlib.util
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import vecops
import vecops.language as T
from dsl_test_utils import make_compiler
from vecops.language._compiler import ir as I
from vecops.language._compiler.backends.cpu.backend import CPUBackend
from vecops.language._compiler.environment import freeze
from vecops.language._compiler.ir.traversal import nodes
from vecops.language._compiler.passes.kernel.reuse import plan_computation_reuse


@pytest.fixture(scope="module")
def catalog_module():
  path = Path(__file__).parents[2] / "benchmarks/dsl_norm_catalog.py"
  spec = importlib.util.spec_from_file_location("vecops_test_norm_catalog", path)
  module = importlib.util.module_from_spec(spec)
  # Dataclass postponed annotations resolve through the defining module.
  sys.modules[spec.name] = module
  spec.loader.exec_module(module)
  return module


def test_complete_normalization_benchmark_metadata(catalog_module):
  cases = catalog_module.catalog()
  assert Counter(case.operation for case in cases) == {"layernorm": 336, "softmax": 480}
  assert len({case.key for case in cases}) == len(cases)
  assert {case.meta for case in cases if case.operation == "layernorm"} == {"Any"}
  assert {case.meta for case in cases if case.operation == "softmax"} == {"Const", "Any"}
  assert {case.affine for case in cases if case.operation == "layernorm"} == set(catalog_module.AFFINE)
  assert {case.accuracy for case in cases if case.operation == "softmax"} == set(catalog_module.ACCURACY)
  both = catalog_module.catalog(variance=("centered", "moments"))
  assert len(both) == 1152
  assert len({case.key for case in both}) == 1152
  assert {case.variance for case in both if case.operation == "layernorm"} == {"centered", "moments"}


def test_any_catalog_pair_reuse_preserves_const_shape_specialization(catalog_module):
  cases = catalog_module.catalog()
  any_cases = [case for case in cases if case.meta == "Any"]
  groups = {}
  for case in any_cases:
    groups.setdefault(case.schema_key, []).append(case)
  assert any(len({case.shape for case in group}) > 1 for group in groups.values())
  for group in groups.values():
    assert len({len(case.shape) for case in group}) == 1
    assert len({(case.operation, case.dtype, case.affine, case.accuracy, case.variance) for case in group}) == 1
  const_cases = [case for case in cases if case.meta == "Const"]
  assert len({case.schema_key for case in const_cases}) == len(const_cases)


def test_catalog_toolchain_defaults_follow_runner_environment(catalog_module, monkeypatch, tmp_path):
  monkeypatch.syspath_prepend(str(catalog_module.ROOT / "benchmarks"))
  monkeypatch.setenv("CC", "/private/toolchain/clang")
  monkeypatch.setenv("CXX", "/private/toolchain/clang++")
  monkeypatch.setenv("VECOPS_TEST_TARGET", "SVE")
  monkeypatch.setattr(
    sys,
    "argv",
    [
      "dsl_norm_catalog.py",
      "--operation",
      "layernorm",
      "--shape",
      "64",
      "--dtype",
      "float32",
      "--build-dir",
      str(tmp_path),
    ],
  )
  observed = {}

  def capture_compiler(**options):
    observed.update(options)
    raise RuntimeError("compiler construction captured without a native build")

  monkeypatch.setattr(catalog_module.vecops, "Compiler", capture_compiler)
  with pytest.raises(RuntimeError, match="construction captured"):
    catalog_module.main()
  assert observed["cc"] == "/private/toolchain/clang"
  assert observed["cxx"] == "/private/toolchain/clang++"
  assert observed["target"] == "SVE"


@pytest.mark.parametrize(
  "arguments",
  [
    ["--shape", "64", "31x47"],
    ["--list", "--shape", "64", "31x47"],
    ["--operation", "layernorm", "--meta", "Any", "Const"],
    ["--operation", "softmax", "--affine", "none", "gamma"],
  ],
)
def test_catalog_rejects_unknown_or_partly_ignored_filter_choices(catalog_module, monkeypatch, arguments):
  monkeypatch.syspath_prepend(str(catalog_module.ROOT / "benchmarks"))
  monkeypatch.setattr(sys, "argv", ["dsl_norm_catalog.py", *arguments])
  with pytest.raises(SystemExit) as error:
    catalog_module.main()
  assert error.value.code == 2


def test_catalog_empty_listing_and_explicit_zero_work_are_not_execution(catalog_module, monkeypatch, capsys):
  monkeypatch.syspath_prepend(str(catalog_module.ROOT / "benchmarks"))

  def unexpected_compile(**kwargs):
    pytest.fail("empty listing/partition must not construct a compiler")

  monkeypatch.setattr(catalog_module.vecops, "Compiler", unexpected_compile)
  monkeypatch.setattr(sys, "argv", ["dsl_norm_catalog.py", "--list", "--operation", "layernorm", "--meta", "Const"])
  catalog_module.main()
  assert len(capsys.readouterr().out.splitlines()) == 1  # Known but empty filter combination: CSV header only.
  monkeypatch.setattr(sys, "argv", ["dsl_norm_catalog.py", "--limit", "0"])
  catalog_module.main()
  assert "0 executions" in capsys.readouterr().out


def test_catalog_collects_representative_schema_and_shares_cpp_variance(catalog_module, tmp_path):
  compiler = make_compiler(jobs=2, cache_dir=tmp_path / "cache", build_dir=tmp_path / "build")
  cases = [
    catalog_module.Case("layernorm", shape, "float32", "Any", "none", "strict", variance)
    for shape in [(7,), (64,)]
    for variance in ["centered", "moments"]
  ]
  with vecops.collect_compile_requests() as collector:
    prepared = catalog_module.collect_kernel_pairs(compiler, cases)
  assert len(prepared) == 4
  assert len(collector) == 3  # Two DSL algorithms, one identical C++ reference.
  assert prepared[0][1][0] is prepared[2][1][0]
  assert prepared[1][1][0] is prepared[3][1][0]
  assert all(pair[1] is prepared[0][1][1] for _, pair in prepared)


def test_assembly_loop_parser_keeps_function_addresses_separate():
  path = Path(__file__).parents[2] / "benchmarks/dsl_norm_inspect.py"
  spec = importlib.util.spec_from_file_location("vecops_test_norm_inspect", path)
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  assembly = """
0000000000000000 <sve_row>:
  0: ld1w { z0.s }, p0/z, [x0]
  4: fadd z0.s, p0/m, z0.s, z1.s
  8: cbnz x1, 0 <sve_row>
0000000000000000 <x86_row>:
  0: vaddps %zmm0, %zmm1, %zmm2
  4: jne 0 <x86_row>
"""
  loops = module.backward_loops(assembly)
  assert [(function, len(body)) for function, _, _, body in loops] == [("sve_row", 3), ("x86_row", 2)]


@pytest.mark.parametrize("rank", [1, 2])
@pytest.mark.parametrize("accuracy", ["strict", "fast", "estimate"])
@pytest.mark.parametrize("unroll", [1, 2])
def test_softmax_accuracy_propagates_to_exp_and_reciprocal(rank, accuracy, unroll):
  """Strict uses typed division; the other inverse tiers retain native rcp."""
  shape = (37,) if rank == 1 else (2, 37)

  @T.template
  def kernel(A: T.Tensor(shape, dtype="float32"), C: T.Tensor(shape, dtype="float32", access="output")):
    T.store(C, T.softmax(T.load(A), accuracy=accuracy, unroll=unroll))

  artifact = CPUBackend().lower(kernel.ir, (), target="SVE")
  source = artifact.source
  expressions = [node for node in nodes(artifact.kernel.body) if isinstance(node, I.Expr)]
  reductions = [
    region for region in artifact.lowered.regions if any(effect.kind == "accumulate" for effect in region.effects)
  ]
  assert reductions
  assert {region.unroll for region in reductions} == {unroll}
  assert f"vecops::vec::opt::math::{accuracy}" in source
  assert "vecops::vec::exp_neg" in source
  assert not any(node.op == "fma" for node in expressions)
  if accuracy == "strict":
    divisions = [
      node
      for node in expressions
      if node.op == "div" and isinstance(node.type, I.ScalarType) and node.type.dtype == "float32"
    ]
    assert any(
      node.args[0].type.dtype == "float32" and node.args[0].op == "cast" and I.constant(node.args[0].args[0]) == 1
      for node in divisions
    )
    assert not any(node.op == "rcp" for node in expressions)
    assert "vecops::vec::rcp" not in source
    assert " / " in source
  else:
    assert "vecops::vec::rcp" in source
    assert any(node.op == "rcp" and node.data == accuracy for node in expressions)
    assert "vecops::vec::opt::math::strict" not in source


@pytest.mark.parametrize("control", ["none", "where", "and", "or", "where-condition", "shared"])
def test_reuse_distinguishes_vector_select_from_scalar_control(control):
  """A vector select is pure; a scalar branch may skip its reduction."""
  view = I.View("A", "float32", (I.literal(137),), (I.literal(1),), I.literal(0))
  x = I.Expr(I.TileType("float32", view.shape), "load", data=view)
  exponential = I.Expr(x.type, "exp_neg", (x,), "estimate")
  # This pure vector select must not disable reuse of the reduction input.
  source = I.Expr(x.type, "where", (I.binary("gt", x, I.literal(-80)), exponential, I.literal(0, "float32")))
  reduction = I.Expr(I.ScalarType("float32"), "reduce_add", (source,), 2)
  flag = I.Expr(I.ScalarType("bool"), "raw", data="enabled")
  zero = I.literal(0, "float32")
  value = reduction
  if control in {"where", "shared"}:
    value = I.Expr(reduction.type, "where", (flag, reduction, zero))
    if control == "shared":
      value = I.binary("add", value, reduction)
  elif control in {"and", "or"}:
    value = I.binary(control, flag, I.binary("gt", reduction, zero))
  elif control == "where-condition":
    value = I.Expr(reduction.type, "where", (I.binary("gt", reduction, zero), I.literal(1, "float32"), zero))
  output = I.View("C", "float32", view.shape, view.strides, view.offset, writable=True)
  ir = I.KernelIR(I.Block([I.Assign("statistic", value), I.Store(output, source)]), {}, {})
  context = SimpleNamespace(environment=freeze({}, target="SVE"))
  result = plan_computation_reuse(ir, context)
  assert (id(source) in result.materializations) == (control in {"none", "where-condition", "shared"})


@pytest.mark.parametrize("rank", [1, 2])
def test_float64_estimate_softmax_retains_native_exp(rank):
  shape = (37,) if rank == 1 else (2, 37)

  @T.template
  def kernel(A: T.Tensor(shape, dtype="float64"), C: T.Tensor(shape, dtype="float64", access="output")):
    T.store(C, T.softmax(T.load(A), accuracy="estimate"))

  source = CPUBackend().lower(kernel.ir, (), target="SVE").source
  assert "vecops::vec::exp" in source
  assert "vecops::vec::opt::math::estimate" in source


def test_estimate_exponential_native_special_values(tmp_path):
  """Native Estimate retains NaNs and the vec output-flush contract."""
  shape = (137,)

  @T.template
  def kernel(A: T.Tensor(shape, dtype="float32"), C: T.Tensor(shape, dtype="float32", access="output")):
    T.store(C, T.exp_neg(T.load(A), accuracy="estimate"))

  compiler = make_compiler(jobs=2, cache_dir=tmp_path / "cache", build_dir=tmp_path / "build")
  compiled = T.compile(kernel, compiler=compiler)
  inputs = np.resize(
    np.array([float("-inf"), -100, -81, -80.00001, -80, -79.99999, -10, -1, -0.0, 0, float("nan")], np.float32),
    shape,
  )
  output = np.empty_like(inputs)
  compiled(inputs, output)
  expected = np.exp(inputs.astype(np.float64)).astype(np.float32)
  expected[expected < np.finfo(np.float32).tiny] = 0
  np.testing.assert_allclose(output, expected, rtol=0.006, atol=0, equal_nan=True)
  np.testing.assert_array_equal(output[inputs < -90], 0)
  assert np.isnan(output[np.isnan(inputs)]).all()
  source = compiled.generated_source
  assert "vecops::vec::exp_neg" in source
  assert "std::exp" not in source
  assert "std::array<vecops::vec::Vec" not in source


@pytest.mark.parametrize("operation", ["rsqrt", "rcp", "exp", "exp_neg", "exp2", "exp2_neg", "log", "log2"])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("accuracy", ["strict", "fast", "estimate"])
def test_scalar_math_uses_native_math_tier(operation, dtype, accuracy):
  operation_fn = getattr(T, operation)

  @T.template
  def kernel(A: T.Tensor((37,), dtype=dtype), C: T.Tensor((), dtype=dtype, access="output")):
    total = T.reduce_add(T.load(A))
    inverse = operation_fn(total, accuracy=accuracy)
    T.store(C, inverse)

  source = CPUBackend().lower(kernel.ir, (), target="SVE").source
  assert f"vecops::vec::{operation}" in source
  assert f"vecops::vec::opt::math::{accuracy}" in source
  assert "std::sqrt" not in source


@pytest.mark.parametrize("operation", ["sqrt", "abs"])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_scalar_exact_unary_math_uses_native_signature(tmp_path, operation, dtype):
  # sqrt/abs have no accuracy option. Compile and execute the scalar call to
  # catch accidental propagation of the approximate-math option pack.
  operation_fn = getattr(T, operation)

  @T.template
  def kernel(A: T.Tensor((37,), dtype=dtype), C: T.Tensor((1,), dtype=dtype, access="output")):
    T.store(C[0], operation_fn(T.reduce_add(T.load(A))))

  compiler = make_compiler(jobs=2, cache_dir=tmp_path / "cache", build_dir=tmp_path / "build")
  compiled = T.compile(kernel, compiler=compiler)
  inputs = np.full((37,), -1 if operation == "abs" else 1, dtype=dtype)
  output = np.empty((1,), dtype=dtype)
  compiled(inputs, output)
  np.testing.assert_allclose(output, 37 if operation == "abs" else np.sqrt(37), rtol=1e-6)
  assert f"vecops::vec::{operation}" in compiled.generated_source
  assert "vecops::vec::opt::math::" not in compiled.generated_source


@pytest.mark.parametrize("rank", [1, 2, 3, 4])
def test_catalog_dynamic_shapes_and_captures(catalog_module, tmp_path, rank):
  case = next(
    case
    for case in catalog_module.catalog()
    if case.operation == "layernorm"
    and len(case.shape) == rank
    and case.dtype == "float32"
    and case.affine == "gamma_beta"
  )
  compiler = make_compiler(jobs=1, cache_dir=tmp_path / "cache", build_dir=tmp_path / "build")
  with vecops.collect_compile_requests() as collector:
    pair = catalog_module.kernels(compiler, case)
    for kernel in pair:
      kernel.operator(*catalog_module.metadata(case), 1, **kernel._core.specialization_values)
  assert len(collector) == 2
  assert "vecops::vec::reduce_add" in pair[0].generated_source
  assert "vecops::ops::" not in pair[0].generated_source
  captures = pair[1]._core.specialization_values
  assert captures["Op"] == 1
  assert captures["Accuracy"] == 0
  assert captures["HasGamma"] == 1
  assert captures["HasBeta"] == 1
