"""Direct rank-two library coverage, distinct from row-wise benchmark wrappers."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import vecops
import vecops.language as T
from dsl_test_utils import assert_no_vec_mask_arrays, make_compiler
from vecops.language._compiler.backends.cpu.backend import CPUBackend

ALGORITHMS = (("layernorm", "centered"), ("layernorm", "moments"), ("softmax", "n/a"))
DTYPES = ("float32", "float64", "float16", "bfloat16")


def direct_template(operation, variance="centered", dtype="float32"):
  M, N = T.dynamic(lower=0), T.dynamic(lower=0)
  AS0, AS1, CS0, CS1 = (T.dynamic() for _ in range(4))
  ComputeType = T.DType("float64" if dtype == "float64" else "float32")

  @T.template
  def kernel(
    A: T.Tensor((M, N), (AS0, AS1), dtype),
    C: T.Tensor((M, N), (CS0, CS1), dtype, access="output"),
  ):
    x = T.load(A, dtype=ComputeType)
    if operation == "layernorm":
      y = T.library.layernorm(x, variance=variance)
    else:
      y = T.library.softmax(x)
    T.store(C, y)

  return kernel


@pytest.fixture(scope="module")
def library_benchmark():
  path = Path(__file__).parents[2] / "benchmarks/dsl_norm_library.py"
  spec = importlib.util.spec_from_file_location("vecops_test_norm_library_benchmark", path)
  module = importlib.util.module_from_spec(spec)
  # Keep imports as the command-line benchmark sees them, without requiring
  # the repository's benchmarks directory to become a product package.
  previous = list(sys.path)
  sys.path.insert(0, str(path.parent))
  sys.modules[spec.name] = module
  try:
    spec.loader.exec_module(module)
    yield module
  finally:
    sys.path[:] = previous
    sys.modules.pop(spec.name, None)


def test_library_catalog_is_explicitly_direct_rank_two(library_benchmark):
  cases = library_benchmark.catalog(variance=("centered", "moments"))
  assert cases
  assert {len(case.shape) for case in cases} == {2}
  assert {case.operation for case in cases} == {"layernorm", "softmax"}
  assert library_benchmark.SCOPE == "direct-rank2-library"
  assert "library_over_row" in library_benchmark.FIELDS


@pytest.mark.parametrize("arguments", (["--list", "--shape", "64"], ["--list", "--shape", "8x128", "31x47"]))
def test_direct_library_catalog_rejects_unsupported_rank_and_unknown_choices(library_benchmark, monkeypatch, arguments):
  monkeypatch.setattr(sys, "argv", ["dsl_norm_library.py", *arguments])
  with pytest.raises(SystemExit) as error:
    library_benchmark.main()
  assert error.value.code == 2


def test_direct_library_catalog_labels_empty_runs_without_compilation(library_benchmark, monkeypatch, capsys):
  def unexpected_compile(**kwargs):
    pytest.fail("an empty direct-library selection must not construct a compiler")

  monkeypatch.setattr(library_benchmark.vecops, "Compiler", unexpected_compile)
  monkeypatch.setattr(sys, "argv", ["dsl_norm_library.py", "--limit", "0"])
  library_benchmark.main()
  assert "0 executions" in capsys.readouterr().out


def test_library_benchmark_collects_full_row_and_cpp_independently(library_benchmark, tmp_path):
  compiler = make_compiler(jobs=2, cache_dir=tmp_path / "cache", build_dir=tmp_path / "build")
  from dsl_norm_catalog import Case

  cases = [Case("layernorm", shape, "float32", "Any", "none", "strict", "moments") for shape in ((8, 128), (3, 37))]
  with vecops.collect_compile_requests() as collector:
    prepared = library_benchmark.collect_kernel_triples(compiler, cases)
  assert len(collector) == 3  # Direct rank-two, row rank-one, and C++.
  assert len(prepared) == 2
  assert all(first is second for first, second in zip(prepared[0][1], prepared[1][1]))
  full_source = prepared[0][1][0].generated_source
  row_source = prepared[0][1][1].generated_source
  assert "dsl_o_norm_" in full_source
  assert "dsl_o_norm_" not in row_source
  assert_no_vec_mask_arrays(full_source)


@pytest.mark.parametrize("operation", ("layernorm", "softmax"))
@pytest.mark.parametrize("rank", (3, 4))
def test_current_higher_rank_library_limit_is_a_located_diagnostic(operation, rank):
  shape = (2,) * (rank - 1) + (37,)

  @T.template
  def kernel(A: T.Tensor(shape, dtype="float32"), C: T.Tensor(shape, dtype="float32", access="output")):
    x = T.load(A)
    if operation == "layernorm":
      T.store(C, T.library.layernorm(x))
    else:
      T.store(C, T.library.softmax(x))

  with pytest.raises(TypeError, match=r"library.py:\d+:.*static_assert"):
    CPUBackend().lower(kernel.ir, (), target="SVE")


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(("operation", "variance"), ALGORITHMS)
def test_direct_library_codegen_preserves_vector_and_accuracy_contracts(dtype, operation, variance):
  source = CPUBackend().lower(direct_template(operation, variance, dtype).ir, (), target="SVE").source
  assert "dsl_o_norm_" in source
  assert "vecops::ops::" not in source
  assert "vecops::vec::opt::math::strict" in source
  assert_no_vec_mask_arrays(source)
  if operation == "layernorm":
    assert "inv_count" in source
    assert "vecops::vec::rsqrt" in source
  else:
    assert "vecops::vec::exp" in source


def test_layernorm_full_reduction_omits_inactive_lane_mask_but_keeps_tail():
  N = T.dynamic(lower=0)

  @T.template
  def kernel(A: T.Tensor((N,), dtype="float32"), C: T.Tensor((N,), dtype="float32", access="output")):
    T.store(C, T.library.layernorm(T.load(A), variance="moments"))

  source = CPUBackend().lower(kernel.ir, (), target="SVE").source
  # Inspect the first legal reduction branch, not another ISA fallback or
  # the later output loop. The partial tail must retain both identities.
  full = source.split("/* full vector groups */", 1)[1].split("/* masked vector tail */", 1)[0]
  assert "vecops::vec::blend" not in full
  assert "vecops::vec::mwhilelt" not in full
  tail = source.split("/* masked vector tail */", 1)[1].split("#else", 1)[0]
  assert "vecops::vec::blend" in tail
  assert "VECOPS_ALWAYS_INLINE Span span" in source
  assert "VECOPS_ALWAYS_INLINE bool overlap" in source


@pytest.fixture(scope="module")
def direct_native_kernels(tmp_path_factory):
  path = tmp_path_factory.mktemp("direct-library-normalization")
  compiler = make_compiler(jobs=4, cache_dir=path / "cache", build_dir=path / "build")
  kernels = {}
  with vecops.collect_compile_requests() as collector:
    for dtype in DTYPES:
      if dtype == "bfloat16" and importlib.util.find_spec("ml_dtypes") is None:
        continue
      for operation, variance in ALGORITHMS:
        kernel = T.compile(direct_template(operation, variance, dtype), compiler=compiler)
        kernel.operator(
          vecops.TensorMeta((3, 37), dtype=dtype),
          vecops.TensorMeta((3, 37), dtype=dtype),
          **kernel._core.specialization_values,
        )
        kernels[dtype, operation, variance] = kernel
  vecops.compile_batch(collector.requests, parallelism=4, execution_parallelism=1)
  return kernels


def expected_normalization(inputs, operation):
  values = inputs.astype(np.float64)
  if operation == "layernorm":
    centered = values - values.mean(axis=-1, keepdims=True)
    return centered / np.sqrt((centered * centered).mean(axis=-1, keepdims=True) + 1e-5)
  exponential = np.exp(values - values.max(axis=-1, keepdims=True))
  return exponential / exponential.sum(axis=-1, keepdims=True)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(("operation", "variance"), ALGORITHMS)
def test_direct_native_library_dynamic_layouts_and_aliases(direct_native_kernels, dtype, operation, variance):
  if dtype == "bfloat16":
    array_dtype = pytest.importorskip("ml_dtypes").bfloat16
  else:
    array_dtype = np.dtype(dtype)
  kernel = direct_native_kernels[dtype, operation, variance]
  rng = np.random.default_rng(31)
  tolerance = {"float32": 3e-5, "float64": 2e-8, "float16": 2e-3, "bfloat16": 8e-3}[dtype]
  for width in (0, 1, 7, 17, 65, 257):
    for layout in ("dense", "strided", "transposed", "negative"):
      if layout == "strided":
        inputs = rng.normal(size=(3, width * 2)).astype(array_dtype)[:, ::2]
      elif layout == "transposed":
        inputs = rng.normal(size=(width, 3)).astype(array_dtype).T
      else:
        inputs = rng.normal(size=(3, width)).astype(array_dtype)
        if layout == "negative":
          inputs = inputs[:, ::-1]
      backing = np.full((width * 2, 3), -123, dtype=array_dtype)
      output = backing[::2, :].T
      kernel(inputs, output)
      if width:
        np.testing.assert_allclose(
          output.astype(np.float64),
          expected_normalization(inputs, operation),
          rtol=tolerance,
          atol=tolerance,
          err_msg=f"direct library {operation}/{variance}/{dtype}: {width=}, {layout=}",
        )
      assert np.all(backing[1::2] == -123)
  for alias in ("inplace", "shifted", "negative"):
    storage = rng.normal(size=3 * 37 + 1).astype(array_dtype)
    inputs = storage[:-1].reshape(3, 37)
    expected = expected_normalization(inputs, operation)
    output = inputs if alias == "inplace" else (inputs[:, ::-1] if alias == "negative" else storage[1:].reshape(3, 37))
    kernel(inputs, output)
    np.testing.assert_allclose(
      output.astype(np.float64), expected, rtol=tolerance, atol=tolerance, err_msg=f"direct {alias=}"
    )
  assert_no_vec_mask_arrays(kernel.generated_source)
