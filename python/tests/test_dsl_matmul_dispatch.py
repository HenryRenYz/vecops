"""Runtime family selection, full/tail K and target-selected packed formats."""

import numpy as np
import pytest
import vecops.language as T
from dsl_test_utils import make_compiler
from vecops._arm_vector_state import VectorState
from vecops.language._compiler.backends.cpu.backend import CPUBackend


@pytest.fixture(scope="module")
def compiler(tmp_path_factory):
  root = tmp_path_factory.mktemp("dsl-matmul-dispatch")
  return make_compiler(target="Native", jobs=6, cache_dir=root / "cache", build_dir=root / "build")


def _template(kind, implementation):
  M, N, K = (T.dynamic(lower=0) for _ in range(3))
  ad, bd, acc = ("bfloat16", "bfloat16", "float32") if kind == "bf16" else ("uint8", "int8", "int32")

  @T.template
  def kernel(
    A: T.Tensor((M, K), (K, 1), ad),
    B: T.Tensor((K, N), (1, K), bd),
    D: T.Tensor((N,), dtype=acc),
    C: T.Tensor((M, N), dtype="float32", access="output"),
  ):
    a = T.pack(T.cast(T.load(A, dtype="float32") * 2 + 1, ad), side="A")
    b = T.pack(T.cast(T.load(B, dtype="float32") * 2 + 1, bd), side="B")
    value = T.dot((a, b), C, acc=T.load(D), implementation=implementation)
    T.store(C, T.cast(value, "float32") * 0.5 + 1)

  return kernel


def _assert_packed_auto_route(source, architecture):
  if architecture == "arm":
    # Auto can choose an ordinary-SVE MMLA leaf *before* its SME fallback.
    # Both packed SME operands already have zero-filled K padding, unlike
    # raw SME operands. A separate raw partial-K branch is not required.
    if "/* ordinary-SVE PackedDot MMLA */" in source:
      assert "/* MMLA two complete K groups */" in source
      assert "/* MMLA odd group: no second-group read */" in source
      assert "sve::mmla(" in source
    elif "sme::mopa<" in source:
      assert "sme_packed_k_cursor" in source
      assert "sme_packed_k_step" in source
    else:
      assert "vecops::vec::widening_dot(" in source, "unsupported matrix geometry must retain the vector fallback"
  elif architecture == "x86":
    assert "full K: unmasked load/convert" in source, "AMX auto must retain its unmasked full-K route"


@pytest.mark.parametrize("kind", ["bf16", "mixed"])
@pytest.mark.parametrize("implementation", ["auto", "vector"])
def test_runtime_families_and_auto_packed_vector_fallback(compiler, kind, implementation):
  kernel = _template(kind, implementation)
  ad, bd, acc = ("bfloat16", "bfloat16", "float32") if kind == "bf16" else ("uint8", "int8", "int32")
  compiled = T.compile(kernel, compiler=compiler)
  import ml_dtypes

  memory = lambda dtype: ml_dtypes.bfloat16 if dtype == "bfloat16" else np.dtype(dtype)
  rng = np.random.default_rng(21)
  for m, n, k in ((8, 1, 129), (1, 19, 257), (19, 23, 65), (16, 48, 128), (33, 65, 257)):
    a = rng.integers(0 if ad == "uint8" else -3, 4, size=(m, k)).astype(memory(ad))
    b = rng.integers(-3, 4, size=(n, k)).astype(memory(bd)).T
    d = rng.integers(-3, 4, size=n).astype(acc)
    c = np.empty((m, n), np.float32)
    compiled.operator.native(a, b, d, c, **compiled._core.specialization_values)
    x = (a.astype(np.float32) * 2 + 1).astype(memory(ad)).astype(acc)
    y = (b.astype(np.float32) * 2 + 1).astype(memory(bd)).astype(acc)
    np.testing.assert_allclose(c, (x @ y + d).astype(np.float32) * 0.5 + 1, rtol=2e-5, atol=2e-5)
  if implementation == "auto":
    environment = compiled.vector_ir.manager.options["_compiler_environment"]
    _assert_packed_auto_route(compiled.generated_source, environment.codegen_target.architecture)


@pytest.mark.parametrize("kind", ("bf16", "mixed"))
@pytest.mark.parametrize("pipeline", ("off", "prefetch", "staged"))
@pytest.mark.parametrize("vl,svl", ((16, 16), (64, 64), (16, 64)))
def test_arm_auto_packed_route_assertions_match_actual_lowering(kind, pipeline, vl, svl):
  artifact = CPUBackend().lower(
    _template(kind, "auto").ir,
    (),
    target="SME",
    options={"cpu_matmul_pipeline": pipeline, "_vector_state": VectorState(vl, svl)},
  )
  _assert_packed_auto_route(artifact.source, "arm")
