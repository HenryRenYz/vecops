"""Explicit register initialization, dynamic prefixes and carried geometry."""

import numpy as np
import pytest
import vecops.language as T
from dsl_test_utils import assert_no_vec_mask_arrays, make_compiler
from vecops.language._compiler.backends.cpu.backend import CPUBackend
from vecops.language._compiler.frontend.parser import lower


def template(dtype="float32", *, fixed=False, power=0):
  descriptor = T.Vector(dtype, N=16) if fixed else T.Vector(dtype, pow=power)
  size = T.dynamic(lower=0)

  @T.template
  def kernel(C: T.Tensor((size,), dtype=dtype, access="output"), count: int = 3):
    value = T.full(descriptor, 2, active=C.shape[0])
    for _ in T.serial(count):
      value = value + 1
    T.store(C, value)

  return kernel


@pytest.mark.parametrize("power", [-1, 0, 2, 5])
def test_vector_fill_retains_geometry_and_has_no_dummy_memory_load(power):
  artifact = CPUBackend().lower(template(power=power).ir, (), target="AVX512")
  from vecops.language._compiler import ir as I
  from vecops.language._compiler.ir.traversal import nodes

  fills = [node for node in nodes(artifact.lowered.kernel.body) if isinstance(node, I.Expr) and node.op == "fill"]
  assert any(isinstance(value.type, I.VectorType) and value.type.tag.pow == power for value in fills)
  assert not any(isinstance(node, I.Expr) and node.op == "load" for node in nodes(artifact.lowered.kernel.body))
  assert_no_vec_mask_arrays(artifact.source)
  if power <= 2:
    assert "load_convert(" not in artifact.source
    assert all(plan.pieces for plan in artifact.backend_ir.states.values())


@pytest.mark.parametrize("case", ["conflicting_dtype", "invalid_active", "tile_active", "boolean_active", "tile_fill"])
def test_vector_fill_rejects_invalid_geometry(case):
  @T.template
  def kernel():
    if case == "conflicting_dtype":
      T.zeros(T.Vector("float32", N=16), dtype="float64")
    elif case == "invalid_active":
      T.full(T.Vector("float32", N=16), 1, active=17)
    elif case == "tile_active":
      T.zeros((16,), active=3)
    elif case == "boolean_active":
      T.zeros(T.Vector("float32", N=16), active=True)
    else:
      T.full(T.Vector("float32", N=16), T.zeros((16,)))

  with pytest.raises(TypeError, match="dtype conflicts|active prefix|scalar fill"):
    lower(kernel.ir)


@pytest.fixture(scope="module")
def compiler(tmp_path_factory):
  root = tmp_path_factory.mktemp("vector-fills")
  return make_compiler(jobs=4, cache_dir=root / "cache", build_dir=root / "build")


@pytest.mark.parametrize("dtype", ["float32", "float16", "float64", "int32"])
@pytest.mark.parametrize("fixed", [False, True])
def test_native_vector_fill_prefix_carried_state_and_canaries(compiler, dtype, fixed):
  kernel = T.compile(template(dtype, fixed=fixed), compiler=compiler)
  for size in (0, 1):
    backing = np.full(size + 8, -19, dtype=dtype)
    output = backing[3 : 3 + size]
    for count in (0, 1, 3):
      kernel.operator.native(output, count, **kernel._core.specialization_values)
      np.testing.assert_array_equal(output, np.full(size, 2 + count, dtype=dtype))
      np.testing.assert_array_equal(backing[:3], np.full(3, -19, dtype=dtype))
      np.testing.assert_array_equal(backing[3 + size :], np.full(5, -19, dtype=dtype))
  # Exceeds any supported native length/power, without guessing this process's
  # SVE width. Failure must precede memory effects and leave canaries intact.
  oversized = np.full(1024, -19, dtype=dtype)
  with pytest.raises((RuntimeError, ValueError), match="active prefix"):
    kernel.operator.native(oversized, 3, **kernel._core.specialization_values)
  np.testing.assert_array_equal(oversized, np.full(1024, -19, dtype=dtype))


@pytest.mark.parametrize("fma", [False, True])
def test_tile_first_update_retains_accumulator_vector_tag(compiler, fma):
  @T.template
  def kernel(A: T.Tensor((16,), dtype="float32"), C: T.Tensor((16,), dtype="float32", access="output")):
    value = T.full(T.Vector("float32", N=16), 2)
    for _ in T.serial(3):
      if fma:
        value = T.fma(T.load(A), 0.5, value)
      else:
        value = T.load(A) + value
    T.store(C, value)

  compiled = T.compile(kernel, compiler=compiler)
  a = np.arange(16, dtype=np.float32) / 16
  c = np.empty_like(a)
  compiled.operator.native(a, c, **compiled._core.specialization_values)
  np.testing.assert_array_equal(c, 2 + a * (1.5 if fma else 3))
  assert_no_vec_mask_arrays(compiled.generated_source)


def test_widened_scalable_state_uses_one_word_per_piece(compiler):
  @T.template
  def kernel(C: T.Tensor((1,), dtype="float32", access="output")):
    value = T.full(T.Vector("float32"), 2, active=1)
    for _ in T.serial(3):
      value = value + T.cast(T.cast(value, "float64") * 0.5, "float32")
    T.store(C, value)

  artifact = CPUBackend().lower(kernel.ir, (), target="AVX512")
  assert all(len(plan.pieces) == 2 for plan in artifact.backend_ir.states.values())
  assert not any(piece.power > 0 for plan in artifact.backend_ir.states.values() for piece in plan.pieces)
  assert "state_read" not in artifact.source  # equivalent rebinds share direct register references
  assert_no_vec_mask_arrays(artifact.source)
  output = np.empty(1, np.float32)
  compiled = T.compile(kernel, compiler=compiler)
  compiled.operator.native(output, **compiled._core.specialization_values)
  np.testing.assert_array_equal(output, np.array([6.75], np.float32))


def test_as_vector_clip_convert_preserves_historical_load(compiler):
  @T.template
  def kernel(
    A: T.Tensor((37,), dtype="float16", access="inout"),
    C: T.Tensor((16,), dtype="float32", access="output"),
  ):
    old = T.load(A) * 2 + 1
    T.store(A, T.full((37,), 99, dtype="float16"))
    T.store(C, T.as_vector(old, T.Vector("float32", N=16)))

  compiled = T.compile(kernel, compiler=compiler)
  a = np.arange(37, dtype=np.float16) / 16
  expected = (a * 2 + 1).astype(np.float16)[:16].astype(np.float32)
  c = np.empty(16, np.float32)
  compiled.operator.native(a, c, **compiled._core.specialization_values)
  np.testing.assert_array_equal(c, expected)
  np.testing.assert_array_equal(a, np.full(37, 99, np.float16))
  assert_no_vec_mask_arrays(compiled.generated_source)
