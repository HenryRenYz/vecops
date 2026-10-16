# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Fixed-prefix register reads preserve transpose tails and shifted state."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import vecops
import vecops.language as T
from dsl_state_consumer_utils import assert_closed_store_consumers, state_consumers
from dsl_test_utils import make_compiler, x86_target_available_for_test
from vecops.language._compiler import ir as I
from vecops.language._compiler.backends.cpu.backend import CPUBackend
from vecops.language._compiler.backends.cpu.cpp.state import StateEmitter
from vecops.language._compiler.backends.cpu.state import StatePiece, StatePlan, state_view

DTYPES = ("float64", "float32", "int8")
SHAPES = ((16, 16), (17, 19))


def transpose_template(dtype, shape):
  rows, columns = shape

  @T.template
  def kernel(
    A: T.Tensor((rows, columns), dtype=dtype),
    C: T.Tensor((columns, rows), dtype=dtype, access="output"),
  ):
    T.transpose_copy(A, C, implementation="vector")

  return kernel


def shifted_masked_template(dtype):
  @T.template
  def kernel(A: T.Tensor((17,), dtype=dtype), C: T.Tensor((17,), dtype=dtype, access="output")):
    value = T.load(A)
    for iteration in T.serial(2):
      value = value + 1
    T.store(C[1:17], value[1:17], mask=value[1:17] > 0)

  return kernel


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
def test_x86_transpose_uses_a_checked_fixed_register_prefix(dtype, shape):
  with state_consumers() as records:
    artifact = CPUBackend().lower(transpose_template(dtype, shape).ir, (), target="AVX512")
  assert_closed_store_consumers(artifact.backend_ir, artifact.source, records, efficient=True)


def test_arm_transpose_never_reinterprets_a_sizeless_register_as_a_fixed_vector():
  source = CPUBackend().lower(transpose_template("float64", (17, 19)).ir, (), target="SVE").source
  assert "/* fixed register prefix */" not in source


@pytest.mark.parametrize(
  "boundary",
  (
    "exact-prefix",
    "requested-dtype",
    "state-dtype",
    "predicate",
    "multiple-pieces",
    "unknown-width",
    "arm",
    "shifted-origin",
    "partial-active",
    "scaled-piece",
    "oversized-prefix",
    "fixed-piece",
    "different-axis",
    "dynamic-length",
  ),
)
def test_fixed_prefix_eligibility_preserves_projection_fallbacks(boundary):
  state = I.TileState("saved", I.TileType("float64", (I.literal(2),)))
  value = I.Expr(state.type, "state_ref", data=state)
  piece = StatePiece((0,), 0, scalable=True, power=0, full=True)
  plan = StatePlan(state, state_view(state), (piece,))
  tag = "vecops::vec::FixedTag<double, 2>"
  target = SimpleNamespace(architecture="x86", vector_bits=512)
  wanted, requested_dtype, coordinates, active, axis = 2, "float64", ("0",), "2", 0
  if boundary == "requested-dtype":
    requested_dtype = "float32"
  elif boundary == "state-dtype":
    plan = replace(plan, state=replace(state, type=replace(state.type, dtype="float32")))
  elif boundary == "predicate":
    value = replace(value, type=replace(value.type, mask=True))
  elif boundary == "multiple-pieces":
    plan = replace(plan, pieces=(piece, piece))
  elif boundary == "unknown-width":
    target.vector_bits = None
  elif boundary == "arm":
    target.architecture = "arm"
  elif boundary == "shifted-origin":
    coordinates = ("1",)
  elif boundary == "partial-active":
    active = "1"
  elif boundary == "scaled-piece":
    plan = replace(plan, pieces=(replace(piece, power=1),))
  elif boundary == "oversized-prefix":
    wanted, active = 16, "16"
    value = replace(value, type=replace(value.type, shape=(I.literal(16),)))
  elif boundary == "fixed-piece":
    plan = replace(plan, pieces=(replace(piece, scalable=False),))
  elif boundary == "different-axis":
    axis = 1
  elif boundary == "dynamic-length":
    value = replace(value, type=replace(value.type, shape=(I.Expr(I.ScalarType("int64"), "raw", data="length"),)))
  emitter = StateEmitter()
  emitter.environment = SimpleNamespace(codegen_target=target)
  emitter.fixed_tag_lanes = {tag: wanted}
  emitter.tag_dtypes = {tag: requested_dtype}
  lines = []
  emitter.line = lines.append
  result = emitter.fixed_register_prefix(plan, value, tag, value.type.shape, coordinates, active, axis)
  if boundary == "exact-prefix":
    assert "/* fixed register prefix */" in result
    assert len(lines) == 1 and "static_assert" in lines[0]
  else:
    assert result is None
    assert not lines


@pytest.fixture(scope="module")
def native_kernels(tmp_path_factory):
  root = tmp_path_factory.mktemp("state-register-prefix")
  compiler = make_compiler(jobs=2, cache_dir=root / "cache", build_dir=root / "build")
  kernels = {}
  with vecops.collect_compile_requests() as collector:
    for dtype in DTYPES:
      for shape in SHAPES:
        with state_consumers() as records:
          compiled = T.compile(transpose_template(dtype, shape), compiler=compiler)
        kernels["proof", dtype, shape] = records
        compiled.operator(
          vecops.TensorMeta(shape, dtype=dtype),
          vecops.TensorMeta(shape[::-1], dtype=dtype),
          **compiled._core.specialization_values,
        )
        kernels[dtype, shape] = compiled
      compiled = T.compile(shifted_masked_template(dtype), compiler=compiler)
      compiled.operator(
        vecops.TensorMeta((17,), dtype=dtype),
        vecops.TensorMeta((17,), dtype=dtype),
        **compiled._core.specialization_values,
      )
      kernels[dtype, "shifted"] = compiled
  vecops.compile_batch(collector.requests, parallelism=2, execution_parallelism=1)
  return kernels


def guarded_array(shape, dtype):
  """Use nonzero origins and retain every byte outside the output view."""
  count = int(np.prod(shape))
  owner = np.empty(count + 38, dtype=dtype)
  array = owner[19 : 19 + count].reshape(shape)
  assert array.ctypes.data != owner.ctypes.data
  return owner, array


@pytest.mark.skipif(not x86_target_available_for_test(), reason="fixed register prefixes target x86")
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
def test_native_transpose_all_lanes_tails_and_owner_guards(native_kernels, dtype, shape):
  compiled = native_kernels[dtype, shape]
  assert_closed_store_consumers(
    compiled.backend_ir, compiled.generated_source, native_kernels["proof", dtype, shape], efficient=True
  )
  input_owner, a = guarded_array(shape, dtype)
  output_owner, c = guarded_array(shape[::-1], dtype)
  input_owner.view(np.uint8).fill(0xA6)
  a[:] = (np.arange(a.size).reshape(shape) % 127 - 63).astype(dtype)
  before = input_owner.view(np.uint8).copy()
  for poison in (0x5A, 0xC3):
    output_owner.view(np.uint8).fill(poison)
    compiled.operator.native(a, c, **compiled._core.specialization_values)
    np.testing.assert_array_equal(c, a.T)
    np.testing.assert_array_equal(input_owner.view(np.uint8), before)
    assert np.all(output_owner[:19].view(np.uint8) == poison)
    assert np.all(output_owner[-19:].view(np.uint8) == poison)


@pytest.mark.skipif(not x86_target_available_for_test(), reason="fixed register prefixes target x86")
@pytest.mark.parametrize("dtype", DTYPES)
def test_native_shifted_state_and_masked_inactive_lanes_remain_correct(native_kernels, dtype):
  compiled = native_kernels[dtype, "shifted"]
  input_owner, a = guarded_array((17,), dtype)
  output_owner, c = guarded_array((17,), dtype)
  input_owner.view(np.uint8).fill(0xA6)
  a[:] = np.arange(-8, 9, dtype=dtype)
  before = input_owner.view(np.uint8).copy()
  for poison in (-37, 41):
    output_owner.fill(poison)
    expected_owner = output_owner.copy()
    expected = expected_owner[19:-19]
    updated = a + np.array(2, dtype=dtype)
    expected[1:] = np.where(updated[1:] > 0, updated[1:], poison)
    compiled.operator.native(a, c, **compiled._core.specialization_values)
    np.testing.assert_array_equal(output_owner, expected_owner)
    np.testing.assert_array_equal(input_owner.view(np.uint8), before)
