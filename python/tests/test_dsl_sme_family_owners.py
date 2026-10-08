# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Dynamic SME dispatch keeps each manual state owner small and self-contained.

The optional real ARM cross tests inspect every emitted state trampoline, not
only the one selected by a tiny constant shape. Target execution/performance is
validated separately: compiler success and source spelling are not inlining
evidence.
"""

import re

import numpy as np
import pytest
import vecops
import vecops.language as T
from arm_sme_asm_utils import assembly_functions, assert_sme_owners
from dsl_test_utils import assert_no_vec_mask_arrays, make_compiler
from test_dsl_matmul_sve_optimized import _cross
from test_dsl_sme_exact_blocks import _body, _cross_wrapper, _sme_available, _template
from vecops._arm_vector_state import VectorState, get_vector_state
from vecops.language._compiler.backends.cpu.backend import CPUBackend
from vecops.language._compiler.backends.cpu.capabilities import CPP_TYPES

_OWNER = (
  r"\w+\.with_resources\(vecops::execution::sme::Region\{\}, "
  r"\[&\]\(auto& \w+\) VECOPS_INLINE_LAMBDA_NOEXCEPT"
)


def _lower(*, vl=64, svl=64, **kwargs):
  return (
    CPUBackend()
    .lower(
      _template(**kwargs).ir,
      (),
      target="SME",
      options={
        "cpu_matmul_pipeline": "off",
        "cpu_matmul_fusion": "always",
        "_vector_state": VectorState(vl, svl),
      },
    )
    .source
  )


@pytest.mark.parametrize("dtype", ("float16", "bfloat16", "float32", "float64", "int8", "uint8"))
@pytest.mark.parametrize("packed", (False, True))
def test_dynamic_geometry_is_selected_before_each_inline_state_owner(dtype, packed):
  output = "float64" if dtype == "float64" else "int32" if dtype in {"int8", "uint8"} else "float32"
  source = _lower(dtype=dtype, output=output, packed=packed, pipeline="affine")
  owners = [_body(source, match[0]) for match in re.finditer(_OWNER, source)]
  expanded = dtype == "float64" and not packed
  assert len(owners) == (11 if expanded else 9)
  schedules = []
  for owner in owners:
    assert ".with_resources(" not in owner
    assert "ZA family */" not in owner  # The family dispatch is outside SM.
    schedules.extend(re.findall(r"/\* SME (\d+x\d+) outer-product schedule", owner))
    for axis in "mnk":
      snapshot = re.search(rf"const vecops::nint_t (\w+_sme_extent_{axis}_value_\d+)=(\w+);", owner)
      assert snapshot is not None
      assert not re.search(rf"\b{snapshot[2]}\b", owner[snapshot.end() :])
    assert "std::min" not in owner and "std::max" not in owner
  assert schedules == ["1x1", "1x2", "1x3", "1x4", "2x1", "3x1", "4x1", "2x2"] + (
    ["4x2", "2x4", "2x2"] if expanded else ["2x2"]
  )
  assert_no_vec_mask_arrays(source)


@pytest.mark.parametrize("vl,svl", ((16, 64), (64, 16), (32, 32)))
def test_owner_geometry_uses_captured_svl_not_ordinary_vl(vl, svl):
  source = _lower(dtype="float16", packed=False, vl=vl, svl=svl)
  lanes = svl // 4  # F32 ZA accumulator lanes; FP16 groups carry two K values.
  declarations = re.findall(r"const vecops::nint_t \w+_sme_matrix_lanes_\d+=([^;]+);", source)
  assert len(declarations) == 10  # Outer dispatch plus nine selected owners.
  assert set(declarations) == {str(lanes)}
  assert "streaming_lanes<" not in _body(source, "void __kernel__")
  assert "clamp_value<vecops::nint_t>" in source


def test_dry_owner_without_environment_keeps_runtime_svl_reader():
  source = _lower(dtype="float16", packed=False, vl=None, svl=None)
  assert "vecops::vec::sme::streaming_lanes<float>()" in source
  assert re.search(
    r"const vecops::nint_t \w+_sme_matrix_lanes_\d+=vecops::vec::sme::streaming_lanes<float>\(\);", source
  )


def _strided_template(dtype, output, *, transformed=False):
  M, N, K = (T.dynamic(lower=0) for _ in range(3))
  A0, A1, B0, B1, D0, D1, C0, C1 = (T.dynamic() for _ in range(8))
  memory = "float32" if transformed else dtype
  acc = "float64" if dtype == "float64" else "int32" if dtype in {"int8", "uint8"} else "float32"

  @T.template
  def kernel(
    A: T.Tensor((M, K), (A0, A1), memory),
    B: T.Tensor((K, N), (B0, B1), dtype),
    D: T.Tensor((M, N), (D0, D1), acc),
    C: T.Tensor((M, N), (C0, C1), output, access="output"),
  ):
    a = T.cast(T.load(A) * 0.5 + 0.125, dtype) if transformed else T.load(A)
    value = T.dot((a, T.load(B)), C, acc=T.load(D), implementation="sme")
    T.store(C, T.cast(T.max(value * 0.5 + 0.125, 0), output))

  return kernel


def _strided_lower(dtype, output, *, transformed, vl, svl):
  return (
    CPUBackend()
    .lower(
      _strided_template(dtype, output, transformed=transformed).ir,
      (),
      target="SME",
      options={
        "cpu_matmul_pipeline": "off",
        "cpu_matmul_fusion": "always",
        "_vector_state": VectorState(vl, svl),
      },
    )
    .source
  )


def _strided_wrapper(source, dtype, output, *, transformed):
  acc = "float64" if dtype == "float64" else "int32" if dtype in {"int8", "uint8"} else "float32"
  memory = "float32" if transformed else dtype
  definitions = [
    '#include "vecops/Kernel.h"',
    "namespace vecops::spec { inline constexpr auto Parallelism = vecops::meta::cint<1>; }",
    source,
    (
      'extern "C" void strided_sme(void* ap,void* bp,void* dp,void* cp,'
      "vecops::nint_t m,vecops::nint_t n,vecops::nint_t k,vecops::nint_t a0,vecops::nint_t a1,"
      "vecops::nint_t b0,vecops::nint_t b1,vecops::nint_t d0,vecops::nint_t d1,"
      "vecops::nint_t c0,vecops::nint_t c1) {"
    ),
  ]
  for name, pointer, element, shape, stride in (
    ("A", "ap", memory, ("m", "k"), ("a0", "a1")),
    ("B", "bp", dtype, ("k", "n"), ("b0", "b1")),
    ("D", "dp", acc, ("m", "n"), ("d0", "d1")),
    ("C", "cp", output, ("m", "n"), ("c0", "c1")),
  ):
    definitions.append(
      f"auto {name}=vecops::tensor::make_tensor(static_cast<{CPP_TYPES[element]}*>({pointer}),"
      f"vecops::tensor::make_shape(vecops::meta::Any{{{shape[0]}}},vecops::meta::Any{{{shape[1]}}}),"
      f"vecops::tensor::make_strides(vecops::meta::Any{{{stride[0]}}},vecops::meta::Any{{{stride[1]}}}));"
    )
  definitions.extend(("__kernel__(A,B,D,C);", "}"))
  return "\n".join(definitions)


def _owners(assembly, *, vl, svl):
  """Return validated compute bodies, not unequal-domain scalar entries.

  Discovery validates every actual owner but does not assert source-to-object
  completeness. Tests with a known family count use _assert_inlined_owners.
  Captured execution domains remain explicit; never infer them from assembly.
  """
  count = sum("with_streaming_za" in name for name in assembly_functions(assembly))
  return list(assert_sme_owners(assembly, count, equal_domains=vl == svl).values())


def _assert_inlined_owners(assembly, count, *, vl, svl):
  owners = assert_sme_owners(assembly, count, equal_domains=vl == svl)
  for index, owner in enumerate(owners.values()):
    assert re.search(r"^\s*\w*mopa\b", owner, re.MULTILINE), f"owner {index} lost its matrix instructions"
    assert len(re.findall(r"^\s*smstart\b", owner, re.MULTILINE)) == 1
    assert re.search(r"^\s*smstop\b", owner, re.MULTILINE)
    calls = re.findall(r"^\s*(?:bl|blr)\b[^\n]*", owner, re.MULTILINE)
    assert not calls, f"owner {index} still contains non-inlined calls: {calls[:12]}"
    assert re.search(r"\bza[0-9][hv]\.", owner), f"owner {index} lost accumulator input/output"


@pytest.mark.parametrize(
  "dtype,output,packed,column_major,pipeline,vl,svl",
  (
    ("float16", "float32", False, False, "plain", 64, 64),
    ("float16", "float16", False, True, "affine", 16, 64),
    ("bfloat16", "float32", False, False, "relu", 64, 16),
    ("bfloat16", "bfloat16", True, True, "affine", 32, 64),
    ("float32", "float32", False, True, "affine", 64, 16),
    ("float64", "float64", False, False, "plain", 16, 64),
    ("int8", "int32", False, False, "relu", 64, 16),
    ("uint8", "int32", True, True, "plain", 16, 64),
  ),
)
def test_dynamic_sme_real_arm_every_state_owner_has_no_calls(
  tmp_path, dtype, output, packed, column_major, pipeline, vl, svl
):
  source = _lower(
    dtype=dtype, output=output, packed=packed, column_major=column_major, pipeline=pipeline, vl=vl, svl=svl
  )
  assembly = _cross(
    tmp_path,
    source,
    lambda source: _cross_wrapper(source, dtype, output, column_major),
    vl=vl,
    svl=svl,
  )
  _assert_inlined_owners(assembly, 11 if dtype == "float64" and not packed else 9, vl=vl, svl=svl)


@pytest.mark.parametrize(
  "dtype,output,transformed,vl,svl",
  (("float16", "float16", True, 16, 64), ("bfloat16", "float32", True, 64, 16), ("float32", "float32", False, 32, 64)),
)
def test_dynamic_sme_real_arm_strided_prologue_epilogue_stays_inlined(tmp_path, dtype, output, transformed, vl, svl):
  source = _strided_lower(dtype, output, transformed=transformed, vl=vl, svl=svl)
  assert "vecops::vec::mul" in source and "vecops::vec::add" in source
  assert "vecops::vec::fmadd" not in source  # Strict separate mul/add contract.
  assert "opt::strided" in source  # Runtime strides may be negative or nonunit.
  assert_no_vec_mask_arrays(source)
  assembly = _cross(
    tmp_path,
    source,
    lambda source: _strided_wrapper(source, dtype, output, transformed=transformed),
    vl=vl,
    svl=svl,
  )
  _assert_inlined_owners(assembly, 9, vl=vl, svl=svl)


def _matrix(shape, dtype, *, reversed_axes=False):
  rows, columns = shape
  itemsize = np.dtype(dtype).itemsize
  stride = (2 * columns + 3, 2)
  capacity = max(1, max(0, rows - 1) * stride[0] + max(0, columns - 1) * stride[1] + 1)
  value = np.ndarray(
    shape, dtype=dtype, buffer=np.empty(capacity, dtype=dtype), strides=tuple(s * itemsize for s in stride)
  )
  return value[::-1, ::-1] if reversed_axes else value


def _native_shapes(lanes, *, expanded=False):
  shapes = [(0, 5, 7), (5, 0, 7), (5, 7, 0)]
  for blocks in (1, 2, 3, 4):
    shapes.extend(((3, blocks * lanes - 1, 17), (blocks * lanes - 1, 3, 17)))
  shapes.extend(((lanes + 1, lanes + 3, 17), (2 * lanes + 1, 2 * lanes + 3, 17), (3, 4 * lanes + 1, 5)))
  if expanded:
    shapes.extend(((4 * lanes + 1, 2 * lanes + 1, 17), (2 * lanes + 1, 4 * lanes + 1, 17)))
  return shapes


def test_native_fixture_preserves_negative_strides_and_all_geometry_boundaries():
  value = _matrix((3, 7), "float32", reversed_axes=True)
  assert all(stride < 0 for stride in value.strides)
  shapes = _native_shapes(4, expanded=True)
  assert (3, 15, 17) in shapes and (15, 3, 17) in shapes
  assert (17, 9, 17) in shapes and (9, 17, 17) in shapes
  for shape in ((0, 5), (5, 0), (5, 7)):
    assert _matrix(shape, "float32").shape == shape


@pytest.mark.skipif(not _sme_available(), reason="state-owner execution requires target SME+FA64")
@pytest.mark.parametrize(
  "dtype,output,transformed",
  (("float16", "float32", True), ("bfloat16", "bfloat16", True), ("float64", "float64", False)),
)
def test_dynamic_sme_arm_native_all_owners_negative_strides_and_alias_prior(tmp_path, dtype, output, transformed):
  if dtype == "bfloat16":
    pytest.importorskip("ml_dtypes")
  acc = "float64" if dtype == "float64" else "float32"
  memory = "float32" if transformed else dtype
  state = get_vector_state()
  assert state.sme_bytes is not None
  lanes = state.sme_bytes // np.dtype(acc).itemsize
  compiler = make_compiler(target="Native", jobs=4, cache_dir=tmp_path / "cache", build_dir=tmp_path / "build")
  compiled = T.compile(
    _strided_template(dtype, output, transformed=transformed),
    compiler=compiler,
    pass_options={"cpu_matmul_pipeline": "off", "cpu_matmul_fusion": "always"},
  )
  # One Dynamic artifact serves every family, including zero dimensions and
  # negative strides. Do not specialize away the runtime geometry dispatch.
  with vecops.collect_compile_requests() as collector:
    compiled.operator(
      vecops.TensorMeta((3, 17), strides=(37, 2), dtype=memory),
      vecops.TensorMeta((17, 19), strides=(41, 2), dtype=dtype),
      vecops.TensorMeta((3, 19), strides=(41, 2), dtype=acc),
      vecops.TensorMeta((3, 19), strides=(41, 2), dtype=output),
      **compiled._core.specialization_values,
    )
  assert len(collector.requests) == 1
  vecops.compile_batch(collector.requests, parallelism=compiler.jobs, execution_parallelism=1)
  rng = np.random.default_rng(624)
  for m, n, k in _native_shapes(lanes, expanded=dtype == "float64"):
    for negative in (False, True):
      a = _matrix((m, k), memory, reversed_axes=negative)
      b = _matrix((k, n), dtype, reversed_axes=not negative)
      a[:] = rng.integers(-4, 5, size=a.shape) / 16
      b[:] = rng.integers(-4, 5, size=b.shape) / 16
      avalue = (a * np.float32(0.5) + np.float32(0.125)).astype(dtype) if transformed else a
      product = avalue.astype(acc) @ b.astype(acc)
      for alias in (False, True) if output == acc else (False,):
        c = _matrix((m, n), output, reversed_axes=negative)
        d = c if alias else _matrix((m, n), acc, reversed_axes=not negative)
        d[:] = rng.integers(-4, 5, size=d.shape) / 16
        # Snapshot prior before an aliased native invocation; the independent
        # oracle cannot accidentally read a C value already overwritten by it.
        expected = np.maximum((product + d.copy()) * np.dtype(acc).type(0.5) + np.dtype(acc).type(0.125), 0).astype(
          output
        )
        compiled.operator.native(a, b, d, c, **compiled._core.specialization_values)
        np.testing.assert_array_equal(
          c, expected, err_msg=f"{dtype}/{output} {(m, n, k)} negative={negative} alias={alias}"
        )
