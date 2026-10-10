# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Exact SME leaves preserve full prefixes without empty-block work in K.

The optional cross tests compile real-pointer wrappers with SME FA64 enabled;
they are instruction/codegen checks, not substitutes for target execution.
"""

import os
from pathlib import Path

import pytest
import vecops
import vecops.language as T
from dsl_test_utils import make_compiler
from vecops._arm_vector_state import get_vector_state
from vecops.language._compiler.backends.cpu.backend import CPUBackend
from vecops.language._compiler.backends.cpu.capabilities import CPP_TYPES


def _template(dtype="bfloat16", output="float32", *, packed=True, column_major=False, pipeline="relu"):
  M, N, K, PSA, PSB = (T.dynamic(lower=0) for _ in range(5))
  acc = "float64" if dtype == "float64" else "int32" if dtype in {"int8", "uint8"} else "float32"
  strides = (1, M) if column_major else (N, 1)

  @T.template
  def kernel(
    A: T.Tensor((M, K), dtype=dtype),
    B: T.Tensor((K, N), (1, K), dtype),
    D: T.Tensor((M, N), strides, acc),
    C: T.Tensor((M, N), strides, output, access="output"),
    AP: T.Tensor((PSA,), dtype=dtype),
    BP: T.Tensor((PSB,), dtype=dtype),
  ):
    a = T.packed(AP, (M, K), side="A", format="sme") if packed else T.load(A)
    b = T.packed(BP, (K, N), side="B", format="sme") if packed else T.load(B)
    result = T.dot((a, b), C, acc=T.load(D), implementation="sme")
    if pipeline == "relu":
      T.store(C, T.max(result, 0))
    elif pipeline == "affine":
      T.store(C, result * 0.5 + 1)

  return kernel


def _lower(**kwargs):
  return CPUBackend().lower(_template(**kwargs).ir, (), target="SME", options={"cpu_matmul_pipeline": "off"}).source


def _body(source, marker):
  start = source.index("{", source.index(marker) + len(marker))
  depth = 1
  for position in range(start + 1, len(source)):
    depth += (source[position] == "{") - (source[position] == "}")
    if depth == 0:
      return source[start + 1 : position]
  raise AssertionError(f"unclosed block following {marker}")


def _sme_available():
  if os.uname().machine.lower() not in {"aarch64", "arm64"}:
    return False
  try:
    features = Path("/proc/cpuinfo").read_text(encoding="utf-8").split()
  except OSError:
    return False
  return "sme" in features and any(flag in features for flag in ("smefa64", "sme_fa64"))


def _runtime_metadata(dtype, output, *, column_major):
  acc = "float64" if dtype == "float64" else "int32" if dtype == "int8" else "float32"
  stride = (1, 19) if column_major else (23, 1)
  return (
    vecops.TensorMeta((19, 17), dtype=dtype),
    vecops.TensorMeta((17, 23), strides=(1, 17), dtype=dtype),
    vecops.TensorMeta((19, 23), strides=stride, dtype=acc),
    vecops.TensorMeta((19, 23), strides=stride, dtype=output),
    vecops.TensorMeta((8192,), dtype=dtype),
    vecops.TensorMeta((8192,), dtype=dtype),
  )


@pytest.mark.parametrize(
  "dtype,output", (("bfloat16", "float32"), ("bfloat16", "bfloat16"), ("int8", "int32"), ("float64", "float64"))
)
@pytest.mark.parametrize("column_major", (False, True))
def test_runtime_metadata_uses_public_keyword_only_strides(dtype, output, column_major):
  # Validate fixture construction even on x86 where SME execution is skipped.
  metadata = _runtime_metadata(dtype, output, column_major=column_major)
  assert metadata[1].strides == (1, 17)
  assert metadata[2].strides == metadata[3].strides == ((1, 19) if column_major else (23, 1))


@pytest.mark.skipif(not _sme_available(), reason="exact-block execution requires a target SME+FA64 CPU")
@pytest.mark.parametrize(
  "dtype,output", (("bfloat16", "float32"), ("bfloat16", "bfloat16"), ("int8", "int32"), ("float64", "float64"))
)
def test_exact_blocks_arm_runtime_boundaries(dtype, output, tmp_path):
  import numpy as np

  if dtype == "bfloat16":
    pytest.importorskip("ml_dtypes")
  state = get_vector_state()
  assert state.sme_bytes is not None
  acc = "float64" if dtype == "float64" else "int32" if dtype == "int8" else "float32"
  lanes = state.sme_bytes // np.dtype(acc).itemsize
  panel, kp = 2 * lanes, max(1, 4 // np.dtype(dtype).itemsize)
  compiler = make_compiler(target="Native", jobs=4, cache_dir=tmp_path / "cache", build_dir=tmp_path / "build")
  kernels = {}
  pipeline = "plain" if dtype == "int8" else "relu"
  with vecops.collect_compile_requests() as collector:
    for packed in (False, True):
      for column_major in (False, True):
        compiled = T.compile(
          _template(dtype, output, packed=packed, column_major=column_major, pipeline=pipeline), compiler=compiler
        )
        compiled.operator(
          *_runtime_metadata(dtype, output, column_major=column_major),
          **compiled._core.specialization_values,
        )
        kernels[packed, column_major] = compiled
  vecops.compile_batch(collector.requests, parallelism=compiler.jobs, execution_parallelism=1)
  rng = np.random.default_rng(263)

  def matrix(shape, element, column_major=False):
    rows, columns = shape
    itemsize = np.dtype(element).itemsize
    stride = (itemsize, rows * itemsize) if column_major else (columns * itemsize, itemsize)
    array = np.ndarray(shape, dtype=element, buffer=np.empty(max(1, rows * columns), dtype=element), strides=stride)
    values = rng.integers(-3, 4, size=shape)
    array[:] = values if element.startswith("int") else values / 8
    return array

  def pack(logical):
    spatial, reduction = logical.shape
    packed = np.zeros(((spatial + panel - 1) // panel, (reduction + kp - 1) // kp, panel, kp), dtype=dtype)
    for row in range(spatial):
      for column in range(reduction):
        packed[row // panel, column // kp, row % panel, column % kp] = logical[row, column]
    return packed.reshape(-1)

  shapes = [(0, 5, 7), (5, 0, 7), (5, 7, 0)]
  for blocks in (1, 2, 3, 4):
    for tail in (0, 1):
      spatial = blocks * lanes - tail
      shapes.extend(((3, spatial, 17), (spatial, 3, 17)))
  shapes.extend(((lanes + 1, lanes + 3, 17), (2 * lanes, 2 * lanes, 17), (2 * lanes + 1, 2 * lanes + 3, 17)))
  for m, n, k in shapes:
    a, b = matrix((m, k), dtype), matrix((k, n), dtype, True)
    ap, bp = pack(a), pack(b.T)
    for (_, column_major), compiled in kernels.items():
      d, c = matrix((m, n), acc, column_major), matrix((m, n), output, column_major)
      expected = a.astype(acc) @ b.astype(acc) + d.copy()
      if pipeline == "relu":
        expected = np.maximum(expected, 0)
      compiled.operator.native(a, b, d, c, ap, bp, **compiled._core.specialization_values)
      np.testing.assert_array_equal(
        c, expected.astype(output), err_msg=f"{dtype}/{output}, {(m, n, k)}, column={column_major}"
      )


def _cross_wrapper(source, dtype, output, column_major):
  acc = "float64" if dtype == "float64" else "int32" if dtype in {"int8", "uint8"} else "float32"
  preamble = "namespace vecops::spec { inline constexpr auto Parallelism = vecops::meta::cint<1>; }\n"
  lines = [preamble, source]
  lines.append(
    'extern "C" void exact_blocks(void* ap,void* bp,void* dp,void* cp,void* packa,void* packb,'
    "vecops::nint_t m,vecops::nint_t n,vecops::nint_t k,vecops::nint_t psa,vecops::nint_t psb) {"
  )
  for name, pointer, element, shape, stride in (
    ("A", "ap", dtype, "vecops::meta::Any{m},vecops::meta::Any{k}", "vecops::meta::Any{k},vecops::meta::cint<1>"),
    ("B", "bp", dtype, "vecops::meta::Any{k},vecops::meta::Any{n}", "vecops::meta::cint<1>,vecops::meta::Any{k}"),
    ("D", "dp", acc, "vecops::meta::Any{m},vecops::meta::Any{n}", ""),
    ("C", "cp", output, "vecops::meta::Any{m},vecops::meta::Any{n}", ""),
    ("AP", "packa", dtype, "vecops::meta::Any{psa}", "vecops::meta::cint<1>"),
    ("BP", "packb", dtype, "vecops::meta::Any{psb}", "vecops::meta::cint<1>"),
  ):
    if not stride:
      stride = (
        "vecops::meta::cint<1>,vecops::meta::Any{m}" if column_major else "vecops::meta::Any{n},vecops::meta::cint<1>"
      )
    lines.append(
      f"auto {name}=vecops::tensor::make_tensor(static_cast<{CPP_TYPES[element]}*>({pointer}),"
      f"vecops::tensor::make_shape({shape}),vecops::tensor::make_strides({stride}));"
    )
  lines.extend(("__kernel__(A,B,D,C,AP,BP);", "}"))
  return "#include <vecops/Kernel.h>\n" + "\n".join(lines)
