# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Exact SME leaves preserve full prefixes without empty-block work in K.

The optional cross tests compile real-pointer wrappers with SME FA64 enabled;
they are instruction/codegen checks, not substitutes for target execution.
"""

import os
import re
import subprocess
from pathlib import Path

import pytest
import vecops
import vecops.language as T
from dsl_test_utils import arm_cross_cxx_include_flags, assert_no_vec_mask_arrays, make_compiler
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


@pytest.mark.parametrize("packed", (False, True))
@pytest.mark.parametrize("column_major", (False, True))
def test_exact_family_dispatch_and_callback_owned_extents(packed, column_major):
  source = _lower(packed=packed, column_major=column_major)
  for shape in ("1x1", "1x2", "1x3", "2x1", "3x1", "2x2"):
    branch = _body(source, f"/* exact {shape} ZA blocks */")
    assert "/* exact blocks: no empty-block tests inside K */" in branch
    assert "/* partial spatial family */" not in branch
    assert "/* full ZA family:" not in branch
    k_body = _body(branch, "/* full K groups: no per-element tail tests */")
    assert not re.search(r"if \(\(dsl_c_sme_[mn]_", k_body)
    rows, columns = map(int, shape.split("x"))
    assert k_body.count("vecops::vec::sme::mopa<") == rows * columns
  headers = re.findall(
    r"\w+\.with_resources\(vecops::execution::sme::Region\{\}, \[&\]\(auto& \w+\) VECOPS_INLINE_LAMBDA_NOEXCEPT",
    source,
  )
  assert len(headers) == 9
  for header in headers:
    owner = _body(source, header)
    assert ".with_resources(" not in owner
    for axis in "mnk":
      match = re.search(rf"const vecops::nint_t (\w+_sme_extent_{axis}_value_\d+)=(\w+_sme_extent_{axis}_\d+);", owner)
      assert match is not None
      # Every selected trampoline reads each escaped capture exactly once;
      # all subsequent uses inside that owner are callback-owned SSA values.
      assert not re.search(rf"\b{match[2]}\b", owner[match.end() :])
  assert "/* partial spatial family */" in source  # Large generic fallback remains.
  assert_no_vec_mask_arrays(source)


@pytest.mark.parametrize("output,column_major", (("float32", False), ("float32", True), ("bfloat16", False)))
def test_exact_full_prefix_drain_keeps_static_extent_and_dynamic_tail(output, column_major):
  source = _lower(output=output, column_major=column_major)
  branch = _body(source, "/* exact 2x2 ZA blocks */")
  stores = re.findall(r"for \(vecops::nint_t (\w+_sme_store_slice_\d+)=0;([^\n]+)", branch)
  assert len(stores) == 4
  # Tile 0 and the tile adjacent along the vector axis have sixteen rows
  # (or columns) for a 64-byte target. The last spatial band remains masked.
  full = (0, 2) if column_major else (0, 1)
  for position in full:
    assert "clamp_value" not in stores[position][1].split(" && ")[0]
    assert re.search(r"<\(dsl_c_sme_matrix_lanes_\d+\)", stores[position][1])
  for position in set(range(4)) - set(full):
    assert "sme::clamp_value<vecops::nint_t>" in stores[position][1]
  axis = "ver" if column_major else "hor"
  assert all(f"sme::read_{axis}<{tile}>" in branch for tile in range(4))
  assert "vecops::vec::max" in branch


@pytest.mark.parametrize("dtype,output", (("float64", "float64"), ("int8", "int32"), ("uint8", "int32")))
def test_exact_native_accumulator_memory_path_preserves_dtypes(dtype, output):
  source = _lower(dtype=dtype, output=output, pipeline="plain")
  branch = _body(source, "/* exact 2x2 ZA blocks */")
  assert "sme::load_hor<" in branch and "sme::store_hor<" in branch
  assert "sme::read_hor<" not in branch
  assert "one partial K group" not in branch  # Both packed inputs are padded.
  if dtype == "float64":
    assert "/* expanded ZA64 catalog */" not in source  # Existing packed F64 policy unchanged.


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


@pytest.mark.parametrize("vl,svl", ((16, 64), (64, 16), (64, 64)))
@pytest.mark.parametrize(
  "dtype,output,packed,column_major,pipeline",
  (
    ("bfloat16", "float32", True, False, "relu"),
    ("bfloat16", "bfloat16", True, True, "affine"),
    ("int8", "int32", False, False, "plain"),
    ("float64", "float64", False, True, "plain"),
  ),
)
def test_exact_blocks_arm_cross_codegen(tmp_path, dtype, output, packed, column_major, pipeline, vl, svl):
  clang = os.environ.get("VECOPS_TEST_ARM_CROSS_CLANG")
  sysroot = os.environ.get("VECOPS_TEST_ARM_CROSS_SYSROOT")
  if not clang or not sysroot:
    pytest.skip("set VECOPS_TEST_ARM_CROSS_CLANG/SYSROOT for real FA64 cross-codegen checks")
  root = Path(__file__).resolve().parents[2]
  cpp = tmp_path / "exact.cpp"
  cpp.write_text(
    _cross_wrapper(
      _lower(dtype=dtype, output=output, packed=packed, column_major=column_major, pipeline=pipeline),
      dtype,
      output,
      column_major,
    ),
    encoding="utf-8",
  )
  assembly = cpp.with_suffix(".s")
  command = [
    clang,
    "--target=aarch64-linux-gnu",
    f"--sysroot={sysroot}",
    "-march=armv9.2-a+sme+sme-f64f64+bf16",
    "-std=c++20",
    "-O3",
    "-S",
    "-DNDEBUG",
    "-DVECOPS_TARGET_SME_FA64=1",
    "-DVECOPS_TARGET_SME_F64F64=1",
    f"-DVECOPS_TARGET_FIXED_STREAMING_SVE_BITS={svl * 8}",
    f"-DVECOPS_KERNEL_SVE_BYTES={vl}",
    f"-DVECOPS_KERNEL_SME_BYTES={svl}",
    "-I",
    str(root / "include"),
    "-I",
    str(root / "include/vecops"),
    *arm_cross_cxx_include_flags(sysroot),
    str(cpp),
    "-o",
    str(assembly),
  ]
  result = subprocess.run(command, capture_output=True, text=True, check=False)
  cpp.with_suffix(".log").write_text(result.stdout + result.stderr, encoding="utf-8")
  assert result.returncode == 0, result.stderr[-6000:]
  text = assembly.read_text(encoding="utf-8")
  assert re.search(r"^\s*\w*mopa\b", text, re.MULTILINE)
  assert re.search(r"^\s*smstart\b", text, re.MULTILINE) and re.search(r"^\s*smstop\b", text, re.MULTILINE)
  assert re.search(r"\bza[0-9][hv]\.", text)
  if packed:
    # Inspect real backwards K-loop edges: at least one complete four-tile
    # kernel must keep all scalar bounds/cursors in registers. The former
    # outlined callback reloaded its escaped K capture after every MOPA set.
    labels = {match[1]: match.start() for match in re.finditer(r"^(\.LBB\d+_\d+):", text, re.MULTILINE)}
    loops = []
    for branch in re.finditer(r"^\s*b\.\w+\s+(\.LBB\d+_\d+)\b", text, re.MULTILINE):
      if branch[1] in labels and labels[branch[1]] < branch.start():
        loops.append(text[labels[branch[1]] : branch.start()])
    assert any(
      len(re.findall(r"^\s*\w*mopa\b", loop, re.MULTILINE)) == 4
      and not re.search(r"^\s*(?:ldr|ldp)\b", loop, re.MULTILINE)
      for loop in loops
    ), "packed K loop still reloads scalar captures/metadata"
    if output == "float32" and svl == 64:
      # Match the target's native full-prefix drain: LLVM can statically
      # unroll sixteen ZA rows rather than leave a dynamic min()/back-edge.
      blocks = re.split(r"^\.LBB\d+_\d+:.*$", text, flags=re.MULTILINE)
      assert max(len(re.findall(r"\bza0h\.s\[", block)) for block in blocks) >= 16
