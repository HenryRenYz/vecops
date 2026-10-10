# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Ordinary vector/state IR must support a source-level dot microtile."""

import os
import re
import subprocess
from pathlib import Path

import numpy as np
import pytest
import vecops.language as T
from dsl_test_utils import arm_cross_cxx_include_flags, assert_no_vec_mask_arrays, make_compiler
from vecops._arm_vector_state import VectorState
from vecops.language._compiler.backends.cpu.backend import CPUBackend
from vecops.language._library.matmul import vector_tile_dot


def template(dtype="float32", *, seed=False, affine=False, columns=16):
  @T.template
  def kernel(
    A: T.Tensor((4, 17), dtype=dtype),
    B: T.Tensor((17, columns), dtype=dtype),
    D: T.Tensor((4, columns), dtype="float32"),
    C: T.Tensor((4, columns), dtype="float32", access="output"),
  ):
    a, b = T.load(A), T.load(B)
    if affine:
      a = a * 2 + 1
      b = b * 2 + 1
    initial = T.load(D) if seed else None
    result = vector_tile_dot((a, b), acc=initial)
    T.store(C, T.max(result * 0.5 + 1, 0))

  return kernel


def test_vector_dot_uses_standard_register_state_and_fma_not_opaque_matmul():
  artifact = CPUBackend().lower(
    template().ir, (), target="AVX512", options={"cpu_ordered_loop_unroll": True, "cpu_state_vector_reuse": True}
  )
  assert not artifact.backend_ir.matmuls
  states = tuple(artifact.backend_ir.states.values())
  assert states and all(plan.pieces for plan in states)
  source = artifact.source
  assert "vecops::vec::fmadd" in source
  assert "#pragma GCC unroll 2" in source
  assert "private scalar scratch" not in "\n".join(artifact.lowered.kernel.diagnostics)
  assert_no_vec_mask_arrays(source)


def dot_template(*, columns=8, k=17, historical=False, packed=False, affine=True, implementation="vector"):
  @T.template
  def kernel(
    A: T.Tensor((4, k), dtype="float32", access="inout"),
    B: T.Tensor((k, columns), dtype="float32"),
    D: T.Tensor((1, columns), dtype="float32"),
    C: T.Tensor((4, columns), dtype="float32", access="output"),
  ):
    a = T.load(A)
    if affine:
      a = a * 2 + 1
    if historical:
      T.store(A, T.full((4, k), 99, dtype="float32"))
    if packed:
      b = T.pack(B, side="B", format="dense")
    else:
      b = T.load(B)
    result = T.dot((a, b), C, acc=T.load(D), acc_dtype="float32", implementation=implementation)
    T.store(C, T.max(result * 0.5 + 1, 0))

  return kernel


def test_default_dot_dispatch_stages_recipe_before_vector_lowering():
  backend = CPUBackend()
  staged = backend.lower(dot_template().ir, (), target="AVX512")
  assert not staged.backend_ir.matmuls
  assert staged.backend_ir.states and all(plan.pieces for plan in staged.backend_ir.states.values())
  assert "cpu-expand-dot-recipes" in staged.lowered.kernel.pass_trace
  assert "staged 1 vector dot" in "\n".join(staged.lowered.kernel.diagnostics)
  assert_no_vec_mask_arrays(staged.source)
  # Larger output rows use the streaming source recipe; packed input still
  # requires the separate ABI-checked family.
  streaming = backend.lower(dot_template(columns=19).ir, (), target="AVX512")
  assert not streaming.backend_ir.matmuls and streaming.backend_ir.states
  fallback = backend.lower(dot_template(packed=True).ir, (), target="AVX512")
  assert fallback.backend_ir.matmuls


def test_automatic_small_x86_dot_uses_recipe_without_changing_sme_selection():
  kernel = dot_template(implementation="auto")
  x86 = CPUBackend().lower(kernel.ir, (), target="AVX512")
  arm = CPUBackend().lower(kernel.ir, (), target="SME")
  assert not x86.backend_ir.matmuls and x86.backend_ir.states
  assert arm.backend_ir.matmuls
  assert_no_vec_mask_arrays(x86.source)


def streaming_template(*, historical=False, scalar_seed=False):
  m, n, k = (T.dynamic(lower=0) for _ in range(3))

  @T.func
  def affine(value):
    return value * 2 + 1

  @T.func
  def finish(value, bias):
    return T.max(value * 0.5 + bias, 0)

  @T.template
  def kernel(
    A: T.Tensor((m, k), (T.dynamic(), T.dynamic()), "float32", access="inout"),
    B: T.Tensor((k, n), (T.dynamic(), T.dynamic()), "float32"),
    D: T.Tensor((1, n), (T.dynamic(), 1), "float32"),
    C: T.Tensor((m, n), (T.dynamic(), T.dynamic()), "float32", access="output"),
  ):
    a = affine(T.load(A))
    if historical:
      T.store(A, 99)
    seed = 2.0 if scalar_seed else T.load(D)
    result = T.dot((a, T.load(B)), C, acc=seed, implementation="vector")
    T.store(C, finish(result, T.load(D)))

  return kernel


@pytest.mark.parametrize("historical,scalar_seed", [(False, False), (True, False), (False, True)])
def test_streaming_dot_has_source_control_and_fused_value_helper(historical, scalar_seed):
  artifact = CPUBackend().lower(
    streaming_template(historical=historical, scalar_seed=scalar_seed).ir, (), target="AVX512"
  )
  assert not artifact.backend_ir.matmuls
  assert artifact.backend_ir.states and all(plan.pieces for plan in artifact.backend_ir.states.values())
  assert "vecops::vec::fmadd" in artifact.source
  assert "private scalar scratch" not in "\n".join(artifact.lowered.kernel.diagnostics)
  assert_no_vec_mask_arrays(artifact.source)


@pytest.mark.parametrize("historical,scalar_seed", [(False, False), (True, False), (False, True)])
def test_native_streaming_dot_strides_aliases_and_broadcast_epilogue(historical, scalar_seed, tmp_path):
  compiler = make_compiler(jobs=4, cache_dir=tmp_path / "cache", build_dir=tmp_path / "build")
  compiled = T.compile(streaming_template(historical=historical, scalar_seed=scalar_seed), compiler=compiler)
  for m, n, k in [(0, 19, 17), (7, 0, 17), (7, 19, 0), (1, 1, 17), (3, 8, 17), (7, 19, 17), (7, 257, 17)]:
    for reverse in (False, True):
      a = (np.arange(m * k, dtype=np.float32).reshape(m, k) % 7) / 16
      b = (np.arange(k * n, dtype=np.float32).reshape(k, n) % 11) / 16
      if reverse:
        a, b = a[::-1, ::-1], b[::-1, ::-1]
      bias = np.arange(n, dtype=np.float32).reshape(1, n) / 16
      expected = np.maximum(((a * 2 + 1) @ b + (2.0 if scalar_seed else bias)) * 0.5 + bias, 0)
      raw = np.full((m + 2, n + 3), -17, dtype=np.float32)
      output = raw[1 : m + 1, 1 : n + 1]
      if reverse:
        output = output[::-1, ::-1]
      compiled.operator.native(a, b, bias, output, **compiled._core.specialization_values)
      np.testing.assert_array_equal(output, expected)
      np.testing.assert_array_equal(raw[[0, -1]], -17)
      np.testing.assert_array_equal(raw[:, [0, -1]], -17)
      if historical:
        np.testing.assert_array_equal(a, np.full(a.shape, 99, np.float32))
  if not historical:
    # Whole producer snapshots must survive a writeback through another ABI
    # parameter, including a broadcast epilogue that aliases the first row.
    for aliased_bias in (False, True):
      a = (np.arange(7 * 23, dtype=np.float32).reshape(7, 23) % 7) / 16
      b = (np.arange(23 * 19, dtype=np.float32).reshape(23, 19) % 11) / 16
      output = a[:, :19]
      bias = output[:1] if aliased_bias else np.arange(19, dtype=np.float32).reshape(1, 19) / 16
      seed = 2.0 if scalar_seed else bias.copy()
      expected = np.maximum(((a.copy() * 2 + 1) @ b + seed) * 0.5 + bias.copy(), 0)
      compiled.operator.native(a, b, bias, output, **compiled._core.specialization_values)
      np.testing.assert_array_equal(output, expected)


def test_staged_value_helper_retains_original_read_identity():
  from vecops.language._compiler import ir as I
  from vecops.language._compiler.frontend.helpers import ValueFunction

  extent = I.Expr(I.ScalarType("int64"), "region_argument")
  parameter = I.Expr(I.VectorType("float32", (extent,)), "region_argument")
  view = I.View("original", "float32", (I.literal(1),), (I.literal(1),), I.literal(0))
  read = I.Expr(I.TileType("float32", view.shape), "load", data=view)
  function = ValueFunction((parameter,), I.binary("add", parameter, read))
  actual = I.Expr(I.VectorType("float32", (I.literal(7),)), "register_input")
  result = function.bind((actual,))
  assert result.args == (actual, read)
  assert I.constant(result.type.shape[0]) == 7


@pytest.mark.parametrize("k,historical", [(0, False), (17, False), (17, True)])
@pytest.mark.parametrize("implementation", ["auto", "vector"])
def test_native_dot_recipe_dispatch_broadcast_seed_and_history(k, historical, implementation, tmp_path):
  compiler = make_compiler(jobs=4, cache_dir=tmp_path / "cache", build_dir=tmp_path / "build")
  compiled = T.compile(dot_template(k=k, historical=historical, implementation=implementation), compiler=compiler)
  if implementation == "vector" or os.uname().machine.lower() in {"x86_64", "amd64"}:
    assert "vector outer-product family" not in compiled.generated_source
  a = np.arange(4 * k, dtype=np.float32).reshape(4, k) / 16
  b = np.arange(k * 8, dtype=np.float32).reshape(k, 8) / 16
  seed = np.arange(8, dtype=np.float32).reshape(1, 8) / 16
  expected = np.maximum(((a * 2 + 1) @ b + seed) * 0.5 + 1, 0)
  output = np.empty((4, 8), np.float32)
  compiled.operator.native(a, b, seed, output, **compiled._core.specialization_values)
  np.testing.assert_array_equal(output, expected)
  if historical:
    np.testing.assert_array_equal(a, np.full(a.shape, 99, np.float32))


@pytest.mark.parametrize("dtype", ["float32", "float16"])
@pytest.mark.parametrize("seed,affine", [(False, False), (True, False), (True, True)])
def test_native_vector_dot_microtile(dtype, seed, affine, tmp_path):
  compiler = make_compiler(jobs=4, cache_dir=tmp_path / "cache", build_dir=tmp_path / "build")
  compiled = T.compile(
    template(dtype, seed=seed, affine=affine),
    compiler=compiler,
    pass_options={"cpu_ordered_loop_unroll": True, "cpu_state_vector_reuse": True},
  )
  rng = np.random.default_rng(127)
  a = (rng.integers(-3, 4, size=(4, 17)) / 8).astype(dtype)
  b = (rng.integers(-3, 4, size=(17, 16)) / 8).astype(dtype)
  initial = (rng.integers(-3, 4, size=(4, 16)) / 8).astype(np.float32)
  output = np.empty((4, 16), np.float32)
  x, y = (a * 2 + 1).astype(dtype), (b * 2 + 1).astype(dtype)
  if not affine:
    x, y = a, b
  expected = x.astype(np.float32) @ y.astype(np.float32)
  if seed:
    expected += initial
  expected = np.maximum(expected * 0.5 + 1, 0)
  compiled.operator.native(a, b, initial, output, **compiled._core.specialization_values)
  np.testing.assert_array_equal(output, expected)


@pytest.mark.parametrize("vl", [16, 64, 128])
@pytest.mark.parametrize("affine", [False, True])
@pytest.mark.parametrize("staged", [False, True])
def test_vector_dot_arm_cross_has_register_fma_loop(vl, affine, staged, tmp_path):
  clang = os.environ.get("VECOPS_TEST_ARM_CROSS_CLANG")
  sysroot = os.environ.get("VECOPS_TEST_ARM_CROSS_SYSROOT")
  if not clang or not sysroot:
    pytest.skip("set VECOPS_TEST_ARM_CROSS_CLANG/SYSROOT for real SVE codegen")
  columns = vl // 4
  kernel = dot_template(columns=columns, affine=affine) if staged else template(columns=columns, affine=affine)
  seed_rows = 1 if staged else 4
  artifact = CPUBackend().lower(
    kernel.ir,
    (),
    target="SVE",
    options={
      "cpu_ordered_loop_unroll": True,
      "cpu_state_vector_reuse": True,
      "_vector_state": VectorState(vl, None),
    },
  )
  assert all(plan.pieces for plan in artifact.backend_ir.states.values())
  source = artifact.source
  wrapper = (
    '#include "vecops/Kernel.h"\nnamespace vecops::spec { inline constexpr auto Parallelism=vecops::meta::cint<1>; }\n'
    + source
    + '\nextern "C" void vector_tile(float* a,float* b,float* d,float* c) {\n'
    + "auto A=vecops::tensor::make_tensor(a,vecops::tensor::make_shape(vecops::meta::cint<4>,vecops::meta::cint<17>));\n"
    + f"auto B=vecops::tensor::make_tensor(b,vecops::tensor::make_shape(vecops::meta::cint<17>,vecops::meta::cint<{columns}>));\n"
    + f"auto D=vecops::tensor::make_tensor(d,vecops::tensor::make_shape(vecops::meta::cint<{seed_rows}>,vecops::meta::cint<{columns}>));\n"
    + f"auto C=vecops::tensor::make_tensor(c,vecops::tensor::make_shape(vecops::meta::cint<4>,vecops::meta::cint<{columns}>));\n__kernel__(A,B,D,C);\n}}\n"
  )
  root = Path(__file__).resolve().parents[2]
  cpp, assembly = tmp_path / "vector_dot.cpp", tmp_path / "vector_dot.s"
  cpp.write_text(wrapper)
  command = [
    clang,
    "--target=aarch64-linux-gnu",
    f"--sysroot={sysroot}",
    "-march=armv9-a+sve2",
    "-std=c++20",
    "-O3",
    "-S",
    "-DNDEBUG",
    "-ffp-contract=off",
    f"-DVECOPS_KERNEL_SVE_BYTES={vl}",
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
  cpp.with_suffix(".log").write_text(result.stdout + result.stderr)
  assert result.returncode == 0, result.stderr[-6000:]
  code = assembly.read_text()
  assert re.search(r"^\s*fm(?:la|ad)\s+z", code, re.MULTILINE)
  assert not re.search(r"^\s*\w*mopa\b|^\s*smstart\b", code, re.MULTILINE)
  # LLVM annotates the actual backedge, which can be an unconditional `b`
  # after an early tail exit. Do not look only for conditional backedges.
  header = re.search(r"^(\.LBB\w+):.*Inner Loop Header", code, re.MULTILINE)
  assert header is not None
  backedge = re.search(rf"^\s*b(?:\.\w+)?\s+{re.escape(header[1])}\s*$", code[header.end() :], re.MULTILINE)
  assert backedge is not None
  hot = code[header.end() : header.end() + backedge.end()]
  assert not re.search(r"^\s*cnt[bhwd]\b", hot, re.MULTILINE)
  assert not re.search(r"^\s*cnt[bhwd]\b", code, re.MULTILINE)
  assert len(re.findall(r"^\s*ld1w\b", hot, re.MULTILINE)) == 2  # one shared B per unrolled K
  assert len(re.findall(r"^\s*ld1rw\b", hot, re.MULTILINE)) == 8  # four independent A broadcasts per K
  assert not re.search(r"^\s*(?:bl|st1w|str)\b", hot, re.MULTILINE)
  assert_no_vec_mask_arrays(source)
