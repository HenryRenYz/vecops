# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Register state geometry is an ISA input, never a cost-profile decision."""

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
from vecops.language._compiler.backends.cpu.capabilities import CPP_TYPES


def _template(shape=(3, 17), *, dtype="float32", output="float32"):
  @T.template
  def kernel(
    A: T.Tensor(shape, dtype=dtype),
    C: T.Tensor(shape, dtype=output, access="output"),
    *,
    count: int = 3,
    choice: bool = False,
  ):
    value = T.load(A)
    for i in T.serial(count):
      old = value
      if choice:
        value = value + old * 0.25 + 1
      else:
        value = value - 0.125
    T.store(C, T.cast(value, output) + 0.5)

  return kernel


def _lower(*, vl=64, svl=64, target="SME", cost_target=None, **kwargs):
  options = {"_vector_state": VectorState(vl, svl)}
  if cost_target is not None:
    options["cpu_cost_target"] = cost_target
  return CPUBackend().lower(_template(**kwargs).ir, (), target=target, options=options)


@pytest.mark.parametrize("vl,svl", ((16, 16), (32, 32), (64, 64), (16, 64), (64, 16)))
@pytest.mark.parametrize("target", ("SVE", "SME"))
def test_captured_sve_state_uses_named_native_words(vl, svl, target):
  artifact = _lower(vl=vl, svl=svl, target=target, shape=(2, 7))
  plans = tuple(artifact.backend_ir.states.values())
  assert plans and all(plan.pieces for plan in plans)
  assert all(piece.scalable for plan in plans for piece in plan.pieces)
  assert all(len(piece.coordinates) == 2 for plan in plans for piece in plan.pieces)
  assert "ScalableTag<float, 0>" in artifact.source
  assert "FixedTag<float, 16>" not in artifact.source
  assert not re.search(r"Buffer<[^>]+> dsl_f_state_", artifact.source)
  assert "state_lane" not in artifact.source
  assert_no_vec_mask_arrays(artifact.source)


@pytest.mark.parametrize("output", ("float32", "float64"))
def test_sve_widening_keeps_a_single_native_word_per_piece(output):
  artifact = _lower(shape=(37,), output=output)
  plans = tuple(artifact.backend_ir.states.values())
  assert plans and all(plan.pieces for plan in plans)
  power = -1 if output == "float64" else 0
  assert all(piece.power == power for plan in plans for piece in plan.pieces)
  assert "state_lane" not in artifact.source
  assert_no_vec_mask_arrays(artifact.source)


def test_unknown_sve_geometry_retains_legal_minimum_fixed_pieces():
  artifact = _lower(vl=None, svl=None, shape=(7,))
  plans = tuple(artifact.backend_ir.states.values())
  assert plans and all(plan.pieces for plan in plans)
  assert all(not piece.scalable and piece.lanes <= 4 for plan in plans for piece in plan.pieces)
  assert_no_vec_mask_arrays(artifact.source)


@pytest.mark.parametrize("target,cost_target", (("SME", "Scalar"), ("Scalar", "AVX512"), ("AVX512", "SME")))
def test_cost_override_cannot_change_state_representation(target, cost_target):
  base = _lower(shape=(7,), target=target)
  override = _lower(shape=(7,), target=target, cost_target=cost_target)
  assert tuple(plan.pieces for plan in base.backend_ir.states.values()) == tuple(
    plan.pieces for plan in override.backend_ir.states.values()
  )


def _wrapper(source, shape, dtype, output):
  dimensions = ",".join(f"vecops::meta::cint<{n}>" for n in shape)
  strides = (1,) if len(shape) == 1 else (shape[-1], 1)
  pitches = ",".join(f"vecops::meta::cint<{s}>" for s in strides)
  tensors = []
  for name, pointer, element in (("A", "a", dtype), ("C", "c", output)):
    tensors.append(
      f"auto {name}=vecops::tensor::make_tensor(static_cast<{CPP_TYPES[element]}*>({pointer}),"
      f"vecops::tensor::make_shape({dimensions}),vecops::tensor::make_strides({pitches}));"
    )
  return "\n".join(
    (
      '#include "vecops/Kernel.h"',
      "namespace vecops::spec { inline constexpr auto Parallelism = vecops::meta::cint<1>; }",
      source,
      'extern "C" void register_state(void* a,void* c,vecops::nint_t count,bool choice) {',
      *tensors,
      "__kernel__(A,C,count,choice);",
      "}",
    )
  )


@pytest.mark.parametrize("vl,svl,fixed", ((16, 64, False), (32, 32, False), (64, 16, False), (64, 64, True)))
@pytest.mark.parametrize("shape,dtype,output", (((2, 7), "float32", "float32"), ((37,), "float32", "float64")))
def test_arm_cross_register_state_in_vla_and_vls(tmp_path, vl, svl, fixed, shape, dtype, output):
  clang = os.environ.get("VECOPS_TEST_ARM_CROSS_CLANG")
  sysroot = os.environ.get("VECOPS_TEST_ARM_CROSS_SYSROOT")
  if not clang or not sysroot:
    pytest.skip("set VECOPS_TEST_ARM_CROSS_CLANG/SYSROOT for actual ARM codegen checks")
  artifact = _lower(vl=vl, svl=svl, shape=shape, dtype=dtype, output=output)
  cpp, assembly = tmp_path / "state.cpp", tmp_path / "state.s"
  cpp.write_text(_wrapper(artifact.source, shape, dtype, output), encoding="utf-8")
  root = Path(__file__).resolve().parents[2]
  command = [
    clang,
    "--target=aarch64-linux-gnu",
    f"--sysroot={sysroot}",
    "-march=armv9.2-a+sme+sme-f64f64+bf16",
    "-std=c++20",
    "-O3",
    "-ffp-contract=off",
    "-S",
    "-DNDEBUG",
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
  if fixed:
    command.insert(5, f"-msve-vector-bits={vl * 8}")
  result = subprocess.run(command, capture_output=True, text=True, check=False)
  cpp.with_suffix(".log").write_text(result.stdout + result.stderr, encoding="utf-8")
  assert result.returncode == 0, result.stderr[-6000:]
  text = assembly.read_text(encoding="utf-8")
  assert re.search(r"\b(?:ld1w|ld1d|st1w|st1d)\b", text)
  assert not re.search(r"\bbl\b[^\n]*(?:4load|3add|3mul|7convert)", text)
  assert not re.search(r"\b(?:st1w|st1d)\b[^\n]*\[sp", text)


@pytest.mark.parametrize("shape,output", (((37,), "float32"), ((3, 17), "float32"), ((37,), "float64")))
def test_native_register_state_branch_snapshot_and_widening(tmp_path, shape, output):
  compiler = make_compiler(target="Native", jobs=4, cache_dir=tmp_path / "cache", build_dir=tmp_path / "build")
  compiled = T.compile(_template(shape=shape, output=output), compiler=compiler)
  a = np.arange(np.prod(shape), dtype=np.float32).reshape(shape) / 16
  c = np.empty(shape, dtype=output)
  for count in (0, 1, 3):
    for choice in (False, True):
      expected = a.copy()
      for _ in range(count):
        old = expected.copy()
        expected = expected + old * np.float32(0.25) + 1 if choice else expected - np.float32(0.125)
      compiled(a, c, count=count, choice=choice)
      np.testing.assert_array_equal(c, expected.astype(output) + 0.5)
  assert not re.search(r"Buffer<[^>]+> dsl_f_state_", compiled.generated_source)
  assert "state_lane" not in compiled.generated_source
  assert_no_vec_mask_arrays(compiled.generated_source)
