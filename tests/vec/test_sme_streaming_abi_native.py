"""Callee-saved ABI and numerical regression for manual SME state owners.

Only scalar metadata crosses the normal/streaming boundary. The high-pressure
probe deliberately spills scalable values; numerical correctness alone does
not detect an incorrectly sized frame that overwrites the caller's d8--d15.
Each native executable changes its own VL/SVL and restores complete prctl flags.
"""

from __future__ import annotations

import ctypes
import json
import os
import platform
import re
import shlex
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from arm_sme_asm_utils import assert_sme_owners

ROOT = Path(__file__).resolve().parents[2]


def raw_lengths():
  """Read full process/thread flags without importing a retained extension."""
  libc = ctypes.CDLL(None, use_errno=True)
  libc.prctl.restype = ctypes.c_int
  result = []
  for option in (51, 64):
    value = libc.prctl(ctypes.c_int(option), *[ctypes.c_ulong(0) for _ in range(4)])
    if value < 0:
      error = ctypes.get_errno()
      raise OSError(error, os.strerror(error))
    result.append(value)
  return tuple(result)


def variants_for(normal, streaming):
  """Cover supported equal powers and both directions of unequal domains."""
  for length in (normal, streaming):
    if not 16 <= length <= 256 or length & (length - 1):
      raise ValueError("power-of-two VL/SVL in [16, 256] required")
  equal, length = set(), 16
  while length <= min(normal, streaming):
    equal.add((length, length))
    length *= 2
  pairs = sorted({*equal, (normal, streaming), (16, streaming), (normal, 16)})
  variants = []
  for vl, svl in pairs:
    variants.extend(((vl, svl, "scalable", True), (vl, svl, "scalable", False)))
    if vl == svl:
      variants.append((vl, svl, str(vl * 8), True))
  return tuple(variants)


def target_options(vl, svl, bits, captured):
  options = [
    "-march=armv9.2-a+sme+sme-fa64+sme-f64f64+bf16+i8mm",
    f"-msve-vector-bits={bits}",
    "-DVECOPS_TARGET_SME_FA64=1",
    "-DVECOPS_TARGET_SME_F64F64=1",
  ]
  if captured:
    options.extend(
      (
        f"-DVECOPS_KERNEL_SVE_BYTES={vl}",
        f"-DVECOPS_KERNEL_SME_BYTES={svl}",
        f"-DVECOPS_TARGET_FIXED_STREAMING_SVE_BITS={svl * 8}",
      )
    )
  return options


def invoke(command, label):
  label.with_suffix(".command.json").write_text(json.dumps(command, indent=2) + "\n")
  result = subprocess.run(command, capture_output=True, text=True, check=False)
  label.with_suffix(".stdout.log").write_text(result.stdout)
  label.with_suffix(".stderr.log").write_text(result.stderr)
  label.with_suffix(".exitcode").write_text(f"{result.returncode}\n")
  assert result.returncode == 0, (command, result.returncode, result.stdout, result.stderr)
  return result.stdout


def check_native_output(output, vl, svl):
  assert f"queried_vl={vl} queried_svl={svl} numerical_exit=0" in output
  canaries = re.findall(r"^d(\d+) expected=([0-9a-f]{16}) actual=([0-9a-f]{16})$", output, re.MULTILINE)
  assert len(canaries) == 8
  for index, (register, expected, actual) in enumerate(canaries, 8):
    assert int(register) == index
    assert expected == actual == f"{index * 0x0101010101010101:016x}"
  restore = re.findall(r"before_vl=(\d+) before_svl=(\d+) after_vl=(\d+) after_svl=(\d+) abi_exit=(\d+)", output)
  assert len(restore) == 1
  before_vl, before_svl, after_vl, after_svl, code = restore[0]
  assert (before_vl, before_svl, code) == (after_vl, after_svl, "0")


@pytest.mark.skipif(
  platform.machine().lower() not in {"aarch64", "arm64"} or sys.platform != "linux",
  reason="Linux AArch64 scheduled SME target required",
)
def test_native_scalable_sme_owner_callee_saved_spills(tmp_path):
  try:
    state = raw_lengths()
  except OSError:
    pytest.skip("CPU/OS must support SVE and SME")
  variants = variants_for(state[0] & 65535, state[1] & 65535)
  cxx = os.environ.get("CXX", "clang++")
  common = [cxx, *shlex.split(os.environ.get("CXXFLAGS", "")), "-O3", "-DNDEBUG", "-MD"]
  includes = ["-I", str(ROOT / "include"), "-I", str(ROOT / "include/vecops")]
  asm, driver = tmp_path / "canary.o", tmp_path / "driver.o"
  invoke(
    [*common, "-march=armv8-a", "-c", str(Path(__file__).with_name("SMEStreamingAbiCanary.S")), "-o", str(asm)],
    tmp_path / "compile-canary",
  )
  invoke(
    [
      *common,
      "-std=c++20",
      "-march=armv8-a",
      *includes,
      "-c",
      str(Path(__file__).with_name("SMEStreamingAbiDriver.cpp")),
      "-o",
      str(driver),
    ],
    tmp_path / "compile-driver",
  )

  def exercise(variant):
    vl, svl, bits, captured = variant
    folder = tmp_path / f"pair-{vl}-{svl}-bits-{bits}-captured-{int(captured)}"
    folder.mkdir()
    flags = [
      *common,
      "-std=c++20",
      "-ffp-contract=off",
      *target_options(*variant),
      *includes,
      "-Dmain=vecops_stride_streaming_main",
    ]
    source = str(Path(__file__).with_name("SMEStreamingAbiProbe.cpp"))
    assembly = invoke([*flags, "-MF", str(folder / "assembly.d"), "-S", source, "-o", "-"], folder / "assembly")
    bodies = assert_sme_owners(assembly, 14, equal_domains=vl == svl and (captured or bits != "scalable"))
    assert any("mul vl" in body or "addvl" in body for body in bodies.values()), "probe lost scalable spill pressure"
    probe, binary = folder / "probe.o", folder / "native"
    invoke([*flags, "-MF", str(folder / "probe.d"), "-c", source, "-o", str(probe)], folder / "compile-probe")
    invoke(
      [cxx, str(asm), str(driver), str(probe), *shlex.split(os.environ.get("LDFLAGS", "")), "-o", str(binary)],
      folder / "link",
    )
    output = invoke([str(binary), str(vl), str(svl)], folder / "native-run")
    check_native_output(output, vl, svl)
    return {
      "vl": vl,
      "svl": svl,
      "bits": bits,
      "captured": captured,
      "owners": len(bodies),
      "numerical_and_eight_abi_canaries": "PASS",
    }

  # Compilation and execution are independent subprocesses: a child's prctl
  # never changes the parent's or another variant's vector lengths.
  workers = min(4, len(os.sched_getaffinity(0)))
  with ThreadPoolExecutor(max_workers=workers) as executor:
    records = list(executor.map(exercise, variants))
  assert raw_lengths() == state
  (tmp_path / "native-evidence.json").write_text(json.dumps(records, indent=2) + "\n")


def test_prepared_variants_do_not_assume_a_machine_wide_vector_length():
  variants = variants_for(64, 64)
  assert len(variants) == len(set(variants)) == 13
  assert {variant[:2] for variant in variants} == {(16, 16), (32, 32), (64, 64), (16, 64), (64, 16)}
  assert len(variants_for(32, 16)) == 5
  assert len(variants_for(16, 16)) == 3
  for variant in variants:
    flags = target_options(*variant)
    assert ("-DVECOPS_KERNEL_SVE_BYTES=" in " ".join(flags)) is variant[3]
  with pytest.raises(ValueError):
    variants_for(48, 64)


def test_prepared_canary_oracle_rejects_corruption_and_raw_flag_loss():
  canaries = "\n".join(
    f"d{index} expected={index * 0x0101010101010101:016x} actual={index * 0x0101010101010101:016x}"
    for index in range(8, 16)
  )
  output = (
    "queried_vl=16 queried_svl=32 numerical_exit=0\n"
    + canaries
    + "\nbefore_vl=65552 before_svl=65568 after_vl=65552 after_svl=65568 abi_exit=0\n"
  )
  check_native_output(output, 16, 32)
  for corrupted in (
    output.replace("actual=0a0a0a0a0a0a0a0a", "actual=0000000000000000"),
    output.replace("after_vl=65552", "after_vl=16"),
    output.replace("numerical_exit=0", "numerical_exit=11"),
    output.replace("d15 expected", "d14 expected"),
  ):
    with pytest.raises(AssertionError):
      check_native_output(corrupted, 16, 32)
