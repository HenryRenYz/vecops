"""Actual SME VLS moves/direct memory/MOPA at supported equal VL/SVL.

Prepared target-only regression; cross success is not this Native test's pass.
The prctl raw flags are restored in finally, with scalar SVCR checked before
each state transition. All vector data stays inside a C++ owner; no live Vec
exists while Python changes the state. There is no JIT/cache/compiler fallback.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
import platform
import shlex
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

native_arm = pytest.mark.skipif(
  platform.machine().lower() not in {"aarch64", "arm64"} or sys.platform != "linux",
  reason="actual Linux AArch64 SME compute allocation required",
)
SVE_GET, SVE_SET, SME_GET, SME_SET, LENGTH_MASK = 51, 50, 64, 63, 65535


def _sha(path):
  return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class VectorControl:
  def __init__(self, library):
    self.library = library
    self.libc = ctypes.CDLL(None, use_errno=True)
    self.libc.prctl.restype = ctypes.c_int
    self.original = self.raw(SVE_GET), self.raw(SME_GET)

  def raw(self, option, value=0):
    result = self.libc.prctl(
      ctypes.c_int(option), ctypes.c_ulong(value), ctypes.c_ulong(0), ctypes.c_ulong(0), ctypes.c_ulong(0)
    )
    if result < 0:
      error = ctypes.get_errno()
      raise OSError(error, os.strerror(error))
    return result

  def clear(self):
    assert self.library.za_asm_read_svcr() & 3 == 0, "live SM/ZA state at Python boundary"

  def set(self, option, value):
    self.clear()
    return self.raw(option, value)

  def select_equal(self, length):
    assert self.set(SVE_SET, length) & LENGTH_MASK == length, "requested ordinary VL not supported"
    assert self.set(SME_SET, length) & LENGTH_MASK == length, "requested streaming SVL not supported"
    self.clear()
    assert (self.raw(SVE_GET) & LENGTH_MASK, self.raw(SME_GET) & LENGTH_MASK) == (length, length)
    assert self.library.za_asm_sve_bytes() == self.library.za_asm_sme_bytes() == length

  def restore(self):
    self.set(SVE_SET, self.original[0])
    self.set(SME_SET, self.original[1])
    assert (self.raw(SVE_GET), self.raw(SME_GET)) == self.original
    self.clear()


def _build(tmp_path, length):
  root = Path(__file__).resolve().parents[2]
  include = (
    Path(os.environ["VECOPS_TEST_SME_ASM_SDK"]) / "include"
    if "VECOPS_TEST_SME_ASM_SDK" in os.environ
    else root / "include"
  )
  assert (include / "vecops/vec/details/sme/ZA.h").read_bytes() == (
    root / "include/vecops/vec/details/sme/ZA.h"
  ).read_bytes()
  source = Path(__file__).with_name("SMEAsmTypesNative.cpp")
  copied, library = tmp_path / "native.cpp", tmp_path / "native.so"
  copied.write_bytes(source.read_bytes())
  cxx = os.environ.get("CXX", "clang++")
  command = [
    cxx,
    *shlex.split(os.environ.get("CXXFLAGS", "")),
    "-std=c++20",
    "-O3",
    "-ffp-contract=off",
    "-fPIC",
    "-shared",
    "-march=armv9.2-a+sme+sme-f64f64+bf16",
    f"-msve-vector-bits={length * 8}",
    "-DVECOPS_TARGET_SME_FA64=1",
    "-DVECOPS_TARGET_SME_F64F64=1",
    f"-DVECOPS_TARGET_FIXED_STREAMING_SVE_BITS={length * 8}",
    f"-DVECOPS_KERNEL_SVE_BYTES={length}",
    f"-DVECOPS_KERNEL_SME_BYTES={length}",
    "-I",
    str(include),
    "-I",
    str(include / "vecops"),
    str(copied),
    *shlex.split(os.environ.get("LDFLAGS", "")),
    "-o",
    str(library),
  ]
  (tmp_path / "command.json").write_text(json.dumps(command, indent=2) + "\n")
  result = subprocess.run(command, capture_output=True, text=True, check=False)
  (tmp_path / "compiler.log").write_text(result.stdout + result.stderr)
  assert result.returncode == 0, result.stderr
  loaded = ctypes.CDLL(str(library))
  for name in ("za_asm_read_svcr", "za_asm_sve_bytes", "za_asm_sme_bytes"):
    getattr(loaded, name).restype = ctypes.c_uint64
    getattr(loaded, name).argtypes = []
  return loaded, {
    "library_sha256": _sha(library),
    "source_sha256": _sha(copied),
    "header_sha256": _sha(include / "vecops/vec/details/sme/ZA.h"),
    "command": command,
  }


def _raw_values(dtype, count):
  dtype = np.dtype(dtype)
  unsigned = np.dtype(f"uint{dtype.itemsize * 8}")
  if dtype.kind == "f":
    patterns = (
      [0, 0x80000000, 0x7FC12345, 0x7FA23456, 0x7F800000, 0xFF800000, 0x3F800000]
      if dtype.itemsize == 4
      else [
        0,
        0x8000000000000000,
        0x7FF8123456789ABC,
        0x7FF0123456789ABC,
        0x7FF0000000000000,
        0xFFF0000000000000,
        0x3FF0000000000000,
      ]
    )
    return np.resize(np.array(patterns, dtype=unsigned), count).view(dtype)
  return (np.arange(count, dtype=unsigned) * unsigned.type(13) + unsigned.type(7)).view(dtype)


def _roundtrips(library, control, length):
  calls = 0
  for family, dtypes in (
    ("move", ("uint8", "uint16", "uint32", "uint64", "int32", "float32", "float64")),
    ("memory", ("uint8", "uint16", "uint32", "uint64")),
  ):
    for dtype in dtypes:
      code = {
        "uint8": "u8",
        "uint16": "u16",
        "uint32": "u32",
        "uint64": "u64",
        "int32": "s32",
        "float32": "f32",
        "float64": "f64",
      }[dtype]
      function = getattr(library, f"za_native_{family}_{code}")
      function.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_bool]
      function.restype = None
      lanes = length // np.dtype(dtype).itemsize
      for count in sorted({0, 1, max(0, lanes - 1), lanes}):
        for vertical in (False, True):
          for alias in ("none", "inplace", "shifted"):
            data = _raw_values(dtype, 3 * lanes + 8).copy()
            input_view = data[2 : 2 + lanes]
            expected = input_view[:count].view(np.uint8).copy()
            output_view = (
              np.empty(lanes + 4, dtype=dtype)[2 : 2 + lanes]
              if alias == "none"
              else input_view
              if alias == "inplace"
              else data[3 : 3 + lanes]
            )
            # The expected whole backing bytes include all inactive canaries.
            output_owner = output_view.base
            if alias == "none":
              output_owner.view(np.uint8)[:] = 0xA5
            before = output_owner.view(np.uint8).copy()
            offset = output_view.ctypes.data - output_owner.ctypes.data
            before[offset : offset + len(expected)] = expected
            function(input_view.ctypes.data, output_view.ctypes.data, count, vertical)
            control.clear()
            np.testing.assert_array_equal(
              output_owner.view(np.uint8), before, err_msg=f"{length}/{family}/{dtype}/{count}/{vertical}/{alias}"
            )
            calls += 1
  return calls


def _outer_products(library, control, length):
  import ml_dtypes  # Required on target; absence is setup error, not a skip.

  cases = (
    ("f32", "float32", "float32", "float32", 1),
    ("f16", "float16", "float16", "float32", 2),
    ("bf16", ml_dtypes.bfloat16, ml_dtypes.bfloat16, "float32", 2),
    ("f64", "float64", "float64", "float64", 1),
    ("ss8", "int8", "int8", "int32", 4),
    ("uu8", "uint8", "uint8", "uint32", 4),
    ("su8", "int8", "uint8", "int32", 4),
    ("us8", "uint8", "int8", "int32", 4),
  )
  calls = 0
  for code, left_type, right_type, result_type, group in cases:
    lanes = length // np.dtype(result_type).itemsize
    function = getattr(library, f"za_native_mopa_{code}")
    function.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64]
    function.restype = None
    left = (np.arange(lanes * group) % 5 - (0 if np.dtype(left_type).kind == "u" else 2)).astype(left_type)
    right = (np.arange(lanes * group) % 7 - (0 if np.dtype(right_type).kind == "u" else 3)).astype(right_type)
    if result_type.startswith("float"):
      left = (left.astype("float64") / 4).astype(left_type)
      right = (right.astype("float64") / 8).astype(right_type)
    # All products/sums are small exact binary rationals (or exact integers).
    expected = (
      left.astype("float64" if result_type.startswith("float") else "int64").reshape(lanes, group)
      @ right.astype("float64" if result_type.startswith("float") else "int64").reshape(lanes, group).T
    )
    expected = expected.astype(result_type)
    for rows, columns in ((0, lanes), (lanes, 0), (1, 1), (max(1, lanes - 1), max(1, lanes - 1)), (lanes, lanes)):
      for alias in (False, True):
        storage = np.zeros(lanes * lanes + 4, dtype=result_type)
        output = storage[2:-2].reshape(lanes, lanes)
        a = left.copy()
        if alias:
          # Raw byte alias is valid even across element types. The full input
          # load completes before any output store and the oracle uses prior A.
          needed = left.nbytes
          assert needed <= output.nbytes
          output.view(np.uint8).reshape(-1)[:needed] = left.view(np.uint8)
          a = np.ndarray(left.shape, dtype=left.dtype, buffer=storage, offset=2 * storage.itemsize)
        before = storage.copy()
        before[2:-2].reshape(lanes, lanes)[:rows, :columns] = expected[:rows, :columns]
        function(a.ctypes.data, right.ctypes.data, output.ctypes.data, rows, columns)
        control.clear()
        np.testing.assert_array_equal(
          storage.view(np.uint8), before.view(np.uint8), err_msg=f"{length}/{code}/{rows}/{columns}/alias={alias}"
        )
        calls += 1
  return calls


@native_arm
@pytest.mark.parametrize("length", (16, 64))
def test_native_sme_equal_vls_public_instruction_bits_masks_aliases(tmp_path, length):
  library, evidence = _build(tmp_path, length)
  control = VectorControl(library)
  evidence["raw_prctl_before"] = control.original
  try:
    control.select_equal(length)
    evidence["actual_pair"] = (library.za_asm_sve_bytes(), library.za_asm_sme_bytes())
    evidence["roundtrip_calls"] = _roundtrips(library, control, length)
    evidence["outer_product_calls"] = _outer_products(library, control, length)
    assert evidence["roundtrip_calls"] == (246 if length == 16 else 264)
    assert evidence["outer_product_calls"] == 80
  finally:
    control.restore()
    evidence["raw_prctl_after"] = (control.raw(SVE_GET), control.raw(SME_GET))
    (tmp_path / "native-identity.json").write_text(json.dumps(evidence, indent=2) + "\n")


def test_prepared_raw_oracle_checks_inactive_and_alias_bytes():
  """Exercise the fixture on this host; this is not execution of SME code."""

  class Control:
    def clear(self):
      pass

  class ScalarMock:
    def __init__(self, corrupt=False):
      self.corrupt = corrupt

    def __getattr__(self, name):
      code = name.rsplit("_", 1)[-1]
      itemsize = {"u8": 1, "u16": 2, "u32": 4, "u64": 8, "s32": 4, "f32": 4, "f64": 8}[code]

      def move(source, destination, count, vertical):
        prior = ctypes.string_at(source, count * itemsize)
        ctypes.memmove(destination, prior, len(prior))
        if self.corrupt:
          ctypes.c_uint8.from_address(destination + count * itemsize).value ^= 1

      return move

  assert _roundtrips(ScalarMock(), Control(), 16) == 246
  assert _roundtrips(ScalarMock(), Control(), 64) == 264
  with pytest.raises(AssertionError):
    _roundtrips(ScalarMock(corrupt=True), Control(), 16)


def test_prepared_outer_product_oracle_checks_cross_dtype_alias():
  """Validate pointer geometry/expected-byte checks without claiming Native."""
  import ml_dtypes

  class Control:
    def clear(self):
      pass

  types = {
    "f32": ("float32", "float32", "float32", 1),
    "f16": ("float16", "float16", "float32", 2),
    "bf16": (ml_dtypes.bfloat16, ml_dtypes.bfloat16, "float32", 2),
    "f64": ("float64", "float64", "float64", 1),
    "ss8": ("int8", "int8", "int32", 4),
    "uu8": ("uint8", "uint8", "uint32", 4),
    "su8": ("int8", "uint8", "int32", 4),
    "us8": ("uint8", "int8", "int32", 4),
  }

  class ScalarMock:
    def __init__(self, length, corrupt=False):
      self.length, self.corrupt = length, corrupt

    def __getattr__(self, name):
      left_type, right_type, result_type, group = types[name.rsplit("_", 1)[-1]]
      lanes = self.length // np.dtype(result_type).itemsize

      def matrix(a, b, output, rows, columns):
        def load(pointer, dtype, count):
          return np.frombuffer(ctypes.string_at(pointer, count * np.dtype(dtype).itemsize), dtype=dtype).copy()

        # Scalar mock computes every element using independent explicit sums.
        # Copy inputs first so output alias cannot destroy prior operands.
        left = load(a, left_type, lanes * group).astype("float64" if result_type.startswith("float") else "int64")
        right = load(b, right_type, lanes * group).astype(left.dtype)
        for row in range(rows):
          for column in range(columns):
            value = sum(left[row * group + k] * right[column * group + k] for k in range(group))
            data = np.array([value], dtype=result_type).tobytes()
            ctypes.memmove(output + (row * lanes + column) * len(data), data, len(data))
        if self.corrupt:
          ctypes.c_uint8.from_address(output).value ^= 1

      return matrix

  assert _outer_products(ScalarMock(16), Control(), 16) == 80
  assert _outer_products(ScalarMock(64), Control(), 64) == 80
  with pytest.raises(AssertionError):
    _outer_products(ScalarMock(16, corrupt=True), Control(), 16)
