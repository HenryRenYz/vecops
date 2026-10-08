# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Canonical dtype names do not depend on optional framework imports."""

import subprocess
import sys

import pytest
import vecops
import vecops.language as T
from vecops._dtype import DTYPE_CONSTANTS, ELEMENT_BYTES, dtype_itemsize, dtype_name, framework_dtype, is_native_dtype


@pytest.mark.parametrize(
  "name",
  [
    "bool",
    "bool_",
    "int8",
    "uint8",
    "int16",
    "uint16",
    "int32",
    "uint32",
    "int64",
    "uint64",
    "float16",
    "bfloat16",
    "float32",
    "float64",
  ],
)
def test_canonical_dtype_name(name):
  expected = getattr(vecops, "bool_" if name == "bool" else name)
  assert vecops.normalize_dtype(name) == expected
  assert T.DType(name).value == expected
  assert vecops.normalize_dtype(T.DType(name)) == expected
  canonical = "bool" if name == "bool_" else name
  assert dtype_name(expected) == canonical
  assert dtype_itemsize(T.DType(name)) == ELEMENT_BYTES[canonical]


def test_language_dtype_exports_follow_the_shared_registry():
  for name, dtype in DTYPE_CONSTANTS.items():
    assert getattr(T, name) is dtype
    assert name in T.__all__
    assert is_native_dtype(dtype)
  assert not is_native_dtype("float32")
  assert not is_native_dtype(None)


def test_canonical_names_without_framework_imports():
  result = subprocess.run(
    [
      sys.executable,
      "-c",
      """
import sys
import vecops
import vecops.language as T
from vecops._dtype import _NAMES
for name, expected in _NAMES.items():
    assert vecops.normalize_dtype(name) == expected
    assert T.DType(name).value == expected
assert 'numpy' not in sys.modules
assert 'torch' not in sys.modules
""",
    ],
    check=False,
    capture_output=True,
    text=True,
  )
  assert result.returncode == 0, result.stdout + result.stderr


def test_unknown_dtype_name_is_rejected():
  with pytest.raises(TypeError, match="unsupported dtype"):
    vecops.normalize_dtype("not-a-dtype")


@pytest.mark.parametrize("name", tuple(ELEMENT_BYTES))
def test_torch_dtypes_and_jit_keys(name):
  torch = pytest.importorskip("torch")
  actual = getattr(torch, name, None)
  if actual is None:
    pytest.skip(f"Torch does not expose {name}")
  from vecops.language.expr import cache_key, normalize_value

  expected = vecops.normalize_dtype(name)
  assert vecops.normalize_dtype(actual) == expected
  assert framework_dtype(T.DType(name), "torch") == actual
  assert cache_key(normalize_value(name, name="dtype")) == cache_key(normalize_value(actual, name="dtype"))


@pytest.mark.parametrize("name", tuple(n for n in ELEMENT_BYTES if n != "bfloat16"))
def test_numpy_dtypes_and_native_views(name):
  np = pytest.importorskip("numpy")
  from vecops.language.expr import cache_key, normalize_value

  actual = np.dtype(name)
  expected = vecops.normalize_dtype(name)
  assert vecops.normalize_dtype(actual) == expected
  assert framework_dtype(T.DType(name), "numpy") == actual
  assert dtype_itemsize(actual) == actual.itemsize
  assert cache_key(normalize_value(name, name="dtype")) == cache_key(normalize_value(actual, name="dtype"))
  view = vecops._C.TensorView(np.zeros(3, dtype=actual))
  assert view.dtype == expected


def test_unspecified_and_non_native_dtypes_are_not_concrete():
  np = pytest.importorskip("numpy")
  with pytest.raises(TypeError, match="unsupported dtype"):
    vecops.normalize_dtype(None)
  swapped = np.dtype("float32").newbyteorder("S")
  with pytest.raises(TypeError, match="native-endian"):
    vecops.normalize_dtype(swapped)
  with pytest.raises(TypeError, match="native-endian"):
    vecops._C.TensorView(np.zeros(3, dtype=swapped))
  with pytest.raises(TypeError, match="unsupported NumPy dtype"):
    vecops._C.TensorView(np.zeros(3, dtype="V2"))


def test_jit_strings_and_source_schema_symbols():
  from vecops.language._compiler.backends.cpu.source import _lower_tensor_dtype
  from vecops.language.expr import normalize_value

  assert normalize_value("not-a-dtype") == "not-a-dtype"
  # Framework aliases must not change ordinary string JIT keys on import.
  assert normalize_value("d") == "d"
  assert _lower_tensor_dtype("bfloat16") == vecops.bfloat16
  assert _lower_tensor_dtype("Element") == "Element"
  assert _lower_tensor_dtype(T.DType("float32", name="dtype")) == "dtype"


def test_numpy_bfloat16_extension_and_jit_keys():
  np = pytest.importorskip("numpy")
  ml_dtypes = pytest.importorskip("ml_dtypes")
  from vecops.language.expr import cache_key, normalize_value

  actual = np.dtype(ml_dtypes.bfloat16)
  assert vecops.normalize_dtype(actual) == vecops.bfloat16
  assert vecops.normalize_dtype(ml_dtypes.bfloat16) == vecops.bfloat16
  assert framework_dtype(T.DType("bfloat16"), "numpy") == actual
  assert cache_key(normalize_value("bfloat16")) == cache_key(normalize_value(ml_dtypes.bfloat16))
  assert vecops._C.TensorView(np.zeros(3, dtype=actual)).dtype == vecops.bfloat16


def test_numpy_bfloat16_without_extension_fails_explicitly():
  result = subprocess.run(
    [
      sys.executable,
      "-c",
      """
import sys
import numpy
from vecops._dtype import framework_dtype
try:
    framework_dtype('bfloat16', 'numpy')
except TypeError as error:
    assert 'registered extension' in str(error)
else:
    raise AssertionError('BF16 must not fall back to another dtype')
assert 'ml_dtypes' not in sys.modules
assert 'torch' not in sys.modules
""",
    ],
    check=False,
    capture_output=True,
    text=True,
  )
  assert result.returncode == 0, result.stdout + result.stderr


def test_torch_bridge_covers_all_vecops_dtypes():
  from vecops._schema_bridge import _dtype_abi, _generate_torch_source

  definition = vecops.KernelDef(
    "dtype_bridge_coverage",
    [vecops.ValueDef(name, getattr(vecops, "bool_" if name == "bool" else name)) for name in ELEMENT_BYTES],
  )
  source = _generate_torch_source("dtype_tests", "coverage", definition._native())
  assert set(_dtype_abi) == set(ELEMENT_BYTES)
  for abi in _dtype_abi.values():
    assert f"return {abi};" in source


def test_framework_storage_dtypes_are_not_kernel_dtypes():
  np = pytest.importorskip("numpy")
  torch = pytest.importorskip("torch")
  from vecops.planner.runtime import _torch_dtype

  assert framework_dtype("torch.bfloat16", "torch") == torch.bfloat16
  assert _torch_dtype("torch.complex64") == torch.complex64
  assert framework_dtype("complex64", "numpy", allow_framework_only=True) == np.dtype("complex64")
  with pytest.raises(TypeError, match="unsupported dtype"):
    framework_dtype("complex64", "numpy")
  with pytest.raises(TypeError, match="unsupported dtype"):
    framework_dtype(torch.complex64, "torch")
  with pytest.raises(TypeError, match="unsupported dtype"):
    _torch_dtype("not-a-dtype")


def test_memory_numpy_uses_shared_dtype_adapter():
  np = pytest.importorskip("numpy")
  if not hasattr(vecops, "memory"):
    pytest.skip("native memory support is unavailable")
  from vecops.memory import numpy as memory_numpy

  class System:
    def __init__(self):
      self.sizes = []

    def allocate(self, size, **options):
      self.sizes.append(size)
      return bytearray(size)

  system = System()
  for value, expected in [
    (vecops.float32, np.dtype("float32")),
    (T.DType("uint16"), np.dtype("uint16")),
    (np.complex64, np.dtype("complex64")),
  ]:
    output = memory_numpy.zeros(3, dtype=value, system=system)
    assert output.dtype == expected
    assert system.sizes[-1] == 3 * expected.itemsize
    np.testing.assert_array_equal(output, 0)
