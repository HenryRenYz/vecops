"""Native CPU metadata shared by paired DSL/C++ benchmark runners."""

import os
import platform

from vecops._arm_vector_state import get_vector_state


def is_arm():
  return platform.machine().lower() in {"aarch64", "arm64"}


def default_cpu():
  return min(os.sched_getaffinity(0))


def packing_geometry(dtype):
  """Return the actual instruction panel geometry, not an AMX assumption.

  This is queried on the compute node. SME GET_VL is safe outside streaming
  mode; it does not modify the process hardware state.
  """
  if not is_arm():
    return "amx", 16, 32 if dtype in {"bfloat16", "float16"} else 64
  svl_bytes = get_vector_state().sme_bytes
  if svl_bytes is None:
    raise RuntimeError("current thread has no valid SME vector length")
  element_bytes = {"float64": 8, "float32": 4, "float16": 2, "bfloat16": 2, "int8": 1, "uint8": 1}[dtype]
  # Public SME Packing uses two ZA tiles per spatial panel. Instruction
  # extents and packed-panel extents are deliberately distinct.
  return "sme", 2 * (svl_bytes // (8 if dtype == "float64" else 4)), max(1, 4 // element_bytes)


def packed_size(spatial, k, dtype):
  _, panel, group = packing_geometry(dtype)
  return ((spatial + panel - 1) // panel) * ((k + group - 1) // group) * panel * group


def add_toolchain_arguments(parser):
  parser.add_argument("--cc", default=os.environ.get("CC", "gcc"))
  parser.add_argument("--cxx", default=os.environ.get("CXX", "g++"))
  parser.add_argument("--target", default=os.environ.get("VECOPS_TEST_TARGET", "Native"))
