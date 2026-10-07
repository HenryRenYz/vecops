"""Tests for read-only, thread-scoped ARM vector-state detection."""

from vecops._arm_vector_state import (
  PR_SME_GET_VL,
  PR_SVE_GET_VL,
  VectorState,
  get_vector_state,
)

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmarks"))
import dsl_matmul_catalog


def test_non_arm_and_non_linux_are_stably_unavailable():
  assert get_vector_state(machine="x86_64", system="linux") == VectorState(None, None)
  assert get_vector_state(machine="aarch64", system="darwin") == VectorState(None, None)


def test_get_operations_are_read_only_and_lengths_validate():
  requests = []

  def prctl(option):
    requests.append(option)
    return 64 if option == PR_SVE_GET_VL else 32

  state = get_vector_state(machine="aarch64", system="linux", prctl=prctl)
  assert requests == [PR_SVE_GET_VL, PR_SME_GET_VL]
  assert state == VectorState(64, 32)
  assert state.compiler_flags == (
    "-msve-vector-bits=512",
    "-msve-streaming-vector-bits=256",
  )
  assert state.cmake_variables == {
    "VECOPS_FIXED_SVE_BITS": "512",
    "VECOPS_FIXED_STREAMING_SVE_BITS": "256",
  }


def test_missing_invalid_or_unsupported_prctl_results_are_none():
  values = {PR_SVE_GET_VL: -1, PR_SME_GET_VL: 24}  # 192 bits is not power-of-two.
  state = get_vector_state(machine="aarch64", system="linux", prctl=lambda option: values[option])
  assert state == VectorState(None, None)
  assert state.env_token == "arm-vector-state:v1:sve=none:sme=none"


def test_no_global_cache_and_token_tracks_thread_vector_changes():
  current = {PR_SVE_GET_VL: 32, PR_SME_GET_VL: 32}
  query = lambda option: current[option]

  first = get_vector_state(machine="aarch64", system="linux", prctl=query)
  current[PR_SVE_GET_VL] = 64
  current[PR_SME_GET_VL] = 64
  second = get_vector_state(machine="aarch64", system="linux", prctl=query)

  assert first == VectorState(32, 32)
  assert second == VectorState(64, 64)
  assert first.env_token != second.env_token


def test_matmul_catalog_uses_explicit_streaming_vector_length():
  cases32 = dsl_matmul_catalog.load_cases(svl_bytes=32)
  cases64 = dsl_matmul_catalog.load_cases(svl_bytes=64)
  cases128 = dsl_matmul_catalog.load_cases(svl_bytes=128)

  def below_tile(cases):
    return next(
      case
      for case in cases
      if case.catalog == "MatmulShape" and case.name == "workload_shape_cases/shape_coverage/below_tile"
    )

  small, medium, large = below_tile(cases32), below_tile(cases64), below_tile(cases128)
  assert (small.m, medium.m, large.m) == (7, 15, 31)
  assert (small.lane, medium.lane, large.lane) == (8, 16, 32)
  assert (small.svl_bytes, medium.svl_bytes, large.svl_bytes) == (32, 64, 128)


def test_matmul_catalog_rejects_invalid_explicit_streaming_vector_length():
  import pytest

  with pytest.raises(ValueError, match="svl_bytes"):
    dsl_matmul_catalog.load_cases(svl_bytes=48)
