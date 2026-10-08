# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Read the calling thread's current ARM SVE/SME vector lengths.

Linux ``prctl`` GET operations are read-only and thread scoped.  Nothing is
cached here: callers should include :attr:`VectorState.env_token` in their
cache identity so a later lookup observes a changed per-thread setting.
"""

from __future__ import annotations

import ctypes
import dataclasses
import platform
import sys
from typing import Callable

PR_SVE_GET_VL = 51
PR_SME_GET_VL = 64
VL_LEN_MASK = 0xFFFF
MIN_VECTOR_BITS = 128
MAX_VECTOR_BITS = 2048


def _validated_bytes(raw_result: int | None) -> int | None:
  if raw_result is None or raw_result < 0:
    return None
  length = int(raw_result) & VL_LEN_MASK  # Linux returns VL in bytes.
  bits = length * 8
  if bits < MIN_VECTOR_BITS or bits > MAX_VECTOR_BITS:
    return None
  if bits & (bits - 1):
    return None
  return length


def _system_prctl(option: int) -> int | None:
  if not sys.platform.startswith("linux"):
    return None
  try:
    libc = ctypes.CDLL(None, use_errno=True)
    # prctl is variadic; leave argtypes unset and supply zero-valued arguments.
    result = libc.prctl(option, 0, 0, 0, 0)
    return int(result)
  except (AttributeError, OSError, TypeError, ValueError):
    return None


@dataclasses.dataclass(frozen=True)
class VectorState:
  """Validated current-thread vector lengths, in bytes (None if unavailable)."""

  sve_bytes: int | None
  sme_bytes: int | None

  @property
  def env_token(self) -> str:
    """Canonical, process-independent token suitable for a build/cache key."""
    sve = str(self.sve_bytes) if self.sve_bytes is not None else "none"
    sme = str(self.sme_bytes) if self.sme_bytes is not None else "none"
    return f"arm-vector-state:v1:sve={sve}:sme={sme}"

  @property
  def compiler_flags(self) -> tuple[str, ...]:
    """Best-effort Clang fixed-width flags; empty if no relevant VL is known."""
    flags = []
    if self.sve_bytes is not None:
      flags.append(f"-msve-vector-bits={self.sve_bytes * 8}")
    if self.sme_bytes is not None:
      flags.append(f"-msve-streaming-vector-bits={self.sme_bytes * 8}")
    return tuple(flags)

  @property
  def cmake_variables(self) -> dict[str, str]:
    """Values for the project's fixed-width CMake configuration knobs."""
    values = {}
    if self.sve_bytes is not None:
      values["VECOPS_FIXED_SVE_BITS"] = str(self.sve_bytes * 8)
    if self.sme_bytes is not None:
      values["VECOPS_FIXED_STREAMING_SVE_BITS"] = str(self.sme_bytes * 8)
    return values


def get_vector_state(
  *,
  machine: str | None = None,
  system: str | None = None,
  prctl: Callable[[int], int | None] | None = None,
) -> VectorState:
  """Query SVE and SME VL for this thread; unsupported systems yield ``None``.

  ``prctl`` is injectable for tests and should return the raw Linux prctl
  result.  Query errors and malformed lengths are represented as unavailable
  rather than raising, which makes this safe to use while forming cache keys.
  """
  machine = (machine or platform.machine()).lower()
  system = system or sys.platform
  if machine not in {"aarch64", "arm64"} or not system.startswith("linux"):
    return VectorState(None, None)
  query = prctl or _system_prctl
  try:
    sve = _validated_bytes(query(PR_SVE_GET_VL))
  except (OSError, RuntimeError, TypeError, ValueError):
    sve = None
  try:
    sme = _validated_bytes(query(PR_SME_GET_VL))
  except (OSError, RuntimeError, TypeError, ValueError):
    sme = None
  return VectorState(sve, sme)


__all__ = ["VectorState", "get_vector_state"]
