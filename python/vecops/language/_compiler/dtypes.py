# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Shared dtype normalization; no parser or target-code dependencies."""

from ..._dtype import ELEMENT_BYTES, scalar_dtype
from ..._dtype import dtype_name as canonical_dtype_name

__all__ = ["ELEMENT_BYTES", "dtype_name"]


def dtype_name(value):
  return canonical_dtype_name(scalar_dtype(value))
