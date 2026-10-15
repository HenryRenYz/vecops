# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Ordinary-SVE single-register, 128-bit-segment operations.

Inputs must be complete scalable Vector(dtype, pow=0) values. These pure
register operations have no streaming owner and must run outside SME regions.
iota(dtype) returns native-word lane indices in an unsigned integer dtype.
join_k_groups zips low-half dwords; broadcast_segment duplicates one 128-bit
segment; row_pair extracts alternating 64-bit pairs into the low half.
mmla accumulates BF16 into FP32 or same-sign bytes into I32 modulo 2**32.
"""

from .dsl import Intrinsic

iota = Intrinsic("sve.iota")
join_k_groups = Intrinsic("sve.join_k_groups")
broadcast_segment = Intrinsic("sve.broadcast_segment")
row_pair = Intrinsic("sve.row_pair")
mmla = Intrinsic("sve.mmla")
__all__ = ["broadcast_segment", "iota", "join_k_groups", "mmla", "row_pair"]
