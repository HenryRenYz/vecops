# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Uncalibrated x86 estimates; not a claim of measured instruction timings."""

from ....cost.model import CostProfile, CostRule, OperationCost

X86 = CostProfile(
  "x86",
  (
    CostRule(OperationCost(4, 12, provenance="estimated x86 division"), (("operation", "div"),)),
    CostRule(OperationCost(4, 12, provenance="estimated x86 square root"), (("operation", "sqrt"),)),
  ),
  # Four independent output columns already hide the dot dependency. Extra
  # K banks add horizontal sums/register pressure on small raw GEMVs.
  policies=(
    ("gemv_k_unroll", 1),
    ("amx_m_major_n_max", 128),
    ("amx_m_major_ratio", 4),
    ("amx_native_b_pack_min_m", 128),
    # B's original panel and its packed copy overlap in the cache during
    # production. Reserving 2/3 of the leaf budget for packed B alone crowds
    # out A/producer lines on long-K panels (e.g. Sapphire Rapids prefill).
    # Runtime L2 bytes, not a host-specific hard-coded column count.
    ("amx_cache_b_panel_divisor", 4),
    # Deeper M reuse amortizes a larger B panel. The bounded small-M policy
    # must not constrain square/tall products (measured on large K cases).
    ("amx_cache_b_panel_max_m", 256),
  ),
)
AVX512 = CostProfile("avx512")
