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
  ),
)
AVX512 = CostProfile("avx512")
