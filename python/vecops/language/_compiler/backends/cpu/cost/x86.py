"""Uncalibrated x86 estimates; not a claim of measured instruction timings."""

from ....cost.model import CostProfile, CostRule, OperationCost

X86 = CostProfile(
  "x86",
  (
    CostRule(OperationCost(4, 12, provenance="estimated x86 division"), (("operation", "div"),)),
    CostRule(OperationCost(4, 12, provenance="estimated x86 square root"), (("operation", "sqrt"),)),
  ),
)
AVX512 = CostProfile("avx512")
