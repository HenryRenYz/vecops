"""SVE math accuracy estimates; strict tiers retain the generic fallback."""

from ....cost.model import CostProfile, CostRule, OperationCost

SVE = CostProfile(
  "sve",
  tuple(
    CostRule(
      OperationCost(cycles, cycles, provenance="estimated SVE reciprocal tier"),
      (("operation", op), ("accuracy", accuracy)),
    )
    for op in ("rcp", "rsqrt")
    for accuracy, cycles in (("estimate", 1), ("fast", 4))
  ),
)
