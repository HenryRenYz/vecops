"""CPU memory estimates; compute work inherits portable defaults."""

from ....cost.model import CostProfile, CostRule, OperationCost

GENERIC = CostProfile(
  "cpu",
  (
    CostRule(
      OperationCost(4, 8, resource="memory", provenance="estimated strided CPU access"),
      (("family", "memory"), ("access", "strided")),
    ),
  ),
)
