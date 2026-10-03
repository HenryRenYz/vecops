"""Conservative SPR dependency estimate; frequency is not baked into cycles."""

from .....cost.model import CostProfile, CostRule, OperationCost

SAPPHIRE_RAPIDS = CostProfile(
  "sapphire-rapids",
  (CostRule(OperationCost(1, 4, provenance="estimated SPR FMA dependency latency"), (("operation", "fma"),)),),
)
