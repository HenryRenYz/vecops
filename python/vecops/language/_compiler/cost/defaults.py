"""Portable estimates; intentionally low-confidence until target calibration."""

from .model import CostProfile, CostRule, OperationCost


def operation_rule(name, cycles, latency=None, **selectors):
  return CostRule(
    OperationCost(cycles, cycles if latency is None else latency),
    tuple(sorted({"operation": name, **selectors}.items())),
    priority=1,
  )


DEFAULT = CostProfile(
  "default",
  (
    *(
      CostRule(OperationCost(1, 1), (("family", family),))
      for family in ("arithmetic", "comparison", "logical", "select", "conversion", "index")
    ),
    *(CostRule(OperationCost(0, 0), (("family", family),)) for family in ("structural", "reference", "shape")),
    CostRule(OperationCost(1, 4, resource="memory"), (("family", "memory"),)),
    CostRule(OperationCost(16, 16), (("family", "math"),)),
    CostRule(
      OperationCost(64, 64, provenance="estimated scalar math fallback"),
      (("family", "math"), ("implementation", "scalar")),
    ),
    *(operation_rule(name, 4) for name in ("div", "sqrt")),
    operation_rule("abs", 1),
    CostRule(OperationCost(1, 1), (("operation", "abs"), ("implementation", "scalar")), priority=1),
    operation_rule("identity", 0),
  ),
  policies=(
    ("reuse_compute_budget", 6),
    ("math_chunk_compute_budget", 6),
    ("raw_fusion_traffic_budget", 256),
    ("raw_fusion_min_reloads", 32),
    ("packed_fusion_traffic_budget", 32),
    ("packed_fusion_min_reloads", 2),
    ("fusion_safety_margin", 1.0),
    ("gemv_epilogue_group_overhead", 8),
    ("vector_register_budget", 24),
  ),
)
