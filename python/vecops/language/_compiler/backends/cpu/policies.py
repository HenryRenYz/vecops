"""Cost-based CPU decisions; no duplicated opcode lists or hidden thresholds."""


def recompute_limit(cost, model, *, packed=False):
  """Extra computation must pay for saved temporary traffic and setup.

  Budgets are heuristic vector-tile traffic equivalents. The lower bound
  protects against allocation/layout overhead omitted by the coarse estimate.
  Unlike an expensive-op list, this varies continuously with total DAG work.
  """
  kind = "packed" if packed else "raw"
  budget = model.policy(kind + "_fusion_traffic_budget", 32 if packed else 256)
  floor = model.policy(kind + "_fusion_min_reloads", 2 if packed else 32)
  margin = model.policy("fusion_safety_margin", 1.0)
  return max(floor, 1 + budget * margin / max(cost, 1e-12))
