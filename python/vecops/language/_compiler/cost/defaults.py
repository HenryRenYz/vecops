# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
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
    CostRule(OperationCost(4, 8, resource="memory"), (("family", "indexed_memory"),)),
    CostRule(OperationCost(1, 3), (("family", "register"),)),
    CostRule(OperationCost(2, 3), (("family", "mask_query"),)),
    CostRule(OperationCost(12, 12, resource="memory"), (("family", "atomic"),)),
    CostRule(OperationCost(16, 16), (("family", "math"),)),
    CostRule(
      OperationCost(64, 64, provenance="estimated scalar math fallback"),
      (("family", "math"), ("implementation", "scalar")),
    ),
    *(operation_rule(name, 4) for name in ("div", "sqrt")),
    operation_rule("abs", 1),
    CostRule(OperationCost(1, 1), (("operation", "abs"), ("implementation", "scalar")), priority=1),
    operation_rule("identity", 0),
    operation_rule("bitreinterpret", 0),
    CostRule(
      OperationCost(1, 1, provenance="grid estimate: cycles per native vector-word MAC"),
      (("operation", "dot"), ("implementation", "grid-word-mac")),
    ),
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
    ("gemv_k_unroll", 2),
    ("gemv_k_unroll_long", 2),
    ("gemv_k_unroll_long_min", 512),
    ("matmul_unblocked_bytes", 512 * 1024),
    ("matmul_runtime_unblocked_bytes", 128 * 1024),
    ("matmul_pipeline", 0),  # 0=off, 1=prefetch, 2=operand lookahead
    ("matmul_prefetch_distance", 2),
    ("matmul_prefetch_lines", 2),
    ("amx_m_major_n_max", 0),
    ("amx_m_major_ratio", 4),
    ("amx_native_b_pack_min_m", 0),
    ("amx_cache_b_panel_divisor", 0),
    ("amx_cache_b_panel_max_m", 0),
    ("vector_register_budget", 24),
    ("grid_worker_cycles", 8000),
    ("grid_single_cycles", 64),
  ),
)
