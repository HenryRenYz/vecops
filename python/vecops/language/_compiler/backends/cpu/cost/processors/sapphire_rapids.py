# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Conservative SPR dependency estimate; frequency is not baked into cycles."""

from .....cost.model import CostProfile, CostRule, OperationCost

SAPPHIRE_RAPIDS = CostProfile(
  "sapphire-rapids",
  (
    CostRule(OperationCost(1, 4, provenance="estimated SPR FMA dependency latency"), (("operation", "fma"),)),
    CostRule(
      OperationCost(1 / 32, 1, provenance="estimated AMX bf16 MAC throughput per 16-lane word"),
      (("operation", "dot"), ("implementation", "grid-word-mac"), ("input_dtype", "bfloat16")),
    ),
    *(
      CostRule(
        OperationCost(1 / 64, 1, provenance="estimated AMX int8 MAC throughput per 16-lane word"),
        (("operation", "dot"), ("implementation", "grid-word-mac"), ("input_dtype", dtype)),
      )
      for dtype in ("int8", "uint8")
    ),
  ),
)
