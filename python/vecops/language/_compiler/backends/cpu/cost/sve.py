"""Central SVE estimates; new layout variants never alter ordinary costs."""

from ....cost.model import CostProfile, CostRule, OperationCost

SVE = CostProfile(
  "sve",
  (
    *tuple(
      CostRule(
        OperationCost(cycles, cycles, provenance="estimated SVE reciprocal tier"),
        (("operation", op), ("accuracy", accuracy)),
      )
      for op in ("rcp", "rsqrt")
      for accuracy, cycles in (("estimate", 1), ("fast", 4))
    ),
    *(
      CostRule(
        OperationCost(
          throughput,
          latency,
          temporary_registers=temporary,
          confidence=0.5,
          provenance="heuristic SVE pair layout; stage34 assembly/whole-kernel A/B, not measured latency",
        ),
        (
          ("operation", operation),
          ("input_dtype", input_dtype),
          ("dtype", dtype),
          ("implementation", implementation),
          ("words", 2),
          ("masked", False),
          ("access", "contiguous"),
        ),
      )
      for implementation, operation, input_dtype, dtype, throughput, latency, temporary in (
        ("sve-ordered-pair", "load", "float16", "float32", 2, 6, 2),
        ("sve-packed-even-odd", "load", "float16", "float32", 1, 6, 1),
        ("sve-ordered-pair", "store", "float32", "float16", 1.5, 4, 2),
        ("sve-packed-even-odd", "store", "float32", "float16", 1.5, 8, 1),
      )
    ),
  ),
  version="2",
  policies=(
    ("pointwise_layout_min_paired_loads", 3),
    ("pointwise_layout_compute_budget", 16),
    ("pointwise_layout_register_budget", 20),
    ("pointwise_layout_min_cost_confidence", 0.5),
    ("pointwise_layout_min_saved_cycles", 2),
    ("pointwise_layout_safety_fraction", 0.1),
  ),
)
