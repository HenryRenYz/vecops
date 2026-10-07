"""Conservative profitability of a semantically proven private CPU layout.

The first automatic candidate is F16-only and requires several independent
full-width pair loads. Estimates come from central implementation cost rules,
not an operator name, library entry point or spelling-based expensive list.
The costs are heuristic relative cycles, not instruction latency measurements.
"""

from dataclasses import dataclass

from .... import ir as I
from ....cost import CostQuery, estimate


@dataclass(frozen=True)
class RegisterLayoutDecision:
  selected: bool
  reason: str
  independent_paired_loads: int = 0
  physical_words: int = 0
  live_registers: int = 0
  pointwise_compute_cycles: float = 0
  ordered_cycles: float = 0
  packed_cycles: float = 0
  cost_provenance: tuple[str, ...] = ()


def conversion_query(operation, memory_dtype, implementation):
  """Describe a complete two-F32-word pair, not a masked tail or scalar fill."""
  output = memory_dtype if operation == "store" else "float32"
  input_dtype = "float32" if operation == "store" else memory_dtype
  return CostQuery(
    operation,
    "memory",
    output,
    input_dtype,
    implementation=implementation,
    words=2,
    access="contiguous",
  )


def assess_register_layout(region, proof, model):
  """Return an auditable choice; this function never changes semantic IR.

  Throughput costs are per F32 native word, while dependency latencies are
  per conversion implementation. Scalars and shared producers are counted
  by the same central DAG estimator as other CPU decisions.
  """
  store = region.effects[0].store
  if store.destination.dtype != "float16" or any(load.data.dtype != "float16" for load in proof.loads):
    return RegisterLayoutDecision(False, "unvalidated physical conversion dtype")
  varying = tuple(load for load in proof.loads if I.constant(load.data.shape[-1]) != 1)
  # The proof has already deduplicated DAG identities. Canonical CSE identifies
  # identical accesses before this CPU pass; do not count repeated use sites or
  # scalar broadcasts as independent memory/convert producers.
  count = len(varying)
  minimum = max(3, model.policy("pointwise_layout_min_paired_loads", 3))
  if count < minimum:
    return RegisterLayoutDecision(False, "too few independent full-width pair loads", count)
  variants = (
    sum((1 << max(0, group.power)) * group.repeat for group in groups)
    for groups in region.groupings.values()
    if any(group.power >= 1 for group in groups)
  )
  words = max(variants, default=0)
  if not words:
    return RegisterLayoutDecision(False, "no complete native pair grouping", count)
  pointwise = estimate(region.roots, model, boundaries=tuple(map(id, proof.loads)))
  compute = pointwise.compute_cycles * 2
  if pointwise.unknown or compute > model.policy("pointwise_layout_compute_budget", 16):
    return RegisterLayoutDecision(
      False, "unknown or compute-heavy pointwise DAG", count, words, pointwise_compute_cycles=compute
    )
  costs = {
    (implementation, operation): model.lookup(conversion_query(operation, "float16", implementation))
    for implementation in ("sve-ordered-pair", "sve-packed-even-odd")
    for operation in ("load", "store")
  }
  provenance = tuple(dict.fromkeys(cost.provenance for cost in costs.values()))
  if any(cost.confidence < model.policy("pointwise_layout_min_cost_confidence", 0.5) for cost in costs.values()):
    return RegisterLayoutDecision(
      False,
      "missing trusted implementation cost rule",
      count,
      words,
      pointwise_compute_cycles=compute,
      cost_provenance=provenance,
    )
  temporary = max(cost.temporary_registers for cost in costs.values())
  # This is a conservative live-range hint, not a claim to be a register
  # allocator. Account for all simultaneously named full groups and a pair's
  # conversion temporary, rather than accepting arbitrary virtual POW2.
  live = (pointwise.live_vectors + pointwise.temporary_registers) * words + temporary
  if live > model.policy("pointwise_layout_register_budget", 20):
    return RegisterLayoutDecision(
      False, "conservative vector-register pressure", count, words, live, compute, cost_provenance=provenance
    )

  def score(implementation):
    load, output = (costs[implementation, operation] for operation in ("load", "store"))
    throughput = 2 * (count * load.throughput_cycles + output.throughput_cycles) + compute
    critical = load.latency_cycles + pointwise.critical_path_cycles + output.latency_cycles
    return max(throughput, critical)

  ordered, packed = score("sve-ordered-pair"), score("sve-packed-even-odd")
  margin = max(
    model.policy("pointwise_layout_min_saved_cycles", 2),
    ordered * model.policy("pointwise_layout_safety_fraction", 0.1),
  )
  selected = ordered - packed >= margin
  return RegisterLayoutDecision(
    selected,
    "estimated profitable" if selected else "insufficient estimated savings",
    count,
    words,
    live,
    compute,
    ordered,
    packed,
    provenance,
  )
