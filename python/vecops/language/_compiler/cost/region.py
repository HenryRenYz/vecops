"""DAG cost and dependency analysis; never executes passes or source emission."""

from dataclasses import dataclass

from ..operations import Trait


@dataclass(frozen=True)
class RegionCost:
  compute_cycles: float = 0
  memory_cycles: float = 0
  critical_path_cycles: float = 0
  operations: int = 0
  temporary_registers: int = 0
  unknown: tuple[str, ...] = ()
  live_vectors: int = 0

  @property
  def estimated_cycles(self):
    return max(self.compute_cycles, self.memory_cycles, self.critical_path_cycles)


def estimate(roots, model, *, boundaries=(), implementation="vector", words=1):
  """Count shared producers once; stop at materialized/opaque/reduction values.

  This estimates one execution of a region, not its whole tile cardinality.
  Profile throughput values are per native word; ``words`` scales physical
  pieces, while the selector can describe their grouping/masking overhead.
  Memory traffic stays separate from transform cost. Scalar conditional arms
  use a worst-case execution path; vector selects evaluate both arms.
  """
  roots = (roots,) if hasattr(roots, "op") else tuple(roots)
  if isinstance(words, bool) or not isinstance(words, int) or words < 1:
    raise ValueError("physical word count must be a positive integer")
  boundaries = set(boundaries)
  seen, unknown, nodes = {}, set(), {}
  compute = memory = 0.0
  registers = 0

  def visit(expr):
    nonlocal compute, memory, registers
    if id(expr) in seen:
      return seen[id(expr)]
    definition = model.catalog.get(expr)
    nodes[id(expr)] = expr
    if id(expr) in boundaries or definition.traits & (Trait.REDUCTION | Trait.VECTOR_BOUNDARY):
      seen[id(expr)] = 0
      return 0
    if definition.family == "unknown":
      unknown.add(expr.op)
    # Shapes/address values are not re-evaluated for every vector lane.
    arguments = expr.args
    scalar = not hasattr(expr.type, "shape")
    if scalar and definition.traits & Trait.CONDITIONAL:
      if expr.op == "where":
        condition = expr.args[0]
        if condition.op == "literal":
          arguments = (condition, expr.args[1 if condition.data else 2])
        else:
          # Analyze both alternatives independently and select the higher
          # compute/memory path, without charging shared producers twice.
          visit(condition)
          branch_start = (compute, memory, registers, dict(seen))
          choices = []
          for arm in expr.args[1:]:
            compute, memory, registers, previous = branch_start
            seen.clear()
            seen.update(previous)
            latency = visit(arm)
            choices.append((compute, memory, registers, dict(seen), latency))
          compute, memory, registers, selected, latency = max(choices, key=lambda x: max(x[0], x[1]))
          seen.clear()
          seen.update(selected)
          own = model.lookup(model.query(expr, implementation="scalar"))
          compute += own.throughput_cycles
          seen[id(expr)] = latency + own.latency_cycles
          return seen[id(expr)]
      elif expr.args[0].op == "literal" and (
        expr.op == "and" and not expr.args[0].data or expr.op == "or" and expr.args[0].data
      ):
        arguments = expr.args[:1]
    latency = max((visit(arg) for arg in arguments), default=0)
    query = model.query(expr, implementation="scalar" if scalar else implementation, words=words)
    cost = model.lookup(query)
    scale = 1 if scalar else words
    if cost.resource == "memory":
      memory += cost.throughput_cycles * scale
      if query.input_dtype != query.dtype:
        # Fused load_convert pays conversion once, not a second load.
        conversion = model.lookup(
          type(query)(
            "cast", "conversion", query.dtype, query.input_dtype, implementation=query.implementation, words=words
          )
        )
        compute += conversion.throughput_cycles * scale
    else:
      compute += cost.throughput_cycles * scale
    registers = max(registers, cost.temporary_registers)
    seen[id(expr)] = latency + cost.latency_cycles
    return seen[id(expr)]

  path = max((visit(root) for root in roots), default=0)
  # Conservative postorder live-value hint, not a physical register allocator.
  # Boundary operands are external values, not recursively produced here.
  uses = {identity: 0 for identity in seen}
  for identity in seen:
    expression = nodes[identity]
    if identity not in boundaries and not model.catalog.has(expression, Trait.REDUCTION | Trait.VECTOR_BOUNDARY):
      for argument in expression.args:
        if id(argument) in uses:
          uses[id(argument)] += 1
  for root in roots:
    uses[id(root)] += 1
  live, peak = set(), 0
  for identity in seen:
    expression = nodes[identity]
    if hasattr(expression.type, "shape"):
      live.add(identity)
      peak = max(peak, len(live))
    for argument in expression.args:
      if id(argument) in uses:
        uses[id(argument)] -= 1
        if not uses[id(argument)]:
          live.discard(id(argument))
  return RegionCost(compute, memory, path, len(seen), registers, tuple(sorted(unknown)), peak * words)
