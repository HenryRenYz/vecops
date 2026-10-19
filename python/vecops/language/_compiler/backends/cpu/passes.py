# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""CPU-specific scheduling policies, separate from generic loop semantics."""

from dataclasses import replace

from ... import ir as I
from ...analysis.facts import ValueFacts
from ...cost import estimate
from ...dtypes import ELEMENT_BYTES
from ...ir import vector as V
from ...passes.manager import PassResult, register_pass


@register_pass("cpu-plan-full-chunks", stage="vector", before="unroll-vector-operations")
def plan_full_chunks(region, context):
  # Full vector groups need neither per-operation empty checks nor partial
  # memory masks. Keep an independently masked tail for arbitrary row lengths.
  if context.options.get("backend") != "cpu":
    return PassResult(region, changed=False)
  region.full_chunks = context.options.get("cpu_full_chunks", True)
  if not isinstance(region.full_chunks, bool):
    raise ValueError("cpu_full_chunks must be bool")
  return region


@register_pass("cpu-choose-vector-chunk", stage="vector", before="unroll-vector-operations")
def choose_vector_chunk(region, context):
  if context.options.get("backend") != "cpu":
    return PassResult(region, changed=False)
  # A larger automatic tile chunk restores independent native math chains.
  # It changes neither explicit vector contracts nor the logical unroll count.
  # Keep it target-local and independently switchable for cost-model tuning.
  power = context.options.get("cpu_math_chunk_pow", 1)
  if isinstance(power, bool) or not isinstance(power, int) or not 0 <= power <= 2:
    raise ValueError("cpu_math_chunk_pow must be 0, 1, or 2")
  reduction_power = context.options.get("cpu_reduction_chunk_pow", 1)
  if isinstance(reduction_power, bool) or not isinstance(reduction_power, int) or not 0 <= reduction_power <= 2:
    raise ValueError("cpu_reduction_chunk_pow must be 0, 1, or 2")
  explicit = any(isinstance(op.expression.type, I.VectorType) for op in region.operations)
  # Grouped operations retain identity-bearing type metadata. A constant
  # extent may therefore still be written as ceildiv(K, group), rather than
  # a literal; use the shared integer analysis for the scheduling decision.
  extent = ValueFacts().infer(region.shape[-1])
  size = extent.constant if extent is not None else None
  environment = context.environment
  region.cost = estimate(
    region.roots, environment.costs, boundaries=(id(op.expression) for op in region.operations if op.opaque)
  )
  compute_bound = region.cost.compute_cycles >= environment.costs.policy("math_chunk_compute_budget", 6)
  # Two logical copies of a single accumulator otherwise expose only two
  # native dependency chains. A multiword base supplies four without changing
  # the requested unroll. Multi-reduction regions already have more chains.
  if not compute_bound:
    power = reduction_power if len(region.initializations) == 1 else 0
  elif not context.options.get("cpu_wordwise_math", True) and any(
    environment.implementations.get(op.expression.op).wordwise_math and not op.opaque for op in region.operations
  ):
    power = 0
  if power and not explicit and region.iteration_axis is None and region.unroll == 2 and (size is None or size >= 256):
    budget = environment.costs.policy("vector_register_budget", 24)
    while (
      power and (region.cost.live_vectors + region.cost.temporary_registers) * region.unroll * (1 << power) > budget
    ):
      power -= 1
    region.base_power = power
  return region


@register_pass("cpu-legalize-math", stage="vector", after="legalize-vector-types")
def legalize_math(region, context):
  if context.options.get("backend") != "cpu":
    return PassResult(region, changed=False)
  enabled = context.options.get("cpu_wordwise_math", True)
  if not isinstance(enabled, bool):
    raise ValueError("cpu_wordwise_math must be bool")
  region.operations = [
    replace(
      op,
      native_math_words=enabled
      and not op.opaque
      and context.environment.implementations.get(op.expression.op).wordwise_math,
    )
    for op in region.operations
  ]
  # Estimate the legal physical word group, not an arbitrarily large virtual
  # vector. Oversized virtual extents stream repeated legal pieces.
  physical_words = 1 << min(5, max(0, region.base_power))
  region.cost = estimate(
    region.roots,
    context.environment.costs,
    boundaries=(id(op.expression) for op in region.operations if op.opaque),
    words=physical_words,
  )
  return region


@register_pass("cpu-plan-native-word-tails", stage="vector", after="legalize-vector-types")
def plan_native_word_tails(region, context):
  """Keep full groups and split masked remainders into physical word fragments.

  SVE grouped predicates can require tuple work even for empty physical
  words. Splitting those words lets ordinary empty-fragment guards remove
  them. AVX masked groups already lower efficiently; fragmenting them can
  duplicate tail setup and increase native code size. Split AVX tails by
  default only when a constant remainder contains a complete physical word,
  exposing a full mask to native optimization. Large AVX-512 private-helper owners and
  hardware regions also split dynamic tails to limit fallback register pressure.
  Large register-layout owners with narrow floating arithmetic split as well;
  copying and converting alone retain the ordinary default.
  Keep other AVX tails grouped.
  An explicit option still overrides this target policy.

  A native compute vector can still widen to multiple memory words. Account
  for every dtype domain with the already-legalized rebind headroom, instead
  of special-casing normalization or individual operations. Explicit vector
  state/iteration contracts retain their own lowering.
  """
  if context.options.get("backend") != "cpu":
    return PassResult(region, changed=False)
  target = context.environment.codegen_target
  enabled = context.options.get("cpu_native_word_tails", True)
  if not isinstance(enabled, bool):
    raise ValueError("cpu_native_word_tails must be bool")  # noqa: TRY004 - Match existing CPU pass-option validation.
  region.native_tails = {}
  if (
    not enabled
    or not (
      target.architecture == "arm" and target.isa in {"sve", "sme"}
      or target.architecture == "x86" and target.isa in {"avx", "avx2", "avx512"}
    )
    or not region.full_chunks
    or region.single_chunk
    or region.iteration_axis is not None
    or any(isinstance(operation.expression.type, I.VectorType) for operation in region.operations)
  ):
    return region
  # Converting stores may deliberately omit their terminal cast from roots.
  # Their physical memory dtype still constrains a native masked fragment.
  memory_bytes = max(
    (ELEMENT_BYTES[effect.store.destination.dtype] for effect in region.effects if effect.store is not None),
    default=ELEMENT_BYTES[region.dtype],
  )
  store_headroom = max(0, memory_bytes.bit_length() - ELEMENT_BYTES[region.dtype].bit_length())
  # A grouped contraction can need several native source words even for
  # one result. Never shrink below one result lane at the minimum SVE VL.
  minimum_result_power = -(16 // ELEMENT_BYTES[region.dtype]).bit_length() + 1
  tail_power = max(-max(region.rebind_headroom, store_headroom), minimum_result_power)
  variants = tuple((power, region.base_power, groups) for power, groups in region.groupings.items())
  fallback_power = min(0, 2 - region.rebind_headroom)
  if region.base_power > fallback_power:
    variants += ((None, fallback_power, V.group_vectors(region.unroll, fallback_power, fallback_power)),)
  for variant, base_power, groups in variants:
    if (
      target.architecture == "x86"
      and "cpu_native_word_tails" not in context.options
      and not context.options.get("_cpu_compound_owner", False)
    ):
      extent = I.constant(region.shape[-1])
      if type(extent) is not int or not target.vector_bits:
        continue
      lanes = target.vector_bits // (8 * ELEMENT_BYTES[region.dtype])
      base_lanes = lanes << base_power if base_power >= 0 else lanes >> -base_power
      fragment_lanes = lanes << tail_power if tail_power >= 0 else lanes >> -tail_power
      # A partial first word provides no complete mask to fold. In that
      # case AVX grouping avoids duplicating per-fragment setup. Use the
      # actual ISA geometry; cost-model and SVE length overrides are unrelated.
      step = base_lanes * region.unroll
      if step <= 0 or fragment_lanes <= 0 or extent % step < fragment_lanes:
        continue
    fragments = []
    for parent, group in enumerate(groups):
      depth = group.power - tail_power
      if depth < 0 or group.repeat != 1:
        break
      for part in range(1 << depth):
        path = tuple(bool(part & (1 << bit)) for bit in reversed(range(depth)))
        first = (group.first << (base_power - tail_power)) + part
        fragments.append(V.VectorTailFragment(parent, first, tail_power, path))
    else:
      # Single native groups already have no tuple predicate to remove.
      if any(fragment.path for fragment in fragments):
        region.native_tails[variant] = tuple(fragments)
  return region
