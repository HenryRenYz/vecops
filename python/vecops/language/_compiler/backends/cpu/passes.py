"""CPU-specific scheduling policies, separate from generic loop semantics."""
from ...passes.manager import PassResult, register_pass
from dataclasses import replace
from ... import ir as I


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
  size = I.constant(region.shape[-1])
  expensive = any(op.expression.op in {"exp", "exp2", "exp_neg", "exp2_neg", "log", "log2"}
                  and not op.opaque for op in region.operations)
  # Two logical copies of a single accumulator otherwise expose only two
  # native dependency chains. A multiword base supplies four without changing
  # the requested unroll. Multi-reduction regions already have more chains.
  if not expensive:
    power = reduction_power if len(region.initializations) == 1 else 0
  elif not context.options.get("cpu_wordwise_math", True):
    power = 0
  if (power and not explicit and region.iteration_axis is None and region.unroll == 2
      and (size is None or size >= 256)):
    region.base_power = power
  return region


@register_pass("cpu-legalize-math", stage="vector", after="legalize-vector-types")
def legalize_math(region, context):
  if context.options.get("backend") != "cpu":
    return PassResult(region, changed=False)
  enabled = context.options.get("cpu_wordwise_math", True)
  if not isinstance(enabled, bool):
    raise ValueError("cpu_wordwise_math must be bool")
  operations = {"exp", "exp2", "exp_neg", "exp2_neg", "log", "log2", "rcp", "rsqrt"}
  region.operations = [replace(op, native_math_words=enabled and not op.opaque
                               and op.expression.op in operations) for op in region.operations]
  return region
