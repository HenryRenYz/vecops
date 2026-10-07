"""Stable, composable AF3-style math; no opaque native operator calls."""

from .. import dsl as T

NEGATIVE_INFINITY = float("-inf")


@T.func
def softmax_inverse(denominator, accuracy="strict"):
  """Softmax's scalar normalization factor in the accumulation dtype.

  Strict uses ordinary IEEE division of a dtype-typed one. Fast/Estimate
  retain the native reciprocal tier; explicit ``T.rcp`` is unaffected.
  """
  denominator = T.as_value(denominator)
  if accuracy == "strict":
    result = T.cast(1, denominator.dtype) / denominator
  else:
    result = T.rcp(denominator, accuracy=accuracy)
  return result


@T.func
def softmax_exp_neg(x, accuracy="strict"):
  """Softmax's FP32 Estimate exponential without the slow FEXPA tail.

  Match ops/details/softmax/Exp.h: approximate exp(x / 128) on
  [-0.625, 0] with a degree-nine polynomial, square seven times, and
  flush centered scores below -80 to zero. NaNs survive clamping.
  Strict/Fast and non-FP32 values retain the native exp_neg operation.
  """
  x = T.as_value(x)
  if x.dtype == "float32" and accuracy == "estimate":
    zero = T.cast(0, x.dtype)
    one = T.cast(1, x.dtype)
    limit = T.cast(-80, x.dtype)
    clamped = T.where(x != x, x, T.max(x, limit))  # noqa: PLR0124 -- preserve NaNs across backend min/max policies
    reduced = clamped * T.cast(1.0 / 128.0, x.dtype)
    result = T.cast(1.0 / 362880.0, x.dtype)
    result = T.fma(result, reduced, T.cast(1.0 / 40320.0, x.dtype))
    result = T.fma(result, reduced, T.cast(1.0 / 5040.0, x.dtype))
    result = T.fma(result, reduced, T.cast(1.0 / 720.0, x.dtype))
    result = T.fma(result, reduced, T.cast(1.0 / 120.0, x.dtype))
    result = T.fma(result, reduced, T.cast(1.0 / 24.0, x.dtype))
    result = T.fma(result, reduced, T.cast(1.0 / 6.0, x.dtype))
    result = T.fma(result, reduced, T.cast(0.5, x.dtype))
    result = T.fma(result, reduced, one)
    result = T.fma(result, reduced, one)
    for i in T.unroll(7):
      result = result * result
    result = T.where(x < limit, zero, result)
  else:
    result = T.exp_neg(x, accuracy=accuracy)
  return result


@T.func
def stable_sigmoid(x, *, accuracy="fast", algorithm="exp"):
  """Clamped sigmoid with the FP32 evaluation schedule of AF3 StableMath.

  Choose polynomial explicitly for SME targets where FEXPA tails are costly.
  Small-float inputs compute in float32; a surrounding cast controls storage.
  Both algorithms clamp magnitude to 80, including infinite inputs.
  """
  x = T.as_value(x)
  if x.dtype == "float16" or x.dtype == "bfloat16":
    x = T.cast(x, "float32")
  magnitude = T.min(T.abs(x), T.cast(80, x.dtype))
  one = T.cast(1, x.dtype)
  if algorithm == "polynomial":
    reduced = -magnitude * T.cast(1.0 / 128.0, x.dtype)
    exponent = T.cast(1.0 / 362880.0, x.dtype)
    exponent = T.fma(exponent, reduced, T.cast(1.0 / 40320.0, x.dtype))
    exponent = T.fma(exponent, reduced, T.cast(1.0 / 5040.0, x.dtype))
    exponent = T.fma(exponent, reduced, T.cast(1.0 / 720.0, x.dtype))
    exponent = T.fma(exponent, reduced, T.cast(1.0 / 120.0, x.dtype))
    exponent = T.fma(exponent, reduced, T.cast(1.0 / 24.0, x.dtype))
    exponent = T.fma(exponent, reduced, T.cast(1.0 / 6.0, x.dtype))
    exponent = T.fma(exponent, reduced, T.cast(0.5, x.dtype))
    exponent = T.fma(exponent, reduced, one)
    exponent = T.fma(exponent, reduced, one)
    for i in T.unroll(7):
      exponent = exponent * exponent
  else:
    T.static_assert(algorithm == "exp")
    exponent = T.exp_neg(-magnitude * T.cast(0.125, x.dtype), accuracy=accuracy)
    for i in T.unroll(3):
      exponent = exponent * exponent
  denominator = one + exponent
  return T.where(x >= 0, one / denominator, exponent / denominator)


@T.func
def silu(x, *, accuracy="fast", algorithm="exp"):
  x = T.as_value(x)
  return x * stable_sigmoid(x, accuracy=accuracy, algorithm=algorithm)


@T.func
def rmsnorm(x, *, eps=1e-5, axis=-1, unroll=2):
  x = T.as_value(x)
  if x.dtype == "float16" or x.dtype == "bfloat16":
    x = T.cast(x, "float32")
  if len(x.shape) == 2 and (axis == -1 or axis == 1):
    return T._normalize(x, kind="rmsnorm", eps=eps, unroll=unroll)
  count = x.shape[axis]
  variance = T.reduce_add(x * x, axis=axis, unroll=unroll) / count
  inverse = T.rsqrt(variance + T.cast(eps, x.dtype))
  if len(x.shape) > 1:
    inverse = T.expand_dims(inverse, axis)
  return x * inverse


@T.func
def masked_softmax(x, mask=None, *, axis=-1, accuracy="fast", unroll=2):
  """Finite-logit softmax; masked/empty rows produce zero, never 0/0.

  Numeric mask values use nonzero truth semantics. Mask/bias remain explicit
  expression operands, so callers can fuse them before a dot or store.
  """
  x = T.as_value(x)
  if x.dtype == "float16" or x.dtype == "bfloat16":
    x = T.cast(x, "float32")
  if mask is None:
    valid = True
  else:
    # T.where already normalizes numeric truth values and preserves native
    # predicate tiles; comparing a predicate Vec to integer zero is invalid.
    valid = T.as_value(mask)
  negative_infinity = T.cast(NEGATIVE_INFINITY, x.dtype)
  scores = T.where(valid, x, negative_infinity)
  if len(x.shape) == 2 and (axis == -1 or axis == 1):
    return T._normalize(scores, kind="masked_softmax", accuracy=accuracy, unroll=unroll)
  maximum = T.reduce_max(scores, axis=axis, unroll=unroll)
  safe_maximum = T.where(maximum == negative_infinity, T.cast(0, x.dtype), maximum)
  if len(x.shape) > 1:
    safe_maximum = T.expand_dims(safe_maximum, axis)
  probability = T.where(valid, softmax_exp_neg(scores - safe_maximum, accuracy=accuracy), T.cast(0, x.dtype))
  total = T.reduce_add(probability, axis=axis, unroll=unroll)
  safe_total = T.where(total > 0, total, T.cast(1, x.dtype))
  if len(x.shape) > 1:
    safe_total = T.expand_dims(safe_total, axis)
  return probability / safe_total
