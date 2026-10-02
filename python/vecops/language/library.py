"""Visible DSL algorithms; no opaque C++ operator calls.

Load with the desired accumulation dtype first. Layernorm uses centered
variance by default; softmax subtracts the maximum. Large tiles stream; the
compiler may cache reused expensive expressions instead of recomputing them.
"""
from . import dsl as T


@T.func
def layernorm(x, *, eps=1e-5, variance="centered", unroll=2):
  T.static_assert(len(x.shape) == 1)
  count = x.shape[0]
  if variance == "moments":
    total = T.reduce_add(x, unroll=unroll)
    total_squared = T.reduce_add(x * x, unroll=unroll)
    mean = total / count
    centered = x - mean
    var = T.max(total_squared / count - mean * mean, T.cast(0, x.dtype))
  else:
    T.static_assert(variance == "centered")
    mean = T.reduce_add(x, unroll=unroll) / count
    centered = x - mean
    var = T.reduce_add(centered * centered, unroll=unroll) / count
  inv_std = T.rsqrt(var + T.cast(eps, x.dtype))
  return centered * inv_std


@T.func
def softmax(x, *, unroll=2):
  T.static_assert(len(x.shape) == 1)
  maximum = T.reduce_max(x, unroll=unroll)
  exponent = T.exp_neg(x - maximum)
  denominator = T.reduce_add(exponent, unroll=unroll)
  inverse = T.rcp(denominator)
  return exponent * inverse
