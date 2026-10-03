"""Visible DSL algorithms; no opaque C++ operator calls.

Load with the desired accumulation dtype first. Layernorm uses centered
variance by default; softmax subtracts the maximum. Large tiles stream; the
compiler may cache reused expensive expressions instead of recomputing them.
"""

from . import dsl as T


@T.func
def layernorm(x, *, eps=1e-5, variance="centered", unroll=2):
  x = T.as_value(x)
  if len(x.shape) == 2:
    result = T._normalize(x, kind="layernorm", eps=eps, variance=variance, unroll=unroll)
  else:
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
    result = centered * inv_std
  return result


@T.func
def softmax(x, *, unroll=2):
  x = T.as_value(x)
  if len(x.shape) == 2:
    result = T._normalize(x, kind="softmax", unroll=unroll)
  else:
    T.static_assert(len(x.shape) == 1)
    maximum = T.reduce_max(x, unroll=unroll)
    exponent = T.exp_neg(x - maximum)
    denominator = T.reduce_add(exponent, unroll=unroll)
    inverse = T.rcp(denominator)
    result = exponent * inverse
  return result


@T.func
def transpose_copy(source, destination, *, compute_dtype=None, implementation="auto"):
  """Transpose two memory axes with fused boundary conversion.

  CPU lowering specializes shape/dtype/layout into register butterflies or
  SME ZA moves; unusual strides retain gather/scatter. This helper is rank 2;
  automatic store planning also handles leading batch axes. `transpose`
  itself remains a zero-copy view operation.
  `implementation='vector'/'sme'/'gather'` is an explicit backend override.
  """
  T.static_assert(len(source.shape) == 2)
  T.static_assert(len(destination.shape) == 2)
  if (
    implementation != "sme"
    and (source.shape[0] == 1 or source.shape[1] == 1)
    and source.strides[1] == 1
    and source.strides[0] == source.shape[1]
    and destination.strides[1] == 1
    and destination.strides[0] == destination.shape[1]
  ):
    # A dense single-row/column transpose has identical linear order. Avoid
    # both a register permutation and one-vector-per-output-row traversal.
    if compute_dtype is None:
      T.copy(
        T.reshape(source, (source.shape[0] * source.shape[1],)),
        T.reshape(destination, (destination.shape[0] * destination.shape[1],)),
      )
    else:
      T.store(
        T.reshape(destination, (destination.shape[0] * destination.shape[1],)),
        T.load(T.reshape(source, (source.shape[0] * source.shape[1],)), dtype=compute_dtype),
      )
  else:
    if compute_dtype is None:
      T.copy(T.transpose(source), destination, implementation=implementation)
    else:
      T.store(destination, T.load(T.transpose(source), dtype=compute_dtype), implementation=implementation)


@T.func
def matmul(
  inputs,
  output,
  acc=None,
  *,
  acc_dtype=None,
  BM=None,
  BN=None,
  BK=None,
  packing="auto",
  swap=True,
  implementation="auto",
):
  """Cache-blocked matrix multiply; B has the conventional [K,N] shape.

  `dot` is the unblocked leaf. This wrapper preserves the input/accumulator
  DAG, so packing folds input transforms and the final store folds epilogues.
  None block sizes use the target cache hierarchy, not the build host's CPU.
  An owned output tile can be forwarded without a memory intermediate.
  """
  if acc_dtype is None:
    result = T._matmul(
      inputs, output, acc=acc, BM=BM, BN=BN, BK=BK, packing=packing, swap=swap, implementation=implementation
    )
  else:
    result = T._matmul(
      inputs,
      output,
      acc=acc,
      acc_dtype=acc_dtype,
      BM=BM,
      BN=BN,
      BK=BK,
      packing=packing,
      swap=swap,
      implementation=implementation,
    )
  return result
