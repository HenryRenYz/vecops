"""Portable online SDPA expressed as visible dot/reduction/control operations.

This implementation is a composition baseline, not a claim of parity with the
CPU's prepared Attention scheduler. It never allocates a full M-by-K score
matrix; users select query tiles/grid scheduling around the helper.
"""

from .. import dsl as T

NEGATIVE_INFINITY = float("-inf")


@T.func
def _floating(dtype):
  return dtype == "float16" or dtype == "bfloat16" or dtype == "float32" or dtype == "float64"


@T.func
def scaled_dot_product_attention(
  query,
  key,
  value,
  output,
  *,
  scale=None,
  bias=None,
  query_mask=None,
  key_mask=None,
  causal=False,
  query_offset=0,
  key_offset=0,
  block_k=64,
  acc_dtype="float32",
  probability_dtype=None,
  accuracy="fast",
  implementation="auto",
):
  """One rank-two query tile against streamed K/V, with online normalization.

  Inputs are memory views, K is [keys,head_dim], V is [keys,value_dim].
  Optional operands are specialized in the outer factory, not nullable runtime
  tensors. All-masked/zero-key rows return zero. Low-dtype probability rounding
  defaults to V.dtype to permit AMX/SME; request float32 explicitly to avoid it.
  Output/accumulator values remain available for a fused epilogue.
  """
  T.static_assert(len(query.shape) == 2)
  T.static_assert(len(key.shape) == 2)
  T.static_assert(len(value.shape) == 2)
  T.static_assert(len(output.shape) == 2)
  T.static_assert(block_k > 0)
  T.static_assert(_floating(query.dtype))
  T.static_assert(_floating(key.dtype))
  T.static_assert(_floating(value.dtype))
  T.static_assert(_floating(output.dtype))
  acc_dtype = T.cast(0, acc_dtype).dtype
  T.static_assert(acc_dtype == "float32" or acc_dtype == "float64")
  assert query.shape[1] == key.shape[1]
  assert key.shape[0] == value.shape[0]
  assert output.shape[0] == query.shape[0] and output.shape[1] == value.shape[1]
  if bias is not None:
    T.static_assert(len(bias.shape) == 2)
    assert bias.shape[0] == query.shape[0] and bias.shape[1] == key.shape[0]
  if query_mask is not None:
    T.static_assert(len(query_mask.shape) == 1)
    assert query_mask.shape[0] == query.shape[0]
  if key_mask is not None:
    T.static_assert(len(key_mask.shape) == 1)
    assert key_mask.shape[0] == key.shape[0]
  if scale is None:
    assert query.shape[1] > 0
    scale = T.rsqrt(T.cast(query.shape[1], acc_dtype))
  if probability_dtype is None:
    probability_dtype = value.dtype
  probability_dtype = T.cast(0, probability_dtype).dtype
  T.static_assert(_floating(probability_dtype))
  negative_infinity = T.cast(NEGATIVE_INFINITY, acc_dtype)
  maximum = T.full((query.shape[0],), negative_infinity, dtype=acc_dtype)
  denominator = T.zeros((query.shape[0],), dtype=acc_dtype)
  accumulator = T.zeros((query.shape[0], value.shape[1]), dtype=acc_dtype)
  for begin in T.serial(0, key.shape[0], block_k, unroll=1):
    end = T.min(begin + block_k, key.shape[0])
    score_storage = T.alloc((query.shape[0], end - begin), dtype=acc_dtype)
    scores = T.dot(query, T.transpose(key[begin:end, :]), acc_dtype=acc_dtype, implementation=implementation)
    scores = scores * T.cast(scale, acc_dtype)
    if bias is not None:
      scores = scores + T.load(bias[:, begin:end], dtype=acc_dtype)
    # A QK product feeds both max and probability consumers. Keep one product
    # per streamed block, with scale/bias fused into its epilogue, rather than
    # recomputing GEMM through each scalar/reduction/state sink.
    T.store(score_storage, scores)
    scores = T.load(score_storage)
    if key_mask is None:
      valid = scores == scores
    else:
      valid = T.broadcast_to(T.expand_dims(T.load(key_mask[begin:end]) != 0, 0), scores.shape)
    if query_mask is not None:
      valid = valid & (T.expand_dims(T.load(query_mask) != 0, 1))
    if causal:
      valid = valid & (
        T.indices(scores.shape, axis=0) + query_offset >= T.indices(scores.shape, axis=1) + begin + key_offset
      )
    scores = T.where(valid, scores, negative_infinity)
    block_maximum = T.reduce_max(scores, axis=1)
    next_maximum = T.max(maximum, block_maximum)
    safe_maximum = T.where(next_maximum == negative_infinity, T.cast(0, acc_dtype), next_maximum)
    correction = T.where(
      maximum == negative_infinity, T.cast(0, acc_dtype), T.exp_neg(maximum - safe_maximum, accuracy=accuracy)
    )
    probability = T.where(
      valid, T.exp_neg(scores - T.expand_dims(safe_maximum, 1), accuracy=accuracy), T.cast(0, acc_dtype)
    )
    denominator = denominator * correction + T.reduce_add(probability, axis=1)
    accumulator = T.dot(
      T.cast(probability, probability_dtype),
      value[begin:end, :],
      acc=accumulator * T.expand_dims(correction, 1),
      acc_dtype=acc_dtype,
      implementation=implementation,
    )
    maximum = next_maximum
  safe_denominator = T.where(denominator > 0, denominator, T.cast(1, acc_dtype))
  result = accumulator / T.expand_dims(safe_denominator, 1)
  T.store(output, result)
  return result
