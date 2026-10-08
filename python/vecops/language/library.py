# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Visible DSL algorithms; no opaque C++ operator calls.

Load with the desired accumulation dtype first. Layernorm uses centered
variance by default; softmax subtracts the maximum. Large tiles stream; the
compiler may cache reused expensive expressions instead of recomputing them.
"""

from . import dsl as T
from ._library.attention import scaled_dot_product_attention
from ._library.math import layernorm_row, masked_softmax, rmsnorm, silu, softmax_row, stable_sigmoid
from ._library.registers import pack_interleave, transpose_registers

__all__ = [
  "layernorm",
  "masked_softmax",
  "matmul",
  "pack_interleave",
  "rmsnorm",
  "scaled_dot_product_attention",
  "silu",
  "softmax",
  "stable_sigmoid",
  "transpose_copy",
  "transpose_registers",
]


@T.func
def layernorm(x, *, eps=1e-5, variance="centered", unroll=2):
  x = T.as_value(x)
  if len(x.shape) == 2:
    if x.dtype != "float32" and x.dtype != "float64":
      x = T.cast(x, "float32")
    result = T.map_rows(layernorm_row, x, eps=eps, variance=variance, unroll=unroll)
  else:
    T.static_assert(len(x.shape) == 1)
    result = layernorm_row(x, eps=eps, variance=variance, unroll=unroll)
  return result


@T.func
def softmax(x, *, unroll=2, accuracy="strict"):
  """Normalize the final axis using the requested exponential accuracy.

  Strict normalization uses dtype-typed ``1 / sum`` (IEEE division).
  Fast/Estimate keep the corresponding native reciprocal accuracy tier.
  """
  x = T.as_value(x)
  if len(x.shape) == 2:
    if x.dtype != "float32" and x.dtype != "float64":
      x = T.cast(x, "float32")
    result = T.map_rows(softmax_row, x, unroll=unroll, accuracy=accuracy)
  else:
    T.static_assert(len(x.shape) == 1)
    result = softmax_row(x, unroll=unroll, accuracy=accuracy)
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
