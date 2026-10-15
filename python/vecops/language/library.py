# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Visible DSL algorithms; no opaque C++ operator calls.

Load with the desired accumulation dtype first. Layernorm uses centered
variance by default; softmax subtracts the maximum. Large tiles stream; the
compiler may cache reused expensive expressions instead of recomputing them.
"""

from . import dsl as T
from ._library.attention import scaled_dot_product_attention
from ._library.batched_matmul import batched_matmul
from ._library.math import layernorm_row, masked_softmax, rmsnorm, silu, softmax_row, stable_sigmoid
from ._library.packing import pack_b_compensated
from ._library.registers import pack_interleave, transpose_registers
from ._library.runtime_quant_int8 import dequantize as dequantize_per_row_column
from ._library.runtime_quant_int8 import quantize as quantize_per_row_u8

__all__ = [
  "batched_matmul",
  "dequantize_per_row_column",
  "layernorm",
  "masked_softmax",
  "matmul",
  "pack_interleave",
  "pack_b_compensated",
  "quantize_per_row_u8",
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
  workspace=None,
):
  """Cache-blocked matrix multiply; B has the conventional [K,N] shape.

  `dot` is the unblocked leaf. This wrapper preserves the input/accumulator
  DAG, so packing folds input transforms and the final store folds epilogues.
  None block sizes use the target cache hierarchy, not the build host's CPU.
  An owned output tile can be forwarded without a memory intermediate.
  Optional workspace is caller-declared T.workspace result storage with the
  exact output shape and accumulation dtype. Disjointness from producer reads,
  initializer and output is checked before writing. This reuses result capacity;
  internal instruction packing and cache scratch retain their own lifetimes.
  """
  if workspace is not None:
    a, b = inputs
    if acc_dtype is None:
      if (a.dtype == "int8" or a.dtype == "uint8") and (b.dtype == "int8" or b.dtype == "uint8"):
        acc_dtype = "int32"
      elif (a.dtype == "int16" or a.dtype == "uint16") and (b.dtype == "int16" or b.dtype == "uint16"):
        acc_dtype = "int64"
      elif a.dtype == "float64" or b.dtype == "float64":
        acc_dtype = "float64"
      else:
        acc_dtype = "float32"
    acc_dtype = T.cast(0, acc_dtype).dtype
    T.static_assert(len(workspace.shape) == 2 and workspace.dtype == acc_dtype)
    assert workspace.shape[0] == output.shape[0] and workspace.shape[1] == output.shape[1], "matmul workspace shape mismatch"
    assert T.view_disjoint(workspace, a) and T.view_disjoint(workspace, b), "matmul workspace overlaps input producer"
    assert T.view_disjoint(workspace, output), "matmul workspace overlaps output"
    if acc is not None:
      assert T.view_disjoint(workspace, T.as_value(acc)), "matmul workspace overlaps initializer"
    target = workspace
  else:
    target = output
  if acc_dtype is None:
    result = T._matmul(
      inputs, target, acc=acc, BM=BM, BN=BN, BK=BK, packing=packing, swap=swap, implementation=implementation
    )
  else:
    result = T._matmul(
      inputs,
      target,
      acc=acc,
      acc_dtype=acc_dtype,
      BM=BM,
      BN=BN,
      BK=BK,
      packing=packing,
      swap=swap,
      implementation=implementation,
    )
  if workspace is not None:
    T.store(output, result)
  return result
