# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Composable quantization and fused grouped-dot over public SME packed I8 B."""

from .. import dsl as T
from .. import sme
from .matrix_recipe import finish_matrix


@T.func
def quantize(value, multipliers, zero_point):
  """FP32 affine map, then truncate/saturate into U8; NaN maps to zero.

  Multipliers broadcast per logical row and zero_point is shared. Multiply
  and add remain separate FP32 operations; use a distinct explicit producer
  when FMA or intermediate rounding is required. This helper does not change
  the behavior of ordinary user casts or recognize arbitrary affine DAGs.
  """
  value = T.as_value(value)
  T.static_assert(value.dtype == "float32")
  affine = value * T.cast(T.as_value(multipliers), "float32") + T.cast(T.as_value(zero_point), "float32")
  bits = T.bitcast(affine, "uint32")
  finite_or_infinite = T.where((bits & 2147483647) > 2139095040, 0.0, affine)
  return T.cast(T.max(0.0, T.min(255.0, finite_or_infinite)), "uint8")


@T.func
def dequantize(value, row_scale, column_scale):
  """I32 -> FP32, then row-scale multiply, then column-scale multiply."""
  value = T.as_value(value)
  T.static_assert(value.dtype == "int32")
  return T.cast(value, "float32") * T.cast(T.as_value(row_scale), "float32") * T.cast(T.as_value(column_scale), "float32")


@T.func(inline=False)
def dot_into(inputs, output, *, packed_b, acc=None, epilogue=None, panel=None):
  """Fuse a supplied U8 activation DAG into I8 packed-B grouped GEMV.

  Each logical activation is evaluated in its producer's dtype/rounding
  contract. Only four valid K elements are quantized at a time, assembled as
  one dword and broadcast to output columns; no quantized MxK allocation is
  created. B is trusted native public SME storage for the identical logical
  K/N domain. Its capacity/density is checked here; the compiler adapter must
  retain its packed declaration and output-alias checks.
  acc is the caller-supplied I32 correction or a running split-K sum, added
  modulo 2**32. No bias correction is invented from zero_point. An epilogue
  may apply dequantize after the complete integer sum; intermediate K phases
  retain native I32. Output writes are delayed until all producers are read.
  panel=None derives public spatial geometry from SVL; an explicit matching
  panel supports portable reference/native testing of the same storage ABI.
  The private call boundary isolates this small grouped leaf from the native
  matrix fallback's allocation and register-state preparation.
  """
  a, b = inputs
  a, b = T.as_value(a), T.as_value(b)
  T.static_assert(a.dtype == "uint8" and b.dtype == "int8" and packed_b.dtype == "int8")
  T.static_assert(len(packed_b.shape) == 1)
  if acc is not None:
    T.static_assert(T.as_value(acc).dtype == "int32")
  m, k, n = a.shape[0], a.shape[1], b.shape[1]
  assert k >= 0 and b.shape[0] == k and output.shape[0] == m and output.shape[1] == n, "runtime quant dot shape mismatch"
  assert packed_b.strides[0] == 1, "runtime quant packed B must be dense"
  span = 2 * sme.streaming_lanes("int32") if panel is None else panel
  assert span > 0 and span <= 2305843009213693951, "runtime quant packed panel overflow"
  assert n <= 9223372036854775807 - (span - 1) and k <= 9223372036854775804, "runtime quant packed extent overflow"
  groups = (k + 3) // 4
  panels = (n + span - 1) // span
  if n > 0 and k > 0:
    assert groups <= packed_b.shape[0] // panels // span // 4, "runtime quant packed B capacity too small"
  lanes = T.lanes(T.Vector("uint32"))
  result = T.alloc((m, n), dtype="int32")
  if acc is not None:
    prior = T.broadcast_to(T.as_value(acc), (m, n))
  for row in T.serial(m, unroll=1):
    for pi in T.serial(panels, unroll=1):
      columns = T.min(span, n - pi * span)
      for ni in T.serial(0, columns, lanes, unroll=1):
        col = pi * span + ni
        active = T.min(lanes, columns - ni)
        if acc is not None:
          total = T.as_vector(T.bitcast(prior[row, col:col + active], "uint32"), T.Vector("uint32"))
        else:
          total = T.as_vector(T.zeros((active,), dtype="uint32"), T.Vector("uint32"))
        for ki in T.serial(groups, unroll=1):
          word = T.cast(0, "uint32")
          for member in T.unroll(4):
            if ki * 4 + member < k:
              activation = T.element_at(a, (row, ki * 4 + member))
              word = word | (T.cast(activation, "uint32") << (member * 8))
          # Keep the physical register descriptor through the byte view.
          # Slicing a full logical tile after reinterpretation loses that
          # geometry and can rebuild the broadcast one byte at a time.
          repeated_word = T.full(T.Vector("uint32"), word)
          repeated = T.as_vector(T.bitcast(repeated_word, "uint8")[:active * 4], T.Vector("uint8"))
          offset = (pi * groups + ki) * span * 4 + ni * 4
          weights = T.load(packed_b[offset:offset + active * 4], vector=T.Vector("int8"))
          total = T.as_vector(total + T.as_vector(T.widening_dot(repeated, weights, dtype="uint32"), T.Vector("uint32")), T.Vector("uint32"))
        T.store(result[row, col:col + active], T.bitcast(total, "int32"))
  finish_matrix(result, output, epilogue=epilogue)
