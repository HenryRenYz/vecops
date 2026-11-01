# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Conservative static-metadata lowering of floating point ATen operations.

The caller owns independent output storage. No ATen dispatcher or allocator is
entered by these grids. Each operation preserves its original dtype boundary;
this is scheduling composition, not algebraic expression fusion.
"""
from functools import lru_cache


def contiguous_layout(shape, strides):
  expected = 1
  for size, stride in reversed(tuple(zip(shape, strides))):
    if size != 1 and stride != expected:
      return False
    expected *= size
  return True


def nonoverlapping_layout(shape, strides):
  if any(size == 0 for size in shape):
    return True
  span = 1
  for size, stride in sorted(((n, s) for n, s in zip(shape, strides) if n > 1), key=lambda x: x[1]):
    if stride < span:
      return False
    span += (size - 1) * stride
  return True


@lru_cache(None)
def kernel(operation, a_shape, a_strides, a_dtype, b_shape, b_strides, b_dtype,
           output_shape, output_strides, output_dtype, parallelism, vector_token):
  import vecops.language as T

  @T.jit(out_idx=[2])
  def factory():
    Operation = T.Const(operation)
    OutputContiguous = T.Const(int(contiguous_layout(output_shape, output_strides)))

    @T.template(source='_Pointwise.cpp', recipe_id='graph-pointwise-v4')
    def entry(a: T.Tensor(a_shape, a_strides, a_dtype),
              b: T.Tensor(b_shape, b_strides, b_dtype),
              output: T.Tensor(output_shape, output_strides, output_dtype),
              alpha: T.float64):
      Operation, OutputContiguous
    return entry

  compiled = factory()
  compiled.explicit_torch_function()  # Declare the bridge; batch it with peers.
  return compiled


def select(schema, arguments, result, metadata):
  """Return (opcode, lhs-ref, rhs-ref, alpha) or None for unsupported semantics."""
  name, overload = schema.name, schema.overload_name
  floating = {'torch.float32', 'torch.bfloat16', 'torch.float16'}
  def broadcast_bool(ref):
    if not hasattr(ref, 'id'):
      return False
    m = metadata(ref)
    return m.dtype == 'torch.bool' and (not m.shape or m.shape[-1] == 1)
  if not hasattr(result, 'id') or metadata(result).dtype not in floating:
    return None
  def operand(ref):
    m = metadata(ref)
    return m.dtype in floating or (
        m.dtype == 'torch.float64' and not m.shape and metadata(result).dtype == 'torch.float32')
  if name == 'aten::fill_' and len(arguments) == 2 and type(arguments[1]) in (int, float):
    # The scalar ABI is FP64. Avoid double rounding of large integer fills
    # whose direct int64->float32 conversion differs from int64->FP64->FP32.
    if type(arguments[1]) is int and abs(arguments[1]) > (1 << 53):
      return None
    return 20, arguments[0], arguments[0], float(arguments[1])
  if name == 'aten::pow' and overload == 'Tensor_Scalar' and arguments[1] == 2:
    return 21, arguments[0], arguments[0], 0.
  if name == 'aten::sqrt' and not metadata(arguments[0]).shape:
    return 22, arguments[0], arguments[0], 0.
  if name == 'aten::copy_':
    a = arguments[1]
    if hasattr(a, 'id') and (operand(a) or broadcast_bool(a)):
      return 0, a, a, 0.
    return None
  if name in {'aten::add_', 'aten::sub_', 'aten::mul_', 'aten::div_'}:
    name = name[:-1]
  if name in {'aten::_to_copy', 'aten::clone', 'aten::contiguous'} or (
      name == 'aten::to' and overload in {'dtype', 'device', 'dtype_layout'}):
    a = arguments[0]
    # Only plain floating conversions/copies. Layout/device/pinning semantics
    # are guarded by capture and by the output metadata below.
    if (operand(a) or broadcast_bool(a)) and a.id != result.id:
      return 0, a, a, 0.
  if name in {'aten::relu', 'aten::neg'}:
    a = arguments[0]
    if metadata(a).dtype in floating:
      return (5 if name == 'aten::relu' else 6), a, a, 0.
  if name in {'aten::add', 'aten::sub', 'aten::mul', 'aten::div'} and overload in {'Tensor', 'Scalar'}:
    a, b = arguments[:2]
    if not hasattr(a, 'id') or not (operand(a) or
                                  (name == 'aten::mul' and broadcast_bool(a))):
      return None
    code = {'aten::add': 1, 'aten::sub': 2, 'aten::mul': 3, 'aten::div': 4}[name]
    if hasattr(b, 'id'):
      if not (operand(b) or (name == 'aten::mul' and broadcast_bool(b))):
        return None
      alpha = arguments[2] if len(arguments) > 2 else 1.
      if type(alpha) not in (int, float):
        return None
      return code, a, b, float(alpha)
    if type(b) in (int, float):
      alpha = arguments[2] if len(arguments) > 2 else 1.
      if type(alpha) not in (int, float):
        return None
      # Scalar addition/subtraction can absorb alpha without changing an
      # intermediate Tensor rounding boundary.
      return code + 10, a, a, float(b * alpha if code in (1, 2) else b)
  return None
