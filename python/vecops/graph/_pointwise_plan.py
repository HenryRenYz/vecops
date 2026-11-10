# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""One metadata-only pointwise selection rule for discovery and real binding."""

import os

from . import _int_scalar, _pointwise, _reduction


def select_kernel(schema, arguments, result, metadata, parallelism, vector_token):
  """Select the existing source kernel without allocating or binding storage."""
  inplace = schema.name in {
    "aten::add_",
    "aten::sub_",
    "aten::mul_",
    "aten::div_",
    "aten::copy_",
    "aten::fill_",
  }
  reduction = (
    _reduction.select(schema, arguments, result, metadata)
    if os.environ.get("VECOPS_GRAPH_SHORT_REDUCTION", "1") != "0"
    else None
  )
  integer = (
    _int_scalar.select(schema, arguments, result, metadata)
    if os.environ.get("VECOPS_GRAPH_INTEGER_SCALARS", "1") != "0"
    else None
  )
  selected = (
    (31, integer[0], integer[0], integer[1])
    if integer is not None
    else (30, reduction, reduction, 0.0)
    if reduction is not None
    else _pointwise.select(schema, arguments, result, metadata)
  )
  if selected is None:
    return None
  code, a, b, alpha = selected
  output = arguments[0] if inplace else result
  am, bm, om = metadata(a), metadata(b), metadata(output)
  if any(m.device != "cpu" or (m.shape and m.shape[-1] != 1 and m.strides[-1] != 1) for m in (am, bm, om)):
    return None
  if not (
    _pointwise.contiguous_layout(om.shape, om.strides)
    or (schema.name == "aten::copy_" and _pointwise.nonoverlapping_layout(om.shape, om.strides))
  ):
    return None
  if code == 31:
    compiled = _int_scalar.kernel(am.shape, am.strides, om.shape, om.strides, parallelism, vector_token)
  elif code == 30:
    compiled = _reduction.kernel(
      am.shape,
      am.strides,
      am.dtype.removeprefix("torch."),
      om.shape,
      om.strides,
      om.dtype.removeprefix("torch."),
      parallelism,
      vector_token,
    )
  else:
    compiled = _pointwise.kernel(
      code,
      am.shape or (1,),
      am.strides or (1,),
      am.dtype.removeprefix("torch."),
      bm.shape or (1,),
      bm.strides or (1,),
      bm.dtype.removeprefix("torch."),
      om.shape or (1,),
      om.strides or (1,),
      om.dtype.removeprefix("torch."),
      parallelism,
      vector_token,
    )
  return compiled, a, b, output, alpha, inplace
