# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Matrix products retain operand DAGs, accumulator and packing provenance.

Logical B is [K,N], unlike the C++ operator's [N,K] convention. Packed
operands record that distinction explicitly; a physical buffer is never
mistaken for a dense logical tensor.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from .types import Node


@dataclass(frozen=True)
class PackedView(Node):
  view: object
  shape: tuple
  side: str
  format: str
  # Logical panel origin; shape above still describes the original physical
  # packed ABI. Slicing must never recompute physical strides from panel sizes.
  origins: tuple = ()


@dataclass(frozen=True)
class Pack(Node):
  source: object
  destination: PackedView


@dataclass(frozen=True)
class DotOptions(Node):
  tiled: bool = False
  packing: str = "auto"
  implementation: str = "auto"
  block_m: int | None = None
  block_n: int | None = None
  block_k: int | None = None
  swap: bool = True


def transpose_matrix(value, memo=None):
  """Transpose a matrix DAG, promoting trailing rank-one broadcasts first."""
  from .kernel import literal
  from .types import TileType

  if not isinstance(value.type, TileType):
    return value
  memo = {} if memo is None else memo
  if id(value) in memo:
    return memo[id(value)]
  rank = len(value.type.shape)
  if rank not in (1, 2) or value.op == "packed_load":
    raise TypeError("matrix transpose requires unpacked rank-one/two operands")
  shape = (value.type.shape[0], literal(1)) if rank == 1 else value.type.shape[::-1]
  data = value.data
  args = tuple(transpose_matrix(a, memo) for a in value.args)
  if value.op == "load":
    region = data.region
    if rank == 1 and region is not None:
      # A transposed trailing broadcast gains a unit axis. Retain every
      # original root coordinate, then append that degenerate coordinate;
      # a rank-two view cannot retain a one-entry axes mapping.
      region = replace(region, root_shape=(*region.root_shape, literal(1)),
                       root_strides=(*region.root_strides, literal(0)),
                       starts=(*region.starts, literal(0)), widths=(*region.widths, literal(1)),
                       steps=(*region.steps, 1), axes=(region.axes[0], len(region.root_shape)),
                       clipped=(*region.clipped, False),
                       exact_starts=(*region.exact_starts, literal(0)) if region.exact_starts is not None else None)
    data = (
      replace(data, shape=shape, strides=(data.strides[0], literal(0)), region=region)
      if rank == 1
      else replace(data, shape=data.shape[::-1], strides=data.strides[::-1],
                   region=replace(data.region, axes=data.region.axes[::-1]) if data.region is not None else None)
    )
  elif value.op == "arange":
    data = 0 if rank == 1 else 1 - (data if data is not None else 1)
  elif value.op == "dot":
    args = (args[1], args[0], *args[2:])
    data = replace(data, block_m=data.block_n, block_n=data.block_m)
  result = replace(value, type=TileType(value.type.dtype, shape, value.type.mask), args=args, data=data)
  memo[id(value)] = result
  return result
