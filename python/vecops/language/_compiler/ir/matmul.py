"""Matrix products retain operand DAGs, accumulator and packing provenance.

Logical B is [K,N], unlike the C++ operator's [N,K] convention. Packed
operands record that distinction explicitly; a physical buffer is never
mistaken for a dense logical tensor.
"""

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


@dataclass(frozen=True)
class MatmulPlan:
  expression: object
  options: DotOptions
  vector_rows: int = 4
  vector_columns: int = 4
  k_unroll: int = 2
  unblocked_bytes: int = 512 * 1024
  runtime_unblocked_bytes: int = 128 * 1024
  amx_m_major: object | None = None
  amx_native_b_pack_min_m: int = 0
  amx_cache_b_panel_divisor: int = 0
  amx_cache_b_panel_max_m: int = 0
  # Experimental ordinary-SVE quantize/packed-B row driver. Profitability
  # remains opt-in until paired target measurements establish a crossover.
  sve_runtime_quant: bool = False
  # Backend-owned immutable native-A gather/packing cost policy. The leaf
  # evaluates actual panel extents, not a guessed size for dynamic metadata.
  sme_native_a_packing: object = None
  # Profile-selected whole-problem traversal for prepacked BF16 AB. Eligibility
  # uses captured SVL and runtime private L2; explicit blocks remain binding.
  sme_packed_ab_cache_bypass: bool = False
  sme_packed_ab_traversal: object = None
  # Narrow memory/elementwise work on a statically singleton ZA slice or
  # unpacked BF16 spatial K slot. ZA instructions retain native-word geometry.
  sme_narrow_singleton_slices: bool = False
  operands: tuple = ()
  amx_b_permutation: object | None = None
  dense_transposes: tuple = ()
  schedule: object | None = None
  pass_trace: tuple = ("cpu-plan-matmul",)


@dataclass(frozen=True)
class OperandFusion:
  expression: object
  materialization: object | None
  condition: object | None
  operation_cost: float
  reason: str


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
    data = (
      replace(data, shape=shape, strides=(data.strides[0], literal(0)))
      if rank == 1
      else replace(data, shape=data.shape[::-1], strides=data.strides[::-1])
    )
  elif value.op == "arange":
    data = 0 if rank == 1 else 1 - (data if data is not None else 1)
  elif value.op == "dot":
    args = (args[1], args[0], *args[2:])
    data = replace(data, block_m=data.block_n, block_n=data.block_m)
  result = replace(value, type=TileType(value.type.dtype, shape, value.type.mask), args=args, data=data)
  memo[id(value)] = result
  return result
