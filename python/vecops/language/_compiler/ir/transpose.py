"""Explicit register permutation DAGs, independent of C++ representation.

Fixed extents refer to *logical* lanes. Named values, rather than aggregate
register arrays, make the same DAG legal for fixed and sizeless SVE vectors.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class PermutationOp:
  kind: str
  lanes: int
  operands: tuple[int, ...]
  # Bitcast/interleave stage width, not a numeric conversion.
  element_bytes: int = 0


@dataclass(frozen=True)
class RegisterTranspose:
  rows: int
  columns: int
  operations: tuple[PermutationOp, ...]
  outputs: tuple[int, ...]
  # A permutation commutes with a sole boundary conversion. The backend may
  # keep narrow source bits through the network and widen at the store.
  dtype: str | None = None


@dataclass(frozen=True)
class TransposePlan:
  statement: object
  dtype: str
  source_axis: int
  variants: tuple[tuple[str, RegisterTranspose], ...]
  sme: bool = False
  sme_preferred: bool = True
  sme_pairs: bool = False
  cache_block: int = 64
  loop_order: str = "output"
  dense_destination: bool = False
  # Maximum lane width of memory and typed intermediates, not just compute.
  lane_bytes: int = 0
  pass_trace: tuple[str, ...] = ("cpu-plan-transpose", "lower-register-permutations")
