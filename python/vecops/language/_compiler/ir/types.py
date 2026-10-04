"""Backend-independent scalar, tile and logical vector types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
  from .kernel import Expr


class Node:
  """Marker for owned IR dataclasses, independent of their module location."""


@dataclass(frozen=True)
class ScalarType(Node):
  dtype: str


@dataclass(frozen=True)
class TileType(Node):
  dtype: str
  shape: tuple[Expr, ...]
  mask: bool = False


@dataclass(frozen=True)
class VectorTag(Node):
  """Logical tag, independent of the native Vec's representation limits."""

  N: int | None = None
  pow: int = 0
  domain: str = "simd"
  basis_dtype: str | None = None


@dataclass(frozen=True)
class VectorType(TileType):
  tag: VectorTag = field(default_factory=VectorTag)
