"""Typed value state across structured control flow, independent of storage.

Backends choose registers or private memory; neither a C++ array nor an ISA
register count is part of the language contract. Updates preserve dtype,
predicate kind and logical extent. Ordinary value aliases remain snapshots.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .types import Node, TileType

if TYPE_CHECKING:
  from .kernel import Block, Expr


@dataclass(frozen=True)
class TileState(Node):
  name: str
  type: TileType


@dataclass(frozen=True)
class DeclareState(Node):
  state: TileState
  value: Expr


@dataclass(frozen=True)
class UpdateState(Node):
  state: TileState
  value: Expr


@dataclass(frozen=True)
class ExitRegion(Node):
  """A structured single-entry region with an early-exit continuation."""

  name: str
  body: Block


@dataclass(frozen=True)
class Exit(Node):
  region: str
