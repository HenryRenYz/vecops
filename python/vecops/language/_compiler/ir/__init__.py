# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
# ruff: noqa: F401 -- this module is the IR re-export facade
"""Internal structured IR vocabulary."""

from .collective import Barrier, Team, TeamReduce
from .kernel import (
  Allocate,
  AMXTile,
  ArchitectureCall,
  Assert,
  Assign,
  BindTile,
  Block,
  Control,
  ElementwiseGroup,
  Expr,
  For,
  Grid,
  HardwareRegion,
  If,
  IndexedStore,
  KernelIR,
  ReductionGroup,
  RowMap,
  SMEMask,
  SMEVector,
  Store,
  UnrolledFor,
  View,
  ViewRegion,
  While,
  binary,
  broadcast_shape,
  constant,
  literal,
  loads,
  loads_unique,
  same_expr,
  shaped_type,
)
from .matmul import DotOptions, Pack, PackedView, transpose_matrix
from .pipeline import PipelineFor, Prefetch
from .printing import FormattedText, Print
from .state import DeclareState, Exit, ExitRegion, TileState, UpdateState
from .types import Node, ScalarType, TileType, VectorTag, VectorType
from .workspace import WorkspaceAllocate
