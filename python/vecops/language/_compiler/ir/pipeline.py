# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Synchronous software pipeline annotations, not asynchronous promises."""

from dataclasses import dataclass

from .types import Node


@dataclass(frozen=True)
class Prefetch(Node):
  view: object
  level: str = "L1"
  intent: str = "read"
  temporal: bool = True


@dataclass(frozen=True)
class PipelineFor(Node):
  loop: object
  stages: int = 2
  distance: int = 1
  buffering: bool = False
