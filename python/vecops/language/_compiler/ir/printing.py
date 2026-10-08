# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Observable record output; not an elementwise value or a synchronization op."""

from dataclasses import dataclass

from .types import Node


@dataclass(frozen=True)
class FormattedText(Node):
  format: str
  arguments: tuple


@dataclass(frozen=True)
class Print(Node):
  arguments: tuple
  separator: str
  end: str
  destination: str
  flush: object
  origin: tuple
  buffer_size: int = 1024
  truncation: str = " ... (trunc)"
