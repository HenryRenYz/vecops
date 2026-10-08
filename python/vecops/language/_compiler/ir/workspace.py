# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Call-scoped semantic allocation sites, supplied by an execution provider."""

from dataclasses import dataclass

from .kernel import View
from .types import Node


@dataclass(frozen=True)
class WorkspaceAllocate(Node):
  view: View
