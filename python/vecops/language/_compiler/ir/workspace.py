"""Call-scoped semantic allocation sites, supplied by an execution provider."""

from dataclasses import dataclass
from .types import Node
from .kernel import View


@dataclass(frozen=True)
class WorkspaceAllocate(Node):
  view: View
