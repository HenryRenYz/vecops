"""Backend compilation boundary; other targets need not use Vector IR or C++."""

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class LoweredArtifact:
  kernel: object
  lowered: object
  source: str
  backend_ir: object | None = None


class Backend(Protocol):
  """Own target legalization and emission, after source capture.

  The current driver consumes a source artifact for the native build adapter.
  Additional targets may provide their own build/runtime adapter rather than
  pretending to have CPU vector tags or a C++ emitter.
  """

  def lower(self, template, out_idx, *, pipeline=None, options=None) -> LoweredArtifact: ...
