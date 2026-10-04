"""Immutable per-compilation diagnostics configuration."""

from dataclasses import dataclass


@dataclass(frozen=True)
class DebugOptions:
  trace: str = "off"
  passes: tuple[str, ...] = ()
  prints: bool = True
  max_print_elements: int = 16
  memory: bool = False
  timeline: bool = False

  def __post_init__(self):
    if self.trace not in {"off", "changed", "all"}:
      raise ValueError("debug trace must be off, changed or all")
    if not isinstance(self.passes, tuple) or any(not isinstance(p, str) for p in self.passes):
      raise TypeError("debug passes must be an immutable tuple of names")
    if type(self.prints) is not bool or type(self.max_print_elements) is not int or self.max_print_elements < 1:
      raise ValueError("prints must be boolean and max_print_elements must be positive")
    if type(self.memory) is not bool:
      raise TypeError("debug memory must be boolean")
    if type(self.timeline) is not bool:
      raise TypeError("debug timeline must be boolean")


@dataclass(frozen=True)
class PrintTarget:
  """A portable output destination, never a captured Python file object."""

  name: str

  def __post_init__(self):
    if self.name not in {"stdout", "stderr"}:
      raise ValueError("kernel print target must be stdout or stderr")


stdout = PrintTarget("stdout")
stderr = PrintTarget("stderr")
