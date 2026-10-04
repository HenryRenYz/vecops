"""Kernel inspection, tracing and profiling; presentation formats may evolve."""

from ._debug.interpreter import UnsupportedInterpretation
from ._debug.machine import llvm_mca
from ._debug.minimize import minimize
from ._debug.options import DebugOptions, PrintTarget, stderr, stdout
from ._debug.profile import ProfileResult, compare
from ._debug.report import DebugInfo, LoweredKernel, Unavailable, lower

__all__ = [
  "DebugInfo",
  "DebugOptions",
  "LoweredKernel",
  "PrintTarget",
  "ProfileResult",
  "Unavailable",
  "UnsupportedInterpretation",
  "compare",
  "llvm_mca",
  "lower",
  "minimize",
  "stderr",
  "stdout",
]
