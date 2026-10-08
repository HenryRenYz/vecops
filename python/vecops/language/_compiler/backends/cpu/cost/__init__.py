# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Resolve CPU profiles from the compilation target, not the build host."""

from __future__ import annotations

import platform
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from ....cost.defaults import DEFAULT
from ....cost.model import CostModel
from ....cost.profiles import register_profile, snapshot
from .generic import GENERIC
from .processors.sapphire_rapids import SAPPHIRE_RAPIDS
from .sme import SME
from .sve import SVE
from .x86 import AVX512, X86

for _profile in (DEFAULT, GENERIC, X86, AVX512, SVE, SME, SAPPHIRE_RAPIDS):
  register_profile(_profile)


@dataclass(frozen=True)
class CPUTarget:
  architecture: str = "generic"
  isa: str = "generic"
  processor: str = "generic"
  vector_bits: int | None = None


@lru_cache(maxsize=64)
def resolve_target(target="generic", *, processor=None):
  name = target.lower()
  architecture, isa, bits = "generic", "generic", None
  if name.startswith("native"):
    machine = platform.machine().lower()
    try:
      facts = Path("/proc/cpuinfo").read_text().lower()
    except OSError:
      facts = ""
    if machine in {"x86_64", "amd64"}:
      architecture = "x86"
      isa, bits = ("avx512", 512) if "avx512f" in facts else ("avx2", 256) if "avx2" in facts else ("scalar", 64)
      if processor is None and ("sapphire rapids" in facts or "w9-3495x" in facts):
        processor = "sapphire-rapids"
    elif machine in {"aarch64", "arm64"}:
      architecture = "arm"
      isa = "sme" if "sme" in facts else "sve" if "sve" in facts else "neon"
  elif name in {"avx", "avx2", "avx512"}:
    architecture, isa, bits = "x86", name, 512 if name == "avx512" else 256
  elif name in {"sve", "sve2", "sme", "neon"}:
    architecture, isa = "arm", "sve" if name == "sve2" else name
  elif name == "scalar":
    isa, bits = "scalar", 64
  elif name != "generic":
    raise ValueError(f"unknown CPU cost target {target!r}; pass an explicit CPUTarget")
  return CPUTarget(architecture, isa, processor or "generic", bits)


def build_model(catalog, target, overrides=()):
  registered = snapshot()
  names = ["default", "cpu"]
  if target.architecture == "x86":
    names.append("x86")
    if target.isa == "avx512":
      names.append("avx512")
  elif target.isa in {"sve", "sme"}:
    names.append("sve")
    if target.isa == "sme":
      names.append("sme")
  if target.processor in registered:
    names.append(target.processor)
  profiles = tuple(registered[name] for name in names)
  for override in overrides:
    profiles += (registered[override] if isinstance(override, str) else override,)
  return CostModel(profiles, catalog)
