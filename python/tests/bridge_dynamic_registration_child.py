# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Subprocess target for the fresh-process bridge cache test."""

from __future__ import annotations

import json
import os
import runpy
import sys
from pathlib import Path


def main() -> None:
  root = Path(os.environ["VECOPS_BRIDGE_TEST_ROOT"]).resolve()
  expected = json.loads(os.environ["VECOPS_BRIDGE_TEST_DISTRIBUTION"])
  entry = runpy.run_path(str(root / "benchmarks/dsl_snapshot_entry.py"))
  package = Path(expected["package"]).parent
  extension = Path(expected["extension"])
  # A complete private wheel/snapshot can be selected without the editable
  # finder. A source-only editable checkout cannot: its finder owns _C, so
  # leave it intact and verify the import rather than silently mixing builds.
  local_extension = package / extension.name
  if local_extension.is_file() and local_extension.resolve() == extension:
    sys.meta_path[:] = [finder for finder in sys.meta_path if not entry["claims_vecops"](finder)]
    sys.path.insert(0, str(package.parent))
  import torch
  import vecops
  import vecops.language as T
  from vecops._compiler import _sdk

  sdk, layout = _sdk()
  actual = {
    "package": str(Path(vecops.__file__).resolve()),
    "extension": str(Path(vecops._C.__file__).resolve()),
    "sdk": str(sdk.resolve()),
    "sdk_layout": str(layout),
  }
  if actual != expected:
    raise RuntimeError(f"child distribution differs from parent: expected {expected}, got {actual}")

  mode = os.environ["VECOPS_BRIDGE_MODE"]
  torch.set_num_threads(2)
  compiler = vecops.Compiler(
    target="Native",
    cache_mode=mode,
    cache_dir=os.environ["VECOPS_BRIDGE_CACHE"],
    build_dir=os.environ["VECOPS_BRIDGE_BUILD"],
    cc=os.environ.get("CC"),
    cxx=os.environ.get("CXX"),
  )

  @T.jit(out_idx=-1, compiler=compiler)
  def factory():
    @T.template
    def copy(A: T.Tensor((4,), dtype="float32"), C: T.Tensor((4,), dtype="float32")):
      for index in T.grid(4, parallel="require"):
        T.store(C[index], T.load(A[index]))

    return copy

  values = torch.arange(4, dtype=torch.float32)
  torch.testing.assert_close(factory()(values), values)
  print(factory()._core.torch().__name__)


if __name__ == "__main__":
  main()
