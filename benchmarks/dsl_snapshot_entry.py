"""Run a frozen source snapshot without changing an editable installation.

Use in a fresh validation process. The snapshot must contain its own native
extension and SDK (copies or links to the matching installed build). Only this
process's vecops editable finder is disabled; other editable packages and the
af3 environment are untouched. Actual import paths are checked, not inferred
from PYTHONPATH, because scikit-build's finder takes precedence over it.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import runpy
import sys
from pathlib import Path


def claims_vecops(finder):
  """Identify only the editable finder that owns vecops import entries."""
  if type(finder).__name__ != "ScikitBuildRedirectingFinder":
    return False
  return any(
    name == "vecops" or name.startswith("vecops.")
    for attribute in ("known_source_files", "known_wheel_files")
    for name in getattr(finder, attribute, {})
  )


def select_snapshot(root):
  root = Path(root).resolve()
  expected = root / "python/vecops/__init__.py"
  if not expected.is_file() or not (root / "CMakeLists.txt").is_file():
    raise ValueError("snapshot needs python/vecops/__init__.py and CMakeLists.txt")
  if any(name == "vecops" or name.startswith("vecops.") for name in sys.modules):
    raise RuntimeError("snapshot validation must start in a fresh process before importing vecops")
  sys.meta_path[:] = [finder for finder in sys.meta_path if not claims_vecops(finder)]
  sys.path.insert(0, str(root / "python"))
  spec = importlib.util.find_spec("vecops")
  if spec is None or not spec.origin or Path(spec.origin).resolve() != expected:
    raise RuntimeError(f"snapshot import was redirected: expected {expected}, got {getattr(spec, 'origin', None)}")
  return root


def main(argv=None):
  arguments = list(sys.argv[1:] if argv is None else argv)
  split = arguments.index("--") if "--" in arguments else len(arguments)
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
  own = parser.parse_args(arguments[:split])
  forwarded = arguments[split + 1 :]
  root = select_snapshot(own.root)
  import vecops

  expected = root / "python/vecops/__init__.py"
  if Path(vecops.__file__).resolve() != expected:
    raise RuntimeError(f"vecops import escaped the source snapshot: {vecops.__file__}")
  # Relative pytest selectors and source inspection must use the same tree.
  # Supply absolute artifact paths when results belong outside that snapshot.
  os.chdir(root)
  print(
    "Frozen source selection: "
    + json.dumps(
      {
        "root": str(root),
        "working_directory": str(Path.cwd()),
        "python_package": vecops.__file__,
        "native_extension": vecops._C.__file__,
      }
    ),
    flush=True,
  )
  # Pytest selectors and inspect.getsource must refer to the same immutable
  # tree as the package. Leaving cwd at a live editable tree makes tests load
  # mutable source, even though vecops itself was selected correctly.
  runner = root / "benchmarks/dsl_arm_validate.py"
  sys.argv = [str(runner), *forwarded]
  runpy.run_path(str(runner), run_name="__main__")


if __name__ == "__main__":
  main()
