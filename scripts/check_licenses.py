# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Check SPDX fields on added files; never infer ownership or rewrite licenses.

Default scope is files added relative to origin/main, including the worktree.
Pass explicit paths to audit another reviewed set. Upstream license texts are
preserved verbatim; uncommentable formats need a .license sidecar. This is a
notice-presence check, not a legal/provenance audit.
"""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ("SPDX-FileCopyrightText:", "SPDX-License-Identifier:")
UPSTREAM_LICENSES = {
  "LICENSES/Arm-optimized-routines.txt",
  "LICENSES/FP16-MIT.txt",
  "LICENSES/KUPL-MulanPSL-2.0.txt",
  "LICENSES/LLVM-libc.txt",
  "LICENSES/PyTorch-BSD.txt",
  "LICENSES/SLEEF-BSL-1.0.txt",
}


def notice_errors(path, root=ROOT):
  """Require both SPDX fields near the top, or in an exact-path sidecar."""
  relative = path.relative_to(root).as_posix()
  if relative == "LICENSE" or relative in UPSTREAM_LICENSES:
    return []  # Full original grants, not project-owned source/prose.
  sidecar = Path(str(path) + ".license")
  notice = sidecar if sidecar.is_file() else path
  try:
    text = notice.read_text(encoding="utf-8").splitlines()[:25]
  except UnicodeError:
    return [f"{relative}: binary/uncommentable content needs a .license sidecar"]
  return [f"{relative}: missing {field}" for field in FIELDS if not any(field in line for line in text)]


def third_party_errors(root=ROOT):
  """Verify exact license-file references in the central provenance document."""
  notices = root / "THIRD_PARTY_NOTICES.md"
  if not notices.is_file():
    return ["missing THIRD_PARTY_NOTICES.md"]
  references = set(re.findall(r"\]\((LICENSES/[^)]+)\)", notices.read_text(encoding="utf-8")))
  return [f"missing referenced full license: {name}" for name in sorted(references) if not (root / name).is_file()]


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--base", default="origin/main")
  parser.add_argument("paths", nargs="*", help="Explicit reviewed repository-relative paths")
  arguments = parser.parse_args()
  names = arguments.paths or subprocess.check_output(
    ["git", "diff", "--diff-filter=A", "--name-only", "-z", arguments.base], cwd=ROOT
  ).decode().split("\0")
  paths = [ROOT / name for name in names if name and (ROOT / name).is_file()]
  errors = [error for path in paths for error in notice_errors(path)] + third_party_errors()
  for error in errors:
    print(error)
  print(f"License notice check: {len(paths)} files, {len(errors)} errors")
  return bool(errors)


if __name__ == "__main__":
  raise SystemExit(main())
