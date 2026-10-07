#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Report Const/Dynamic median ratios from Google Benchmark JSON files."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path


EXTENT_RE = re.compile(r"/extent:(Dynamic|Const)(?=/)")


def normalized_name(entry: dict[str, object]) -> tuple[str, str] | None:
    name = str(entry.get("run_name", entry.get("name", "")))
    match = EXTENT_RE.search(name)
    if match is None:
        return None
    return EXTENT_RE.sub("", name), match.group(1)


def load_entries(
    paths: list[Path],
) -> dict[str, dict[str, dict[str, dict[str, object]]]]:
    pairs: dict[str, dict[str, dict[str, dict[str, object]]]] = {}
    for path in paths:
        with path.open(encoding="utf-8") as stream:
            document = json.load(stream)
        for entry in document.get("benchmarks", []):
            aggregate = str(entry.get("aggregate_name", ""))
            if aggregate not in {"median", "cv"}:
                continue
            normalized = normalized_name(entry)
            if normalized is None:
                continue
            key, extent = normalized
            pairs.setdefault(key, {}).setdefault(extent, {})[aggregate] = entry
    return pairs


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: compare_matmul_extents.py RESULT.json [...]", file=sys.stderr)
        return 2
    pairs = load_entries([Path(argument) for argument in sys.argv[1:]])
    missing = [
        key
        for key, variants in pairs.items()
        if any(
            extent not in variants or "median" not in variants[extent]
            for extent in ("Dynamic", "Const")
        )
    ]
    print(
        "case\tdynamic_cpu\tconst_cpu\tunit\t"
        "dynamic_cv_pct\tconst_cv_pct\tconst_over_dynamic"
    )
    for key, variants in sorted(pairs.items()):
        if "Dynamic" not in variants or "Const" not in variants:
            continue
        dynamic = variants["Dynamic"]["median"]
        constant = variants["Const"]["median"]
        dynamic_cpu = float(dynamic["cpu_time"])
        constant_cpu = float(constant["cpu_time"])
        dynamic_cv = float(
            variants["Dynamic"].get("cv", {}).get("cpu_time", float("nan"))
        )
        constant_cv = float(
            variants["Const"].get("cv", {}).get("cpu_time", float("nan"))
        )
        ratio = constant_cpu / dynamic_cpu if dynamic_cpu else float("nan")
        unit = str(dynamic.get("time_unit", ""))
        print(
            f"{key}\t{dynamic_cpu:.9g}\t{constant_cpu:.9g}\t{unit}\t"
            f"{100.0 * dynamic_cv:.4f}\t{100.0 * constant_cv:.4f}\t"
            f"{ratio:.6f}"
        )
    if missing:
        for key in sorted(missing):
            print(f"missing median peer: {key}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
