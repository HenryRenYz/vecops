#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Audit provider coverage and vecops Meta extent pairs from benchmark lists."""

from __future__ import annotations

import argparse
import subprocess
from collections import Counter, defaultdict
from pathlib import Path


def list_names(executable: Path) -> list[str]:
    result = subprocess.run(
        [str(executable), "--benchmark_list_tests"], check=True,
        text=True, stdout=subprocess.PIPE)
    return [line.strip() for line in result.stdout.splitlines()
            if line.strip().startswith("MatmulOtherlibs/")]


def fields(name: str) -> dict[str, str]:
    result = {}
    for component in name.split("/")[1:]:
        if ":" in component:
            key, value = component.split(":", 1)
            result[key] = value
    return result


def semantic_key(item: dict[str, str]) -> tuple[str, ...]:
    return tuple(item[key] for key in (
        "case", "model", "subgraph", "shape", "batch", "tokens", "chunk",
        "op", "dtype", "orientation"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True, type=Path,
                        help="complete external provider, normally OpenBLAS")
    parser.add_argument("--vecops", required=True, action="append", type=Path,
                        help="repeat for the core and fusion executables")
    parser.add_argument("--provider", action="append", default=[], type=Path)
    args = parser.parse_args()

    failures: list[str] = []
    reference = [fields(name) for name in list_names(args.reference)]
    if any(item.get("phase") != "raw_e2e" for item in reference):
        failures.append("reference provider contains a non-raw formal row")
    expected = {semantic_key(item) for item in reference}
    if len(expected) != len(reference):
        failures.append("reference provider contains duplicate semantic cases")

    vecops = [fields(name) for executable in args.vecops
              for name in list_names(executable)]
    extent_sets: dict[tuple[str, ...], set[str]] = defaultdict(set)
    for item in vecops:
        if item.get("phase") != "raw_e2e":
            failures.append(
                f"vecops formal row is not raw_e2e: {item.get('case')}/"
                f"{item.get('op')}")
        extent_sets[semantic_key(item)].add(item["extent"])
    if set(extent_sets) != expected:
        failures.append(
            f"vecops coverage differs: missing={len(expected-set(extent_sets))}, "
            f"extra={len(set(extent_sets)-expected)}")
    for key, extents in extent_sets.items():
        model = key[1]
        wanted = {"AllDynamic", "AllConst"} if model == "general" else {
            "TokenDynamic", "AllConst"}
        if extents != wanted:
            failures.append(
                f"{key[0]}/{key[7]} extent mismatch: {sorted(extents)} != "
                f"{sorted(wanted)}")

    for executable in args.provider:
        records = [fields(name) for name in list_names(executable)]
        if any(item.get("phase") != "raw_e2e" for item in records):
            failures.append(f"{executable.name} contains a non-raw formal row")
        actual = {semantic_key(item) for item in records}
        label = executable.name
        if actual != expected:
            failures.append(
                f"{label} coverage differs: missing={len(expected-actual)}, "
                f"extra={len(actual-expected)}")
        counts = Counter(semantic_key(item) for item in records)
        duplicates = sum(count != 1 for count in counts.values())
        if duplicates:
            failures.append(f"{label} has {duplicates} duplicate cases")

    if failures:
        for failure in failures:
            print(f"ERROR: {failure}")
        return 1
    print(f"catalog audit passed: {len(expected)} semantic cases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
