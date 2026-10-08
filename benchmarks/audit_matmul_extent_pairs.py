#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Verify that every shape-bearing Matmul benchmark has Dynamic/Const peers."""

from __future__ import annotations

import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path


EXTENT_RE = re.compile(r"/extent:(Dynamic|Const)(?=/)")


def list_benchmarks(executable: Path) -> list[str]:
    result = subprocess.run(
        [str(executable), "--benchmark_list_tests"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return [line.strip() for line in result.stdout.splitlines() if "/shape:" in line]


def audit(executable: Path) -> tuple[int, int, int]:
    variants: dict[str, set[str]] = defaultdict(set)
    malformed: list[str] = []
    benchmarks = list_benchmarks(executable)
    duplicate_names = {
        name: count for name, count in Counter(benchmarks).items() if count != 1
    }
    for benchmark in benchmarks:
        match = EXTENT_RE.search(benchmark)
        if match is None:
            malformed.append(benchmark)
            continue
        extent = match.group(1)
        key = EXTENT_RE.sub("", benchmark)
        variants[key].add(extent)

    unpaired = {
        key: modes
        for key, modes in variants.items()
        if modes != {"Dynamic", "Const"}
    }
    for benchmark in malformed:
        print(f"{executable}: missing extent: {benchmark}", file=sys.stderr)
    for key, modes in sorted(unpaired.items()):
        print(
            f"{executable}: unpaired {key}: {','.join(sorted(modes))}",
            file=sys.stderr,
        )
    for name, count in sorted(duplicate_names.items()):
        print(
            f"{executable}: duplicate registration x{count}: {name}",
            file=sys.stderr,
        )
    print(
        f"{executable}: logical_cases={len(variants)} "
        f"registered_variants={sum(len(modes) for modes in variants.values())} "
        f"malformed={len(malformed)} unpaired={len(unpaired)} "
        f"duplicates={len(duplicate_names)}"
    )
    return len(malformed), len(unpaired), len(duplicate_names)


def main() -> int:
    if len(sys.argv) < 2:
        print(
            "usage: audit_matmul_extent_pairs.py BENCHMARK [BENCHMARK ...]",
            file=sys.stderr,
        )
        return 2
    failures = 0
    for argument in sys.argv[1:]:
        executable = Path(argument).resolve()
        if not executable.is_file():
            print(f"missing benchmark executable: {executable}", file=sys.stderr)
            failures += 1
            continue
        malformed, unpaired, duplicates = audit(executable)
        failures += malformed + unpaired + duplicates
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
