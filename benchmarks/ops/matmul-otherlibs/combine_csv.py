#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Combine already-normalized benchmark CSVs without losing host metadata."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from merge_results import FIELDS


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--common-cases-across-hosts", action="store_true",
        help="retain only case IDs present on every input host")
    parser.add_argument(
        "--latest-wins", action="store_true",
        help="deduplicate result keys, with rows from later inputs winning")
    args = parser.parse_args()

    rows: list[dict[str, str]] = []
    for path in args.inputs:
        with path.open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != FIELDS:
                parser.error(f"{path} does not use the current result schema")
            rows.extend(reader)
    if args.latest_wins:
        keyed: dict[tuple[str, ...], dict[str, str]] = {}
        for row in rows:
            key = tuple(row[field] for field in (
                "host", "library", "case", "op", "extent", "phase", "tuning"))
            keyed[key] = row
        rows = list(keyed.values())
    if args.common_cases_across_hosts:
        cases_by_host: dict[str, set[str]] = {}
        for row in rows:
            cases_by_host.setdefault(row["host"], set()).add(row["case"])
        if cases_by_host:
            common_cases = set.intersection(*cases_by_host.values())
            rows = [row for row in rows if row["case"] in common_cases]
    rows.sort(key=lambda row: tuple(row[field] for field in (
        "host", "case", "op", "phase", "library", "extent", "tuning")))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
