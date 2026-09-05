#!/usr/bin/env python3
"""Reject incomplete, phase-mixed, or under-referenced formal result CSVs."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

from merge_results import FIELDS


PROVIDERS = ("vecops", "OpenBLAS", "oneDNN", "LIBXSMM", "ACL", "KUPL-MMA")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    args = parser.parse_args()
    with args.input.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != FIELDS:
            parser.error("input does not use the current result schema")
        rows = list(reader)

    errors: list[str] = []
    keys = [tuple(row[field] for field in (
        "host", "library", "case", "op", "extent", "phase", "tuning"))
        for row in rows]
    duplicates = sum(count != 1 for count in Counter(keys).values())
    if duplicates:
        errors.append(f"duplicate result keys: {duplicates}")

    semantics = {(row["case"], row["op"]) for row in rows}
    hosts = {row["host"] for row in rows}
    expected = len(hosts) * len(semantics) * (len(PROVIDERS) + 1)
    # Each semantic has one row per external provider and two vecops extents.
    if len(rows) != expected:
        errors.append(f"row count {len(rows)} != expected {expected}")

    for row in rows:
        identity = f"{row['host']}/{row['library']}/{row['case']}/{row['op']}"
        if row["status"] not in ("ok", "timeout", "unsupported_arch"):
            errors.append(
                f"unexpected terminal status {row['status']!r}: {identity}: "
                f"{row['error']}")
        if row["status"] == "ok":
            if row["phase"] != "raw_e2e":
                errors.append(f"non-raw formal row: {identity}")
            if row["packing_included"] not in ("1", "1.0"):
                errors.append(f"packing not included: {identity}")
            if row["weights_prepacked"] not in ("0", "0.0"):
                errors.append(f"prepacked formal weights: {identity}")
            if not row["median_us"]:
                errors.append(f"ok row has no median: {identity}")

    ok = {(row["host"], row["library"], row["case"], row["op"]): row
          for row in rows if row["status"] == "ok"}
    for row in rows:
        if row["status"] != "ok":
            continue
        identity = f"{row['host']}/{row['library']}/{row['case']}/{row['op']}"
        openblas = ok.get((row["host"], "OpenBLAS", row["case"], row["op"]))
        if openblas is not None and not row["speedup_vs_openblas"]:
            errors.append(f"missing same-phase OpenBLAS speedup: {identity}")
        dynamic_extent = "AllDynamic" if row["model"] == "general" \
            else "TokenDynamic"
        dynamic = next((candidate for candidate in rows
                        if candidate["host"] == row["host"]
                        and candidate["library"] == "vecops"
                        and candidate["case"] == row["case"]
                        and candidate["op"] == row["op"]
                        and candidate["extent"] == dynamic_extent
                        and candidate["status"] == "ok"), None)
        const = next((candidate for candidate in rows
                      if candidate["host"] == row["host"]
                      and candidate["library"] == "vecops"
                      and candidate["case"] == row["case"]
                      and candidate["op"] == row["op"]
                      and candidate["extent"] == "AllConst"
                      and candidate["status"] == "ok"), None)
        if dynamic is not None and not row["vecops_dynamic_speedup_vs_row"]:
            errors.append(f"missing vecops dynamic speedup: {identity}")
        if const is not None and not row["vecops_const_speedup_vs_row"]:
            errors.append(f"missing vecops const speedup: {identity}")

    if errors:
        for error in errors[:100]:
            print(f"ERROR: {error}")
        if len(errors) > 100:
            print(f"ERROR: ... {len(errors) - 100} more")
        return 1
    print(f"validated {len(rows)} rows, {len(hosts)} hosts, "
          f"{len(semantics)} semantic cases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
