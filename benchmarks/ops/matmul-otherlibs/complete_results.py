#!/usr/bin/env python3
"""Expand measured rows into a complete host/case/provider result matrix."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from merge_results import FIELDS


PROVIDERS = ("vecops", "OpenBLAS", "oneDNN", "LIBXSMM", "ACL", "KUPL-MMA")
PHASE = {
    "vecops": "raw_e2e",
    "OpenBLAS": "raw_e2e",
    "oneDNN": "prepared_execute",
    "LIBXSMM": "prepared_execute",
    "ACL": "prepared_execute",
    "KUPL-MMA": "raw_e2e",
}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != FIELDS:
            raise ValueError(f"{path} does not use the current result schema")
        return list(reader)


def epilogue(provider: str, operation: str) -> str:
    if operation in ("gemm", "gemm_add"):
        return "native"
    if provider in ("vecops", "oneDNN", "ACL"):
        return "native_fused"
    if provider == "KUPL-MMA" and operation == "bias":
        return "native_accumulate"
    return "separate"


def key(row: dict[str, str]) -> tuple[str, ...]:
    return tuple(row[field] for field in (
        "host", "library", "case", "op", "extent"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--catalog", required=True, type=Path,
                        help="full-catalog normalized CSV")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    rows = [row for path in args.inputs for row in read_rows(path)]
    catalog_rows = read_rows(args.catalog)
    semantics: dict[tuple[str, str], dict[str, str]] = {}
    for row in catalog_rows:
        semantics.setdefault((row["case"], row["op"]), row)

    measured = {key(row): row for row in rows}
    host_metadata: dict[str, dict[str, str]] = {}
    versions: dict[str, str] = {}
    for row in rows:
        host_metadata.setdefault(row["host"], row)
        if row["version"]:
            versions.setdefault(row["library"], row["version"])

    completed: list[dict[str, str]] = []
    for host, host_row in sorted(host_metadata.items()):
        for semantic in semantics.values():
            for provider in PROVIDERS:
                extents = ("AllConst", "AllDynamic") \
                    if provider == "vecops" and semantic["model"] == "general" \
                    else ("AllConst", "TokenDynamic") \
                    if provider == "vecops" else ("NA",)
                for extent in extents:
                    measured_row = measured.get((
                        host, provider, semantic["case"], semantic["op"], extent))
                    if measured_row is not None:
                        completed.append(measured_row)
                        continue
                    row = {field: "" for field in FIELDS}
                    for field in (
                            "arch", "cpu", "compiler", "backend_isa"):
                        row[field] = host_row[field]
                    for field in (
                            "case", "model", "subgraph", "shape", "batch",
                            "tokens", "chunk", "op", "dtype", "orientation"):
                        row[field] = semantic[field]
                    row.update({
                        "host": host,
                        "library": provider,
                        "version": versions.get(provider, ""),
                        "extent": extent,
                        "phase": PHASE[provider],
                        "epilogue": epilogue(provider, semantic["op"]),
                        "tuning": "default",
                        "mc": "auto", "nc": "auto", "kc": "auto",
                    })
                    if host == "local-x86" and provider in ("ACL", "KUPL-MMA"):
                        row["status"] = "unsupported_arch"
                        row["error"] = f"{provider} adapter is ARM-only"
                    else:
                        row["status"] = "not_measured"
                    completed.append(row)

    completed.sort(key=lambda row: tuple(row[field] for field in (
        "host", "case", "op", "phase", "library", "extent", "tuning")))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(completed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
