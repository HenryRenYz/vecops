#!/usr/bin/env python3
"""Aggregate vecops throughput and provider-relative speedups."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from pathlib import Path


PROVIDERS = ("OpenBLAS", "oneDNN", "LIBXSMM", "ACL", "KUPL-MMA")
FIELDS = [
    "host", "split", "group", "extent", "samples", "gflops_geomean",
    "gflops_median",
    *(item for provider in PROVIDERS for item in (
        f"speedup_vs_{provider.lower().replace('-', '_')}_geomean",
        f"speedup_vs_{provider.lower().replace('-', '_')}_samples")),
]


def geomean(values: list[float]) -> float | str:
    return math.exp(statistics.fmean(math.log(value) for value in values)) \
        if values else ""


def case_group(row: dict[str, str]) -> str:
    if row["model"] == "general":
        return "General"
    if row["model"] == "llm":
        return "LLM"
    if row["subgraph"] in ("gsa_projection", "pair_bias_projection"):
        return "AF3Projection"
    if row["subgraph"] in ("gsa_qk", "gsa_pv"):
        return "AF3Attention"
    return "AF3Triangle"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    with args.input.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    ok = [row for row in rows if row["status"] == "ok" and row["median_us"]]
    providers = {
        (row["host"], row["case"], row["op"], row["library"]):
            float(row["median_us"])
        for row in ok if row["library"] != "vecops"
    }
    vecops = [row for row in ok if row["library"] == "vecops"]

    dimensions: list[tuple[str, str, object]] = [("overall", "All", lambda row: True)]
    for group in ("General", "LLM", "AF3Projection", "AF3Attention", "AF3Triangle"):
        dimensions.append(("case_group", group,
                           lambda row, wanted=group: case_group(row) == wanted))
    for operation in ("gemm", "gemm_add", "bias", "bias_relu", "bias_silu"):
        dimensions.append(("epilogue", operation,
                           lambda row, wanted=operation: row["op"] == wanted))

    output_rows: list[dict[str, object]] = []
    for host in sorted({row["host"] for row in vecops}):
        host_rows = [row for row in vecops if row["host"] == host]
        for extent in ("Dynamic", "Const"):
            extent_rows = [
                row for row in host_rows
                if (row["extent"] == "AllConst") == (extent == "Const")]
            for split, group, predicate in dimensions:
                selected = [row for row in extent_rows if predicate(row)]
                if not selected:
                    continue
                result: dict[str, object] = {field: "" for field in FIELDS}
                throughputs = [float(row["logical_gflops"]) for row in selected]
                result.update({
                    "host": host, "split": split, "group": group,
                    "extent": extent, "samples": len(selected),
                    "gflops_geomean": geomean(throughputs),
                    "gflops_median": statistics.median(throughputs),
                })
                for provider in PROVIDERS:
                    ratios = []
                    for row in selected:
                        provider_time = providers.get((
                            host, row["case"], row["op"], provider))
                        if provider_time is not None:
                            ratios.append(provider_time / float(row["median_us"]))
                    stem = provider.lower().replace("-", "_")
                    result[f"speedup_vs_{stem}_geomean"] = geomean(ratios)
                    result[f"speedup_vs_{stem}_samples"] = len(ratios)
                output_rows.append(result)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(output_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
