#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Compare one benchmark cycle with peers and a previous vecops snapshot."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from pathlib import Path


def read_rows(paths: list[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        with path.open(encoding="utf-8", newline="") as stream:
            rows.extend(csv.DictReader(stream))
    return [row for row in rows if row.get("status") == "ok" and row.get("median_us")]


def write_rows(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def extent_class(extent: str) -> str:
    return "Const" if extent == "AllConst" else "Dynamic"


def group(row: dict[str, str]) -> str:
    if row["model"] == "general":
        return "General"
    if row["model"] == "llm":
        return "LLM"
    if row["subgraph"] in ("gsa_projection", "pair_bias_projection"):
        return "AF3Projection"
    if row["subgraph"] in ("gsa_qk", "gsa_pv"):
        return "AF3Attention"
    return "AF3Triangle"


def phase_aligned(vecops_phase: str, peer_phase: str) -> bool:
    if vecops_phase == peer_phase:
        return True
    return vecops_phase == "prepared_b_execute" and peer_phase == "prepared_execute"


def geomean(values: list[float]) -> float | str:
    return math.exp(statistics.fmean(math.log(value) for value in values)) \
        if values else ""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--current", nargs="+", required=True, type=Path)
    parser.add_argument("--peers", nargs="*", default=[], type=Path,
                        help="fallback peer rows, normally the last full CSV")
    parser.add_argument("--baseline", type=Path,
                        help="previous vecops snapshot for regression matching")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--slower-than", type=float, default=1.0,
                        help="emit peer rows faster than vecops by this factor")
    args = parser.parse_args()

    current = read_rows(args.current)
    fallback = read_rows(args.peers)
    # Current rows win over the fallback snapshot for an identical result key.
    result: dict[tuple[str, str, str, str, str, str], dict[str, str]] = {}
    for row in [*fallback, *current]:
        result[(row["host"], row["library"], row["case"], row["op"],
                row["extent"], row["phase"])] = row
    rows = list(result.values())
    vecops = [row for row in current if row["library"] == "vecops"]

    peers: dict[tuple[str, str, str], list[dict[str, str]]] = {}
    for row in rows:
        if row["library"] != "vecops":
            peers.setdefault((row["host"], row["case"], row["op"]), []).append(row)

    provider_columns = {
        "OpenBLAS": "openblas", "oneDNN": "onednn", "LIBXSMM": "libxsmm",
        "ACL": "acl", "KUPL-MMA": "kupl",
    }
    comparison_fields = [
        "host", "extent", "group", "case", "shape", "op",
        "vecops_phase", "vecops_us",
    ]
    for column in provider_columns.values():
        comparison_fields += [
            f"{column}_phase", f"{column}_us",
            f"vecops_speedup_vs_{column}", f"phase_aligned_{column}",
        ]
    comparison_fields += [
        "best_aligned_provider", "best_aligned_peer_us",
        "vecops_speedup_vs_best_aligned", "best_any_provider",
        "best_any_peer_us", "vecops_speedup_vs_best_any",
    ]
    comparison: list[dict[str, object]] = []
    for row in vecops:
        vec_time = float(row["median_us"])
        output_row: dict[str, object] = {
            "host": row["host"], "extent": row["extent"],
            "group": group(row), "case": row["case"],
            "shape": row["shape"], "op": row["op"],
            "vecops_phase": row["phase"], "vecops_us": vec_time,
        }
        candidates = peers.get((row["host"], row["case"], row["op"]), [])
        by_provider: dict[str, dict[str, str]] = {}
        for peer in candidates:
            current_best = by_provider.get(peer["library"])
            if current_best is None or float(peer["median_us"]) < float(current_best["median_us"]):
                by_provider[peer["library"]] = peer
        for provider, column in provider_columns.items():
            peer = by_provider.get(provider)
            if peer is None:
                output_row.update({
                    f"{column}_phase": "", f"{column}_us": "",
                    f"vecops_speedup_vs_{column}": "",
                    f"phase_aligned_{column}": "",
                })
                continue
            peer_time = float(peer["median_us"])
            output_row.update({
                f"{column}_phase": peer["phase"],
                f"{column}_us": peer_time,
                # > 1 means vecops is faster, < 1 means the peer is faster.
                f"vecops_speedup_vs_{column}": peer_time / vec_time,
                f"phase_aligned_{column}": int(
                    phase_aligned(row["phase"], peer["phase"])),
            })
        best_any = min(by_provider.values(), key=lambda peer: float(peer["median_us"]),
                       default=None)
        aligned = [peer for peer in by_provider.values()
                   if phase_aligned(row["phase"], peer["phase"])]
        best_aligned = min(aligned, key=lambda peer: float(peer["median_us"]),
                           default=None)
        for prefix, peer in (("best_aligned", best_aligned), ("best_any", best_any)):
            if peer is None:
                output_row[f"{prefix}_provider"] = ""
                output_row[f"{prefix}_peer_us"] = ""
                output_row[f"vecops_speedup_vs_{prefix}"] = ""
            else:
                peer_time = float(peer["median_us"])
                output_row[f"{prefix}_provider"] = peer["library"]
                output_row[f"{prefix}_peer_us"] = peer_time
                output_row[f"vecops_speedup_vs_{prefix}"] = peer_time / vec_time
        comparison.append(output_row)
    comparison.sort(key=lambda row: (
        str(row["host"]), str(row["case"]), str(row["op"]), str(row["extent"])))
    write_rows(args.output_dir / "vecops-comparison.csv", comparison_fields, comparison)

    slower_fields = [
        "host", "extent", "group", "case", "shape", "op",
        "vecops_phase", "provider", "peer_phase", "phase_aligned",
        "vecops_us", "peer_us", "peer_faster_x",
    ]
    slower: list[dict[str, object]] = []
    for row in vecops:
        vec_time = float(row["median_us"])
        for peer in peers.get((row["host"], row["case"], row["op"]), []):
            ratio = vec_time / float(peer["median_us"])
            if ratio <= args.slower_than:
                continue
            slower.append({
                "host": row["host"], "extent": row["extent"],
                "group": group(row), "case": row["case"],
                "shape": row["shape"], "op": row["op"],
                "vecops_phase": row["phase"], "provider": peer["library"],
                "peer_phase": peer["phase"],
                "phase_aligned": int(phase_aligned(row["phase"], peer["phase"])),
                "vecops_us": vec_time, "peer_us": float(peer["median_us"]),
                "peer_faster_x": ratio,
            })
    slower.sort(key=lambda row: (-float(row["peer_faster_x"]), str(row["host"]),
                                 str(row["case"]), str(row["op"])))
    write_rows(args.output_dir / "vecops-slower.csv", slower_fields, slower)

    regression_fields = [
        "host", "extent", "group", "case", "shape", "op", "phase",
        "baseline_us", "current_us", "baseline_over_current",
        "change_percent",
    ]
    regressions: list[dict[str, object]] = []
    if args.baseline:
        baseline = {
            (row["host"], row["case"], row["op"], row["extent"], row["phase"]): row
            for row in read_rows([args.baseline]) if row["library"] == "vecops"
        }
        for row in vecops:
            old = baseline.get((row["host"], row["case"], row["op"],
                                row["extent"], row["phase"]))
            if old is None:
                continue
            old_us = float(old["median_us"])
            new_us = float(row["median_us"])
            regressions.append({
                "host": row["host"], "extent": row["extent"],
                "group": group(row), "case": row["case"],
                "shape": row["shape"], "op": row["op"], "phase": row["phase"],
                "baseline_us": old_us, "current_us": new_us,
                "baseline_over_current": old_us / new_us,
                "change_percent": (new_us / old_us - 1.0) * 100.0,
            })
        regressions.sort(key=lambda row: float(row["change_percent"]), reverse=True)
    write_rows(args.output_dir / "vecops-regressions.csv", regression_fields, regressions)

    summary_fields = [
        "host", "split", "group", "extent", "samples",
        "gflops_geomean", "median_us_geomean",
        "baseline_speedup_geomean", "aligned_peer_speedup_geomean",
        "aligned_peer_samples",
    ]
    summary: list[dict[str, object]] = []
    predicates = [("overall", "All", lambda _: True)]
    predicates += [("case_group", name, lambda row, wanted=name: group(row) == wanted)
                   for name in ("General", "LLM", "AF3Projection",
                                "AF3Attention", "AF3Triangle")]
    predicates += [("operation", name,
                    lambda row, wanted=name: row["op"] == wanted)
                   for name in ("gemm", "gemm_add", "bias", "bias_relu", "bias_silu")]
    regression_speedups = {
        (row["host"], row["case"], row["op"], row["extent"], row["phase"]):
            float(row["baseline_over_current"])
        for row in regressions
    }
    # Use every phase-aligned comparison, not only the subset where vecops is
    # slower. Keep the same direction as the detailed comparison columns:
    # peer_us / vecops_us, so values above one mean vecops is faster.
    aligned_peer: dict[tuple[str, str, str, str], float] = {}
    for row in comparison:
        value = row.get("vecops_speedup_vs_best_aligned", "")
        if value == "":
            continue
        key = (str(row["host"]), str(row["case"]), str(row["op"]),
               str(row["extent"]))
        aligned_peer[key] = float(value)
    for host in sorted({row["host"] for row in vecops}):
        for extent in ("Dynamic", "Const"):
            host_rows = [row for row in vecops
                         if row["host"] == host and extent_class(row["extent"]) == extent]
            for split, name, predicate in predicates:
                selected = [row for row in host_rows if predicate(row)]
                if not selected:
                    continue
                base_values = [regression_speedups[key] for row in selected
                               if (key := (row["host"], row["case"], row["op"],
                                           row["extent"], row["phase"]))
                               in regression_speedups]
                peer_values = [aligned_peer[key] for row in selected
                               if (key := (row["host"], row["case"], row["op"],
                                           row["extent"])) in aligned_peer]
                summary.append({
                    "host": host, "split": split, "group": name,
                    "extent": extent, "samples": len(selected),
                    "gflops_geomean": geomean([
                        float(row["logical_gflops"]) for row in selected]),
                    "median_us_geomean": geomean([
                        float(row["median_us"]) for row in selected]),
                    "baseline_speedup_geomean": geomean(base_values),
                    "aligned_peer_speedup_geomean": geomean(peer_values),
                    "aligned_peer_samples": len(peer_values),
                })
    write_rows(args.output_dir / "vecops-summary.csv", summary_fields, summary)
    print(f"wrote {len(comparison)} comparison rows, {len(slower)} slower rows, "
          f"{len(regressions)} matched regressions, and {len(summary)} summary rows "
          f"to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
