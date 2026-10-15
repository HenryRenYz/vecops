#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Freeze nested, coverage-directed Matmul performance cohorts from admitted runs.

Selection uses metadata, explicit historical sentinels and compile-group reuse,
not a random sample or an estimate of full-catalog application performance.
An unselected regression can always escape a subset. Keep full runs for release.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

TIERS = (("smoke", 64, 16), ("standard", 256, 64), ("extended", 1024, 256))


def read(path):
    return json.loads(Path(path).read_text())


def write(path, document):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n")
    path.with_name(path.name + ".license").write_text(
        "SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors\n"
        "SPDX-License-Identifier: MIT\n")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def geometry(m, n, k):
    return ("gemv" if min(m, n) == 1 else "tiny" if m*n <= 64 else
            "skinny" if min(m, n) < 16 else "large" if m*n*k >= 2**26 else "matrix")


def features(row):
    """Observable catalog dimensions only; no inferred alias/stride guarantees."""
    tags = {}
    def add(name, value, weight=8):
        if value is not None:
            tags[f"{name}={json.dumps(value, sort_keys=True)}"] = weight
    for key in ("catalog", "suite", "meta", "metadata", "level", "packing", "native_a",
                "native_b", "accumulator", "memory_a", "memory_b", "memory_c", "memory",
                "native", "input_pipeline_a", "input_pipeline_b", "output_pipeline", "pipeline",
                "reference_family", "reference_policy", "reference_compensated_pack", "orientation",
                "broadcast", "batch_strategy", "selector", "side", "compensation_mode", "strategy",
                "calls_per_pack", "cohort"):
        if key == 'metadata' and 'meta' in row:
            continue
        add(key, row.get(key))
    m, n, k = row.get('shape', (row.get('m', 1), row.get('n', row.get('spatial', 1)), row.get('k', 1)))
    m, n, k = int(m), int(n), int(k)
    geom = geometry(m, n, k)
    add('geometry', geom, 14)
    add('shape', (m, n, k), 1)
    batch = int(row.get('batch') or 1)
    add('batch', batch, 5)
    for axis, value in zip('mnk', (m, n, k)):
        add(axis + '_size_class', 'zero' if value == 0 else 'one' if value == 1 else int(math.log2(value)), 3)
        for alignment in (4, 16, 32, 64):
            residue = value % alignment
            add(f'{axis}_tail{alignment}', 'aligned' if residue == 0 else 'one' if residue == 1 else
                'last' if residue == alignment-1 else 'interior', 4)
    meta = row.get('meta', row.get('metadata'))
    dtype = (row.get('native_a', row.get('native')), row.get('native_b', row.get('native')))
    packing = row.get('packing', row.get('strategy', row.get('side')))
    add('pair/meta_dtype', (meta, dtype), 5)
    add('pair/geometry_packing', (geom, packing), 5)
    add('pair/dtype_packing', (dtype, packing), 5)
    add('pair/geometry_epilogue', (geom, row.get('output_pipeline', row.get('pipeline'))), 5)
    add('triple/meta_dtype_packing', (meta, dtype, packing), 2)
    add('case_name', row.get('name'), 1)
    add('combination', (meta, dtype, packing, row.get('input_pipeline_a'),
                       row.get('input_pipeline_b'), row.get('output_pipeline', row.get('pipeline')),
                       batch > 1, row.get('broadcast'), row.get('batch_strategy'),
                       row.get('reference_family', row.get('selector')), row.get('reference_policy')), .7)
    for risk in row.get('_risks', ()):
        add('risk', risk, 30)
    return tags


def nested_select(rows, budgets, *, key, groups=None, pins=None, stress=()):
    """Deterministic greedy coverage; prefixes are stable and strictly nested."""
    by_id = {r[key]: r for r in rows}
    if len(by_id) != len(rows):
        raise ValueError('duplicate catalog ID')
    groups = groups or {identity: identity for identity in by_id}
    pins = pins or {}
    unknown = set(stress) | set().union(*(set(v) for v in pins.values()))
    if unknown - by_id.keys():
        raise ValueError('sentinel outside admitted catalog')
    tagged = {identity: features(row) for identity, row in by_id.items()}
    selected, selected_set, covered, used_groups, outputs = [], set(), set(), set(), {}
    def take(identity):
        if identity not in selected_set:
            selected.append(identity); selected_set.add(identity)
            covered.update(tagged[identity]); used_groups.add(groups[identity])
    for tier, budget in budgets:
        limit = min(budget, len(rows))
        if tier in ('smoke', 'standard') and set(pins.get(tier, ())).intersection(stress):
            raise ValueError('compile stress sentinel belongs in extended tier')
        for identity in pins.get(tier, ()):
            take(identity)
        if len(selected) > limit:
            raise ValueError('sentinels exceed tier budget')
        while len(selected) < limit:
            candidates = by_id.keys() - selected_set
            if tier in ('smoke', 'standard'):
                candidates -= set(stress)
            if not candidates:
                raise ValueError('not enough non-stress rows for tier')
            def score(identity):
                tags = tagged[identity]
                gain = sum(weight for tag, weight in tags.items() if tag not in covered)
                # Mild reuse preference; do not sacrifice rare metadata for reuse.
                reuse = groups[identity] in used_groups
                return (gain * (1.08 if reuse else 1.0), reuse, identity)
            take(max(candidates, key=score))
        outputs[tier] = list(selected)
    outputs['full'] = list(selected) + sorted(by_id.keys() - selected_set)
    return outputs


def coverage(rows, chosen, key, groups=None):
    all_tags = set().union(*(features(r) for r in rows))
    chosen_set = set(chosen)
    selected = [r for r in rows if r[key] in chosen_set]
    tags = set().union(*(features(r) for r in selected))
    return {'count': len(selected), 'metadata_features': len(tags), 'available_features': len(all_tags),
            'missing_features': sorted(all_tags-tags),
            'compile_groups': len({groups[i] for i in chosen}) if groups else None,
            'by_catalog': dict(Counter(r.get('catalog', r.get('suite')) for r in selected)),
            'by_meta': dict(Counter(r.get('meta', r.get('metadata')) for r in selected)),
            'risk_tags': sorted({risk for r in selected for risk in r.get('_risks', ())})}


def generate(args):
    plan = read(args.plan)
    arch = plan['architecture']
    rows = plan['executions']
    ids = {r['execution_id'] for r in rows}
    if ids != set(plan['selected_execution_ids']) or len(ids) != len(rows):
        raise ValueError('plan is not an exact unique execution scope')
    ledger = read(args.plan.with_name('ledger.json'))
    if (len(ledger['rows']) != len(ids) or
        any(s['status'] not in ('passed', 'not_applicable') for r in ledger['rows'] for s in r['stages'].values())):
        raise ValueError('baseline contains incomplete or failed stages')
    passed = {r['execution_id'] for r in ledger['rows'] if r['stages']['compute']['status'] == 'passed'}
    if passed != ids:
        raise ValueError('baseline must have every selected compute admitted')
    packing_ids = {r['execution_id'] for r in ledger['rows']
                   if r['stages'].get('packing', {}).get('status') == 'passed'}
    summary = read(args.supplemental)
    supplemental = [s['registration'] for s in summary['states'] if s['timed']]
    if any(not s['timed'] and s['registration'].get('dsl_equivalent') for s in summary['states']):
        raise ValueError('comparable supplemental baseline is incomplete')
    if any(not s['built'] or not s['validated'] for s in summary['states'] if s['timed']):
        raise ValueError('supplemental timing without correctness admission')
    controls = read(args.sentinels) if args.sentinels else {}
    if controls and controls['architecture'] != arch:
        raise ValueError('sentinel architecture mismatch')
    risks = controls.get('risks', {})
    rows = [{**r, '_risks': risks.get(r['execution_id'], [])} for r in rows]
    groups = {identity:g['id'] for g in plan['groups'] for identity in g['execution_ids']}
    if groups.keys() != ids or sum(len(g['execution_ids']) for g in plan['groups']) != len(ids):
        raise ValueError('compilation group inventory differs from execution scope')
    main = nested_select(rows, [(name,m) for name,m,s in TIERS], key='execution_id', groups=groups,
                         pins=controls.get('main_pins'), stress=controls.get('compile_stress_ids', ()))
    # Even the smallest companion must contain actual packing, reuse and Legacy.
    supp_pins = controls.get('supplemental_pins', {})
    supp_pins.setdefault('smoke', [])
    for category in sorted({r['suite'] for r in supplemental}):
        candidates = [r['registration_id'] for r in supplemental if r['suite'] == category]
        if not set(candidates).intersection(supp_pins['smoke']):
            supp_pins['smoke'].append(min(candidates))
    supp = nested_select(supplemental, [(name,s) for name,m,s in TIERS], key='registration_id', pins=supp_pins)
    evidence = {'main_plan':str(args.plan.resolve()), 'main_plan_sha256':digest(args.plan),
                'ledger_sha256':digest(args.plan.with_name('ledger.json')),
                'supplemental_summary':str(args.supplemental.resolve()), 'supplemental_sha256':digest(args.supplemental),
                'sentinels_sha256':digest(args.sentinels) if args.sentinels else None}
    report = {'architecture':arch, 'evidence':evidence, 'tiers':{},
              'method':'Fixed nested weighted metadata coverage, explicit sentinels, mild compile-group reuse. Not a random or unbiased sample. No guarantee for unselected workloads.',
              'excluded_api_gaps':[s['registration'] for s in summary['states'] if not s['timed']
                                   and s['registration'].get('comparison_status') == 'unsupported_public_api'],
              'excluded_other_registrations':[s['registration'] for s in summary['states'] if not s['timed']
                                   and s['registration'].get('comparison_status') != 'unsupported_public_api'],
              'compile_stress_ids':controls.get('compile_stress_ids', [])}
    for tier in (*[t[0] for t in TIERS], 'full'):
        base = args.output / arch / tier
        write(base/'main.json', {'architecture':arch, 'execution_ids':main[tier], 'tier':tier, 'evidence':evidence})
        packing = [identity for identity in main[tier] if identity in packing_ids]
        write(base/'main-packing.json', {'architecture':arch, 'execution_ids':packing, 'tier':tier, 'evidence':evidence})
        picked_supp = [r for r in supplemental if r['registration_id'] in set(supp[tier])]
        for kind in ('lifecycle','packing'):
            chosen = [r['registration_id'] for r in picked_supp if (r['suite']=='MatmulPack') == (kind=='packing')]
            write(base/f'{kind}.json', {'architecture':arch, 'registration_ids':chosen, 'tier':tier, 'evidence':evidence})
        picked_main = [r for r in rows if r['execution_id'] in set(main[tier])]
        write(base/'workloads.json', {'architecture':arch, 'main':picked_main, 'supplemental':picked_supp})
        report['tiers'][tier] = {'main':coverage(rows,main[tier],'execution_id',groups),
                               'supplemental':coverage(supplemental,supp[tier],'registration_id'),
                               'additional_main_packing_stages':len(packing),
                               'total_registrations':len(main[tier])+len(supp[tier])}
    write(args.output/arch/'coverage.json',report)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--supplemental',type=Path,required=True)
    p.add_argument('--sentinels',type=Path)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    print(json.dumps(generate(args)['tiers'],indent=2))


if __name__=='__main__':
    main()
