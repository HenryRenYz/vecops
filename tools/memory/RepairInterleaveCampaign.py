# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Complete undersampled interleave comparisons without choosing by speed.

Repeat short-window cases first. If tails persist, rerun every placement in
the affected comparison with an identical longer window. The unchanged 0.1%
skew limit determines the window, never the measured bandwidth. Prior runs
remain retained, and only identical conditions are pooled by the summarizer.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from SummarizeSocketCampaign import summarize


def idle_check(excluded):
    def snapshot():
        return {int(v[0][3:]): list(map(int, v[1:])) for line in Path('/proc/stat').read_text().splitlines()
                if (v := line.split()) and v[0].startswith('cpu') and v[0][3:].isdigit()}
    a = snapshot()
    # Short-lived kernel/SSH activity can dominate a two-second snapshot.
    # Keep the same occupancy threshold but observe a sustained ten-second mean.
    time.sleep(10)
    b = snapshot()
    busy = []
    for cpu in set(range(304, 608)) - excluded:
        delta = [y - x for x, y in zip(a[cpu], b[cpu])]
        usage = 1 - (delta[3] + delta[4]) / max(1, sum(delta))
        if usage > .25:
            busy.append((cpu, round(usage * 100)))
    if busy:
        raise RuntimeError(f'Defer heavy rerun: socket-1 CPUs occupied: {busy}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--binary', type=Path, required=True)
    p.add_argument('--wait-log', type=Path, required=True)
    p.add_argument('--exclude-cpus', default='306')
    p.add_argument('--control-cpu', type=int, default=345,
                   help='Idle socket-1 core outside every worker map; pins analysis only')
    args = p.parse_args()
    root = args.root.resolve()
    script = Path(__file__).with_name('RunSocketCampaign.py')
    summary_script = Path(__file__).with_name('SummarizeSocketCampaign.py')
    excluded = set(map(int, args.exclude_cpus.split(','))) if args.exclude_cpus else set()
    manifest = json.loads((root / 'input.json').read_text())
    cases = manifest['cases']
    for c in cases:
        for node in c['execution']:
            cpus=[node*38+offset for offset in list(range(4,38))+list(range(4))
                  if node*38+offset not in excluded][:c['threads_per_group']]
            if args.control_cpu in cpus:
                raise ValueError('Control CPU overlaps a benchmark worker')
    if args.control_cpu in excluded or not 304 <= args.control_cpu < 608:
        raise ValueError('Control CPU must be an available socket-1 core')
    os.sched_setaffinity(0, {args.control_cpu})
    while 'CAMPAIGN_COMPLETE' not in args.wait_log.read_text():
        time.sleep(5)
    by_id = {c['id']: c for c in cases}

    def rows():
        return [r for r in summarize(root)['selected_summaries'] if r['scope'] == 'total']

    def run(name, selected, seconds=.5):
        plan = root / (name + '-input.json')
        if plan.exists():
            # Resume the original prescribed plan, including all its controls.
            selected=json.loads(plan.read_text())['cases']
        elif selected:
            plan.write_text(json.dumps(dict(schema=1, cases=selected), indent=2))
        if not selected:
            return
        idle_check(excluded | {args.control_cpu})
        print(json.dumps(dict(phase=name, cases=len(selected), seconds=seconds,
                              case_seconds={c['id']: c.get('seconds', seconds) for c in selected})), flush=True)
        subprocess.run([sys.executable, '-u', str(script), '--binary', str(args.binary.resolve()),
                        '--output', str(root / name), '--replay-manifest', str(plan),
                        '--samples', '5', '--seconds', str(seconds), '--retries', '0',
                        '--exclude-cpus', args.exclude_cpus, '--default-memory-bind', '8-15,24-31'], check=True)

    selected = [by_id[r['case_id']] for r in rows() if r['samples_usable'] < 3]
    run('repair-01-short', selected)
    current = rows()
    poor = [r for r in current if r['samples_usable'] < 3]
    conditions = ('die', 'memory_kind', 'pages', 'threads_per_group', 'operation', 'access')
    key = lambda c: tuple(c[k] for k in conditions)
    windows = {}
    for row in poor:
        # Conservative preset windows keep all placement variants comparable.
        worst_us = max(row['max_start_skew_us'], row['max_end_skew_us'])
        seconds = 2 if worst_us < 1500 else 5 if worst_us < 4000 else 20 if worst_us < 18000 else 40
        k = key(by_id[row['case_id']])
        windows[k] = max(windows.get(k, 0), seconds)
    long_cases = [dict(c, seconds=windows[key(c)]) for c in cases if key(c) in windows]
    run('repair-02-long', long_cases)
    # A final identical-condition repeat adds observations without changing the
    # latest prescribed window or selecting a faster result.
    latest = {c['id']: c for c in long_cases}
    remaining = [r for r in rows() if r['samples_usable'] < 3]
    run('repair-03-repeat', [latest.get(r['case_id'], by_id[r['case_id']]) for r in remaining])
    subprocess.run([sys.executable, str(summary_script), str(root)], check=True)
    remaining = [r for r in rows() if r['samples_usable'] < 3]
    print(json.dumps(dict(undersampled=[r['case_id'] for r in remaining])), flush=True)
    if remaining:
        raise RuntimeError('Some configurations remain undersampled; retain their quality flags')
    print('INTERLEAVE_REPAIR_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
