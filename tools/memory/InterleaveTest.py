# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Check real cyclic page placement, role pools and exported aggregation.

The benchmark verifies every page's exact node before and after timing. These
checks independently validate exported page counts, arithmetic and row sums.
All files remain in the explicit output directory; no system policy is changed.
"""
import argparse
import csv
import json
import math
from pathlib import Path
import subprocess


def main():
    p = argparse.ArgumentParser()
    p.add_argument('binary', type=Path)
    p.add_argument('--nodes', default='0')
    p.add_argument('--execution-nodes')
    p.add_argument('--destination-nodes')
    p.add_argument('--pages', default='base')
    p.add_argument('--cpu', default='4')
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    nodes = list(map(int, args.nodes.split(',')))
    execution = args.execution_nodes or args.nodes
    args.output_dir.mkdir(parents=True, exist_ok=True)
    common = [str(args.binary.resolve()), '--numa', args.nodes, '--exec-node', execution,
              '--cpu', args.cpu, '--seconds', '.03', '--samples', '2', '--warmup', '.01',
              '--retries', '0', '--max-skew', '.02']

    def counts(text):
        return {int(k): int(v) for k, v in (s.split(':') for s in text.split('|'))} if text else {}

    def run(name, extra, source, destination, enabled=True, command='bandwidth'):
        csv_path = args.output_dir / (name + '.csv')
        json_path = args.output_dir / (name + '.jsonl')
        cmd = [common[0], command, *common[1:], *(['--interleave'] if enabled else []),
               '--csv', str(csv_path), '--json', str(json_path), *extra]
        result = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=120)
        (args.output_dir / (name + '.log')).write_text(result.stdout)
        assert result.returncode in (0, 1), result.stdout
        rows = list(csv.DictReader(csv_path.open()))
        assert rows and rows == [json.loads(s) for s in json_path.read_text().splitlines()]
        for row in rows:
            assert row['schema_version'] == '2' and row['placement'] == 'all-pages-verified', row
            assert row['memory_policy'] == ('interleave' if enabled else 'bind')
            assert row['status'] in ('ok', 'interrupted', 'skew', 'page-fault'), row
            if not row['scope'].startswith('thread:'):
                continue
            op = row['operation']
            page = 2**21 if row['page_policy'] == 'huge' else int(row['base_page_bytes'])
            assert int(row['interleave_bytes']) == (page if enabled else 0)
            n = (int(row['per_thread_array_bytes']) + page - 1) // page
            for role, pool, arrays in [('source', source, 0 if op == 'write' else 2 if op == 'triad' else 1),
                                       ('destination', destination, 0 if op in ('read', 'chase') else 1)]:
                pool = sorted(set(pool)) if enabled else [int(row[role + '_node'])]
                expected = {node: arrays * (n // len(pool) + (i < n % len(pool)))
                            for i, node in enumerate(pool)} if arrays else {}
                assert counts(row[role + '_page_counts']) == expected, (row, expected)
                assert row[role + '_memory_nodes'] == (','.join(map(str, pool)) if arrays else ''), row
            duration = (int(row['end_ns']) - int(row['start_ns'])) / 1e9
            assert math.isclose(float(row['elapsed_seconds']), duration, rel_tol=1e-10)
        for total in (r for r in rows if r['scope'] == 'total' or r['scope'].startswith('group:')):
            participants = [r for r in rows if r['scope'].startswith('thread:') and r['kind'] == 'bandwidth'
                            and all(r[k] == total[k] for k in ('config', 'sample', 'attempt'))
                            and (total['scope'] == 'total' or r['group'] == total['group'])]
            for role in ('source', 'destination'):
                expected = {}
                for row in participants:
                    for node, n in counts(row[role + '_page_counts']).items():
                        expected[node] = expected.get(node, 0) + n
                assert counts(total[role + '_page_counts']) == expected, total
            assert int(total['read_bytes']) == sum(int(r['read_bytes']) for r in participants)
            assert int(total['write_bytes']) == sum(int(r['write_bytes']) for r in participants)
        print(f'PASS {name}: {len(rows)} rows')

    for pages in args.pages.split(','):
        # Incomplete cycles, one-page arrays, non-page-aligned lengths, full cycles.
        sizes = '2MiB,10MiB,16MiB' if pages == 'huge' else '256B,4KiB,28KiB,1MiB'
        extra = ['--pages', pages, '--size', sizes, '--op', 'read,write,copy,triad', '--kernel', 'vec4']
        run(pages + '-cyclic', extra, nodes, nodes)
        run(pages + '-bind-control', extra, nodes, nodes, enabled=False)
        run(pages + '-chase', ['--pages', pages, '--size', '16MiB'], nodes, [], command='latency')
        run(pages + '-loaded-chase', ['--pages', pages, '--size', '16MiB', '--loaded',
            '--latency-cpu', args.cpu, '--cpu', str(int(args.cpu) + 1), '--op', 'triad',
            '--kernel', 'vec4'], nodes, nodes, command='latency')
        repeated = ','.join([str(nodes[0])] * len(nodes))
        run(pages + '-duplicate-pool', [*extra, '--numa', repeated], [nodes[0]], [nodes[0]])
        if args.destination_nodes:
            dst = list(map(int, args.destination_nodes.split(',')))
            assert len(dst) == len(nodes)
            routes = ','.join(f'{a}:{b}' for a, b in zip(nodes, dst))
            run(pages + '-separate-roles', [*extra, '--numa', routes, '--op', 'copy,triad'], nodes, dst)
    for command, extra in [('bandwidth', ['--pages', 'auto']), ('pingpong', ['--peer-cpu', '5'])]:
        result = subprocess.run([common[0], command, *common[1:], '--interleave', '--dry-run', *extra],
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=10)
        assert result.returncode == 2, result.stdout.decode()
    print('PASS rejected ambiguous/unsupported policies')


if __name__ == '__main__':
    main()
