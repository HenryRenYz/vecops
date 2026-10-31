# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Predeclare paired socket-1 placement comparisons for 920f-4.

Die 0/A is NUMA 8-11, die 1/B is NUMA 12-15. HBM nodes are CPU node + 16.
HBM huge-page cases use larger arrays and therefore are not isolated page-size
comparisons. Local controls are rerun with the same binary and actual CPU map.
"""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--cores', default='16,37')
    args = p.parse_args()
    cases = []
    for die, execution in [('a', list(range(8, 12))), ('b', list(range(12, 16)))]:
        for memory, pages, size in [('ddr', 'base', '4MiB'), ('hbm', 'base', '4MiB'),
                                     ('hbm', 'huge', '16MiB')]:
            targets = [n + (16 if memory == 'hbm' else 0) for n in execution]
            for n in map(int, args.cores.split(',')):
                for access in ('temporal', 'stream'):
                    for op in ('read', 'write', 'copy', 'triad'):
                        modes = ['local', 'interleave']
                        if memory == 'hbm' and pages == 'base':
                            modes.append('ring')
                        for placement in modes:
                            chosen = targets[1:] + targets[:1] if placement == 'ring' else targets
                            case = dict(name=f'interleave_die_{die}_{memory}_{pages}_{placement}',
                                        family='interleave', targets=chosen, execution=execution,
                                        threads_per_group=n, operation=op, access=access,
                                        kernel='vec4', size=size, pages=pages, command='bandwidth',
                                        interleave=placement == 'interleave', placement=placement,
                                        die=die, memory_kind=memory)
                            case['id'] = hashlib.sha256(json.dumps(case, sort_keys=True).encode()).hexdigest()[:14]
                            cases.append(case)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(schema=1, cases=cases), indent=2))
    print(f'Predeclared {len(cases)} cases in {args.output}')


if __name__ == '__main__':
    main()
