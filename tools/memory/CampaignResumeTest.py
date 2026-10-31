# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Regression: resuming a subset must retain all completed manifest entries.

Exercise the runner's real CLI/output path with completed cases. CPU-affinity
and health probes are mocked; no benchmark or remote command is launched.
"""
import argparse
import contextlib
import io
import json
from pathlib import Path
import sys
from unittest.mock import patch

import RunSocketCampaign as campaign


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    binary = root / 'unused-binary'
    cases = [dict(id=f'completed-{i}', name=f'case-{i}', family='interleave',
                  targets=[28], execution=[12], threads_per_group=16,
                  operation='read', access='stream', pages='base', size='4MiB',
                  command='bandwidth', kernel='vec4', interleave=False) for i in range(2)]
    prior = dict(binary=str(binary), samples=5, seconds=.5, excluded_cpus=[306],
                 cases=cases)  # Legacy manifests omitted the empty default mask.
    (root / 'manifest.json').write_text(json.dumps(prior))
    before_health = json.dumps(dict(retained_original_health=True))
    (root / 'before.json').write_text(before_health)
    (root / 'states.jsonl').write_text(''.join(json.dumps(dict(id=c['id'])) + '\n' for c in cases))
    subset = root / 'subset.json'
    subset.write_text(json.dumps(dict(cases=cases[:1])))
    command = ['RunSocketCampaign.py', '--binary', str(binary), '--output', str(root),
               '--replay-manifest', str(subset), '--samples', '5', '--seconds', '.5',
               '--exclude-cpus', '306']
    capture = io.StringIO()
    with patch.object(sys, 'argv', command), patch.object(campaign.os, 'sched_setaffinity'), \
            patch.object(campaign, 'snapshot', return_value={}), \
            patch.object(campaign.subprocess, 'run') as launch, contextlib.redirect_stdout(capture):
        campaign.main()
        launch.assert_not_called()
    retained = json.loads((root / 'manifest.json').read_text())
    assert {c['id'] for c in retained['cases']} == {c['id'] for c in cases}
    assert 'CAMPAIGN_COMPLETE' in capture.getvalue()
    assert (root / 'before.json').read_text() == before_health
    before = (root / 'manifest.json').read_bytes()
    changed = list(command)
    changed[changed.index('--seconds') + 1] = '1'
    for argv, replacement in [(changed, None), (command, dict(cases=[dict(cases[0], size='8MiB')]))]:
        if replacement:
            subset.write_text(json.dumps(replacement))
        with patch.object(sys, 'argv', argv), patch.object(campaign.os, 'sched_setaffinity'):
            try:
                campaign.main()
            except ValueError:
                pass
            else:
                raise AssertionError('Changed resume conditions must be rejected')
        assert (root / 'manifest.json').read_bytes() == before
    print('PASS subset resume retains completed cases, skips work, and rejects changed conditions')


if __name__ == '__main__':
    main()
