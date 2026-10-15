# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Continue one guarded supplemental Ninja batch without changing its inputs."""
import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--jobs', type=int, default=8)
    args = parser.parse_args()
    root, directory = args.root.resolve(), args.directory.resolve()
    sys.path.insert(0, str(root / 'benchmarks'))
    from dsl_snapshot_entry import select_snapshot
    select_snapshot(root)
    from dsl_matmul_artifact_ab import adapter_contract, template_contract
    from dsl_matmul_repair_suite import BatchInputAttestation
    from dsl_matmul_supplemental_pack import templates

    attestation = BatchInputAttestation.restore(directory)
    original = json.loads((directory / 'build-status.json').read_text())
    (directory / 'original-build-status.json').write_text(json.dumps(original, indent=2) + '\n')
    events = []
    for cache in sorted((directory / 'build').glob('kernel-batch-*/build/CMakeCache.txt')):
        text = cache.read_text()
        if not re.search(r'^CMAKE_GENERATOR:INTERNAL=Ninja$', text, re.MULTILINE):
            raise ValueError('Continuation requires the original configured Ninja graph')
        attestation.guard('pre_continue_build')
        command = ['cmake', '--build', str(cache.parent), '--target', 'vecops_batch_all',
                   '--config', 'Release', '--parallel', str(args.jobs), '--', '-k', '0']
        log = cache.parent / 'supplemental-keepgoing.log'
        with log.open('w') as output:
            result = subprocess.run(command, stdout=output, stderr=subprocess.STDOUT, check=False,
                                    cwd=root.parents[1], env=os.environ.copy())
        events.append({'command': command, 'returncode': result.returncode, 'log': str(log)})
    attestation.guard('post_build')
    records = []
    for adapter in (directory / 'build').glob('kernel-batch-*/tasks/*/input/vecops_adapter.cpp'):
        text = adapter.read_text()
        source = re.search(r'#include "([^"\n]+\.cpp)"', text)
        if source is None:
            continue
        spec = adapter.with_name('vecops_spec.h').read_text()
        captures = {name: int(value) for name, value in re.findall(
            r'inline constexpr auto (\w+) = ::vecops::meta::cint<(-?\d+)>;', spec)}
        captures.update(re.findall(r'using (\w+) = ::vecops::(\w+);', spec))
        binaries = list((adapter.parents[3] / 'artifacts' / adapter.parents[1].name).glob('*.so'))
        if len(binaries) == 1:
            records.append((Path(source.group(1)).resolve(), adapter_contract(text), captures,
                            str(binaries[0]), hashlib.sha256(binaries[0].read_bytes()).hexdigest()))
    manifest_path = directory / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    built, failures = [], []
    for artifact in manifest['artifacts']:
        missing = []
        for label, template in zip(('dsl', 'cpp'), templates(artifact['row'])):
            contract, captures = template_contract(template)
            matches = [record for record in records
                       if record[0] == Path(artifact[label + '_source']).resolve()
                       and json.dumps(record[1], sort_keys=True) == json.dumps(contract, sort_keys=True)
                       and all(record[2].get(key) == value for key, value in captures.items())]
            if len(matches) != 1:
                missing.append(label)
        if missing:
            failures.append({'registration_id': artifact['registration_id'], 'row': artifact['row'],
                             'phase': 'native_compilation', 'missing_roles': missing,
                             'error': 'Original compiler graph failed to produce an exact source/ABI/capture artifact'})
        else:
            built.append(artifact['registration_id'])
    manifest['failures'] = failures
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    checkpoint = directory / 'batch-input-attestation.json'
    status = {'admitted': True, 'registrations': built, 'failed_registrations': failures,
              'original_error': original.get('error'), 'continuation': events,
              'attestation_checkpoint': str(checkpoint),
              'attestation_checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest()}
    (directory / 'build-status.json').write_text(json.dumps(status, indent=2) + '\n')
    print(json.dumps({'available_pairs': len(built), 'failed': len(failures)}, indent=2))


if __name__ == '__main__':
    main()
