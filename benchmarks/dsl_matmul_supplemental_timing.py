# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Guarded parallel pre/post attestation around one quiet serial timing phase.

Pre/post workers are untimed. Measurement writes pending evidence only; a
complete postguard and unchanged exact input chain are required for admission.
"""
import argparse
import hashlib
import json
import math
import os
import sys
from contextlib import ExitStack
from pathlib import Path

from dsl_matmul_supplemental_validate import batch_lock, result_document

HARNESS_INPUTS = ('dsl_matmul_supplemental_timing.py', 'dsl_matmul_supplemental_artifacts.py',
    'dsl_matmul_supplemental_validate.py', 'dsl_matmul_supplemental_lifecycle.py',
    'dsl_matmul_supplemental_pack.py', 'dsl_matmul_supplemental_catalog.py',
    'dsl_matmul_source_catalog.py', 'dsl_matmul_artifact_ab.py', 'dsl_matmul_fusion_catalog.py',
    'dsl_timing.py', 'dsl_snapshot_entry.py', 'dsl_matmul_repair_suite.py',
    'dsl_benchmark_utils.py', 'dsl_transpose.py', 'dsl_matmul_catalog.py')
TIMING_FIELDS = ('median_ns', 'dsl_over_cpp', 'speedup', 'samples', 'rounds', 'cpu')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_inputs(inputs):
    for name, expected in inputs.items():
        if digest(name) != expected:
            raise ValueError(f'Timing input changed; private batch cannot be admitted: {name}')


def save(path, document):
    temporary = path.with_suffix(path.suffix + '.writing')
    temporary.write_text(json.dumps(document, indent=2) + '\n')
    temporary.replace(path)


def admit_pending(pre, pending, post_inputs):
    """Pure-data proof linkage; SDK/source truth is checked by the postguard."""
    if pending.get('admitted') or pending['pre_sha256'] != pre['record_sha256']:
        raise ValueError('Pending timing does not reference the original preguard')
    if pending['inputs'] != pre['inputs'] or post_inputs != pre['inputs']:
        raise ValueError('Timing input chain changed between pre, measurement and post')
    rows = pending['rows']
    ids = [row['registration']['registration_id'] for row in rows]
    if len(ids) != len(set(ids)) or set(ids) != set(pre['registrations']):
        raise ValueError('Pending timing rows do not exactly cover the preguard registrations')
    for row in rows:
        identity = row['registration']['registration_id']
        if row.get('artifacts') != pre['oracle_artifacts'][identity]:
            raise ValueError('Pending timing artifacts differ from the oracle proof')
        if row.get('error'):
            if row.get('validated') or row.get('timed') or any(name in row for name in TIMING_FIELDS):
                raise ValueError('Failed measurement cannot carry a valid timing ratio')
        elif not row.get('validated') or not row.get('timed') or not row.get('median_ns'):
            raise ValueError('Successful measurement requires a fresh oracle and timing result')
        elif (set(row['median_ns']) != {'dsl', 'cpp'}
              or any(not math.isfinite(value) or value <= 0 for value in row['median_ns'].values())):
            raise ValueError('Successful measurement requires finite positive DSL/CPP medians')
    return rows


def verify_timing_chains(document):
    chains = document.get('timing_chains', [document.get('timing_chain')])
    covered = set()
    for chain in chains:
        if not chain or digest(chain['pre']) != chain['pre_sha256'] or digest(chain['pending']) != chain['pending_sha256']:
            raise ValueError('Timing pre/pending record changed')
        pre = json.loads(Path(chain['pre']).read_text())
        pre['record_sha256'] = chain['pre_sha256']
        pending = json.loads(Path(chain['pending']).read_text())
        if chain['inputs'] != pre['inputs']:
            raise ValueError('Final timing chain differs from its original preguard inputs')
        check_inputs(chain['inputs'])
        rows = admit_pending(pre, pending, chain['inputs'])
        ids = {row['registration']['registration_id'] for row in rows}
        if covered.intersection(ids):
            raise ValueError('Final timing chains overlap registrations')
        covered.update(ids)
        actual = [row for row in document['rows'] if row['registration']['registration_id'] in ids]
        if actual != rows:
            raise ValueError('Final timing rows differ from the actual pending measurement')
    if {row['registration']['registration_id'] for row in document['rows']} != covered:
        raise ValueError('Final timing rows are not completely covered by pre/measurement/post chains')


def measure_rows(selected, originals, measure):
    results = []
    for row in selected:
        result = dict(originals[row['registration_id']])
        try:
            result.update(measure(row))
            result.update(validated=True, timed=True)
        except (AssertionError, RuntimeError, ValueError, TypeError) as error:
            result.update(validated=False, timed=False, phase='measurement_oracle_or_timing', error=repr(error))
            for name in TIMING_FIELDS:
                result.pop(name, None)
        results.append(result)
        print('PENDING' if result['timed'] else 'FAILED', row['registration_id'], flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--work-dir', type=Path, required=True)
    parser.add_argument('--phase', choices=('pre', 'measure', 'post'), required=True)
    parser.add_argument('--batch-index', type=int)
    parser.add_argument('--cpu', type=int, required=True)
    parser.add_argument('--rounds', type=int, default=12)
    parser.add_argument('--quiet-barrier', type=Path)
    args = parser.parse_args()
    if (args.phase == 'measure') != (args.batch_index is None):
        parser.error('measure is serial across all batches; pre/post require one batch-index')
    root, work = args.root.resolve(), args.work_dir.resolve()
    sys.path.insert(0, str(root / 'benchmarks'))
    from dsl_snapshot_entry import select_snapshot
    select_snapshot(root)
    os.chdir(root.parents[1])
    import numpy as np
    from dsl_matmul_repair_suite import BatchInputAttestation
    from dsl_matmul_supplemental_artifacts import artifact_inventory, bind_artifact
    from dsl_matmul_supplemental_lifecycle import lifecycle_arguments, scenario_for, templates
    from dsl_matmul_supplemental_pack import measure_pair, packing_arguments
    from dsl_matmul_supplemental_pack import templates as packing_templates
    from dsl_timing import reserve_cpu

    plan_path = work / 'plan.json'
    plan = json.loads(plan_path.read_text())
    directories = sorted((work / 'native').glob('batch-*'))
    indices = list(range(len(directories))) if args.batch_index is None else [args.batch_index]
    if any(not 0 <= index < len(directories) for index in indices):
        parser.error('batch-index is outside the native inventory')
    lease = reserve_cpu(args.cpu)
    os.sched_setaffinity(0, {args.cpu})
    try:
        with ExitStack() as locks:
            for index in indices:
                locks.enter_context(batch_lock(directories[index]))
            if args.phase == 'measure':
                if args.quiet_barrier is None:
                    parser.error('measure requires a root-approved quiet-barrier record')
                barrier = json.loads(args.quiet_barrier.read_text())
                if not barrier.get('approved') or barrier.get('cpu') != args.cpu:
                    raise ValueError('Quiet barrier is not approved for the measurement CPU')
                for pid in barrier.get('guard_worker_pids', []):
                    if Path(f'/proc/{pid}').exists():
                        raise ValueError('Guard worker still exists; timing cannot start')
                for pid in barrier.get('compiler_pids', []):
                    if Path(f'/proc/{pid}').exists():
                        raise ValueError('Compiler still exists; timing cannot start')
                for index in indices:
                    directory = directories[index]
                    if (work / f'timing-pending-batch-{index:03d}.json').exists():
                        raise ValueError('Pending timing already exists; archive an explicit failed attempt before retrying')
                    pre_path = directory / 'timing-pre.json'
                    pre = json.loads(pre_path.read_text())
                    if barrier.get('pre_records', {}).get(str(pre_path)) != digest(pre_path):
                        raise ValueError('Quiet barrier does not cover this exact preguard record')
                    check_inputs(pre['inputs'])
                    if pre['batch_index'] != index or pre['directory'] != str(directory):
                        raise ValueError('Preguard does not belong to this private batch')
                    records = artifact_inventory(directory)
                    manifest = json.loads((directory / 'manifest.json').read_text())
                    artifacts = {row['registration_id']: row for row in manifest['artifacts']}
                    oracle = json.loads((work / f'validation-batch-{index:03d}.json').read_text())
                    valid = {row['registration']['registration_id']: row for row in oracle['rows'] if row.get('validated')}
                    rows = []
                    for group in plan['groups'][index::len(directories)]:
                        selected = [row for row in group['rows'] if row['registration_id'] in pre['registrations']]
                        if not selected:
                            continue
                        try:
                            pair = packing_templates(selected[0]) if plan.get('suite') == 'packing' else templates(selected[0])[1]
                            artifact = artifacts[selected[0]['registration_id']]
                            bound = [bind_artifact(template, artifact[label + '_source'], records)
                                     for label, template in zip(('dsl', 'cpp'), pair)]
                        except (AssertionError, RuntimeError, ValueError, TypeError) as error:
                            for row in selected:
                                failure = dict(valid[row['registration_id']])
                                failure.update(validated=False, timed=False, phase='measurement_oracle_or_timing', error=repr(error))
                                for name in TIMING_FIELDS:
                                    failure.pop(name, None)
                                rows.append(failure)
                                print('FAILED', row['registration_id'], flush=True)
                            continue
                        def measure_row(row, bound=bound, valid=valid):
                            for label, (_, evidence) in zip(('dsl', 'cpp'), bound):
                                original = valid[row['registration_id']]['artifacts'][label]
                                if (evidence['binary'], evidence['binary_sha256']) != (original['binary'], original['binary_sha256']):
                                    raise ValueError('Exact timing binary differs from the oracle binary')
                            packing = plan.get('suite') == 'packing'
                            if packing:
                                values, expected, correction = packing_arguments(row)
                            else:
                                values, expected = lifecycle_arguments(scenario_for(row))
                            if row['suite'] == 'StatefulMatmul':
                                values[0].fill(1)
                                values[1].fill(1)
                                expected.fill(row['shape'][2])
                            calls = {label: lambda repeat, invoke=invoke, values=values: invoke(values, repeat)
                                     for label, (invoke, _) in zip(('dsl', 'cpp'), bound)}
                            output = values[1] if packing else values[2]
                            tolerance = 0.02 if output.dtype.itemsize <= 2 else (8e-4 if output.dtype.itemsize == 4 else 2e-10)
                            for call in calls.values():
                                for poison in (0xA5, 0x5A):
                                    output.view(np.uint8).fill(poison)
                                    if packing and correction is not None:
                                        values[2].view(np.uint8).fill(poison)
                                    call(1)
                                    if packing:
                                        np.testing.assert_array_equal(output.view(np.uint8), expected.view(np.uint8))
                                        if correction is not None:
                                            np.testing.assert_array_equal(values[2], correction)
                                    elif output.dtype.kind in 'iu':
                                        np.testing.assert_array_equal(output, expected)
                                    else:
                                        np.testing.assert_allclose(output, expected, rtol=tolerance, atol=tolerance)
                            return measure_pair(calls, args.cpu, args.rounds)
                        rows.extend(measure_rows(selected, valid, measure_row))
                    pending = {'admitted': False, 'rows': rows, 'inputs': pre['inputs'],
                               'pre_sha256': digest(pre_path), 'quiet_barrier': str(args.quiet_barrier),
                               'quiet_barrier_sha256': digest(args.quiet_barrier)}
                    check_inputs(pre['inputs'])
                    save(work / f'timing-pending-batch-{index:03d}.json', pending)
                return

            index = indices[0]
            directory = directories[index]
            status_path = directory / 'build-status.json'
            status = json.loads(status_path.read_text())
            checkpoint = Path(status['attestation_checkpoint'])
            if not status['admitted'] or digest(checkpoint) != status['attestation_checkpoint_sha256']:
                raise ValueError('Native batch checkpoint is not admitted')
            attestation = BatchInputAttestation.restore(directory)
            try:
                if args.phase == 'pre':
                    if (directory / 'timing-pre.json').exists():
                        raise ValueError('Immutable timing preguard already exists')
                    oracle_path = work / f'validation-batch-{index:03d}.json'
                    oracle = json.loads(oracle_path.read_text())
                    if not oracle['completed']:
                        raise ValueError('Oracle batch is not complete')
                    inputs = {str(plan_path): digest(plan_path), str(directory / 'manifest.json'): digest(directory / 'manifest.json'),
                              str(oracle_path): digest(oracle_path)}
                    inputs.update({str(root / 'benchmarks' / name): digest(root / 'benchmarks' / name)
                                   for name in HARNESS_INPUTS})
                    inputs[str(Path(__file__).resolve())] = digest(Path(__file__).resolve())
                    for admission in oracle['batch_admissions']:
                        if digest(admission['checkpoint']) != admission['checkpoint_sha256']:
                            raise ValueError('Oracle checkpoint changed')
                        inputs[admission['checkpoint']] = admission['checkpoint_sha256']
                    for row in oracle['rows']:
                        if not row.get('validated'):
                            continue
                        for artifact in row['artifacts'].values():
                            if digest(artifact['binary']) != artifact['binary_sha256']:
                                raise ValueError('Oracle binary changed before timing')
                            inputs[artifact['binary']] = artifact['binary_sha256']
                    attestation.guard('supplemental_timing_pre')
                    for source in attestation.sources:
                        for name, expected in source['inputs'][0]:
                            inputs[name] = expected
                    retained = directory / 'timing-pre-attestation.json'
                    retained.write_bytes(checkpoint.read_bytes())
                    retained.chmod(0o444)
                    inputs[str(retained)] = digest(retained)
                    check_inputs(inputs)
                    pre = {'directory': str(directory), 'batch_index': index, 'inputs': inputs,
                           'registrations': [row['registration']['registration_id'] for row in oracle['rows'] if row.get('validated')],
                           'expected_registered': oracle['registered'], 'native_available_pairs': oracle['available_pairs'],
                           'native_failures': oracle['native_failures'],
                           'oracle_failures': [row for row in oracle['rows'] if not row.get('validated')],
                           'oracle_artifacts': {row['registration']['registration_id']: row['artifacts']
                                                for row in oracle['rows'] if row.get('validated')},
                           'admitted': False}
                    save(directory / 'timing-pre.json', pre)
                    (directory / 'timing-pre.json').chmod(0o444)
                else:
                    if (work / f'timing-batch-{index:03d}.json').exists():
                        raise ValueError('Admitted timing batch already exists')
                    pre_path = directory / 'timing-pre.json'
                    pre = json.loads(pre_path.read_text())
                    pre['record_sha256'] = digest(pre_path)
                    pending = json.loads((work / f'timing-pending-batch-{index:03d}.json').read_text())
                    check_inputs({pending['quiet_barrier']: pending['quiet_barrier_sha256']})
                    check_inputs(pre['inputs'])
                    attestation.guard('supplemental_timing_post')
                    check_inputs(pre['inputs'])
                    rows = admit_pending(pre, pending, pre['inputs'])
                    retained = directory / 'timing-attestation.json'
                    retained.write_bytes(checkpoint.read_bytes())
                    retained.chmod(0o444)
                    admission = {'directory': str(directory), 'checkpoint': str(retained), 'checkpoint_sha256': digest(retained),
                                 'manifest_sha256': digest(directory / 'manifest.json'),
                                 'registrations': [row['registration']['registration_id'] for row in rows]}
                    result = result_document(rows, pre['expected_registered'], len(rows), pre['native_failures'], [admission])
                    result.update(native_available_pairs=pre['native_available_pairs'], oracle_failures=pre['oracle_failures'])
                    result['all_passed'] = result['all_passed'] and not pre['oracle_failures']
                    result['timing_chain'] = {'pre': str(pre_path), 'pre_sha256': pre['record_sha256'],
                        'pending': str(work / f'timing-pending-batch-{index:03d}.json'),
                        'pending_sha256': digest(work / f'timing-pending-batch-{index:03d}.json'),
                        'inputs': pre['inputs']}
                    save(work / f'timing-batch-{index:03d}.json', result)
            finally:
                status['attestation_checkpoint_sha256'] = digest(checkpoint)
                save(status_path, status)
    finally:
        lease.close()


if __name__ == '__main__':
    main()
