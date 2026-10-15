# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Restore guarded supplemental batches for independent oracle checks and timing."""
import argparse
import hashlib
import json
import os
import re
import sys
from contextlib import ExitStack, contextmanager
from pathlib import Path


@contextmanager
def batch_lock(directory):
    """All validation and timing phases share one exclusive checkpoint lock."""
    import fcntl
    with (directory / 'supplemental-driver.lock').open('a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f'Another driver owns this native batch: {directory}') from error
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def result_document(rows, registered, available_pairs, native_failures, admissions):
    """Execution completion excludes unavailable native pairs; success does not."""
    completed = len(rows) == available_pairs
    return {'rows': rows, 'completed': completed,
            'all_passed': completed and not native_failures and available_pairs == registered
                          and all(row.get('validated') and not row.get('error') for row in rows),
            'registered': registered, 'available_pairs': available_pairs,
            'native_failures': native_failures, 'batch_admissions': admissions}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--work-dir', type=Path, required=True)
    parser.add_argument('--cc', required=True)
    parser.add_argument('--cxx', required=True)
    parser.add_argument('--target', default='Native')
    parser.add_argument('--cpu', type=int, required=True)
    parser.add_argument('--rounds', type=int, default=12)
    parser.add_argument('--time', action='store_true')
    parser.add_argument('--smoke', action='store_true', help='one direct-artifact oracle row, never formal timing')
    parser.add_argument('--batch-index', type=int, help='validate one private batch without timing')
    args = parser.parse_args()
    if args.smoke and args.time:
        parser.error('smoke is an oracle-only check')
    if args.batch_index is not None and (args.time or args.smoke):
        parser.error('batch-index is an untimed full-batch validation only')
    root, work = args.root.resolve(), args.work_dir.resolve()
    sys.path.insert(0, str(root / 'benchmarks'))
    from dsl_snapshot_entry import select_snapshot
    select_snapshot(root)
    os.chdir(root.parents[1])
    import numpy as np
    import vecops
    from dsl_matmul_artifact_ab import adapter_contract, template_contract
    from dsl_matmul_repair_suite import BatchInputAttestation
    from dsl_matmul_supplemental_lifecycle import lifecycle_arguments, scenario_for, templates
    from dsl_matmul_supplemental_pack import measure_pair
    from dsl_timing import reserve_cpu

    plan = json.loads((work / 'plan.json').read_text())
    if plan.get('suite') == 'packing':
        from dsl_matmul_supplemental_pack import packing_arguments
        from dsl_matmul_supplemental_pack import templates as packing_templates
    completion = json.loads((work / 'complete.json').read_text())
    if not completion['completed']:
        raise RuntimeError('All guarded native batches must be admitted before validation')
    groups = plan['groups']
    native_directories = sorted((work / 'native').glob('batch-*'))
    native_statuses = [json.loads((directory / 'build-status.json').read_text())
                       for directory in native_directories]
    if args.batch_index is not None and not 0 <= args.batch_index < len(native_directories):
        parser.error('batch-index is outside the private batch inventory')
    selected = [args.batch_index] if args.batch_index is not None else list(range(len(native_directories)))
    available_count = sum(len(native_statuses[index]['registrations']) for index in selected)
    native_failures = [failure for index in selected for status in [native_statuses[index]]
                       for failure in status.get('failed_registrations', [])]
    registered_count = sum(len(group['rows']) for index in selected
                           for group in groups[index::len(native_directories)])
    lease = reserve_cpu(args.cpu)
    locks = ExitStack()
    os.sched_setaffinity(0, {args.cpu})
    results = []
    admissions = []
    def canonical(value):
        return json.dumps(value, sort_keys=True)

    def inventory(directory):
        records = []
        for adapter in sorted((directory / 'build').glob('kernel-batch-*/tasks/*/input/vecops_adapter.cpp')):
            text = adapter.read_text()
            source = re.search(r'#include "([^"\n]+\.cpp)"', text)
            if source is None:
                continue
            spec = adapter.with_name('vecops_spec.h').read_text()
            captures = {name: int(value) for name, value in re.findall(
                r'inline constexpr auto (\w+) = ::vecops::meta::cint<(-?\d+)>;', spec)}
            captures.update(re.findall(r'using (\w+) = ::vecops::(\w+);', spec))
            for binary in (adapter.parents[3] / 'artifacts' / adapter.parents[1].name).glob('*.so'):
                records.append((Path(source.group(1)).resolve(), captures, adapter_contract(text),
                                vecops._C.Executable.load(str(binary)), str(binary.resolve()),
                                hashlib.sha256(binary.read_bytes()).hexdigest()))
        return records

    def bind(template, source, records):
        contract, captures = template_contract(template)
        matches = [record for record in records if record[0] == Path(source).resolve()
                   and canonical(record[2]) == canonical(contract)
                   and all(record[1].get(name) == value for name, value in captures.items())]
        if len(matches) != 1:
            raise RuntimeError(f'Expected one exact source/ABI/capture artifact, found {len(matches)}: {source}')
        executable = matches[0][3]
        access = [parameter[7] for parameter in contract[0] if parameter[0] == 'TENSOR']
        def invoke(values, repeat):
            arguments = vecops._C.ArgumentMetadata([
                vecops._C.TensorView.from_numpy(value, writable=bool(mode & 2))
                for value, mode in zip(values, access)] + [repeat])
            if not executable.can_invoke(arguments):
                raise RuntimeError('Exact artifact rejects invocation metadata')
            executable.invoke(arguments)
        return invoke, {'binary': matches[0][4], 'binary_sha256': matches[0][5],
                        'source': str(Path(source).resolve()), 'contract': contract, 'captures': captures}
    try:
        for index, directory in enumerate(native_directories):
            if index not in selected:
                continue
            locks.enter_context(batch_lock(directory))
            status = json.loads((directory / 'build-status.json').read_text())
            checkpoint = Path(status['attestation_checkpoint'])
            if not status['admitted'] or hashlib.sha256(checkpoint.read_bytes()).hexdigest() != status['attestation_checkpoint_sha256']:
                raise RuntimeError(f'Native batch or its checkpoint was not admitted: {directory}')
            attestation = BatchInputAttestation.restore(directory)
            attestation.guard('supplemental_validation_pre')
            records = inventory(directory)
            manifest_document = json.loads((directory / 'manifest.json').read_text())
            by_id = {item['registration_id']: item for item in manifest_document['artifacts']}
            pending = []
            for group in groups[index::len(native_directories)]:
                available = set(status['registrations'])
                if not any(row['registration_id'] in available for row in group['rows']):
                    continue
                group_dir = work / 'groups' / hashlib.sha256(group['key'].encode()).hexdigest()
                proof = json.loads((group_dir / 'proof.json').read_text())
                pair = packing_templates(group['rows'][0]) if plan.get('suite') == 'packing' else templates(group['rows'][0])[1]
                if hashlib.sha256(Path(proof['source']).read_bytes()).hexdigest() != proof['raw_sha256']:
                    raise RuntimeError('Proved generated source changed before validation')
                artifact = by_id[group['rows'][0]['registration_id']]
                try:
                    bound = [bind(template, artifact[label + '_source'], records)
                             for label, template in zip(('dsl', 'cpp'), pair)]
                except (RuntimeError, ValueError, TypeError) as error:
                    for row in group['rows']:
                        if row['registration_id'] not in available:
                            continue
                        pending.append({'registration': row, 'validated': False, 'timed': False,
                                        'phase': 'artifact_binding', 'error': repr(error)})
                        print('FAILED', row['registration_id'], repr(error), flush=True)
                    continue
                for row in group['rows']:
                    if row['registration_id'] not in available:
                        continue
                    if args.smoke and pending:
                        break
                    try:
                        if plan.get('suite') == 'packing':
                            values, expected, correction = packing_arguments(row)
                            calls = {label: lambda repeat, invoke=invoke, values=values: invoke(values, repeat)
                                     for label, (invoke, _) in zip(('dsl', 'cpp'), bound)}
                            for call in calls.values():
                                for poison in (0xA5, 0x5A):
                                    values[1].view(np.uint8).fill(poison)
                                    if correction is not None:
                                        values[2].view(np.uint8).fill(poison)
                                    call(1)
                                    np.testing.assert_array_equal(values[1].view(np.uint8), expected.view(np.uint8))
                                    if correction is not None:
                                        np.testing.assert_array_equal(values[2], correction)
                            result = {'registration': row, 'validated': True, 'timed': args.time,
                                      'artifacts': {label: evidence for label, (_, evidence) in zip(('dsl', 'cpp'), bound)}}
                            if args.time:
                                result.update(measure_pair(calls, args.cpu, args.rounds))
                            pending.append(result)
                            print('PASS', row['registration_id'], flush=True)
                            continue
                        scenario = scenario_for(row)
                        values, expected = lifecycle_arguments(scenario)
                        if row['suite'] == 'StatefulMatmul':
                            values[0].fill(1)
                            values[1].fill(1)
                            expected.fill(row['shape'][2])
                        calls = {label: lambda repeat, invoke=invoke, values=values: invoke(values, repeat)
                                 for label, (invoke, _) in zip(('dsl', 'cpp'), bound)}
                        tolerance = 0.02 if values[2].dtype.itemsize <= 2 else (8e-4 if values[2].dtype.itemsize == 4 else 2e-10)
                        for call in calls.values():
                            for poison in (0xA5, 0x5A):
                                values[2].view(np.uint8).fill(poison)
                                call(1)
                                if values[2].dtype.kind in 'iu':
                                    np.testing.assert_array_equal(values[2], expected)
                                else:
                                    np.testing.assert_allclose(values[2], expected, rtol=tolerance, atol=tolerance)
                        result = {'registration': row, 'validated': True, 'timed': args.time,
                                  'artifacts': {label: evidence for label, (_, evidence) in zip(('dsl', 'cpp'), bound)}}
                        if args.time:
                            result.update(measure_pair(calls, args.cpu, args.rounds))
                        pending.append(result)
                        print('PASS', row['registration_id'], flush=True)
                    except (AssertionError, RuntimeError, ValueError, TypeError) as error:
                        pending.append({'registration': row, 'validated': False, 'timed': False,
                                        'phase': 'oracle_or_timing', 'error': repr(error)})
                        print('FAILED', row['registration_id'], repr(error), flush=True)
            attestation.guard('supplemental_validation_post')
            results.extend(pending)
            # Validation extends the checkpoint event log; preserve its new exact digest.
            status['attestation_checkpoint_sha256'] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            (directory / 'build-status.json').write_text(json.dumps(status, indent=2) + '\n')
            retained = directory / ('timing-attestation.json' if args.time else 'validation-attestation.json')
            retained.write_bytes(checkpoint.read_bytes())
            manifest = directory / 'manifest.json'
            admissions.append({'directory': str(directory), 'checkpoint': str(retained),
                               'checkpoint_sha256': hashlib.sha256(retained.read_bytes()).hexdigest(),
                               'manifest_sha256': hashlib.sha256(manifest.read_bytes()).hexdigest(),
                               'registrations': [r['registration']['registration_id'] for r in pending]})
            document = result_document(results, registered_count, available_count,
                                       native_failures, admissions)
            filename = 'smoke-results.json' if args.smoke else ('timing-results.json' if args.time else 'validation-results.json')
            if args.batch_index is not None:
                filename = f'validation-batch-{args.batch_index:03d}.json'
            (work / filename).write_text(json.dumps(document, indent=2) + '\n')
            if args.smoke:
                return
    finally:
        locks.close()
        lease.close()


if __name__ == '__main__':
    main()
