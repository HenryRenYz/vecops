# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Aggregate disjoint, fully post-guarded untimed supplemental batch results."""
import argparse
import hashlib
import json
from pathlib import Path

from dsl_matmul_supplemental_validate import result_document


def validate_batch(document, expected_rows, directory):
    expected = {row['registration_id']: row for row in expected_rows}
    rows = document['rows']
    row_ids = [row['registration']['registration_id'] for row in rows]
    failed = document.get('native_failures', [])
    failed_ids = [row['registration_id'] for row in failed]
    if (len(set(row_ids)) != len(row_ids) or len(set(failed_ids)) != len(failed_ids)
            or set(row_ids).intersection(failed_ids)
            or set(row_ids).union(failed_ids) != set(expected)):
        raise ValueError('Private batch does not exactly cover its expected registration IDs')
    if any(row['registration'] != expected[row['registration']['registration_id']] for row in rows):
        raise ValueError('Private batch registration metadata differs from the plan')
    if any('row' in failure and failure['row'] != expected[failure['registration_id']] for failure in failed):
        raise ValueError('Private batch failure metadata differs from the plan')
    admissions = document['batch_admissions']
    if (len(admissions) != 1 or Path(admissions[0]['directory']).resolve() != directory.resolve()
            or len(admissions[0]['registrations']) != len(row_ids)
            or set(admissions[0]['registrations']) != set(row_ids)):
        raise ValueError('Private batch admission does not match its directory and healthy registrations')


def aggregate(documents):
    rows, failures, oracle_failures, admissions = [], [], [], []
    seen = set()
    for document in documents:
        if not document['completed']:
            raise ValueError('Cannot aggregate an incomplete private batch')
        for row in document['rows']:
            identity = row['registration']['registration_id']
            if identity in seen:
                raise ValueError('Duplicate registration in validation batches')
            seen.add(identity)
            rows.append(row)
        failures.extend(document.get('native_failures', []))
        oracle_failures.extend(document.get('oracle_failures', []))
        admissions.extend(document['batch_admissions'])
    failed_ids = [failure['registration_id'] for failure in failures]
    if len(set(failed_ids)) != len(failed_ids) or seen.intersection(failed_ids):
        raise ValueError('Native failure registrations overlap admitted pairs')
    oracle_failed_ids = [row['registration']['registration_id'] for row in oracle_failures]
    if (len(set(oracle_failed_ids)) != len(oracle_failed_ids)
            or set(oracle_failed_ids).intersection(seen.union(failed_ids))):
        raise ValueError('Original oracle failures overlap measured or native-failed registrations')
    result = result_document(rows, sum(doc['registered'] for doc in documents),
                             sum(doc['available_pairs'] for doc in documents), failures, admissions)
    if result['registered'] != result['available_pairs'] + len(failures) + len(oracle_failures):
        raise ValueError('Registered coverage is not available pairs plus native failures')
    result['oracle_failures'] = oracle_failures
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work-dir', type=Path, required=True)
    parser.add_argument('--phase', choices=('validation', 'timing'), default='validation')
    args = parser.parse_args()
    paths = sorted(args.work_dir.glob(f'{args.phase}-batch-*.json'))
    expected = len(list((args.work_dir / 'native').glob('batch-*')))
    if len(paths) != expected:
        raise ValueError('Every native batch needs a separate completed validation result')
    documents = [json.loads(path.read_text()) for path in paths]
    plan = json.loads((args.work_dir / 'plan.json').read_text())
    for index, (path, document) in enumerate(zip(paths, documents)):
        if path.name != f'{args.phase}-batch-{index:03d}.json':
            raise ValueError('Validation batch filenames do not match the complete native batch inventory')
        expected_rows = [row for group in plan['groups'][index::expected] for row in group['rows']]
        if args.phase == 'timing':
            oracle = json.loads((args.work_dir / f'validation-batch-{index:03d}.json').read_text())
            failed_oracle_ids = {row['registration']['registration_id'] for row in oracle['rows'] if not row.get('validated')}
            expected_rows = [row for row in expected_rows if row['registration_id'] not in failed_oracle_ids]
        validate_batch(document, expected_rows, args.work_dir / 'native' / f'batch-{index:03d}')
    # Check retained proof bytes before admitting the aggregate. The strict
    # summary additionally hashes every attested source and exact binary.
    for document in documents:
        for admission in document['batch_admissions']:
            for name, digest in [(admission['checkpoint'], admission['checkpoint_sha256']),
                                 (str(Path(admission['directory']) / 'manifest.json'), admission['manifest_sha256'])]:
                if hashlib.sha256(Path(name).read_bytes()).hexdigest() != digest:
                    raise ValueError('Retained batch provenance changed before aggregation')
    result = aggregate(documents)
    if args.phase == 'validation' and result['registered'] != plan['registrations']:
        raise ValueError('Aggregate does not cover the complete registration plan')
    if args.phase == 'timing':
        from dsl_matmul_supplemental_timing import verify_timing_chains
        for document in documents:
            verify_timing_chains(document)
        for document in documents:
            chain = document['timing_chain']
            for name, expected_digest in [(chain['pre'], chain['pre_sha256']), (chain['pending'], chain['pending_sha256']),
                                          *chain['inputs'].items()]:
                if hashlib.sha256(Path(name).read_bytes()).hexdigest() != expected_digest:
                    raise ValueError('Timing pre/measurement/post input chain changed')
        oracle = json.loads((args.work_dir / 'validation-results.json').read_text())
        if not oracle['completed']:
            raise ValueError('Full oracle aggregate is not complete')
        result.update(registered=plan['registrations'], native_available_pairs=oracle['available_pairs'],
                      native_failures=oracle['native_failures'],
                      oracle_failures=[row for row in oracle['rows'] if not row.get('validated')],
                      timing_chains=[document['timing_chain'] for document in documents])
        result['all_passed'] = result['all_passed'] and not result['native_failures'] and not result['oracle_failures']
    (args.work_dir / f'{args.phase}-results.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
