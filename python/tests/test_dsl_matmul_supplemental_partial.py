# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Pure-data partial native admission and performance-denominator regressions."""
import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'benchmarks'))
from dsl_matmul_supplemental_aggregate import aggregate, validate_batch
from dsl_matmul_supplemental_catalog import supplemental_cases
from dsl_matmul_supplemental_summary import summarize
from dsl_matmul_supplemental_timing import admit_pending, check_inputs, measure_rows, verify_timing_chains
from dsl_matmul_supplemental_validate import batch_lock, result_document


def test_partial_completion_is_distinct_from_all_passed():
    rows = [{'validated': True} for _ in range(247)]
    failures = [{'registration_id': str(index)} for index in range(5)]
    result = result_document(rows, 252, 247, failures, [])
    assert result['completed'] and not result['all_passed']
    assert result['registered'] == 252 and result['available_pairs'] == 247
    assert len(result['native_failures']) == 5
    missing = result_document(rows[:-1], 252, 247, failures, [])
    assert not missing['completed'] and not missing['all_passed']


def test_private_batch_lock_excludes_other_cpu_or_phase(tmp_path):
    with batch_lock(tmp_path), pytest.raises(RuntimeError, match='Another driver owns this native batch'), batch_lock(tmp_path):
        pass
    with batch_lock(tmp_path):
        pass


@pytest.mark.parametrize('filename', ['source.cpp', 'binary.so', 'manifest.json', 'plan.json', 'checkpoint.json'])
def test_timing_changed_real_input_prevents_admission(tmp_path, filename):
    path = tmp_path / filename
    path.write_bytes(b'original input')
    inputs = {str(path): hashlib.sha256(path.read_bytes()).hexdigest()}
    check_inputs(inputs)
    path.write_bytes(b'changed input')
    with pytest.raises(ValueError, match='private batch cannot be admitted'):
        check_inputs(inputs)


def test_pending_timing_requires_original_pre_and_unchanged_post_chain():
    artifacts = {'dsl': {'binary': 'dsl.so', 'binary_sha256': 'dsl-sha'},
                 'cpp': {'binary': 'cpp.so', 'binary_sha256': 'cpp-sha'}}
    row = {'registration': {'registration_id': 'one'}, 'validated': True, 'timed': True,
           'median_ns': {'dsl': 2, 'cpp': 1}, 'artifacts': artifacts}
    pre = {'record_sha256': 'pre-proof', 'inputs': {'source.cpp': 'original-sha'}, 'registrations': ['one'],
           'oracle_artifacts': {'one': artifacts}}
    pending = {'admitted': False, 'pre_sha256': 'pre-proof', 'inputs': pre['inputs'], 'rows': [row]}
    assert admit_pending(pre, pending, pre['inputs']) == [row]
    assert not pending['admitted']
    with pytest.raises(ValueError, match='input chain changed'):
        admit_pending(pre, pending, {'source.cpp': 'changed-sha'})
    with pytest.raises(ValueError, match='original preguard'):
        admit_pending(pre, {**pending, 'pre_sha256': 'wrong-pre'}, pre['inputs'])
    with pytest.raises(ValueError, match='original preguard'):
        admit_pending(pre, {**pending, 'admitted': True}, pre['inputs'])
    with pytest.raises(ValueError, match='exactly cover'):
        admit_pending(pre, {**pending, 'rows': [row, row]}, pre['inputs'])
    with pytest.raises(ValueError, match='artifacts differ'):
        admit_pending(pre, {**pending, 'rows': [{**row, 'artifacts': {}}]}, pre['inputs'])
    failure = {**row, 'validated': False, 'timed': False, 'error': 'numeric failure'}
    failure.pop('median_ns')
    assert admit_pending(pre, {**pending, 'rows': [failure]}, pre['inputs']) == [failure]
    with pytest.raises(ValueError, match='valid timing ratio'):
        admit_pending(pre, {**pending, 'rows': [{**failure, 'median_ns': {'dsl': 2, 'cpp': 1}}]}, pre['inputs'])


def test_measurement_row_failure_continues_without_a_ratio():
    selected = [{'registration_id': str(index)} for index in range(3)]
    originals = {row['registration_id']: {'registration': row, 'validated': True, 'timed': False}
                 for row in selected}
    def measure(row):
        if row['registration_id'] == '1':
            raise AssertionError('independent fresh oracle failed')
        return {'median_ns': {'dsl': 2, 'cpp': 1}}
    rows = measure_rows(selected, originals, measure)
    assert len(rows) == 3 and rows[0]['timed'] and rows[2]['timed']
    assert not rows[1]['validated'] and not rows[1]['timed'] and 'median_ns' not in rows[1]


def test_final_timing_statistics_cannot_drift_from_pending(tmp_path):
    artifacts = {'dsl': {'binary': 'dsl.so', 'binary_sha256': 'dsl-sha'}}
    row = {'registration': {'registration_id': 'one'}, 'validated': True, 'timed': True,
           'median_ns': {'dsl': 2, 'cpp': 1}, 'artifacts': artifacts}
    pre_path, pending_path = tmp_path / 'pre.json', tmp_path / 'pending.json'
    pre = {'inputs': {}, 'registrations': ['one'], 'oracle_artifacts': {'one': artifacts}}
    pre_path.write_text(json.dumps(pre))
    pre_sha = hashlib.sha256(pre_path.read_bytes()).hexdigest()
    pending = {'admitted': False, 'inputs': {}, 'pre_sha256': pre_sha, 'rows': [row]}
    pending_path.write_text(json.dumps(pending))
    chain = {'pre': str(pre_path), 'pre_sha256': pre_sha, 'pending': str(pending_path),
             'pending_sha256': hashlib.sha256(pending_path.read_bytes()).hexdigest(), 'inputs': {}}
    result = {'rows': [row], 'timing_chain': chain}
    verify_timing_chains(result)
    changed = {**row, 'median_ns': {'dsl': 999, 'cpp': 1}}
    with pytest.raises(ValueError, match='actual pending measurement'):
        verify_timing_chains({**result, 'rows': [changed]})


def test_disjoint_partial_batch_aggregation_and_rejection():
    rows = [{'registration': {'registration_id': str(index)}, 'validated': True}
            for index in range(247)]
    failures = [{'registration_id': str(index)} for index in range(247, 252)]
    first = result_document(rows[:123], 123, 123, [], [])
    second = result_document(rows[123:], 129, 124, failures, [])
    result = aggregate([first, second])
    assert result['registered'] == 252 and result['available_pairs'] == 247
    assert result['completed'] and not result['all_passed']
    assert len(result['rows']) == 247 and len(result['native_failures']) == 5
    with pytest.raises(ValueError, match='Duplicate registration'):
        aggregate([first, first])
    with pytest.raises(ValueError, match='incomplete private batch'):
        aggregate([first, result_document(rows[123:-1], 129, 124, failures, [])])
    with pytest.raises(ValueError, match='Native failure registrations overlap'):
        aggregate([first, result_document(rows[123:], 129, 124,
            [{'registration_id': '0'}, *failures[1:]], [])])


def test_batch_aggregation_requires_exact_ids_metadata_and_admission(tmp_path):
    expected = [{'registration_id': 'healthy', 'metadata': 'Const'},
                {'registration_id': 'failed', 'metadata': 'Dynamic'}]
    directory = tmp_path / 'native' / 'batch-000'
    row = {'registration': expected[0], 'validated': True}
    admission = {'directory': str(directory), 'registrations': ['healthy']}
    document = result_document([row], 2, 1, [{'registration_id': 'failed', 'row': expected[1]}], [admission])
    validate_batch(document, expected, directory)
    document['native_failures'] = [{'registration_id': 'replacement'}]
    with pytest.raises(ValueError, match='expected registration IDs'):
        validate_batch(document, expected, directory)

    document['native_failures'] = [{'registration_id': 'failed', 'row': expected[1]}]
    document['batch_admissions'][0]['directory'] = str(directory.with_name('batch-001'))
    with pytest.raises(ValueError, match='directory and healthy registrations'):
        validate_batch(document, expected, directory)
    document['batch_admissions'][0]['directory'] = str(directory)
    document['rows'][0]['registration'] = {'registration_id': 'healthy', 'metadata': 'Dynamic'}
    with pytest.raises(ValueError, match='metadata differs'):
        validate_batch(document, expected, directory)


def test_partial_timing_keeps_original_compile_and_oracle_failures_separate(tmp_path):
    healthy = {'registration_id': 'healthy'}
    native_failed = {'registration_id': 'native-failed'}
    numeric_failed = {'registration_id': 'oracle-failed'}
    directory = tmp_path / 'native' / 'batch-000'
    row = {'registration': healthy, 'validated': True, 'timed': True}
    admission = {'directory': str(directory), 'registrations': ['healthy']}
    document = result_document([row], 3, 1, [{'registration_id': 'native-failed', 'row': native_failed}], [admission])
    document['oracle_failures'] = [{'registration': numeric_failed, 'validated': False, 'error': 'numeric'}]
    validate_batch(document, [healthy, native_failed], directory)
    result = aggregate([document])
    assert result['registered'] == 3 and result['available_pairs'] == 1
    assert result['completed'] and not result['all_passed']
    assert len(result['native_failures']) == len(result['oracle_failures']) == 1


def test_partial_summary_counts_only_guarded_pairs_and_valid_timings(tmp_path):
    packing = [row for row in supplemental_cases('arm', ('sme_f64',))
               if row['suite'] == 'MatmulPack']
    assert len(packing) == 252
    batch = tmp_path / 'native' / 'batch-000'
    batch.mkdir(parents=True)
    source = batch / 'kernel.cpp'
    source.write_text('guarded fixture source\n')
    binary = batch / 'fixture.so'
    binary.write_bytes(b'fixture binary provenance')
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    checkpoint = batch / 'batch-input-attestation.json'
    checkpoint.write_text(json.dumps({'invalid_reason': None, 'policies': ['fixture'],
        'events': [{'phase': 'post_build'}],
        'sources': [{'inputs': [[(str(source), digest(source))]]}]}))
    failures = [{'registration_id': row['registration_id'], 'phase': 'native_compilation',
                 'error': 'LLVM compiler failure'} for row in packing[247:]]
    manifest = batch / 'manifest.json'
    manifest.write_text(json.dumps({'artifacts': [
        {'registration_id': row['registration_id'], 'row': row} for row in packing],
        'failures': failures}))
    (batch / 'build-status.json').write_text(json.dumps({'admitted': True,
        'registrations': [row['registration_id'] for row in packing[:247]],
        'attestation_checkpoint': str(checkpoint), 'attestation_checkpoint_sha256': digest(checkpoint)}))
    artifact = {'binary': str(binary), 'binary_sha256': digest(binary)}
    rows = [{'registration': row, 'validated': True, 'timed': True,
             'median_ns': {'dsl': 4, 'cpp': 1}, 'artifacts': {'dsl': artifact, 'cpp': artifact}}
            for row in packing[:247]]
    admission = {'directory': str(batch), 'checkpoint': str(checkpoint),
                 'checkpoint_sha256': digest(checkpoint), 'manifest_sha256': digest(manifest),
                 'registrations': [row['registration_id'] for row in packing[:247]]}
    (tmp_path / 'timing-results.json').write_text(json.dumps(
        result_document(rows, 252, 247, failures, [admission])))
    data = summarize('arm', ('sme_f64',), [tmp_path])['categories']['MatmulPack']
    assert data['registered'] == data['comparable'] == 252
    assert data['built'] == data['validated'] == data['timed'] == data['timed_rows'] == 247
    assert data['failed_unrecovered'] == 5 and data['public_api_unsupported'] == 0
    assert data['dsl_slower_than_2x_count'] == 247
    assert abs(data['dsl_over_cpp_geomean'] - 4) < 1e-12
