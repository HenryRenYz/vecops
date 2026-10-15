# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Subset scope and deterministic coverage contracts, without native compilation."""
import importlib.util
from pathlib import Path

import pytest

path = Path(__file__).resolve().parents[2] / 'benchmarks/dsl_matmul_regression_subsets.py'
spec = importlib.util.spec_from_file_location('regression_subsets_test', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def rows():
    return [dict(execution_id=str(i), m=1 if i%2 else 16, n=17+i, k=32+i,
                 meta='Const' if i%2 else 'Dynamic', packing='raw' if i<3 else 'packedb',
                 native_a='bf16', native_b='bf16', catalog='test') for i in range(7)]


def test_nested_selection_is_order_independent_and_preserves_future_stress():
    options = dict(key='execution_id', pins={'smoke':['1'], 'extended':['6']}, stress=['6'])
    budgets = [('smoke',2),('standard',4),('extended',6)]
    result = module.nested_select(rows(), budgets, **options)
    assert result == module.nested_select(list(reversed(rows())), budgets, **options)
    assert result['smoke'][0] == '1'
    assert '6' not in result['standard'] and '6' in result['extended']
    for left, right in [('smoke','standard'),('standard','extended'),('extended','full')]:
        assert result[right][:len(result[left])] == result[left]
    assert set(result['full']) == {str(i) for i in range(7)}


def test_selection_rejects_unknown_duplicate_and_over_budget_pins():
    with pytest.raises(ValueError, match='outside'):
        module.nested_select(rows(), [('smoke',2)], key='execution_id', pins={'smoke':['missing']})
    with pytest.raises(ValueError, match='duplicate'):
        module.nested_select(rows()+rows()[:1], [('smoke',2)], key='execution_id')
    with pytest.raises(ValueError, match='budget'):
        module.nested_select(rows(), [('smoke',1)], key='execution_id', pins={'smoke':['1','2']})
    with pytest.raises(ValueError, match='extended'):
        module.nested_select(rows(), [('smoke',2)], key='execution_id', pins={'smoke':['6']}, stress=['6'])


def test_coverage_exposes_missing_dimensions_and_counts_unique_compilations():
    all_rows = rows()
    groups = {str(i):'shared' if i<3 else str(i) for i in range(7)}
    report = module.coverage(all_rows,['0','1'], 'execution_id',groups)
    assert report['count'] == 2 and report['compile_groups'] == 1
    assert report['missing_features'] and report['metadata_features'] < report['available_features']
    assert report['by_meta'] == {'Dynamic':1,'Const':1}
    assert module.features(all_rows[0]) != module.features(all_rows[1])


def test_features_do_not_claim_catalog_unsupported_alias_or_stride_coverage():
    tags = module.features(rows()[0])
    assert not any('alias' in key or 'stride' in key for key in tags)
