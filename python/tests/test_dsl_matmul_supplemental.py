# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Supplemental coverage preserves original selectors, metadata and lifecycle."""

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'benchmarks'))
from dsl_matmul_supplemental_catalog import manifest, supplemental_cases
from dsl_matmul_supplemental_lifecycle import lifecycle_arguments, scenario_for
from dsl_matmul_supplemental_lifecycle import templates as lifecycle_templates
from dsl_matmul_supplemental_pack import compensation_oracle, oracle, templates
from dsl_matmul_supplemental_summary import summarize
from vecops.language._compiler.backends.cpu.backend import CPUBackend


@pytest.mark.parametrize('arch,features,counts,equivalent',[
  ('x86',(),{'MatmulPack':220,'StatefulMatmul':10,'MatmulWeightReuse':394},220),
  ('x86',('amx_fp16',),{'MatmulPack':220,'StatefulMatmul':10,'MatmulWeightReuse':410},220),
  ('arm',(),{'MatmulPack':220,'StatefulMatmul':10,'MatmulWeightReuse':426,'MatmulMMLAExperiment':576},220),
  ('arm',('sme_f64',),{'MatmulPack':252,'StatefulMatmul':10,'MatmulWeightReuse':442,'MatmulMMLAExperiment':576},252),
])
def test_real_registration_counts_and_unsupported_contracts(arch,features,counts,equivalent):
  data=manifest(arch,features)
  assert data['counts']==counts
  assert data['equivalent_packing_count']==equivalent
  ids=[row['registration_id'] for row in data['registrations']]
  assert len(ids)==len(set(ids))
  assert all(row.get('unsupported_reason') for row in data['registrations'] if not row['dsl_equivalent'])


def test_weight_reuse_preserves_actual_shard_metadata():
  rows=[row for row in supplemental_cases('x86') if row['suite']=='MatmulWeightReuse']
  lifecycle=[row for row in rows if row['cohort']=='lifecycle']
  assert Counter(row['metadata'] for row in lifecycle)=={'Dynamic':141,'Const':141}
  assert all(row['metadata']==('Dynamic' if row['strategy']=='raw' else 'Const') for row in lifecycle)
  assert all(row['broadcast']=='B' and row['calls_per_pack'] in (1,2,4,8,16) for row in rows)


def test_fused_byte_pipeline_is_platform_specific():
  for arch,expected in [('x86','multiply4'),('arm','identity')]:
    rows=[row for row in supplemental_cases(arch) if row['suite']=='MatmulPack' and row['name']=='FP32ToS8']
    assert rows and all(row['pipeline']==expected for row in rows)
    assert {row['selector'] for row in rows}==({'Vector'} if arch=='x86' else {'Vector','SMEPostprocess'})


@pytest.mark.parametrize('side',['A','B'])
def test_amx_public_oracle_matches_element_abi(side, monkeypatch):
  # This byte oracle checks the AMX contract on every test host. Select
  # x86 in both the oracle dispatcher and its shared geometry provider;
  # otherwise an ARM host correctly returns the unrelated SME layout.
  monkeypatch.setattr('dsl_matmul_supplemental_pack.is_arm',lambda:False)
  monkeypatch.setattr('dsl_benchmark_utils.is_arm',lambda:False)
  row={'native':'int8','pipeline':'identity','side':side}
  logical=np.arange(19*67,dtype=np.int16).astype(np.int8).reshape(19,67)
  actual=oracle(row,logical)
  expected=np.zeros(2*2*16*64,dtype=np.int8)
  for s in range(19):
    for k in range(67):
      base=(s//16*2+k//64)*16*64
      offset=s%16*64+k%64 if side=='A' else (k%64//4)*16*4+s%16*4+k%4
      expected[base+offset]=logical[s,k]
  np.testing.assert_array_equal(actual,expected)


def test_fused_packing_dsl_retains_transform_and_cpp_selector():
  row=next(row for row in supplemental_cases('x86') if row['suite']=='MatmulPack' and row['name']=='FP32ToS8')
  dsl,cpp=templates(row)
  source=CPUBackend().lower(dsl.ir,(),target='AVX512').source
  assert any('vecops::vec::mul(' in line and 'std::int64_t{4}' in line for line in source.splitlines())
  assert cpp is not None and row['selector']=='Vector'


def test_float_to_unsigned_byte_oracle_saturates_before_truncation():
  row={'native':'uint8','pipeline':'identity','side':'A'}
  result=oracle(row,np.asarray([[-2.9,0.9,255.9,300]],dtype=np.float32))
  np.testing.assert_array_equal(result[:4],np.asarray([0,0,255,255],dtype=np.uint8))


def test_lifecycle_coverage_distinguishes_api_gap_from_pending_validation():
  rows=supplemental_cases('x86')
  legacy=[r for r in rows if r['suite']=='StatefulMatmul' and r['name']=='Legacy']
  prepared=[r for r in rows if r['suite']=='StatefulMatmul' and r['name']=='Prepared']
  reuse=[r for r in rows if r['suite']=='MatmulWeightReuse']
  assert len(legacy)==5 and len(prepared)==5 and len(reuse)==394
  assert all(r['dsl_equivalent'] for r in (*legacy,*reuse))
  assert all(r['comparison_status']=='unsupported_public_api' for r in prepared)
  with pytest.raises(ValueError,match='Prepared operation plan'):
    scenario_for(prepared[0])


@pytest.mark.parametrize('strategy',['raw','pack_once_then_reuse'])
def test_lifecycle_preserves_batch_native_types_and_inner_reuse(strategy):
  row=next(r for r in supplemental_cases('x86') if r['suite']=='MatmulWeightReuse'
           and r['strategy']==strategy and r['calls_per_pack']==16 and r['name']=='small')
  scenario,pair=lifecycle_templates(row)
  assert scenario.batch==row['batch'] and scenario.shared_b
  assert scenario.online==(strategy=='pack_once_then_reuse')
  assert scenario.reference_bound_invocation
  assert scenario.types[4:6]==('bfloat16','bfloat16')
  assert pair[0].ir is not None and pair[1] is not None
  reference=Path(__file__).resolve().parents[2]/'benchmarks/dsl_matmul_supplemental_lifecycle_reference.cpp'
  source=reference.read_text()
  assert source.count('call < spec::CallsPerPack')==2
  assert source.index('execution::ExecutionSession execution{workspace}')<source.index('call < spec::CallsPerPack')


def test_lifecycle_mutable_bias_preserves_shared_storage_schema():
  row=next(r for r in supplemental_cases('x86') if r['suite']=='MatmulWeightReuse'
           and r['batch']==8 and r['name']=='small' and r['output_pipeline']=='BiasRelu')
  scenario=scenario_for(row)
  values,expected=lifecycle_arguments(scenario)
  prior=values[3]
  assert prior.flags.writeable and prior.strides[0]==0 and prior.shape==(8,1,64)
  before=prior[0,0,0].copy()
  prior[0,0,0]=before+1
  assert np.all(prior[:,0,0]==before+1)
  assert expected.shape==(8,1,64)


@pytest.mark.parametrize('mode',['SeparateSum','Fused'])
def test_compensation_covers_same_outputs_and_preserves_cpp_algorithm(mode):
  row=next(r for r in supplemental_cases('x86') if r.get('compensation_mode')==mode)
  assert row['dsl_equivalent'] and row['dsl_schedule']=='pack_then_unsigned_modular_column_reduce'
  dsl,cpp=templates(row)
  assert dsl.ir is not None and cpp is not None
  values=np.asarray([[-31,-1,7],[3,-3,0]],dtype=np.float32)*0.25
  np.testing.assert_array_equal(compensation_oracle(row,values),np.asarray([75,0],dtype=np.int32))


def test_strict_summary_does_not_turn_absent_results_into_unsupported(tmp_path):
  data=summarize('x86',(),[tmp_path])
  assert data['categories']['MatmulPack']['registered']==220
  assert data['categories']['MatmulPack']['comparable']==220
  assert data['categories']['MatmulWeightReuse']['comparable']==394
  assert data['categories']['StatefulMatmul']['public_api_unsupported']==5
  assert all(category['built']==category['validated']==category['timed']==0 for category in data['categories'].values())
