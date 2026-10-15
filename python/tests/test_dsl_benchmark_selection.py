# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Exact subset scope and immutable selection identity, without native builds."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'benchmarks'))
from dsl_benchmark_selection import selection_identity, select_rows, verify_selection_resume
from dsl_matmul_full_suite import make_plan


@pytest.mark.parametrize('field', ['execution_ids', 'registration_ids'])
@pytest.mark.parametrize('ids', [[], ['a', 'a'], [1], ['']])
def test_reject_invalid_ids(tmp_path, field, ids):
  p=tmp_path/'selection.json';p.write_text(json.dumps({'architecture':'arm',field:ids}))
  with pytest.raises(ValueError):selection_identity(p,'arm',field)


@pytest.mark.parametrize('field', ['execution_ids', 'registration_ids'])
def test_scope_and_resume_content_identity(tmp_path, field):
  p=tmp_path/'selection.json';p.write_text(json.dumps({'architecture':'arm',field:['b']}))
  identity=selection_identity(p,'arm',field)
  assert select_rows(['a','b','c'],identity,field,lambda x:x)==('b',)
  with pytest.raises(ValueError,match='architecture'):selection_identity(p,'x86',field)
  with pytest.raises(ValueError,match='out-of-scope'):select_rows(['a'],identity,field,lambda x:x)
  verify_selection_resume(identity,selection_identity(p,'arm',field))
  with pytest.raises(ValueError,match='resume'):verify_selection_resume(identity,None)
  p.write_text(json.dumps({'architecture':'arm',field:['b']},indent=2))
  with pytest.raises(ValueError,match='resume'):verify_selection_resume(identity,selection_identity(p,'arm',field))
  p.write_text(json.dumps({'architecture':'arm',field:['a']}))
  with pytest.raises(ValueError,match='resume'):verify_selection_resume(identity,selection_identity(p,'arm',field))


def test_filter_before_grouping_retains_full_source_audit(tmp_path, monkeypatch):
  import dsl_matmul_catalog
  cases=['source-a','source-b'];monkeypatch.setattr(dsl_matmul_catalog,'load_cases',lambda **kwargs:cases)
  rows=[SimpleNamespace(execution_id=x,compilation_key='shared') for x in ['a','b']]
  source=SimpleNamespace(scenarios=lambda cases,levels:rows,
    audit_document=lambda cases, executions:{'source_hashes':{c:'hash' for c in cases},
      'executions':[{'execution_id':x.execution_id} for x in executions]})
  p=tmp_path/'ids.json';p.write_text(json.dumps({'architecture':'arm','execution_ids':['b']}))
  args=SimpleNamespace(architecture='arm',feature=[],complete_registration=True,level=['matmul'],
    execution_ids_file=p,snapshot=tmp_path,extra_options='{}',label='default',cflag=[],target='Native',
    cc='clang',cxx='clang++',cpu=1,rounds=2,stages=['compute'])
  plan=make_plan(args,source)
  assert plan['selected_execution_ids']==['b']
  assert plan['groups'][0]['execution_ids']==['b']
  assert plan['executions']==[{'execution_id':'b'}]
  assert set(plan['source_hashes'])==set(cases)


def test_supplemental_real_plan_exact_scope_and_resume(tmp_path):
  from dsl_matmul_supplemental_accelerated import make_plan as supplemental_plan, verify_plan_resume
  from dsl_matmul_supplemental_catalog import supplemental_cases
  rows=[r for r in supplemental_cases('arm',['sme_f64']) if r['suite']=='MatmulPack' and r['dsl_equivalent']]
  # Two registrations may share a native template; the unselected row must stay out.
  field='registration_ids';p=tmp_path/'supp.json'
  p.write_text(json.dumps({'architecture':'arm',field:[rows[0]['registration_id']]}))
  identity=selection_identity(p,'arm',field)
  plan=supplemental_plan(rows,'packing',None,identity)
  assert plan['registrations']==1
  assert [r['registration_id'] for g in plan['groups'] for r in g['rows']]==identity[field]
  verify_plan_resume(plan,supplemental_plan(rows,'packing',None,identity))
  with pytest.raises(ValueError,match='resume'):verify_plan_resume(plan,supplemental_plan(rows,'packing',None))
  changed=dict(plan,groups=[])
  with pytest.raises(ValueError,match='scope'):verify_plan_resume(plan,changed)
  with pytest.raises(ValueError,match='out-of-scope'):supplemental_plan(rows[1:],'packing',None,identity)


def test_plan_only_rejects_worker_before_loading_snapshot(tmp_path,monkeypatch):
  import dsl_matmul_full_suite as suite
  import dsl_matmul_supplemental_accelerated as supplemental
  with pytest.raises(SystemExit) as error:
    suite.main(['--snapshot',str(tmp_path),'--suite-dir',str(tmp_path/'work'),'--architecture','arm','--cpu','1',
      '--plan-only','--worker','build'])
  assert error.value.code==2
  monkeypatch.setattr(sys,'argv',['supplemental','--root',str(tmp_path),'--work-dir',str(tmp_path/'work'),
    '--architecture','arm','--cc','clang','--cxx','clang++','--plan-only','--lower-only'])
  with pytest.raises(SystemExit) as error:supplemental.main()
  assert error.value.code==2
