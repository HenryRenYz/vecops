# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Strict regression evidence and individual thresholds without native execution."""
import copy
import json
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'benchmarks'))
from dsl_matmul_regression_compare import compare, load_main, load_supplemental, main


def data(dsl=100,cpp=100):return {'a':{'metadata':{'m':4},'dsl':dsl,'cpp':cpp}}


def test_individual_findings_not_hidden_by_geomean():
  old=data(1000);old['b']=dict(old['a'])
  new=data(1200);new['b']={**old['b'],'dsl':100}
  report=compare(old,new,confirmation={**new,'a':{**new['a'],'dsl':1190}})
  assert report['metrics']['dsl_new_over_old']['geomean']<1
  assert report['suspected_regressions']==['a'] and report['improvements']==['b']
  assert report['repeated_regressions']==['a'] and report['review_required']
  assert report['rows'][0]['cpp_new_over_old']==1


def test_strict_dual_threshold_and_confirmation():
  assert not compare(data(100),data(151),relative=.6)['review_required']
  assert not compare(data(1000),data(1050))['review_required']
  report=compare(data(100),data(200),confirmation=data(100))
  assert report['review_required'] and not report['repeated_regressions']


@pytest.mark.parametrize('value',[float('nan'),float('inf'),0,-1,True])
def test_nonfinite_and_invalid_timings(value):
  with pytest.raises(ValueError):compare(data(),data(value))


def test_scope_and_metadata_drift():
  with pytest.raises(ValueError):compare(data(),{})
  with pytest.raises(ValueError):compare(data(),{'a':{'metadata':{'m':8},'dsl':100,'cpp':100}})


def write_main(path, status='passed'):
  path.mkdir()
  (path/'plan.json').write_text(json.dumps({'architecture':'arm','executions':[{'execution_id':'a','m':4}]}))
  row={'execution_id':'a','build':'built','stages':{'compute':{'status':status,'median_ns':{'candidate':100,'cpp':90}}}}
  (path/'ledger.json').write_text(json.dumps({'architecture':'arm','rows':[row]}))
  return row


def test_main_admission_unknown_duplicate_and_architecture(tmp_path):
  row=write_main(tmp_path/'suite');identity={'architecture':'arm','execution_ids':['a']}
  assert load_main(tmp_path/'suite',identity,'compute')['a']['dsl']==100
  for ids in (['missing'],):
    with pytest.raises(ValueError):load_main(tmp_path/'suite',{**identity,'execution_ids':ids},'compute')
  with pytest.raises(ValueError):load_main(tmp_path/'suite',{**identity,'architecture':'x86'},'compute')
  with pytest.raises(ValueError):load_main(tmp_path/'suite',identity,'packing')
  (tmp_path/'suite/ledger.json').write_text(json.dumps({'architecture':'arm','rows':[row,row]}))
  with pytest.raises(ValueError,match='duplicate'):load_main(tmp_path/'suite',identity,'compute')


def test_failed_main_and_cli_self_compare(tmp_path):
  write_main(tmp_path/'old');write_main(tmp_path/'new','failed')
  identity={'architecture':'arm','execution_ids':['a']}
  with pytest.raises(ValueError,match='unpassed'):load_main(tmp_path/'new',identity,'compute')
  selection=tmp_path/'ids.json';selection.write_text(json.dumps(identity))
  output=tmp_path/'out.json'
  assert main(['--kind','main','--architecture','arm','--selection',str(selection),'--baseline',str(tmp_path/'old'),
    '--candidate',str(tmp_path/'old'),'--output',str(output)])==0
  assert json.loads(output.read_text())['metrics']['dsl_new_over_old']['geomean']==1


def test_supplemental_rejects_unadmitted_and_uncovered_chain(tmp_path):
  path=tmp_path/'timing-results.json';identity={'architecture':'arm','registration_ids':['a']}
  path.write_text(json.dumps({'rows':[]}))
  with pytest.raises(ValueError,match='admissions'):load_supplemental(path,identity)
  path.write_text(json.dumps({'completed':True,'batch_admissions':[{}],'timing_chains':[{}],'rows':[]}))
  with pytest.raises(ValueError,match='pre/pending'):load_supplemental(path,identity)


def test_supplemental_complete_chain_and_tamper(tmp_path):
  import hashlib
  def write(name, doc):
    path=tmp_path/name;path.write_text(json.dumps(doc));return path,hashlib.sha256(path.read_bytes()).hexdigest()
  binary=tmp_path/'kernel.so';binary.write_bytes(b'fixture')
  artifacts={role:{'binary':str(binary),'binary_sha256':hashlib.sha256(binary.read_bytes()).hexdigest()} for role in ('dsl','cpp')}
  row={'registration':{'registration_id':'a','architecture':'arm','side':'B'},'validated':True,'timed':True,
    'median_ns':{'dsl':100,'cpp':90},'artifacts':artifacts}
  pre,pre_sha=write('pre.json',{'inputs':{},'registrations':['a'],'oracle_artifacts':{'a':artifacts}})
  pending,pending_sha=write('pending.json',{'admitted':False,'pre_sha256':pre_sha,'inputs':{},'rows':[row]})
  checkpoint,checkpoint_sha=write('checkpoint.json',{'policies':['fixture'],'events':[{'phase':'post_build'}],'sources':[]})
  _,manifest_sha=write('manifest.json',{'artifacts':[]})
  chain={'pre':str(pre),'pre_sha256':pre_sha,'pending':str(pending),'pending_sha256':pending_sha,'inputs':{}}
  doc={'completed':True,'rows':[row],'timing_chains':[chain],'batch_admissions':[{'directory':str(tmp_path),
    'checkpoint':str(checkpoint),'checkpoint_sha256':checkpoint_sha,'manifest_sha256':manifest_sha,'registrations':['a']}]}
  path,_=write('timing-results.json',doc);identity={'architecture':'arm','registration_ids':['a']}
  assert load_supplemental(path,identity)['a']['cpp']==90
  doc['rows']=[row,row];write('timing-results.json',doc)
  with pytest.raises(ValueError):load_supplemental(path,identity)
  doc['rows']=[row];write('timing-results.json',doc)
  binary.write_bytes(b'changed')
  with pytest.raises(ValueError,match='binary'):load_supplemental(path,identity)


def test_cpp_only_regression_also_requires_review():
  report=compare(data(100,100),data(100,200),confirmation=data(100,201))
  assert report['review_required'] and report['repeated_regressions']==['a']
  assert report['rows'][0]['regressed_roles']==['cpp']
  assert report['rows'][0]['relative_dsl_cpp_new_over_old']==.5


@pytest.mark.parametrize('selected',[['a','a'],[],['a','other']])
def test_main_plan_scope_must_match_ledger(tmp_path,selected):
  write_main(tmp_path/'suite')
  p=tmp_path/'suite/plan.json';doc=json.loads(p.read_text());doc['selected_execution_ids']=selected;p.write_text(json.dumps(doc))
  with pytest.raises(ValueError,match='scope'):load_main(tmp_path/'suite',{'architecture':'arm','execution_ids':['a']},'compute')
