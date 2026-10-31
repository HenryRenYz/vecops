# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Audit DMA raw accounting and summarize every retained scope without peak picking."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import platform
import subprocess
import sys
from datetime import datetime, timezone, timedelta


def eligible(r):
    target=float(r['target_seconds'])*1e9
    return (r['status'] in {'ok','interrupted'} and r['placement']=='all-pages-verified'
            and int(r['minor_faults'])==0 and int(r['major_faults'])==0
            and int(r['start_skew_ns'])<=target*.001 and int(r['end_skew_ns'])<=target*.001
            and (r['scope'].startswith('thread:') or int(r['common_ns'])>=target*.998))


def audit(manifest, root):
    summaries=[]; errors=[]; count=0; rejected={}; compiled={}
    def require(condition, label):
        if not condition: errors.append(label)
    for c in manifest['cases']:
        path=root/c['csv']
        require(path.exists(),c['id']+': missing CSV')
        if not path.exists():continue
        require(hashlib.sha256(path.read_bytes()).hexdigest()==c['csv_sha256'],c['id']+': CSV hash')
        rows=list(csv.DictReader(path.open(newline='')))
        require(c['exit'] in [0,1],c['id']+': process exit')
        require(bool(rows),c['id']+': empty rows')
        count+=len(rows)
        for r in rows:
            label=c['id']+':'+r['scope']+':'+r['sample']
            if not compiled:
                compiled={k:r[k] for k in ['compiler','tool_source_sha256','base_page_bytes','backend','cpu_model']}
            require(all(r[k]==v for k,v in compiled.items()),label+': compiled environment')
            duration=float(r['elapsed_seconds']);payload=int(r['write_bytes'])
            require(r['host']==manifest['host'],label+': original compute node')
            require(duration>0,label+': positive duration')
            require(payload==int(r['read_bytes']) and payload>0,label+': copy bytes')
            require(math.isclose(duration,(int(r['end_ns'])-int(r['start_ns']))*1e-9,rel_tol=1e-10),label+': elapsed envelope')
            require(math.isclose(float(r['payload_GB_s']),payload/duration/1e9,rel_tol=1e-10),label+': payload rate')
            require(math.isclose(float(r['total_GB_s']),2*payload/duration/1e9,rel_tol=1e-10),label+': read+write rate')
            require(r['placement']=='all-pages-verified',label+': placement/data verification')
            if r['scope'].startswith('thread:'):
                page=2*1024*1024 if r['page_policy']=='huge' else int(r['base_page_bytes'])
                pages=(int(r['per_thread_array_bytes'])+page-1)//page
                require(int(r['working_set_bytes'])==2*int(r['per_thread_array_bytes']),label+': private copy arrays')
                routes=[list(map(int,x.split(':'))) for x in c['routes'].split(',')]
                for role,column in [(0,'source'),(1,'destination')]:
                    nodes=sorted({pair[role] for pair in routes}) if c['interleave'] else [routes[int(r['group'])][role]]
                    require(r[column+'_memory_nodes']==','.join(map(str,nodes)),label+': requested '+column+' pool')
                    expected={n:pages//len(nodes)+(i<pages%len(nodes)) for i,n in enumerate(nodes)}
                    actual={int(x.split(':')[0]):int(x.split(':')[1]) for x in r[column+'_page_counts'].split('|')}
                    require(actual==expected,label+': private '+column+' page counts')
            if c['engine']=='sdma':
                require(r['engine']=='sdma' and r['schema_version']=='3',label+': DMA schema')
                require(int(r['completed_descriptors'])==int(r['submissions'])*int(r['sdma_batch']),label+': completed submissions')
                require(int(r['completed_descriptors'])>0,label+': completed work')
                require(int(r['drain_ns'])<=int(r['end_ns'])-int(r['start_ns']),label+': drain envelope')
                require(0<=float(r['cpu_seconds'])<=duration*int(r['threads'])*1.05+.001,label+': CPU time')
            if not eligible(r):rejected[r['status']]=rejected.get(r['status'],0)+1
        # Accounting and envelope checks are independent of the implementation.
        for total in [r for r in rows if r['scope']=='total' or r['scope'].startswith('group:')]:
            workers=[r for r in rows if r['scope'].startswith('thread:')
                     and all(r[k]==total[k] for k in ['config','sample','attempt'])
                     and (total['scope']=='total' or r['group']==total['group'])]
            label=c['id']+':'+total['scope']+':'+total['sample']
            require(bool(workers),label+': workers')
            if not workers:continue
            for key in ['read_bytes','write_bytes','completed_descriptors','submissions','sdma_channels','working_set_bytes','threads','minor_faults','major_faults','context_switches']:
                require(int(total[key])==sum(int(r[key]) for r in workers),label+': sum '+key)
            require(int(total['start_ns'])==min(int(r['start_ns']) for r in workers),label+': earliest start')
            require(int(total['end_ns'])==max(int(r['end_ns']) for r in workers),label+': latest completion')
            for key in ['source_page_counts','destination_page_counts']:
                def counts(v):return dict((int(x.split(':')[0]),int(x.split(':')[1])) for x in v.split('|') if x)
                actual={}
                for r in workers:
                    for node,n in counts(r[key]).items():actual[node]=actual.get(node,0)+n
                require(counts(total[key])==actual,label+': page sums')
        for scope in sorted(set(r['scope'] for r in rows)):
            if scope.startswith('thread:'):continue
            own=[r for r in rows if r['scope']==scope]
            valid=[r for r in own if eligible(r)]
            result={k:v for k,v in c.items() if k!='command'}
            result['size_scope']=c['command'][c['command'].index('--size-scope')+1]
            result.update(scope=scope,recorded=len(own),usable=len(valid),strict=sum(r['status']=='ok' for r in valid))
            for key in ['payload_GB_s','total_GB_s','elapsed_seconds','cpu_seconds','setup_seconds','drain_ns']:
                values=[float(r[key]) for r in valid]
                result[key]=statistics.median(values) if values else None
                if key=='payload_GB_s':
                    result['minimum']=min(values) if values else None
                    result['maximum']=max(values) if values else None
            if valid:
                result['controller_equivalent_cores']=statistics.median(float(r['cpu_seconds'])/float(r['elapsed_seconds']) for r in valid) if c['engine']=='sdma' else None
                result['cpu_seconds_per_GiB']=statistics.median(float(r['cpu_seconds'])/(int(r['write_bytes'])/1024**3) for r in valid) if c['engine']=='sdma' else None
                result['threads']=int(valid[0]['threads']);result['source_nodes']=valid[0]['source_memory_nodes'];result['destination_nodes']=valid[0]['destination_memory_nodes']
            summaries.append(result)
    if manifest.get('finished') and (root/'plan.json').exists():
        plan=json.loads((root/'plan.json').read_text())
        originals={c['id']:c for c in manifest['cases'] if not c.get('repair_rank')}
        require(set(originals)=={c['id'] for c in plan},'planned case coverage')
        for c in plan:
            if c['id'] in originals:require(originals[c['id']]['command']==c['command'],c['id']+': planned command')
    selected={}
    for r in summaries:
        key=(r['name'],r['scope'])
        if key not in selected or r.get('repair_rank',0)>selected[key].get('repair_rank',0):selected[key]=r
    return dict(selected_summaries=list(selected.values()),environment={**{k:v for k,v in manifest.items() if k!='cases'},**compiled},cases=len(manifest['cases']),
                raw_rows=count,rejected_status_counts=rejected,audit_errors=errors,summaries=summaries)


def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--repair',action='store_true');args=p.parse_args()
    manifest=json.loads((args.root/'manifest.json').read_text())
    if platform.node()==manifest['host']:
        evidence={}
        for n in range(32):
            p=Path(f'/sys/devices/system/node/node{n}/distance')
            if p.exists():evidence[str(n)]=p.read_text().strip()
        manifest['numa_distance_evidence']=evidence
        (args.root/'manifest.json').write_text(json.dumps(manifest,indent=2))
    result=audit(manifest,args.root)
    repair=args.repair or ((args.root/'repair.enabled').exists() and platform.node()==manifest['host'])
    poor=[r for r in result['selected_summaries'] if r['scope']=='total' and r['family']!='correctness' and r['usable']<3]
    if repair and (poor or not any(c['family']=='local-saturation' for c in manifest['cases'])):
        if platform.node()!=manifest['host']:raise RuntimeError('Repairs must run in the original compute allocation')
        if result['audit_errors']:raise RuntimeError('Fix accounting errors before repairing sample quality')
        subprocess.run([sys.executable,str(Path(__file__).with_name('RepairSdmaCampaign.py')),manifest['binary'],str(args.root)],check=True)
        manifest=json.loads((args.root/'manifest.json').read_text())
        result=audit(manifest,args.root)
    (args.root/'analysis.json').write_text(json.dumps(result,indent=2))
    if result['summaries']:
        with (args.root/'summary.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=sorted(set().union(*(r.keys() for r in result['summaries']))))
            writer.writeheader();writer.writerows(result['summaries'])
    main_rows=[r for r in result['selected_summaries'] if r['scope']=='total' and r['family']!='correctness']
    print(json.dumps(dict(cases=result['cases'],rows=result['raw_rows'],errors=result['audit_errors'][:20],
                         sufficient=sum(r['usable']>=3 for r in main_rows),insufficient=sum(r['usable']<3 for r in main_rows)),indent=2))
    if result['audit_errors']:raise SystemExit(1)
    history=manifest.get('repair_history',[])
    ready=bool(manifest.get('finished')) and (not (args.root/'repair.enabled').exists() or bool(history and all(h.get('finished') for h in history)))
    if ready:
        (args.root/'audit-complete.json').write_text(json.dumps(dict(binary_sha256=manifest['binary_sha256'],
            cases=result['cases'],raw_rows=result['raw_rows'],finished=datetime.now(timezone(timedelta(hours=8))).isoformat()),indent=2))


if __name__=='__main__':main()
