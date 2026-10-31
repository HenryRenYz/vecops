# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Repeat insufficient cases with a longer window, never select by speed."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import time
from RunSdmaCampaign import stamp, sha, read
from SummarizeSdmaCampaign import eligible


def main():
    p=argparse.ArgumentParser();p.add_argument('binary',type=Path);p.add_argument('root',type=Path);a=p.parse_args()
    root=a.root.resolve();binary=a.binary.resolve();path=root/'manifest.json'
    m=json.loads(path.read_text())
    if sha(binary)!=m['binary_sha256'] or platform.node()!=m['host']:
        raise RuntimeError('Repair requires the identical binary and compute node')
    if sha('/usr/lib64/libsdma_dk.so')!=m['sdk_sha256'] or read('/sys/module/sdma_dae/parameters/share_chns')!=m['share_chns'] or read('/sys/module/sdma_dae/parameters/safe_mode')!=m['safe_mode']:
        raise RuntimeError('SDK/driver conditions changed')
    allowed=os.sched_getaffinity(0)
    if set(m['affinity'])!=allowed:raise RuntimeError('Repair allocation CPU mask changed')
    os.sched_setaffinity(0,{606})
    def child():os.sched_setaffinity(0,allowed)
    history=dict(started=stamp(),host=platform.node(),script_sha256=sha(__file__))
    m.setdefault('repair_history',[]).append(history)
    def save():
        temporary=path.with_suffix('.new');temporary.write_text(json.dumps(m,indent=2));temporary.replace(path)
    def rows(c):return list(csv.DictReader((root/c['csv']).open(newline='')))
    def total(c):return [r for r in rows(c) if r['scope']=='total']
    # Add symmetric local-memory controls for each driver-recommended device.
    # The main fixed-pool sweep used source nodes 8/24 for all visible devices.
    plan_path=root/'plan.json';plan=json.loads(plan_path.read_text())
    first_node={}
    for node,device in sorted(m['recommended_devices'].items(),key=lambda x:int(x[0])):
        first_node.setdefault(int(device),int(node))
    extensions=[]
    for device,node in first_node.items():
        if node==8:continue
        templates=[c for c in m['cases'] if c['family']=='saturation' and int(c['devices'])==device and not c.get('repair_rank')]
        for original in templates:
            case=dict(original);case['id']='x-local-'+original['id'];case['name']='local-'+original['name']
            case['family']='local-saturation';case['local_execution_node']=node
            route=','.join(':'.join(str(node if int(x)==8 else node+16 if int(x)==24 else int(x)) for x in r.split(':')) for r in original['routes'].split(','))
            command=list(original['command']);command[command.index('--numa')+1]=route
            command[command.index('--exec-node')+1]=str(node)
            case['routes']=route;case['command']=command
            case['csv']=case['id']+'.csv'
            extensions.append(case)
    # Exercise uneven channel ownership and non-cache-line-aligned array tails.
    template=next(c for c in m['cases'] if c['family']=='saturation' and int(c['devices'])==m['recommended_devices']['8'])
    for route in ['8:24','24:24']:
        case=dict(template);case['id']='x-edge-odd-'+route;case['name']='edge-odd-channels-'+route
        case.update(family='correctness',routes=route,controllers='0-1',size='65540',channels=3,
                    depth=2,batch=1,chunk='16KiB',seconds=.05,samples=2)
        command=list(template['command'])
        replacements={'--numa':route,'--exec-node':'8','--cpu':'0-1','--size':'65540',
            '--size-scope':'thread-array','--sdma-channels':'3','--sdma-depth':'2',
            '--sdma-batch':'1','--sdma-chunk':'16KiB','--seconds':'.05','--samples':'2'}
        for key,value in replacements.items():command[command.index(key)+1]=value
        case['command']=command;case['csv']=case['id']+'.csv';extensions.append(case)
    known_plan={c['id'] for c in plan}
    for c in extensions:
        if c['id'] not in known_plan:plan.append(c)
    plan_path.write_text(json.dumps(plan,indent=2));m['planned_cases']=len(plan)
    for c in extensions:
        if any(old['id']==c['id'] for old in m['cases']):continue
        print(stamp(),'LOCAL CONTROL',c['id'],flush=True)
        start=time.monotonic()
        with (root/(c['id']+'.log')).open('w') as log:
            r=subprocess.run(c['command']+['--csv',str(root/c['csv'])],stdout=log,stderr=subprocess.STDOUT,timeout=240,preexec_fn=child)
        c.update(exit=r.returncode,wall_seconds=time.monotonic()-start,finished=stamp(),csv_sha256=sha(root/c['csv']))
        m['cases'].append(c);save()
        if r.returncode not in [0,1]:raise RuntimeError('Local control failed: '+c['id'])
    for round_id in [1,2]:
        originals=[c for c in m['cases'] if not c.get('repair_rank') and c['family']!='correctness']
        for original in originals:
            own=[c for c in m['cases'] if c['name']==original['name']]
            latest=max(own,key=lambda c:c.get('repair_rank',0))
            observations=total(latest)
            if sum(eligible(r) for r in observations)>=3:continue
            if latest.get('repair_rank',0)>=round_id:continue
            spread=max(max(int(r['start_skew_ns']),int(r['end_skew_ns'])) for r in observations)
            duration=min(60,max(2,latest['seconds']*2,math.ceil(spread*1.5e-6)))
            case=dict(original);case.update(id=f'r{round_id}-'+original['id'],repair_rank=round_id,
                replaces=original['id'],seconds=duration,samples=7,repair_reason='fewer than 3 synchronized fault-free observations')
            command=list(original['command'])
            command[command.index('--seconds')+1]=str(duration)
            command[command.index('--samples')+1]='7'
            command[command.index('--warmup')+1]='.1'
            case['command']=command;case['csv']=case['id']+'.csv'
            print(stamp(),'REPAIR',case['id'],'seconds',duration,flush=True)
            start=time.monotonic()
            with (root/(case['id']+'.log')).open('w') as log:
                r=subprocess.run(command+['--csv',str(root/case['csv'])],stdout=log,stderr=subprocess.STDOUT,timeout=900,preexec_fn=child)
            case.update(exit=r.returncode,wall_seconds=time.monotonic()-start,finished=stamp(),csv_sha256=sha(root/case['csv']))
            m['cases'].append(case);save()
            if r.returncode not in [0,1]:raise RuntimeError('Repair failed: '+case['id'])
    history['finished']=stamp();save()
    print('REPAIR COMPLETE',flush=True)


if __name__=='__main__':main()
