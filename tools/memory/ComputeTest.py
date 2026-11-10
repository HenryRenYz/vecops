# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Native integration checks: BF16 tails, LN affine/constant rows, unequal shards,
strict placement, independent operator references, and global timing accounting.
"""
import argparse, collections, csv, json, math, subprocess
from pathlib import Path


def main():
    p=argparse.ArgumentParser()
    p.add_argument('binary',type=Path)
    p.add_argument('--exec-node',default='0')
    p.add_argument('--nodes',default='0')
    p.add_argument('--cpus',default='0-2')
    p.add_argument('--output-dir',type=Path,required=True)
    args=p.parse_args();args.output_dir.mkdir(parents=True,exist_ok=True)
    binary=str(args.binary.resolve());nodes=sorted(set(map(int,args.nodes.split(','))))
    common=[binary,'compute','--numa',str(nodes[0]),'--exec-node',args.exec_node,'--cpu',args.cpus,
            '--input-nodes',args.nodes,'--output-nodes',args.nodes,'--workspace-node',str(nodes[0]),
            '--interleave','--pages','base','--seconds','.02','--warmup','.03','--samples','2',
            '--retries','20','--max-skew','.02']
    for name,op,shape,extra in [('gemm-tail','gemm','37,23,35',[]),('gemm-sampled','gemm','96,64,2048',[]),
                               ('ln-tail','ln','13,37',[]),('ln-no-affine','ln','13,37',['--ln-affine','off']),
                               ('ln-constant-row','ln','13,1',[])]:
        path=args.output_dir/(name+'.csv');jp=path.with_suffix('.jsonl')
        cmd=common+['--op',op,'--shape',shape,'--csv',str(path),'--json',str(jp),*extra]
        result=subprocess.run(cmd,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=180)
        path.with_suffix('.log').write_text(result.stdout)
        assert result.returncode==0,result.stdout
        rows=list(csv.DictReader(path.open()));assert rows==[json.loads(x) for x in jp.read_text().splitlines()]
        batches=collections.defaultdict(list)
        for r in rows:
            assert r['schema_version']=='4' and r['placement']=='all-pages-verified'
            assert r['verification'] and r['shape']==shape
            batches[(r['sample'],r['attempt'])].append(r)
        totals=[]
        for rs in batches.values():
            total=next(r for r in rs if r['scope']=='total');workers=[r for r in rs if r['scope'].startswith('thread:')]
            spans=sorted((int(r['row_begin']),int(r['row_end'])) for r in workers)
            assert spans[0][0]==0 and spans[-1][1]==int(shape.split(',')[0])
            assert all(a<b for a,b in spans) and all(a[1]==b[0] for a,b in zip(spans,spans[1:]))
            assert len(workers)==int(total['threads'])
            assert all(r['iterations']==total['iterations'] for r in workers)
            start=min(int(r['start_ns']) for r in workers);end=max(int(r['end_ns']) for r in workers)
            assert int(total['start_ns'])==start and int(total['end_ns'])==end
            seconds=(end-start)/1e9;iterations=int(total['iterations'])
            assert math.isclose(float(total['operation_seconds']),seconds/iterations,rel_tol=1e-9)
            dims=list(map(int,shape.split(',')))
            metric=2*dims[0]*dims[1]*dims[2] if op=='gemm' else dims[0]*dims[1]
            assert math.isclose(float(total['GFLOP_s' if op=='gemm' else 'GElements_s']),metric*iterations/seconds/1e9,rel_tol=1e-9)
            assert all(not r['source_page_counts'] and not r['destination_page_counts'] for r in workers)
            if total['status']=='ok':
                assert all(r['status']=='ok' for r in rs)
                assert all(int(total[k])==0 for k in ('context_switches','minor_faults','major_faults'))
                totals.append(total)
        assert len(totals)==2
        assert all(float(r['warmup_actual_seconds'])>=.03 for r in totals)
        print('PASS',name,'disjoint row coverage, scalar references, placement and envelope accounting')
    for extra in [[],['--op','gemm','--shape','1,2'],['--op','ln','--shape','1,2,3'],
                  ['--op','ln','--shape','4,0'],['--op','ln','--shape','4,2','--size','1MiB'],
                  ['--op','gemm','--shape','7,4,4','--access','stream'],
                  ['--op','ln','--shape','1,4'],['--op','ln','--shape','4,2','--ln-eps','0']]:
        result=subprocess.run(common+extra+['--dry-run'],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=20)
        assert result.returncode==2,(extra,result.stdout.decode())
    print('PASS invalid compute arguments')

if __name__=='__main__':main()
