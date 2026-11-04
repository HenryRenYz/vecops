# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Validate a completed matched-CPU lx-run probe and summarize five rounds."""
import argparse
import json
import math
import hashlib
import re
from pathlib import Path
import statistics

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('directory',type=Path)
parser.add_argument('--output',type=Path)
args=parser.parse_args()
if not (args.directory/'exitcode').exists() or (args.directory/'exitcode').read_text().strip()!='0':
  raise ValueError('Job has not completed successfully')
metadata=[];samples=[];source_hashes=[]
for path in sorted(args.directory.glob('rank-*.stdout.log')):
  for line in path.read_text().splitlines():
    if match:=re.fullmatch(r'([0-9a-f]{64})\s+.*mpi_proxy_probe\.cpp',line):source_hashes.append(match.group(1))
    if not line.startswith('{"type":'):continue
    row=json.loads(line)
    if row['type']=='metadata':metadata.append(row)
    elif row['type']=='sample':samples.append(row)
source=args.directory/'mpi_proxy_probe.cpp'
if not source.exists():source=Path(__file__).with_name('mpi_proxy_probe.cpp')
expected_hash=hashlib.sha256(source.read_bytes()).hexdigest()
if len(source_hashes)!=2 or set(source_hashes)!={expected_hash}:
  raise ValueError('Missing or changed benchmark source hashes')
if len(metadata)!=2 or {r['rank'] for r in metadata}!={0,1} or len({r['host'] for r in metadata})!=2:
  raise ValueError('Need exactly two distinct compute nodes/ranks')
if any(len(r['allowed_cpus'])!=2 or r['mpi_thread_level']<2 or r.get('mpi_caller_cpu')!=r['allowed_cpus'][1] for r in metadata):
  raise ValueError('Need matched MPI caller CPU and SERIALIZED support')
if any(r['iterations']!=max(8,min(4096,(2<<30)//r['bytes_per_remote_peer'])) for r in samples):
  raise ValueError('Wrong iteration count; exploratory or old probe must not be mixed')
if len(samples)!=210 or any(not r['correct'] or not math.isfinite(r['seconds']) or r['seconds']<=0 for r in samples):
  raise ValueError('Incomplete or invalid sample set')
records=[]
for op in ['sendrecv','allgather','alltoall']:
  for size in [1,64,4096,65536,1<<20,16<<20,64<<20]:
    record={'operation':op,'bytes_per_remote_peer':size}
    for mode in ['direct_mpi','spsc_proxy']:
      group=[r for r in samples if r['operation']==op and r['bytes_per_remote_peer']==size and r['mode']==mode]
      if len(group)!=5 or {r['round'] for r in group}!=set(range(5)):
        raise ValueError('Missing or duplicated control rounds')
      seconds=[r['seconds']/r['iterations'] for r in group]
      mean=statistics.mean(seconds)
      record[mode]={'median_us':statistics.median(seconds)*1e6,
                    'median_remote_GBps':statistics.median(size/t/1e9 for t in seconds),
                    'cv_percent':100*statistics.stdev(seconds)/mean,
                    'round_us':[t*1e6 for t in seconds]}
    record['proxy_extra_us']=record['spsc_proxy']['median_us']-record['direct_mpi']['median_us']
    record['proxy_throughput_ratio']=record['spsc_proxy']['median_remote_GBps']/record['direct_mpi']['median_remote_GBps']
    records.append(record)
result={'scope':'2 nodes, 1 MPI rank/node, 2 CPUs/rank, DDR0; same MPI caller physical CPU in both modes; in-process SPSC proxy, not production cross-process proxy; caller/idle thread deliberately busy-spins; 5 block medians, not per-call p99; no computation-overlap test; no Torch MPI comparison','source_sha256':expected_hash,'metadata':metadata,'samples':len(samples),'records':records}
if args.output:args.output.write_text(json.dumps(result,indent=2)+'\n')
for row in records:
  if row['bytes_per_remote_peer'] in [1,64<<20]:
    print(row['operation'],row['bytes_per_remote_peer'],json.dumps(row))
