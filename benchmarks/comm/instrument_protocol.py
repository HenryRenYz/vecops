# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Build an opt-in private source copy; never alter production timing paths.

Records the same collective epoch on every rank, blocked wait time, copy
submissions including caller-only work, and actual mapping/capacity changes.
Call comm_diag_start/stop outside inference; JSON writes are outside timing.
"""
from pathlib import Path
import argparse
import shutil

HEADER = r'''
// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
// Private diagnostic instrumentation, deliberately absent from production.
#include <cstdio>
namespace {
using DU=unsigned long long;
DU dn(){return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();}
struct DP{unsigned label;DU ns;};
struct DC{DU begin,end,bytes,strided,rows,minrun,maxrun,tasks,lanes,kind;};
struct DR{DU context,epoch,descriptor,op,rank,size,borrowed,input_bytes,output_bytes;
 DU map_checks=0,mmap_calls=0,growths=0,descriptor_checks=0,wait_calls=0,blocked_waits=0,wait_yields=0;
 unsigned np=0,nc=0;DP points[48]{};DC copies[20]{};DU waits[48]{};};
bool denabled=false;std::vector<DR> drecs;thread_local DR* dcurrent=nullptr;
void dp(unsigned label){if(dcurrent && dcurrent->np<48)dcurrent->points[dcurrent->np++]={label,dn()};}
struct DScope{DR r{};bool active=false;DScope(){active=denabled;if(active){dcurrent=&r;dp(0);}}~DScope(){if(active){dp(999);drecs.push_back(r);dcurrent=nullptr;}}};
void dc_save(DC& c){if(dcurrent && dcurrent->nc<20){c.end=dn();dcurrent->copies[dcurrent->nc++]=c;}}
}
extern "C" void comm_diag_start(){drecs.clear();drecs.reserve(30000);denabled=true;}
extern "C" void comm_diag_stop(const char* path){denabled=false;FILE* f=fopen(path,"w");if(!f)return;fputs("[",f);bool sep=false;for(auto& r:drecs){
fprintf(f,"%s{\"context\":%llu,\"epoch\":%llu,\"descriptor\":%llu,\"op\":%llu,\"rank\":%llu,\"size\":%llu,\"borrowed\":%llu,\"input_bytes\":%llu,\"output_bytes\":%llu,\"map_checks\":%llu,\"mmap_calls\":%llu,\"growths\":%llu,\"descriptor_checks\":%llu,\"wait_calls\":%llu,\"blocked_waits\":%llu,\"wait_yields\":%llu,\"points\":[",sep?",":"",r.context,r.epoch,r.descriptor,r.op,r.rank,r.size,r.borrowed,r.input_bytes,r.output_bytes,r.map_checks,r.mmap_calls,r.growths,r.descriptor_checks,r.wait_calls,r.blocked_waits,r.wait_yields);sep=true;
for(unsigned i=0;i<r.np;++i)fprintf(f,"%s[%u,%llu,%llu]",i?",":"",r.points[i].label,r.points[i].ns,r.waits[i]);
fputs("],\"copies\":[",f);for(unsigned i=0;i<r.nc;++i){auto& c=r.copies[i];fprintf(f,"%s[%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu]",i?",":"",c.begin,c.end,c.bytes,c.strided,c.rows,c.minrun,c.maxrun,c.tasks,c.lanes,c.kind);}fputs("]}",f);}fputs("]\n",f);fclose(f);}
'''

def instrument(text):
  def replace(old,new,count=1):
    nonlocal text
    if text.count(old)!=count:raise ValueError(f'instrumentation anchor changed ({text.count(old)} != {count}): {old[:90]!r}')
    text=text.replace(old,new,count)
  replace('namespace vecops::comm {',HEADER+'\nnamespace vecops::comm {')
  replace('    if (mapped == bytes) return;', '    if(dcurrent)++dcurrent->map_checks;\n    if (mapped == bytes) return;\n    if(dcurrent)++dcurrent->mmap_calls;')
  replace('    ++stats.capacity_growths;', '    ++stats.capacity_growths;if(dcurrent)++dcurrent->growths;')
  replace('    size_t spin=0;\n    while (load(token)<expected)', '    if(dcurrent)++dcurrent->wait_calls;\n    DU ds=dcurrent && load(token)<expected?dn():0;\n    if(ds)++dcurrent->blocked_waits;\n    size_t spin=0;\n    while (load(token)<expected)')
  replace('        if ((spin&65535)==0) std::this_thread::yield();', '        if ((spin&65535)==0){if(dcurrent)++dcurrent->wait_yields;std::this_thread::yield();}')
  replace('    }\n  }\n  void reserve(size_t required)', '    }\n    if(ds && dcurrent->np)dcurrent->waits[dcurrent->np-1]+=dn()-ds;\n  }\n  void reserve(size_t required)')
  replace('    Batch b{&jobs,tasks,copy};', '''    DC dc{};dc.begin=dcurrent?dn():0;dc.minrun=~DU(0);dc.kind=1;
    if(dcurrent)for(const auto& j:jobs){dc.bytes+=j.shape->bytes;dc.minrun=std::min<DU>(dc.minrun,j.shape->run);dc.maxrun=std::max<DU>(dc.maxrun,j.shape->run);if(j.shape->offsets.size()>1){dc.strided+=j.shape->bytes;dc.rows+=j.shape->offsets.size();}}
    Batch b{&jobs,tasks,copy};''')
  replace('    } else Batch::work(&b,0,1);','    } else Batch::work(&b,0,1);\n    if(dcurrent){dc.tasks=tasks;dc.lanes=executor?lanes:1;dc_save(dc);}')
  replace('    const uint32_t lanes=std::min<size_t>(options.threads,rows);','    const uint32_t lanes=std::min<size_t>(options.threads,rows);\n    DC dc{};dc.begin=dcurrent?dn():0;dc.kind=2;')
  replace('    stats.copy_bytes+=bytes;stats.row_copy_bytes+=bytes;++stats.row_copy_calls;', '    stats.copy_bytes+=bytes;stats.row_copy_bytes+=bytes;++stats.row_copy_calls;\n    if(dcurrent){dc.bytes=bytes;dc.strided=bytes;dc.rows=rows;dc.minrun=dc.maxrun=run;dc.tasks=rows;dc.lanes=executor?lanes:1;dc_save(dc);}')
  replace('  auto& p=*state_; auto& c=*p.owner->state_;\n  std::lock_guard lock(c.mutex);\n  try {','''  DScope diagnostic;
  auto& p=*state_; auto& c=*p.owner->state_;
  std::lock_guard lock(c.mutex);dp(1);
  if(dcurrent){auto& d=*dcurrent;d.context=reinterpret_cast<DU>(&c);d.epoch=c.epoch+1;d.descriptor=p.descriptor_id;d.op=static_cast<DU>(p.spec.operation);d.rank=c.rank;d.size=c.size;d.borrowed=consumer!=nullptr;for(auto& x:p.inputs)d.input_bytes+=x.bytes;for(auto& x:p.outputs)d.output_bytes+=x.bytes;}
  try {''')
  replace('    c.reserve(p.required);','    dp(2);c.reserve(p.required);')
  replace('    for(auto& peer:c.peers) c.wait(&peer.header->arrived,epoch);','    dp(3);for(auto& peer:c.peers) c.wait(&peer.header->arrived,epoch);dp(4);')
  replace('      if(p.validated_descriptors[peer]!=h->descriptor) {','      if(p.validated_descriptors[peer]!=h->descriptor) {\n        if(dcurrent)++dcurrent->descriptor_checks;')
  replace('    p.jobs.clear();\n    if(op==Operation::AllToAll)','    dp(5);p.jobs.clear();\n    if(op==Operation::AllToAll)')
  replace('    c.batch(p.jobs);store(&own->ready,epoch);\n    for(auto& peer:c.peers)c.wait(&peer.header->ready,epoch);','    c.batch(p.jobs);store(&own->ready,epoch);dp(6);\n    for(auto& peer:c.peers)c.wait(&peer.header->ready,epoch);dp(7);')
  replace('      c.batch(p.jobs);store(&own->stages[level],epoch);','      c.batch(p.jobs);store(&own->stages[level],epoch);dp(100+level*2);')
  replace('      for(auto& peer:c.peers)c.wait(&peer.header->stages[level],epoch);','      for(auto& peer:c.peers)c.wait(&peer.header->stages[level],epoch);dp(101+level*2);')
  replace('    p.jobs.clear();\n    if(op==Operation::AllGather)','    dp(200);p.jobs.clear();\n    if(op==Operation::AllGather)')
  replace('    if(consumer) (*consumer)(p.borrowed.data(),p.borrowed.size());','    dp(201);if(consumer) (*consumer)(p.borrowed.data(),p.borrowed.size());dp(202);')
  replace('    store(&own->done,epoch);++c.stats.calls;','    store(&own->done,epoch);++c.stats.calls;dp(203);')
  return text

def main():
  p=argparse.ArgumentParser();p.add_argument('--source-checkout',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
  if a.output.exists():raise FileExistsError(a.output)
  for part in ('include','src/comm','python/vecops/distributed','tests/comm'):
    shutil.copytree(a.source_checkout/part,a.output/part,ignore=shutil.ignore_patterns('__pycache__'))
  f=a.output/'src/comm/Communication.cpp';f.write_text(instrument(f.read_text()))
  print(a.output)
if __name__=='__main__':main()
