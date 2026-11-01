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
struct DScope{DR r{};bool active=false;DScope(bool lane=true){active=denabled && lane;if(active){dcurrent=&r;dp(0);}}~DScope(){if(active){dp(999);drecs.push_back(r);dcurrent=nullptr;}}};
void dc_save(DC& c){if(dcurrent && dcurrent->nc<20){c.end=dn();dcurrent->copies[dcurrent->nc++]=c;}}
}
extern "C" void comm_diag_start(){drecs.clear();drecs.reserve(30000);denabled=true;}
extern "C" void comm_diag_stop(const char* path){denabled=false;FILE* f=fopen(path,"w");if(!f)return;fputs("[",f);bool sep=false;for(auto& r:drecs){
fprintf(f,"%s{\"context\":%llu,\"epoch\":%llu,\"descriptor\":%llu,\"op\":%llu,\"rank\":%llu,\"size\":%llu,\"borrowed\":%llu,\"input_bytes\":%llu,\"output_bytes\":%llu,\"map_checks\":%llu,\"mmap_calls\":%llu,\"growths\":%llu,\"descriptor_checks\":%llu,\"wait_calls\":%llu,\"blocked_waits\":%llu,\"wait_yields\":%llu,\"points\":[",sep?",":"",r.context,r.epoch,r.descriptor,r.op,r.rank,r.size,r.borrowed,r.input_bytes,r.output_bytes,r.map_checks,r.mmap_calls,r.growths,r.descriptor_checks,r.wait_calls,r.blocked_waits,r.wait_yields);sep=true;
for(unsigned i=0;i<r.np;++i)fprintf(f,"%s[%u,%llu,%llu]",i?",":"",r.points[i].label,r.points[i].ns,r.waits[i]);
fputs("],\"copies\":[",f);for(unsigned i=0;i<r.nc;++i){auto& c=r.copies[i];fprintf(f,"%s[%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu]",i?",":"",c.begin,c.end,c.bytes,c.strided,c.rows,c.minrun,c.maxrun,c.tasks,c.lanes,c.kind);}fputs("]}",f);}fputs("]\n",f);fclose(f);}
'''

SNAPSHOT_HEADER = r'''
// Diagnostic-only raw bindings. The group/plans/input tensors must outlive use;
// one caller per process, contiguous equal-sized AllToAll only.
namespace {
struct DSnap {const void* sources[128]{};void* destinations[128]{};size_t bytes[128]{};size_t count=0;} dsnapshot;
}
extern "C" size_t comm_diag_snapshot(const void** sources,void** destinations,size_t* bytes,size_t capacity) {
  if(capacity<dsnapshot.count)return 0;
  for(size_t i=0;i<dsnapshot.count;++i){sources[i]=dsnapshot.sources[i];destinations[i]=dsnapshot.destinations[i];bytes[i]=dsnapshot.bytes[i];}
  return dsnapshot.count;
}
'''

def instrument(text, snapshot_addresses=False):
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
  batch_record='''
    DC dc{};dc.begin=dcurrent?dn():0;dc.minrun=~DU(0);dc.kind=1;
    if(dcurrent)for(const auto& j:JOBS){dc.bytes+=j.shape->bytes;dc.minrun=std::min<DU>(dc.minrun,j.shape->run);dc.maxrun=std::max<DU>(dc.maxrun,j.shape->run);if(j.shape->offsets.size()>1){dc.strided+=j.shape->bytes;dc.rows+=j.shape->offsets.size();}}
'''
  replace('    Batch b{&jobs,tasks,copy};',batch_record.replace('JOBS','jobs')+'    Batch b{&jobs,tasks,copy};')
  replace('    } else Batch::work(&b,0,1);','    } else Batch::work(&b,0,1);\n    if(dcurrent){dc.tasks=tasks;dc.lanes=executor?lanes:1;dc_save(dc);}')
  replace('    const uint32_t lanes=std::min<size_t>(options.threads,rows);','    const uint32_t lanes=std::min<size_t>(options.threads,rows);\n    DC dc{};dc.begin=dcurrent?dn():0;dc.kind=2;')
  replace('    stats.copy_bytes+=bytes;stats.row_copy_bytes+=bytes;++stats.row_copy_calls;', '    stats.copy_bytes+=bytes;stats.row_copy_bytes+=bytes;++stats.row_copy_calls;\n    if(dcurrent){dc.bytes=bytes;dc.strided=bytes;dc.rows=rows;dc.minrun=dc.maxrun=run;dc.tasks=rows;dc.lanes=executor?lanes:1;dc_save(dc);}')
  replace('  auto& p=*state_; auto& c=*p.owner->state_;\n  const uint32_t lane=', '  DScope diagnostic(!team || team->worker_id==0);\n  auto& p=*state_; auto& c=*p.owner->state_;\n  const uint32_t lane=')
  replace('      lock.lock();','''      lock.lock();dp(1);
      if(dcurrent){auto& d=*dcurrent;d.context=reinterpret_cast<DU>(&c);d.epoch=c.epoch+1;d.descriptor=p.descriptor_id;d.op=static_cast<DU>(p.spec.operation);d.rank=c.rank;d.size=c.size;d.borrowed=borrowed;for(auto& x:p.inputs)d.input_bytes+=x.bytes;for(auto& x:p.outputs)d.output_bytes+=x.bytes;}
''')
  replace('      c.reserve(p.required);','      dp(2);c.reserve(p.required);')
  replace('      store(&own->arrived,p.team_epoch);','      store(&own->arrived,p.team_epoch);dp(3);')
  replace('      if(early_snapshot)validate_and_map();','      if(early_snapshot){dp(4);validate_and_map();}')
  replace('      for(auto& peer:c.peers)c.wait(&peer.header->arrived,p.team_epoch);','      for(auto& peer:c.peers)c.wait(&peer.header->arrived,p.team_epoch);dp(4);')
  replace('        if(p.validated_descriptors[peer]!=h->descriptor) {','        if(p.validated_descriptors[peer]!=h->descriptor) {\n          if(dcurrent)++dcurrent->descriptor_checks;')
  replace('      p.jobs.clear();\n      if(op==Operation::AllToAll)', '      dp(5);p.jobs.clear();\n      if(op==Operation::AllToAll)')
  replace('      store(&own->ready,p.team_epoch);\n      for(auto& peer:c.peers)c.wait(&peer.header->ready,p.team_epoch);','      store(&own->ready,p.team_epoch);dp(6);\n      for(auto& peer:c.peers)c.wait(&peer.header->ready,p.team_epoch);dp(7);')
  replace('        store(&own->stages[level],p.team_epoch);','        store(&own->stages[level],p.team_epoch);dp(100+level*2);')
  replace('        for(auto& peer:c.peers)c.wait(&peer.header->stages[level],p.team_epoch);','        for(auto& peer:c.peers)c.wait(&peer.header->stages[level],p.team_epoch);dp(101+level*2);')
  replace('      p.jobs.clear();p.team_rows=false;','      dp(200);p.jobs.clear();p.team_rows=false;')
  replace('    if(consumer)(*consumer)(p.borrowed.data(),p.borrowed.size());','    dp(201);if(consumer)(*consumer)(p.borrowed.data(),p.borrowed.size());')
  replace('    if(team_consumer){(*team_consumer)(p.borrowed.data(),p.borrowed.size(),*team);sync();}', '    if(team_consumer){(*team_consumer)(p.borrowed.data(),p.borrowed.size(),*team);sync();}dp(202);')
  replace('      store(&own->done,p.team_epoch);++c.stats.calls;','      store(&own->done,p.team_epoch);++c.stats.calls;dp(203);')
  replace('    const uint32_t lanes=std::min<size_t>({c.options.threads,team->worker_count,p.team_tasks});','    const uint32_t lanes=std::min<size_t>({c.options.threads,team->worker_count,p.team_tasks});'+batch_record.replace('JOBS','p.jobs'))
  replace('    if(lane<lanes){Batch b{&p.jobs,p.team_tasks,c.copy};Batch::work(&b,lane,lanes);}\n    sync();','    if(lane<lanes){Batch b{&p.jobs,p.team_tasks,c.copy};Batch::work(&b,lane,lanes);}\n    sync();if(dcurrent && p.team_tasks){dc.tasks=p.team_tasks;dc.lanes=lanes;dc_save(dc);}')
  replace('      if(lane<lanes)c.copy->rows(&job,lane,lanes);\n      sync();', '      DC dc{};dc.begin=dcurrent?dn():0;dc.kind=2;\n      if(lane<lanes)c.copy->rows(&job,lane,lanes);\n      sync();if(dcurrent){dc.bytes=job.rows*job.run*job.peers;dc.strided=dc.bytes;dc.rows=job.rows;dc.minrun=dc.maxrun=job.run;dc.tasks=job.rows;dc.lanes=lanes;dc_save(dc);}')
  if snapshot_addresses:
    replace('namespace vecops::comm {',SNAPSHOT_HEADER+'\nnamespace vecops::comm {')
    replace('      prepare_copy();\n    });\n    if(op!=Operation::Barrier)copy();', '''      if(op==Operation::AllToAll) {
        dsnapshot.count=0;bool contiguous=p.jobs.size()<=128;
        for(const auto& j:p.jobs)contiguous &= j.shape->offsets.size()==1;
        if(contiguous)for(const auto& j:p.jobs){auto i=dsnapshot.count++;dsnapshot.sources[i]=j.source;dsnapshot.destinations[i]=j.destination;dsnapshot.bytes[i]=j.shape->bytes;}
      }
      prepare_copy();
    });
    if(op!=Operation::Barrier)copy();''')
  return text

def main():
  p=argparse.ArgumentParser();p.add_argument('--source-checkout',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--snapshot-addresses',action='store_true',help='diagnostic raw contiguous AllToAll bindings; retained owners required');a=p.parse_args()
  if a.output.exists():raise FileExistsError(a.output)
  for part in ('include','src/comm','python/vecops/distributed','tests/comm'):
    shutil.copytree(a.source_checkout/part,a.output/part,ignore=shutil.ignore_patterns('__pycache__'))
  f=a.output/'src/comm/Communication.cpp';f.write_text(instrument(f.read_text(),a.snapshot_addresses))
  print(a.output)
if __name__=='__main__':main()
