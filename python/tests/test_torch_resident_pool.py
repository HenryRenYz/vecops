# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Resident teams must be concurrent, share one provider and cancel on failure."""
import ctypes
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch


@pytest.fixture(scope='module')
def probe(tmp_path_factory):
  root = tmp_path_factory.mktemp('torch-team-probe')
  source = root/'probe.cpp'
  source.write_text(r'''
#include "vecops/runtime/CallAbi.h"
#include <atomic>
#include <stdexcept>
#include <sys/syscall.h>
#include <unistd.h>
struct Work { unsigned values[64]{}; unsigned answer=0; int fail; };
void body(void* p,const VecopsTeamContextV1* t) {
  auto& w=*static_cast<Work*>(p);
  for(unsigned i=0;i<12;++i) {
    if(w.fail==int(t->worker_id) && i==2)throw std::runtime_error("injected worker failure");
    w.values[t->worker_id]=i+1;
    if(t->barrier(t->context,nullptr))throw std::runtime_error("cancelled");
    if(!t->worker_id)for(unsigned j=0;j<t->worker_count;++j)w.answer+=w.values[j];
    if(t->barrier(t->context,nullptr))throw std::runtime_error("cancelled");
  }
}
extern "C" int check(const VecopsThreadPoolV1* pool,unsigned n,int fail,unsigned* answer) {
  Work work{};work.fail=fail;char text[1024]{};VecopsError error{sizeof(error),0,text,sizeof(text),0};
  auto result=pool->team_executor->run(pool->context,n,&work,body,&error);
  *answer=work.answer;return result;
}
''')
  with source.open('a') as f:
    f.write(r'''
struct Nested { const VecopsThreadPoolV1* pool; std::atomic<unsigned> flags{0}; long tid=0; };
void forbidden_nested(void* p,const VecopsTeamContextV1*) {static_cast<Nested*>(p)->flags.fetch_or(1);}
void nested_task(void* p,uint32_t,uint32_t) {
  auto& w=*static_cast<Nested*>(p);
  if(syscall(SYS_gettid)!=w.tid)w.flags.fetch_or(8);
}
void nested_outer(void* p,const VecopsTeamContextV1* team) {
  auto& w=*static_cast<Nested*>(p);
  if(!team->worker_id) {
    w.tid=syscall(SYS_gettid);
    if(!w.pool->in_parallel_region(w.pool->context))w.flags.fetch_or(4);
    if(!w.pool->team_executor->run(w.pool->context,1,p,forbidden_nested,nullptr))w.flags.fetch_or(2);
    if(w.pool->parallel_for(w.pool->context,8,p,nested_task,nullptr))w.flags.fetch_or(16);
  }
  if(team->barrier(team->context,nullptr))w.flags.fetch_or(32);
}
extern "C" int check_nested(const VecopsThreadPoolV1* pool,unsigned n,unsigned* flags) {
  Nested work{pool};auto result=pool->team_executor->run(pool->context,n,&work,nested_outer,nullptr);
  *flags=work.flags.load();return result;
}
''')
  sdk = Path(__file__).resolve().parents[2]
  library = root/'probe.so'
  subprocess.run(['g++','-std=c++20','-O2','-shared','-fPIC','-I'+str(sdk/'include'),
                  str(source),'-o',str(library)],check=True)
  return library


def test_shared_team_and_nonleader_failure_recover_in_fresh_process(probe):
  code = r'''
import ctypes, os, torch
from vecops._torch_pool import address
from vecops import _schema_bridge
os.environ['VECOPS_THREAD_POOL']='torch'
os.environ['VECOPS_TORCH_RESIDENT_TEAM']='1'
torch.set_num_threads(4)
a=address()
from pathlib import Path
import vecops._torch_pool as provider
command=Path(provider._owner._name).with_name('build-command.txt').read_text()
assert '-DVECOPS_TORCH_PROVIDER_BUILD_FLAG=1' in command
assert '-Wl,--as-needed' in command
assert a.value==_schema_bridge._thread_pool_pointer(torch,'test::first').value
assert a.value==_schema_bridge._thread_pool_pointer(torch,'test::second').value
library=ctypes.CDLL(os.environ['TEST_PROBE'])
library.check.argtypes=[ctypes.c_void_p,ctypes.c_uint,ctypes.c_int,ctypes.POINTER(ctypes.c_uint)]
for n,fail in [(1,-1),(4,-1),(3,2),(4,-1),(2,-1),(5,-1)]:
 answer=ctypes.c_uint()
 result=library.check(a,n,fail,ctypes.byref(answer))
 if n>4:assert result!=0 and answer.value==0
 elif fail<0:assert result==0 and answer.value==n*78,(n,result,answer.value)
 else:assert result!=0
'''
  env = dict(os.environ, TEST_PROBE=str(probe), OMP_DYNAMIC='FALSE',
      CXXFLAGS=os.environ.get('CXXFLAGS','')+' -DVECOPS_TORCH_PROVIDER_BUILD_FLAG=1',
      LDFLAGS=os.environ.get('LDFLAGS','')+' -Wl,--as-needed')
  if 'GNU' in torch.__config__.parallel_info() or 'OpenMP 201511' in torch.__config__.parallel_info():
    env.update(VECOPS_TORCH_BRIDGE_CC='gcc', VECOPS_TORCH_BRIDGE_CXX='g++')
  subprocess.run([sys.executable,'-c',code],env=env,check=True,
      timeout=int(os.environ.get('VECOPS_TEST_BUILD_TIMEOUT','120')))


def test_torch_pool_preserves_native_graph_islands_in_fresh_process():
  code = r'''
import torch
from vecops.graph import GraphCapture
torch.set_num_threads(4)
with torch.inference_mode():
 x=torch.randn(4,257)
 c=GraphCapture('torch-team-island')
 c.capture_torch(lambda x:((x+1)*2).relu(),x)
 p=c.compile_torch(native_pointwise=True,borrow_outputs=True)
 assert max(p.allocator_policy['native_island_sizes'])>=3,p.allocator_policy
 for i in range(3):
  y=x+i
  torch.testing.assert_close(p(y),((y+1)*2).relu())
'''
  env=dict(os.environ,VECOPS_THREAD_POOL='torch',VECOPS_TORCH_RESIDENT_TEAM='1',OMP_DYNAMIC='FALSE')
  if 'OpenMP 201511' in torch.__config__.parallel_info():
    env.update(VECOPS_TORCH_BRIDGE_CC='gcc',VECOPS_TORCH_BRIDGE_CXX='g++')
  subprocess.run([sys.executable,'-c',code],env=env,check=True,
      timeout=int(os.environ.get('VECOPS_TEST_BUILD_TIMEOUT','180')))


def test_one_worker_team_is_nested_and_logical_tasks_remain_serial(probe):
  code=r'''
import ctypes,torch
from vecops._torch_pool import address
torch.set_num_threads(4)
module=ctypes.CDLL(__import__('sys').argv[1])
module.check_nested.argtypes=[ctypes.c_void_p,ctypes.c_uint,ctypes.POINTER(ctypes.c_uint)]
pool=address().value
for workers in (1,4,1):
 flags=ctypes.c_uint()
 assert module.check_nested(pool,workers,ctypes.byref(flags))==0
 assert flags.value==0,flags.value
'''
  env=dict(os.environ,VECOPS_THREAD_POOL='torch',VECOPS_TORCH_RESIDENT_TEAM='1')
  subprocess.run([sys.executable,'-c',code,str(probe)],env=env,check=True,timeout=180)
