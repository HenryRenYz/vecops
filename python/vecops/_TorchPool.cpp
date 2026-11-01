// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
/** Torch's OpenMP workers as one synchronous, cancellable resident team.
 * Exactly requested_workers callbacks must be concurrent: a task-style
 * parallel_for is insufficient because its chunks may execute sequentially.
 * The library is loaded once per process and shared by every operator/collective.
 */
#include <ATen/Parallel.h>
#include <c10/util/Exception.h>
#include <omp.h>
#include <algorithm>
#include <atomic>
#include <cstring>
#include <exception>
#include <mutex>
#include "vecops/runtime/CallAbi.h"

namespace {
void error_message(VecopsError* e, const char* text) noexcept {
  if (!e || e->struct_size < sizeof(VecopsError)) return;
  e->code=VECOPS_STATUS_EXECUTION_ERROR;
  e->message_required=std::strlen(text)+1;
  if(e->message && e->message_capacity) {
    const auto n=std::min(std::strlen(text),e->message_capacity-1);
    std::memcpy(e->message,text,n);e->message[n]=0;
  }
}
void relax() noexcept {
#if defined(__aarch64__)
  asm volatile("yield" ::: "memory");
#elif defined(__x86_64__)
  __builtin_ia32_pause();
#else
  std::atomic_signal_fence(std::memory_order_seq_cst);
#endif
}
struct Barrier {
  explicit Barrier(uint32_t n):count(n){}
  const uint32_t count;
  std::atomic<uint32_t> arrived{0};
  alignas(128) std::atomic<uint32_t> generation{0};
  std::atomic<bool> cancelled{false};
  static int32_t wait(void* p, VecopsError* error) {
    auto& b=*static_cast<Barrier*>(p);
    const auto gen=b.generation.load(std::memory_order_acquire);
    if(!b.cancelled.load(std::memory_order_acquire)) {
      if(b.arrived.fetch_add(1,std::memory_order_acq_rel)+1==b.count) {
        b.arrived.store(0,std::memory_order_relaxed);
        b.generation.fetch_add(1,std::memory_order_release);
      } else while(b.generation.load(std::memory_order_acquire)==gen && !b.cancelled.load(std::memory_order_acquire)) relax();
    }
    if(!b.cancelled.load(std::memory_order_acquire))return VECOPS_STATUS_OK;
    error_message(error,"Torch resident team cancelled");return VECOPS_STATUS_EXECUTION_ERROR;
  }
  void cancel(){cancelled.store(true,std::memory_order_release);generation.fetch_add(1,std::memory_order_release);}
};
uint32_t maximum(void*) {return std::max(1,at::get_num_threads());}
uint32_t nested(void*) {return omp_in_parallel() || at::in_parallel_region() || c10::ParallelGuard::is_enabled();}
int32_t tasks(void*,uint32_t count,void* context,VecopsParallelTaskFn body,VecopsError* error) noexcept {
  if(!body || !count)return VECOPS_STATUS_OK;
  try {
    // A one-worker OpenMP region is inactive to omp_in_parallel(). The C10
    // guard still makes it a resident callback and forbids another team.
    if(nested(nullptr)) {
      for(uint32_t i=0;i<count;++i)body(context,i,count);
      return VECOPS_STATUS_OK;
    }
    at::parallel_for(0,count,1,[&](int64_t first,int64_t last){
      for(auto i=first;i<last;++i)body(context,static_cast<uint32_t>(i),count);
    });return VECOPS_STATUS_OK;
  }catch(const std::exception& e){error_message(error,e.what());}
  catch(...){error_message(error,"Torch task failed");}
  return VECOPS_STATUS_EXECUTION_ERROR;
}
int32_t run(void*,uint32_t requested,void* context,VecopsTeamBodyFn body,VecopsError* error) noexcept {
  if(!body || !requested)return VECOPS_STATUS_OK;
  try {
    TORCH_CHECK(!nested(nullptr),"nested Torch resident team is unsupported");
    at::internal::lazy_init_num_threads();
    TORCH_CHECK(requested<=maximum(nullptr),"Torch resident team exceeds prepared thread budget");
    Barrier barrier(requested);
    std::exception_ptr exception;
    std::mutex mutex;
    uint32_t actual=0;
#pragma omp parallel num_threads(requested) shared(actual,exception,barrier)
    {
#pragma omp single
      actual=static_cast<uint32_t>(omp_get_num_threads());
      // The single's implicit barrier publishes actual before any callback.
      if(actual==requested) {
        const auto id=static_cast<uint32_t>(omp_get_thread_num());
        at::internal::ThreadIdGuard thread_id(id);
        c10::ParallelGuard guard(true);
        const VecopsTeamContextV1 team{sizeof(VecopsTeamContextV1),id,requested,&barrier,Barrier::wait};
        try{body(context,&team);}
        catch(...){
          {
            std::lock_guard lock(mutex);
            if(!exception)exception=std::current_exception();
          }
          barrier.cancel();
        }
      }
    }
    TORCH_CHECK(actual==requested,"OpenMP reduced a prepared resident team (OMP_DYNAMIC/thread limit); callbacks were not entered");
    if(exception)std::rethrow_exception(exception);
    return VECOPS_STATUS_OK;
  }catch(const std::exception& e){error_message(error,e.what());}
  catch(...){error_message(error,"Torch resident team failed");}
  return VECOPS_STATUS_EXECUTION_ERROR;
}
const VecopsTeamExecutorV1 executor{sizeof(VecopsTeamExecutorV1),1,run};
VecopsThreadPoolV1 provider{sizeof(VecopsThreadPoolV1),VECOPS_THREAD_POOL_ABI_MAJOR,VECOPS_THREAD_POOL_ABI_MINOR,
 UINT64_C(0x746f726368746561),nullptr,maximum,nested,tasks,nullptr,nullptr,0,&executor};
}
extern "C" VECOPS_RUNTIME_EXPORT const VecopsThreadPoolV1* vecops_torch_resident_pool_v1(){return &provider;}
