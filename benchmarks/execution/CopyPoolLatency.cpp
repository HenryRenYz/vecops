// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
/** Real vec payload on the loaded provider ABI; allocation, verification and
 * simulated producer gaps are excluded from every reported copy interval. */
#include "vecops/comm/Copy.h"
#include "vecops/runtime/CallAbi.h"
#include <chrono>
#include <cstdint>
#include <thread>
#include <vector>
namespace {
struct Job {
  const vecops::comm::detail::CopyBackend* copy;
  vecops::comm::detail::CopyRows rows;
  const std::uint8_t* source;
  std::uint8_t* output;
  std::size_t bytes;
};
void copy(void* context,std::uint32_t lane,std::uint32_t lanes) {
  auto& j=*static_cast<Job*>(context);
  if(j.rows.peers>1)j.copy->rows(&j.rows,lane,lanes);
  else {auto a=j.bytes*lane/lanes,b=j.bytes*(lane+1)/lanes;j.copy->bytes(j.output+a,j.source+a,b-a);}
}
}
extern "C" int measure_copy_pool(const VecopsThreadPoolV1* pool,unsigned workers,
    std::size_t rows,std::size_t run,std::size_t peers,unsigned repeats,unsigned gap_us,double* ns) {
  if(!pool || !workers || workers>pool->max_parallelism(pool->context) || !rows || !run || !peers)return -1;
  auto block=rows*run,bytes=block*peers;
  std::vector<std::uint8_t> source(bytes),output(bytes+128,99);
  for(std::size_t i=0;i<bytes;++i)source[i]=static_cast<std::uint8_t>(i*13+7);
  std::vector<const void*> pointers(peers);
  for(std::size_t p=0;p<peers;++p)pointers[p]=source.data()+p*block;
  vecops::comm::detail::CopyRows geometry{.sources=pointers.data(),.destination=output.data()+64,
      .peers=peers,.rows=rows,.run=run,.row_stride=peers*run};
  Job job{&vecops::comm::detail::copy_backend(),geometry,source.data(),output.data()+64,bytes};
  using Clock=std::chrono::steady_clock;
  for(unsigned i=0;i<repeats+10;++i) {
    if(gap_us)std::this_thread::sleep_for(std::chrono::microseconds(gap_us));
    source[0]=static_cast<std::uint8_t>(i);
    auto begin=Clock::now();
    int status=pool->parallel_for(pool->context,workers,&job,copy,nullptr);
    auto elapsed=std::chrono::duration<double,std::nano>(Clock::now()-begin).count();
    if(status || output[64]!=source[0])return -2;
    if(i>=10)ns[i-10]=elapsed;
  }
  for(std::size_t row=0;row<rows;++row)for(std::size_t p=0;p<peers;++p)for(std::size_t i=0;i<run;++i)
    if(output[64+row*peers*run+p*run+i]!=source[p*block+row*run+i])return -3;
  for(unsigned i=0;i<64;++i)if(output[i]!=99 || output[64+bytes+i]!=99)return -4;
  return 0;
}
