// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
/** @file Copy.cpp
 * Byte transport through the public vec API. Dispatch once per worker/job,
 * not per peer fragment. Four independent vectors cover long runs;
 * predication handles short runs/tails without reading outside a snapshot.
 */
#include "vecops/comm/Copy.h"
#include "vecops/vec/Vec.h"
#include <algorithm>
#ifndef VECOPS_COMM_COPY_VARIANT
#define VECOPS_COMM_COPY_VARIANT portable
#endif
namespace vecops::comm::detail::VECOPS_COMM_COPY_VARIANT {
namespace {
using Tag=vec::ScalableTag<uint8_t,0>;
VECOPS_ALWAYS_INLINE void copy(uint8_t* to,const uint8_t* from,std::size_t bytes) noexcept {
  const auto tag=Tag{};
  const auto width=static_cast<std::size_t>(vec::size(tag));
  std::size_t i=0;
  for(;i+4*width<=bytes;i+=4*width) {
    auto a=vec::load(tag,from+i);
    auto b=vec::load(tag,from+i+width);
    auto c=vec::load(tag,from+i+2*width);
    auto d=vec::load(tag,from+i+3*width);
    vec::store(tag,to+i,a);vec::store(tag,to+i+width,b);
    vec::store(tag,to+i+2*width,c);vec::store(tag,to+i+3*width,d);
  }
  for(;i<bytes;i+=width) {
    const auto active=vec::opt::first(static_cast<nint_t>(std::min(width,bytes-i)));
    auto value=vec::load(tag,from+i,active,vec::opt::zero);
    vec::store(tag,to+i,value,active);
  }
}
template<std::size_t Run>
void rows(const CopyRows& job,std::uint32_t lane,std::uint32_t lanes) noexcept {
  const std::size_t first=job.rows*lane/lanes,last=job.rows*(lane+1)/lanes;
  for(std::size_t row=first;row<last;++row)
    for(std::size_t peer=0;peer<job.peers;++peer)
      copy(static_cast<uint8_t*>(job.destination)+row*job.row_stride+peer*Run,
           static_cast<const uint8_t*>(job.sources[peer])+row*Run,Run);
}
}
void copy_rows(void* opaque,std::uint32_t lane,std::uint32_t lanes) noexcept {
  const auto& job=*static_cast<const CopyRows*>(opaque);
  switch(job.run) {
    case 16:return rows<16>(job,lane,lanes);
    case 32:return rows<32>(job,lane,lanes);
    case 64:return rows<64>(job,lane,lanes);
    case 128:return rows<128>(job,lane,lanes);
    case 192:return rows<192>(job,lane,lanes);
    default:
      const std::size_t first=job.rows*lane/lanes,last=job.rows*(lane+1)/lanes;
      for(std::size_t row=first;row<last;++row)
        for(std::size_t peer=0;peer<job.peers;++peer)
          copy(static_cast<uint8_t*>(job.destination)+row*job.row_stride+peer*job.run,
               static_cast<const uint8_t*>(job.sources[peer])+row*job.run,job.run);
  }
}
void copy_bytes(void* destination,const void* source,std::size_t bytes) noexcept {
  copy(static_cast<uint8_t*>(destination),static_cast<const uint8_t*>(source),bytes);
}
template<std::size_t Run>
void strided(uint8_t* to,const uint8_t* from,const std::size_t* offsets,
             std::size_t first,std::size_t last,bool pack) noexcept {
  for(std::size_t row=first;row<last;++row) {
    if(pack)copy(to+row*Run,from+offsets[row],Run);
    else copy(to+offsets[row],from+row*Run,Run);
  }
}
void copy_strided(void* destination,const void* source,const std::size_t* offsets,
                  std::size_t first,std::size_t last,std::size_t run,bool pack) noexcept {
  auto* to=static_cast<uint8_t*>(destination);
  auto* from=static_cast<const uint8_t*>(source);
  switch(run) {
    case 16:return strided<16>(to,from,offsets,first,last,pack);
    case 32:return strided<32>(to,from,offsets,first,last,pack);
    case 64:return strided<64>(to,from,offsets,first,last,pack);
    case 128:return strided<128>(to,from,offsets,first,last,pack);
    case 192:return strided<192>(to,from,offsets,first,last,pack);
    default:
      for(std::size_t row=first;row<last;++row) {
        if(pack)copy(to+row*run,from+offsets[row],run);
        else copy(to+offsets[row],from+row*run,run);
      }
  }
}
const CopyBackend& backend() noexcept {
  static const CopyBackend result{copy_bytes,copy_strided,copy_rows};
  return result;
}
}
