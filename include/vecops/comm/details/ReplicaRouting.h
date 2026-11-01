// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
#pragma once
#include <algorithm>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

namespace vecops::comm::detail {
/** Cold deterministic ingress assignment for one replica partition.
 * Local sources have target -1. Each remote source goes to the least-loaded
 * local member, using payload bytes and global-rank order to break ties.
 * Members of one domain derive identical targets without negotiation. Group
 * labels and membership need not be contiguous or equally sized. Arithmetic
 * overflow is rejected before communication begins.
 */
inline std::vector<int> balanced_relay_targets(std::span<const int> groups,
                                             int rank,std::span<const std::uint64_t> bytes) {
  if(groups.size()!=bytes.size() || rank<0 || static_cast<std::size_t>(rank)>=groups.size())
    throw std::invalid_argument("invalid replica ingress partition");
  std::vector<int> members;
  for(std::size_t r=0;r<groups.size();++r)
    if(groups[r]==groups[rank])members.push_back(static_cast<int>(r));
  std::vector<std::uint64_t> loads(members.size(),0);
  std::vector<int> result(groups.size(),-1);
  for(std::size_t source=0;source<groups.size();++source) {
    if(groups[source]==groups[rank])continue;
    const auto selected=static_cast<std::size_t>(std::min_element(loads.begin(),loads.end())-loads.begin());
    if(bytes[source]>std::numeric_limits<std::uint64_t>::max()-loads[selected])
      throw std::overflow_error("replica ingress byte count overflow");
    loads[selected]+=bytes[source];result[source]=members[selected];
  }
  return result;
}
}
