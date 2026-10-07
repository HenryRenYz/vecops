#include <cstdint>
#include <cstdio>
#include <limits>

#include "vecops/vec/Vec.h"

namespace v = vecops::vec;

template <typename Memory, typename Tag, int Scale>
bool check() {
  using Request = v::LoadRequest<Tag, v::Active::Unmasked,
      v::Addressing::Strided, v::Populate::Zero, v::mem::Unaligned,
      v::mem::Temporal, Scale>;
  Request request;
  constexpr vecops::nint_t scale = std::same_as<v::details::CurrentBackend, v::details::SVEBackend>
      ? (Scale == 0 ? sizeof(Memory) : Scale) : 1;
  const auto last = v::size(Tag{}) > 1 ? v::size(Tag{}) - 1 : 1;
  const auto lower = std::numeric_limits<int32_t>::min() / scale / last;
  const auto upper = std::numeric_limits<int32_t>::max() / scale / last;
  request.stride = lower;
  if (!v::details::strided_indices_fit<Memory>(Tag{}, request)) return false;
  request.stride = lower - 1;
  if (v::details::strided_indices_fit<Memory>(Tag{}, request)) return false;
  request.stride = upper;
  if (!v::details::strided_indices_fit<Memory>(Tag{}, request)) return false;
  request.stride = upper + 1;
  if (v::details::strided_indices_fit<Memory>(Tag{}, request)) return false;
  request.stride = std::numeric_limits<vecops::nint_t>::min();
  if (v::details::strided_indices_fit<Memory>(Tag{}, request)) return false;
  request.stride = std::numeric_limits<vecops::nint_t>::max();
  if (v::details::strided_indices_fit<Memory>(Tag{}, request)) return false;
  request.stride = 0;
  return v::details::strided_indices_fit<Memory>(Tag{}, request);
}

int main() {
  using Fixed = v::FixedTag<vecops::float32_t, 4>;
  using Scalable = v::ScalableTag<vecops::float32_t, 1>;
  if (!check<vecops::float32_t, Fixed, 0>() ||
      !check<vecops::float16_t, Fixed, 0>() ||
      !check<vecops::float64_t, Fixed, 0>() ||
      !check<vecops::float32_t, Fixed, 1>() ||
      !check<vecops::float32_t, Fixed, 8>() ||
      !check<vecops::float32_t, Scalable, 0>() ||
      !check<vecops::float16_t, Scalable, 0>()) return 1;
  std::puts("strided i32/byte-offset signed boundary gate passed");
  return 0;
}
