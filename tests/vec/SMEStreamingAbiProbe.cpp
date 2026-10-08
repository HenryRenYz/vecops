// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
// Streaming-only wide-address validation; allocate and diagnose outside SM.
// Arrays below are ordinary memory inputs, never vector/Mask representations.
#include <array>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <initializer_list>
#include <sys/mman.h>

#include "vecops/vec/Vec.h"
#include "vecops/vec/SME.h"

namespace v = vecops::vec;

template <typename Memory>
struct Sparse {
  static constexpr vecops::nint_t stride = vecops::nint_t{1} << 30;
  vecops::nint_t lanes;
  std::size_t bytes;
  void* mapping;

  explicit Sparse(vecops::nint_t count)
      : lanes(count), bytes((2 * count + 1) * stride * sizeof(Memory) + 4096),
        mapping(mmap(nullptr, bytes, PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0)) {}

  ~Sparse() {
    if (mapping != MAP_FAILED) munmap(mapping, bytes);
  }

  Memory* base() {
    return reinterpret_cast<Memory*>(static_cast<unsigned char*>(mapping)
                                    + lanes * stride * sizeof(Memory));
  }
};

template <typename Tag, typename Memory, typename Layout, typename Policy>
VECOPS_ALWAYS_INLINE int run_access(
    Tag tag, Memory* pointer, vecops::nint_t lanes,
    Layout layout, Policy policy) noexcept {
  using MemoryTag = v::Rebind<Memory, Tag>;
  using Compute = v::ElementOf<Tag>;
  if (lanes != v::size(tag)) return 2;
  // Only active lanes are read; avoid a library memset inside the SME owner.
  std::array<Memory, 512> input;
  for (vecops::nint_t lane = 0; lane < lanes; ++lane) {
    input[lane] = static_cast<Memory>(static_cast<int>(lane) - 7);
  }
  const auto dense = v::load(MemoryTag{}, input.data());
  const auto all_expected = v::convert(tag, MemoryTag{}, dense, layout, v::cvt::saturate);
  auto compute = v::zeros(tag);
  for (vecops::nint_t lane = 0; lane < lanes; ++lane)
    compute = v::set(tag, compute, lane, static_cast<Compute>(300 - lane * 29));
  const auto stored = v::convert(MemoryTag{}, tag, compute, layout, policy);
  for (bool negative : {false, true}) {
    const auto stride = negative ? -Sparse<Memory>::stride : Sparse<Memory>::stride;
    for (vecops::nint_t lane = 0; lane < lanes; ++lane)
      pointer[lane * stride] = input[lane];
    const auto loaded = v::load_convert(tag, pointer, v::strided(stride), layout);
    for (vecops::nint_t lane = 0; lane < lanes; ++lane)
      if (v::get(tag, loaded, lane) != v::get(tag, all_expected, lane)) return 11;

    auto inactive = v::zeros(tag);
    for (vecops::nint_t lane = 0; lane < lanes; ++lane)
      inactive = v::set(tag, inactive, lane, static_cast<Compute>(71 + lane));
    using RequestMaskTag = std::conditional_t<std::same_as<Layout, v::cvt::Unordered>, MemoryTag, Tag>;
    using WideRequest = v::LoadConvertRequest<Tag, Memory, v::Active::First,
        v::Addressing::Strided, v::Populate::MergeVector, v::mem::Unaligned,
        v::mem::Temporal, 0, v::Vec<v::IndexTag<Tag>>, Layout,
        v::cvt::Saturate, v::Mask<RequestMaskTag>>;
    using DenseRequest = v::LoadConvertRequest<Tag, Memory, v::Active::First,
        v::Addressing::Contiguous, v::Populate::MergeVector, v::mem::Unaligned,
        v::mem::Temporal, 0, v::Vec<v::IndexTag<Tag>>, Layout,
        v::cvt::Saturate, v::Mask<RequestMaskTag>>;
    WideRequest wide_request;
    DenseRequest dense_request;
    wide_request.stride = stride;
    wide_request.first_count = dense_request.first_count = lanes / 2;
    wide_request.merge_vector = dense_request.merge_vector = &inactive;
    const auto populated = v::load_convert(tag, pointer, wide_request);
    const auto populated_expected = v::load_convert(tag, input.data(), dense_request);
    for (vecops::nint_t lane = 0; lane < lanes; ++lane)
      if (v::get(tag, populated, lane) != v::get(tag, populated_expected, lane)) return 16;

    for (int first : {0, 1, static_cast<int>(lanes / 2), static_cast<int>(lanes)}) {
      for (vecops::nint_t lane = 0; lane < lanes; ++lane)
        pointer[lane * stride] = input[lane];
      const auto expected = v::load_convert(tag, input.data(), layout, v::opt::first(first));
      const auto partial = v::load_convert(tag, pointer, v::strided(stride), layout, v::opt::first(first));
      for (vecops::nint_t lane = 0; lane < lanes; ++lane)
        if (v::get(tag, partial, lane) != v::get(tag, expected, lane)) return 12;
      for (vecops::nint_t lane = 0; lane < lanes; ++lane)
        pointer[lane * stride] = static_cast<Memory>(-3);
      v::store_convert(tag, pointer, compute, v::strided(stride), layout, policy, v::opt::first(first));
      for (vecops::nint_t lane = 0; lane < lanes; ++lane) {
        const Memory expected_lane = lane < first ? v::get(MemoryTag{}, stored, lane) : static_cast<Memory>(-3);
        if (pointer[lane * stride] != expected_lane) return 13;
      }
    }

    using MaskTag = std::conditional_t<std::same_as<Layout, v::cvt::Unordered>, MemoryTag, Tag>;
    auto mask = v::mfill(MaskTag{}, false);
    for (vecops::nint_t lane = 0; lane < lanes; ++lane)
      mask = v::set(MaskTag{}, mask, lane, lane % 3 == 1);
    for (vecops::nint_t lane = 0; lane < lanes; ++lane)
      pointer[lane * stride] = input[lane];
    const auto masked_expected = v::load_convert(tag, input.data(), layout, v::opt::masked(mask));
    const auto masked = v::load_convert(tag, pointer, v::strided(stride), layout, v::opt::masked(mask));
    for (vecops::nint_t lane = 0; lane < lanes; ++lane)
      if (v::get(tag, masked, lane) != v::get(tag, masked_expected, lane)) return 14;
    for (vecops::nint_t lane = 0; lane < lanes; ++lane)
      pointer[lane * stride] = static_cast<Memory>(-3);
    v::store_convert(tag, pointer, compute, v::strided(stride), layout, policy, v::opt::masked(mask));
    for (vecops::nint_t lane = 0; lane < lanes; ++lane) {
      const Memory expected_lane = lane % 3 == 1 ? v::get(MemoryTag{}, stored, lane) : static_cast<Memory>(-3);
      if (pointer[lane * stride] != expected_lane) return 15;
    }
  }
  return 0;
}

template <typename Tag, typename Memory, typename Layout, typename Policy>
int run(Tag tag, Layout layout, Policy policy) {
  // Every test uses POW=1; rdsvl queries the streaming domain before entry.
  const auto lanes = 2 * v::sme::streaming_lanes<v::ElementOf<Tag>>();
  if (lanes > 512) return 4;
  Sparse<Memory> sparse(lanes);
  if (sparse.mapping == MAP_FAILED) {
    std::fprintf(stderr, "streaming sparse mmap failed errno=%d\n", errno);
    return 3;
  }
  Memory* pointer = sparse.base();
  const auto state_before = v::sme::read_svcr();
  if (state_before != 0) return 5;
  int result = -1;
  v::sme::with_streaming_za([&]() VECOPS_INLINE_LAMBDA_NOEXCEPT {
    if (v::sme::read_svcr() != 3) {
      result = 6;
      return;
    }
    result = run_access(tag, pointer, lanes, layout, policy);
  });
  if (v::sme::read_svcr() != state_before) return 7;
  return result;
}

int main() {
  using Float = v::ScalableTag<vecops::float32_t, 1>;
  using Integer = v::ScalableTag<int32_t, 1>;
  using Unsigned = v::ScalableTag<uint32_t, 1>;
  using WideInteger = v::ScalableTag<int64_t, 1>;
  using WideUnsigned = v::ScalableTag<uint64_t, 1>;
  const auto check = [](int code, const char* name) {
    if (code) std::fprintf(stderr, "%s failed: %d\n", name, code);
    return code;
  };
  int result = check(run<Float, vecops::float16_t>(Float{}, v::cvt::ordered, v::cvt::saturate), "f16 ordered");
  if (result) return result;
  result = check(run<Float, vecops::float16_t>(Float{}, v::cvt::unordered, v::cvt::saturate), "f16 unordered");
  if (result) return result;
  result = check(run<Float, vecops::bfloat16_t>(Float{}, v::cvt::unordered, v::cvt::saturate), "bf16 unordered");
  if (result) return result;
  result = check(run<Integer, int8_t>(Integer{}, v::cvt::ordered, v::cvt::saturate), "i8 saturate");
  if (result) return result;
  result = check(run<Integer, int8_t>(Integer{}, v::cvt::unordered, v::cvt::wrap), "i8 unordered wrap");
  if (result) return result;
  result = check(run<Integer, int16_t>(Integer{}, v::cvt::unordered, v::cvt::wrap), "i16 unordered wrap");
  if (result) return result;
  result = check(run<Float, vecops::float32_t>(Float{}, v::cvt::unordered, v::cvt::saturate), "f32 identity");
  if (result) return result;
  result = check(run<Float, vecops::float64_t>(Float{}, v::cvt::unordered, v::cvt::saturate), "f64 unordered");
  if (result) return result;
  result = check(run<Unsigned, uint8_t>(Unsigned{}, v::cvt::unordered, v::cvt::wrap), "u8 unordered wrap");
  if (result) return result;
  result = check(run<Unsigned, uint16_t>(Unsigned{}, v::cvt::unordered, v::cvt::wrap), "u16 unordered wrap");
  if (result) return result;
  result = check(run<Integer, int32_t>(Integer{}, v::cvt::unordered, v::cvt::saturate), "i32 identity");
  if (result) return result;
  result = check(run<Unsigned, uint32_t>(Unsigned{}, v::cvt::unordered, v::cvt::saturate), "u32 identity");
  if (result) return result;
  result = check(run<WideInteger, int64_t>(WideInteger{}, v::cvt::unordered, v::cvt::saturate), "i64 identity");
  if (result) return result;
  result = check(run<WideUnsigned, uint64_t>(WideUnsigned{}, v::cvt::unordered, v::cvt::saturate), "u64 identity");
  if (result) return result;
  std::puts("streaming wide-stride layout/mask/first/saturate/wrap gate passed");
  return 0;
}
