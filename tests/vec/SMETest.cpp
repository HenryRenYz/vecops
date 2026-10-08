// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>

#include <array>
#include <cstdint>

#include "vecops/vec/Vec.h"
#include "vecops/vec/details/sme/SME.h"

namespace {

using namespace vecops;

template <typename T>
VECOPS_ALWAYS_INLINE void roundtrip(
    const T* input, T* output) noexcept {
  using Tag = vec::ScalableTag<T, 0>;
  const auto pg = vec::mtrue(Tag{});
  const auto value = vec::load(Tag{}, input);
  vec::details::sme::zero_za();
  vec::details::sme::write_hor<0>(0, pg, value);
  vec::store(
      Tag{}, output,
      vec::details::sme::read_hor<0>(Tag{}, 0, pg));
}

template <typename T>
VECOPS_ALWAYS_INLINE void vertical_roundtrip(
    const T* input, T* output) noexcept {
  using Tag = vec::ScalableTag<T, 0>;
  const auto pg = vec::mtrue(Tag{});
  vec::details::sme::zero_za();
  vec::details::sme::write_ver<0>(0, pg, vec::load(Tag{}, input));
  vec::store(
      Tag{}, output,
      vec::details::sme::read_ver<0>(Tag{}, 0, pg));
}

struct HelperBody {
  const uint8_t* in8;
  const uint16_t* in16;
  const uint32_t* in32;
  const uint64_t* in64;
  uint8_t* out8;
  uint16_t* out16;
  uint32_t* out32;
  uint64_t* out64;
  uint8_t* vertical8;
  uint16_t* vertical16;
  uint32_t* vertical32;
  uint64_t* vertical64;
  uint32_t* partial32;

  VECOPS_ALWAYS_INLINE void operator()() const noexcept {
    roundtrip(in8, out8);
    roundtrip(in16, out16);
    roundtrip(in32, out32);
    roundtrip(in64, out64);

    vertical_roundtrip(in8, vertical8);
    vertical_roundtrip(in16, vertical16);
    vertical_roundtrip(in32, vertical32);
    vertical_roundtrip(in64, vertical64);

    using U32Tag = vec::ScalableTag<uint32_t, 0>;
    const auto pg32 = vec::mtrue(U32Tag{});
    const nint_t lanes = vec::size(U32Tag{});
    vec::details::sme::zero_za();
    vec::details::sme::write_hor<0>(
        0, pg32, vec::load(U32Tag{}, in32));
    const auto partial_pg = vec::mwhilelt(U32Tag{}, 0, lanes / 2);
    vec::store(
        U32Tag{}, partial32,
        vec::details::sme::read_hor<0>(U32Tag{}, 0, partial_pg));

    using F32Tag = vec::ScalableTag<vecops::float32_t, 0>;
    const auto pg_f32 = vec::mtrue(F32Tag{});
    vec::details::sme::zero_za();
    vec::details::sme::mopa<
        0, vecops::float32_t, vecops::float32_t, vecops::float32_t>(
        pg_f32, pg_f32, vec::fill(F32Tag{}, 1.0f),
        vec::fill(F32Tag{}, 2.0f));

    using F16Tag = vec::ScalableTag<vecops::float16_t, 0>;
    const auto pg_f16 = vec::mtrue(F16Tag{});
    vec::details::sme::mopa<
        0, vecops::float32_t, vecops::float16_t, vecops::float16_t>(
        pg_f16, pg_f16, vec::zeros(F16Tag{}), vec::zeros(F16Tag{}));

    using BF16Tag = vec::ScalableTag<vecops::bfloat16_t, 0>;
    const auto pg_bf16 = vec::mtrue(BF16Tag{});
    vec::details::sme::mopa<
        0, vecops::float32_t, vecops::bfloat16_t, vecops::bfloat16_t>(
        pg_bf16, pg_bf16, vec::zeros(BF16Tag{}), vec::zeros(BF16Tag{}));

    using S8Tag = vec::ScalableTag<int8_t, 0>;
    using U8Tag = vec::ScalableTag<uint8_t, 0>;
    const auto pg_s8 = vec::mtrue(S8Tag{});
    const auto pg_u8 = vec::mtrue(U8Tag{});
    const auto s8 = vec::zeros(S8Tag{});
    const auto u8 = vec::zeros(U8Tag{});
    vec::details::sme::mopa<0, int32_t, int8_t, int8_t>(
        pg_s8, pg_s8, s8, s8);
    vec::details::sme::mopa<0, int32_t, uint8_t, uint8_t>(
        pg_u8, pg_u8, u8, u8);
    vec::details::sme::mopa<0, int32_t, int8_t, uint8_t>(
        pg_s8, pg_u8, s8, u8);
    vec::details::sme::mopa<0, int32_t, uint8_t, int8_t>(
        pg_u8, pg_s8, u8, s8);

#if defined(HAS_SME_F64F64)
    using F64Tag = vec::ScalableTag<vecops::float64_t, 0>;
    const auto pg_f64 = vec::mtrue(F64Tag{});
    vec::details::sme::mopa<
        0, vecops::float64_t, vecops::float64_t, vecops::float64_t>(
        pg_f64, pg_f64, vec::zeros(F64Tag{}), vec::zeros(F64Tag{}));
#endif
  }
};

TEST(VecSMETest, ManualRegionPreservesStateAndDispatchesHelpers) {
  constexpr std::size_t Bytes = 64;
  alignas(64) std::array<uint8_t, Bytes> in8{}, out8{};
  alignas(64) std::array<uint16_t, Bytes / 2> in16{}, out16{};
  alignas(64) std::array<uint32_t, Bytes / 4> in32{}, out32{}, partial32{};
  alignas(64) std::array<uint64_t, Bytes / 8> in64{}, out64{};
  alignas(64) std::array<uint8_t, Bytes> vertical8{};
  alignas(64) std::array<uint16_t, Bytes / 2> vertical16{};
  alignas(64) std::array<uint32_t, Bytes / 4> vertical32{};
  alignas(64) std::array<uint64_t, Bytes / 8> vertical64{};
  for (std::size_t i = 0; i < in8.size(); ++i) in8[i] = i + 1;
  for (std::size_t i = 0; i < in16.size(); ++i) in16[i] = i + 101;
  for (std::size_t i = 0; i < in32.size(); ++i) in32[i] = i + 1001;
  for (std::size_t i = 0; i < in64.size(); ++i) in64[i] = i + 10001;

  const auto before = vec::details::sme::read_svcr();
  vec::details::sme::with_streaming_za(HelperBody{
      in8.data(), in16.data(), in32.data(), in64.data(),
      out8.data(), out16.data(), out32.data(), out64.data(),
      vertical8.data(), vertical16.data(), vertical32.data(),
      vertical64.data(), partial32.data()});
  const auto after = vec::details::sme::read_svcr();

  EXPECT_EQ(before, 0u);
  EXPECT_EQ(after, before);
  EXPECT_EQ(out8, in8);
  EXPECT_EQ(out16, in16);
  EXPECT_EQ(out32, in32);
  EXPECT_EQ(out64, in64);
  EXPECT_EQ(vertical8, in8);
  EXPECT_EQ(vertical16, in16);
  EXPECT_EQ(vertical32, in32);
  EXPECT_EQ(vertical64, in64);
  for (std::size_t lane = 0; lane < partial32.size(); ++lane) {
    EXPECT_EQ(partial32[lane],
              lane < partial32.size() / 2 ? in32[lane] : 0u);
  }
}

} // namespace
