// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
// Native oracle companion to SMEAsmTypesProbe.cpp. Every Vec/Mask is local
// to a single manual state owner; the caller must select equal VL/SVL in VLS.
#include <cstdint>
#include "vecops/vec/SME.h"

#if defined(HAS_SME)
namespace V = vecops::vec;
template <typename T> using Tag = V::ScalableTag<T, 0>;

extern "C" std::uint64_t za_asm_read_svcr() { return V::sme::read_svcr(); }
extern "C" std::uint64_t za_asm_sve_bytes() {
  // Query actual ordinary VL even in VLS: svcntb() can fold to the compile
  // width and would not independently attest the caller's runtime state.
  std::uint64_t value;
  asm volatile("cntb %0" : "=r"(value));
  return value;
}
extern "C" std::uint64_t za_asm_sme_bytes() { return V::sme::streaming_vector_bytes(); }

#define VECOPS_NATIVE_ZA_MOVE(Name, Type)                                 \
extern "C" void Name(const Type* input, Type* output,                    \
                     vecops::nint_t count, bool vertical) {              \
  V::sme::with_streaming_za([&]() VECOPS_INLINE_LAMBDA_NOEXCEPT {          \
    const auto pg = V::mwhilelt(Tag<Type>{}, 0, count);                   \
    const auto value = V::load(Tag<Type>{}, input, V::opt::masked(pg));   \
    V::sme::zero_za();                                                  \
    if (vertical) {                                                    \
      V::sme::write_ver<0>(0, pg, value);                               \
      V::store(Tag<Type>{}, output,                                    \
               V::sme::read_ver<0>(Tag<Type>{}, 0, pg), V::opt::masked(pg)); \
    } else {                                                           \
      V::sme::write_hor<0>(0, pg, value);                               \
      V::store(Tag<Type>{}, output,                                    \
               V::sme::read_hor<0>(Tag<Type>{}, 0, pg), V::opt::masked(pg)); \
    }                                                                  \
  });                                                                  \
}
VECOPS_NATIVE_ZA_MOVE(za_native_move_u8, std::uint8_t)
VECOPS_NATIVE_ZA_MOVE(za_native_move_u16, std::uint16_t)
VECOPS_NATIVE_ZA_MOVE(za_native_move_u32, std::uint32_t)
VECOPS_NATIVE_ZA_MOVE(za_native_move_u64, std::uint64_t)
VECOPS_NATIVE_ZA_MOVE(za_native_move_s32, std::int32_t)
VECOPS_NATIVE_ZA_MOVE(za_native_move_f32, vecops::float32_t)
VECOPS_NATIVE_ZA_MOVE(za_native_move_f64, vecops::float64_t)
#undef VECOPS_NATIVE_ZA_MOVE

#define VECOPS_NATIVE_ZA_MEMORY(Name, Type)                               \
extern "C" void Name(const Type* input, Type* output,                    \
                     vecops::nint_t count, bool vertical) {              \
  V::sme::with_streaming_za([&]() VECOPS_INLINE_LAMBDA_NOEXCEPT {          \
    const auto pg = V::mwhilelt(Tag<Type>{}, 0, count);                   \
    V::sme::zero_za();                                                  \
    if (vertical) {                                                    \
      V::sme::load_ver<0>(0, pg, input);                                \
      V::sme::store_ver<0>(0, pg, output);                              \
    } else {                                                           \
      V::sme::load_hor<0>(0, pg, input);                                \
      V::sme::store_hor<0>(0, pg, output);                              \
    }                                                                  \
  });                                                                  \
}
VECOPS_NATIVE_ZA_MEMORY(za_native_memory_u8, std::uint8_t)
VECOPS_NATIVE_ZA_MEMORY(za_native_memory_u16, std::uint16_t)
VECOPS_NATIVE_ZA_MEMORY(za_native_memory_u32, std::uint32_t)
VECOPS_NATIVE_ZA_MEMORY(za_native_memory_u64, std::uint64_t)
#undef VECOPS_NATIVE_ZA_MEMORY

#define VECOPS_NATIVE_ZA_MOPA(Name, Acc, Left, Right, Group)              \
extern "C" void Name(const Left* a, const Right* b, Acc* output,         \
                     vecops::nint_t rows, vecops::nint_t columns) {       \
  V::sme::with_streaming_za([&]() VECOPS_INLINE_LAMBDA_NOEXCEPT {          \
    const auto pa = V::mwhilelt(Tag<Left>{}, 0, rows * Group);           \
    const auto pb = V::mwhilelt(Tag<Right>{}, 0, columns * Group);        \
    const auto va = V::load(Tag<Left>{}, a, V::opt::masked(pa));          \
    const auto vb = V::load(Tag<Right>{}, b, V::opt::masked(pb));         \
    const auto pc = V::mwhilelt(Tag<Acc>{}, 0, columns);                 \
    const auto lanes = V::size(Tag<Acc>{});                             \
    V::sme::zero_za();                                                  \
    V::sme::mopa<0, Acc, Left, Right>(pa, pb, va, vb);                   \
    for (vecops::nint_t row = 0; row < rows; ++row) {                    \
      V::store(Tag<Acc>{}, output + row * lanes,                       \
               V::sme::read_hor<0>(Tag<Acc>{},                         \
                   static_cast<std::uint32_t>(row), pc), V::opt::masked(pc)); \
    }                                                                  \
  });                                                                  \
}
VECOPS_NATIVE_ZA_MOPA(za_native_mopa_f32, vecops::float32_t, vecops::float32_t, vecops::float32_t, 1)
VECOPS_NATIVE_ZA_MOPA(za_native_mopa_f16, vecops::float32_t, vecops::float16_t, vecops::float16_t, 2)
VECOPS_NATIVE_ZA_MOPA(za_native_mopa_bf16, vecops::float32_t, vecops::bfloat16_t, vecops::bfloat16_t, 2)
VECOPS_NATIVE_ZA_MOPA(za_native_mopa_f64, vecops::float64_t, vecops::float64_t, vecops::float64_t, 1)
VECOPS_NATIVE_ZA_MOPA(za_native_mopa_ss8, std::int32_t, std::int8_t, std::int8_t, 4)
VECOPS_NATIVE_ZA_MOPA(za_native_mopa_uu8, std::uint32_t, std::uint8_t, std::uint8_t, 4)
VECOPS_NATIVE_ZA_MOPA(za_native_mopa_su8, std::int32_t, std::int8_t, std::uint8_t, 4)
VECOPS_NATIVE_ZA_MOPA(za_native_mopa_us8, std::int32_t, std::uint8_t, std::int8_t, 4)
#undef VECOPS_NATIVE_ZA_MOPA
#endif
