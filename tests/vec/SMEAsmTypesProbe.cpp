// Cross-compilation probe: public SME instruction operands must normalize to
// sizeless ACLE register types, even when the public Vec/Mask is VLS-sized.
// No Vec/Mask is created before SMSTART or survives SMSTOP; no arrays are used.
#include <type_traits>
#include <utility>

#include "vecops/vec/SME.h"

#if defined(HAS_SME)
namespace {
namespace V = vecops::vec;

template <typename T>
using Tag = V::ScalableTag<T, 0>;

template <typename T>
consteval bool sizeless_asm_operands() {
  using Raw = typename V::details::SVERawVector<T>::type;
  return std::is_same_v<
             decltype(V::details::sme::raw_word<T>(
                 std::declval<V::Vec<Tag<T>>>())), Raw> &&
         std::is_same_v<
             decltype(V::details::sme::raw_mask<T>(
                 std::declval<V::Mask<Tag<T>>>())), svbool_t>;
}

static_assert(sizeless_asm_operands<std::uint8_t>());
static_assert(sizeless_asm_operands<std::uint16_t>());
static_assert(sizeless_asm_operands<std::uint32_t>());
static_assert(sizeless_asm_operands<std::uint64_t>());
static_assert(sizeless_asm_operands<std::int8_t>());
static_assert(sizeless_asm_operands<std::int16_t>());
static_assert(sizeless_asm_operands<std::int32_t>());
static_assert(sizeless_asm_operands<std::int64_t>());
static_assert(sizeless_asm_operands<vecops::float16_t>());
static_assert(sizeless_asm_operands<vecops::bfloat16_t>());
static_assert(sizeless_asm_operands<vecops::float32_t>());
static_assert(sizeless_asm_operands<vecops::float64_t>());
} // namespace

#define VECOPS_PROBE_ZA_MOVE(Name, Type)                                  \
extern "C" void Name(const Type* input, Type* output,                     \
                     vecops::nint_t count, std::uint32_t slice) {          \
  V::sme::with_streaming_za([&]() VECOPS_INLINE_LAMBDA_NOEXCEPT {           \
    const auto pg = V::mwhilelt(Tag<Type>{}, 0, count);                    \
    const auto value = V::load(Tag<Type>{}, input, V::opt::masked(pg));    \
    V::sme::zero_za();                                                   \
    V::sme::write_hor<0>(slice, pg, value);                               \
    V::sme::write_ver<0>(slice, pg, value);                               \
    const auto horizontal = V::sme::read_hor<0>(Tag<Type>{}, slice, pg);   \
    V::store(Tag<Type>{}, output, horizontal, V::opt::masked(pg));        \
    const auto vertical = V::sme::read_ver<0>(Tag<Type>{}, slice, pg);     \
    V::store(Tag<Type>{}, output, vertical, V::opt::masked(pg));          \
  });                                                                  \
}

VECOPS_PROBE_ZA_MOVE(za_move_u8, std::uint8_t)
VECOPS_PROBE_ZA_MOVE(za_move_u16, std::uint16_t)
VECOPS_PROBE_ZA_MOVE(za_move_u32, std::uint32_t)
VECOPS_PROBE_ZA_MOVE(za_move_u64, std::uint64_t)
VECOPS_PROBE_ZA_MOVE(za_move_s32, std::int32_t)
VECOPS_PROBE_ZA_MOVE(za_move_f32, vecops::float32_t)
VECOPS_PROBE_ZA_MOVE(za_move_f64, vecops::float64_t)
#undef VECOPS_PROBE_ZA_MOVE

#define VECOPS_PROBE_ZA_MEMORY(Name, Type)                                \
extern "C" void Name(const Type* input, Type* output,                     \
                     vecops::nint_t count, std::uint32_t slice) {          \
  V::sme::with_streaming_za([&]() VECOPS_INLINE_LAMBDA_NOEXCEPT {           \
    const auto pg = V::mwhilelt(Tag<Type>{}, 0, count);                    \
    V::sme::load_hor<0>(slice, pg, input);                                \
    V::sme::store_hor<0>(slice, pg, output);                              \
    V::sme::load_ver<0>(slice, pg, input);                                \
    V::sme::store_ver<0>(slice, pg, output);                              \
  });                                                                  \
}
VECOPS_PROBE_ZA_MEMORY(za_memory_u8, std::uint8_t)
VECOPS_PROBE_ZA_MEMORY(za_memory_u16, std::uint16_t)
VECOPS_PROBE_ZA_MEMORY(za_memory_u32, std::uint32_t)
VECOPS_PROBE_ZA_MEMORY(za_memory_u64, std::uint64_t)
#undef VECOPS_PROBE_ZA_MEMORY

#define VECOPS_PROBE_ZA_ADDHA(Name, Type)                                 \
extern "C" void Name(const Type* input, Type* output, vecops::nint_t count) { \
  V::sme::with_streaming_za([&]() VECOPS_INLINE_LAMBDA_NOEXCEPT {           \
    const auto pg = V::mwhilelt(Tag<Type>{}, 0, count);                    \
    const auto value = V::load(Tag<Type>{}, input, V::opt::masked(pg));    \
    V::sme::zero_za();                                                   \
    V::sme::addha<0, Type>(pg, pg, value);                                \
    V::store(Tag<Type>{}, output, V::sme::read_hor<0>(Tag<Type>{}, 0, pg), \
             V::opt::masked(pg));                                       \
  });                                                                  \
}
VECOPS_PROBE_ZA_ADDHA(za_addha_s32, std::int32_t)
VECOPS_PROBE_ZA_ADDHA(za_addha_u32, std::uint32_t)
#undef VECOPS_PROBE_ZA_ADDHA

#define VECOPS_PROBE_ZA_MOPA(Name, Acc, Left, Right)                      \
extern "C" void Name(const Left* a, const Right* b, Acc* output,          \
                     vecops::nint_t count) {                             \
  V::sme::with_streaming_za([&]() VECOPS_INLINE_LAMBDA_NOEXCEPT {           \
    const auto pga = V::mwhilelt(Tag<Left>{}, 0, count);                   \
    const auto pgb = V::mwhilelt(Tag<Right>{}, 0, count);                  \
    const auto va = V::load(Tag<Left>{}, a, V::opt::masked(pga));          \
    const auto vb = V::load(Tag<Right>{}, b, V::opt::masked(pgb));         \
    V::sme::zero_za();                                                   \
    V::sme::mopa<0, Acc, Left, Right>(pga, pgb, va, vb);                  \
    const auto pg = V::mwhilelt(Tag<Acc>{}, 0, count);                     \
    V::store(Tag<Acc>{}, output, V::sme::read_hor<0>(Tag<Acc>{}, 0, pg),   \
             V::opt::masked(pg));                                       \
  });                                                                  \
}
VECOPS_PROBE_ZA_MOPA(za_mopa_f32, vecops::float32_t, vecops::float32_t, vecops::float32_t)
VECOPS_PROBE_ZA_MOPA(za_mopa_f16, vecops::float32_t, vecops::float16_t, vecops::float16_t)
VECOPS_PROBE_ZA_MOPA(za_mopa_bf16, vecops::float32_t, vecops::bfloat16_t, vecops::bfloat16_t)
VECOPS_PROBE_ZA_MOPA(za_mopa_f64, vecops::float64_t, vecops::float64_t, vecops::float64_t)
VECOPS_PROBE_ZA_MOPA(za_mopa_ss8, std::int32_t, std::int8_t, std::int8_t)
VECOPS_PROBE_ZA_MOPA(za_mopa_uu8, std::uint32_t, std::uint8_t, std::uint8_t)
VECOPS_PROBE_ZA_MOPA(za_mopa_su8, std::int32_t, std::int8_t, std::uint8_t)
VECOPS_PROBE_ZA_MOPA(za_mopa_us8, std::int32_t, std::uint8_t, std::int8_t)
#undef VECOPS_PROBE_ZA_MOPA
#endif
