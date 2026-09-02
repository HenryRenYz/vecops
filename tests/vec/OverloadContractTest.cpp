#include <gtest/gtest.h>

#include <cstdint>
#include <type_traits>

#include "vecops/vec/Vec.h"

namespace vec = vecops::vec;

namespace {

template <typename Op, typename... Args>
inline constexpr bool accepts = requires(Op op, Args... args) {
  op(args...);
};

using FTag = vec::ScalableTag<float>;
using ITag = vec::ScalableTag<int32_t>;
using HTag = vec::Half<FTag>;
using FV = vec::Vec<FTag>;
using IV = vec::Vec<ITag>;
using HV = vec::Vec<HTag>;
using FM = vec::Mask<FTag>;
using IM = vec::Mask<ITag>;
using Indexed = decltype(vec::indexed(std::declval<IV&>()));
using MaskedF = vec::opt::Masked<FM>;
using MaskedI = vec::opt::Masked<IM>;
using MathStrict = decltype(vec::opt::math::strict);
using MathFast = decltype(vec::opt::math::fast);
using MathEstimate = decltype(vec::opt::math::estimate);
using Saturate = decltype(vec::opt::saturate);
using Wrap = decltype(vec::opt::wrap);

using DotToTag = vec::ScalableTag<vecops::float32_t>;
using DotF16Tag = vec::ViewAs<vecops::float16_t, DotToTag>;
using DotBF16Tag = vec::ViewAs<vecops::bfloat16_t, DotToTag>;
using DotF16V = vec::Vec<DotF16Tag>;
using DotBF16V = vec::Vec<DotBF16Tag>;
using DotToV = vec::Vec<DotToTag>;

static_assert(accepts<
              decltype(vec::widening_dot), DotToTag,
              DotF16Tag, DotBF16Tag, DotF16V, DotBF16V>);
static_assert(accepts<
              decltype(vec::widening_dot), DotToTag,
              DotF16Tag, DotBF16Tag, DotF16V, DotBF16V, DotToV>);
static_assert(accepts<
              decltype(vec::widening_dot), DotToTag,
              DotF16V, DotBF16V>);
static_assert(accepts<
              decltype(vec::widening_dot), DotToTag,
              DotF16V, DotBF16V, DotToV>);
static_assert(!accepts<
              decltype(vec::widening_dot), DotToTag,
              DotToTag, DotF16Tag, DotToV, DotF16V>);
static_assert(!accepts<
              decltype(vec::widening_dot), DotF16Tag,
              DotF16Tag, DotBF16Tag, DotF16V, DotBF16V>);

#define CHECK_BINARY_CPO(Name, Tag, Value, Masked)                    \
  static_assert(accepts<decltype(vec::Name), Tag, Value, Value>);     \
  static_assert(accepts<                                              \
                decltype(vec::Name), Tag, Value, Value,              \
                Masked, vec::opt::Zero>);                             \
  static_assert(accepts<decltype(vec::Name), Value, Value>);          \
  static_assert(accepts<                                              \
                decltype(vec::Name), Value, Value,                   \
                Masked, vec::opt::Zero>)

#define CHECK_UNARY_CPO(Name, Tag, Value, Masked)                     \
  static_assert(accepts<decltype(vec::Name), Tag, Value>);            \
  static_assert(accepts<                                              \
                decltype(vec::Name), Tag, Value,                     \
                Masked, vec::opt::Zero>);                             \
  static_assert(accepts<decltype(vec::Name), Value>);                 \
  static_assert(accepts<                                              \
                decltype(vec::Name), Value,                          \
                Masked, vec::opt::Zero>)

CHECK_BINARY_CPO(add, FTag, FV, MaskedF);
CHECK_BINARY_CPO(sub, FTag, FV, MaskedF);
CHECK_BINARY_CPO(mul, FTag, FV, MaskedF);
CHECK_BINARY_CPO(div, FTag, FV, MaskedF);
CHECK_BINARY_CPO(min, FTag, FV, MaskedF);
CHECK_BINARY_CPO(max, FTag, FV, MaskedF);
CHECK_BINARY_CPO(copysign, FTag, FV, MaskedF);
static_assert(accepts<decltype(vec::add), ITag, IV, IV, Saturate>);
static_assert(accepts<decltype(vec::add), IV, IV, Saturate>);
static_assert(accepts<decltype(vec::sub), ITag, IV, IV, Wrap>);
static_assert(accepts<
              decltype(vec::sub), ITag, IV, IV,
              Saturate, MaskedI, vec::opt::Zero>);
CHECK_UNARY_CPO(neg, FTag, FV, MaskedF);
CHECK_UNARY_CPO(abs, FTag, FV, MaskedF);
CHECK_UNARY_CPO(sqrt, FTag, FV, MaskedF);
CHECK_UNARY_CPO(floor, FTag, FV, MaskedF);
CHECK_UNARY_CPO(ceil, FTag, FV, MaskedF);
CHECK_UNARY_CPO(trunc, FTag, FV, MaskedF);
CHECK_UNARY_CPO(round, FTag, FV, MaskedF);
CHECK_UNARY_CPO(round_even, FTag, FV, MaskedF);
CHECK_UNARY_CPO(nearbyint, FTag, FV, MaskedF);
CHECK_UNARY_CPO(rint, FTag, FV, MaskedF);
CHECK_UNARY_CPO(rcp, FTag, FV, MaskedF);
CHECK_UNARY_CPO(rsqrt, FTag, FV, MaskedF);
CHECK_UNARY_CPO(exp, FTag, FV, MaskedF);
CHECK_UNARY_CPO(exp_strict, FTag, FV, MaskedF);
CHECK_UNARY_CPO(exp_fast, FTag, FV, MaskedF);
CHECK_UNARY_CPO(exp_est, FTag, FV, MaskedF);
CHECK_UNARY_CPO(exp_neg, FTag, FV, MaskedF);
CHECK_UNARY_CPO(exp_neg_strict, FTag, FV, MaskedF);
CHECK_UNARY_CPO(exp_neg_fast, FTag, FV, MaskedF);
CHECK_UNARY_CPO(exp_neg_est, FTag, FV, MaskedF);
CHECK_UNARY_CPO(log, FTag, FV, MaskedF);
CHECK_UNARY_CPO(log_strict, FTag, FV, MaskedF);
CHECK_UNARY_CPO(log_fast, FTag, FV, MaskedF);
CHECK_UNARY_CPO(log_est, FTag, FV, MaskedF);
CHECK_UNARY_CPO(log2, FTag, FV, MaskedF);
CHECK_UNARY_CPO(log2_strict, FTag, FV, MaskedF);
CHECK_UNARY_CPO(log2_fast, FTag, FV, MaskedF);
CHECK_UNARY_CPO(log2_est, FTag, FV, MaskedF);
CHECK_UNARY_CPO(log10, FTag, FV, MaskedF);
CHECK_UNARY_CPO(log10_strict, FTag, FV, MaskedF);
CHECK_UNARY_CPO(log10_fast, FTag, FV, MaskedF);
CHECK_UNARY_CPO(log10_est, FTag, FV, MaskedF);
CHECK_UNARY_CPO(sin, FTag, FV, MaskedF);
CHECK_UNARY_CPO(sin_strict, FTag, FV, MaskedF);
CHECK_UNARY_CPO(sin_fast, FTag, FV, MaskedF);
CHECK_UNARY_CPO(sin_est, FTag, FV, MaskedF);
CHECK_UNARY_CPO(cos, FTag, FV, MaskedF);
CHECK_UNARY_CPO(cos_strict, FTag, FV, MaskedF);
CHECK_UNARY_CPO(cos_fast, FTag, FV, MaskedF);
CHECK_UNARY_CPO(cos_est, FTag, FV, MaskedF);
CHECK_UNARY_CPO(tan, FTag, FV, MaskedF);
CHECK_UNARY_CPO(tan_strict, FTag, FV, MaskedF);
CHECK_UNARY_CPO(tan_fast, FTag, FV, MaskedF);
CHECK_UNARY_CPO(tan_est, FTag, FV, MaskedF);
CHECK_UNARY_CPO(sinpi, FTag, FV, MaskedF);
CHECK_UNARY_CPO(sinpi_strict, FTag, FV, MaskedF);
CHECK_UNARY_CPO(sinpi_fast, FTag, FV, MaskedF);
CHECK_UNARY_CPO(sinpi_est, FTag, FV, MaskedF);
CHECK_UNARY_CPO(cospi, FTag, FV, MaskedF);
CHECK_UNARY_CPO(cospi_strict, FTag, FV, MaskedF);
CHECK_UNARY_CPO(cospi_fast, FTag, FV, MaskedF);
CHECK_UNARY_CPO(cospi_est, FTag, FV, MaskedF);
CHECK_UNARY_CPO(tanpi, FTag, FV, MaskedF);
CHECK_UNARY_CPO(tanpi_strict, FTag, FV, MaskedF);
CHECK_UNARY_CPO(tanpi_fast, FTag, FV, MaskedF);
CHECK_UNARY_CPO(tanpi_est, FTag, FV, MaskedF);

static_assert(accepts<decltype(vec::sincos), FTag, FV, FV, FV>);
static_assert(accepts<decltype(vec::sincos), FV, FV, FV>);
static_assert(accepts<decltype(vec::sincos_strict), FTag, FV, FV, FV>);
static_assert(accepts<decltype(vec::sincos_fast), FV, FV, FV, MaskedF>);
static_assert(accepts<decltype(vec::sincos_est), FV, FV, FV>);
static_assert(accepts<
              decltype(vec::sincospi), FTag, FV, FV, FV,
              MaskedF, vec::opt::Zero, MathFast>);
static_assert(accepts<decltype(vec::sincospi_strict), FTag, FV, FV, FV>);
static_assert(accepts<decltype(vec::sincospi_fast), FV, FV, FV, MaskedF>);
static_assert(accepts<decltype(vec::sincospi_est), FV, FV, FV>);
static_assert(!accepts<
              decltype(vec::sincos), FTag, FV, FV, FV,
              MathStrict, MathFast>);
static_assert(!accepts<
              decltype(vec::sincos_fast), FV, FV, FV, MathEstimate>);

static_assert(accepts<decltype(vec::rsqrt), FTag, FV, MathStrict>);
static_assert(accepts<decltype(vec::rsqrt), FV, MathFast>);
static_assert(accepts<
              decltype(vec::rcp), FTag, FV,
              MaskedF, MathEstimate, vec::opt::Zero>);
static_assert(!accepts<
              decltype(vec::rsqrt), FTag, FV, MathStrict, MathFast>);
static_assert(!accepts<decltype(vec::rsqrt_fast), FTag, FV, MathFast>);
static_assert(!accepts<decltype(vec::rcp_est), FV, MathStrict>);
static_assert(!accepts<
              decltype(vec::rcp), FTag, FV, MaskedF, MaskedF>);
static_assert(accepts<decltype(vec::exp), FTag, FV, MathStrict>);
static_assert(accepts<decltype(vec::exp), FV, MathFast>);
static_assert(accepts<
              decltype(vec::exp_neg), FTag, FV,
              MaskedF, MathEstimate, vec::opt::Zero>);
static_assert(!accepts<
              decltype(vec::exp), FTag, FV, MathStrict, MathFast>);
static_assert(!accepts<decltype(vec::exp_fast), FTag, FV, MathFast>);
static_assert(!accepts<decltype(vec::exp_est), FV, MathStrict>);
static_assert(!accepts<
              decltype(vec::exp_neg_strict), FTag, FV, MathEstimate>);
static_assert(!accepts<decltype(vec::exp_neg_fast), FV, MathFast>);
static_assert(accepts<decltype(vec::sin), FTag, FV, MathStrict>);
static_assert(accepts<decltype(vec::tanpi), FV, MathEstimate>);
static_assert(!accepts<
              decltype(vec::cos), FTag, FV, MathStrict, MathFast>);
static_assert(!accepts<decltype(vec::sin_fast), FV, MathEstimate>);
CHECK_BINARY_CPO(bit_and, ITag, IV, MaskedI);
CHECK_BINARY_CPO(bit_or, ITag, IV, MaskedI);
CHECK_BINARY_CPO(bit_xor, ITag, IV, MaskedI);
CHECK_BINARY_CPO(bit_andnot, ITag, IV, MaskedI);
CHECK_UNARY_CPO(bit_not, ITag, IV, MaskedI);

#undef CHECK_BINARY_CPO
#undef CHECK_UNARY_CPO

#define CHECK_FMA_CPO(Name)                                            \
  static_assert(accepts<decltype(vec::Name), FTag, FV, FV, FV>);       \
  static_assert(accepts<                                               \
                decltype(vec::Name), FTag, FV, FV, FV,                \
                MaskedF, vec::opt::Zero>);                             \
  static_assert(accepts<decltype(vec::Name), FV, FV, FV>)

CHECK_FMA_CPO(fmadd);
CHECK_FMA_CPO(fmsub);
CHECK_FMA_CPO(fnmadd);
CHECK_FMA_CPO(fnmsub);
CHECK_FMA_CPO(clamp);

static_assert(!accepts<decltype(vec::clamp), FTag, FV, FV, IV>);
static_assert(!accepts<decltype(vec::clamp), FV, FV, IV>);

#undef CHECK_FMA_CPO

#define CHECK_COMPARISON_CPO(Name)                                     \
  static_assert(accepts<decltype(vec::Name), FTag, FV, FV>);           \
  static_assert(accepts<                                               \
                decltype(vec::Name), FTag, FV, FV,                    \
                MaskedF, vec::opt::Zero>);                             \
  static_assert(accepts<decltype(vec::Name), FV, FV>)

CHECK_COMPARISON_CPO(cmpeq);
CHECK_COMPARISON_CPO(cmpne);
CHECK_COMPARISON_CPO(cmplt);
CHECK_COMPARISON_CPO(cmpgt);
CHECK_COMPARISON_CPO(cmple);
CHECK_COMPARISON_CPO(cmpge);

#undef CHECK_COMPARISON_CPO

#define CHECK_CLASSIFICATION_CPO(Name)                                 \
  static_assert(accepts<decltype(vec::Name), FTag, FV>);               \
  static_assert(accepts<                                               \
                decltype(vec::Name), FTag, FV,                        \
                MaskedF, vec::opt::Zero>);                             \
  static_assert(accepts<decltype(vec::Name), FV>)

CHECK_CLASSIFICATION_CPO(isnan);
CHECK_CLASSIFICATION_CPO(isposinf);
CHECK_CLASSIFICATION_CPO(isneginf);
CHECK_CLASSIFICATION_CPO(isinf);
CHECK_CLASSIFICATION_CPO(isfinite);
CHECK_CLASSIFICATION_CPO(isnormal);
CHECK_CLASSIFICATION_CPO(signbit);

#undef CHECK_CLASSIFICATION_CPO

static_assert(accepts<decltype(vec::fill), FTag, float>);
static_assert(accepts<
              decltype(vec::fill), FTag, float,
              vec::opt::First, vec::opt::Zero>);
static_assert(accepts<decltype(vec::mfill), FTag, bool>);
static_assert(accepts<decltype(vec::mwhilelt), FTag, vecops::nint_t,
                      vecops::nint_t>);
static_assert(accepts<decltype(vec::mwhilele), FTag, vecops::nint_t,
                      vecops::nint_t>);
static_assert(accepts<decltype(vec::mwhilege), FTag, vecops::nint_t,
                      vecops::nint_t>);
static_assert(accepts<decltype(vec::mwhilegt), FTag, vecops::nint_t,
                      vecops::nint_t>);
static_assert(accepts<decltype(vec::zeros), FTag>);
static_assert(accepts<decltype(vec::mtrue), FTag>);
static_assert(accepts<decltype(vec::mfalse), FTag>);
static_assert(accepts<decltype(vec::blend), FTag, FV, FM, FV>);
static_assert(accepts<decltype(vec::blend), FV, FM, FV>);
static_assert(accepts<decltype(vec::mask_and), FTag, FM, FM>);
static_assert(accepts<decltype(vec::mask_or), FTag, FM, FM>);
static_assert(accepts<decltype(vec::mask_xor), FTag, FM, FM>);
static_assert(accepts<decltype(vec::mask_andnot), FTag, FM, FM>);
static_assert(accepts<decltype(vec::mask_not), FTag, FM>);
static_assert(accepts<decltype(vec::mask_all), FTag, FM>);
static_assert(accepts<decltype(vec::mask_any), FTag, FM>);
static_assert(accepts<decltype(vec::mask_none), FTag, FM>);
static_assert(accepts<decltype(vec::mask_count), FTag, FM>);
static_assert(accepts<decltype(vec::mask_first), FTag, FM>);
static_assert(accepts<decltype(vec::mask_last), FTag, FM>);
static_assert(accepts<decltype(vec::iota), FTag, float>);
static_assert(accepts<decltype(vec::iota), FTag, float, float>);
static_assert(!accepts<decltype(vec::mask_and), FM, FM>);
static_assert(accepts<decltype(vec::get), FTag, FV, vecops::nint_t>);
static_assert(accepts<decltype(vec::get), FTag, FM, vecops::nint_t>);
static_assert(accepts<decltype(vec::get), FV, vecops::nint_t>);
static_assert(!accepts<decltype(vec::get), FM, vecops::nint_t>);
static_assert(accepts<decltype(vec::set), FTag, FV, vecops::nint_t, float>);
static_assert(accepts<decltype(vec::set), FTag, FM, vecops::nint_t, bool>);
static_assert(accepts<decltype(vec::set), FV, vecops::nint_t, float>);
static_assert(accepts<decltype(vec::bitcast), ITag, FTag, FV>);
static_assert(accepts<decltype(vec::lower), FTag, FV>);
static_assert(accepts<decltype(vec::upper), FTag, FV>);
static_assert(accepts<decltype(vec::concat), FTag, HV, HV>);
static_assert(accepts<decltype(vec::even), FTag, FV>);
static_assert(accepts<decltype(vec::odd), FTag, FV>);
static_assert(accepts<decltype(vec::concat_even), FTag, FV, FV>);
static_assert(accepts<decltype(vec::concat_odd), FTag, FV, FV>);
static_assert(accepts<decltype(vec::interleave), FTag, HV, HV>);
static_assert(accepts<decltype(vec::interleave_even), FTag, FV, FV>);
static_assert(accepts<decltype(vec::interleave_even), FV, FV>);
static_assert(accepts<decltype(vec::interleave_odd), FTag, FV, FV>);
static_assert(accepts<decltype(vec::interleave_odd), FV, FV>);
static_assert(accepts<decltype(vec::local_interleave_lower), FTag, FV, FV>);
static_assert(accepts<decltype(vec::local_interleave_lower), FV, FV>);
static_assert(accepts<decltype(vec::local_interleave_upper), FTag, FV, FV>);
static_assert(accepts<decltype(vec::local_interleave_upper), FV, FV>);
static_assert(accepts<decltype(vec::shuf), FTag, FV, IV>);
static_assert(accepts<decltype(vec::shuf), FV, IV>);
static_assert(accepts<decltype(vec::local_shuf), FTag, FV, IV>);
static_assert(accepts<decltype(vec::local_shuf), FV, IV>);

static_assert(accepts<decltype(vec::shl), ITag, IV, int>);
static_assert(accepts<decltype(vec::shl), ITag, IV,
                      vecops::meta::Const<3>>);
static_assert(!accepts<decltype(vec::shl), ITag, IV,
                       vecops::meta::Const<-1>>);
static_assert(!accepts<decltype(vec::shl), ITag, IV,
                       vecops::meta::Dynamic<1>>);
static_assert(accepts<decltype(vec::shl), ITag, IV, IV>);
static_assert(accepts<decltype(vec::shl), IV, int>);
static_assert(accepts<decltype(vec::shr), ITag, IV, int>);
static_assert(accepts<decltype(vec::shr), ITag, IV,
                      vecops::meta::Const<3>>);
static_assert(!accepts<decltype(vec::shr), ITag, IV,
                       vecops::meta::Const<-1>>);
static_assert(!accepts<decltype(vec::shr), ITag, IV,
                       vecops::meta::Dynamic<1>>);
static_assert(accepts<decltype(vec::shr), ITag, IV, IV>);
static_assert(accepts<decltype(vec::shr), IV, int>);
static_assert(accepts<decltype(vec::popcount), ITag, IV>);
static_assert(accepts<decltype(vec::popcount), IV>);
static_assert(accepts<decltype(vec::countl_zero), ITag, IV>);
static_assert(accepts<decltype(vec::countl_one), IV>);
static_assert(accepts<decltype(vec::countr_zero), ITag, IV>);
static_assert(accepts<decltype(vec::countr_one), IV>);
static_assert(!accepts<decltype(vec::popcount), FTag, FV>);
static_assert(accepts<decltype(vec::rotl), ITag, IV, int>);
static_assert(accepts<decltype(vec::rotl), ITag, IV,
                      vecops::meta::Const<-1>>);
static_assert(accepts<decltype(vec::rotl), ITag, IV, IV>);
static_assert(accepts<decltype(vec::rotl), IV, int>);
static_assert(accepts<decltype(vec::rotr), ITag, IV, int>);
static_assert(accepts<decltype(vec::rotr), ITag, IV,
                      vecops::meta::Const<-1>>);
static_assert(accepts<decltype(vec::rotr), ITag, IV, IV>);
static_assert(accepts<decltype(vec::rotr), IV, int>);
static_assert(!accepts<decltype(vec::rotl), FTag, FV, int>);
static_assert(!accepts<decltype(vec::rotl), ITag, IV,
                       vecops::meta::Dynamic<1>>);

static_assert(accepts<decltype(vec::reduce_add), FTag, FV>);
static_assert(accepts<decltype(vec::reduce_add), FTag, FV, MaskedF>);
static_assert(accepts<decltype(vec::reduce_max), FTag, FV>);
static_assert(accepts<decltype(vec::reduce_max), FTag, FV, MaskedF>);
static_assert(accepts<decltype(vec::reduce_min), FTag, FV>);
static_assert(accepts<decltype(vec::reduce_min), FTag, FV, MaskedF>);
static_assert(!accepts<decltype(vec::reduce_add), FV>);
static_assert(!accepts<decltype(vec::reduce_max), FV>);
static_assert(!accepts<decltype(vec::reduce_min), FV>);

static_assert(accepts<decltype(vec::load), FTag, const float*>);
static_assert(accepts<decltype(vec::load), FTag, const float*, MaskedF,
                      vec::opt::Zero>);
static_assert(accepts<decltype(vec::load), FTag, const float*, Indexed>);
static_assert(accepts<decltype(vec::store), FTag, float*, FV>);
static_assert(accepts<decltype(vec::store), FTag, float*, FV, MaskedF>);
static_assert(!accepts<decltype(vec::load), const float*>);
static_assert(!accepts<decltype(vec::store), float*, FV>);
static_assert(accepts<decltype(vec::prefetch), FTag, const float*>);

static_assert(accepts<decltype(vec::convert), ITag, FTag, FV>);
static_assert(accepts<decltype(vec::convert), ITag, FTag, FM>);
static_assert(!accepts<decltype(vec::convert), FTag, FV>);
static_assert(accepts<decltype(vec::load_convert), FTag, const int16_t*>);
static_assert(accepts<
              decltype(vec::load_convert), FTag, const int16_t*,
              vec::cvt::Ordered, vec::cvt::Saturate,
              vec::mem::Unaligned, vec::mem::Temporal>);
static_assert(accepts<decltype(vec::store_convert), FTag, int16_t*, FV>);
static_assert(accepts<
              decltype(vec::store_convert), FTag, int16_t*, FV,
              vec::cvt::Ordered, vec::cvt::Saturate,
              vec::mem::Unaligned, vec::mem::Temporal, vec::mem::Packed>);

TEST(VecOverloadContractTest, PublicCPOContractTableCompiles) {
  SUCCEED();
}

} // namespace
