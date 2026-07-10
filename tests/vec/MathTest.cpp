#include <gtest/gtest.h>

#include <algorithm>
#include <bit>
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <random>
#include <type_traits>
#include <vector>

#include "TestUtils.h"
#include "vecops/vec/Vec.h"

using namespace vecops;
using namespace vecops::vec;

namespace {

enum class Tier { Strict, Fast, Estimate };

class ScopedNearestRounding {
 public:
  ScopedNearestRounding() : old_(std::fegetround()) { std::fesetround(FE_TONEAREST); }
  ~ScopedNearestRounding() { std::fesetround(old_); }
 private:
  int old_;
};

template <typename E>
double as_double(E x) { return static_cast<double>(x); }

template <typename E>
E reference_exp(E x) {
  if constexpr (std::is_same_v<E, vecops::float16_t> ||
                std::is_same_v<E, vecops::bfloat16_t>) {
    return E(static_cast<float>(std::exp(static_cast<long double>(static_cast<float>(x)))));
  } else {
    return static_cast<E>(std::exp(static_cast<long double>(x)));
  }
}

template <typename E>
uint64_t bits(E x) {
  if constexpr (std::is_same_v<E, float32_t>) {
    uint32_t u;
    std::memcpy(&u, &x, sizeof(u));
    return u;
  } else if constexpr (std::is_same_v<E, float64_t>) {
    uint64_t u;
    std::memcpy(&u, &x, sizeof(u));
    return u;
  } else {
    return x.to_bits();
  }
}

template <typename E>
uint64_t ulp_distance(E a, E b) {
  if (std::isnan(as_double(a)) && std::isnan(as_double(b))) return 0;
  const auto aa = bits(a), bb = bits(b);
  return aa > bb ? aa - bb : bb - aa;
}

template <Tier tier, TLV_DECL_VEC(V)>
V call_exp(V v) {
  if constexpr (tier == Tier::Strict) return vecops::vec::exp(v);
  else if constexpr (tier == Tier::Fast) return exp_fast(v);
  else return exp_est(v);
}

template <Tier tier, TLV_DECL_VEC(V)>
V call_exp_neg(V v) {
  if constexpr (tier == Tier::Strict) return vecops::vec::exp_neg(v);
  else if constexpr (tier == Tier::Fast) return exp_neg_fast(v);
  else return exp_neg_est(v);
}

template <Tier tier, TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
V call_exp(V v, Mask<T> m, V d) {
  if constexpr (tier == Tier::Strict) return vecops::vec::exp(v, m, d);
  else if constexpr (tier == Tier::Fast) return exp_fast(v, m, d);
  else return exp_est(v, m, d);
}

template <Tier tier, TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
V call_exp_neg(V v, Mask<T> m, V d) {
  if constexpr (tier == Tier::Strict) return vecops::vec::exp_neg(v, m, d);
  else if constexpr (tier == Tier::Fast) return exp_neg_fast(v, m, d);
  else return exp_neg_est(v, m, d);
}

template <typename E>
bool is_normal_result(E x) {
  const double v = as_double(x);
  return std::isfinite(v) && v >= as_double(std::numeric_limits<E>::min());
}

template <Tier tier, typename E>
void check_accuracy(E input, E actual) {
  const E expected = reference_exp(input);
  ASSERT_EQ(std::isnan(as_double(expected)), std::isnan(as_double(actual)))
      << "x=" << as_double(input);
  if (std::isnan(as_double(expected))) return;
#ifdef VECOPS_MATH_ASSUME_VALID_INPUTS
  if (!std::isfinite(as_double(input)) || !is_normal_result(expected)) return;
#endif
#ifndef VECOPS_PRESERVE_SUBNORMALS
  const double expected_value = as_double(expected);
  if (expected_value > 0.0 &&
      expected_value < as_double(std::numeric_limits<E>::min()) &&
      as_double(actual) == 0.0) {
    return;
  }
#endif
  if constexpr (tier == Tier::Strict) {
    EXPECT_LE(ulp_distance(expected, actual), 1u) << "x=" << as_double(input)
        << " expected=" << as_double(expected) << " actual=" << as_double(actual);
  } else if constexpr (tier == Tier::Fast) {
    if (is_normal_result(expected)) {
      EXPECT_LE(ulp_distance(expected, actual), 4u) << "x=" << as_double(input)
          << " expected=" << as_double(expected) << " actual=" << as_double(actual);
    }
  } else if (is_normal_result(expected)) {
    const double rel = std::abs(as_double(actual) / as_double(expected) - 1.0);
    EXPECT_LE(rel, 0.006) << "x=" << as_double(input) << " rel=" << rel;
  }
}

template <Tier tier, typename Tag>
void run_values(Tag t, const std::vector<TypeOf<Tag>>& values) {
  ScopedNearestRounding rounding;
  using E = TypeOf<Tag>;
  const nint_t n = size(t);
  std::vector<E> in(static_cast<size_t>(n));
  std::vector<E> out(static_cast<size_t>(n));
  for (size_t pos = 0; pos < values.size(); pos += static_cast<size_t>(n)) {
    for (nint_t i = 0; i < n; ++i) in[static_cast<size_t>(i)] = values[(pos + i) % values.size()];
    storeu(t, out.data(), call_exp<tier>(loadu(t, in.data())));
    const size_t count = std::min(static_cast<size_t>(n), values.size() - pos);
    for (size_t i = 0; i < count; ++i) check_accuracy<tier>(in[i], out[i]);
  }
}

template <Tier tier, typename Tag>
void run_neg_values(Tag t, const std::vector<TypeOf<Tag>>& values) {
  ScopedNearestRounding rounding;
  using E = TypeOf<Tag>;
  const nint_t n = size(t);
  std::vector<E> in(static_cast<size_t>(n));
  std::vector<E> out(static_cast<size_t>(n));
  for (size_t pos = 0; pos < values.size(); pos += static_cast<size_t>(n)) {
    for (nint_t i = 0; i < n; ++i) in[static_cast<size_t>(i)] = values[(pos + i) % values.size()];
    storeu(t, out.data(), call_exp_neg<tier>(loadu(t, in.data())));
    const size_t count = std::min(static_cast<size_t>(n), values.size() - pos);
    for (size_t i = 0; i < count; ++i) {
      ASSERT_FALSE(std::isnan(as_double(in[i])));
      ASSERT_LE(as_double(in[i]), 0.0);
      check_accuracy<tier>(in[i], out[i]);
      EXPECT_GE(as_double(out[i]), 0.0);
      EXPECT_LE(as_double(out[i]), 1.0);
    }
  }
}

template <typename E>
void check_subnormal_and_thresholds() {
  ScalableTag<E, 0> t;
  constexpr bool f32 = std::is_same_v<E, float32_t>;
  const E normal_log = E(f32 ? -87.3365447505531 : -708.3964185322641);
  const E zero_log = E(f32 ? -103.972084045410 : -745.1332191019411);
  const E overflow_log = E(f32 ? 88.72283905206835 : 709.782712893384);
  std::vector<E> values;
  for (E center : {normal_log, zero_log, overflow_log}) {
    values.push_back(std::nextafter(center, -std::numeric_limits<E>::infinity()));
    values.push_back(center);
    values.push_back(std::nextafter(center, std::numeric_limits<E>::infinity()));
  }
  for (int i = 0; i <= 8192; ++i) {
    values.push_back(E(as_double(zero_log) +
                       (as_double(normal_log) - as_double(zero_log)) * i / 8192.0));
  }
  run_values<Tier::Strict>(t, values);

  const nint_t n = size(t);
  std::vector<E> in(static_cast<size_t>(n), E(as_double(normal_log) - 0.25));
  std::vector<E> out(static_cast<size_t>(n));
  auto v = loadu(t, in.data());
  // Fast tiers may either retain a subnormal when FTZ is disabled by the test
  // process or flush it to zero in the library's default execution mode.
  storeu(t, out.data(), exp_fast(v));
  for (E y : out) {
    EXPECT_GE(as_double(y), 0.0);
    EXPECT_LT(as_double(y), as_double(std::numeric_limits<E>::min()));
  }
  storeu(t, out.data(), exp_est(v));
  for (E y : out) {
    EXPECT_GE(as_double(y), 0.0);
    EXPECT_LT(as_double(y), as_double(std::numeric_limits<E>::min()));
  }
}

template <Tier tier, typename E>
void check_monotonic() {
  ScalableTag<E, 0> t;
  const nint_t n = size(t);
  std::vector<E> in(static_cast<size_t>(n)), out(static_cast<size_t>(n));
  double previous = 0.0;
  bool first = true;
  constexpr int count = 16384;
  for (int base = 0; base < count; base += static_cast<int>(n)) {
    for (nint_t i = 0; i < n; ++i) {
      const int index = std::min(base + static_cast<int>(i), count - 1);
      in[static_cast<size_t>(i)] = E(-80.0 + 160.0 * index / (count - 1));
    }
    storeu(t, out.data(), call_exp<tier>(loadu(t, in.data())));
    for (nint_t i = 0; i < n && base + i < count; ++i) {
      const double current = as_double(out[static_cast<size_t>(i)]);
      if (!first) EXPECT_GE(current, previous) << "index=" << base + i;
      previous = current;
      first = false;
    }
  }
}

template <typename E>
std::vector<E> regular_samples() {
  std::vector<E> v;
  const double lo = std::is_same_v<E, float64_t> ? -708.0 :
                    std::is_same_v<E, float32_t> ? -87.0 : -9.0;
  const double hi = std::is_same_v<E, float64_t> ? 709.0 :
                    std::is_same_v<E, float32_t> ? 88.0 : 10.0;
  constexpr int count = std::is_same_v<E, float64_t> ? 12000 : 24000;
  for (int i = 0; i <= count; ++i) {
    v.push_back(static_cast<E>(lo + (hi - lo) * i / count));
  }
  std::mt19937_64 rng(0x51D4E77u);
  std::uniform_real_distribution<double> dist(lo, hi);
  for (int i = 0; i < count; ++i) v.push_back(static_cast<E>(dist(rng)));
  for (int q = -128; q <= 128; ++q) {
    E x = static_cast<E>((q + 0.5) * 0.6931471805599453094);
    v.push_back(std::nextafter(x, -std::numeric_limits<E>::infinity()));
    v.push_back(x);
    v.push_back(std::nextafter(x, std::numeric_limits<E>::infinity()));
  }
  return v;
}

template <typename E>
std::vector<E> all_low_precision_values() {
  std::vector<E> v;
  v.reserve(65536);
  for (uint32_t i = 0; i <= 0xffffu; ++i) {
    E x = E::from_bits(static_cast<uint16_t>(i));
#ifdef VECOPS_MATH_ASSUME_VALID_INPUTS
    if (!std::isfinite(as_double(x))) continue;
#endif
    v.push_back(x);
  }
  return v;
}

template <typename E, int POW2>
void run_regular_tiers() {
  ScalableTag<E, POW2> t;
  auto samples = regular_samples<E>();
  run_values<Tier::Strict>(t, samples);
  run_values<Tier::Fast>(t, samples);
  run_values<Tier::Estimate>(t, samples);
}

template <typename E>
void run_low_precision_tiers() {
  ScalableTag<E, 0> t;
  auto samples = all_low_precision_values<E>();
  run_values<Tier::Strict>(t, samples);
  run_values<Tier::Fast>(t, samples);
  run_values<Tier::Estimate>(t, samples);
}

template <typename E, int POW2>
void run_negative_tiers() {
  ScalableTag<E, POW2> t;
  auto samples = regular_samples<E>();
  samples.erase(std::remove_if(samples.begin(), samples.end(), [](E x) {
    return std::isnan(as_double(x)) || as_double(x) > 0.0;
  }), samples.end());
  samples.push_back(E(0.0));
  samples.push_back(E(-0.0));
  samples.push_back(-std::numeric_limits<E>::min());
  if constexpr (std::is_same_v<E, float32_t>) {
    samples.insert(samples.end(), {E(-87.5), E(-100.0), E(-104.0), E(-120.0)});
  } else if constexpr (std::is_same_v<E, float64_t>) {
    samples.insert(samples.end(), {E(-709.0), E(-740.0), E(-746.0), E(-800.0)});
  }
#ifndef VECOPS_MATH_ASSUME_VALID_INPUTS
  samples.push_back(-std::numeric_limits<E>::infinity());
#endif
  run_neg_values<Tier::Strict>(t, samples);
  run_neg_values<Tier::Fast>(t, samples);
  run_neg_values<Tier::Estimate>(t, samples);
}

template <typename E>
void run_negative_low_precision_tiers() {
  ScalableTag<E, 0> t;
  auto samples = all_low_precision_values<E>();
  samples.erase(std::remove_if(samples.begin(), samples.end(), [](E x) {
    return std::isnan(as_double(x)) || as_double(x) > 0.0;
  }), samples.end());
  run_neg_values<Tier::Strict>(t, samples);
  run_neg_values<Tier::Fast>(t, samples);
  run_neg_values<Tier::Estimate>(t, samples);
}

template <Tier tier, typename E>
void check_masked() {
  ScalableTag<E, 1> t;
  const nint_t n = size(t);
  std::vector<E> in(static_cast<size_t>(n)), defaults(static_cast<size_t>(n)), out(static_cast<size_t>(n));
  std::vector<bool> pattern(static_cast<size_t>(n));
  for (nint_t i = 0; i < n; ++i) {
    pattern[static_cast<size_t>(i)] = i % 2 == 0;
    in[static_cast<size_t>(i)] = pattern[static_cast<size_t>(i)] ? E((i % 9) * 0.25 - 1.0)
                                                                 : std::numeric_limits<E>::quiet_NaN();
    defaults[static_cast<size_t>(i)] = E(-3.0 - i);
  }
  auto m = test_utils::make_mask(t, pattern);
  auto vi = loadu(t, in.data());
  storeu(t, out.data(), call_exp<tier>(vi, m, loadu(t, defaults.data())));
  for (nint_t i = 0; i < n; ++i) {
    if (pattern[static_cast<size_t>(i)]) check_accuracy<tier>(in[i], out[i]);
    else EXPECT_EQ(bits(defaults[i]), bits(out[i])) << "inactive lane " << i;
  }
  Vec<ScalableTag<E, 1>> preserved;
  if constexpr (tier == Tier::Strict) preserved = vecops::vec::exp(vi, m);
  else if constexpr (tier == Tier::Fast) preserved = exp_fast(vi, m);
  else preserved = exp_est(vi, m);
  storeu(t, out.data(), preserved);
  for (nint_t i = 0; i < n; ++i) if (!pattern[i]) EXPECT_EQ(bits(in[i]), bits(out[i]));
}

template <Tier tier, typename E>
void check_neg_masked() {
  ScalableTag<E, 1> t;
  const nint_t n = size(t);
  std::vector<E> in(static_cast<size_t>(n)), defaults(static_cast<size_t>(n)), out(static_cast<size_t>(n));
  std::vector<bool> pattern(static_cast<size_t>(n));
  for (nint_t i = 0; i < n; ++i) {
    pattern[static_cast<size_t>(i)] = i % 2 == 0;
    in[static_cast<size_t>(i)] = pattern[static_cast<size_t>(i)]
        ? E(-0.125 * (i + 1))
        : (i % 4 == 1 ? std::numeric_limits<E>::quiet_NaN() : E(4.0));
    defaults[static_cast<size_t>(i)] = E(-7.0 - i);
  }
  auto m = test_utils::make_mask(t, pattern);
  auto vi = loadu(t, in.data());
  storeu(t, out.data(), call_exp_neg<tier>(vi, m, loadu(t, defaults.data())));
  for (nint_t i = 0; i < n; ++i) {
    if (pattern[static_cast<size_t>(i)]) check_accuracy<tier>(in[i], out[i]);
    else EXPECT_EQ(bits(defaults[i]), bits(out[i])) << "inactive lane " << i;
  }
}

}  // namespace

TEST(VecMathTest, Float32AccuracyAllVectorShapes) {
  run_regular_tiers<float32_t, 0>();
  run_regular_tiers<float32_t, 1>();
  run_regular_tiers<float32_t, 2>();
}

TEST(VecMathTest, Float64AccuracyAllVectorShapes) {
  run_regular_tiers<float64_t, 0>();
  run_regular_tiers<float64_t, 1>();
  run_regular_tiers<float64_t, 2>();
}

TEST(VecMathTest, Float16Exhaustive) { run_low_precision_tiers<vecops::float16_t>(); }

#if defined(HAS_BFLOAT16) || defined(ARCH_X86_FAMILY)
TEST(VecMathTest, BFloat16Exhaustive) { run_low_precision_tiers<vecops::bfloat16_t>(); }
#endif

TEST(VecMathTest, NegativeDomainFloat32AllVectorShapes) {
  run_negative_tiers<float32_t, 0>();
  run_negative_tiers<float32_t, 1>();
  run_negative_tiers<float32_t, 2>();
}

TEST(VecMathTest, NegativeDomainFloat64AllVectorShapes) {
  run_negative_tiers<float64_t, 0>();
  run_negative_tiers<float64_t, 1>();
  run_negative_tiers<float64_t, 2>();
}

TEST(VecMathTest, NegativeDomainFloat16Exhaustive) {
  run_negative_low_precision_tiers<vecops::float16_t>();
}

#if defined(HAS_BFLOAT16) || defined(ARCH_X86_FAMILY)
TEST(VecMathTest, NegativeDomainBFloat16Exhaustive) {
  run_negative_low_precision_tiers<vecops::bfloat16_t>();
}
#endif

TEST(VecMathTest, MaskedOverloadsPreserveInactiveLanes) {
  check_masked<Tier::Strict, float32_t>();
  check_masked<Tier::Fast, float32_t>();
  check_masked<Tier::Estimate, float32_t>();
  check_masked<Tier::Strict, float64_t>();
  check_masked<Tier::Fast, float64_t>();
  check_masked<Tier::Estimate, float64_t>();
  check_masked<Tier::Strict, vecops::float16_t>();
  check_masked<Tier::Fast, vecops::float16_t>();
  check_masked<Tier::Estimate, vecops::float16_t>();
#if defined(HAS_BFLOAT16) || defined(ARCH_X86_FAMILY)
  check_masked<Tier::Strict, vecops::bfloat16_t>();
  check_masked<Tier::Fast, vecops::bfloat16_t>();
  check_masked<Tier::Estimate, vecops::bfloat16_t>();
#endif
}

TEST(VecMathTest, NegativeDomainMaskedOverloadsIgnoreInvalidInactiveLanes) {
  check_neg_masked<Tier::Strict, float32_t>();
  check_neg_masked<Tier::Fast, float32_t>();
  check_neg_masked<Tier::Estimate, float32_t>();
  check_neg_masked<Tier::Strict, float64_t>();
  check_neg_masked<Tier::Fast, float64_t>();
  check_neg_masked<Tier::Estimate, float64_t>();
  check_neg_masked<Tier::Strict, vecops::float16_t>();
  check_neg_masked<Tier::Fast, vecops::float16_t>();
  check_neg_masked<Tier::Estimate, vecops::float16_t>();
#if defined(HAS_BFLOAT16) || defined(ARCH_X86_FAMILY)
  check_neg_masked<Tier::Strict, vecops::bfloat16_t>();
  check_neg_masked<Tier::Fast, vecops::bfloat16_t>();
  check_neg_masked<Tier::Estimate, vecops::bfloat16_t>();
#endif
}

TEST(VecMathTest, SubnormalAndThresholdBoundaries) {
  check_subnormal_and_thresholds<float32_t>();
  check_subnormal_and_thresholds<float64_t>();
}

TEST(VecMathTest, Float32UpperTailDense) {
  ScalableTag<float32_t, 0> t;
  constexpr float lo = 88.0f;
  constexpr float overflow = 88.72283905206835f;
  const float finite_hi = std::nextafter(overflow, -std::numeric_limits<float>::infinity());
  std::vector<float> values;
  values.reserve(8195);
  for (int i = 0; i <= 8192; ++i) {
    values.push_back(lo + (finite_hi - lo) * static_cast<float>(i) / 8192.0f);
  }
  values.push_back(overflow);
  values.push_back(std::nextafter(overflow, std::numeric_limits<float>::infinity()));
  run_values<Tier::Strict>(t, values);
  run_values<Tier::Fast>(t, values);
  run_values<Tier::Estimate>(t, values);
}

TEST(VecMathTest, MonotonicAcrossRangeReductionBoundaries) {
  check_monotonic<Tier::Strict, float32_t>();
  check_monotonic<Tier::Fast, float32_t>();
  check_monotonic<Tier::Estimate, float32_t>();
  check_monotonic<Tier::Strict, float64_t>();
  check_monotonic<Tier::Fast, float64_t>();
  check_monotonic<Tier::Estimate, float64_t>();
}

#ifndef VECOPS_MATH_ASSUME_VALID_INPUTS
template <Tier tier, typename E>
void check_edges() {
  ScalableTag<E, 0> t;
  std::vector<E> in(static_cast<size_t>(size(t)), E(0));
  in[0] = E(0);
  if (in.size() > 1) in[1] = E(-0.0);
  if (in.size() > 2) in[2] = std::numeric_limits<E>::infinity();
  if (in.size() > 3) in[3] = -std::numeric_limits<E>::infinity();
  if (in.size() > 4) in[4] = std::numeric_limits<E>::quiet_NaN();
  std::vector<E> out(in.size());
  storeu(t, out.data(), call_exp<tier>(loadu(t, in.data())));
  EXPECT_EQ(as_double(out[0]), 1.0);
  if (in.size() > 1) EXPECT_EQ(as_double(out[1]), 1.0);
  if (in.size() > 2) EXPECT_TRUE(std::isinf(as_double(out[2]))) << as_double(out[2]);
  if (in.size() > 3) EXPECT_EQ(as_double(out[3]), 0.0);
  if (in.size() > 4) EXPECT_TRUE(std::isnan(as_double(out[4])));
}

TEST(VecMathTest, IeeeEdgeCases) {
  check_edges<Tier::Strict, float32_t>();
  check_edges<Tier::Fast, float32_t>();
  check_edges<Tier::Estimate, float32_t>();
  check_edges<Tier::Strict, float64_t>();
  check_edges<Tier::Fast, float64_t>();
  check_edges<Tier::Estimate, float64_t>();
  check_edges<Tier::Strict, vecops::float16_t>();
  check_edges<Tier::Fast, vecops::float16_t>();
  check_edges<Tier::Estimate, vecops::float16_t>();
#if defined(HAS_BFLOAT16) || defined(ARCH_X86_FAMILY)
  check_edges<Tier::Strict, vecops::bfloat16_t>();
  check_edges<Tier::Fast, vecops::bfloat16_t>();
  check_edges<Tier::Estimate, vecops::bfloat16_t>();
#endif
}
#endif
