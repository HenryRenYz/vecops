// Standalone full-range accuracy probe for the vec math module: the exp
// family (exp, exp2, exp10, and their *_neg domains) and the reciprocal
// family (rcp, rsqrt). Mirrors the tolerance contracts of the regular test
// suites (tests/vec/MathTest.cpp, tests/vec/ReciprocalTest.cpp) but sweeps
// the whole input domain — directed boundary sweeps, random bit patterns,
// boundary-bit neighborhoods, masked mixed-tail words, and (with
// --exhaustive) the complete f32 bit pattern space. Not part of the test
// suite and not registered in CMake; compile manually, e.g.
//
//   x86: g++ -std=c++20 -O2 -march=native -Iinclude -Iinclude/vecops \
//            tests/vec/MathAccuracyProbe.cpp -o /tmp/mathprobe
//   SVE: clang++ -std=c++20 -O2 -march=armv8-a+sve -Iinclude \
//            -Iinclude/vecops tests/vec/MathAccuracyProbe.cpp -o ~/tmp/mathprobe
//
// Usage: mathprobe [--family=exp|exp2|exp10|rcp|rsqrt|all]
//                  [--mode=quick|exhaustive]   (default quick)
//
// quick      : sweeps + random + boundary neighborhoods + masked mixed-tail
// exhaustive : quick + the full 2^32 f32 bit-pattern sweep for Strict

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <type_traits>
#include <vector>

#include "vecops/vec/Vec.h"

namespace vec = vecops::vec;
using namespace vecops;

constexpr long double kLog2Base10Long = 3.321928094887362347870319429489L;

/* **************************************************************************** */
//    Shared bit/format helpers                                                 //
/* **************************************************************************** */

template <typename T>
double as_double(T v) {
  return static_cast<double>(v);
}

template <typename T>
auto bits(T v) {
  if constexpr (sizeof(T) == 4) {
    uint32_t r;
    memcpy(&r, &v, 4);
    return (uint64_t)r;
  } else if constexpr (sizeof(T) == 8) {
    uint64_t r;
    memcpy(&r, &v, 8);
    return r;
  } else {
    return (uint64_t)v.to_bits();
  }
}

template <typename T>
T from_bits(uint64_t b) {
  if constexpr (sizeof(T) == 4) {
    float32_t v;
    uint32_t u = (uint32_t)b;
    memcpy(&v, &u, 4);
    return v;
  } else if constexpr (sizeof(T) == 8) {
    float64_t v;
    memcpy(&v, &b, 8);
    return v;
  } else {
    return T::from_bits((uint16_t)b);
  }
}

template <typename T>
const char* format_name() {
  if constexpr (std::is_same_v<T, float32_t>) return "f32";
  else if constexpr (std::is_same_v<T, float64_t>) return "f64";
  else if constexpr (std::is_same_v<T, vecops::float16_t>) return "f16";
  else return "bf16";
}

struct Lcg {
  uint64_t state;
  uint64_t next() {
    state = state * 6364136223846793005ull + 1442695040888963407ull;
    return state;
  }
};

struct Stats {
  long long checked = 0;
  long long violations = 0;
  double max_ulp = 0;
  double max_rel = 0;
  double worst_x = 0;
  double worst_rel_x = 0;
  bool nan_bad = false;
  double nan_bad_x = 0;
};

/* **************************************************************************** */
//    Probe families                                                            //
/* **************************************************************************** */

// A family names itself, evaluates the op under a tier, computes a
// long-double reference, and owns its tier contracts.
template <vec::ExpBase Base, bool NegativeOnly>
struct ExpFamily {
  static constexpr vec::ExpBase base = Base;
  static constexpr bool is_recip = false;
  static const char* name() {
    if constexpr (NegativeOnly) {
      if constexpr (Base == vec::ExpBase::E) return "exp_neg";
      else if constexpr (Base == vec::ExpBase::Base2) return "exp2_neg";
      else return "exp10_neg";
    } else {
      if constexpr (Base == vec::ExpBase::E) return "exp";
      else if constexpr (Base == vec::ExpBase::Base2) return "exp2";
      else return "exp10";
    }
  }
  // The *_neg contract only defines x <= 0 active lanes.
  static constexpr bool negative_only = NegativeOnly;

  template <vec::FloatingTag Tag, typename... Options>
  static auto invoke(Tag tag, vec::Vec<Tag> v, Options&&... options) {
    return vec::ExpCpo<Base, NegativeOnly>{}(
        tag, v, std::forward<Options>(options)...);
  }

  template <typename T>
  static long double reference_widened(T v) {
    if constexpr (Base == vec::ExpBase::E)
      return std::exp(static_cast<long double>(as_double(v)));
    else if constexpr (Base == vec::ExpBase::Base2)
      return std::exp2(static_cast<long double>(as_double(v)));
    else
      return std::exp2(
          static_cast<long double>(as_double(v)) * kLog2Base10Long);
  }

  template <typename T>
  static T reference(T v) {
    const long double r = reference_widened(v);
    if constexpr (sizeof(T) < 4) return T((float)r);
    else return (T)r;
  }

  // Contract table (matches tests/vec/MathTest.cpp): Strict <= 1 ULP,
  // Fast <= 4 ULP, Estimate <= 4 ULP or 0.006 relative.
  static constexpr bool fast_ulp_contract = true;
  static constexpr unsigned fast_ulp_bound = 4;
  static constexpr bool estimate_ulp_contract = true;
  static constexpr unsigned estimate_ulp_bound = 4;
  static constexpr long double estimate_rel_bound = 0.006L;
};

template <bool IsRsqrt>
struct RecipFamily {
  static const char* name() {
    return IsRsqrt ? "rsqrt" : "rcp";
  }
  static constexpr bool negative_only = false;
  static constexpr bool is_recip = true;

  template <vec::FloatingTag Tag, typename... Options>
  static auto invoke(Tag tag, vec::Vec<Tag> v, Options&&... options) {
    if constexpr (IsRsqrt)
      return vec::rsqrt(tag, v, std::forward<Options>(options)...);
    else
      return vec::rcp(tag, v, std::forward<Options>(options)...);
  }

  template <typename T>
  static long double reference_widened(T v) {
    if constexpr (IsRsqrt)
      return 1.0L / std::sqrt(static_cast<long double>(as_double(v)));
    else
      return 1.0L / static_cast<long double>(as_double(v));
  }

  template <typename T>
  static T reference(T v) {
    if constexpr (IsRsqrt) {
      if (v == T(0)) return std::numeric_limits<T>::infinity();
    }
    const long double r = reference_widened(v);
    if constexpr (sizeof(T) < 4) return T((float)r);
    else return (T)r;
  }

  // Contract table (matches tests/vec/ReciprocalTest.cpp and the Math.h
  // header): Strict <= 1 ULP, Fast <= 2^-15 relative (narrow: 2 ULP),
  // Estimate <= 2^-7 relative.
  static constexpr bool fast_ulp_contract = false;
  static constexpr unsigned fast_ulp_bound = 0;
  static constexpr long double fast_rel_bound_wide = 3.0517578125e-5L;  // 2^-15
  static constexpr bool estimate_ulp_contract = false;
  static constexpr unsigned estimate_ulp_bound = 0;
  static constexpr long double estimate_rel_bound = 7.8125e-3L;  // 2^-7
};

/* **************************************************************************** */
//    Tier checking                                                             //
/* **************************************************************************** */

/**
 * Checks one (input, actual) pair under the family's tier contract. Shared
 * rules: NaN expectations require NaN results; non-preserving builds accept
 * flushed subnormal outputs. Family specifics: exp allows documented
 * flush-to-zero; the recip family additionally pins the exact +-inf and
 * signed-zero results and accepts the two narrow-format boundary behaviors
 * (finite near-max instead of inf for estimate tiers, and a flush at the
 * smallest normal).
 */
template <typename Family, vec::Accuracy Tier, typename T>
bool check_one(T input, T actual, Stats& s) {
  const T expected = Family::template reference<T>(input);
  const double in = as_double(input);
  ++s.checked;
  if (std::isnan(as_double(expected))) {
    if (!std::isnan(as_double(actual))) {
      ++s.violations;
      if (!s.nan_bad) {
        s.nan_bad = true;
        s.nan_bad_x = in;
      }
      return false;
    }
    return true;
  }

  const double ed = as_double(expected);
  const double ad = as_double(actual);

  // Reciprocal exact-class results: overflow to +-inf (and back), and
  // rcp(+-inf) = signed zero, compare bit-exactly (with an estimate-tier
  // overflow allowance for finite values near the format maximum).
  if constexpr (Family::is_recip) {
    if (std::isinf(ed)) {
      if constexpr (Tier == vec::Accuracy::Strict) {
        if (bits(expected) != bits(actual)) {
          ++s.violations;
          if (s.violations < 8)
            printf("  INF MISMATCH x=%g got=%.9g want=%.9g\n", in, ad, ed);
          return false;
        }
        return true;
      } else {
        const bool near = std::isinf(ad) ||
            std::abs(ad) >= 0.5 * (double)std::numeric_limits<T>::max();
        if (!near) {
          ++s.violations;
          if (s.violations < 8)
            printf("  INF-EST x=%g got=%.9g want=%.9g\n", in, ad, ed);
          return false;
        }
        return true;
      }
    }
    if (expected == T(0)) {  // rcp(+-inf) / rsqrt(+inf): exact signed zero
      if (bits(expected) != bits(actual)) {
        ++s.violations;
        if (s.violations < 8)
          printf("  ZERO MISMATCH x=%g got=%.9g want=%.9g\n", in, ad, ed);
        return false;
      }
      return true;
    }
  }

  const double min_normal = (double)std::numeric_limits<T>::min();
#ifndef VECOPS_PRESERVE_SUBNORMALS
  if (ed != 0.0 && std::fabs(ed) < min_normal && ad == 0.0)
    return true;  // documented flush-to-zero allowance (all tiers, both signs)
  // Reciprocal boundary: a result one narrow-format ULP below the smallest
  // normal flushes in non-preserving builds.
  if constexpr (Family::is_recip) {
    if (std::abs(ed) == min_normal && ad == 0.0) return true;
  }
#endif

  const uint64_t eb = bits(expected);
  const uint64_t ab = bits(actual);
  const uint64_t ulps = eb > ab ? eb - ab : ab - eb;
  if (ulps > s.max_ulp) {
    s.max_ulp = (double)ulps;
    s.worst_x = in;
  }
  const bool finite_pair = std::isfinite(ed) && std::isfinite(ad) &&
                           std::abs(ed) >= min_normal;

  const auto rel_error = [&]() -> long double {
    const long double exact = Family::template reference_widened<T>(input);
    return std::fabs(static_cast<long double>(ad) / exact - 1.0L);
  };

  if constexpr (Tier == vec::Accuracy::Strict) {
    if (ulps > 1ull && !(ad == 0.0 && ed == 0.0)) {
      ++s.violations;
      if (s.violations < 8)
        printf("  STRICT VIOLATION x=%g expected=%.9g actual=%.9g "
               "ulp=%llu\n",
               in, ed, ad, (unsigned long long)ulps);
      return false;
    }
  } else if constexpr (Tier == vec::Accuracy::Fast) {
    if (finite_pair) {
      if constexpr (Family::fast_ulp_contract) {
        if (ulps > (uint64_t)Family::fast_ulp_bound) {
          ++s.violations;
          if (s.violations < 8)
            printf("  FAST VIOLATION x=%g expected=%.9g actual=%.9g "
                   "ulp=%llu\n",
                   in, ed, ad, (unsigned long long)ulps);
          return false;
        }
      } else {
        // Relative bound; narrow formats relax to 2 ULP of their mantissa.
        const long double bound = [&]() -> long double {
          if constexpr (sizeof(T) >= 4)
            return Family::fast_rel_bound_wide;
          else if constexpr (std::is_same_v<T, vecops::float16_t>)
            return 9.765625e-4L;  // 2 ULP of f16
          else
            return 7.8125e-3L;    // 2 ULP of bf16
        }();
        if (rel_error() > bound) {
          ++s.violations;
          if (s.violations < 8)
            printf("  FAST VIOLATION x=%g expected=%.9g actual=%.9g "
                   "rel=%.3g\n",
                   in, ed, ad, (double)rel_error());
          return false;
        }
      }
    }
  } else {  // Estimate
    if (finite_pair) {
      const long double rel = rel_error();
      if ((double)rel > s.max_rel) {
        s.max_rel = (double)rel;
        s.worst_rel_x = in;
      }
      const bool ulp_ok = !Family::estimate_ulp_contract ||
          ulps <= (uint64_t)Family::estimate_ulp_bound;
      if (!ulp_ok && rel > Family::estimate_rel_bound) {
        ++s.violations;
        if (s.violations < 8)
          printf("  EST VIOLATION x=%g expected=%.9g actual=%.9g "
                 "ulp=%llu rel=%.3g\n",
                 in, ed, ad, (unsigned long long)ulps, (double)rel);
        return false;
      }
    }
  }
  return true;
}

template <typename Family, vec::Accuracy Tier, typename T>
void update_rel_stat(T input, T actual, Stats& s) {
  if constexpr (Tier != vec::Accuracy::Strict) {
    const T expected = Family::template reference<T>(input);
    if (std::isnan(as_double(expected))) return;
    const double ed = as_double(expected);
    const double ad = as_double(actual);
    const double min_normal = (double)std::numeric_limits<T>::min();
    if (!std::isfinite(ed) || !std::isfinite(ad) || std::abs(ed) < min_normal)
      return;
    const long double exact = Family::template reference_widened<T>(input);
    const long double rel =
        std::fabs(static_cast<long double>(ad) / exact - 1.0L);
    if ((double)rel > s.max_rel) {
      s.max_rel = (double)rel;
      s.worst_rel_x = as_double(input);
    }
  }
}

/* **************************************************************************** */
//    Vector drivers                                                            //
/* **************************************************************************** */

template <typename Family, vec::Accuracy Tier, vec::FloatingTag Tag>
void run_vector(Tag tag, const std::vector<vec::ElementOf<Tag>>& xs,
                Stats& s) {
  using T = vec::ElementOf<Tag>;
  const nint_t lanes = vec::size(tag);
  for (size_t base = 0; base < xs.size(); base += (size_t)lanes) {
    auto input = vec::zeros(tag);
    nint_t filled = 0;
    for (nint_t lane = 0; lane < lanes; ++lane) {
      const size_t idx = base + (size_t)lane;
      if (idx >= xs.size()) break;
      input = vec::set(tag, input, lane, xs[idx]);
      ++filled;
    }
    auto result = Family::invoke(tag, input, vec::opt::math::accuracy<Tier>);
    for (nint_t lane = 0; lane < filled; ++lane) {
      const T x = vec::get(tag, input, lane);
      const T y = vec::get(tag, result, lane);
      update_rel_stat<Family, Tier, T>(x, y, s);
      check_one<Family, Tier, T>(x, y, s);
    }
  }
}

template <typename Family, vec::Accuracy Tier, typename T>
void report(const char* mode, const Stats& s) {
  const char* tier = Tier == vec::Accuracy::Strict
      ? "strict"
      : (Tier == vec::Accuracy::Fast ? "fast" : "est");
  printf("%-9s/%-4s/%-7s/%-10s checked=%-10lld viol=%-4lld "
         "max_ulp=%-8.1f(@%.9g) max_rel=%-10.3g(@%.9g)%s%s\n",
         Family::name(), format_name<T>(), mode, tier, s.checked,
         s.violations, s.max_ulp, s.worst_x, s.max_rel, s.worst_rel_x,
         s.nan_bad ? " NAN-PROPAGATION-BROKEN" : "",
         s.violations ? "  <-- FAIL" : "");
}

template <typename Family, typename T>
void run_all_tiers(const char* mode, const std::vector<T>& inputs) {
  vec::ScalableTag<T, 0> tag;
  std::vector<T> selected;
  selected.reserve(inputs.size());
  for (T v : inputs) {
    const bool positive_or_nan =
        as_double(v) > 0 || std::isnan(as_double(v));
    if constexpr (Family::negative_only) {
      if (positive_or_nan) continue;  // *_neg contract: only x <= 0 defined
    }
    selected.push_back(v);
  }
  const auto drive = [&]<vec::Accuracy Tier>() {
    Stats s;
    run_vector<Family, Tier>(tag, selected, s);
    report<Family, Tier, T>(mode, s);
    return s.violations == 0;
  };
  const bool ok = drive.template operator()<vec::Accuracy::Strict>() &&
      drive.template operator()<vec::Accuracy::Fast>() &&
      drive.template operator()<vec::Accuracy::Estimate>();
  (void)ok;
}

/* **************************************************************************** */
//    Input generation                                                          //
/* **************************************************************************** */

// Boundary neighborhoods: every tail threshold of every family gets +-4
// bit-neighbors so the special-path edges are exercised on purpose. Values
// are kept in double; consumers widen to the target format.
template <typename Family, typename T>
std::vector<double> boundary_probes() {
  std::vector<double> probes;
  if constexpr (Family::is_recip) {
    probes = {0.0, -0.0, 1.0, -1.0, 4.0, 9.0, 16.0, 1e-30, -1e-30, 1e30,
              -1e30, 1e38, -1e38, 1e-300, 1e300,
              std::numeric_limits<double>::infinity(),
              -std::numeric_limits<double>::infinity()};
    if constexpr (std::is_same_v<T, float32_t>) {
      probes.push_back(0x1p-126);
      probes.push_back(0x1p126);
      probes.push_back(-0x1p126);
      probes.push_back(0x1p-125);
    } else if constexpr (std::is_same_v<T, float64_t>) {
      probes.push_back(0x1p-1022);
      probes.push_back(0x1p1022);
      probes.push_back(-0x1p1022);
      probes.push_back(0x1p-1021);
    }
  } else {
    constexpr vec::ExpBase Base = Family::base;
    if constexpr (std::is_same_v<T, float32_t>) {
      if constexpr (Base == vec::ExpBase::E)
        probes = {87.3365447505531, 88.72283905206835,
                  -87.3365447505531, -103.972084045410, 126.0, 128.0,
                  -150.0};
      else if constexpr (Base == vec::ExpBase::Base2)
        probes = {126.0, 128.0, -126.0, -150.0, -149.0, 127.9};
      else
        probes = {37.9250618474, 38.56, -45.15, -46.0, 38.53, 39.0};
    } else if constexpr (std::is_same_v<T, float64_t>) {
      if constexpr (Base == vec::ExpBase::E)
        probes = {708.3964185322641, 709.782712893384,
                  -708.3964185322641, -745.1332191019411, 1024.0, -1076.0};
      else if constexpr (Base == vec::ExpBase::Base2)
        probes = {1022.001953125, 1024.0, -1076.0, -1075.0, 1023.9};
      else
        probes = {307.6526, 309.0, -323.6, -324.0, 308.9};
    }
  }
  return probes;
}

template <typename Family, typename T>
std::vector<T> make_sweep_inputs() {
  std::vector<T> xs;
  if constexpr (std::is_same_v<T, float32_t>) {
    const double lo = [] {
      if constexpr (Family::is_recip)
        return -40.0;
      else if constexpr (Family::base == vec::ExpBase::E) return -110.0;
      else if constexpr (Family::base == vec::ExpBase::Base2) return -152.0;
      else return -46.0;
    }();
    const double hi = [] {
      if constexpr (Family::is_recip)
        return 40.0;
      else if constexpr (Family::base == vec::ExpBase::E) return 95.0;
      else if constexpr (Family::base == vec::ExpBase::Base2) return 129.0;
      else return 39.0;
    }();
    for (int i = 0; i < 300000; ++i)
      xs.push_back((T)(float32_t(lo + (hi - lo) * i / 299999.0)));
  } else if constexpr (std::is_same_v<T, float64_t>) {
    const double lo = [] {
      if constexpr (Family::is_recip)
        return -300.0;
      else if constexpr (Family::base == vec::ExpBase::E) return -760.0;
      else if constexpr (Family::base == vec::ExpBase::Base2) return -1078.0;
      else return -325.0;
    }();
    const double hi = [] {
      if constexpr (Family::is_recip)
        return 300.0;
      else if constexpr (Family::base == vec::ExpBase::E) return 715.0;
      else if constexpr (Family::base == vec::ExpBase::Base2) return 1025.0;
      else return 310.0;
    }();
    for (int i = 0; i < 300000; ++i)
      xs.push_back((T)(float64_t(lo + (hi - lo) * i / 299999.0)));
  } else {
    // Narrow formats: exhaustive 65536 patterns.
    for (uint32_t b = 0; b < 65536; ++b) xs.push_back(from_bits<T>(b));
  }
  // Random bit patterns on top of the directed sweep (f32/f64).
  if constexpr (std::is_same_v<T, float32_t> ||
                std::is_same_v<T, float64_t>) {
    Lcg rng{0x12345678deadbeefull};
    constexpr int kRandom = 2000000;
    for (int i = 0; i < kRandom; ++i)
      xs.push_back(from_bits<T>(rng.next()));
    for (double p : boundary_probes<Family, T>()) {
      const uint64_t b = bits((T)p);
      for (int d = -4; d <= 4; ++d) xs.push_back(from_bits<T>(b + d));
    }
  }
  return xs;
}

// Complete f32 bit-pattern sweep (Strict tier only; the provable bounds of
// the estimate tiers need no exhaustive pass).
template <typename Family>
void exhaustive_f32() {
  vec::ScalableTag<float32_t, 0> tag;
  const nint_t lanes = vec::size(tag);
  Stats s;
  std::vector<float32_t> inputs((size_t)lanes);
  for (uint64_t pattern = 0; pattern <= 0xffffffffull;
       pattern += (uint64_t)lanes) {
    for (nint_t i = 0; i < lanes; ++i)
      inputs[(size_t)i] =
          from_bits<float32_t>(pattern + (uint64_t)i);
    auto v = vec::load(tag, inputs.data());
    auto r = Family::invoke(
        tag, v, vec::opt::math::accuracy<vec::Accuracy::Strict>);
    for (nint_t i = 0; i < lanes; ++i)
      check_one<Family, vec::Accuracy::Strict, float32_t>(
          inputs[(size_t)i], vec::get(tag, r, i), s);
  }
  report<Family, vec::Accuracy::Strict, float32_t>("exhaust", s);
}

/* **************************************************************************** */
//    Masked mixed-tail probe                                                   */
/* **************************************************************************** */

// Active lanes cycle through boundary/tail/NaN values while inactive lanes
// hold garbage; verifies active lanes against the tier contract and that
// inactive lanes pass through untouched under plain masked semantics.
template <typename Family, typename T>
void probe_masked_mixed() {
  std::vector<double> probes;
  if constexpr (Family::is_recip) {
    probes = {0.0, -0.0, 1.0, 4.0, 9.0, 1e-38, 1e38, -1e38, 1e-40,
              1e-300, 1e300, 2.5e-39, 3e38,
              std::numeric_limits<double>::infinity(),
              -std::numeric_limits<double>::infinity()};
  } else {
    probes = boundary_probes<Family, T>();
    const double extra[] = {0.0, 1.0, -1.0};
    for (double e : extra) probes.push_back(e);
    if (probes.empty()) probes = {0.0, 1.0, -1.0};
  }
  // The *_neg contract only defines x <= 0 active lanes; drop positive
  // probes so "unspecified" results are not flagged as violations.
  if constexpr (Family::negative_only) {
    probes.erase(
        std::remove_if(
            probes.begin(), probes.end(),
            [](double p) { return p > 0.0; }),
        probes.end());
  }
  vec::ScalableTag<T, 0> tag;
  const nint_t lanes = vec::size(tag);
  const auto run = [&]<vec::Accuracy Tier>() {
    Stats s;
    for (size_t rep = 0; rep < 4096; ++rep) {
      auto input = vec::zeros(tag);
      auto mask = vec::mfalse(tag);
      for (nint_t lane = 0; lane < lanes; ++lane) {
        const bool active = (lane + (nint_t)rep) % 3 != 0;
        const double base_value =
            probes[(rep * 7 + (size_t)lane * 13) % probes.size()];
        const double value = active ? base_value : (lane % 2 ? 1e30 : NAN);
        input = vec::set(tag, input, lane, (T)value);
        mask = vec::set(tag, mask, lane, active);
      }
      auto passthrough =
          Family::invoke(tag, input, vec::opt::masked(mask));
      auto result_tier = Family::invoke(
          tag, input, vec::opt::masked(mask),
          vec::opt::math::accuracy<Tier>);
      for (nint_t lane = 0; lane < lanes; ++lane) {
        const bool active = (lane + (nint_t)rep) % 3 != 0;
        const T x = vec::get(tag, input, lane);
        if (active) {
          check_one<Family, Tier, T>(x, vec::get(tag, result_tier, lane), s);
        } else {
          if (bits(vec::get(tag, passthrough, lane)) != bits(x)) {
            ++s.violations;
            if (s.violations < 8)
              printf("  MASKED-INACTIVE-BROKEN rep=%zu lane=%d\n", rep,
                     (int)lane);
          }
        }
      }
    }
    report<Family, Tier, T>("masked", s);
  };
  run.template operator()<vec::Accuracy::Strict>();
  run.template operator()<vec::Accuracy::Fast>();
  run.template operator()<vec::Accuracy::Estimate>();
}

/* **************************************************************************** */
//    Family driver and CLI                                                     */
/* **************************************************************************** */

template <typename Family>
void probe_family(bool exhaustive) {
  run_all_tiers<Family, float32_t>("quick", make_sweep_inputs<Family, float32_t>());
  run_all_tiers<Family, float64_t>("quick", make_sweep_inputs<Family, float64_t>());
  run_all_tiers<Family, vecops::float16_t>(
      "quick", make_sweep_inputs<Family, vecops::float16_t>());
#if defined(HAS_BFLOAT16) || !defined(ARCH_X86_FAMILY)
  run_all_tiers<Family, vecops::bfloat16_t>(
      "quick", make_sweep_inputs<Family, vecops::bfloat16_t>());
#endif
  probe_masked_mixed<Family, float32_t>();
  probe_masked_mixed<Family, float64_t>();
  if (exhaustive) {
    fflush(stdout);
    exhaustive_f32<Family>();
  }
}

int main(int argc, char** argv) {
  std::string family = "all";
  bool exhaustive = false;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg.rfind("--family=", 0) == 0) family = arg.substr(9);
    else if (arg == "--mode=exhaustive" || arg == "--exhaustive")
      exhaustive = true;
    else if (arg == "--mode=quick") exhaustive = false;
    else {
      fprintf(stderr,
              "usage: %s [--family=exp|exp2|exp10|exp_neg|rcp|rsqrt|all] "
              "[--mode=quick|exhaustive]\n",
              argv[0]);
      return 2;
    }
  }
  printf("preserve_subnormals=%d mode=%s\n",
#ifdef VECOPS_PRESERVE_SUBNORMALS
         1,
#else
         0,
#endif
         exhaustive ? "exhaustive" : "quick");

  const bool want_all = family == "all";
  if (want_all || family == "exp")
    probe_family<ExpFamily<vec::ExpBase::E, false>>(exhaustive);
  if (want_all || family == "exp2")
    probe_family<ExpFamily<vec::ExpBase::Base2, false>>(exhaustive);
  if (want_all || family == "exp10")
    probe_family<ExpFamily<vec::ExpBase::Base10, false>>(exhaustive);
  if (want_all || family == "exp_neg")
    probe_family<ExpFamily<vec::ExpBase::E, true>>(exhaustive);
  if (want_all || family == "rcp")
    probe_family<RecipFamily<false>>(exhaustive);
  if (want_all || family == "rsqrt")
    probe_family<RecipFamily<true>>(exhaustive);
  printf("PROBE DONE\n");
  return 0;
}
