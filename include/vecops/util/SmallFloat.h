//
// Created by HenryRenYz on 2026/8/22.
//

#ifndef VECOPS_SMALL_FLOAT_H
#define VECOPS_SMALL_FLOAT_H

#include <cstdint>
#include <limits>
#include <ostream>

#include "vecops/CoreDefs.h"

namespace vecops {

/**
 * @brief CRTP base supplying the arithmetic / relational operator surface
 *        shared by the small-float wrappers (Float16, BFloat16).
 *
 * Every operation round-trips through float: the derived class only needs an
 * implicit conversion to float and a constructor from float. The base is
 * empty, so the wrappers keep their two-byte size and alignment.
 */
template <typename Derived>
struct SmallFloatOps {
  friend VECOPS_INLINE std::ostream& operator<<(
      std::ostream& out, const Derived& value) {
    out << float(value);
    return out;
  }

#define VECOPS_DEFINE_SMALL_FLOAT_BINARY(Op, Apply)                     \
  friend VECOPS_INLINE Derived operator Op(                             \
      const Derived& a, const Derived& b) {                             \
    return float(a) Apply float(b);                                     \
  }
  VECOPS_DEFINE_SMALL_FLOAT_BINARY(+, +)
  VECOPS_DEFINE_SMALL_FLOAT_BINARY(-, -)
  VECOPS_DEFINE_SMALL_FLOAT_BINARY(*, *)
  VECOPS_DEFINE_SMALL_FLOAT_BINARY(/, /)
#undef VECOPS_DEFINE_SMALL_FLOAT_BINARY

  friend VECOPS_INLINE Derived operator-(const Derived& a) {
    return -float(a);
  }

#define VECOPS_DEFINE_SMALL_FLOAT_COMPOUND(OpAssign, Op)                \
  friend VECOPS_INLINE Derived& operator OpAssign(                      \
      Derived& a, const Derived& b) {                                    \
    a = a Op b;                                                          \
    return a;                                                            \
  }
  VECOPS_DEFINE_SMALL_FLOAT_COMPOUND(+=, +)
  VECOPS_DEFINE_SMALL_FLOAT_COMPOUND(-=, -)
  VECOPS_DEFINE_SMALL_FLOAT_COMPOUND(*=, *)
  VECOPS_DEFINE_SMALL_FLOAT_COMPOUND(/=, /)
#undef VECOPS_DEFINE_SMALL_FLOAT_COMPOUND

#define VECOPS_DEFINE_SMALL_FLOAT_RELATIONAL(Op)                         \
  friend VECOPS_INLINE bool operator Op(                                 \
      const Derived& a, const Derived& b) {                              \
    return float(a) Op float(b);                                         \
  }
  VECOPS_DEFINE_SMALL_FLOAT_RELATIONAL(<)
  VECOPS_DEFINE_SMALL_FLOAT_RELATIONAL(>)
  VECOPS_DEFINE_SMALL_FLOAT_RELATIONAL(<=)
  VECOPS_DEFINE_SMALL_FLOAT_RELATIONAL(>=)
  VECOPS_DEFINE_SMALL_FLOAT_RELATIONAL(==)
  VECOPS_DEFINE_SMALL_FLOAT_RELATIONAL(!=)
#undef VECOPS_DEFINE_SMALL_FLOAT_RELATIONAL
};

} // namespace vecops

#define VECOPS_DEFINE_STD_ABS(Type)                                      \
  VECOPS_INLINE constexpr vecops::Type fabs(vecops::Type x) {            \
    return vecops::Type::from_bits(x.to_bits() & 0x7fff);                \
  }                                                                      \
  VECOPS_INLINE constexpr vecops::Type abs(vecops::Type x) {             \
    return std::fabs(x);                                                 \
  }

/** Defines std::numeric_limits for a small-float wrapper from its format
 *  parameters and the bit patterns of the special values. */
#define VECOPS_DEFINE_SMALL_FLOAT_LIMITS(                                \
    Type, IsIec559, Digits, Digits10, MaxDigits10, MinExp, MinExp10,     \
    MaxExp, MaxExp10, MinBits, LowestBits, MaxBits, EpsilonBits,         \
    RoundErrorBits, InfinityBits, QuietNaNBits, SignalingNaNBits,        \
    DenormMinBits)                                                       \
  template <>                                                            \
  class numeric_limits<vecops::Type> {                                   \
   public:                                                               \
    static constexpr bool is_specialized = true;                         \
    static constexpr bool is_signed = true;                              \
    static constexpr bool is_integer = false;                            \
    static constexpr bool is_exact = false;                              \
    static constexpr bool has_infinity = true;                           \
    static constexpr bool has_quiet_NaN = true;                          \
    static constexpr bool has_signaling_NaN = true;                      \
    static constexpr auto has_denorm =                                   \
        numeric_limits<float>::has_denorm;                               \
    static constexpr auto has_denorm_loss =                              \
        numeric_limits<float>::has_denorm_loss;                          \
    static constexpr auto round_style =                                  \
        numeric_limits<float>::round_style;                              \
    static constexpr bool is_iec559 = IsIec559;                          \
    static constexpr bool is_bounded = true;                             \
    static constexpr bool is_modulo = false;                             \
    static constexpr int digits = Digits;                                \
    static constexpr int digits10 = Digits10;                            \
    static constexpr int max_digits10 = MaxDigits10;                     \
    static constexpr int radix = 2;                                      \
    static constexpr int min_exponent = MinExp;                          \
    static constexpr int min_exponent10 = MinExp10;                      \
    static constexpr int max_exponent = MaxExp;                          \
    static constexpr int max_exponent10 = MaxExp10;                      \
    static constexpr auto traps = numeric_limits<float>::traps;          \
    static constexpr auto tinyness_before =                              \
        numeric_limits<float>::tinyness_before;                          \
    static constexpr vecops::Type min() {                                \
      return vecops::Type::from_bits(MinBits);                           \
    }                                                                    \
    static constexpr vecops::Type lowest() {                             \
      return vecops::Type::from_bits(LowestBits);                        \
    }                                                                    \
    static constexpr vecops::Type max() {                                \
      return vecops::Type::from_bits(MaxBits);                           \
    }                                                                    \
    static constexpr vecops::Type epsilon() {                            \
      return vecops::Type::from_bits(EpsilonBits);                       \
    }                                                                    \
    static constexpr vecops::Type round_error() {                        \
      return vecops::Type::from_bits(RoundErrorBits);                    \
    }                                                                    \
    static constexpr vecops::Type infinity() {                           \
      return vecops::Type::from_bits(InfinityBits);                      \
    }                                                                    \
    static constexpr vecops::Type quiet_NaN() {                          \
      return vecops::Type::from_bits(QuietNaNBits);                      \
    }                                                                    \
    static constexpr vecops::Type signaling_NaN() {                      \
      return vecops::Type::from_bits(SignalingNaNBits);                  \
    }                                                                    \
    static constexpr vecops::Type denorm_min() {                         \
      return vecops::Type::from_bits(DenormMinBits);                     \
    }                                                                    \
  };

#endif // VECOPS_SMALL_FLOAT_H
