//
// Created by renyz on 2026/6/2.
//

#ifndef VECOPS_LAYOUT_H
#define VECOPS_LAYOUT_H

#include <algorithm>
#include <array>
#include <ostream>
#include <tuple>
#include <utility>

#include "vecops/CoreTypes.h"
#include "vecops/Assertion.h"
#include "vecops/util/TypeTraits.h"

namespace vecops::gemm {

// ======================== Sentinel values for unbounded Dynamic ========================

constexpr nint_t kLoInf = std::numeric_limits<nint_t>::min();
constexpr nint_t kHiInf = std::numeric_limits<nint_t>::max();

// ======================== constexpr helpers ========================

namespace details {

constexpr nint_t lsb(nint_t x) {
  using U = std::make_unsigned_t<nint_t>;
  return static_cast<nint_t>(static_cast<U>(x) & -static_cast<U>(x));
}

template <nint_t A, nint_t N>
struct AlignAddSub {
  static constexpr nint_t value = (N == 0) ? A : ((A < lsb(N)) ? A : lsb(N));
};

template <nint_t A1, nint_t A2>
struct AlignAddSubDyn {
  static constexpr nint_t value = (A1 < A2) ? A1 : A2;
};

template <nint_t A, nint_t N>
struct AlignMul {
  static constexpr bool is_zero = (N == 0);
  static constexpr nint_t value = A * lsb(N);
};

template <nint_t A1, nint_t A2>
struct AlignMulDyn {
  static constexpr nint_t value = A1 * A2;
};

} // namespace details

// ======================== Value hierarchy ========================

/**
 * 用于表示一个张量元数据中的值，比如Shape或者Stride中的一个值。
 * 使用类型而不是运行时nint_t是用于存储编译期常量、对齐信息和约束，用于编译期优化。
 * 对于张量元数据而言，所有数值均为常量，一旦张量初始化其元数据则不再变化。
 */
struct Value {
  /**
   * 表示此值是否为编译期常量。
   */
  static constexpr bool is_const = false;
  /**
   * 表示此值是否为运行时常量。
   */
  static constexpr bool is_runtime = false;

  /**
   * 对于传入值v，其是否符合当前Value定义的约束（比如是否对齐等等）
   */
  static constexpr bool conforms(nint_t v) { return false; }

  /**
   * 检查此Value定义是否一定符合对齐到v的要求
   * 此函数返回false不代表Value携带的值一定不符合对齐要求（比如实际的运行时量可能是对齐的）
   */
  static constexpr bool aligns(nint_t v) { return false; }

  /**
   * 检查此Value携带的值是否符合对齐到v的要求
   * 可能会出现aligns(v) == false但this->is_aligned(v) == true的情况：Any{4}.is_aligned(4)
   */
  bool is_aligned(nint_t v) { return false; }

  constexpr explicit operator nint_t() const { return -100; }
}; // struct Value

/**
 * 表示编译器常量值，其值必定为N
 */
template <nint_t N>
struct Const : public Value {
  static constexpr bool is_const = true;
  static constexpr bool is_runtime = false;
  static constexpr nint_t value = N;

  static constexpr bool conforms(nint_t v) { return v == N; }

  static constexpr bool aligns(nint_t v) { return N % v == 0; }

  constexpr explicit Const(nint_t v = N) {
    VECOPS_ASSERT(v == N, "%td != %td", v, N);
  }

  constexpr bool is_aligned(nint_t v) { return N % v == 0; }

  constexpr explicit operator nint_t() const { return N; }
}; // struct Const

/**
 * constexpr field template用于方便初始化
 */
template <nint_t N>
inline constexpr Const<N> cint{N};

/**
 * 携带对齐和上下界约束的运行时量.
 * @tparam Alignment 对齐量，必须是2的整数幂且不能是0，运行时值v满足 v % Alignment == 0
 * @tparam Lo 编译期保证的下界（包含），v >= Lo；kLoInf 表示无下界约束
 * @tparam Hi 编译期保证的上界（包含），v <= Hi；kHiInf 表示无上界约束
 */
template <nint_t Alignment, nint_t Lo = kLoInf, nint_t Hi = kHiInf>
struct Dynamic : public Value {
  static_assert((Alignment & (Alignment - 1)) == 0 && Alignment > 0,
                "Alignment must be positive and power of 2"
  );
  static constexpr bool is_const = false;
  static constexpr bool is_runtime = true;
  static constexpr nint_t alignment = Alignment;
  static constexpr nint_t lo = Lo;
  static constexpr nint_t hi = Hi;
  static constexpr bool has_lower = (Lo != kLoInf);
  static constexpr bool has_upper = (Hi != kHiInf);

  /**
   * 检查编译期值v是否满足此Dynamic定义的约束：
   *   对齐约束：v & (Alignment-1) == 0
   *   上下界约束（若生效）：Lo <= v <= Hi
   */
  static constexpr bool conforms(nint_t v) {
    return (v & (Alignment - 1)) == 0
           && (!has_lower || v >= Lo)
           && (!has_upper || v <= Hi);
  }

  /**
   * 检查任何可能的此Dynamic值是否一定对齐到v。
   * 如果任何符合约束的运行时值都能被v整除则返回true。
   * 保守返回false是安全的。
   */
  static constexpr bool aligns(nint_t v) { return Alignment % v == 0; }

  /**
   * 构造Dynamic，运行时值v必须在编译期符合约束条件。
   */
  constexpr explicit Dynamic(nint_t v) : value(v) {
    VECOPS_ASSERT(conforms(v),
                  "value %td fails Dynamic<A=%td, Lo=%td, Hi=%td> constraints", v, Alignment, Lo, Hi);
  }

  /**
   * 检查this的运行时值是否对齐到v。
   */
  constexpr bool is_aligned(nint_t v) const { return value % v == 0; }

  constexpr explicit operator nint_t() const { return value; }

  const nint_t value;
}; // struct Dynamic

/**
 * 表示任意运行时量（无约束）。等价于 Dynamic<1>（对齐1，无上下界）
 */
using Any = Dynamic<1>;

// ======================== dyn<> 辅助构造 ========================

template <nint_t A, nint_t L, nint_t H>
constexpr Dynamic<A, L, H> dyn(nint_t v) { return Dynamic<A, L, H>{v}; }

template <nint_t A>
constexpr Dynamic<A> dyn(nint_t v) { return Dynamic<A>{v}; }

// ======================== 算术操作符 ========================
//
// 操作符语义：对Const与Dynamic做算术运算，在编译期传播约束信息（对齐量、上下界）。
// int/nint_t 类型的操作数会被自动包装为 Any（即 Dynamic<1>），等价于无约束运行时量。
//
// 约束传播规则摘要：
//   Const ± Const  → Const（值在编译期合并）
//   Const ± Dyn    → Dyn<gcd(A,N), ...>（对齐退化为gcd，界平移）
//   Dyn  ± Dyn     → Dyn<min(A1,A2), ...>（对齐取更弱，界线性合并）
//   Const × Const  → Const
//   Const × Dyn    → 若N=0 → Const<0>；否则 Dyn<A·lsb(N), ...>
//   Dyn  × Dyn     → Dyn<A1·A2, 四角积min/max>
//   Dyn  / Const<N>: A%N=0 → Dyn<A/N,...>; 否则 Dyn<1,...>（对齐退化）
//   Const / Dyn, Dyn / Dyn: 对齐退化到1
//   Dyn  % Const<N>: A%N=0 → Const<0>; 否则 Dyn<1,...>
//   Const % Dyn, Dyn % Dyn: 对齐退化到1
// 除法和求余Bounds由端点分析法精确计算（利用截断除法的分段单调性）

// ---- 加法 ----

template <nint_t N, nint_t M>
constexpr Const<N + M> operator+(Const<N>, Const<M>) { return Const<N + M>(); }

template <nint_t N, nint_t A, nint_t L, nint_t H>
constexpr auto operator+(Const<N>, Dynamic<A, L, H> rhs) {
  constexpr nint_t g = details::AlignAddSub<A, N>::value;
  constexpr nint_t rl = (L == kLoInf) ? kLoInf : L + N;
  constexpr nint_t rh = (H == kHiInf) ? kHiInf : H + N;
  return Dynamic<g, rl, rh>{N + rhs.value};
}

template <nint_t A, nint_t L, nint_t H, nint_t N>
constexpr auto operator+(Dynamic<A, L, H> lhs, Const<N>) {
  constexpr nint_t g = details::AlignAddSub<A, N>::value;
  constexpr nint_t rl = (L == kLoInf) ? kLoInf : L + N;
  constexpr nint_t rh = (H == kHiInf) ? kHiInf : H + N;
  return Dynamic<g, rl, rh>{lhs.value + N};
}

template <nint_t A1, nint_t L1, nint_t H1, nint_t A2, nint_t L2, nint_t H2>
constexpr auto operator+(Dynamic<A1, L1, H1> lhs, Dynamic<A2, L2, H2> rhs) {
  constexpr nint_t g = details::AlignAddSubDyn<A1, A2>::value;
  constexpr nint_t rl = (L1 == kLoInf || L2 == kLoInf) ? kLoInf : L1 + L2;
  constexpr nint_t rh = (H1 == kHiInf || H2 == kHiInf) ? kHiInf : H1 + H2;
  return Dynamic<g, rl, rh>{lhs.value + rhs.value};
}

// Value + nint_t  → Value + Any{nint_t}
template <typename T, std::enable_if_t<std::is_base_of_v<Value, T> && !is_int<T>, bool> = true>
constexpr auto operator+(T lhs, nint_t rhs) {
  return lhs + Any{rhs};
}

template <typename T, std::enable_if_t<std::is_base_of_v<Value, T> && !is_int<T>, bool> = true>
constexpr auto operator+(nint_t lhs, T rhs) {
  return Any{lhs} + rhs;
}

// ---- 减法 ----

template <nint_t N, nint_t M>
constexpr Const<N - M> operator-(Const<N>, Const<M>) { return Const<N - M>(); }

template <nint_t N, nint_t A, nint_t L, nint_t H>
constexpr auto operator-(Const<N>, Dynamic<A, L, H> rhs) {
  constexpr nint_t g = details::AlignAddSub<A, N>::value;
  constexpr nint_t rl = (H == kHiInf) ? kLoInf : N - H;
  constexpr nint_t rh = (L == kLoInf) ? kHiInf : N - L;
  return Dynamic<g, rl, rh>{N - rhs.value};
}

template <nint_t A, nint_t L, nint_t H, nint_t N>
constexpr auto operator-(Dynamic<A, L, H> lhs, Const<N>) {
  constexpr nint_t g = details::AlignAddSub<A, N>::value;
  constexpr nint_t rl = (L == kLoInf) ? kLoInf : L - N;
  constexpr nint_t rh = (H == kHiInf) ? kHiInf : H - N;
  return Dynamic<g, rl, rh>{lhs.value - N};
}

template <nint_t A1, nint_t L1, nint_t H1, nint_t A2, nint_t L2, nint_t H2>
constexpr auto operator-(Dynamic<A1, L1, H1> lhs, Dynamic<A2, L2, H2> rhs) {
  constexpr nint_t g = details::AlignAddSubDyn<A1, A2>::value;
  constexpr nint_t rl = (L1 == kLoInf || H2 == kHiInf) ? kLoInf : L1 - H2;
  constexpr nint_t rh = (H1 == kHiInf || L2 == kLoInf) ? kHiInf : H1 - L2;
  return Dynamic<g, rl, rh>{lhs.value - rhs.value};
}

// Value - nint_t
template <typename T, std::enable_if_t<std::is_base_of_v<Value, T> && !is_int<T>, bool> = true>
constexpr auto operator-(T lhs, nint_t rhs) {
  return lhs - Any{rhs};
}

template <typename T, std::enable_if_t<std::is_base_of_v<Value, T> && !is_int<T>, bool> = true>
constexpr auto operator-(nint_t lhs, T rhs) {
  return Any{lhs} - rhs;
}

// ---- 一元取负 ----

template <nint_t N>
constexpr Const<-N> operator-(Const<N>) { return Const<-N>(); }

template <nint_t A, nint_t L, nint_t H>
constexpr auto operator-(Dynamic<A, L, H> x) {
  constexpr nint_t rl = (H == kHiInf) ? kLoInf : -H;
  constexpr nint_t rh = (L == kLoInf) ? kHiInf : -L;
  return Dynamic<A, rl, rh>{-x.value};
}

// ---- 乘法 ----

template <nint_t N, nint_t M>
constexpr Const<N * M> operator*(Const<N>, Const<M>) { return Const<N * M>(); }

template <nint_t N, nint_t A, nint_t L, nint_t H>
constexpr auto operator*(Const<N>, Dynamic<A, L, H> rhs) {
  if constexpr (details::AlignMul<A, N>::is_zero) {
    return Const<0>{};
  } else {
    constexpr nint_t g = details::AlignMul<A, N>::value;
    if constexpr (L == kLoInf || H == kHiInf) {
      return Dynamic<g>{N * rhs.value};
    } else {
      constexpr nint_t rl = (N >= 0) ? N * L : N * H;
      constexpr nint_t rh = (N >= 0) ? N * H : N * L;
      return Dynamic<g, rl, rh>{N * rhs.value};
    }
  }
}

template <nint_t A, nint_t L, nint_t H, nint_t N>
constexpr auto operator*(Dynamic<A, L, H> lhs, Const<N>) {
  return Const<N>{} * lhs;
}

template <nint_t A1, nint_t L1, nint_t H1, nint_t A2, nint_t L2, nint_t H2>
constexpr auto operator*(Dynamic<A1, L1, H1> lhs, Dynamic<A2, L2, H2> rhs) {
  constexpr nint_t g = details::AlignMulDyn<A1, A2>::value;
  if constexpr (L1 == kLoInf || H1 == kHiInf || L2 == kLoInf || H2 == kHiInf) {
    return Dynamic<g>{lhs.value * rhs.value};
  } else {
    constexpr nint_t rl = std::min({L1 * L2, L1 * H2, H1 * L2, H1 * H2});
    constexpr nint_t rh = std::max({L1 * L2, L1 * H2, H1 * L2, H1 * H2});
    return Dynamic<g, rl, rh>{lhs.value * rhs.value};
  }
}

// Value * nint_t
template <typename T, std::enable_if_t<std::is_base_of_v<Value, T> && !is_int<T>, bool> = true>
constexpr auto operator*(T lhs, nint_t rhs) {
  return lhs * Any{rhs};
}

template <typename T, std::enable_if_t<std::is_base_of_v<Value, T> && !is_int<T>, bool> = true>
constexpr auto operator*(nint_t lhs, T rhs) {
  return Any{lhs} * rhs;
}

// ---- 除法 ----

namespace details {

constexpr nint_t ceil_div(nint_t a, nint_t b) {
  return a >= 0 ? (a + b - 1) / b : a / b;
}

constexpr nint_t floor_div(nint_t a, nint_t b) {
  return a >= 0 ? a / b : (a - b + 1) / b;
}

} // namespace details

template <nint_t N, nint_t M>
constexpr Const<N / M> operator/(Const<N>, Const<M>) { return Const<N / M>(); }

/// Dyn<A,L,H> / Const<N>:
///   对齐量：如果 A%N==0，保留 A/N；否则退化到1（因为不能保证商的可整除性）
///   Bounds：利用 f(k) = k*A/N 在k上单调，极值在k_min和k_max处
template <nint_t A, nint_t L, nint_t H, nint_t N>
constexpr auto operator/(Dynamic<A, L, H> lhs, Const<N>) {
  static_assert(N != 0, "division by zero");
  if constexpr (L == kLoInf || H == kHiInf) {
    constexpr nint_t g = (A % N == 0) ? A / N : 1;
    return Dynamic<g>{lhs.value / N};
  } else {
    constexpr nint_t k_min = details::ceil_div(L, A);
    constexpr nint_t k_max = details::floor_div(H, A);
    constexpr nint_t min_val = k_min * A / N;
    constexpr nint_t max_val = k_max * A / N;
    constexpr nint_t rl = std::min(min_val, max_val);
    constexpr nint_t rh = std::max(min_val, max_val);
    if constexpr (A % N == 0) {
      constexpr nint_t g = A / N;
      return Dynamic<g, rl, rh>{lhs.value / N};
    } else {
      return Dynamic<1, rl, rh>{lhs.value / N};
    }
  }
}

/// Const<N> / Dyn<A,L,H>:
///   对齐量退化为1（除非所有可能结果相同 → Const）
///   Bounds：利用 f(k)=N/(k*A) 在 k>0 和 k<0 上分别单调，极值在端点和±1处
template <nint_t N, nint_t A, nint_t L, nint_t H>
constexpr auto operator/(Const<N>, Dynamic<A, L, H> rhs) {
  if constexpr (L == kLoInf || H == kHiInf) {
    return Dynamic<1>{N / rhs.value};
  } else {
    constexpr nint_t k_min = details::ceil_div(L, A);
    constexpr nint_t k_max = details::floor_div(H, A);
    constexpr bool has_pos = (k_max >= 1);
    constexpr bool has_neg = (k_min <= -1);
    constexpr nint_t k_min_pos = has_pos ? ((k_min < 1) ? 1 : k_min) : 0;
    constexpr nint_t k_max_neg = has_neg ? ((k_max > -1) ? -1 : k_max) : 0;
    constexpr nint_t c1 = has_pos ? N / (k_min_pos * A) : 0;
    constexpr nint_t c2 = has_pos ? N / (k_max * A) : 0;
    constexpr nint_t c3 = has_neg ? N / (k_min * A) : 0;
    constexpr nint_t c4 = has_neg ? N / (k_max_neg * A) : 0;
    if constexpr (has_pos && has_neg) {
      constexpr nint_t rl = std::min({c1, c2, c3, c4});
      constexpr nint_t rh = std::max({c1, c2, c3, c4});
      return Dynamic<1, rl, rh>{N / rhs.value};
    } else if constexpr (has_pos) {
      constexpr nint_t rl = std::min(c1, c2);
      constexpr nint_t rh = std::max(c1, c2);
      return Dynamic<1, rl, rh>{N / rhs.value};
    } else {
      constexpr nint_t rl = std::min(c3, c4);
      constexpr nint_t rh = std::max(c3, c4);
      return Dynamic<1, rl, rh>{N / rhs.value};
    }
  }
}

/// Dyn / Dyn: 对齐退化为1
///   Bounds：若v2区间不跨0，在固定符号上除法对v1单调增、对v2单调（视v1符号），
///   极值在四角取得；若v2跨0，退回到无界（v2可能接近0导致商无界）
template <nint_t A1, nint_t L1, nint_t H1, nint_t A2, nint_t L2, nint_t H2>
constexpr auto operator/(Dynamic<A1, L1, H1> lhs, Dynamic<A2, L2, H2> rhs) {
    if constexpr (L1 == kLoInf || H1 == kHiInf || L2 == kLoInf || H2 == kHiInf) {
        return Dynamic<1>{lhs.value / rhs.value};
    } else if constexpr (L2 > 0) {
        // v2 all positive: f(v1,v2) monotonic → four-corner is exact
        constexpr nint_t rl = std::min({L1 / L2, L1 / H2, H1 / L2, H1 / H2});
        constexpr nint_t rh = std::max({L1 / L2, L1 / H2, H1 / L2, H1 / H2});
        return Dynamic<1, rl, rh>{lhs.value / rhs.value};
    } else if constexpr (H2 < 0) {
        // v2 all negative: similar, monotonic
        constexpr nint_t rl = std::min({L1 / L2, L1 / H2, H1 / L2, H1 / H2});
        constexpr nint_t rh = std::max({L1 / L2, L1 / H2, H1 / L2, H1 / H2});
        return Dynamic<1, rl, rh>{lhs.value / rhs.value};
    } else {
        // v2 crosses 0: quotient potentially unbounded
        return Dynamic<1>{lhs.value / rhs.value};
    }
}

// Value / nint_t
template <typename T, std::enable_if_t<std::is_base_of_v<Value, T> && !is_int<T>, bool> = true>
constexpr auto operator/(T lhs, nint_t rhs) {
  return lhs / Any{rhs};
}

template <typename T, std::enable_if_t<std::is_base_of_v<Value, T> && !is_int<T>, bool> = true>
constexpr auto operator/(nint_t lhs, T rhs) {
  return Any{lhs} / rhs;
}

// ---- 求余 (remainder) ----

template <nint_t N, nint_t M>
constexpr Const<N % M> operator%(Const<N>, Const<M>) { return Const<N % M>(); }

/// Dyn<A,L,H> % Const<N>:
///   如果 A%N==0，每个v都是N的倍数，结果必为0 → Const<0>
///   否则，对齐退化为1，余数范围取决于被除数符号
template <nint_t A, nint_t L, nint_t H, nint_t N_in>
constexpr auto operator%(Dynamic<A, L, H> lhs, Const<N_in>) {
  static_assert(N_in != 0, "modulo by zero");
  constexpr nint_t N = N_in;
  if constexpr (A % N == 0) {
    return Const<0>{};
  } else {
    if constexpr (L == kLoInf || H == kHiInf) {
      return Dynamic<1>{lhs.value % N};
    } else {
      constexpr nint_t rl = (L >= 0) ? 0 : -(N > 0 ? N : -N) + 1;
      constexpr nint_t rh = (H < 0) ? 0 : (N > 0 ? N : -N) - 1;
      constexpr nint_t lo = std::min(rl, rh);
      constexpr nint_t hi = std::max(rl, rh);
      return Dynamic<1, lo, hi>{lhs.value % N};
    }
  }
}

/// Const<N> % Dyn<A,L,H>: 对齐退化为1
template <nint_t N, nint_t A, nint_t L, nint_t H>
constexpr auto operator%(Const<N>, Dynamic<A, L, H> rhs) {
  if constexpr (L == kLoInf || H == kHiInf) {
    return Dynamic<1>{N % rhs.value};
  } else {
    constexpr nint_t k_min = details::ceil_div(L, A);
    constexpr nint_t k_max = details::floor_div(H, A);
    constexpr bool has_pos = (k_max >= 1);
    constexpr bool has_neg = (k_min <= -1);
    constexpr nint_t k_min_pos = has_pos ? ((k_min < 1) ? 1 : k_min) : 0;
    constexpr nint_t k_max_neg = has_neg ? ((k_max > -1) ? -1 : k_max) : 0;
    constexpr nint_t c1 = has_pos ? N % (k_min_pos * A) : 0;
    constexpr nint_t c2 = has_pos ? N % (k_max * A) : 0;
    constexpr nint_t c3 = has_neg ? N % (k_min * A) : 0;
    constexpr nint_t c4 = has_neg ? N % (k_max_neg * A) : 0;
    if constexpr (has_pos && has_neg) {
      constexpr nint_t rl = std::min({c1, c2, c3, c4});
      constexpr nint_t rh = std::max({c1, c2, c3, c4});
      return Dynamic<1, rl, rh>{N % rhs.value};
    } else if constexpr (has_pos) {
      constexpr nint_t rl = std::min(c1, c2);
      constexpr nint_t rh = std::max(c1, c2);
      return Dynamic<1, rl, rh>{N % rhs.value};
    } else {
      constexpr nint_t rl = std::min(c3, c4);
      constexpr nint_t rh = std::max(c3, c4);
      return Dynamic<1, rl, rh>{N % rhs.value};
    }
  }
}

/// Dyn % Dyn:
///  对齐退化到1，Bounds利用C++余数性质保守估计：
///  结果符号跟随被除数，|result| < |v2|。
///  故 |result| <= max(|L2|,|H2|) - 1
template <nint_t A1, nint_t L1, nint_t H1, nint_t A2, nint_t L2, nint_t H2>
constexpr auto operator%(Dynamic<A1, L1, H1> lhs, Dynamic<A2, L2, H2> rhs) {
    if constexpr (L1 == kLoInf || H1 == kHiInf || L2 == kLoInf || H2 == kHiInf) {
        return Dynamic<1>{lhs.value % rhs.value};
    } else {
        constexpr nint_t max_abs_v2 = std::max(L2 < 0 ? -L2 : L2, H2 < 0 ? -H2 : H2);
        constexpr nint_t rl = (L1 < 0) ? -(max_abs_v2 - 1) : 0;
        constexpr nint_t rh = (H1 >= 0) ? (max_abs_v2 - 1) : 0;
        constexpr nint_t lo = std::min(rl, rh);
        constexpr nint_t hi = std::max(rl, rh);
        return Dynamic<1, lo, hi>{lhs.value % rhs.value};
    }
}

// Value % nint_t
template <typename T, std::enable_if_t<std::is_base_of_v<Value, T> && !is_int<T>, bool> = true>
constexpr auto operator%(T lhs, nint_t rhs) {
  return lhs % Any{rhs};
}

template <typename T, std::enable_if_t<std::is_base_of_v<Value, T> && !is_int<T>, bool> = true>
constexpr auto operator%(nint_t lhs, T rhs) {
  return Any{lhs} % rhs;
}

namespace details {

/**
 * 将用户参数输入中的整数v转化为Any{v}
 * @tparam T
 */
template <typename T, typename = void/*SFINAE*/>
struct ValuePromote { using Type = T; };
template <typename T>
struct ValuePromote<T, std::enable_if_t<is_int<T>>> { using Type = Any; };

template <typename T>
struct PickConstValue { static constexpr nint_t value = 0; };

template <nint_t N>
struct PickConstValue<Const < N>> {
static constexpr nint_t value = N;
};

template <typename... Is>
VECOPS_INLINE constexpr void zip_packed_values(nint_t* v_out, const Is& ... v_in) {
  int idx = 0;
  ([&] {
    if constexpr (!std::remove_cvref_t<Is>::is_const) {
      v_out[idx++] = nint_t(v_in);
    }
  }(), ...);
}

template <typename... Is>
VECOPS_INLINE constexpr void zip_packed_values(nint_t* v_out, const nint_t* v_in) {
  int out_idx = 0;
  int in_idx = 0;
  ([&] {
    if constexpr (std::remove_cvref_t<Is>::is_const) {
      ++in_idx;
    } else {
      v_out[out_idx++] = v_in[in_idx++];
    }
  }(), ...);
}

template <typename... Is>
VECOPS_INLINE constexpr void unzip_packed_values(nint_t* v_out, const nint_t* v_in) {
  int out_idx = 0;
  int in_idx = 0;
  ([&] {
    if constexpr (std::remove_cvref_t<Is>::is_const) {
      v_out[out_idx++] = PickConstValue<std::remove_cvref_t<Is>>::value;
    } else {
      v_out[out_idx++] = v_in[in_idx++];
    }
  }(), ...);
}

/**
 * 压缩存储，存储一系列整数值，不会实际存储其中的编译期常量，从而节省存储空间
 * @tparam Is
 */
template <typename... Is>
struct PackedStorage {
  static constexpr bool is_runtime[] = {Is::is_runtime...};
  static constexpr int n_dim = sizeof...(Is);
private:
  static constexpr nint_t const_values[] = {PickConstValue<Is>::value...};
  /**
  * 生成一个将Is下标映射到压缩数组下标的映射表
  * @return (映射表数组, 压缩数组长度)
  */
  static constexpr auto stor_data = [] {
    std::array<int, sizeof...(Is)> arr;
    int off = 0;
    for (int i = 0; i < sizeof...(Is); ++i) {
      arr[i] = off;
      off += int(is_runtime[i]);
    }
    return std::make_pair(arr, off);
  }();
public:
  static constexpr std::array<int, sizeof...(Is)> offsets = stor_data.first;
  static constexpr int num_stor = stor_data.second;

  constexpr PackedStorage() = default;

  /**
   * 构造压缩存储，输入非压缩原始数据，函数内部会进行压缩
   */
  constexpr PackedStorage(Is... values) {
    VECOPS_ASSERT((Is::conforms(nint_t(values)) && ...), "values do not conform to type constraints");
    zip_packed_values<Is...>(this->values.data(), values...);
  }

  constexpr PackedStorage(const nint_t* values) {
    int idx = 0;
    VECOPS_ASSERT(((Is::conforms(values[idx++])) && ...), "values do not conform to type constraints");
    zip_packed_values<Is...>(this->values.data(), values);
  }

  /**
   * 拿到第I维的数据，如果是常量则函数返回constexpr，否则返回运行时值。
   */
  template <int I>
  [[nodiscard]] constexpr nint_t get() const {
    if constexpr (is_runtime[I]) {
      return values[offsets[I]];
    } else {
      return const_values[I];
    }
  }

  /**
   * 拿到第I维的数据，总是返回运行时量。
   */
  [[nodiscard]] nint_t operator[](int i) const {
    VECOPS_ASSERT(0 <= i && i < n_dim, "%d !in 0..%d", i, n_dim);
    if (is_runtime[i]) {
      return values[offsets[i]];
    } else {
      return const_values[i];
    }
  }

  /**
   * 拿到此压缩存储对应的非压缩原始数据
   */
  [[nodiscard]] constexpr std::array<nint_t, n_dim> to_array() const {
    std::array<nint_t, n_dim> arr;
    unzip_packed_values<Is...>(arr.data(), this->values.data());
    return arr;
  }

  /**
   * 拿到此压缩存储的压缩数据数组
   */
  [[nodiscard]] constexpr const std::array<nint_t, num_stor>& to_packed_array() const {
    return this->values;
  }

  /**
   * 从压缩数组直接初始化一个压缩存储
   */
  static constexpr PackedStorage from_packed_array(const nint_t* values) {
    PackedStorage self;
    std::copy(values, values + num_stor, self.values.begin());
    return self;
  }

private:
  std::array<nint_t, num_stor> values;
}; // struct PackedStorage

} // namespace details

/**
 * 将类型转换为Value包装，比如整数会被转为Any{v}
 */
template <typename T>
using ToValue = details::ValuePromote<T>::Type;

/**
 * 多维数组元数据基类，比如Shape或者Stride
 * @tparam Is
 */
template <typename... Is>
struct ArrayMeta {
  static_assert((std::is_base_of_v<Value, Is> && ...), "Is is not Value");
  static_assert(sizeof...(Is) > 0, "ndim cannot be 0");
  static constexpr int Ndim = sizeof...(Is);

  constexpr ArrayMeta() = default;

  template <typename... Ints>
  constexpr explicit ArrayMeta(Ints... vs) {
    static_assert(sizeof...(Ints) == sizeof...(Is), "MatrixMeta: argument count mismatch");

    _stor = details::PackedStorage < Is...>{ Is{vs}... };
  }

  template <int I>
  constexpr nint_t get() const {
    return _stor.template get<I>();
  }

  constexpr nint_t operator[](int i) const {
    return _stor[i];
  }

  template <int I>
  constexpr bool is_const() const {
    return !this->template is_runtime<I>();
  }

  template <int I>
  constexpr bool is_runtime() const {
    return decltype(_stor)::is_runtime[I];
  }

  constexpr int ndim() const {
    return Ndim;
  }

  details::PackedStorage<Is...> _stor;
}; // struct ArrayMeta

/**
 * 高维数组形状参数，每一维度参数必须非负
 */
template <typename... Is>
struct Shape : public ArrayMeta<Is...> {
  template <typename... Ints>
  constexpr Shape(Ints... is) : ArrayMeta<Is...>(is...) {
    VECOPS_ASSERT(((nint_t(is) >= 0) && ...), "is must be non-negative");
  }

  constexpr Shape() = default;
}; // struct Shape

template <typename... Ints>
constexpr auto make_shape(Ints&& ... is) -> Shape<ToValue<std::remove_cvref_t<Ints>>...> {
  return {std::forward<Ints>(is)...};
}

/**
 * 高维数组步长参数，每一维参数可为负数
 */
template <typename... Is>
struct Strides : public ArrayMeta<Is...> {
  template <typename... Ints>
  constexpr Strides(Ints... is) : ArrayMeta<Is...>(is...) {
  }

  constexpr Strides() = default;
}; // class Strides

template <typename... Ints>
constexpr auto make_strides(Ints&& ... is) -> Strides<ToValue<std::remove_cvref_t<Ints>>...> {
  return {std::forward<Ints>(is)...};
}

namespace details {

template <typename T>
struct IsArrayMeta : std::false_type {};
template <typename... Is>
struct IsArrayMeta<ArrayMeta<Is...>> : std::true_type {};
template <typename... Is>
struct IsArrayMeta<Shape<Is...>> : std::true_type {};
template <typename... Is>
struct IsArrayMeta<Strides<Is...>> : std::true_type {};

template <typename T>
struct IsShape : std::false_type {};
template <typename... Is>
struct IsShape<Shape<Is...>> : std::true_type {};

template <typename T>
struct IsStrides : std::false_type {};
template <typename... Is>
struct IsStrides<Strides<Is...>> : std::true_type {};


template <
    template <typename... xIs> typename Meta, // 目标Meta类型，模板类型用于随后填入类型值
    int N, // 原始Is长度
    int I, // 当前下标
    int J, // 目标下标
    typename = void, // SFINAE
    typename... InIs // 剩余未处理的类型值
>
struct ArrayMetaRemoveDim {
  static_assert(sizeof(N) == 0, "Unreachable");
};

template <
    template <typename... xIs> typename Meta,
    int N,
    int I,
    int J,
    typename InI0,
    typename... InIs
>
struct ArrayMetaRemoveDim<Meta, N, I, J, std::enable_if_t<(I < J)>, InI0, InIs...> {
  static_assert(0 <= I && I < N, "I out of range");

  // 由于有两个variadic template args，所以需要套两层
  template <typename... OutIs> // 已处理的类型值
  struct Holder {
    // 直接从高一级转发
    using Inner = typename ArrayMetaRemoveDim<Meta, N, I + 1, J, void, InIs...>
    ::template Holder<OutIs..., InI0>;
    using Type = Inner::Type;

    constexpr Type transform(const Meta<OutIs..., InI0, InIs...>& m) {
      return Inner{}.transform(m);
    }
  };
};

template <
    template <typename... xIs> typename Meta,
    int N,
    int I,
    typename InI0,
    typename... InIs
>
struct ArrayMetaRemoveDim<Meta, N, I, I, void, InI0, InIs...> {
  static_assert(0 <= I && I < N, "I out of range");

  // 由于有两个variadic template args，所以需要套两层
  template <typename... OutIs> // 已处理的类型值
  struct Holder {
    using Type = Meta<OutIs..., InIs...>; // 输入InI0被移除

    constexpr Type transform(const Meta<OutIs..., InI0, InIs...>& m) {
      std::array<nint_t, N - 1> out;
      auto in = m._stor.to_array();
      std::copy(in.data(), in.data() + I, out.data());
      std::copy(in.data() + I + 1, in.data() + N, out.data() + I);
      return [&] <size_t... Idx>(std::index_sequence<Idx...>) {
        return Type{out[Idx]...};
      }(std::make_index_sequence<N - 1>{});
    }
  };
};

template <int I, template <typename... xIs> typename TMeta, typename... Is>
constexpr auto remove_dim(const TMeta<Is...>& m) {
  return typename details::ArrayMetaRemoveDim<TMeta, int(sizeof...(Is)), 0, I, void, Is...>::template Holder<>{}.transform(m);
}


template <
    template <typename... xIs> typename Meta, // 目标Meta类型，模板类型用于随后填入类型值
    int N, // 原始Is长度
    int I, // 当前下标
    int J, // 目标下标
    typename InNew,
    typename = void, // SFINAE
    typename... InIs // 剩余未处理的类型值
>
struct ArrayMetaSetDim {
  static_assert(sizeof(N) == 0, "Unreachable");
};

template <
    template <typename... xIs> typename Meta,
    int N,
    int I,
    int J,
    typename InNew,
    typename InI0,
    typename... InIs
>
struct ArrayMetaSetDim<Meta, N, I, J, InNew, std::enable_if_t<(I < J)>, InI0, InIs...> {
  static_assert(0 <= I && I < N, "I out of range");

  // 由于有两个variadic template args，所以需要套两层
  template <typename... OutIs> // 已处理的类型值
  struct Holder {
    // 直接从高一级转发
    using Inner = typename ArrayMetaSetDim<Meta, N, I + 1, J, InNew, void, InIs...>
    ::template Holder<OutIs..., InI0>;
    using Type = Inner::Type;

    constexpr Type transform(const Meta<OutIs..., InI0, InIs...>& m, InNew v) {
      return Inner{}.transform(m, v);
    }
  };
};

template <
    template <typename... xIs> typename Meta,
    int N,
    int I,
    typename InNew,
    typename InI0,
    typename... InIs
>
struct ArrayMetaSetDim<Meta, N, I, I, InNew, void, InI0, InIs...> {
  static_assert(0 <= I && I < N, "I out of range");

  // 由于有两个variadic template args，所以需要套两层
  template <typename... OutIs> // 已处理的类型值
  struct Holder {
    using Type = Meta<OutIs..., InNew, InIs...>; // 输入InI0被替换为InNew

    constexpr Type transform(const Meta<OutIs..., InI0, InIs...>& m, InNew v) {
      std::array<nint_t, N> out;
      auto in = m._stor.to_array();
      std::copy(in.data(), in.data() + I, out.data());
      out[I] = nint_t(v);
      std::copy(in.data() + I + 1, in.data() + N, out.data() + I + 1);
      return [&] <size_t... Idx>(std::index_sequence<Idx...>) {
        return Type{out[Idx]...};
      }(std::make_index_sequence<N>{});
    }
  };
};

template <int I, typename Inew, template <typename... xIs> typename TMeta, typename... Is>
constexpr auto set_dim(const TMeta<Is...>& m, Inew v) {
  return typename details::ArrayMetaSetDim<TMeta, int(sizeof...(Is)), 0, I, Inew, void, Is...>::template Holder<>{}.transform(m, v);
}


template <
    template <typename... xIs> typename Meta, // 目标Meta类型，模板类型用于随后填入类型值
    int N, // 原始Is长度
    int I, // 当前下标
    int J, // 目标下标
    typename InNew,
    typename = void, // SFINAE
    typename... InIs // 剩余未处理的类型值
>
struct ArrayMetaInsertDim {
  static_assert(sizeof(N) == 0, "Unreachable");
};

template <
    template <typename... xIs> typename Meta,
    int N,
    int I,
    int J,
    typename InNew,
    typename InI0,
    typename... InIs
>
struct ArrayMetaInsertDim<Meta, N, I, J, InNew, std::enable_if_t<(I < J)>, InI0, InIs...> {
  static_assert(0 <= I && I <= N, "I out of range");

  // 由于有两个variadic template args，所以需要套两层
  template <typename... OutIs> // 已处理的类型值
  struct Holder {
    // 直接从高一级转发
    using Inner = typename ArrayMetaInsertDim<Meta, N, I + 1, J, InNew, void, InIs...>
    ::template Holder<OutIs..., InI0>;
    using Type = Inner::Type;

    constexpr Type transform(const Meta<OutIs..., InI0, InIs...>& m, InNew v) {
      return Inner{}.transform(m, v);
    }
  };
};

template <
    template <typename... xIs> typename Meta,
    int N,
    int I,
    typename InNew,
    typename... InIs
>
struct ArrayMetaInsertDim<Meta, N, I, I, InNew, void, InIs...> {
  static_assert(0 <= I && I <= N, "I out of range");

  // 由于有两个variadic template args，所以需要套两层
  template <typename... OutIs> // 已处理的类型值
  struct Holder {
    using Type = Meta<OutIs..., InNew, InIs...>; // 输入InI0被替换为InNew

    constexpr Type transform(const Meta<OutIs..., InIs...>& m, InNew v) {
      std::array<nint_t, N + 1> out;
      auto in = m._stor.to_array();
      std::copy(in.data(), in.data() + I, out.data());
      out[I] = nint_t(v);
      std::copy(in.data() + I, in.data() + N, out.data() + I + 1);
      return [&] <size_t... Idx>(std::index_sequence<Idx...>) {
        return Type{out[Idx]...};
      }(std::make_index_sequence<N + 1>{});
    }
  };
};

template <int I, typename Inew, template <typename... xIs> typename TMeta, typename... Is>
constexpr auto insert_dim(const TMeta<Is...>& m, Inew v) {
  return typename details::ArrayMetaInsertDim<TMeta, int(sizeof...(Is)), 0, I, Inew, void, Is...>::template Holder<>{}.transform(m, v);
}

} // namespace details

/**
 * 检查类型T是否是ArrayMeta
 */
template <typename T>
static constexpr bool is_array_meta = details::IsArrayMeta<T>::value;
/**
 * 检查类型T是否是Shape
 */
template <typename T>
static constexpr bool is_shape = details::IsShape<T>::value;
/**
 * 检查类型T是否是Strides
 */
template <typename T>
static constexpr bool is_strides = details::IsStrides<T>::value;

/**
 * 获取ArrayMeta (Shape, Stride)中第I维的数据
 * 如果第I维是Const，则函数返回constexpr，否则返回运行时量
 */
template <int I, typename TMeta, std::enable_if_t<is_array_meta<TMeta>, bool> = true>
constexpr nint_t get(const TMeta& m) {
  return m.template get<I>();
}

/**
 * 获取ArrayMeta中第I维的数据是否为Const
 */
template <int I, typename TMeta, std::enable_if_t<is_array_meta<TMeta>, bool> = true>
constexpr bool is_const(const TMeta& m) {
  return m.template is_const<I>();
}

/**
 * 获取ArrayMeta中第I维的数据是否为运行时量
 */
template <int I, typename TMeta, std::enable_if_t<is_array_meta<TMeta>, bool> = true>
constexpr bool is_runtime(const TMeta& m) {
  return m.template is_runtime<I>();
}


/**
 * 一个高维数组的内存布局，包含Shape和Stride。
 * @tparam TShape
 * @tparam TStrides
 */
template <typename TShape, typename TStrides>
struct Layout {
  static_assert(is_shape<TShape>, "TShape must be Shape<_,_>");
  static_assert(is_strides<TStrides>, "TStride must be Stride<_,_>");
  static_assert(TShape::Ndim == TStrides::Ndim, "TShape and TStride must have the same rank");
  static constexpr int Ndim = TShape::Ndim;

  using Shape = TShape;
  using Stride = TStrides;

  constexpr Layout(TShape shape, TStrides stride) : _shape(shape), _stride(stride) {
  }

  constexpr const TShape& shape() const {
    return _shape;
  }

  constexpr const TStrides& strides() const {
    return _stride;
  }

  constexpr int ndim() const {
    return Ndim;
  }

private:
  TShape _shape;
  TStrides _stride;
}; // struct Layout

template <typename TShape, typename TStrides>
constexpr auto make_layout(TShape&& shape, TStrides&& stride) -> Layout<std::remove_cvref_t<TShape>, std::remove_cvref_t<TStrides>> {
  return {std::forward<TShape>(shape), std::forward<TStrides>(stride)};
}

namespace details {

template <typename T>
struct IsLayout : std::false_type {};
template <typename TShape, typename TStrides>
struct IsLayout<Layout<TShape, TStrides>> : std::true_type {};

} // namespace details

/**
 * 检查类型T是否是Layout
 */
template <typename T>
static constexpr bool is_layout = details::IsLayout<T>::value;

/**
 * 获取布局layout第I维的大小（Shape数值），此值应是非负的。
 */
template <int I, typename TLayout, std::enable_if_t<is_layout<TLayout>, bool> = true>
constexpr nint_t size(const TLayout& layout) {
  return get<I>(layout.shape());
}

/**
 * 获取布局layout第I维的步长（Stride数值）
 */
template <int I, typename TLayout, std::enable_if_t<is_layout<TLayout>, bool> = true>
constexpr nint_t stride(const TLayout& layout) {
  return get<I>(layout.strides());
}

/**
 * 移除第I维
 */
template <int I, typename TMeta, std::enable_if_t<is_array_meta<TMeta>, bool> = true>
constexpr auto remove(const TMeta& m) { return details::remove_dim<I>(m); }

template <int I, typename TLayout, std::enable_if_t<is_layout<TLayout>, bool> = true>
constexpr auto remove(const TLayout& m) {
  return make_layout(remove<I>(m.shape()), remove<I>(m.strides()));
}

/**
 * 将第I维设置为指定的值
 */
template <int I, typename TMeta, typename Inew, std::enable_if_t<is_array_meta<TMeta>, bool> = true>
constexpr auto set(const TMeta& m, Inew v) { return details::set_dim<I>(m, v); }

template <int I, typename TLayout, typename IShapeNew, typename IStridesNew, std::enable_if_t<is_layout<TLayout>, bool> = true>
constexpr auto set(const TLayout& m, IShapeNew v_size, IStridesNew v_stride) {
  return make_layout(set<I>(m.shape(), v_size), set<I>(m.strides(), v_stride));
}

/**
 * 在第I维之前插入指定的值，如果I==Ndim，则在最后一维后面插入此值
 */
template <int I, typename TMeta, typename Inew, std::enable_if_t<is_array_meta<TMeta>, bool> = true>
constexpr auto insert(const TMeta& m, Inew v) { return details::insert_dim<I>(m, v); }

template <int I, typename TLayout, typename IShapeNew, typename IStridesNew, std::enable_if_t<is_layout<TLayout>, bool> = true>
constexpr auto insert(const TLayout& m, IShapeNew v_size, IStridesNew v_stride) {
  return make_layout(insert<I>(m.shape(), v_size), insert<I>(m.strides(), v_stride));
}


// ============ I/O Support ============

/**
 * 输出 ArrayMeta (Shape / Strides) 为紧凑元组形式 "(v0, v1, ...)"。
 *
 * 维度值输出规则：
 *   - Const<N>             → N!
 *   - Any{v} (=Dynamic<1>)  → v
 *   - Dynamic<A,_,_>{v}     → v@A   (当 A>1)
 *   - Dynamic<A,Lo,Hi>{v}   → v@A[Lo,Hi] (当有上下界时)
 */
template <typename... Is>
std::ostream& operator<<(std::ostream& os, const ArrayMeta<Is...>& m) {
  os << '(';
  [&] <size_t... Idx>(std::index_sequence<Idx...>) {
    int i = 0;
    ([&] {
      if (i++) os << ", ";
      nint_t v = m.template get<Idx>();
      using DimType = std::tuple_element_t<Idx, std::tuple<Is...>>;
      os << v;
      if constexpr (DimType::is_const) {
        os << '!';
      } else {
        if constexpr (DimType::alignment > 1) {
          os << '@' << DimType::alignment;
        }
        if constexpr (DimType::has_lower && DimType::has_upper) {
          os << '[' << DimType::lo << ',' << DimType::hi << ']';
        } else if constexpr (DimType::has_lower) {
          os << '[' << DimType::lo << ",∞)";
        } else if constexpr (DimType::has_upper) {
          os << "[0," << DimType::hi << ']';
        }
      }
    }(), ...);
  }(std::index_sequence_for<Is...>{});
  os << ')';
  return os;
}

/**
 * 输出 Layout<TShape, TStrides> 为 "Layout(s=(...), st=(...))" 形式。
 */
template <typename TShape, typename TStrides>
std::ostream& operator<<(std::ostream& os, const Layout<TShape, TStrides>& l) {
  os << "Layout(s=" << l.shape() << ", st=" << l.strides() << ')';
  return os;
}


} // namespace vecops::gemm

#endif //VECOPS_LAYOUT_H
