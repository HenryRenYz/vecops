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

/**
 * @file Layout.h
 * @brief Compile-time tensor layout metadata and constraint system.
 *
 * This header defines the foundational types for describing multi-dimensional
 * tensor shapes and strides with compile-time constraint propagation. The
 * core concept is a Value hierarchy that separates compile-time constants
 * from run-time values while preserving alignment and bound constraints
 * through arithmetic operations.
 *
 * ## Key components
 *
 * | Component      | Purpose                                                   |
 * |----------------|-----------------------------------------------------------|
 * | Value hierarchy| Compile-time/run-time typed integers with constraints     |
 * | PackedStorage  | Space-efficient storage eliding compile-time constants    |
 * | ArrayMeta      | Multi-dimensional typed integer arrays (base for Shape/Strides) |
 * | Shape          | Non-negative dimension sizes                              |
 * | Strides        | Stride values (may be negative)                           |
 * | Layout         | Pairs a Shape and Strides into a memory layout descriptor |
 * | Continuity     | Compile-time and run-time row-major contiguity checks     |
 *
 * ## Usage overview
 *
 * @code
 * #include "vecops/gemm/Layout.h"
 * using namespace vecops::gemm;
 *
 * // Create compile-time shapes
 * auto sh = make_shape(cint<2>, cint<3>);       // Shape<Const<2>, Const<3>>
 * auto st = make_strides(cint<3>, cint<1>);     // Strides<Const<3>, Const<1>>
 *
 * // Create runtime shapes with alignment constraints
 * auto sh2 = make_shape(cint<16>, dyn<4>(128)); // Shape<Const<16>, Dynamic<4>>
 *
 * // Build a layout
 * auto layout = make_layout(sh, st);
 *
 * // Arithmetic propagates constraints at compile time
 * auto stride2 = st.get<0>() * sh.get<1>();     // Const<6>
 * @endcode
 *
 * ## Pitfalls
 *
 * - `Const` cannot be default-constructed: its constructor asserts the value equals N.
 * - `Dynamic` construction asserts alignment/bounds constraints at runtime.
 * - Arithmetic with raw `nint_t` promotes to `Any`, losing all constraint info.
 * - `aligns(v)` is conservative: returning false means "not guaranteed", not "impossible".
 * - `remove_dim`, `set_dim`, `insert_dim` are O(N) compile-time type transformations.
 */

namespace vecops::gemm {

// ======================== Sentinel values for unbounded Dynamic ========================

/**
 * Sentinel for "no lower bound". Used as the default `Lo` parameter in
 * `Dynamic<A,Lo,Hi>` to indicate the runtime value has no compile-time
 * lower bound guarantee.
 */
constexpr nint_t kLoInf = std::numeric_limits<nint_t>::min();

/**
 * Sentinel for "no upper bound". Used as the default `Hi` parameter in
 * `Dynamic<A,Lo,Hi>` to indicate the runtime value has no compile-time
 * upper bound guarantee.
 */
constexpr nint_t kHiInf = std::numeric_limits<nint_t>::max();

// ======================== constexpr helpers ========================

namespace details {

/**
 * Extract the least significant set bit of `x`.
 * Used internally for computing alignment constraints during
 * arithmetic constraint propagation.
 *
 * @note Works correctly on two's-complement signed integers because
 *       `x & -x` isolates the lowest 1 bit regardless of sign.
 */
constexpr nint_t lsb(nint_t x) {
  using U = std::make_unsigned_t<nint_t>;
  return static_cast<nint_t>(static_cast<U>(x) & -static_cast<U>(x));
}

/**
 * Helper for addition/subtraction alignment constraint propagation.
 * Computes the resulting alignment when adding/subtracting a value
 * with alignment `A` and a Const value `N`.
 *
 * - A = 0 means the input is Const with alignment "infinity" (cannot be
 *   undermined by another operand) → result alignment is N.
 * - Otherwise the result alignment is min(A, lsb(N)).
 */
template <nint_t A, nint_t N>
struct AlignAddSub {
  static constexpr nint_t value = (N == 0) ? A : ((A < lsb(N)) ? A : lsb(N));
};

/**
 * Helper for Dynamic+Dynamic addition/subtraction alignment propagation.
 * Computes the minimum of two alignments (the weaker alignment survives).
 */
template <nint_t A1, nint_t A2>
struct AlignAddSubDyn {
  static constexpr nint_t value = (A1 < A2) ? A1 : A2;
};

/**
 * Helper for multiplication alignment constraint propagation.
 * - When `N == 0`, multiplication produces `Const<0>` (handled by the caller).
 * - Otherwise result alignment = `A * lsb(N)`.
 */
template <nint_t A, nint_t N>
struct AlignMul {
  static constexpr bool is_zero = (N == 0);
  static constexpr nint_t value = A * lsb(N);
};

/**
 * Helper for Dynamic*Dynamic multiplication alignment propagation.
 * Result alignment = `A1 * A2`.
 */
template <nint_t A1, nint_t A2>
struct AlignMulDyn {
  static constexpr nint_t value = A1 * A2;
};

} // namespace details

// ======================== Value hierarchy ========================

/**
 * @brief Abstract base for tensor metadata values (shape entries, stride entries).
 *
 * The Value hierarchy represents integer metadata as **types** rather than
 * bare `nint_t`, enabling compile-time storage of:
 * - Whether the value is a compile-time constant or run-time variable
 * - Alignment constraints (runtime value % A == 0)
 * - Lower/upper bound guarantees
 *
 * These constraints propagate through arithmetic operations, allowing
 * the compiler to optimize based on partial knowledge of runtime values.
 *
 * ## Subclass contract
 *
 * All subclasses must implement:
 * - `is_const` / `is_runtime` – class-level predicates
 * - `conforms(v)` – does compile-time value `v` satisfy this type's constraints?
 * - `aligns(v)` – does **every** possible runtime value of this type align to `v`?
 * - `is_aligned(v)` – does **this** specific runtime instance align to `v`?
 * - `explicit operator nint_t()` – conversion to raw integer
 *
 * ## Pitfalls
 *
 * - `aligns(v)` is **conservative**: returning `false` does not mean the
 *   runtime value definitely misaligns, it only means "cannot guarantee".
 *   For example, `Any{4}.is_aligned(4)` is `true` even though `Any::aligns(4)`
 *   is `false`.
 * - `Value` provides default implementations (`is_const=false`, `is_runtime=false`,
 *   `aligns`/`is_aligned`/`conforms` all return `false`, `operator nint_t` returns -100).
 *   These exist to simplify the operator-overload SFINAE machinery; override
 *   all of them in your subclass.
 */
struct Value {
  /// Whether all instances of this type carry a compile-time known value.
  static constexpr bool is_const = false;
  /// Whether instances of this type carry a run-time value (stored per instance).
  static constexpr bool is_runtime = false;

  /**
   * Check whether a compile-time value `v` satisfies this type's constraints
   * (e.g., alignment, bounds).
   */
  static constexpr bool conforms(nint_t v) { return false; }

  /**
   * Check whether **every** possible runtime value of this type is guaranteed
   * to be divisible by `v`. This is a conservative check: returning `true`
   * means "always aligned"; returning `false` means "not proven".
   *
   * @note `false` does **not** mean a specific instance is misaligned
   *       (use `is_aligned()` for that).
   */
  static constexpr bool aligns(nint_t v) { return false; }

  /**
   * Check whether **this specific instance's** runtime value is divisible by `v`.
   *
   * @note `aligns(v)` may return `false` while `is_aligned(v)` returns `true`
   *       (e.g., `Any{4}.is_aligned(4) == true` whereas `Any::aligns(4) == false`).
   */
  bool is_aligned(nint_t v) { return false; }

  /// Convert to raw `nint_t`. Default implementation returns an invalid sentinel.
  constexpr explicit operator nint_t() const { return -100; }
}; // struct Value

/**
 * @brief A compile-time constant integer with value N.
 *
 * `Const<N>` always carries the value `N` and nothing else. The constructor
 * **asserts** that the provided runtime value equals `N` — so you cannot
 * accidentally construct a `Const<N>` with a mismatched value.
 *
 * ## Usage
 *
 * @code
 * Const<16> c;              // c == 16
 * Const<16> c2(16);         // OK
 * Const<16> c3(8);          // RUNTIME ASSERTION FAILURE — 8 != 16
 *
 * auto c = cint<16>;        // Convenience: same as Const<16>{}
 * @endcode
 *
 * ## Arithmetic behavior
 *
 * Arithmetically, `Const<N>` acts as the integer `N`, but operations with
 * `Dynamic` degrade the result to `Dynamic` (losing constness) because the
 * result depends on a runtime value.
 *
 * @tparam N The compile-time integer value. Can be negative.
 */
template <nint_t N>
struct Const : public Value {
  static constexpr bool is_const = true;
  static constexpr bool is_runtime = false;
  static constexpr nint_t value = N;

  static constexpr bool conforms(nint_t v) { return v == N; }

  static constexpr bool aligns(nint_t v) { return N % v == 0; }

  /**
   * Construct a `Const<N>`. The value `v` must equal `N`, otherwise a runtime
   * assertion fires.
   *
   * @note The default parameter (`v = N`) allows default construction.
   */
  constexpr explicit Const(nint_t v = N) {
    VECOPS_ASSERT(v == N, "%td != %td", v, N);
  }

  constexpr bool is_aligned(nint_t v) { return N % v == 0; }

  constexpr explicit operator nint_t() const { return N; }
}; // struct Const

/**
 * @brief Convenience variable template for constructing `Const<N>`.
 *
 * @code
 * auto c = cint<32>;   // Equivalent to Const<32>{}
 * @endcode
 */
template <nint_t N>
inline constexpr Const<N> cint{N};

/**
 * @brief A run-time integer with compile-time alignment and bound constraints.
 *
 * `Dynamic<Alignment, Lo, Hi>` stores a run-time `nint_t` value while
 * providing compile-time guarantees about its properties:
 *
 * - **Alignment**: The runtime value `v` satisfies `v % Alignment == 0`.
 *   `Alignment` must be a positive power of 2.
 * - **Lower bound**: `v >= Lo` (if `Lo != kLoInf`).
 * - **Upper bound**: `v <= Hi` (if `Hi != kHiInf`).
 *
 * These constraints propagate through arithmetic operations (see the
 * operator overloads below).
 *
 * ## Usage
 *
 * @code
 * // Aligned to 16, range [0, 1024]
 * Dynamic<16, 0, 1024> d(512);   // OK
 *
 * // Aligned to 4, no bounds
 * Dynamic<4> d2(100);            // Equivalent to Dynamic<4, kLoInf, kHiInf>
 *
 * // Construction asserts:
 * Dynamic<8, 0, 16> bad(10);     // RUNTIME ASSERTION — 10 % 8 != 0
 * @endcode
 *
 * ## Convenience constructors
 *
 * Use the `dyn<A,L,H>(v)` or `dyn<A>(v)` factory functions for cleaner syntax:
 * @code
 * auto d = dyn<8>(64);           // Dynamic<8>{64}
 * auto d2 = dyn<4,0,256>(128);   // Dynamic<4,0,256>{128}
 * @endcode
 *
 * ## Any: The unconstrained runtime value
 *
 * `Any` is an alias for `Dynamic<1>` — alignment 1 (everything aligns to 1),
 * no bounds. It is the default when a raw `nint_t` participates in Value
 * arithmetic.
 *
 * @tparam Alignment  Positive power-of-2 alignment requirement. Default 1 (no constraint).
 * @tparam Lo         Compile-time inclusive lower bound. `kLoInf` means no lower bound.
 * @tparam Hi         Compile-time inclusive upper bound. `kHiInf` means no upper bound.
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
   * Check whether a compile-time value `v` satisfies all constraints of
   * this Dynamic type: alignment, lower bound (if active), upper bound (if active).
   */
  static constexpr bool conforms(nint_t v) {
    return (v & (Alignment - 1)) == 0
           && (!has_lower || v >= Lo)
           && (!has_upper || v <= Hi);
  }

  /**
   * Check whether **every** possible value of this Dynamic type is
   * guaranteed to be divisible by `v`. Returns `true` when
   * `Alignment % v == 0`.
   *
   * @note Conservative: returns `false` when uncertain, even if some
   *       specific values may still align.
   */
  static constexpr bool aligns(nint_t v) { return Alignment % v == 0; }

  /**
   * Construct a `Dynamic` with the given runtime value.
   * Asserts that `v` satisfies all compile-time constraints (alignment, bounds).
   */
  constexpr explicit Dynamic(nint_t v) : value(v) {
    VECOPS_ASSERT(conforms(v),
                  "value %td fails Dynamic<A=%td, Lo=%td, Hi=%td> constraints", v, Alignment, Lo, Hi);
  }

  /**
   * Check whether **this specific instance's** runtime value is divisible by `v`.
   */
  constexpr bool is_aligned(nint_t v) const { return value % v == 0; }

  constexpr explicit operator nint_t() const { return value; }

  const nint_t value;
}; // struct Dynamic

/**
 * @brief Alias for `Dynamic<1>` — a completely unconstrained runtime value.
 *
 * `Any` is the default Value type when a raw `nint_t` participates in
 * Value arithmetic. It carries no alignment constraint (1 divides everything)
 * and no bounds.
 *
 * @code
 * Any a(42);        // Unconstrained runtime value
 * auto b = 42 + cint<3>;  // operator+(nint_t, Const<3>) → wraps 42 in Any
 * @endcode
 */
using Any = Dynamic<1>;

/**
 * @brief Wildcard type for compile-time metadata pattern matching.
 *
 * `any` is not a runtime metadata value. It is a type-level pattern used by
 * traits such as `is_lenient_v<Meta, Patterns...>` to mean "accept any Value
 * type in this position". Use `Any` when you need an unconstrained runtime
 * `Value`; use `any` only as a template matching wildcard.
 *
 * @code
 * using St = Strides<Const<8>, Const<2>>;
 * static_assert(is_lenient_v<St, any, Const<2>>);
 * @endcode
 */
struct any {};

// ======================== dyn<> auxiliary constructors ========================

/**
 * @brief Create a `Dynamic<A, L, H>` with explicit alignment and bounds.
 *
 * @code
 * auto d = dyn<8, 0, 1024>(512);   // Dynamic<8, 0, 1024>
 * @endcode
 *
 * @tparam A  Alignment (positive power of 2).
 * @tparam L  Lower bound.
 * @tparam H  Upper bound.
 * @param v  The runtime value.
 * @return Dynamic<A, L, H>{v}
 */
template <nint_t A, nint_t L, nint_t H>
constexpr Dynamic<A, L, H> dyn(nint_t v) { return Dynamic<A, L, H>{v}; }

/**
 * @brief Create a `Dynamic<A>` with alignment only (no bounds).
 *
 * @code
 * auto d = dyn<4>(128);   // Dynamic<4>
 * @endcode
 *
 * @tparam A  Alignment (positive power of 2).
 * @param v   The runtime value.
 * @return Dynamic<A>{v}
 */
template <nint_t A>
constexpr Dynamic<A> dyn(nint_t v) { return Dynamic<A>{v}; }

// ======================== Arithmetic operators ========================
//
// Operator semantics: Const and Dynamic arithmetic propagates constraint
// information (alignment, bounds) at compile time. Raw nint_t operands
// are automatically wrapped as Any (Dynamic<1>, unconstrained).
//
// Constraint propagation rules summary:
//   Const ± Const  → Const   (value merged at compile time)
//   Const ± Dyn    → Dyn<gcd(A,N), ...>  (alignment degrades, bounds shift)
//   Dyn  ± Dyn     → Dyn<min(A1,A2), ...> (weaker alignment, bounds merged)
//   Const × Const  → Const
//   Const × Dyn    → if N=0: Const<0>; else Dyn<A·lsb(N), ...>
//   Dyn  × Dyn     → Dyn<A1·A2, four-corner min/max bounds>
//   Dyn  / Const<N>: if A%N=0: Dyn<A/N,...>; else Dyn<1,...> (alignment degrades)
//   Const / Dyn, Dyn / Dyn: alignment degrades to 1
//   Dyn  % Const<N>: if A%N=0: Const<0>; else Dyn<1,...>
//   Const % Dyn, Dyn % Dyn: alignment degrades to 1
// Division and remainder bounds use endpoint analysis (piecewise monotonicity
// of truncating division).

// ---- Addition ----

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

/// Value + nint_t → Value + Any{nint_t}
template <typename T, std::enable_if_t<std::is_base_of_v<Value, T> && !is_int<T>, bool> = true>
constexpr auto operator+(T lhs, nint_t rhs) {
  return lhs + Any{rhs};
}

/// nint_t + Value → Any{nint_t} + Value
template <typename T, std::enable_if_t<std::is_base_of_v<Value, T> && !is_int<T>, bool> = true>
constexpr auto operator+(nint_t lhs, T rhs) {
  return Any{lhs} + rhs;
}

// ---- Subtraction ----

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

/// Value - nint_t → Value - Any{nint_t}
template <typename T, std::enable_if_t<std::is_base_of_v<Value, T> && !is_int<T>, bool> = true>
constexpr auto operator-(T lhs, nint_t rhs) {
  return lhs - Any{rhs};
}

/// nint_t - Value → Any{nint_t} - Value
template <typename T, std::enable_if_t<std::is_base_of_v<Value, T> && !is_int<T>, bool> = true>
constexpr auto operator-(nint_t lhs, T rhs) {
  return Any{lhs} - rhs;
}

// ---- Unary negation ----

template <nint_t N>
constexpr Const<-N> operator-(Const<N>) { return Const<-N>(); }

template <nint_t A, nint_t L, nint_t H>
constexpr auto operator-(Dynamic<A, L, H> x) {
  constexpr nint_t rl = (H == kHiInf) ? kLoInf : -H;
  constexpr nint_t rh = (L == kLoInf) ? kHiInf : -L;
  return Dynamic<A, rl, rh>{-x.value};
}

// ---- Multiplication ----

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

/// Value * nint_t → Value * Any{nint_t}
template <typename T, std::enable_if_t<std::is_base_of_v<Value, T> && !is_int<T>, bool> = true>
constexpr auto operator*(T lhs, nint_t rhs) {
  return lhs * Any{rhs};
}

/// nint_t * Value → Any{nint_t} * Value
template <typename T, std::enable_if_t<std::is_base_of_v<Value, T> && !is_int<T>, bool> = true>
constexpr auto operator*(nint_t lhs, T rhs) {
  return Any{lhs} * rhs;
}

// ---- Division ----

namespace details {

/**
 * Ceiling division: `ceil(a / b)` for integer `a`, `b` (b != 0).
 * Handles negative numerators correctly with truncating division.
 */
constexpr nint_t ceil_div(nint_t a, nint_t b) {
  return a >= 0 ? (a + b - 1) / b : a / b;
}

/**
 * Floor division: `floor(a / b)` for integer `a`, `b` (b != 0).
 * Handles negative numerators correctly with truncating division.
 */
constexpr nint_t floor_div(nint_t a, nint_t b) {
  return a >= 0 ? a / b : (a - b + 1) / b;
}

} // namespace details

template <nint_t N, nint_t M>
constexpr Const<N / M> operator/(Const<N>, Const<M>) { return Const<N / M>(); }

/**
 * Dynamic<A,L,H> / Const<N>:
 *   Alignment: if A % N == 0, result alignment = A/N; otherwise degrades to 1
 *     (quotient divisibility by N is not guaranteed).
 *   Bounds: f(k) = k*A/N is monotonic in k, so extreme values are at k_min and k_max.
 */
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

/**
 * Const<N> / Dynamic<A,L,H>:
 *   Alignment degrades to 1 (unless all possible results produce the same value → Const).
 *   Bounds: f(k) = N/(k*A) is monotonic on k>0 and k<0 separately,
 *   so extreme values are at endpoints and ±1.
 *
 *   @note When the Dynamic denominator range crosses zero, the quotient
 *         is potentially unbounded (division by arbitrarily small values).
 */
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

/**
 * Dynamic / Dynamic: alignment always degrades to 1.
 *   Bounds: if the denominator range does not cross zero, division is
 *   monotonic in both variables (with sign conventions); extreme values
 *   are at the four corners. If denominator crosses zero, bounds become
 *   unbounded (quotient can be arbitrarily large as denominator → 0).
 */
template <nint_t A1, nint_t L1, nint_t H1, nint_t A2, nint_t L2, nint_t H2>
constexpr auto operator/(Dynamic<A1, L1, H1> lhs, Dynamic<A2, L2, H2> rhs) {
  if constexpr (L1 == kLoInf || H1 == kHiInf || L2 == kLoInf || H2 == kHiInf) {
    return Dynamic<1>{lhs.value / rhs.value};
  } else if constexpr (L2 > 0) {
    // Denominator all positive: monotonic → four-corner is exact
    constexpr nint_t rl = std::min({L1 / L2, L1 / H2, H1 / L2, H1 / H2});
    constexpr nint_t rh = std::max({L1 / L2, L1 / H2, H1 / L2, H1 / H2});
    return Dynamic<1, rl, rh>{lhs.value / rhs.value};
  } else if constexpr (H2 < 0) {
    // Denominator all negative: similar, monotonic
    constexpr nint_t rl = std::min({L1 / L2, L1 / H2, H1 / L2, H1 / H2});
    constexpr nint_t rh = std::max({L1 / L2, L1 / H2, H1 / L2, H1 / H2});
    return Dynamic<1, rl, rh>{lhs.value / rhs.value};
  } else {
    // Denominator crosses zero: quotient potentially unbounded
    return Dynamic<1>{lhs.value / rhs.value};
  }
}

/// Value / nint_t → Value / Any{nint_t}
template <typename T, std::enable_if_t<std::is_base_of_v<Value, T> && !is_int<T>, bool> = true>
constexpr auto operator/(T lhs, nint_t rhs) {
  return lhs / Any{rhs};
}

/// nint_t / Value → Any{nint_t} / Value
template <typename T, std::enable_if_t<std::is_base_of_v<Value, T> && !is_int<T>, bool> = true>
constexpr auto operator/(nint_t lhs, T rhs) {
  return Any{lhs} / rhs;
}

// ---- Remainder ----

template <nint_t N, nint_t M>
constexpr Const<N % M> operator%(Const<N>, Const<M>) { return Const<N % M>(); }

/**
 * Dynamic<A,L,H> % Const<N>:
 *   If A % N == 0, every runtime value is a multiple of N, so the result
 *   is always 0 → Const<0>. Otherwise alignment degrades to 1 and the
 *   remainder range depends on the sign of the dividend.
 */
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

/**
 * Const<N> % Dynamic<A,L,H>: alignment degrades to 1.
 * Bounds computed from endpoint analysis similar to division.
 */
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

/**
 * Dynamic % Dynamic: alignment degrades to 1.
 * Uses C++ remainder semantics (sign follows dividend, |result| < |v2|).
 * Conservative bounds: |result| <= max(|L2|, |H2|) - 1.
 */
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

/// Value % nint_t → Value % Any{nint_t}
template <typename T, std::enable_if_t<std::is_base_of_v<Value, T> && !is_int<T>, bool> = true>
constexpr auto operator%(T lhs, nint_t rhs) {
  return lhs % Any{rhs};
}

/// nint_t % Value → Any{nint_t} % Value
template <typename T, std::enable_if_t<std::is_base_of_v<Value, T> && !is_int<T>, bool> = true>
constexpr auto operator%(nint_t lhs, T rhs) {
  return Any{lhs} % rhs;
}

namespace details {

/**
 * @brief Converts integer types to their Value wrapper equivalent.
 *
 * Non-integer types (already Value subclasses) pass through unchanged.
 * Integer types (int8_t, uint32_t, etc.) are promoted to `Any`.
 *
 * Used by `ToValue<T>` to normalize user-provided shape/stride parameters
 * so that bare integers are automatically wrapped as `Any{v}`.
 */
template <typename T, typename = void/*SFINAE*/>
struct ValuePromote { using Type = T; };
template <typename T>
struct ValuePromote<T, std::enable_if_t<is_int<T>>> { using Type = Any; };

// ======================== IsMoreLenientValue ========================

/**
 * @brief Check whether Value type `VSrc` can be implicitly converted to `VDst`.
 *
 * A Value type is "more lenient" (less strict) when it carries less
 * compile-time information: `Const<N>` is strictest, `Dynamic<A,L,H>` is
 * intermediate, and `Any` is the most lenient (no constraints at all).
 *
 * The implicit conversion is safe when:
 * - Same type → always OK
 * - `Const<N>` → `Dynamic<A,L,H>` if N satisfies A, L, H
 * - `Const<N>` → `Any` (always OK)
 * - `Dynamic<A1,L1,H1>` → `Dynamic<A2,L2,H2>` if constraints are relaxed
 * - `Dynamic<A,L,H>` → `Any` (always OK)
 *
 * Any other direction (e.g., `Dynamic` → `Const`) requires explicit `as<>()`.
 */
template <typename VSrc, typename VDst>
struct IsMoreLenientValue : std::false_type {};

// Same type
template <nint_t N>
struct IsMoreLenientValue<Const<N>, Const<N>> : std::true_type {};
template <nint_t A, nint_t L, nint_t H>
struct IsMoreLenientValue<Dynamic<A, L, H>, Dynamic<A, L, H>> : std::true_type {};
template <>
struct IsMoreLenientValue<Any, Any> : std::true_type {};

// Const<N> → Dynamic<A, L, H>: N must conform to alignment and bounds
template <nint_t N, nint_t A, nint_t L, nint_t H>
struct IsMoreLenientValue<Const<N>, Dynamic<A, L, H>>
    : std::bool_constant<(N & (A - 1)) == 0
                         && (L == kLoInf || N >= L)
                         && (H == kHiInf || N <= H)> {};

// Const<N> → Any
template <nint_t N>
struct IsMoreLenientValue<Const<N>, Any> : std::true_type {};

// Dynamic<A1,L1,H1> → Dynamic<A2,L2,H2>: A1 % A2 == 0 and bounds are relaxed
template <nint_t A1, nint_t L1, nint_t H1, nint_t A2, nint_t L2, nint_t H2>
struct IsMoreLenientValue<Dynamic<A1, L1, H1>, Dynamic<A2, L2, H2>>
    : std::bool_constant<(A1 % A2 == 0)
                         && (L2 == kLoInf || (L1 != kLoInf && L1 >= L2))
                         && (H2 == kHiInf || (H1 != kHiInf && H1 <= H2))> {};

// Dynamic<A, L, H> → Any
template <nint_t A, nint_t L, nint_t H>
struct IsMoreLenientValue<Dynamic<A, L, H>, Any> : std::true_type {};

/**
 * Helper to extract the compile-time value from a Const type, or 0
 * for non-Const types. Used by PackedStorage for the const_values array.
 */
template <typename T>
struct PickConstValue { static constexpr nint_t value = 0; };

template <nint_t N>
struct PickConstValue<Const < N>> {
static constexpr nint_t value = N;
};

/**
 * Pack runtime values from a parameter pack into a flat array,
 * skipping Const entries (they are stored outside the packed array).
 *
 * @param v_out  Output array (must have enough space for runtime entries).
 * @param v_in   Input values (mix of Const and Dynamic).
 */
template <typename... Is>
VECOPS_INLINE constexpr void zip_packed_values(nint_t* v_out, const Is& ... v_in) {
  int idx = 0;
  ([&] {
    if constexpr (!std::remove_cvref_t<Is>::is_const) {
      v_out[idx++] = nint_t(v_in);
    }
  }(), ...);
}

/**
 * Copy a packed array into an output array, filling Const entries from
 * compile-time type info and runtime entries from the packed array.
 * Inverse of zip_packed_values.
 */
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

/**
 * Unpack compress storage into a full array, filling Const entries from
 * compile-time type information.
 */
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
 * @brief Compressed storage for a sequence of Values.
 *
 * `PackedStorage<Is...>` stores the run-time values of a parameter pack
 * `Is...` while eliding compile-time constants from the actual storage.
 * This saves memory for metadata types where many dimensions are
 * compile-time `Const<N>`.
 *
 * ## Internal layout
 *
 * - `is_runtime[i]`: whether the i-th type is a Dynamic (run-time) type.
 * - `offsets[i]`: maps logical index i to its position in the compressed
 *   `values` array (skipping Const entries).
 * - `const_values[i]`: the compile-time value for Const entries.
 * - `num_stor`: number of actually stored runtime values.
 *
 * ## Usage
 *
 * @code
 * // PackedStorage<Const<2>, Dynamic<4>, Const<3>>
 * // stores only 1 runtime value (the Dynamic<4> entry)
 * PackedStorage<Const<2>, Dynamic<4>, Const<3>> stor(2, 128, 3);
 * auto v0 = stor.get<0>();   // 2 (constexpr)
 * auto v1 = stor.get<1>();   // 128 (runtime read)
 * auto arr = stor.to_array(); // {2, 128, 3}
 * @endcode
 *
 * @tparam Is  Parameter pack of Value subclasses (Const<N> or Dynamic<A,L,H>).
 */
template <typename... Is>
struct PackedStorage {
  static constexpr bool is_runtime[] = {Is::is_runtime...};
  static constexpr int n_dim = sizeof...(Is);
private:
  static constexpr nint_t const_values[] = {PickConstValue<Is>::value...};

  /// Fold over the type pack directly to compute offsets at compile time.
  /// Uses Is::is_runtime (class-level static) rather than is_runtime[i]
  /// which avoids clang's "array without known bound" constexpr limitation.
  template <size_t... Js>
  static constexpr auto _compute_offsets(std::index_sequence<Js...>) {
    std::array<int, sizeof...(Is)> arr{};
    int off = 0;
    ((arr[Js] = off, off += int(Is::is_runtime)), ...);
    return std::make_pair(arr, off);
  }
  static constexpr auto stor_data = _compute_offsets(std::index_sequence_for<Is...>{});
public:
  /**
   * Offset mapping: `offsets[i]` gives the position in the compressed
   * `values` array for logical dimension i (meaningless for Const entries).
   */
  static constexpr std::array<int, sizeof...(Is)> offsets = stor_data.first;
  /// Number of actually stored runtime values.
  static constexpr int num_stor = stor_data.second;

  constexpr PackedStorage() = default;

  /**
   * Construct from unpacked values. Internally compresses by storing
   * only the runtime entries.
   *
   * @note Asserts that each value conforms to its type's constraints.
   */
  constexpr PackedStorage(Is... values) {
    VECOPS_ASSERT((Is::conforms(nint_t(values)) && ...), "values do not conform to type constraints");
    zip_packed_values<Is...>(this->values.data(), values...);
  }

  /**
   * Construct from a packed (already compressed) array.
   *
   * @note Asserts that each value conforms to its type's constraints.
   */
  constexpr PackedStorage(const nint_t* values) {
    int idx = 0;
    VECOPS_ASSERT(((Is::conforms(values[idx++])) && ...), "values do not conform to type constraints");
    zip_packed_values<Is...>(this->values.data(), values);
  }

  /**
   * Get the value for dimension I. Returns a `constexpr` value for
   * Const entries, reads from the compressed array for Dynamic entries.
   *
   * @tparam I  Zero-based dimension index.
   * @return The value (constexpr or runtime depending on type).
   */
  template <int I>
  [[nodiscard]] constexpr nint_t get() const {
    using DimType = std::tuple_element_t<I, std::tuple<Is...>>;
    if constexpr (DimType::is_runtime) {
      return values[offsets[I]];
    } else {
      return PickConstValue<DimType>::value;
    }
  }

  /**
   * Get the value for dimension `i` at runtime. Always returns a
   * runtime `nint_t` regardless of whether the type is Const or Dynamic.
   *
   * @param i  Zero-based dimension index (runtime value).
   * @return The integer value.
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
   * Expand to a full unpacked array (restoring Const entries).
   */
  [[nodiscard]] constexpr std::array<nint_t, n_dim> to_array() const {
    std::array<nint_t, n_dim> arr;
    unzip_packed_values<Is...>(arr.data(), this->values.data());
    return arr;
  }

  /**
   * Get the internal compressed array (runtime values only, no Const entries).
   */
  [[nodiscard]] constexpr const std::array<nint_t, num_stor>& to_packed_array() const {
    return this->values;
  }

  /**
   * Construct from a pre-existing compressed array.
   *
   * @note The caller must ensure the array was produced by `zip_packed_values`
   *       with the same type parameter pack. No validation is performed.
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
 * @brief Convert a type to its Value wrapper.
 *
 * Integer types are promoted to `Any`; Value subclasses pass through unchanged.
 * This is used to normalize user-provided parameters in `make_shape`,
 * `make_strides`, etc.
 *
 * @code
 * using T = ToValue<int32_t>;   // T = Any
 * using U = ToValue<Const<4>>;  // U = Const<4>
 * @endcode
 */
template <typename T>
using ToValue = details::ValuePromote<T>::Type;

/**
 * @brief Base class for multi-dimensional typed integer arrays (Shape, Strides).
 *
 * `ArrayMeta<Is...>` is a fixed-rank container where each dimension's value
 * is represented by a Value subclass type (`Const<N>` or `Dynamic<A,L,H>`).
 * It uses `PackedStorage` internally to store only runtime values.
 *
 * ## Subclass usage
 *
 * Subclasses `Shape<Is...>` and `Strides<Is...>` enforce additional
 * semantic constraints (e.g., Shape values must be non-negative).
 *
 * ## Construction
 *
 * @code
 * // Explicit types
 * ArrayMeta<Const<2>, Dynamic<4>> m(2, 128);
 *
 * // Via make_shape
 * auto s = make_shape(cint<2>, 128);  // Shape<Const<2>, Any>
 * @endcode
 *
 * @tparam Is  Value subclass types for each dimension.
 */
template <typename... Is>
struct ArrayMeta {
  static_assert((std::is_base_of_v<Value, Is> && ...), "Is is not Value");
  static_assert(sizeof...(Is) > 0, "ndim cannot be 0");
  static constexpr int Ndim = sizeof...(Is);

  constexpr ArrayMeta() = default;

  /**
   * Construct from per-dimension values. Types are deduced from the
   * parameter types via ValuePromote.
   */
  template <typename... Ints>
  constexpr explicit ArrayMeta(Ints... vs) {
    static_assert(sizeof...(Ints) == sizeof...(Is), "MatrixMeta: argument count mismatch");

    _stor = details::PackedStorage < Is...>{ Is{vs}... };
  }

  /**
   * Get the compile-time (if Const) or runtime value for dimension I.
   * @tparam I  Zero-based dimension index.
   */
  template <int I>
  constexpr nint_t get() const {
    return _stor.template get<I>();
  }

  constexpr nint_t operator[](int i) const {
    return _stor[i];
  }

  /**
   * Check whether dimension I is a compile-time constant.
   */
  template <int I>
  constexpr bool is_const() const {
    return !this->template is_runtime<I>();
  }

  /**
   * Check whether dimension I holds a runtime value.
   */
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
 * @brief Multi-dimensional shape descriptor. All dimension values must be non-negative.
 *
 * `Shape<Is...>` inherits from `ArrayMeta<Is...>` and adds the constraint
 * that every dimension size >= 0. Construction asserts this.
 *
 * ## Usage
 *
 * @code
 * Shape<Const<2>, Const<3>> s(2, 3);          // OK
 * Shape<Const<2>> bad(-1);                     // RUNTIME ASSERTION
 *
 * auto s2 = make_shape(cint<2>, dyn<4>(128));  // Convenience factory
 * @endcode
 *
 * @tparam Is  Value types for each dimension.
 */
template <typename... Is>
struct Shape : public ArrayMeta<Is...> {
  template <typename... Ints>
  constexpr Shape(Ints... is) : ArrayMeta<Is...>(is...) {
    VECOPS_ASSERT(((nint_t(is) >= 0) && ...), "is must be non-negative");
  }

  constexpr Shape() = default;
}; // struct Shape

/**
 * @brief Create a Shape, automatically wrapping bare integers as Any.
 *
 * @code
 * auto s = make_shape(cint<3>, 128);  // Shape<Const<3>, Any>
 * @endcode
 */
template <typename... Ints>
constexpr auto make_shape(Ints&& ... is) -> Shape<ToValue<std::remove_cvref_t<Ints>>...> {
  return {std::forward<Ints>(is)...};
}

/**
 * @brief Multi-dimensional stride descriptor. Stride values may be negative
 *        (for reversed dimensions).
 *
 * Unlike `Shape`, `Strides` does **not** enforce a sign constraint.
 *
 * @tparam Is  Value types for each dimension.
 */
template <typename... Is>
struct Strides : public ArrayMeta<Is...> {
  template <typename... Ints>
  constexpr Strides(Ints... is) : ArrayMeta<Is...>(is...) {
  }

  constexpr Strides() = default;
}; // class Strides

/**
 * @brief Create Strides, automatically wrapping bare integers as Any.
 */
template <typename... Ints>
constexpr auto make_strides(Ints&& ... is) -> Strides<ToValue<std::remove_cvref_t<Ints>>...> {
  return {std::forward<Ints>(is)...};
}

namespace details {

// --- Type trait helpers for ArrayMeta / Shape / Strides ---

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


/**
 * @brief Compile-time dimension removal from an ArrayMeta type.
 *
 * Removes dimension `J` from the type `Meta<InIs...>`, producing
 * `Meta<OutIs..., InIs_without_J...>`.
 *
 * This is a compile-time O(N) operation implemented via recursive
 * template specialization with two nested Holder levels (two variadic
 * packs require double nesting).
 *
 * @tparam Meta  The target container template (e.g., Shape, Strides, ArrayMeta).
 * @tparam N     Original number of dimensions.
 * @tparam I     Current input index (internal recursion counter).
 * @tparam J     Target dimension to remove.
 * @tparam InIs  Remaining input types.
 */
template <
    template <typename... xIs> typename Meta,
    int N,
    int I,
    int J,
    typename = void, // SFINAE
    typename... InIs
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

  template <typename... OutIs>
  struct Holder {
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

  template <typename... OutIs>
  struct Holder {
    using Type = Meta<OutIs..., InIs...>;  // InI0 removed

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


/**
 * @brief Compile-time dimension replacement in an ArrayMeta type.
 *
 * Replaces dimension `J` with a new Value type `InNew`.
 *
 * @tparam Meta  The target container template.
 * @tparam N     Original number of dimensions.
 * @tparam I     Current input index (recursion counter).
 * @tparam J     Target dimension to replace.
 * @tparam InNew The new Value type for dimension J.
 */
template <
    template <typename... xIs> typename Meta,
    int N,
    int I,
    int J,
    typename InNew,
    typename = void,
    typename... InIs
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

  template <typename... OutIs>
  struct Holder {
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

  template <typename... OutIs>
  struct Holder {
    using Type = Meta<OutIs..., InNew, InIs...>;  // InI0 replaced by InNew

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


/**
 * @brief Compile-time dimension insertion into an ArrayMeta type.
 *
 * Inserts a new Value type `InNew` at position `J` (0 <= J <= N).
 * When `J == N`, the dimension is appended at the end.
 *
 * @tparam Meta  The target container template.
 * @tparam N     Original number of dimensions.
 * @tparam I     Current input index (recursion counter).
 * @tparam J     Target position for insertion.
 * @tparam InNew The Value type to insert at position J.
 */
template <
    template <typename... xIs> typename Meta,
    int N,
    int I,
    int J,
    typename InNew,
    typename = void,
    typename... InIs
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

  template <typename... OutIs>
  struct Holder {
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

  template <typename... OutIs>
  struct Holder {
    using Type = Meta<OutIs..., InNew, InIs...>;

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

/// Type trait: `true` if T is an ArrayMeta, Shape, or Strides.
template <typename T>
static constexpr bool is_array_meta = details::IsArrayMeta<T>::value;
/// Type trait: `true` if T is a Shape.
template <typename T>
static constexpr bool is_shape = details::IsShape<T>::value;
/// Type trait: `true` if T is a Strides.
template <typename T>
static constexpr bool is_strides = details::IsStrides<T>::value;

/**
 * @brief Get the value of dimension I from an ArrayMeta.
 *
 * Returns `constexpr` if the dimension is a compile-time `Const<N>`.
 *
 * @tparam I     Dimension index.
 * @tparam TMeta The ArrayMeta type (Shape, Strides, or ArrayMeta itself).
 * @param  m     The metadata container.
 * @return The integer value for dimension I.
 */
template <int I, typename TMeta, std::enable_if_t<is_array_meta<TMeta>, bool> = true>
constexpr nint_t get(const TMeta& m) {
  return m.template get<I>();
}

/**
 * @brief Check whether dimension I of an ArrayMeta is a compile-time constant.
 */
template <int I, typename TMeta, std::enable_if_t<is_array_meta<TMeta>, bool> = true>
constexpr bool is_const(const TMeta& m) {
  return m.template is_const<I>();
}

/**
 * @brief Check whether dimension I of an ArrayMeta is a runtime value.
 */
template <int I, typename TMeta, std::enable_if_t<is_array_meta<TMeta>, bool> = true>
constexpr bool is_runtime(const TMeta& m) {
  return m.template is_runtime<I>();
}

namespace details {

// ======================== Lenient metadata matching ========================

/**
 * @brief Check whether a Shape or Strides type can be converted to another
 *        metadata type dimension by dimension.
 *
 * Returns true only when the source and destination are both `Shape<...>` or
 * both `Strides<...>`, have the same rank, and each source dimension can be
 * converted to the corresponding destination dimension according to
 * `IsMoreLenientValue`.
 */
template <typename MSrc, typename MDst>
struct IsMoreLenientMeta : std::false_type {};

template <bool SameRank, typename MSrc, typename MDst>
struct IsMoreLenientMetaImpl : std::false_type {};

template <typename... Ss, typename... Ds>
struct IsMoreLenientMetaImpl<true, Shape<Ss...>, Shape<Ds...>>
    : std::bool_constant<(IsMoreLenientValue<Ss, Ds>::value && ...)> {};

template <typename... Ss, typename... Ds>
struct IsMoreLenientMetaImpl<true, Strides<Ss...>, Strides<Ds...>>
    : std::bool_constant<(IsMoreLenientValue<Ss, Ds>::value && ...)> {};

template <typename... Ss, typename... Ds>
struct IsMoreLenientMeta<Shape<Ss...>, Shape<Ds...>>
    : IsMoreLenientMetaImpl<sizeof...(Ss) == sizeof...(Ds), Shape<Ss...>, Shape<Ds...>> {};

template <typename... Ss, typename... Ds>
struct IsMoreLenientMeta<Strides<Ss...>, Strides<Ds...>>
    : IsMoreLenientMetaImpl<sizeof...(Ss) == sizeof...(Ds), Strides<Ss...>, Strides<Ds...>> {};

/**
 * @brief Match one metadata Value against a lenient pattern.
 *
 * The `any` wildcard accepts every Value type. Non-wildcard patterns reuse
 * `IsMoreLenientValue`, so `Const<N>` can match compatible `Dynamic<A,L,H>`
 * patterns, `Dynamic` can match less constrained `Dynamic` patterns, and so on.
 */
template <typename Actual, typename Pattern>
struct IsLenientPatternValue : IsMoreLenientValue<Actual, Pattern> {};

template <typename Actual>
struct IsLenientPatternValue<Actual, any> : std::true_type {};

template <bool SameRank, typename Meta, typename... Patterns>
struct IsLenientPatternMetaImpl : std::false_type {};

template <typename... Ss, typename... Patterns>
struct IsLenientPatternMetaImpl<true, Shape<Ss...>, Patterns...>
    : std::bool_constant<(IsLenientPatternValue<Ss, Patterns>::value && ...)> {};

template <typename... Ss, typename... Patterns>
struct IsLenientPatternMetaImpl<true, Strides<Ss...>, Patterns...>
    : std::bool_constant<(IsLenientPatternValue<Ss, Patterns>::value && ...)> {};

/**
 * @brief Check whether a Shape or Strides type matches a lenient pattern list.
 *
 * Rank must match exactly. Each pattern is either the `any` wildcard or a
 * Value type accepted by `IsMoreLenientValue`.
 */
template <typename Meta, typename... Patterns>
struct IsLenientPatternMeta : std::false_type {};

template <typename... Ss, typename... Patterns>
struct IsLenientPatternMeta<Shape<Ss...>, Patterns...>
    : IsLenientPatternMetaImpl<sizeof...(Ss) == sizeof...(Patterns), Shape<Ss...>, Patterns...> {};

template <typename... Ss, typename... Patterns>
struct IsLenientPatternMeta<Strides<Ss...>, Patterns...>
    : IsLenientPatternMetaImpl<sizeof...(Ss) == sizeof...(Patterns), Strides<Ss...>, Patterns...> {};

} // namespace details

/**
 * @brief Check whether a Shape or Strides type matches a lenient pattern list.
 *
 * This is a static type predicate. It does not inspect runtime shape or stride
 * values. Use `any` as a per-dimension wildcard, and use Value types such as
 * `Const<N>`, `Dynamic<A,L,H>`, or `Any` for constrained dimensions.
 *
 * @code
 * using St = Strides<Const<64>, Const<8>, Const<2>, Const<2>>;
 * static_assert(is_lenient_v<St, any, any, Dynamic<1, -1, 3>, Const<2>>);
 * @endcode
 */
template <typename Meta, typename... Patterns>
static constexpr bool is_lenient_v =
    details::IsLenientPatternMeta<std::remove_cvref_t<Meta>, Patterns...>::value;


/**
 * @brief Memory layout descriptor pairing a Shape and Strides.
 *
 * `Layout<TShape, TStrides>` describes the memory layout for a
 * multi-dimensional tensor: the shape specifies the size of each
 * dimension, and the strides specify the spacing (in elements) between
 * consecutive entries along each dimension. This follows row-major
 * conventions but allows arbitrary striding.
 *
 * ## Template parameters
 *
 * - `TShape`: A `Shape<...>` type encoding per-dimension sizes.
 * - `TStrides`: A `Strides<...>` type encoding per-dimension strides.
 *
 * Both must have the same rank (number of dimensions).
 *
 * ## Usage
 *
 * @code
 * auto s = make_shape(cint<2>, cint<3>);
 * auto st = make_strides(cint<3>, cint<1>);
 * auto layout = make_layout(s, st);
 *
 * // or directly
 * auto layout2 = make_layout(make_shape(2, 3), make_strides(3, 1));
 * @endcode
 *
 * @tparam TShape   Shape type.
 * @tparam TStrides Strides type.
 */
template <typename TShape, typename TStrides>
struct Layout {
  static_assert(is_shape<TShape>, "TShape must be Shape<_,_>");
  static_assert(is_strides<TStrides>, "TStride must be Stride<_,_>");
  static_assert(TShape::Ndim == TStrides::Ndim, "TShape and TStride must have the same rank");
  static constexpr int Ndim = TShape::Ndim;

  using Shape = TShape;
  using Stride = TStrides;
  using Strides = TStrides;

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

  /**
   * @brief Reinterpret this Layout with different Shape and Strides metadata.
   *
   * Runtime shape and stride values are copied from the current layout into
   * the target metadata types. Target constructors enforce their own
   * constraints, so converting to incompatible `Const<N>` or
   * `Dynamic<A,L,H>` types triggers the same assertions as direct
   * construction.
   *
   * @code
   * auto layout = make_layout(make_shape(Any{4}, Any{5}),
   *                           make_strides(Any{5}, Any{1}));
   * auto typed = layout.as<Shape<Const<4>, Const<5>>,
   *                        Strides<Const<5>, Const<1>>>();
   * @endcode
   *
   * @tparam TShape2   Target Shape type with the same rank.
   * @tparam TStrides2 Target Strides type with the same rank.
   */
  template <typename TShape2, typename TStrides2>
  constexpr Layout<TShape2, TStrides2> as() const {
    static_assert(is_shape<TShape2>, "TShape2 must be Shape<...>");
    static_assert(is_strides<TStrides2>, "TStrides2 must be Strides<...>");
    static_assert(TShape2::Ndim == Ndim, "Target Shape rank must match Layout rank");
    static_assert(TStrides2::Ndim == Ndim, "Target Strides rank must match Layout rank");

    auto s_arr = _shape._stor.to_array();
    auto t_arr = _stride._stor.to_array();
    return [&] <size_t... Idx>(std::index_sequence<Idx...>) {
      return Layout<TShape2, TStrides2>{
          TShape2{s_arr[Idx]...},
          TStrides2{t_arr[Idx]...}
      };
    }(std::make_index_sequence<Ndim>{});
  }

  /**
   * @brief Implicit conversion to a more lenient Layout type.
   *
   * A Layout can convert implicitly only when both Shape and Strides metadata
   * move toward equal-or-more-lenient Value types. Examples include
   * `Const<N>` to compatible `Dynamic<A,L,H>`, `Const<N>` to `Any`, and
   * constrained `Dynamic` to less constrained `Dynamic`. More strict
   * conversions remain explicit through `as()`.
   */
  template <typename TShape2, typename TStrides2,
      std::enable_if_t<
          !(std::is_same_v<TShape, TShape2> && std::is_same_v<TStrides, TStrides2>) &&
          details::IsMoreLenientMeta<TShape, TShape2>::value &&
          details::IsMoreLenientMeta<TStrides, TStrides2>::value,
      bool> = true>
  constexpr operator Layout<TShape2, TStrides2>() const {
    return as<TShape2, TStrides2>();
  }

private:
  TShape _shape;
  TStrides _stride;
}; // struct Layout

namespace details {

// ---- Repeat type helper ----

template <int N, template <typename...> class Meta, typename V, typename... Acc>
struct RepeatImpl {
  using type = typename RepeatImpl<N - 1, Meta, V, V, Acc...>::type;
};

template <template <typename...> class Meta, typename V, typename... Acc>
struct RepeatImpl<0, Meta, V, Acc...> {
  using type = Meta<Acc...>;
};

template <int N, template <typename...> class Meta, typename V>
using repeat_t = typename RepeatImpl<N, Meta, V>::type;

// ---- Inferred contiguous Strides from Shape ----

/// Compute the product type of a pack of Value types (right-to-left accumulation).
template <typename... Ts> struct Product;
template <> struct Product<> { using type = Const<1>; };
template <typename T> struct Product<T> { using type = T; };
template <typename T0, typename T1, typename... Ts>
struct Product<T0, T1, Ts...> {
  using type = decltype(std::declval<T0>() * std::declval<typename Product<T1, Ts...>::type()>());
};

/// Extract types from index I to end of pack.
template <int I, typename... Ts> struct SuffixOf;
template <int I, typename T0, typename... Ts>
struct SuffixOf<I, T0, Ts...> {
  using type = std::conditional_t<(I == 0), std::tuple<T0, Ts...>,
      typename SuffixOf<I - 1, Ts...>::type>;
};
template <int I> struct SuffixOf<I> { using type = std::tuple<>; };

/// Compute the product of types in a std::tuple.
template <typename Tuple> struct TupleProduct;
template <typename... Ts>
struct TupleProduct<std::tuple<Ts...>> : Product<Ts...> {};

/// Compute the stride type for dimension I: product of sizes[I+1..N-1].
template <int I, typename TShape> struct StrideTypeForDim;
template <int I, typename... Ss>
struct StrideTypeForDim<I, Shape<Ss...>> {
  using suffix = typename SuffixOf<I + 1, Ss...>::type;
  using type = typename TupleProduct<suffix>::type;
};

/// Compute the full Strides<...> type from a Shape<...>.
template <typename TShape> struct InferredStrides;
template <typename... Ss>
struct InferredStrides<Shape<Ss...>> {
  template <size_t... Idx>
  static auto deduce(std::index_sequence<Idx...>)
      -> Strides<typename StrideTypeForDim<Idx, Shape<Ss...>>::type...>;
  using type = decltype(deduce(std::make_index_sequence<sizeof...(Ss)>{}));
};

} // namespace details

// ======================== make_layout ========================

/**
 * @brief Create a Layout, forwarding the shape and strides arguments.
 */
template <typename TShape, typename TStrides>
constexpr auto make_layout(TShape&& shape, TStrides&& stride) -> Layout<std::remove_cvref_t<TShape>, std::remove_cvref_t<TStrides>> {
  return {std::forward<TShape>(shape), std::forward<TStrides>(stride)};
}

/**
 * @brief Create a Layout from a typed Shape only; Strides are inferred as
 *        row-major contiguous, preserving compile-time type constraints.
 *
 * @code
 * auto s = make_shape(cint<2>, Any{5}, cint<4>, cint<3>);
 * auto L = make_layout(s);
 * // L.strides() has type Strides<Dynamic<4>, Const<12>, Const<3>, Const<1>>
 * @endcode
 */
template <typename TShape>
constexpr auto make_layout(TShape&& shape) -> Layout<
    std::remove_cvref_t<TShape>,
    typename details::InferredStrides<std::remove_cvref_t<TShape>>::type
> {
  using S = std::remove_cvref_t<TShape>;
  static_assert(S::Ndim > 0, "Shape must have at least 1 dimension");
  constexpr int N = S::Ndim;

  std::array<nint_t, N> stride_vals{};
  stride_vals[N - 1] = 1;
  for (int i = N - 2; i >= 0; --i)
    stride_vals[i] = stride_vals[i + 1] * nint_t(shape[i + 1]);

  using St = typename details::InferredStrides<S>::type;
  return [&] <size_t... Idx>(std::index_sequence<Idx...>) {
    return Layout<S, St>(
        std::forward<TShape>(shape),
        St{stride_vals[Idx]...}
    );
  }(std::make_index_sequence<N>{});
}

/**
 * @brief Create a Layout of rank Ndim from initializer_lists for shape and
 *        strides. All dimensions are treated as `Any`.
 */
template <int Ndim>
constexpr auto make_layout(
    std::initializer_list<nint_t> shape_vals,
    std::initializer_list<nint_t> stride_vals
) {
  static_assert(Ndim > 0, "Ndim must be positive");
  VECOPS_ASSERT(shape_vals.size() == Ndim, "shape_vals.size() != Ndim");
  VECOPS_ASSERT(stride_vals.size() == Ndim, "stride_vals.size() != Ndim");

  using S = details::repeat_t<Ndim, Shape, Any>;
  using St = details::repeat_t<Ndim, Strides, Any>;

  return [&] <size_t... Idx>(std::index_sequence<Idx...>) {
    return Layout<S, St>{
        S{Any{*(shape_vals.begin() + (nint_t)Idx)}...},
        St{Any{*(stride_vals.begin() + (nint_t)Idx)}...}
    };
  }(std::make_index_sequence<Ndim>{});
}

/**
 * @brief Create a Layout of rank Ndim from a shape initializer_list only.
 *        Strides are auto-computed as row-major contiguous, all Any.
 */
template <int Ndim>
constexpr auto make_layout(
    std::initializer_list<nint_t> shape_vals
) {
  static_assert(Ndim > 0, "Ndim must be positive");
  VECOPS_ASSERT(shape_vals.size() == Ndim, "shape_vals.size() != Ndim");

  using S = details::repeat_t<Ndim, Shape, Any>;
  using St = details::repeat_t<Ndim, Strides, Any>;

  std::array<nint_t, Ndim> shapes{};
  std::copy(shape_vals.begin(), shape_vals.end(), shapes.begin());

  std::array<nint_t, Ndim> stride_vals{};
  stride_vals[Ndim - 1] = 1;
  for (int i = Ndim - 2; i >= 0; --i)
    stride_vals[i] = stride_vals[i + 1] * shapes[i + 1];

  return [&] <size_t... Idx>(std::index_sequence<Idx...>) {
    return Layout<S, St>{
        S{Any{shapes[Idx]}...},
        St{Any{stride_vals[Idx]}...}
    };
  }(std::make_index_sequence<Ndim>{});
}

namespace details {

template <typename T>
struct IsLayout : std::false_type {};
template <typename TShape, typename TStrides>
struct IsLayout<Layout<TShape, TStrides>> : std::true_type {};

} // namespace details

/// Type trait: `true` if T is a Layout.
template <typename T>
static constexpr bool is_layout = details::IsLayout<T>::value;

/**
 * @brief Get the size (shape value) of dimension I from a Layout.
 *
 * @tparam I       Dimension index.
 * @tparam TLayout Layout type.
 * @param  layout  The layout.
 * @return The size of dimension I (always non-negative).
 */
template <int I, typename TLayout, std::enable_if_t<is_layout<TLayout>, bool> = true>
constexpr nint_t size(const TLayout& layout) {
  return get<I>(layout.shape());
}

/**
 * @brief Get the stride value of dimension I from a Layout.
 */
template <int I, typename TLayout, std::enable_if_t<is_layout<TLayout>, bool> = true>
constexpr nint_t stride(const TLayout& layout) {
  return get<I>(layout.strides());
}

/**
 * @brief Compute the linear storage offset for a full coordinate tuple.
 *
 * Bounds are checked with VECOPS_ASSERT, so release builds do not pay for the
 * validation when assertions are disabled.
 */
template <typename TLayout, std::enable_if_t<is_layout<std::remove_cvref_t<TLayout>>, bool> = true>
constexpr nint_t offset_at(
    const TLayout& layout,
    const std::array<nint_t, std::remove_cvref_t<TLayout>::Ndim>& coords) {
  using LayoutT = std::remove_cvref_t<TLayout>;
  nint_t offset = 0;
  VECOPS_UNROLL
  for (int d = 0; d < LayoutT::Ndim; ++d) {
    VECOPS_ASSERT(0 <= coords[d] && coords[d] < layout.shape()[d], "index out of range");
    offset += coords[d] * layout.strides()[d];
  }
  return offset;
}

/**
 * @brief Compute the linear storage offset from one integer coordinate per
 *        layout dimension.
 */
template <
    typename TLayout,
    typename... Is,
    std::enable_if_t<is_layout<std::remove_cvref_t<TLayout>>, bool> = true>
constexpr nint_t offset_at(const TLayout& layout, Is... is) {
  using LayoutT = std::remove_cvref_t<TLayout>;
  static_assert(sizeof...(Is) == LayoutT::Ndim, "coordinate count must match layout rank");
  static_assert((std::is_integral_v<std::decay_t<Is>> && ...), "coordinates must be integers");
  return offset_at(layout, std::array<nint_t, LayoutT::Ndim>{static_cast<nint_t>(is)...});
}

/**
 * @brief Remove dimension I from an ArrayMeta (Shape or Strides).
 *
 * Returns a new ArrayMeta of rank `Ndim-1`.
 *
 * @note This is a compile-time O(N) type transformation — the return
 *       type carries the modified type parameter pack.
 */
template <int I, typename TMeta, std::enable_if_t<is_array_meta<TMeta>, bool> = true>
constexpr auto remove(const TMeta& m) { return details::remove_dim<I>(m); }

/**
 * @brief Remove dimension I from a Layout (both shape and strides).
 */
template <int I, typename TLayout, std::enable_if_t<is_layout<TLayout>, bool> = true>
constexpr auto remove(const TLayout& m) {
  return make_layout(remove<I>(m.shape()), remove<I>(m.strides()));
}

/**
 * @brief Set dimension I in an ArrayMeta to a new value.
 *
 * @tparam I     Dimension index.
 * @tparam TMeta ArrayMeta type.
 * @tparam Inew  New Value type for this dimension.
 * @param  m     The metadata container.
 * @param  v     The new value.
 * @return A new ArrayMeta with dimension I replaced.
 */
template <int I, typename TMeta, typename Inew, std::enable_if_t<is_array_meta<TMeta>, bool> = true>
constexpr auto set(const TMeta& m, Inew v) { return details::set_dim<I>(m, v); }

/**
 * @brief Set dimension I in a Layout to new shape and stride values.
 */
template <int I, typename TLayout, typename IShapeNew, typename IStridesNew, std::enable_if_t<is_layout<TLayout>, bool> = true>
constexpr auto set(const TLayout& m, IShapeNew v_size, IStridesNew v_stride) {
  return make_layout(set<I>(m.shape(), v_size), set<I>(m.strides(), v_stride));
}

/**
 * @brief Insert a new dimension before position I in an ArrayMeta.
 *
 * If `I == Ndim`, the new dimension is appended after the last dimension.
 *
 * @tparam I     Insertion position (0 <= I <= Ndim).
 * @tparam TMeta ArrayMeta type.
 * @tparam Inew  Value type for the new dimension.
 * @param  m     The metadata container.
 * @param  v     The value for the new dimension.
 * @return A new ArrayMeta of rank `Ndim+1`.
 */
template <int I, typename TMeta, typename Inew, std::enable_if_t<is_array_meta<TMeta>, bool> = true>
constexpr auto insert(const TMeta& m, Inew v) { return details::insert_dim<I>(m, v); }

/**
 * @brief Insert a new dimension before position I in a Layout
 *        (both shape and strides).
 */
template <int I, typename TLayout, typename IShapeNew, typename IStridesNew, std::enable_if_t<is_layout<TLayout>, bool> = true>
constexpr auto insert(const TLayout& m, IShapeNew v_size, IStridesNew v_stride) {
  return make_layout(insert<I>(m.shape(), v_size), insert<I>(m.strides(), v_stride));
}

// ======================== ArrayMeta Swap Dim ========================

namespace details {

/**
 * @brief Compile-time dimension swapping for an ArrayMeta type.
 *
 * Swaps dimensions I and J in the type parameter pack, producing a new
 * type with the dimensions exchanged at compile time.
 *
 * @tparam Meta  The target container template.
 * @tparam I     First dimension index to swap.
 * @tparam J     Second dimension index to swap.
 * @tparam Is    Original type parameter pack.
 */
template <
    template <typename... xIs> typename Meta,
    int I,
    int J,
    typename... Is
>
struct ArrayMetaSwapDim {
private:
  static constexpr int N = sizeof...(Is);
  static_assert(0 <= I && I < N, "I out of range");
  static_assert(0 <= J && J < N, "J out of range");

  template <int Idx>
  struct select {
    using type = std::conditional_t<
        Idx == I,
        std::tuple_element_t<J, std::tuple<Is...>>,
        std::conditional_t<
            Idx == J,
            std::tuple_element_t<I, std::tuple<Is...>>,
            std::tuple_element_t<Idx, std::tuple<Is...>>
        >
    >;
  };

  template <int... Idx>
  static constexpr Meta<typename select<Idx>::type...> _make_type(std::integer_sequence<int, Idx...>);

public:
  using Type = decltype(_make_type(std::make_integer_sequence<int, N>{}));

  static constexpr Type transform(const Meta<Is...>& m) {
    std::array<nint_t, N> arr = m._stor.to_array();
    std::swap(arr[I], arr[J]);
    return [&] <size_t... Idx>(std::index_sequence<Idx...>) {
      return Type{arr[Idx]...};
    }(std::make_index_sequence<N>{});
  }
};

template <int I, int J, template <typename... xIs> typename TMeta, typename... Is>
constexpr auto swap_dim(const TMeta<Is...>& m) {
  return ArrayMetaSwapDim<TMeta, I, J, Is...>::transform(m);
}

} // namespace details

/**
 * @brief Compile-time transpose: swap dimensions I and J in a Layout, preserving
 *        Const type information.
 *
 * Unlike the runtime `transpose(layout, i, j)`, this overload preserves the
 * compile-time Const/Dynamic types for each dimension, producing the exact
 * swapped type at compile time.
 *
 * @code
 * auto s = make_shape(cint<2>, cint<3>, cint<4>);
 * auto st = make_strides(cint<12>, cint<4>, cint<1>);
 * auto L = make_layout(s, st);
 * auto Lt = transpose<0, 1>(L);
 * // Lt.shape() == Shape<Const<3>, Const<2>, Const<4>>
 * @endcode
 *
 * @tparam I       First dimension index to swap.
 * @tparam J       Second dimension index to swap.
 * @tparam TLayout Layout type.
 * @param  layout  The layout to transpose.
 * @return A new Layout with dimensions I and J exchanged, preserving type info.
 */
template <int I, int J, typename TLayout, std::enable_if_t<is_layout<TLayout>, bool> = true>
constexpr auto transpose(const TLayout& layout) {
  static_assert(0 <= I && I < TLayout::Ndim, "I out of range");
  static_assert(0 <= J && J < TLayout::Ndim, "J out of range");
  return make_layout(
      details::swap_dim<I, J>(layout.shape()),
      details::swap_dim<I, J>(layout.strides())
  );
}

/**
 * @brief Runtime transpose: swap dimensions i and j in a Layout.
 *
 * The return type degrades to all-`Any` (all dimensions become `Any`)
 * because the swap targets are runtime values.
 *
 * @note Prefer the compile-time overload `transpose<I, J>(layout)` when
 *       the swap indices are known at compile time.
 *
 * @tparam TLayout Layout type.
 * @param  layout  The layout to transpose.
 * @param  i       First dimension index (runtime).
 * @param  j       Second dimension index (runtime).
 * @return A new Layout with all-Any Shape and Strides.
 */
template <typename TLayout, std::enable_if_t<is_layout<TLayout>, bool> = true>
constexpr auto transpose(const TLayout& layout, int i, int j) {
  constexpr int ndim = TLayout::Ndim;
  VECOPS_ASSERT(0 <= i && i < ndim, "i(%d) out of range [0, %d)", i, ndim);
  VECOPS_ASSERT(0 <= j && j < ndim, "j(%d) out of range [0, %d)", j, ndim);

  auto shape_arr = layout.shape()._stor.to_array();
  auto stride_arr = layout.strides()._stor.to_array();
  std::swap(shape_arr[i], shape_arr[j]);
  std::swap(stride_arr[i], stride_arr[j]);

  return [&] <size_t... Idx>(std::index_sequence<Idx...>) {
    return make_layout(
        make_shape(Any{shape_arr[Idx]}...),
        make_strides(Any{stride_arr[Idx]}...)
    );
  }(std::make_index_sequence<ndim>{});
}

// ======================== Continuity Traits ========================

namespace details {

/**
 * @brief Compile-time check: are the last N dimensions of TLayout contiguous?
 *
 * A layout is **row-major contiguous** when each stride equals the product
 * of all subsequent dimensions' sizes:
 *   stride[D] == shape[D+1] * stride[D+1]
 * and the last stride == 1.
 *
 * This trait performs the check at compile time using only `Const` dimension
 * information. If any involved dimension is non-Const, the check returns
 * `false` (conservative).
 *
 * @tparam TLayout  The Layout type.
 * @tparam N        How many trailing dimensions to check.
 */
template <typename TLayout, int N>
struct is_ct_last_contiguous_impl;

template <typename... Ss, typename... Ts, int N>
struct is_ct_last_contiguous_impl<Layout<Shape<Ss...>, Strides<Ts...>>, N> {
private:
  static constexpr int ndim = sizeof...(Ss);

  template <int D>
  static constexpr bool _check_one() {
    if constexpr (D < 0 || D > ndim) {
      return false;
    } else if constexpr (D >= ndim) {
      return true;
    } else if constexpr (D == ndim - 1) {
      using St = std::tuple_element_t<D, std::tuple<Ts...>>;
      if constexpr (!St::is_const) {
        return false;
      } else {
        return St::value == 1;
      }
    } else {
      using Sd = std::tuple_element_t<D, std::tuple<Ts...>>;
      using Sd1 = std::tuple_element_t<D + 1, std::tuple<Ts...>>;
      using Zd1 = std::tuple_element_t<D + 1, std::tuple<Ss...>>;
      if constexpr (!Sd::is_const) {
        return false;
      } else if constexpr (!Sd1::is_const) {
        return false;
      } else if constexpr (!Zd1::is_const) {
        return false;
      } else {
        return Sd::value == Zd1::value * Sd1::value && _check_one<D + 1>();
      }
    }
  }

public:
  static constexpr bool value = (N >= 0) && (N <= ndim) && _check_one<ndim - N>();
};

} // namespace details

/**
 * @brief Compile-time check: are the last N dimensions of TLayout contiguous?
 *
 * Evaluates at compile time. When all involved dimensions are `Const`,
 * can resolve to `true` without any runtime check.
 *
 * @tparam TLayout  Layout type.
 * @tparam N        Number of trailing dimensions to check.
 */
template <typename TLayout, int N>
struct is_ct_last_contiguous : details::is_ct_last_contiguous_impl<std::remove_cvref_t<TLayout>, N> {};

/**
 * @brief Compile-time check: are all dimensions of TLayout contiguous?
 *
 * Equivalent to `is_ct_last_contiguous<TLayout, TLayout::Ndim>`.
 */
template <typename TLayout>
struct is_ct_contiguous : is_ct_last_contiguous<std::remove_cvref_t<TLayout>, std::remove_cvref_t<TLayout>::Ndim> {};

/**
 * @brief Runtime check: are the last N dimensions of `layout` contiguous?
 *
 * If the compile-time check (`is_ct_last_contiguous`) already returns `true`,
 * this function returns `true` at compile time with no runtime cost.
 *
 * @tparam N       Number of trailing dimensions to check.
 * @tparam TLayout Layout type.
 * @param  layout  The layout.
 * @return `true` if the last N dimensions are row-major contiguous.
 */
template <int N, typename TLayout, std::enable_if_t<is_layout<TLayout>, bool> = true>
inline bool is_last_contiguous(const TLayout& layout) {
  if constexpr (is_ct_last_contiguous<TLayout, N>::value) {
    return true;
  }
  if constexpr (N <= 0) {
    return true;
  }
  int ndim = layout.ndim();
  if (N > ndim) return false;
  nint_t expected = 1;
  for (int d = ndim - 1; d >= ndim - N; --d) {
    if (layout.strides()[d] != expected) return false;
    expected *= layout.shape()[d];
  }
  return true;
}

/**
 * @brief Runtime check: are all dimensions of `layout` contiguous?
 *
 * @tparam TLayout Layout type.
 * @param  layout  The layout.
 * @return `true` if the layout is fully row-major contiguous.
 */
template <typename TLayout, std::enable_if_t<is_layout<TLayout>, bool> = true>
inline bool is_contiguous(const TLayout& layout) {
  return is_last_contiguous<TLayout::Ndim>(layout);
}


// ============ I/O Support ============

/**
 * @brief Stream output for ArrayMeta (Shape / Strides).
 *
 * Prints as compact tuple form `(v0, v1, ...)`.
 *
 * Value decoration rules:
 *   - `Const<N>`             → `N!`
 *   - `Any{v}` (=Dynamic<1>) → `v`
 *   - `Dynamic<A,_,_>{v}`    → `v@A`  (when A > 1)
 *   - `Dynamic<A,Lo,Hi>{v}`  → `v@A[Lo,Hi]` (when bounds are active)
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
 * @brief Stream output for Layout.
 *
 * Prints as `Layout(s=(...), st=(...))`.
 */
template <typename TShape, typename TStrides>
std::ostream& operator<<(std::ostream& os, const Layout<TShape, TStrides>& l) {
  os << "Layout(s=" << l.shape() << ", st=" << l.strides() << ')';
  return os;
}


} // namespace vecops::gemm

#endif //VECOPS_LAYOUT_H
