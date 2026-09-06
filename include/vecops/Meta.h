//
// Created by renyz on 2026/6/2.
//

#ifndef VECOPS_META_H
#define VECOPS_META_H

#include <algorithm>
#include <array>
#include <concepts>
#include <numeric>
#include <ostream>
#include <type_traits>
#include <tuple>
#include <utility>

#include "vecops/CoreTypes.h"
#include "vecops/Assertion.h"
#include "vecops/util/Math.h"
#include "vecops/util/TypeTraits.h"

/**
 * @file Meta.h
 * @brief Typed integer metadata with compile-time constraint propagation.
 *
 * This header defines the foundational types for describing multi-dimensional
 * sizes, strides, loop bounds, and vector addressing parameters. The core
 * concept is the `Value` hierarchy: an integer is represented by a type that
 * records whether it is constant or runtime and, for runtime values, what
 * alignment and bounds the compiler may assume. Arithmetic preserves every
 * constraint that can be proven from the operand types.
 *
 * This header deliberately has no dependency on `tensor::Layout`. Tensor and
 * vector layers may both use the metadata types without creating a dependency
 * cycle. `Shape`, `Strides`, and `Layout` are defined in
 * `vecops/tensor/Layout.h`.
 *
 * ## Key components
 *
 * | Component      | Purpose                                                   |
 * |----------------|-----------------------------------------------------------|
 * | Value hierarchy| Compile-time/run-time typed integers with constraints     |
 * | ValueInput     | Accepts Value subtypes and fixed-width integer inputs      |
 * | to_value(_t)   | Normalizes raw integer inputs to unconstrained `Any`        |
 * | PackedStorage  | Stores only runtime members of a heterogeneous Value pack  |
 *
 * ## Usage overview
 *
 * @code
 * #include "vecops/Meta.h"
 * using namespace vecops::meta;
 *
 * auto four = cint<4>;                          // Const<4>
 * auto n = dyn<8, 0, 1024>(128);               // Dynamic<8, 0, 1024>
 * auto bytes = n * cint<4>;                    // Dynamic<32, 0, 4096>
 * static_assert(decltype(bytes)::aligns(32));
 *
 * // Raw integers are accepted, but are deliberately promoted to Any.
 * auto unknown = n + nint_t{3};                // Any-like result
 * @endcode
 *
 * ## Constraint vocabulary
 *
 * - `Const<N>` means the value is exactly `N` and occupies no runtime storage
 *   inside `PackedStorage`.
 * - `Dynamic<A, Lo, Hi>` stores a runtime value that is a multiple of `A` and
 *   lies in the inclusive range `[Lo, Hi]`. Omitted bounds are unbounded.
 * - `Any` is `Dynamic<1>`: a runtime integer with no useful constraints.
 * - `any` (lowercase) is not a value. It is a wildcard used only by metadata
 *   matching traits.
 *
 * A type guarantee and an instance observation are intentionally distinct.
 * `T::aligns(k)` asks whether every value described by `T` is divisible by
 * `k`; `value.is_aligned(k)` inspects one runtime instance.
 *
 * ## Pitfalls
 *
 * - `Const<N>` can be default-constructed as `N`; constructing it from a
 *   different runtime value triggers `VECOPS_ASSERT`.
 * - `Dynamic` construction asserts alignment/bounds constraints at runtime.
 * - Arithmetic with raw `nint_t` promotes to `Any`, losing all constraint info.
 * - `aligns(v)` is conservative: returning false means "not guaranteed", not "impossible".
 * - Alignment describes integer divisibility, not pointer alignment and not a
 *   byte unit. Its unit is whatever the surrounding abstraction assigns.
 * - Division and remainder propagation are conservative; inspect the result
 *   type before relying on a bound or alignment in a hot-path specialization.
 * - Runtime validation uses `VECOPS_ASSERT`; callers must not rely on it as
 *   input sanitization in builds where assertions are disabled.
 */

namespace vecops::meta {

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
 * - `operator nint_t()` – conversion to raw integer
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
  /// Conservative bounds used by generic range queries for custom values.
  static constexpr bool has_lower = false;
  static constexpr bool has_upper = false;
  static constexpr nint_t lo = kLoInf;
  static constexpr nint_t hi = kHiInf;

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
  constexpr operator nint_t() const { return -100; }
}; // struct Value

template <typename T>
concept ValueType = std::derived_from<std::remove_cvref_t<T>, Value>;

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
  VECOPS_ALWAYS_INLINE constexpr explicit Const(nint_t v = N) {
    VECOPS_ASSERT(v == N, "%td != %td", v, N);
  }

  VECOPS_ALWAYS_INLINE constexpr bool is_aligned(nint_t v) {
    return N % v == 0;
  }

  VECOPS_ALWAYS_INLINE constexpr operator nint_t() const {
    return N;
  }
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
  VECOPS_ALWAYS_INLINE constexpr explicit Dynamic(nint_t v) : value(v) {
    VECOPS_ASSERT(conforms(v),
                  "value %td fails Dynamic<A=%td, Lo=%td, Hi=%td> constraints", v, Alignment, Lo, Hi);
  }

  /**
   * Check whether **this specific instance's** runtime value is divisible by `v`.
   */
  VECOPS_ALWAYS_INLINE constexpr bool is_aligned(nint_t v) const {
    return value % v == 0;
  }

  VECOPS_ALWAYS_INLINE constexpr operator nint_t() const {
    // Spell the constraint out here: some Clang versions conservatively
    // treat even this constexpr helper call as potentially side-effecting and
    // consequently discard __builtin_assume(conforms(value)).
    VECOPS_ASSUME(
        (value & (Alignment - 1)) == 0 &&
        (!has_lower || value >= Lo) &&
        (!has_upper || value <= Hi));
    return value;
  }

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

/** A non-nint_t integer accepted by Value arithmetic through Any. */
template <typename T>
concept OtherInteger =
    std::integral<std::remove_cvref_t<T>> &&
    !std::same_as<std::remove_cvref_t<T>, bool> &&
    !std::same_as<std::remove_cvref_t<T>, nint_t>;

/**
 * @brief Wildcard type for compile-time metadata pattern matching.
 *
 * `any` is not a runtime metadata value. It is a type-level pattern used by
 * traits such as `is_lenient_v<Meta, Patterns...>` to mean "accept any Value
 * type in this position". Use `Any` when you need an unconstrained runtime
 * `Value`; use `any` only as a template matching wildcard.
 *
 * @code
 * // A consumer such as tensor::Layout may accept `any` for one metadata slot
 * // while constraining another slot to Const<2>.
 * static_assert(!std::is_base_of_v<Value, any>);
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
// information (alignment, bounds) at compile time. Raw integral operands
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
template <typename T>
  requires (std::derived_from<T, Value> && !is_int_v<T>)
constexpr auto operator+(T lhs, nint_t rhs) {
  return lhs + Any{rhs};
}

/// nint_t + Value → Any{nint_t} + Value
template <typename T>
  requires (std::derived_from<T, Value> && !is_int_v<T>)
constexpr auto operator+(nint_t lhs, T rhs) {
  return Any{lhs} + rhs;
}

template <ValueType T, OtherInteger I>
constexpr auto operator+(T lhs, I rhs) {
  return lhs + static_cast<nint_t>(rhs);
}

template <OtherInteger I, ValueType T>
constexpr auto operator+(I lhs, T rhs) {
  return static_cast<nint_t>(lhs) + rhs;
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
template <typename T>
  requires (std::derived_from<T, Value> && !is_int_v<T>)
constexpr auto operator-(T lhs, nint_t rhs) {
  return lhs - Any{rhs};
}

/// nint_t - Value → Any{nint_t} - Value
template <typename T>
  requires (std::derived_from<T, Value> && !is_int_v<T>)
constexpr auto operator-(nint_t lhs, T rhs) {
  return Any{lhs} - rhs;
}

template <ValueType T, OtherInteger I>
constexpr auto operator-(T lhs, I rhs) {
  return lhs - static_cast<nint_t>(rhs);
}

template <OtherInteger I, ValueType T>
constexpr auto operator-(I lhs, T rhs) {
  return static_cast<nint_t>(lhs) - rhs;
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
template <typename T>
  requires (std::derived_from<T, Value> && !is_int_v<T>)
constexpr auto operator*(T lhs, nint_t rhs) {
  return lhs * Any{rhs};
}

/// nint_t * Value → Any{nint_t} * Value
template <typename T>
  requires (std::derived_from<T, Value> && !is_int_v<T>)
constexpr auto operator*(nint_t lhs, T rhs) {
  return Any{lhs} * rhs;
}

template <ValueType T, OtherInteger I>
constexpr auto operator*(T lhs, I rhs) {
  return lhs * static_cast<nint_t>(rhs);
}

template <OtherInteger I, ValueType T>
constexpr auto operator*(I lhs, T rhs) {
  return static_cast<nint_t>(lhs) * rhs;
}

// ---- Division ----

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
    constexpr nint_t k_min = ceil_div(L, A);
    constexpr nint_t k_max = floor_div(H, A);
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
    constexpr nint_t k_min = ceil_div(L, A);
    constexpr nint_t k_max = floor_div(H, A);
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
template <typename T>
  requires (std::derived_from<T, Value> && !is_int_v<T>)
constexpr auto operator/(T lhs, nint_t rhs) {
  return lhs / Any{rhs};
}

/// nint_t / Value → Any{nint_t} / Value
template <typename T>
  requires (std::derived_from<T, Value> && !is_int_v<T>)
constexpr auto operator/(nint_t lhs, T rhs) {
  return Any{lhs} / rhs;
}

template <ValueType T, OtherInteger I>
constexpr auto operator/(T lhs, I rhs) {
  return lhs / static_cast<nint_t>(rhs);
}

template <OtherInteger I, ValueType T>
constexpr auto operator/(I lhs, T rhs) {
  return static_cast<nint_t>(lhs) / rhs;
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
    constexpr nint_t k_min = ceil_div(L, A);
    constexpr nint_t k_max = floor_div(H, A);
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
template <typename T>
  requires (std::derived_from<T, Value> && !is_int_v<T>)
constexpr auto operator%(T lhs, nint_t rhs) {
  return lhs % Any{rhs};
}

/// nint_t % Value → Any{nint_t} % Value
template <typename T>
  requires (std::derived_from<T, Value> && !is_int_v<T>)
constexpr auto operator%(nint_t lhs, T rhs) {
  return Any{lhs} % rhs;
}

template <ValueType T, OtherInteger I>
constexpr auto operator%(T lhs, I rhs) {
  return lhs % static_cast<nint_t>(rhs);
}

template <OtherInteger I, ValueType T>
constexpr auto operator%(I lhs, T rhs) {
  return static_cast<nint_t>(lhs) % rhs;
}

namespace details {

/**
 * @brief Converts integer types to their Value wrapper equivalent.
 *
 * Non-integer types (already Value subclasses) pass through unchanged.
 * Integer types (int8_t, uint32_t, etc.) are promoted to `Any`.
 *
 * Used by `to_value_t<T>` to normalize user-provided shape/stride parameters
 * so that bare integers are automatically wrapped as `Any{v}`.
 */
template <typename T, typename = void/*SFINAE*/>
struct ValuePromote { using type = T; };
template <typename T>
  requires is_int_v<T>
struct ValuePromote<T, void> {
  using type = Any;
};

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
 * PackedStorage<Const<2>, Dynamic<4>, Const<3>> stor(
 *     Const<2>{}, Dynamic<4>{128}, Const<3>{});
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

  struct StorageData {
    ::vecops::details::InlineArray<int, sizeof...(Is)> offsets{};
    int count = 0;
  };

  /// Fold over the type pack directly to compute offsets at compile time.
  /// Uses Is::is_runtime (class-level static) rather than is_runtime[i]
  /// which avoids clang's "array without known bound" constexpr limitation.
  template <size_t... Js>
  static constexpr auto _compute_offsets(std::index_sequence<Js...>) {
    ::vecops::details::InlineArray<int, sizeof...(Is)> arr{};
    int off = 0;
    ((arr[Js] = off, off += int(Is::is_runtime)), ...);
    return StorageData{arr, off};
  }
  static constexpr auto stor_data = _compute_offsets(std::index_sequence_for<Is...>{});
public:
  /**
   * Offset mapping: `offsets[i]` gives the position in the compressed
   * `values` array for logical dimension i (meaningless for Const entries).
   */
  static constexpr auto offsets = stor_data.offsets;
  /// Number of actually stored runtime values.
  static constexpr int num_stor = stor_data.count;

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
  [[nodiscard]] VECOPS_ALWAYS_INLINE constexpr nint_t get() const {
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
  [[nodiscard]] VECOPS_ALWAYS_INLINE nint_t operator[](int i) const {
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
  [[nodiscard]] constexpr const auto& to_packed_array() const {
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
  ::vecops::details::InlineArray<nint_t, num_stor> values;
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
 * using T = to_value_t<int32_t>;   // T = Any
 * using U = to_value_t<Const<4>>;  // U = Const<4>
 * @endcode
 */
template <typename T>
using to_value_t = details::ValuePromote<std::remove_cvref_t<T>>::type;

/**
 * @brief Input accepted by APIs that normalize arguments to a Value.
 *
 * A ValueInput is either an existing `Value` subtype or one of vecops' fixed
 * width integer types. Integer inputs normalize to `Any` (`Dynamic<1>`), while
 * existing Value types retain their compile-time alignment and bound metadata.
 */
template <typename T>
concept ValueInput = ValueType<to_value_t<T>>;

/**
 * @brief Normalize an integer or existing Value to its canonical Value type.
 *
 * @code
 * auto runtime = to_value(int32_t{17}); // Any{17}
 * auto fixed = to_value(cint<4>);       // Const<4>{}
 * @endcode
 */
template <ValueInput T>
VECOPS_ALWAYS_INLINE constexpr to_value_t<T> to_value(T&& value) {
  using V = to_value_t<T>;
  return V{static_cast<nint_t>(value)};
}

/** Compare two Value objects without discarding their metadata beforehand. */
template <ValueType Lhs, ValueType Rhs>
VECOPS_ALWAYS_INLINE constexpr bool operator==(Lhs lhs, Rhs rhs) {
  return static_cast<nint_t>(lhs) == static_cast<nint_t>(rhs);
}

template <ValueType Lhs, ValueType Rhs>
VECOPS_ALWAYS_INLINE constexpr bool operator!=(Lhs lhs, Rhs rhs) {
  return !(lhs == rhs);
}

template <ValueType Lhs, ValueType Rhs>
VECOPS_ALWAYS_INLINE constexpr bool operator<(Lhs lhs, Rhs rhs) {
  return static_cast<nint_t>(lhs) < static_cast<nint_t>(rhs);
}

template <ValueType Lhs, ValueType Rhs>
VECOPS_ALWAYS_INLINE constexpr bool operator<=(Lhs lhs, Rhs rhs) {
  return !(rhs < lhs);
}

template <ValueType Lhs, ValueType Rhs>
VECOPS_ALWAYS_INLINE constexpr bool operator>(Lhs lhs, Rhs rhs) {
  return rhs < lhs;
}

template <ValueType Lhs, ValueType Rhs>
VECOPS_ALWAYS_INLINE constexpr bool operator>=(Lhs lhs, Rhs rhs) {
  return !(lhs < rhs);
}

template <ValueType Value>
VECOPS_ALWAYS_INLINE constexpr bool operator==(Value lhs, nint_t rhs) {
  return static_cast<nint_t>(lhs) == rhs;
}

template <ValueType Value>
VECOPS_ALWAYS_INLINE constexpr bool operator==(nint_t lhs, Value rhs) {
  return rhs == lhs;
}

template <ValueType Value>
VECOPS_ALWAYS_INLINE constexpr bool operator!=(Value lhs, nint_t rhs) {
  return !(lhs == rhs);
}

template <ValueType Value>
VECOPS_ALWAYS_INLINE constexpr bool operator!=(nint_t lhs, Value rhs) {
  return !(lhs == rhs);
}

template <ValueType Value>
VECOPS_ALWAYS_INLINE constexpr bool operator<(Value lhs, nint_t rhs) {
  return static_cast<nint_t>(lhs) < rhs;
}

template <ValueType Value>
VECOPS_ALWAYS_INLINE constexpr bool operator<(nint_t lhs, Value rhs) {
  return lhs < static_cast<nint_t>(rhs);
}

template <ValueType Value>
VECOPS_ALWAYS_INLINE constexpr bool operator<=(Value lhs, nint_t rhs) {
  return !(rhs < lhs);
}

template <ValueType Value>
VECOPS_ALWAYS_INLINE constexpr bool operator<=(nint_t lhs, Value rhs) {
  return !(rhs < lhs);
}

template <ValueType Value>
VECOPS_ALWAYS_INLINE constexpr bool operator>(Value lhs, nint_t rhs) {
  return rhs < lhs;
}

template <ValueType Value>
VECOPS_ALWAYS_INLINE constexpr bool operator>(nint_t lhs, Value rhs) {
  return rhs < lhs;
}

template <ValueType Value>
VECOPS_ALWAYS_INLINE constexpr bool operator>=(Value lhs, nint_t rhs) {
  return !(lhs < rhs);
}

template <ValueType Value>
VECOPS_ALWAYS_INLINE constexpr bool operator>=(nint_t lhs, Value rhs) {
  return !(lhs < rhs);
}

#define VECOPS_META_DEFINE_INTEGER_COMPARISON(Op)                           \
  template <ValueType Value, OtherInteger I>                                \
  VECOPS_ALWAYS_INLINE constexpr bool operator Op(Value lhs, I rhs) {       \
    return lhs Op static_cast<nint_t>(rhs);                                 \
  }                                                                          \
  template <OtherInteger I, ValueType Value>                                \
  VECOPS_ALWAYS_INLINE constexpr bool operator Op(I lhs, Value rhs) {       \
    return static_cast<nint_t>(lhs) Op rhs;                                 \
  }

VECOPS_META_DEFINE_INTEGER_COMPARISON(==)
VECOPS_META_DEFINE_INTEGER_COMPARISON(!=)
VECOPS_META_DEFINE_INTEGER_COMPARISON(<)
VECOPS_META_DEFINE_INTEGER_COMPARISON(<=)
VECOPS_META_DEFINE_INTEGER_COMPARISON(>)
VECOPS_META_DEFINE_INTEGER_COMPARISON(>=)

#undef VECOPS_META_DEFINE_INTEGER_COMPARISON

template <ValueType T>
inline constexpr bool has_lower_bound_v = [] {
  using V = std::remove_cvref_t<T>;
  if constexpr (V::is_const) return true;
  else return V::has_lower;
}();

template <ValueType T>
inline constexpr bool has_upper_bound_v = [] {
  using V = std::remove_cvref_t<T>;
  if constexpr (V::is_const) return true;
  else return V::has_upper;
}();

template <ValueType T>
inline constexpr nint_t lower_bound_v = [] {
  using V = std::remove_cvref_t<T>;
  if constexpr (V::is_const) return V::value;
  else return V::lo;
}();

template <ValueType T>
inline constexpr nint_t upper_bound_v = [] {
  using V = std::remove_cvref_t<T>;
  if constexpr (V::is_const) return V::value;
  else return V::hi;
}();

template <ValueType T>
inline constexpr bool is_bounded_v =
    has_lower_bound_v<T> && has_upper_bound_v<T>;

/** True when a Value type denotes exactly one runtime value. */
template <ValueType T>
inline constexpr bool is_singleton_v =
    is_bounded_v<T> && lower_bound_v<T> == upper_bound_v<T>;

/** The unique value denoted by a singleton Value type. */
template <ValueType T>
inline constexpr nint_t singleton_value_v = lower_bound_v<T>;

template <ValueType T, nint_t Lo, nint_t Hi>
inline constexpr bool range_within_v =
    is_bounded_v<T> && lower_bound_v<T> >= Lo && upper_bound_v<T> <= Hi;

template <ValueType T, nint_t Lo>
inline constexpr bool lower_bound_at_least_v =
    has_lower_bound_v<T> && lower_bound_v<T> >= Lo;

template <ValueType T, nint_t Hi>
inline constexpr bool upper_bound_at_most_v =
    has_upper_bound_v<T> && upper_bound_v<T> <= Hi;

} // namespace vecops::meta

namespace vecops {

// ================== Value-aware overloads of util/Math.h ==================
//
// min/max/clamp and the integer-division family accept meta::Const /
// meta::Dynamic operands and propagate constraints, mirroring the arithmetic
// operators above:
//   Const  op  Const  → Const (folded at compile time)
//   Const  op  Dyn    → tighter bounds, alignment degrades to gcd
//   Dyn    op  Dyn    → merged bounds (min/max) or degraded (division)
//   integer op Value   → Any{integer} op Value, matching the arithmetic
//                        operators: constraints on the raw integer side are
//                       lost.
//
// Alignment notes (A, A1, A2 are positive powers of two, so every gcd below
// is a power of two as required by Dynamic):
//   min/max      : the result is one of the two operands, hence a multiple
//                  of gcd(A1, A2).
//   clamp        : the result is v, Lo, or Hi, hence a multiple of
//                  gcd(gcd(A, Lo), Hi).
//   ceil_div/floor_div by Const<N>: A % N == 0 keeps divisibility (result
//                  alignment A/N), otherwise degrades to 1 — same rule as
//                  operator/.
//   align_up/align_down by Const<N>: A % N == 0 means every runtime value is
//                  already an N-multiple, so the alignment A survives
//                  (align_up is the identity); otherwise gcd(A, N).
// The division family is monotonic for positive divisors, so bounds are
// computed from the endpoints via the scalar util/Math.h primitives.
//
// Bound sentinels participate naturally in min/max comparisons (kLoInf is
// the smallest representable nint_t, kHiInf the largest), so min/max bound
// merging needs no sentinel special case. The division family, in contrast,
// must keep the explicit unbounded branches: kLoInf / N would silently
// destroy the sentinel.

/**
 * @brief Compile-time minimum of two Const values.
 * @code
 * static_assert(std::same_as<decltype(min(cint<3>, cint<7>)), Const<3>>);
 * @endcode
 */
template <nint_t N, nint_t M>
constexpr meta::Const<(N < M) ? N : M> min(meta::Const<N>, meta::Const<M>) {
  return meta::Const<(N < M) ? N : M>();
}

/**
 * @brief min of a Const and a Dynamic, with constraint propagation.
 *
 * If every runtime value of the Dynamic side is >= N (or <= N), the result
 * folds to Const<N> (or passes the Dynamic through unchanged). Otherwise the
 * upper bound tightens to min(H, N) and the alignment degrades to gcd(A, N).
 */
template <nint_t N, nint_t A, nint_t L, nint_t H>
constexpr auto min(meta::Const<N>, meta::Dynamic<A, L, H> rhs) {
  if constexpr (L != meta::kLoInf && N <= L) {
    return meta::Const<N>();
  } else if constexpr (H != meta::kHiInf && H < N) {
    return rhs;
  } else {
    constexpr nint_t g = std::gcd(A, N);
    constexpr nint_t rh = (H < N) ? H : N;
    return meta::Dynamic<g, L, rh>((rhs.value < N) ? rhs.value : N);
  }
}

/// @copydoc min(meta::Const<N>, meta::Dynamic<A,L,H>)
template <nint_t N, nint_t A, nint_t L, nint_t H>
constexpr auto min(meta::Dynamic<A, L, H> lhs, meta::Const<N> rhs) {
  return min(rhs, lhs);
}

/**
 * @brief min of two Dynamics: bounds merge component-wise, alignment degrades
 * to gcd(A1, A2).
 */
template <nint_t A1, nint_t L1, nint_t H1, nint_t A2, nint_t L2, nint_t H2>
constexpr auto min(meta::Dynamic<A1, L1, H1> lhs, meta::Dynamic<A2, L2, H2> rhs) {
  constexpr nint_t lo = (L1 < L2) ? L1 : L2;
  constexpr nint_t hi = (H1 < H2) ? H1 : H2;
  return meta::Dynamic<std::gcd(A1, A2), lo, hi>(
      (rhs.value < lhs.value) ? rhs.value : lhs.value);
}

/// Value min nint_t → Value min Any{nint_t}
template <typename T>
  requires (std::derived_from<T, meta::Value> && !is_int_v<T>)
constexpr auto min(T lhs, nint_t rhs) {
  return min(lhs, meta::Any{rhs});
}

/// nint_t min Value → Any{nint_t} min Value
template <typename T>
  requires (std::derived_from<T, meta::Value> && !is_int_v<T>)
constexpr auto min(nint_t lhs, T rhs) {
  return min(meta::Any{lhs}, rhs);
}

/**
 * @brief Compile-time maximum of two Const values.
 * @code
 * static_assert(std::same_as<decltype(max(cint<3>, cint<7>)), Const<7>>);
 * @endcode
 */
template <nint_t N, nint_t M>
constexpr meta::Const<(N > M) ? N : M> max(meta::Const<N>, meta::Const<M>) {
  return meta::Const<(N > M) ? N : M>();
}

/**
 * @brief max of a Const and a Dynamic, with constraint propagation.
 *
 * Dual of min(Const<N>, Dynamic<A,L,H>): folds to Const<N> when every
 * runtime value is <= N, passes the Dynamic through when every runtime value
 * is >= N, otherwise the lower bound tightens to max(L, N) and the alignment
 * degrades to gcd(A, N).
 */
template <nint_t N, nint_t A, nint_t L, nint_t H>
constexpr auto max(meta::Const<N>, meta::Dynamic<A, L, H> rhs) {
  if constexpr (H != meta::kHiInf && N >= H) {
    return meta::Const<N>();
  } else if constexpr (L != meta::kLoInf && N <= L) {
    return rhs;
  } else {
    constexpr nint_t g = std::gcd(A, N);
    constexpr nint_t rl = (N > L) ? N : L;
    return meta::Dynamic<g, rl, H>((N > rhs.value) ? N : rhs.value);
  }
}

/// @copydoc max(meta::Const<N>, meta::Dynamic<A,L,H>)
template <nint_t N, nint_t A, nint_t L, nint_t H>
constexpr auto max(meta::Dynamic<A, L, H> lhs, meta::Const<N> rhs) {
  return max(rhs, lhs);
}

/**
 * @brief max of two Dynamics: bounds merge component-wise, alignment degrades
 * to gcd(A1, A2).
 */
template <nint_t A1, nint_t L1, nint_t H1, nint_t A2, nint_t L2, nint_t H2>
constexpr auto max(meta::Dynamic<A1, L1, H1> lhs, meta::Dynamic<A2, L2, H2> rhs) {
  constexpr nint_t lo = (L1 > L2) ? L1 : L2;
  constexpr nint_t hi = (H1 > H2) ? H1 : H2;
  return meta::Dynamic<std::gcd(A1, A2), lo, hi>(
      (lhs.value > rhs.value) ? lhs.value : rhs.value);
}

/// Value max nint_t → Value max Any{nint_t}
template <typename T>
  requires (std::derived_from<T, meta::Value> && !is_int_v<T>)
constexpr auto max(T lhs, nint_t rhs) {
  return max(lhs, meta::Any{rhs});
}

/// nint_t max Value → Any{nint_t} max Value
template <typename T>
  requires (std::derived_from<T, meta::Value> && !is_int_v<T>)
constexpr auto max(nint_t lhs, T rhs) {
  return max(meta::Any{lhs}, rhs);
}

/**
 * @brief Compile-time clamp of a Const into [Const<Lo>, Const<Hi>].
 */
template <nint_t N, nint_t Lo, nint_t Hi>
constexpr meta::Const<(N < Lo) ? Lo : (Hi < N) ? Hi : N>
clamp(meta::Const<N>, meta::Const<Lo>, meta::Const<Hi>) {
  constexpr nint_t v = (N < Lo) ? Lo : (Hi < N) ? Hi : N;
  return meta::Const<v>();
}

/**
 * @brief clamp of a Dynamic into compile-time bounds [Lo, Hi].
 *
 * Bounds tighten to the intersection [max(L, Lo), min(H, Hi)]; when the
 * intersection collapses (or the Dynamic lies entirely outside [Lo, Hi]) the
 * result folds to a Const. The result is v, Lo, or Hi, so the alignment
 * degrades to gcd(gcd(A, Lo), Hi).
 *
 * No nint_t-bound overloads exist: runtime bounds have no constraint to
 * propagate, so callers should convert explicitly.
 */
template <nint_t A, nint_t L, nint_t H, nint_t Lo, nint_t Hi>
constexpr auto clamp(meta::Dynamic<A, L, H> v, meta::Const<Lo>, meta::Const<Hi>) {
  static_assert(Lo <= Hi, "clamp bounds must be ordered");
  if constexpr (L != meta::kLoInf && L > Hi) {
    return meta::Const<Hi>();
  } else if constexpr (H != meta::kHiInf && H < Lo) {
    return meta::Const<Lo>();
  } else {
    constexpr nint_t rl = (Lo > L) ? Lo : L;
    constexpr nint_t rh = (Hi < H) ? Hi : H;
    if constexpr (rl == rh) {
      return meta::Const<rl>();
    } else {
      constexpr nint_t g = std::gcd(std::gcd(A, Lo), Hi);
      return meta::Dynamic<g, rl, rh>(::vecops::clamp(nint_t(v), Lo, Hi));
    }
  }
}

/**
 * @brief Compile-time ceiling division of Const values.
 */
template <nint_t N, nint_t M>
constexpr meta::Const<::vecops::ceil_div(N, M)> ceil_div(meta::Const<N>, meta::Const<M>) {
  return meta::Const<::vecops::ceil_div(N, M)>();
}

/**
 * @brief ceil_div of a Dynamic by a Const divisor (N > 0).
 *
 * Alignment follows operator/: A % N == 0 → A/N, otherwise 1. Bounds are the
 * endpoint ceil_div values (monotonic for positive divisors).
 */
template <nint_t A, nint_t L, nint_t H, nint_t N>
constexpr auto ceil_div(meta::Dynamic<A, L, H> lhs, meta::Const<N>) {
  static_assert(N > 0, "ceil_div divisor must be positive");
  constexpr nint_t g = (A % N == 0) ? A / N : 1;
  constexpr nint_t rl = L == meta::kLoInf
      ? meta::kLoInf : ::vecops::ceil_div(L, N);
  constexpr nint_t rh = H == meta::kHiInf
      ? meta::kHiInf : ::vecops::ceil_div(H, N);
  return meta::Dynamic<g, rl, rh>(::vecops::ceil_div(lhs.value, N));
}

/// @brief ceil_div by a runtime divisor: no constraint survives.
template <nint_t N, nint_t A, nint_t L, nint_t H>
constexpr meta::Any ceil_div(meta::Const<N> lhs, meta::Dynamic<A, L, H> rhs) {
  return meta::Any(::vecops::ceil_div(nint_t(lhs), rhs.value));
}

/// @brief ceil_div with runtime dividend and divisor: no constraint survives.
template <nint_t A1, nint_t L1, nint_t H1, nint_t A2, nint_t L2, nint_t H2>
constexpr meta::Any
ceil_div(meta::Dynamic<A1, L1, H1> lhs, meta::Dynamic<A2, L2, H2> rhs) {
  return meta::Any(::vecops::ceil_div(lhs.value, rhs.value));
}

/// Value ceil_div nint_t → Value ceil_div Any{nint_t}
template <typename T>
  requires (std::derived_from<T, meta::Value> && !is_int_v<T>)
constexpr auto ceil_div(T lhs, nint_t rhs) {
  return ceil_div(lhs, meta::Any{rhs});
}

/// nint_t ceil_div Value → Any{nint_t} ceil_div Value
template <typename T>
  requires (std::derived_from<T, meta::Value> && !is_int_v<T>)
constexpr auto ceil_div(nint_t lhs, T rhs) {
  return ceil_div(meta::Any{lhs}, rhs);
}

/**
 * @brief Compile-time floor division of Const values.
 */
template <nint_t N, nint_t M>
constexpr meta::Const<::vecops::floor_div(N, M)> floor_div(meta::Const<N>, meta::Const<M>) {
  return meta::Const<::vecops::floor_div(N, M)>();
}

/**
 * @brief floor_div of a Dynamic by a Const divisor (N > 0).
 *
 * Alignment follows operator/: A % N == 0 → A/N, otherwise 1. Bounds are the
 * endpoint floor_div values (monotonic for positive divisors).
 */
template <nint_t A, nint_t L, nint_t H, nint_t N>
constexpr auto floor_div(meta::Dynamic<A, L, H> lhs, meta::Const<N>) {
  static_assert(N > 0, "floor_div divisor must be positive");
  constexpr nint_t g = (A % N == 0) ? A / N : 1;
  constexpr nint_t rl = L == meta::kLoInf
      ? meta::kLoInf : ::vecops::floor_div(L, N);
  constexpr nint_t rh = H == meta::kHiInf
      ? meta::kHiInf : ::vecops::floor_div(H, N);
  return meta::Dynamic<g, rl, rh>(::vecops::floor_div(lhs.value, N));
}

/// @brief floor_div by a runtime divisor: no constraint survives.
template <nint_t N, nint_t A, nint_t L, nint_t H>
constexpr meta::Any floor_div(meta::Const<N> lhs, meta::Dynamic<A, L, H> rhs) {
  return meta::Any(::vecops::floor_div(nint_t(lhs), rhs.value));
}

/// @brief floor_div with runtime dividend and divisor: no constraint survives.
template <nint_t A1, nint_t L1, nint_t H1, nint_t A2, nint_t L2, nint_t H2>
constexpr meta::Any
floor_div(meta::Dynamic<A1, L1, H1> lhs, meta::Dynamic<A2, L2, H2> rhs) {
  return meta::Any(::vecops::floor_div(lhs.value, rhs.value));
}

/// Value floor_div nint_t → Value floor_div Any{nint_t}
template <typename T>
  requires (std::derived_from<T, meta::Value> && !is_int_v<T>)
constexpr auto floor_div(T lhs, nint_t rhs) {
  return floor_div(lhs, meta::Any{rhs});
}

/// nint_t floor_div Value → Any{nint_t} floor_div Value
template <typename T>
  requires (std::derived_from<T, meta::Value> && !is_int_v<T>)
constexpr auto floor_div(nint_t lhs, T rhs) {
  return floor_div(meta::Any{lhs}, rhs);
}

/**
 * @brief Compile-time upward alignment of a Const.
 */
template <nint_t N, nint_t M>
constexpr meta::Const<::vecops::align_up(N, M)> align_up(meta::Const<N>, meta::Const<M>) {
  return meta::Const<::vecops::align_up(N, M)>();
}

/**
 * @brief align_up of a Dynamic to a Const alignment (N > 0).
 *
 * A % N == 0 means every runtime value is already N-aligned, so align_up is
 * the identity and alignment A survives; otherwise the result is a multiple
 * of gcd(A, N).
 */
template <nint_t A, nint_t L, nint_t H, nint_t N>
constexpr auto align_up(meta::Dynamic<A, L, H> lhs, meta::Const<N>) {
  static_assert(N > 0, "align_up alignment must be positive");
  constexpr nint_t g = (A % N == 0) ? A : std::gcd(A, N);
  if constexpr (L == meta::kLoInf || H == meta::kHiInf) {
    return meta::Dynamic<g>(::vecops::align_up(lhs.value, N));
  } else {
    return meta::Dynamic<g, ::vecops::align_up(L, N), ::vecops::align_up(H, N)>(
        ::vecops::align_up(lhs.value, N));
  }
}

/// @brief align_up to a runtime alignment: the result keeps only the gcd.
template <nint_t N, nint_t A, nint_t L, nint_t H>
constexpr meta::Any align_up(meta::Const<N> lhs, meta::Dynamic<A, L, H> rhs) {
  return meta::Any(::vecops::align_up(nint_t(lhs), rhs.value));
}

/// @brief align_up with runtime value and alignment: no constraint survives.
template <nint_t A1, nint_t L1, nint_t H1, nint_t A2, nint_t L2, nint_t H2>
constexpr meta::Any
align_up(meta::Dynamic<A1, L1, H1> lhs, meta::Dynamic<A2, L2, H2> rhs) {
  return meta::Any(::vecops::align_up(lhs.value, rhs.value));
}

/// Value align_up nint_t → Value align_up Any{nint_t}
template <typename T>
  requires (std::derived_from<T, meta::Value> && !is_int_v<T>)
constexpr auto align_up(T lhs, nint_t rhs) {
  return align_up(lhs, meta::Any{rhs});
}

/// nint_t align_up Value → Any{nint_t} align_up Value
template <typename T>
  requires (std::derived_from<T, meta::Value> && !is_int_v<T>)
constexpr auto align_up(nint_t lhs, T rhs) {
  return align_up(meta::Any{lhs}, rhs);
}

/**
 * @brief Compile-time downward alignment of a Const.
 */
template <nint_t N, nint_t M>
constexpr meta::Const<::vecops::align_down(N, M)> align_down(meta::Const<N>, meta::Const<M>) {
  return meta::Const<::vecops::align_down(N, M)>();
}

/**
 * @brief align_down of a Dynamic to a Const alignment (N > 0).
 *
 * A % N == 0 means every runtime value is already N-aligned, so align_down is
 * the identity and alignment A survives; otherwise the result is a multiple
 * of gcd(A, N).
 */
template <nint_t A, nint_t L, nint_t H, nint_t N>
constexpr auto align_down(meta::Dynamic<A, L, H> lhs, meta::Const<N>) {
  static_assert(N > 0, "align_down alignment must be positive");
  constexpr nint_t g = (A % N == 0) ? A : std::gcd(A, N);
  if constexpr (L == meta::kLoInf || H == meta::kHiInf) {
    return meta::Dynamic<g>(::vecops::align_down(lhs.value, N));
  } else {
    return meta::Dynamic<g, ::vecops::align_down(L, N), ::vecops::align_down(H, N)>(
        ::vecops::align_down(lhs.value, N));
  }
}

/// @brief align_down to a runtime alignment: the result keeps only the gcd.
template <nint_t N, nint_t A, nint_t L, nint_t H>
constexpr meta::Any align_down(meta::Const<N> lhs, meta::Dynamic<A, L, H> rhs) {
  return meta::Any(::vecops::align_down(nint_t(lhs), rhs.value));
}

/// @brief align_down with runtime value and alignment: no constraint survives.
template <nint_t A1, nint_t L1, nint_t H1, nint_t A2, nint_t L2, nint_t H2>
constexpr meta::Any
align_down(meta::Dynamic<A1, L1, H1> lhs, meta::Dynamic<A2, L2, H2> rhs) {
  return meta::Any(::vecops::align_down(lhs.value, rhs.value));
}

/// Value align_down nint_t → Value align_down Any{nint_t}
template <typename T>
  requires (std::derived_from<T, meta::Value> && !is_int_v<T>)
constexpr auto align_down(T lhs, nint_t rhs) {
  return align_down(lhs, meta::Any{rhs});
}

/// nint_t align_down Value → Any{nint_t} align_down Value
template <typename T>
  requires (std::derived_from<T, meta::Value> && !is_int_v<T>)
constexpr auto align_down(nint_t lhs, T rhs) {
  return align_down(meta::Any{lhs}, rhs);
}

// Preserve Value-aware overload resolution when the ordinary integer operand
// is narrower or wider than nint_t. The raw side is still intentionally
// normalized to Any: only a Value operand carries compile-time constraints.
#define VECOPS_DEFINE_VALUE_INTEGER_OVERLOADS(Function)                     \
  template <meta::ValueType Value, meta::OtherInteger I>                    \
  constexpr auto Function(Value lhs, I rhs) {                               \
    return Function(lhs, static_cast<nint_t>(rhs));                          \
  }                                                                           \
  template <meta::OtherInteger I, meta::ValueType Value>                    \
  constexpr auto Function(I lhs, Value rhs) {                               \
    return Function(static_cast<nint_t>(lhs), rhs);                          \
  }

VECOPS_DEFINE_VALUE_INTEGER_OVERLOADS(min)
VECOPS_DEFINE_VALUE_INTEGER_OVERLOADS(max)
VECOPS_DEFINE_VALUE_INTEGER_OVERLOADS(ceil_div)
VECOPS_DEFINE_VALUE_INTEGER_OVERLOADS(floor_div)
VECOPS_DEFINE_VALUE_INTEGER_OVERLOADS(align_up)
VECOPS_DEFINE_VALUE_INTEGER_OVERLOADS(align_down)

#undef VECOPS_DEFINE_VALUE_INTEGER_OVERLOADS

} // namespace vecops

#endif // VECOPS_META_H
