//
// Created by renyz on 2026/3/16.
//

#ifndef VECOPS_VEC_H
#define VECOPS_VEC_H

#include "./VecBase.h"
#include "./VecPolicy.h"
#include "./impl/VectorizedUtil.h"
#include "./Capabilities.h"
#include "vecops/util/ScalarConvert.h"

/**
 * @file Vec.h
 * @brief High-level SIMD vector operations API.
 * 
 * This file provides the user-facing API for SIMD vector operations, designed
 * with a Highway-like interface. The API abstracts over different SIMD
 * architectures and automatically handles:
 * 
 * - Single-word vectors (one hardware register)
 * - Multi-word vectors (multiple registers concatenated)
 * - Scalable vectors (runtime-determined size like SVE)
 * - Scalar fallback (when no SIMD is available)
 * 
 * ## Usage Example
 * @code
 *   using namespace vecops::vec;
 *   Tag<float32_t, 4> t;  // 4-element float vector
 *   
 *   // Fill with a value
 *   auto v = fill(t, 1.0f);
 *   
 *   // Load from memory
 *   float data[4] = {1, 2, 3, 4};
 *   auto v2 = load(t, data);
 *   
 *   // Store to memory
 *   store(t, data, v);
 *   
 *   // Process partial data (n elements)
 *   auto v3 = load(t, data, 3, v);  // Load 3 elements, 4th from default_v
 * @endcode
 * 
 * ## Design Notes
 * 
 * The API uses a tag-based dispatch system. All operations take a Tag parameter
 * that specifies the vector type. This design:
 * 
 * 1. Enables compile-time type checking and optimization
 * 2. Supports both fixed-size and scalable vectors uniformly
 * 3. Allows the same API to work across different architectures
 * 
 * For multi-word vectors, operations are automatically unrolled across all words.
 * The internal `vmap` functions handle this transparently.
 */

#if defined(ARCH_X86_FAMILY) && defined(HAS_AVX)
  #include "./impl/x86_Basic.h"
  #include "./impl/x86_Bit.h"
  #include "./impl/x86_Conversions.h"
  #include "./impl/x86_MaskConversions.h"
  #include "./impl/x86_LoadStore.h"
  #include "./impl/x86_Arithmetic.h"
  #include "./impl/x86_Math.h"
#elif defined(ARCH_ARM_FAMILY) && defined(HAS_SVE)
  #include "./impl/SVE_Basic.h"
  #include "./impl/SVE_Bit.h"
  #include "./impl/SVE_Conversions.h"
  #include "./impl/SVE_MaskConversions.h"
  #include "./impl/SVE_LoadStore.h"
  #include "./impl/SVE_Arithmetic.h"
  #include "./impl/SVE_Math.h"
#elif defined(ARCH_ARM_FAMILY) && defined(HAS_NEON)
  #warning "NEON not implemented yet, falling back to Scalar implementation"
  #include "./impl/Scalar.h"
#else
  #include "./impl/Scalar.h"
#endif

#include "./impl/Common.h"

namespace vecops::vec {
using namespace CPU_CAPABILITY;

/* ************************************************************************** */
//                               Constructors                                 //
/* ************************************************************************** */

/**
 * @brief Create a vector filled with a single value.
 *
 * All elements of the result vector are set to `value`.
 *
 * @example
 *   Tag<float32_t, 4> t;
 *   auto v = fill(t, 3.14f);  // v = [3.14, 3.14, 3.14, 3.14]
 */
template <TLV_DECL_TAG(T)>
VECOPS_VFUNC Vec<T> fill(T t, TypeOf<T> value) {
  using namespace details;
  return vmap(
      t, [=](auto tt) VECOPS_INLINE_LAMBDA { return word::fill(tt, value); }
  );
}

/**
 * @brief Create a vector with first n elements filled with a value.
 *
 * Elements [0, n) are set to `value`; elements [n, size(t)) are taken from `default_v`.
 *
 * @return Vector with first n elements from value, rest from default_v
 */
template <TLV_DECL_TAG(T)>
VECOPS_VFUNC Vec<T> fill(T t, TypeOf<T> value, nint_t n, Vec<T> default_v) {
  using namespace details;
  return vmap(
      t, n,
      [=](auto tt, auto&& dd) VECOPS_INLINE_LAMBDA { return word::fill(tt, value); },
      [=](auto tt, nint_t rem, auto&& dd) VECOPS_INLINE_LAMBDA { return word::fill(tt, value, rem, dd); },
      ShardVec(t, default_v)
  );
}

template <TLV_DECL_TAG(T)>
VECOPS_VFUNC Vec<T> fill(T t, TypeOf<T> value, nint_t n, TypeOf<T> default_v = T()) {
  return vec::fill(t, value, n, vec::fill(t, default_v));
}

/**
 * @brief Create a vector where masked lanes are set to a value.
 *
 * For each lane i where mask[i] is true, result[i] = value.
 * Masked-out lanes take values from default_v.
 *
 * @return Vector with masked lanes from value, rest from default_v
 */
template <TLV_DECL_TAG(T)>
VECOPS_VFUNC Vec<T> fill(T t, TypeOf<T> value, Mask<T> m, Vec<T> default_v) {
  using namespace details;
  return vmap(
      t, [=](auto tt, auto&& mm, auto&& dd) VECOPS_INLINE_LAMBDA { return word::fill(tt, value, mm, dd); },
      ShardMask(t, m), ShardVec(t, default_v)
  );
}

template <TLV_DECL_TAG(T)>
VECOPS_VFUNC Vec<T> fill(T t, TypeOf<T> value, Mask<T> m, TypeOf<T> default_v = TypeOf<T>()) {
  return vec::fill(t, value, m, vec::fill(t, default_v));
}


/**
 * @brief Create a vector filled with zeros (default-constructed values).
 *
 * Equivalent to fill(t, T()) where T is default-constructible.
 *
 * @return Vector with all elements zero-initialized
 */
template <TLV_DECL_TAG(T)>
VECOPS_VFUNC Vec<T> zeros(T t) {
  using namespace details;
  return vmap(
      t, [=](auto tt) VECOPS_INLINE_LAMBDA { return word::zeros(tt); }
  );
}

/**
 * @brief Blend two vectors based on a mask: result[i] = m[i] ? v1[i] : v0[i].
 *
 * For each element, if the corresponding mask element is true, the element
 * from v1 is selected; otherwise, the element from v0 is selected.
 *
 * For multi-word vectors, the operation is automatically unrolled across
 * all constituent words.
 *
 * @return Blended vector
 *
 * @example
 *   Tag<float32_t, 4> t;
 *   auto v0 = fill(t, 0.0f);
 *   auto v1 = fill(t, 1.0f);
 *   auto m  = mwhilelt(t, 0, 2);  // m = [T, T, F, F]
 *   auto r  = blend(v0, m, v1);   // r = [1, 1, 0, 0]
 */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
VECOPS_VFUNC V blend(V v0, Mask<T> m, V v1) {
  using namespace details;
  constexpr T t;
  return vmap(
      t, [=](auto tt, auto&& vv0, auto&& mm, auto&& vv1) VECOPS_INLINE_LAMBDA { return word::blend(vv0, mm, vv1); },
      ShardVec(t, v0), ShardMask(t, m), ShardVec(t, v1)
  );
}

/**
 * @brief Create a mask filled with a single boolean value.
 *
 * @return Mask with all elements set to `value`
 *
 * @example
 *   Tag<float32_t, 4> t;
 *   auto m = mfill(t, true);  // All lanes active
 */
template <TLV_DECL_TAG(T)>
VECOPS_VFUNC Mask<T> mfill(T t, bool value) {
  using namespace details;
  return vmap(
      t, [=](auto tt) { return word::mfill(tt, value); }
  );
}

/**
 * @brief Create a mask with all lanes active (all true).
 *
 * @return Mask with all elements true
 */
template <TLV_DECL_TAG(T)>
VECOPS_VFUNC Mask<T> mtrue(T t) {
  return vec::mfill(t, true);
}

/**
 * @brief Create a mask with all lanes inactive (all false).
 *
 * @return Mask with all elements false
 */
template <TLV_DECL_TAG(T)>
VECOPS_VFUNC Mask<T> mfalse(T t) {
  return vec::mfill(t, false);
}

namespace details {

template <TLV_DECL_TAG(T)>
struct MWhileLtWords {
  Mask<T>& r;
  nint_t a;
  nint_t b;
  nint_t ws;

  template <nint_t I>
  VECOPS_VFUNC void operator()() const {
    constexpr auto wt = word_tag(T{});
    r = set_word_mask<I>(T{}, r, word::mwhilelt(wt, a + I * ws, b));
  }
};

} // namespace details

/**
 * @brief Create a mask where lanes i are true if (a + i) < b.
 *
 * This is useful for creating masks for processing a partial number of
 * elements. For a vector of size N:
 *   result[i] = (a + i) < b
 *
 * For multi-word vectors, each word's mask is computed independently
 * with appropriate offsets.
 *
 * @return Mask where lanes [a, b) are true
 *
 * @example
 *   Tag<float32_t, 4> t;
 *   auto m = mwhilelt(t, 0, 3);  // m = [true, true, true, false]
 *
 *   // Useful for processing partial data:
 *   auto m = mwhilelt(t, 0, n);  // Process n elements
 */
template <TLV_DECL_TAG(T)>
VECOPS_VFUNC Mask<T> mwhilelt(T t, nint_t a, nint_t b) {
  using namespace details;
  constexpr auto wt = word_tag(T{});
  if constexpr (is_word_vec(T{})) {
    return word::mwhilelt(wt, a, b);
  } else {
    constexpr nint_t nloop = num_words(T{});
    const nint_t ws = word_size(T{});
    Mask<T> r;
    foreach<nloop>(details::MWhileLtWords<T>{r, a, b, ws});
    return r;
  }
}

/**
 * @brief Create a mask where lanes i are true if (a + i) <= b.
 *
 * Equivalent to mwhilelt(t, a, b + 1).
 *
 * @return Mask where lanes [a, b] are true
 */
template <TLV_DECL_TAG(T)>
VECOPS_VFUNC Mask<T> mwhilele(T t, nint_t a, nint_t b) {
  return vec::mwhilelt(t, a, b + 1);
}

/**
 * @brief Create a mask where lanes i are true if (a + i) >= b.
 *
 * For a vector of size N:
 *   result[i] = (a + i) >= b
 *
 * @return Mask where lanes >= b are true
 */
template <TLV_DECL_TAG(T)>
VECOPS_VFUNC Mask<T> mwhilege(T t, nint_t a, nint_t b) {
  using namespace details;
  nint_t ws = word_size(t);
  return vmap(
      t, [=] <nint_t I>(auto tt) { return word::mwhilege(tt, a + I * ws, b); }
  );
}

/**
 * @brief Create a mask where lanes i are true if (a + i) > b.
 *
 * Equivalent to mwhilege(t, a, b + 1).
 *
 * @return Mask where lanes > b are true
 */
template <TLV_DECL_TAG(T)>
VECOPS_VFUNC Mask<T> mwhilegt(T t, nint_t a, nint_t b) {
  return vec::mwhilege(t, a, b + 1);
}


/* ************************************************************************** */
//                          Consecutive load & store                          //
/* ************************************************************************** */

namespace details {

template <TLV_DECL_TAG(T), typename Population>
VECOPS_VFUNC Vec<T> population_value(T t, Population&& population) {
  using P = policy_details::remove_cvref_t<Population>;
  if constexpr (policy_details::is_zero<P>) {
    return vec::zeros(t);
  } else {
    using V = policy_details::remove_cvref_t<decltype(population.value)>;
    if constexpr (is_vec<V>) {
      static_assert(std::is_same_v<V, Vec<T>>,
                    "opt::merge vector must match the load/convert result tag");
      return population.value;
    } else {
      static_assert(std::is_convertible_v<V, TypeOf<T>>,
                    "opt::merge scalar must be convertible to the result element type");
      return vec::fill(t, static_cast<TypeOf<T>>(population.value));
    }
  }
}

template <TLV_DECL_TAG(T), typename Alignment>
VECOPS_VFUNC Vec<T> load_full(
    T t, const TypeOf<T>* p, Alignment) {
  using namespace details;
  if constexpr (policy_details::is_aligned<Alignment>) {
    VECOPS_ASSERT(is_aligned(memory_alignment(t), p), "Not aligned");
    if constexpr (is_word_vec(t)) {
      return word::load(t, p);
    }
    return vmap(
        t,
        [](auto tt, const TypeOf<T>* pp) VECOPS_INLINE_LAMBDA {
          return word::load(tt, pp);
        },
        StepPointer(t, p));
  } else {
    if constexpr (is_word_vec(t)) {
      return word::loadu(t, p);
    }
    return vmap(
        t,
        [](auto tt, const TypeOf<T>* pp) VECOPS_INLINE_LAMBDA {
          return word::loadu(tt, pp);
        },
        StepPointer(t, p));
  }
}

template <TLV_DECL_TAG(T), typename Alignment>
VECOPS_VFUNC Vec<T> load_masked(
    T t, const TypeOf<T>* p, Mask<T> m, Alignment) {
  using namespace details;
  if constexpr (policy_details::is_aligned<Alignment>) {
    VECOPS_ASSERT(is_aligned(memory_alignment(t), p), "Not aligned");
    if constexpr (is_word_vec(t)) {
      return word::load(t, p, m);
    }
    return vmap(
        t,
        [](auto tt, const TypeOf<T>* pp,
           auto mm) VECOPS_INLINE_LAMBDA {
          return word::load(tt, pp, mm);
        },
        StepPointer(t, p), ShardMask(t, m));
  } else {
    if constexpr (is_word_vec(t)) {
      return word::loadu(t, p, m);
    }
    return vmap(
        t,
        [](auto tt, const TypeOf<T>* pp,
           auto mm) VECOPS_INLINE_LAMBDA {
          return word::loadu(tt, pp, mm);
        },
        StepPointer(t, p), ShardMask(t, m));
  }
}

template <TLV_DECL_TAG(T), typename Alignment>
VECOPS_VFUNC Vec<T> load_masked(
    T t, const TypeOf<T>* p, Mask<T> m, Vec<T> default_v, Alignment) {
  using namespace details;
  if constexpr (policy_details::is_aligned<Alignment>) {
    VECOPS_ASSERT(is_aligned(memory_alignment(t), p), "Not aligned");
    if constexpr (is_word_vec(t)) {
      return word::load(t, p, m, default_v);
    }
    return vmap(
        t,
        [](auto tt, const TypeOf<T>* pp, auto mm,
           auto dd) VECOPS_INLINE_LAMBDA {
          return word::load(tt, pp, mm, dd);
        },
        StepPointer(t, p), ShardMask(t, m), ShardVec(t, default_v));
  } else {
    if constexpr (is_word_vec(t)) {
      return word::loadu(t, p, m, default_v);
    }
    return vmap(
        t,
        [](auto tt, const TypeOf<T>* pp, auto mm,
           auto dd) VECOPS_INLINE_LAMBDA {
          return word::loadu(tt, pp, mm, dd);
        },
        StepPointer(t, p), ShardMask(t, m), ShardVec(t, default_v));
  }
}

template <TLV_DECL_TAG(T), typename Alignment>
VECOPS_VFUNC void store_full(
    T t, TypeOf<T>* p, Vec<T> v, Alignment) {
  using namespace details;
  if constexpr (policy_details::is_aligned<Alignment>) {
    VECOPS_ASSERT(is_aligned(memory_alignment(t), p), "Not aligned");
    if constexpr (is_word_vec(t)) {
      word::store(t, p, v);
      return;
    }
    vmap(
        t,
        [](auto tt, TypeOf<T>* pp, auto vv) VECOPS_INLINE_LAMBDA {
          word::store(tt, pp, vv);
        },
        StepPointer(t, p), ShardVec(t, v));
  } else {
    if constexpr (is_word_vec(t)) {
      word::storeu(t, p, v);
      return;
    }
    vmap(
        t,
        [](auto tt, TypeOf<T>* pp, auto vv) VECOPS_INLINE_LAMBDA {
          word::storeu(tt, pp, vv);
        },
        StepPointer(t, p), ShardVec(t, v));
  }
}

template <TLV_DECL_TAG(T), typename Alignment>
VECOPS_VFUNC void store_masked(
    T t, TypeOf<T>* p, Mask<T> m, Vec<T> v, Alignment) {
  using namespace details;
  if constexpr (policy_details::is_aligned<Alignment>) {
    VECOPS_ASSERT(is_aligned(memory_alignment(t), p), "Not aligned");
    if constexpr (is_word_vec(t)) {
      word::store(t, p, m, v);
      return;
    }
    vmap(
        t,
        [](auto tt, TypeOf<T>* pp, auto mm,
           auto vv) VECOPS_INLINE_LAMBDA {
          word::store(tt, pp, mm, vv);
        },
        StepPointer(t, p), ShardMask(t, m), ShardVec(t, v));
  } else {
    if constexpr (is_word_vec(t)) {
      word::storeu(t, p, m, v);
      return;
    }
    vmap(
        t,
        [](auto tt, TypeOf<T>* pp, auto mm,
           auto vv) VECOPS_INLINE_LAMBDA {
          word::storeu(tt, pp, mm, vv);
        },
        StepPointer(t, p), ShardMask(t, m), ShardVec(t, v));
  }
}

template <typename... Options>
consteval void validate_memory_options() {
  static_assert((policy_details::is_memory_option<Options> && ...),
                "unsupported load/store option");
  static_assert(
      policy_details::count_options<policy_details::IsAlignment, Options...> <= 1,
      "load/store accepts at most one alignment option");
  static_assert(
      policy_details::count_options<policy_details::IsTemporality, Options...> <= 1,
      "load/store accepts at most one temporal option");
  static_assert(
      policy_details::count_options<policy_details::IsActive, Options...> <= 1,
      "opt::masked and opt::first are mutually exclusive");
  static_assert(
      policy_details::count_options<policy_details::IsPopulation, Options...> <= 1,
      "load/store accepts at most one population option");
}

template <bool IsStore, typename... Options>
consteval bool valid_memory_options() {
  if constexpr (!(policy_details::is_memory_option<Options> && ...)) {
    return false;
  } else if constexpr (
      policy_details::count_options<
          policy_details::IsAlignment, Options...> > 1 ||
      policy_details::count_options<
          policy_details::IsTemporality, Options...> > 1 ||
      policy_details::count_options<
          policy_details::IsActive, Options...> > 1 ||
      policy_details::count_options<
          policy_details::IsPopulation, Options...> > 1) {
    return false;
  } else {
    constexpr int active_count =
        policy_details::count_options<
            policy_details::IsActive, Options...>;
    constexpr int population_count =
        policy_details::count_options<
            policy_details::IsPopulation, Options...>;
    return IsStore ? population_count == 0
                   : (population_count == 0 || active_count == 1);
  }
}

} // namespace details

/**
 * @brief Load consecutive elements with orthogonal memory options.
 *
 * Defaults to unaligned, temporal, unmasked access. Inactive lanes from
 * opt::masked/opt::first are zero unless opt::merge is supplied.
 */
template <TLV_DECL_TAG(T), typename... Options>
  requires (details::valid_memory_options<false, Options...>())
VECOPS_VFUNC Vec<T> load(
    T t, const TypeOf<T>* p, Options&&... options) {
  details::validate_memory_options<Options...>();
  constexpr int active_count =
      policy_details::count_options<policy_details::IsActive, Options...>;
  constexpr int population_count =
      policy_details::count_options<policy_details::IsPopulation, Options...>;
  static_assert(active_count != 0 || population_count == 0,
                "load population requires opt::masked or opt::first");

  auto&& alignment = policy_details::select_option<
      policy_details::IsAlignment>(
      mem::unaligned, std::forward<Options>(options)...);
  if constexpr (active_count == 0) {
    return details::load_full(t, p, alignment);
  } else {
    auto&& active = policy_details::select_option<
        policy_details::IsActive>(
        opt::first(size(t)), std::forward<Options>(options)...);
    Mask<T> m;
    if constexpr (policy_details::is_masked<decltype(active)>) {
      static_assert(std::is_same_v<
                        policy_details::remove_cvref_t<decltype(active.value)>,
                        Mask<T>>,
                    "opt::masked mask must match the load tag");
      m = active.value;
    } else {
      VECOPS_ASSERT(
          0 <= active.value && active.value <= size(t),
          "load count %zd !in 0..%zd", active.value, size(t));
      m = vec::mwhilelt(t, 0, active.value);
    }
    auto&& population = policy_details::select_option<
        policy_details::IsPopulation>(
        opt::zero, std::forward<Options>(options)...);
    if constexpr (policy_details::is_zero<decltype(population)>) {
      return details::load_masked(t, p, m, alignment);
    } else {
      return details::load_masked(
          t, p, m, details::population_value(t, population), alignment);
    }
  }
}

template <TLV_DECL_TAG(T)>
VECOPS_VFUNC Vec<T> load(
    T t, std::initializer_list<TypeOf<T>> list) {
  VECOPS_ASSERT(
      list.size() >= static_cast<size_t>(size(t)),
      "insufficient elements: %zd v.s. %zd",
      static_cast<nint_t>(list.size()), size(t));
  return vec::load(t, list.begin());
}

/**
 * @brief Store consecutive elements with orthogonal memory options.
 *
 * Defaults to unaligned, temporal, unmasked access. Inactive lanes selected
 * by opt::masked/opt::first are not written.
 */
template <TLV_DECL_TAG(T), typename... Options>
  requires (details::valid_memory_options<true, Options...>())
VECOPS_VFUNC void store(
    T t, TypeOf<T>* p, Vec<T> v, Options&&... options) {
  details::validate_memory_options<Options...>();
  constexpr int active_count =
      policy_details::count_options<policy_details::IsActive, Options...>;
  constexpr int population_count =
      policy_details::count_options<policy_details::IsPopulation, Options...>;
  static_assert(population_count == 0,
                "store does not accept zero/merge population options");

  auto&& alignment = policy_details::select_option<
      policy_details::IsAlignment>(
      mem::unaligned, std::forward<Options>(options)...);
  if constexpr (active_count == 0) {
    details::store_full(t, p, v, alignment);
  } else {
    auto&& active = policy_details::select_option<
        policy_details::IsActive>(
        opt::first(size(t)), std::forward<Options>(options)...);
    Mask<T> m;
    if constexpr (policy_details::is_masked<decltype(active)>) {
      static_assert(std::is_same_v<
                        policy_details::remove_cvref_t<decltype(active.value)>,
                        Mask<T>>,
                    "opt::masked mask must match the store tag");
      m = active.value;
    } else {
      VECOPS_ASSERT(
          0 <= active.value && active.value <= size(t),
          "store count %zd !in 0..%zd", active.value, size(t));
      m = vec::mwhilelt(t, 0, active.value);
    }
    details::store_masked(t, p, m, v, alignment);
  }
}


/* ************************************************************************** */
//                         Indexed gather & scatter                           //
/* ************************************************************************** */

/**
 * @brief Gather elements from memory using an index vector.
 *
 * For each lane i, loads from p[i[i]] and returns the result.
 * This is the vectorized equivalent of:
 *   result[i] = p[index[i]]
 *
 * @return Gathered vector
 *
 * @note Gather operations can be significantly slower than consecutive loads
 *       due to memory access patterns. Use consecutive loads when possible.
 *
 * @example
 *   Tag<float32_t, 4> t;
 *   float data[100] = {...};
 *   int32_t indices[4] = {10, 20, 5, 15};
 *   auto idx = load(Tag<int32_t, 4>(), indices);
 *   auto v = gather(t, data, idx);  // v[i] = data[indices[i]]
 */
template <TLV_DECL_TAG(T), TL_IF(sizeof(TypeOf<T>) < 4)>
VECOPS_VFUNC Vec<T> gather(T t, const TypeOf<T>* p, Vec<Rebind<GatherScatterIndex<TypeOf<T>>, T>> i) {
  if constexpr (num_words(t) > 1) {
    Half<T> th;
    using Ti = Rebind<GatherScatterIndex<TypeOf<T>>, T>;
    constexpr Ti ti;
    auto lo = vec::gather(th, p, word::lower(ti, i));
    auto hi = vec::gather(th, p, word::upper(ti, i));
    return word::concat(t, lo, hi);
  } else {
    return word::gather(t, p, i);
  }
}
template <TLV_DECL_TAG(T), TL_IF(sizeof(TypeOf<T>) >= 4)>
VECOPS_VFUNC Vec<T> gather(T t, const TypeOf<T>* p, Vec<Rebind<GatherScatterIndex<TypeOf<T>>, T>> i) {
  if constexpr (num_words(t) > 1) {
    Half<T> th;
    using Ti = Rebind<GatherScatterIndex<TypeOf<T>>, T>;
    constexpr Ti ti;
    auto lo = vec::gather(th, p, word::lower(ti, i));
    auto hi = vec::gather(th, p, word::upper(ti, i));
    return word::concat(t, lo, hi);
  } else {
    return word::gather(t, p, i);
  }
}

/**
 * @brief Gather first n elements using an index vector, with default for rest.
 *
 * For lanes [0, n), loads from p[index[i]]. Lanes [n, size(t)) take
 * values from default_v.
 *
 * @return Gathered vector
 */
template <TLV_DECL_TAG(T), TL_IF(sizeof(TypeOf<T>) < 4)>
VECOPS_VFUNC Vec<T> gather(T t, const TypeOf<T>* p, Vec<Rebind<GatherScatterIndex<TypeOf<T>>, T>> i, nint_t n, Vec<T> default_v) {
  VECOPS_ASSERT(0 <= n && n <= size(t), "%zd !in 0..%zd", n, size(t));
  if constexpr (num_words(t) > 1) {
    Half<T> th;
    using Ti = Rebind<GatherScatterIndex<TypeOf<T>>, T>;
    constexpr Ti ti;
    auto half_size = size(th);
    auto default_lo = word::lower(t, default_v);
    auto default_hi = word::upper(t, default_v);
    if (n <= half_size) {
      auto lo = vec::gather(th, p, word::lower(ti, i), n, default_lo);
      return word::concat(t, lo, default_hi);
    }
    auto lo = vec::gather(th, p, word::lower(ti, i));
    auto hi = vec::gather(th, p, word::upper(ti, i), n - half_size, default_hi);
    return word::concat(t, lo, hi);
  } else {
    return word::gather(t, p, i, n, default_v);
  }
}
template <TLV_DECL_TAG(T), TL_IF(sizeof(TypeOf<T>) >= 4)>
VECOPS_VFUNC Vec<T> gather(T t, const TypeOf<T>* p, Vec<Rebind<GatherScatterIndex<TypeOf<T>>, T>> i, nint_t n, Vec<T> default_v) {
  VECOPS_ASSERT(0 <= n && n <= size(t), "%zd !in 0..%zd", n, size(t));
  if constexpr (num_words(t) > 1) {
    Half<T> th;
    using Ti = Rebind<GatherScatterIndex<TypeOf<T>>, T>;
    constexpr Ti ti;
    auto half_size = size(th);
    auto default_lo = word::lower(t, default_v);
    auto default_hi = word::upper(t, default_v);
    if (n <= half_size) {
      auto lo = vec::gather(th, p, word::lower(ti, i), n, default_lo);
      return word::concat(t, lo, default_hi);
    }
    auto lo = vec::gather(th, p, word::lower(ti, i));
    auto hi = vec::gather(th, p, word::upper(ti, i), n - half_size, default_hi);
    return word::concat(t, lo, hi);
  } else {
    return word::gather(t, p, i, n, default_v);
  }
}

/**
 * @brief Gather first n elements using an index vector, with scalar default.
 *
 * Convenience overload that broadcasts a scalar default value.
 *
 * @return Gathered vector
 */
template <TLV_DECL_TAG(T)>
VECOPS_VFUNC Vec<T> gather(T t, const TypeOf<T>* p, Vec<Rebind<GatherScatterIndex<TypeOf<T>>, T>> i, nint_t n, TypeOf<T> default_v) {
  return vec::gather(t, p, i, n, vec::fill(t, default_v));
}

/**
 * @brief Masked gather from memory using an index vector.
 *
 * For each lane i where mask[i] is true, loads from p[index[i]].
 * For masked-out lanes, takes the value from default_v[i].
 *
 * @return Gathered vector
 *
 * @note May access p[index[i]] for masked-out lanes; ensure indices are valid.
 */
template <TLV_DECL_TAG(T), TL_IF(sizeof(TypeOf<T>) < 4)>
VECOPS_VFUNC Vec<T> gather(T t, const TypeOf<T>* p, Vec<Rebind<GatherScatterIndex<TypeOf<T>>, T>> i, Mask<T> m, Vec<T> default_v) {
  if constexpr (num_words(t) > 1) {
    Half<T> th;
    using Ti = Rebind<GatherScatterIndex<TypeOf<T>>, T>;
    constexpr Ti ti;
    auto lo = vec::gather(th, p, word::lower(ti, i), word::lower(t, m), word::lower(t, default_v));
    auto hi = vec::gather(th, p, word::upper(ti, i), word::upper(t, m), word::upper(t, default_v));
    return word::concat(t, lo, hi);
  } else {
    return word::gather(t, p, i, m, default_v);
  }
}
template <TLV_DECL_TAG(T), TL_IF(sizeof(TypeOf<T>) >= 4)>
VECOPS_VFUNC Vec<T> gather(T t, const TypeOf<T>* p, Vec<Rebind<GatherScatterIndex<TypeOf<T>>, T>> i, Mask<T> m, Vec<T> default_v) {
  if constexpr (num_words(t) > 1) {
    Half<T> th;
    using Ti = Rebind<GatherScatterIndex<TypeOf<T>>, T>;
    constexpr Ti ti;
    auto lo = vec::gather(th, p, word::lower(ti, i), word::lower(t, m), word::lower(t, default_v));
    auto hi = vec::gather(th, p, word::upper(ti, i), word::upper(t, m), word::upper(t, default_v));
    return word::concat(t, lo, hi);
  } else {
    return word::gather(t, p, i, m, default_v);
  }
}

/**
 * @brief Masked gather with scalar default.
 *
 * For each lane i where mask[i] is true, loads from p[index[i]].
 * Masked-out lanes take the scalar default_v.
 *
 * @return Gathered vector
 */
template <TLV_DECL_TAG(T)>
VECOPS_VFUNC Vec<T> gather(T t, const TypeOf<T>* p, Vec<Rebind<GatherScatterIndex<TypeOf<T>>, T>> i, Mask<T> m, TypeOf<T> default_v) {
  return vec::gather(t, p, i, m, vec::fill(t, default_v));
}

/**
 * @brief Scatter elements to memory using an index vector.
 *
 * For each lane i, stores v[i] to p[index[i]].
 * This is the vectorized equivalent of:
 *   p[index[i]] = v[i]
 *
 * @warning If indices are not unique, the result depends on execution order.
 *          Multiple writes to the same location may race.
 *
 * @note Scatter operations can be significantly slower than consecutive stores.
 */
template <TLV_DECL_TAG(T), TL_IF(sizeof(TypeOf<T>) < 4)>
VECOPS_VFUNC void scatter(T t, TypeOf<T>* p, Vec<Rebind<GatherScatterIndex<TypeOf<T>>, T>> i, Vec<T> v) {
  if constexpr (num_words(t) > 1) {
    Half<T> th;
    using Ti = Rebind<GatherScatterIndex<TypeOf<T>>, T>;
    constexpr Ti ti;
    vec::scatter(th, p, word::lower(ti, i), word::lower(t, v));
    vec::scatter(th, p, word::upper(ti, i), word::upper(t, v));
  } else {
    word::scatter(t, p, i, v);
  }
}
template <TLV_DECL_TAG(T), TL_IF(sizeof(TypeOf<T>) >= 4)>
VECOPS_VFUNC void scatter(T t, TypeOf<T>* p, Vec<Rebind<GatherScatterIndex<TypeOf<T>>, T>> i, Vec<T> v) {
  if constexpr (num_words(t) > 1) {
    Half<T> th;
    using Ti = Rebind<GatherScatterIndex<TypeOf<T>>, T>;
    constexpr Ti ti;
    vec::scatter(th, p, word::lower(ti, i), word::lower(t, v));
    vec::scatter(th, p, word::upper(ti, i), word::upper(t, v));
  } else {
    word::scatter(t, p, i, v);
  }
}

/**
 * @brief Scatter first n elements to memory using an index vector.
 *
 * Only lanes [0, n) are written to p[index[i]].
 */
template <TLV_DECL_TAG(T), TL_IF(sizeof(TypeOf<T>) < 4)>
VECOPS_VFUNC void scatter(T t, TypeOf<T>* p, Vec<Rebind<GatherScatterIndex<TypeOf<T>>, T>> i, nint_t n, Vec<T> v) {
  VECOPS_ASSERT(0 <= n && n <= size(t), "%zd !in 0..%zd", n, size(t));
  if constexpr (num_words(t) > 1) {
    Half<T> th;
    using Ti = Rebind<GatherScatterIndex<TypeOf<T>>, T>;
    constexpr Ti ti;
    auto half_size = size(th);
    if (n <= half_size) {
      vec::scatter(th, p, word::lower(ti, i), n, word::lower(t, v));
    } else {
      vec::scatter(th, p, word::lower(ti, i), word::lower(t, v));
      vec::scatter(th, p, word::upper(ti, i), n - half_size, word::upper(t, v));
    }
  } else {
    word::scatter(t, p, i, n, v);
  }
}
template <TLV_DECL_TAG(T), TL_IF(sizeof(TypeOf<T>) >= 4)>
VECOPS_VFUNC void scatter(T t, TypeOf<T>* p, Vec<Rebind<GatherScatterIndex<TypeOf<T>>, T>> i, nint_t n, Vec<T> v) {
  VECOPS_ASSERT(0 <= n && n <= size(t), "%zd !in 0..%zd", n, size(t));
  if constexpr (num_words(t) > 1) {
    Half<T> th;
    using Ti = Rebind<GatherScatterIndex<TypeOf<T>>, T>;
    constexpr Ti ti;
    auto half_size = size(th);
    if (n <= half_size) {
      vec::scatter(th, p, word::lower(ti, i), n, word::lower(t, v));
    } else {
      vec::scatter(th, p, word::lower(ti, i), word::lower(t, v));
      vec::scatter(th, p, word::upper(ti, i), n - half_size, word::upper(t, v));
    }
  } else {
    word::scatter(t, p, i, n, v);
  }
}

/**
 * @brief Masked scatter to memory using an index vector.
 *
 * For each lane i where mask[i] is true, stores v[i] to p[index[i]].
 * Masked-out lanes are not written.
 */
template <TLV_DECL_TAG(T), TL_IF(sizeof(TypeOf<T>) < 4)>
VECOPS_VFUNC void scatter(T t, TypeOf<T>* p, Vec<Rebind<GatherScatterIndex<TypeOf<T>>, T>> i, Mask<T> m, Vec<T> v) {
  if constexpr (num_words(t) > 1) {
    Half<T> th;
    using Ti = Rebind<GatherScatterIndex<TypeOf<T>>, T>;
    constexpr Ti ti;
    vec::scatter(th, p, word::lower(ti, i), word::lower(t, m), word::lower(t, v));
    vec::scatter(th, p, word::upper(ti, i), word::upper(t, m), word::upper(t, v));
  } else {
    word::scatter(t, p, i, m, v);
  }
}
template <TLV_DECL_TAG(T), TL_IF(sizeof(TypeOf<T>) >= 4)>
VECOPS_VFUNC void scatter(T t, TypeOf<T>* p, Vec<Rebind<GatherScatterIndex<TypeOf<T>>, T>> i, Mask<T> m, Vec<T> v) {
  if constexpr (num_words(t) > 1) {
    Half<T> th;
    using Ti = Rebind<GatherScatterIndex<TypeOf<T>>, T>;
    constexpr Ti ti;
    vec::scatter(th, p, word::lower(ti, i), word::lower(t, m), word::lower(t, v));
    vec::scatter(th, p, word::upper(ti, i), word::upper(t, m), word::upper(t, v));
  } else {
    word::scatter(t, p, i, m, v);
  }
}


/* ************************************************************************** */
//                             Get / set element                              //
/* ************************************************************************** */

/**
 * @brief Get a single element from a vector by index.
 *
 * @warning This operation is relatively slow as it requires extracting
 *          the element from a SIMD register. Avoid using in performance-
 *          critical inner loops.
 * @return The element at the specified index
 */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
VECOPS_VFUNC TypeOf<T> get(V v, nint_t index) {
  constexpr T t;
  nint_t ws = word_size(t);
  nint_t ord = index / ws, off = index % ws;
  auto word = get_word(t, v, ord);
  return word::get(word, off);
}

template <TLV_DECL_TAG(T)>
VECOPS_VFUNC TypeOf<T> get(T t, Vec<T> v, nint_t index) {
  VECOPS_ASSERT(0 <= index && index < size(t), "%zd !in 0:%zd", index, size(t));
  return vec::get<Vec<T>, T>(v, index);
}


/**
 * @brief Get a single element from a mask by index.
 *
 * @warning This operation is relatively slow. Avoid using in hot loops.
 * @return The boolean value at the specified index
 */
template <TLV_DECL_TAG(T)>
VECOPS_VFUNC bool get(T t, Mask<T> m, nint_t index) {
  VECOPS_ASSERT(0 <= index && index < size(t), "%zd !in 0:%zd", index, size(t));
  nint_t ws = word_size(t);
  nint_t ord = index / ws, off = index % ws;
  auto word = get_word_mask(t, m, ord);
  constexpr auto wt = word_tag(t);
  return word::get(wt, word, off);
}

/**
 * @brief Set a single element in a vector by index.
 *
 * @warning This operation is relatively slow as it requires modifying
 *          the SIMD register. Avoid using in performance-critical loops.
 *
 * @return Vector with the element at index set to x
 */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
VECOPS_VFUNC V set(V v, nint_t index, TypeOf<T> x) {
  constexpr T t;
  nint_t ws = word_size(t);
  nint_t ord = index / ws, off = index % ws;
  auto word = get_word(t, v, ord);
  return set_word(t, v, ord, word::set(word, off, x));
}
template <TLV_DECL_TAG(T)>
VECOPS_VFUNC Vec<T> set(T t, Vec<T> v, nint_t index, TypeOf<T> x) {
  VECOPS_ASSERT(0 <= index && index < size(t), "%zd !in 0:%zd", index, size(t));
  return vec::set<Vec<T>, T>(v, index, x);
}

/**
 * @brief Set a single element in a mask by index.
 *
 * @warning This operation is relatively slow. Avoid using in hot loops.
 *
 * @return Mask with the element at index set to x
 */
template <TLV_DECL_TAG(T)>
VECOPS_VFUNC auto set(T t, Mask<T> m, nint_t index, bool x) -> Mask<T> {
  VECOPS_ASSERT(0 <= index && index < size(t), "%zd !in 0:%zd", index, size(t));
  nint_t ws = word_size(t);
  nint_t ord = index / ws, off = index % ws;
  auto word = get_word_mask(t, m, ord);
  constexpr auto wt = word_tag(t);
  return set_word_mask(t, m, ord, word::set(wt, word, off, x));
}



/* ************************************************************************** */
//                       Basic arithmetic operations                          //
/* ************************************************************************** */

/**
 * @brief Element-wise addition: result[i] = a[i] + b[i].
 * @return Sum of the two vectors
 */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
VECOPS_VFUNC V add(V a, V b) {
  using namespace details;
  constexpr T t;
  return vmap(
      t, [=](auto tt, auto&& aa, auto&& bb) VECOPS_INLINE_LAMBDA { return word::add(aa, bb); },
      ShardVec(t, a), ShardVec(t, b)
  );
}

/**
 * @brief Masked element-wise addition: result[i] = a[i] + b[i] for masked lanes.
 *
 * For masked-out lanes, result[i] = a[i].
 *
 * @return Sum of the two vectors for masked lanes
 */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
VECOPS_VFUNC V add(V a, V b, Mask<T> m) {
  using namespace details;
  constexpr T t;
  return vmap(
      t, [=](auto tt, auto&& aa, auto&& bb, auto&& mm) VECOPS_INLINE_LAMBDA { return word::add(aa, bb, mm); },
      ShardVec(t, a), ShardVec(t, b), ShardMask(t, m)
  );
}

/**
 * @brief Element-wise subtraction: result[i] = a[i] - b[i].
 * @return Difference of the two vectors
 */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
VECOPS_VFUNC V sub(V a, V b) {
  using namespace details;
  constexpr T t;
  return vmap(
      t, [=](auto tt, auto&& aa, auto&& bb) VECOPS_INLINE_LAMBDA { return word::sub(aa, bb); },
      ShardVec(t, a), ShardVec(t, b)
  );
}

/**
 * @brief Masked element-wise subtraction: result[i] = a[i] - b[i] for masked lanes.
 *
 * For masked-out lanes, result[i] = a[i].
 */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
VECOPS_VFUNC V sub(V a, V b, Mask<T> m) {
  using namespace details;
  constexpr T t;
  return vmap(
      t, [=](auto tt, auto&& aa, auto&& bb, auto&& mm) VECOPS_INLINE_LAMBDA { return word::sub(aa, bb, mm); },
      ShardVec(t, a), ShardVec(t, b), ShardMask(t, m)
  );
}

/**
 * @brief Element-wise multiplication: result[i] = a[i] * b[i].
 * @return Product of the two vectors
 */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
VECOPS_VFUNC V mul(V a, V b) {
  using namespace details;
  constexpr T t;
  return vmap(
      t, [=](auto tt, auto&& aa, auto&& bb) VECOPS_INLINE_LAMBDA { return word::mul(aa, bb); },
      ShardVec(t, a), ShardVec(t, b)
  );
}

/**
 * @brief Masked element-wise multiplication: result[i] = a[i] * b[i] for masked lanes.
 *
 * For masked-out lanes, result[i] = a[i].
 */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
VECOPS_VFUNC V mul(V a, V b, Mask<T> m) {
  using namespace details;
  constexpr T t;
  return vmap(
      t, [=](auto tt, auto&& aa, auto&& bb, auto&& mm) VECOPS_INLINE_LAMBDA { return word::mul(aa, bb, mm); },
      ShardVec(t, a), ShardVec(t, b), ShardMask(t, m)
  );
}

/**
 * @brief Element-wise fused multiply-add: result[i] = a[i] * b[i] + c[i].
 *
 * Uses a native FMA instruction when available for the element type and
 * target backend; otherwise falls back to multiply plus add.
 *
 * @return Fused multiply-add result
 */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
VECOPS_VFUNC V fmadd(V a, V b, V c) {
  using namespace details;
  constexpr T t;
  return vmap(
      t, [=](auto tt, auto&& aa, auto&& bb, auto&& cc) VECOPS_INLINE_LAMBDA { return word::fmadd(aa, bb, cc); },
      ShardVec(t, a), ShardVec(t, b), ShardVec(t, c)
  );
}

/**
 * @brief Masked fused multiply-add for masked lanes.
 *
 * For masked lanes, result[i] = a[i] * b[i] + c[i].
 * For masked-out lanes, result[i] = a[i].
 *
 * @return Fused multiply-add result for masked lanes
 */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
VECOPS_VFUNC V fmadd(V a, V b, V c, Mask<T> m) {
  using namespace details;
  constexpr T t;
  return vmap(
      t, [=](auto tt, auto&& aa, auto&& bb, auto&& cc, auto&& mm) VECOPS_INLINE_LAMBDA { return word::fmadd(aa, bb, cc, mm); },
      ShardVec(t, a), ShardVec(t, b), ShardVec(t, c), ShardMask(t, m)
  );
}

/**
 * @brief Element-wise fused multiply-subtract: result[i] = a[i] * b[i] - c[i].
 *
 * Uses a native FMA instruction when available for the element type and
 * target backend; otherwise falls back to multiply plus subtract.
 *
 * @return Fused multiply-subtract result
 */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
VECOPS_VFUNC V fmsub(V a, V b, V c) {
  using namespace details;
  constexpr T t;
  return vmap(
      t, [=](auto tt, auto&& aa, auto&& bb, auto&& cc) { return word::fmsub(aa, bb, cc); },
      ShardVec(t, a), ShardVec(t, b), ShardVec(t, c)
  );
}

/**
 * @brief Masked fused multiply-subtract for masked lanes.
 *
 * For masked lanes, result[i] = a[i] * b[i] - c[i].
 * For masked-out lanes, result[i] = a[i].
 *
 * @return Fused multiply-subtract result for masked lanes
 */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
VECOPS_VFUNC V fmsub(V a, V b, V c, Mask<T> m) {
  using namespace details;
  constexpr T t;
  return vmap(
      t, [=](auto tt, auto&& aa, auto&& bb, auto&& cc, auto&& mm) { return word::fmsub(aa, bb, cc, mm); },
      ShardVec(t, a), ShardVec(t, b), ShardVec(t, c), ShardMask(t, m)
  );
}

/**
 * @brief Element-wise negative fused multiply-add: result[i] = -(a[i] * b[i]) + c[i].
 *
 * Uses a native FMA instruction when available for the element type and
 * target backend; otherwise falls back to multiply plus add/subtract.
 *
 * @return Negative fused multiply-add result
 */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
VECOPS_VFUNC V fnmadd(V a, V b, V c) {
  using namespace details;
  constexpr T t;
  return vmap(
      t, [=](auto tt, auto&& aa, auto&& bb, auto&& cc) { return word::fnmadd(aa, bb, cc); },
      ShardVec(t, a), ShardVec(t, b), ShardVec(t, c)
  );
}

/**
 * @brief Masked negative fused multiply-add for masked lanes.
 *
 * For masked lanes, result[i] = -(a[i] * b[i]) + c[i].
 * For masked-out lanes, result[i] = a[i].
 *
 * @return Negative fused multiply-add result for masked lanes
 */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
VECOPS_VFUNC V fnmadd(V a, V b, V c, Mask<T> m) {
  using namespace details;
  constexpr T t;
  return vmap(
      t, [=](auto tt, auto&& aa, auto&& bb, auto&& cc, auto&& mm) { return word::fnmadd(aa, bb, cc, mm); },
      ShardVec(t, a), ShardVec(t, b), ShardVec(t, c), ShardMask(t, m)
  );
}

/**
 * @brief Element-wise negative fused multiply-subtract: result[i] = -(a[i] * b[i]) - c[i].
 *
 * Uses a native FMA instruction when available for the element type and
 * target backend; otherwise falls back to multiply plus add/subtract.
 *
 * @return Negative fused multiply-subtract result
 */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
VECOPS_VFUNC V fnmsub(V a, V b, V c) {
  using namespace details;
  constexpr T t;
  return vmap(
      t, [=](auto tt, auto&& aa, auto&& bb, auto&& cc) { return word::fnmsub(aa, bb, cc); },
      ShardVec(t, a), ShardVec(t, b), ShardVec(t, c)
  );
}

/**
 * @brief Masked negative fused multiply-subtract for masked lanes.
 *
 * For masked lanes, result[i] = -(a[i] * b[i]) - c[i].
 * For masked-out lanes, result[i] = a[i].
 *
 * @return Negative fused multiply-subtract result for masked lanes
 */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
VECOPS_VFUNC V fnmsub(V a, V b, V c, Mask<T> m) {
  using namespace details;
  constexpr T t;
  return vmap(
      t, [=](auto tt, auto&& aa, auto&& bb, auto&& cc, auto&& mm) { return word::fnmsub(aa, bb, cc, mm); },
      ShardVec(t, a), ShardVec(t, b), ShardVec(t, c), ShardMask(t, m)
  );
}

/**
 * @brief Element-wise division: result[i] = a[i] / b[i].
 * @return Quotient of the two vectors
 */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
VECOPS_VFUNC V div(V a, V b) {
  using namespace details;
  constexpr T t;
  return vmap(
      t, [=](auto tt, auto&& aa, auto&& bb) { return word::div(aa, bb); },
      ShardVec(t, a), ShardVec(t, b)
  );
}

/**
 * @brief Masked element-wise division: result[i] = a[i] / b[i] for masked lanes.
 *
 * For masked-out lanes, result[i] = a[i].
 */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
VECOPS_VFUNC V div(V a, V b, Mask<T> m) {
  using namespace details;
  constexpr T t;
  return vmap(
      t, [=](auto tt, auto&& aa, auto&& bb, auto&& mm) { return word::div(aa, bb, mm); },
      ShardVec(t, a), ShardVec(t, b), ShardMask(t, m)
  );
}

/**
 * @brief Element-wise maximum: result[i] = max(a[i], b[i]).
 * @return Vector with maximum elements
 */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
VECOPS_VFUNC V max(V a, V b) {
  using namespace details;
  constexpr T t;
  return vmap(
      t, [=](auto tt, auto&& aa, auto&& bb) { return word::max(aa, bb); },
      ShardVec(t, a), ShardVec(t, b)
  );
}

/**
 * @brief Masked element-wise maximum: result[i] = max(a[i], b[i]) for masked lanes.
 *
 * For masked-out lanes, result[i] = a[i].
 */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
VECOPS_VFUNC V max(V a, V b, Mask<T> m) {
  using namespace details;
  constexpr T t;
  return vmap(
      t, [=](auto tt, auto&& aa, auto&& bb, auto&& mm) { return word::max(aa, bb, mm); },
      ShardVec(t, a), ShardVec(t, b), ShardMask(t, m)
  );
}

/**
 * @brief Element-wise minimum: result[i] = min(a[i], b[i]).
 * @return Vector with minimum elements
 */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
VECOPS_VFUNC V min(V a, V b) {
  using namespace details;
  constexpr T t;
  return vmap(
      t, [=](auto tt, auto&& aa, auto&& bb) { return word::min(aa, bb); },
      ShardVec(t, a), ShardVec(t, b)
  );
}

/**
 * @brief Masked element-wise minimum: result[i] = min(a[i], b[i]) for masked lanes.
 *
 * For masked-out lanes, result[i] = a[i].
 */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
VECOPS_VFUNC V min(V a, V b, Mask<T> m) {
  using namespace details;
  constexpr T t;
  return vmap(
      t, [=](auto tt, auto&& aa, auto&& bb, auto&& mm) { return word::min(aa, bb, mm); },
      ShardVec(t, a), ShardVec(t, b), ShardMask(t, m)
  );
}

/**
 * @brief Element-wise bitwise AND: result[i] = a[i] & b[i].
 * @return Bitwise AND of the two vectors
 */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
VECOPS_VFUNC V bit_and(V a, V b) {
  using namespace details;
  constexpr T t;
  return vmap(
      t, [=](auto tt, auto&& aa, auto&& bb) { return word::bit_and(aa, bb); },
      ShardVec(t, a), ShardVec(t, b)
  );
}

/**
 * @brief Masked element-wise bitwise AND for masked lanes.
 *
 * For masked lanes, result[i] = a[i] & b[i].
 * For masked-out lanes, result[i] = a[i].
 */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
VECOPS_VFUNC V bit_and(V a, V b, Mask<T> m) {
  using namespace details;
  constexpr T t;
  return vmap(
      t, [=](auto tt, auto&& aa, auto&& bb, auto&& mm) { return word::bit_and(aa, bb, mm); },
      ShardVec(t, a), ShardVec(t, b), ShardMask(t, m)
  );
}

/**
 * @brief Element-wise bitwise OR: result[i] = a[i] | b[i].
 * @return Bitwise OR of the two vectors
 */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
VECOPS_VFUNC V bit_or(V a, V b) {
  using namespace details;
  constexpr T t;
  return vmap(
      t, [=](auto tt, auto&& aa, auto&& bb) { return word::bit_or(aa, bb); },
      ShardVec(t, a), ShardVec(t, b)
  );
}

/**
 * @brief Masked element-wise bitwise OR for masked lanes.
 *
 * For masked lanes, result[i] = a[i] | b[i].
 * For masked-out lanes, result[i] = a[i].
 */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
VECOPS_VFUNC V bit_or(V a, V b, Mask<T> m) {
  using namespace details;
  constexpr T t;
  return vmap(
      t, [=](auto tt, auto&& aa, auto&& bb, auto&& mm) { return word::bit_or(aa, bb, mm); },
      ShardVec(t, a), ShardVec(t, b), ShardMask(t, m)
  );
}

/**
 * @brief Element-wise bitwise XOR: result[i] = a[i] ^ b[i].
 * @return Bitwise XOR of the two vectors
 */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
VECOPS_VFUNC V bit_xor(V a, V b) {
  using namespace details;
  constexpr T t;
  return vmap(
      t, [=](auto tt, auto&& aa, auto&& bb) { return word::bit_xor(aa, bb); },
      ShardVec(t, a), ShardVec(t, b)
  );
}

/**
 * @brief Masked element-wise bitwise XOR for masked lanes.
 *
 * For masked lanes, result[i] = a[i] ^ b[i].
 * For masked-out lanes, result[i] = a[i].
 */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
VECOPS_VFUNC V bit_xor(V a, V b, Mask<T> m) {
  using namespace details;
  constexpr T t;
  return vmap(
      t, [=](auto tt, auto&& aa, auto&& bb, auto&& mm) { return word::bit_xor(aa, bb, mm); },
      ShardVec(t, a), ShardVec(t, b), ShardMask(t, m)
  );
}

/**
 * @brief Element-wise bitwise AND-NOT: result[i] = (~a[i]) & b[i].
 * @return Bitwise AND-NOT of the two vectors
 */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
VECOPS_VFUNC V bit_andnot(V a, V b) {
  using namespace details;
  constexpr T t;
  return vmap(
      t, [=](auto tt, auto&& aa, auto&& bb) { return word::bit_andnot(aa, bb); },
      ShardVec(t, a), ShardVec(t, b)
  );
}

/**
 * @brief Masked element-wise bitwise AND-NOT for masked lanes.
 *
 * For masked lanes, result[i] = (~a[i]) & b[i].
 * For masked-out lanes, result[i] = a[i].
 */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
VECOPS_VFUNC V bit_andnot(V a, V b, Mask<T> m) {
  using namespace details;
  constexpr T t;
  return vmap(
      t, [=](auto tt, auto&& aa, auto&& bb, auto&& mm) { return word::bit_andnot(aa, bb, mm); },
      ShardVec(t, a), ShardVec(t, b), ShardMask(t, m)
  );
}

/**
 * @brief Element-wise left shift: result[i] = v[i] << count.
 * @param count Shift count (same for all lanes)
 * @return Shifted vector
 */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
VECOPS_VFUNC V bit_shl(V v, int count) {
  using namespace details;
  constexpr T t;
  return vmap(
      t, [=](auto tt, auto&& vv) { return word::bit_shl(vv, count); },
      ShardVec(t, v)
  );
}

/**
 * @brief Masked element-wise left shift for masked lanes.
 *
 * For masked lanes, result[i] = v[i] << count.
 * For masked-out lanes, result[i] = v[i].
 */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
VECOPS_VFUNC V bit_shl(V v, int count, Mask<T> m) {
  using namespace details;
  constexpr T t;
  return vmap(
      t, [=](auto tt, auto&& vv, auto&& mm) { return word::bit_shl(vv, count, mm); },
      ShardVec(t, v), ShardMask(t, m)
  );
}

/**
 * @brief Element-wise right shift: result[i] = v[i] >> count.
 * @param count Shift count (same for all lanes)
 * @return Shifted vector
 */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
VECOPS_VFUNC V bit_shr(V v, int count) {
  using namespace details;
  constexpr T t;
  return vmap(
      t, [=](auto tt, auto&& vv) { return word::bit_shr(vv, count); },
      ShardVec(t, v)
  );
}

/**
 * @brief Masked element-wise right shift for masked lanes.
 *
 * For masked lanes, result[i] = v[i] >> count.
 * For masked-out lanes, result[i] = v[i].
 */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
VECOPS_VFUNC V bit_shr(V v, int count, Mask<T> m) {
  using namespace details;
  constexpr T t;
  return vmap(
      t, [=](auto tt, auto&& vv, auto&& mm) { return word::bit_shr(vv, count, mm); },
      ShardVec(t, v), ShardMask(t, m)
  );
}


/**
 * @brief Element-wise bitwise NOT: result[i] = ~v[i].
 * @return Bitwise complement of the vector
 */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
VECOPS_VFUNC V bit_not(V v) {
  using namespace details;
  constexpr T t;
  return vmap(
      t, [=](auto tt, auto&& vv) { return word::bit_not(vv); },
      ShardVec(t, v)
  );
}

/**
 * @brief Masked bitwise NOT: result[i] = ~v[i] for masked lanes, default_v[i] otherwise.
 */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
VECOPS_VFUNC V bit_not(V v, Mask<T> m, V default_v) {
  using namespace details;
  constexpr T t;
  return vmap(
      t, [=](auto tt, auto&& vv, auto&& mm, auto&& dd) { return word::bit_not(vv, mm, dd); },
      ShardVec(t, v), ShardMask(t, m), ShardVec(t, default_v)
  );
}

/**
 * @brief Masked bitwise NOT with original value as default for masked-out lanes.
 *
 * For masked lanes, result[i] = ~v[i].
 * For masked-out lanes, result[i] = v[i].
 */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
VECOPS_VFUNC V bit_not(V v, Mask<T> m) {
  return vec::bit_not(v, m, v);
}

/* ********************************************************************** */
//                     Mask Bitwise Operations                             //
/* ********************************************************************** */

namespace details {
template <typename F, TLV_DECL_TAG(T)>
VECOPS_VFUNC Mask<T> mask_bitwise_op(F&& f, T t, Mask<T> a, Mask<T> b) {
  using namespace details;
  constexpr nint_t NW = num_words(t);
  if constexpr (NW <= 1) {
    return f(a, b);
  } else {
    Mask<T> r;
    foreach<NW>([&]<nint_t I>{
      r = set_word_mask<I>(t, r, f(get_word_mask<I>(t, a), get_word_mask<I>(t, b)));
    });
    return r;
  }
}
template <typename F, TLV_DECL_TAG(T)>
VECOPS_VFUNC Mask<T> mask_unary_op(F&& f, T t, Mask<T> a) {
  using namespace details;
  constexpr nint_t NW = num_words(t);
  if constexpr (NW <= 1) {
    return f(a);
  } else {
    Mask<T> r;
    foreach<NW>([&]<nint_t I>{
      r = set_word_mask<I>(t, r, f(get_word_mask<I>(t, a)));
    });
    return r;
  }
}
}  // namespace details

/**
 * @brief Bitwise AND of two masks: result[i] = a[i] & b[i].
 */
template <TLV_DECL_TAG(T)>
VECOPS_VFUNC Mask<T> bit_and(T t, Mask<T> a, Mask<T> b) {
  return details::mask_bitwise_op(
    [](auto aa, auto bb) { return word::bit_and(aa, bb); }, t, a, b);
}

/**
 * @brief Bitwise OR of two masks: result[i] = a[i] | b[i].
 */
template <TLV_DECL_TAG(T)>
VECOPS_VFUNC Mask<T> bit_or(T t, Mask<T> a, Mask<T> b) {
  return details::mask_bitwise_op(
    [](auto aa, auto bb) { return word::bit_or(aa, bb); }, t, a, b);
}

/**
 * @brief Bitwise XOR of two masks: result[i] = a[i] ^ b[i].
 */
template <TLV_DECL_TAG(T)>
VECOPS_VFUNC Mask<T> bit_xor(T t, Mask<T> a, Mask<T> b) {
  return details::mask_bitwise_op(
    [](auto aa, auto bb) { return word::bit_xor(aa, bb); }, t, a, b);
}

/**
 * @brief Bitwise AND-NOT of two masks: result[i] = ~a[i] & b[i].
 */
template <TLV_DECL_TAG(T)>
VECOPS_VFUNC Mask<T> bit_andnot(T t, Mask<T> a, Mask<T> b) {
  return details::mask_bitwise_op(
    [](auto aa, auto bb) { return word::bit_andnot(aa, bb); }, t, a, b);
}

/**
 * @brief Bitwise NOT of a mask: result[i] = ~a[i].
 */
template <TLV_DECL_TAG(T)>
VECOPS_VFUNC Mask<T> bit_not(T t, Mask<T> a) {
  return details::mask_unary_op(
    [](auto aa) { return word::bit_not(aa); }, t, a);
}

/**
 * @brief Element-wise negation: result[i] = -v[i].
 * @return Negated vector
 */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
VECOPS_VFUNC V neg(V v) {
  using namespace details;
  constexpr T t;
  return vmap(
      t, [=](auto tt, auto&& vv) { return word::neg(vv); },
      ShardVec(t, v)
  );
}

/**
 * @brief Masked negation: result[i] = -v[i] for masked lanes, default_v[i] otherwise.
 */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
VECOPS_VFUNC V neg(V v, Mask<T> m, V default_v) {
  using namespace details;
  constexpr T t;
  return vmap(
      t, [=](auto tt, auto&& vv, auto&& mm, auto&& dd) { return word::neg(vv, mm, dd); },
      ShardVec(t, v), ShardMask(t, m), ShardVec(t, default_v)
  );
}

/**
 * @brief Masked negation with original value as default for masked-out lanes.
 *
 * For masked lanes, result[i] = -v[i].
 * For masked-out lanes, result[i] = v[i].
 */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
VECOPS_VFUNC V neg(V v, Mask<T> m) {
  return vec::neg(v, m, v);
}


/**
 * @brief Element-wise absolute value: result[i] = |v[i]|.
 * @return Absolute value of the vector
 */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
VECOPS_VFUNC V abs(V v) {
  using namespace details;
  constexpr T t;
  return vmap(
      t, [=](auto tt, auto&& vv) { return word::abs(vv); },
      ShardVec(t, v)
  );
}

/**
 * @brief Masked absolute value: result[i] = |v[i]| for masked lanes, default_v[i] otherwise.
 */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
VECOPS_VFUNC V abs(V v, Mask<T> m, V default_v) {
  using namespace details;
  constexpr T t;
  return vmap(
      t, [=](auto tt, auto&& vv, auto&& mm, auto&& dd) { return word::abs(vv, mm, dd); },
      ShardVec(t, v), ShardMask(t, m), ShardVec(t, default_v)
  );
}

/**
 * @brief Masked absolute value with original value as default for masked-out lanes.
 *
 * For masked lanes, result[i] = |v[i]|.
 * For masked-out lanes, result[i] = v[i].
 */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
VECOPS_VFUNC V abs(V v, Mask<T> m) {
  return vec::abs(v, m, v);
}


/**
 * @brief Element-wise square root: result[i] = sqrt(v[i]).
 * @return Square root of the vector (floating-point only)
 */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
VECOPS_VFUNC V sqrt(V v) {
  using namespace details;
  constexpr T t;
  return vmap(
      t, [=](auto tt, auto&& vv) { return word::sqrt(vv); },
      ShardVec(t, v)
  );
}

/**
 * @brief Masked square root: result[i] = sqrt(v[i]) for masked lanes, default_v[i] otherwise.
 */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
VECOPS_VFUNC V sqrt(V v, Mask<T> m, V default_v) {
  using namespace details;
  constexpr T t;
  return vmap(
      t, [=](auto tt, auto&& vv, auto&& mm, auto&& dd) { return word::sqrt(vv, mm, dd); },
      ShardVec(t, v), ShardMask(t, m), ShardVec(t, default_v)
  );
}

/**
 * @brief Masked square root with original value as default for masked-out lanes.
 *
 * For masked lanes, result[i] = sqrt(v[i]).
 * For masked-out lanes, result[i] = v[i].
 */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
VECOPS_VFUNC V sqrt(V v, Mask<T> m) {
  return vec::sqrt(v, m, v);
}


/**
 * @brief Element-wise reciprocal square root: result[i] = 1/sqrt(v[i]).
 * @return Reciprocal square root of the vector (floating-point only)
 */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
VECOPS_VFUNC V rsqrt(V v) {
  using namespace details;
  constexpr T t;
  return vmap(
      t, [=](auto tt, auto&& vv) { return word::rsqrt(vv); },
      ShardVec(t, v)
  );
}

/**
 * @brief Masked reciprocal square root: result[i] = 1/sqrt(v[i]) for masked lanes.
 *
 * For masked-out lanes, result[i] = default_v[i].
 */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
VECOPS_VFUNC V rsqrt(V v, Mask<T> m, V default_v) {
  using namespace details;
  constexpr T t;
  return vmap(
      t, [=](auto tt, auto&& vv, auto&& mm, auto&& dd) { return word::rsqrt(vv, mm, dd); },
      ShardVec(t, v), ShardMask(t, m), ShardVec(t, default_v)
  );
}

/**
 * @brief Masked reciprocal square root with original value as default.
 *
 * For masked lanes, result[i] = 1/sqrt(v[i]).
 * For masked-out lanes, result[i] = v[i].
 */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
VECOPS_VFUNC V rsqrt(V v, Mask<T> m) {
  return vec::rsqrt(v, m, v);
}


/**
 * @brief Element-wise reciprocal: result[i] = 1/v[i].
 * @return Reciprocal of the vector (floating-point only)
 */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
VECOPS_VFUNC V rcp(V v) {
  using namespace details;
  constexpr T t;
  return vmap(
      t, [=](auto tt, auto&& vv) { return word::rcp(vv); },
      ShardVec(t, v)
  );
}

/**
 * @brief Masked reciprocal: result[i] = 1/v[i] for masked lanes.
 *
 * For masked-out lanes, result[i] = default_v[i].
 */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
VECOPS_VFUNC V rcp(V v, Mask<T> m, V default_v) {
  using namespace details;
  constexpr T t;
  return vmap(
      t, [=](auto tt, auto&& vv, auto&& mm, auto&& dd) { return word::rcp(vv, mm, dd); },
      ShardVec(t, v), ShardMask(t, m), ShardVec(t, default_v)
  );
}

/**
 * @brief Masked reciprocal with original value as default.
 *
 * For masked lanes, result[i] = 1/v[i].
 * For masked-out lanes, result[i] = v[i].
 */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
VECOPS_VFUNC V rcp(V v, Mask<T> m) {
  return vec::rcp(v, m, v);
}

/* ************************************************************************** */
//                           Transcendental math                              //
/* ************************************************************************** */

/**
 * @brief Computes the element-wise natural exponential with strict accuracy.
 *
 * Normal results are within one ULP of the destination type. The default build
 * may flush subnormal results to zero; defining `VECOPS_PRESERVE_SUBNORMALS`
 * enables gradual underflow with the same one-ULP bound. This is the default
 * choice for numerically sensitive workloads.
 */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(is_float<TypeOf<T>>)>
VECOPS_VFUNC V exp(V v) {
  using namespace details;
  constexpr T t;
  return vmap(t, [=](auto, auto&& vv) VECOPS_INLINE_LAMBDA { return word::exp(vv); }, ShardVec(t, v));
}

/**
 * @brief Computes strict element-wise exp on active lanes.
 *
 * Inactive lanes are copied from default_v without modification.
 */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(is_float<TypeOf<T>>)>
VECOPS_VFUNC V exp(V v, Mask<T> m, V default_v) {
  using namespace details;
  constexpr T t;
  return vmap(t, [=](auto, auto&& vv, auto&& mm, auto&& dd) VECOPS_INLINE_LAMBDA {
    return word::exp(vv, mm, dd);
  }, ShardVec(t, v), ShardMask(t, m), ShardVec(t, default_v));
}

/**
 * @brief Computes strict element-wise exp on active lanes.
 *
 * Inactive lanes retain their original input values.
 */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(is_float<TypeOf<T>>)>
VECOPS_VFUNC V exp(V v, Mask<T> m) {
  return vec::exp(v, m, v);
}

/**
 * @brief Computes a fast element-wise natural exponential.
 *
 * Normal results are within four ULP. Subnormal mathematical results may be
 * flushed to positive zero. If `VECOPS_MATH_ASSUME_VALID_INPUTS` is defined,
 * the input must not be NaN or infinity. Finite subnormal inputs are allowed.
 */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(is_float<TypeOf<T>>)>
VECOPS_VFUNC V exp_fast(V v) {
  using namespace details;
  constexpr T t;
  return vmap(t, [=](auto, auto&& vv) VECOPS_INLINE_LAMBDA { return word::exp_fast(vv); }, ShardVec(t, v));
}

/**
 * @brief Computes fast element-wise exp on active lanes.
 *
 * Inactive lanes are copied from default_v without modification.
 * With `VECOPS_MATH_ASSUME_VALID_INPUTS`, active inputs must not be NaN or
 * infinity. Finite subnormal inputs are allowed.
 */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(is_float<TypeOf<T>>)>
VECOPS_VFUNC V exp_fast(V v, Mask<T> m, V default_v) {
  using namespace details;
  constexpr T t;
  return vmap(t, [=](auto, auto&& vv, auto&& mm, auto&& dd) VECOPS_INLINE_LAMBDA {
    return word::exp_fast(vv, mm, dd);
  }, ShardVec(t, v), ShardMask(t, m), ShardVec(t, default_v));
}

/**
 * @brief Computes fast element-wise exp on active lanes.
 *
 * Inactive lanes retain their original input values.
 * With `VECOPS_MATH_ASSUME_VALID_INPUTS`, active inputs must not be NaN or
 * infinity. Finite subnormal inputs are allowed.
 */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(is_float<TypeOf<T>>)>
VECOPS_VFUNC V exp_fast(V v, Mask<T> m) {
  return vec::exp_fast(v, m, v);
}

/**
 * @brief Computes a maximum-throughput estimate of element-wise exp.
 *
 * Normal results are within four ULP of the correctly rounded destination
 * result or have at most 0.6 percent relative error. This quantization-aware
 * bound applies uniformly to every floating-point type. Subnormal
 * mathematical results may be flushed to positive zero. If
 * `VECOPS_MATH_ASSUME_VALID_INPUTS` is defined, the input must not be NaN or
 * infinity. Finite subnormal inputs are allowed.
 */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(is_float<TypeOf<T>>)>
VECOPS_VFUNC V exp_est(V v) {
  using namespace details;
  constexpr T t;
  return vmap(t, [=](auto, auto&& vv) VECOPS_INLINE_LAMBDA { return word::exp_est(vv); }, ShardVec(t, v));
}

/**
 * @brief Computes estimated element-wise exp on active lanes.
 *
 * Inactive lanes are copied from default_v without modification.
 * With `VECOPS_MATH_ASSUME_VALID_INPUTS`, active inputs must not be NaN or
 * infinity. Finite subnormal inputs are allowed.
 */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(is_float<TypeOf<T>>)>
VECOPS_VFUNC V exp_est(V v, Mask<T> m, V default_v) {
  using namespace details;
  constexpr T t;
  return vmap(t, [=](auto, auto&& vv, auto&& mm, auto&& dd) VECOPS_INLINE_LAMBDA {
    return word::exp_est(vv, mm, dd);
  }, ShardVec(t, v), ShardMask(t, m), ShardVec(t, default_v));
}

/**
 * @brief Computes estimated element-wise exp on active lanes.
 *
 * Inactive lanes retain their original input values.
 * With `VECOPS_MATH_ASSUME_VALID_INPUTS`, active inputs must not be NaN or
 * infinity. Finite subnormal inputs are allowed.
 */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(is_float<TypeOf<T>>)>
VECOPS_VFUNC V exp_est(V v, Mask<T> m) {
  return vec::exp_est(v, m, v);
}

/**
 * @brief Computes strict element-wise exp for inputs known to be non-positive.
 *
 * Every lane must satisfy `v[i] <= 0`; results for positive values and NaNs are
 * undefined. Negative infinity is supported unless
 * `VECOPS_MATH_ASSUME_VALID_INPUTS` is defined. Normal results are within one
 * ULP, and `VECOPS_PRESERVE_SUBNORMALS` enables the strict gradual-underflow
 * guarantee.
 */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(is_float<TypeOf<T>>)>
VECOPS_VFUNC V exp_neg(V v) {
  using namespace details;
  constexpr T t;
  return vmap(t, [=](auto, auto&& vv) VECOPS_INLINE_LAMBDA { return word::exp_neg(vv); }, ShardVec(t, v));
}

/**
 * @brief Computes strict non-positive-domain exp on active lanes.
 *
 * Active lanes must be non-positive and not NaN. Inactive lanes are copied
 * from `default_v` without modification.
 */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(is_float<TypeOf<T>>)>
VECOPS_VFUNC V exp_neg(V v, Mask<T> m, V default_v) {
  using namespace details;
  constexpr T t;
  return vmap(t, [=](auto, auto&& vv, auto&& mm, auto&& dd) VECOPS_INLINE_LAMBDA {
    return word::exp_neg(vv, mm, dd);
  }, ShardVec(t, v), ShardMask(t, m), ShardVec(t, default_v));
}

/**
 * @brief Computes strict non-positive-domain exp on active lanes.
 *
 * Active lanes must be non-positive and not NaN. Inactive lanes retain their
 * original input values.
 */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(is_float<TypeOf<T>>)>
VECOPS_VFUNC V exp_neg(V v, Mask<T> m) {
  return vec::exp_neg(v, m, v);
}

/**
 * @brief Computes fast element-wise exp for inputs known to be non-positive.
 *
 * Every lane must satisfy `v[i] <= 0`; positive values and NaNs have undefined
 * results. Normal results are within four ULP. Subnormal mathematical results
 * may be flushed to positive zero.
 */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(is_float<TypeOf<T>>)>
VECOPS_VFUNC V exp_neg_fast(V v) {
  using namespace details;
  constexpr T t;
  return vmap(t, [=](auto, auto&& vv) VECOPS_INLINE_LAMBDA { return word::exp_neg_fast(vv); }, ShardVec(t, v));
}

/**
 * @brief Computes fast non-positive-domain exp on active lanes.
 *
 * Active lanes must be non-positive and not NaN. Inactive lanes are copied
 * from `default_v` without modification.
 */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(is_float<TypeOf<T>>)>
VECOPS_VFUNC V exp_neg_fast(V v, Mask<T> m, V default_v) {
  using namespace details;
  constexpr T t;
  return vmap(t, [=](auto, auto&& vv, auto&& mm, auto&& dd) VECOPS_INLINE_LAMBDA {
    return word::exp_neg_fast(vv, mm, dd);
  }, ShardVec(t, v), ShardMask(t, m), ShardVec(t, default_v));
}

/**
 * @brief Computes fast non-positive-domain exp on active lanes.
 *
 * Active lanes must be non-positive and not NaN. Inactive lanes retain their
 * original input values.
 */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(is_float<TypeOf<T>>)>
VECOPS_VFUNC V exp_neg_fast(V v, Mask<T> m) {
  return vec::exp_neg_fast(v, m, v);
}

/**
 * @brief Estimates element-wise exp for inputs known to be non-positive.
 *
 * Every lane must satisfy `v[i] <= 0`; positive values and NaNs have undefined
 * results. Normal results are within four ULP of the correctly rounded
 * destination result or have at most 0.6 percent relative error. This
 * quantization-aware bound applies uniformly to every floating-point type,
 * and subnormal mathematical results may be flushed to positive zero.
 */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(is_float<TypeOf<T>>)>
VECOPS_VFUNC V exp_neg_est(V v) {
  using namespace details;
  constexpr T t;
  return vmap(t, [=](auto, auto&& vv) VECOPS_INLINE_LAMBDA { return word::exp_neg_est(vv); }, ShardVec(t, v));
}

/**
 * @brief Estimates non-positive-domain exp on active lanes.
 *
 * Active lanes must be non-positive and not NaN. Inactive lanes are copied
 * from `default_v` without modification.
 */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(is_float<TypeOf<T>>)>
VECOPS_VFUNC V exp_neg_est(V v, Mask<T> m, V default_v) {
  using namespace details;
  constexpr T t;
  return vmap(t, [=](auto, auto&& vv, auto&& mm, auto&& dd) VECOPS_INLINE_LAMBDA {
    return word::exp_neg_est(vv, mm, dd);
  }, ShardVec(t, v), ShardMask(t, m), ShardVec(t, default_v));
}

/**
 * @brief Estimates non-positive-domain exp on active lanes.
 *
 * Active lanes must be non-positive and not NaN. Inactive lanes retain their
 * original input values.
 */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(is_float<TypeOf<T>>)>
VECOPS_VFUNC V exp_neg_est(V v, Mask<T> m) {
  return vec::exp_neg_est(v, m, v);
}

/* ************************************************************************** */
//                                Comparison                                  //
/* ************************************************************************** */

/**
 * @brief Element-wise equality comparison: result[i] = (a[i] == b[i]).
 * @return Mask where lanes are true if elements are equal
 */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
VECOPS_VFUNC Mask<T> cmpeq(V a, V b) {
  using namespace details;
  constexpr T t;
  return vmap(
      t, [=](auto tt, auto&& aa, auto&& bb) { return word::cmpeq(aa, bb); },
      ShardVec(t, a), ShardVec(t, b)
  );
}

/**
 * @brief Masked equality comparison: only compare lanes where mask is true.
 *
 * For masked-out lanes, result[i] = false.
 *
 * @return Mask where lanes are true if elements are equal and mask is true
 */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
VECOPS_VFUNC Mask<T> cmpeq(V a, V b, Mask<T> m) {
  using namespace details;
  constexpr T t;
  return vmap(
      t, [=](auto tt, auto&& aa, auto&& bb, auto&& mm) { return word::cmpeq(aa, bb, mm); },
      ShardVec(t, a), ShardVec(t, b), ShardMask(t, m)
  );
}


/**
 * @brief Element-wise inequality comparison: result[i] = (a[i] != b[i]).
 * @return Mask where lanes are true if elements are not equal
 */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
VECOPS_VFUNC Mask<T> cmpne(V a, V b) {
  using namespace details;
  constexpr T t;
  return vmap(
      t, [=](auto tt, auto&& aa, auto&& bb) { return word::cmpne(aa, bb); },
      ShardVec(t, a), ShardVec(t, b)
  );
}

/**
 * @brief Masked inequality comparison: only compare lanes where mask is true.
 *
 * For masked-out lanes, result[i] = false.
 */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
VECOPS_VFUNC Mask<T> cmpne(V a, V b, Mask<T> m) {
  using namespace details;
  constexpr T t;
  return vmap(
      t, [=](auto tt, auto&& aa, auto&& bb, auto&& mm) { return word::cmpne(aa, bb, mm); },
      ShardVec(t, a), ShardVec(t, b), ShardMask(t, m)
  );
}


/**
 * @brief Element-wise less-than comparison: result[i] = (a[i] < b[i]).
 * @return Mask where lanes are true if a[i] < b[i]
 */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
VECOPS_VFUNC Mask<T> cmplt(V a, V b) {
  using namespace details;
  constexpr T t;
  return vmap(
      t, [=](auto tt, auto&& aa, auto&& bb) { return word::cmplt(aa, bb); },
      ShardVec(t, a), ShardVec(t, b)
  );
}

/**
 * @brief Masked less-than comparison: only compare lanes where mask is true.
 *
 * For masked-out lanes, result[i] = false.
 */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
VECOPS_VFUNC Mask<T> cmplt(V a, V b, Mask<T> m) {
  using namespace details;
  constexpr T t;
  return vmap(
      t, [=](auto tt, auto&& aa, auto&& bb, auto&& mm) { return word::cmplt(aa, bb, mm); },
      ShardVec(t, a), ShardVec(t, b), ShardMask(t, m)
  );
}


/**
 * @brief Element-wise greater-than comparison: result[i] = (a[i] > b[i]).
 * @return Mask where lanes are true if a[i] > b[i]
 */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
VECOPS_VFUNC Mask<T> cmpgt(V a, V b) {
  using namespace details;
  constexpr T t;
  return vmap(
      t, [=](auto tt, auto&& aa, auto&& bb) { return word::cmpgt(aa, bb); },
      ShardVec(t, a), ShardVec(t, b)
  );
}

/**
 * @brief Masked greater-than comparison: only compare lanes where mask is true.
 *
 * For masked-out lanes, result[i] = false.
 */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
VECOPS_VFUNC Mask<T> cmpgt(V a, V b, Mask<T> m) {
  using namespace details;
  constexpr T t;
  return vmap(
      t, [=](auto tt, auto&& aa, auto&& bb, auto&& mm) { return word::cmpgt(aa, bb, mm); },
      ShardVec(t, a), ShardVec(t, b), ShardMask(t, m)
  );
}


/**
 * @brief Element-wise less-than-or-equal comparison: result[i] = (a[i] <= b[i]).
 * @return Mask where lanes are true if a[i] <= b[i]
 */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
VECOPS_VFUNC Mask<T> cmple(V a, V b) {
  using namespace details;
  constexpr T t;
  return vmap(
      t, [=](auto tt, auto&& aa, auto&& bb) { return word::cmple(aa, bb); },
      ShardVec(t, a), ShardVec(t, b)
  );
}

/**
 * @brief Masked less-than-or-equal comparison: only compare lanes where mask is true.
 *
 * For masked-out lanes, result[i] = false.
 */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
VECOPS_VFUNC Mask<T> cmple(V a, V b, Mask<T> m) {
  using namespace details;
  constexpr T t;
  return vmap(
      t, [=](auto tt, auto&& aa, auto&& bb, auto&& mm) { return word::cmple(aa, bb, mm); },
      ShardVec(t, a), ShardVec(t, b), ShardMask(t, m)
  );
}


/**
 * @brief Element-wise greater-than-or-equal comparison: result[i] = (a[i] >= b[i]).
 * @return Mask where lanes are true if a[i] >= b[i]
 */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
VECOPS_VFUNC Mask<T> cmpge(V a, V b) {
  using namespace details;
  constexpr T t;
  return vmap(
      t, [=](auto tt, auto&& aa, auto&& bb) { return word::cmpge(aa, bb); },
      ShardVec(t, a), ShardVec(t, b)
  );
}

/**
 * @brief Masked greater-than-or-equal comparison: only compare lanes where mask is true.
 *
 * For masked-out lanes, result[i] = false.
 */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
VECOPS_VFUNC Mask<T> cmpge(V a, V b, Mask<T> m) {
  using namespace details;
  constexpr T t;
  return vmap(
      t, [=](auto tt, auto&& aa, auto&& bb, auto&& mm) { return word::cmpge(aa, bb, mm); },
      ShardVec(t, a), ShardVec(t, b), ShardMask(t, m)
  );
}


/**
 * @brief Check if elements are NaN: result[i] = isnan(v[i]).
 * @return Mask where lanes are true if elements are NaN (floating-point only)
 */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
VECOPS_VFUNC Mask<T> isnan(V v) {
  using namespace details;
  constexpr T t;
  return vmap(
      t, [=](auto tt, auto&& vv) { return word::isnan(vv); },
      ShardVec(t, v)
  );
}

/**
 * @brief Masked NaN check: only check lanes where mask is true.
 *
 * For masked-out lanes, result[i] = false.
 */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
VECOPS_VFUNC Mask<T> isnan(V v, Mask<T> m) {
  using namespace details;
  constexpr T t;
  return vmap(
      t, [=](auto tt, auto&& vv, auto&& mm) { return word::isnan(vv, mm); },
      ShardVec(t, v), ShardMask(t, m)
  );
}


/**
 * @brief Check if elements are positive infinity: result[i] = (v[i] == +inf).
 * @return Mask where lanes are true if elements are positive infinity
 */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
VECOPS_VFUNC Mask<T> isposinf(V v) {
  using namespace details;
  constexpr T t;
  return vmap(
      t, [=](auto tt, auto&& vv) { return word::isposinf(vv); },
      ShardVec(t, v)
  );
}

/**
 * @brief Masked positive infinity check: only check lanes where mask is true.
 *
 * For masked-out lanes, result[i] = false.
 */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
VECOPS_VFUNC Mask<T> isposinf(V v, Mask<T> m) {
  using namespace details;
  constexpr T t;
  return vmap(
      t, [=](auto tt, auto&& vv, auto&& mm) { return word::isposinf(vv, mm); },
      ShardVec(t, v), ShardMask(t, m)
  );
}


/**
 * @brief Check if elements are negative infinity: result[i] = (v[i] == -inf).
 * @return Mask where lanes are true if elements are negative infinity
 */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
VECOPS_VFUNC Mask<T> isneginf(V v) {
  using namespace details;
  constexpr T t;
  return vmap(
      t, [=](auto tt, auto&& vv) { return word::isneginf(vv); },
      ShardVec(t, v)
  );
}

/**
 * @brief Masked negative infinity check: only check lanes where mask is true.
 *
 * For masked-out lanes, result[i] = false.
 */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
VECOPS_VFUNC Mask<T> isneginf(V v, Mask<T> m) {
  using namespace details;
  constexpr T t;
  return vmap(
      t, [=](auto tt, auto&& vv, auto&& mm) { return word::isneginf(vv, mm); },
      ShardVec(t, v), ShardMask(t, m)
  );
}


/**
 * @brief Check if elements are infinity (positive or negative): result[i] = isinf(v[i]).
 * @return Mask where lanes are true if elements are infinity
 */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
VECOPS_VFUNC Mask<T> isinf(V v) {
  using namespace details;
  constexpr T t;
  return vmap(
      t, [=](auto tt, auto&& vv) { return word::isinf(vv); },
      ShardVec(t, v)
  );
}

/**
 * @brief Masked infinity check: only check lanes where mask is true.
 *
 * For masked-out lanes, result[i] = false.
 */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
VECOPS_VFUNC Mask<T> isinf(V v, Mask<T> m) {
  using namespace details;
  constexpr T t;
  return vmap(
      t, [=](auto tt, auto&& vv, auto&& mm) { return word::isinf(vv, mm); },
      ShardVec(t, v), ShardMask(t, m)
  );
}


/* ************************************************************************** */
//                           Shuffle & Permutation                            //
/* ************************************************************************** */

/**
 * @brief Shuffle elements within each 16-byte block using compile-time indices.
 *
 * Permutes elements within each 16-byte lane independently. All lanes apply
 * the same shuffle pattern. For float32_t (4 elements per block), indices
 * must be in [0, 3].
 *
 * Result: result[j] = v[Is[j % M] + floor(j / M)]
 * where M = 16 / sizeof(element_type), j = 0...N-1.
 *
 * @tparam Is Compile-time shuffle indices
 * @return Shuffled vector
 *
 * @note Vectors smaller than word_size are treated as word_size vectors.
 */
template <int... Is, TLV_DECL_VEC(V)>
VECOPS_VFUNC V local_shuf(V v) {
  using namespace details;
  constexpr Vec2Tag<V> t;
  return vmap(
      t, [=](auto tt, auto&& vv) { return word::local_shuf<Is...>(vv); },
      ShardVec(t, v)
  );
}

/**
 * @brief Shuffle elements within each 16-byte block using runtime indices.
 *
 * Permutes elements within each 16-byte lane independently. Each lane can
 * use different indices from the index vector. Index vector element type
 * must be a signed integer with same width as the data element type.
 *
 * Result: result[j] = v[i[j] + floor(j / M)]
 * where M = 16 / sizeof(element_type), j = 0...N-1.
 * Undefined if i[j] not in [0, M).
 *
 * @param v Input vector
 * @param i Index vector (signed integer, same bit-width as v)
 * @return Shuffled vector
 */
template <TLV_DECL_VEC(V), TLV_DECL_VEC(Vi)>
VECOPS_VFUNC V local_shuf(V v, Vi i) {
  using namespace details;
  constexpr Vec2Tag<V> t;
  constexpr Vec2Tag<Vi> ti;
  return vmap(
      t, [=](auto tt, auto&& vv, auto&& ii) { return word::local_shuf(vv, ii); },
      ShardVec(t, v), ShardVec(ti, i)
  );
}

/**
 * @brief Shuffle elements within each 16-byte block using runtime integer indices.
 *
 * Same as the compile-time index version, but accepts runtime values.
 * All lanes apply the same shuffle pattern.
 *
 * @param v Input vector
 * @param is Runtime shuffle indices (one per element in block)
 * @return Shuffled vector
 */
template <TLV_DECL_VEC(V), typename... Is, TL_IF(is_any<Is, int> && ...)>
VECOPS_VFUNC V local_shuf(V v, Is... is) {
  using namespace details;
  constexpr Vec2Tag<V> t;
  return vmap(
      t, [=](auto tt, auto&& vv) { return word::local_shuf(vv, is...); },
      ShardVec(t, v)
  );
}

//template <int... Is, TLV_DECL_VEC(V)>
//V block_shuf(V v) {
//  using namespace details;
//  constexpr Vec2Tag<V> t;
//  return vmap(
//      t, [=](auto tt, auto&& vv) { return word::block_shuf<Is...>(vv); },
//      ShardVec(t, v)
//  );
//}
//
//template <TLV_DECL_VEC(V), typename... Is, TL_IF(is_any<Is, int> && ...)>
//V block_shuf(V v, Is... is) {
//  using namespace details;
//  constexpr Vec2Tag<V> t;
//  return vmap(
//      t, [=](auto tt, auto&& vv) { return word::block_shuf(vv, is...); },
//      ShardVec(t, v)
//  );
//}

/**
 * @brief Shuffle elements across the entire vector using an index vector.
 *
 * Permutes elements across the whole vector (word). Each element can be
 * selected from any position in the input. Index vector element type must
 * be a signed integer with same width as the data element type.
 *
 * Result: result[j] = v[i[j]], where j = 0...N-1.
 * Undefined if i[j] not in [0, N).
 *
 * @param v Input vector
 * @param i Index vector (signed integer, same bit-width as v)
 * @return Shuffled vector
 *
 * @note On x86 AVX2+, this involves cross-lane data movement and is slower
 *       than local_shuf. Without AVX512, performance is further reduced.
 *       For int8_t/uint8_t without AVX512_VBMI, this is slower than wider types.
 */
template <TLV_DECL_VEC(V), TLV_DECL_VEC(Vi)>
VECOPS_VFUNC V shuf(V v, Vi i) {
  using namespace details;
  constexpr Vec2Tag<V> t;
  constexpr Vec2Tag<Vi> ti;
  return vmap(
      t, [=](auto tt, auto&& vv, auto&& ii) { return word::shuf(vv, ii); },
      ShardVec(t, v), ShardVec(ti, i)
  );
}

/**
 * @brief Extract the upper half of a vector.
 *
 * Returns a vector of half the length containing the upper (high-indexed)
 * elements of the input vector.
 *
 * For a vector v of size N: result[i] = v[i + N/2]
 *
 * @tparam T Input vector tag type
 * @tparam V Output vector type (Vec<Half<T>>)
 * @param t Input vector tag
 * @param v Input vector of size N
 * @return Vector of size N/2 containing upper half elements
 *
 * @note For multi-word vectors, this extracts words from the upper half
 *       without additional processing. For single-word vectors, delegates
 *       to the architecture-specific word::upper implementation.
 *
 * @example
 *   Tag<float32_t, 8> t;
 *   auto v = load(t, {0, 1, 2, 3, 4, 5, 6, 7});
 *   auto u = upper(t, v);  // u = [4, 5, 6, 7]
 */
template <TLV_DECL_TAG(T), typename V = Vec<Half<T>>>
VECOPS_VFUNC V upper(T t, Vec<T> v) {
  using namespace details;
  constexpr nint_t NWi = num_words(t);
  if constexpr (NWi > 1) {
    static_assert(NWi == 2 * num_words(Vec2Tag<V>{}));
    return vmap(Half<T>{}, [&]<nint_t I>(auto tt) { return get_word<I + NWi / 2>(t, v); });
  } else {
    return word::upper(t, v);
  }
}

/**
 * @brief Extract the lower half of a vector.
 *
 * Returns a vector of half the length containing the lower (low-indexed)
 * elements of the input vector.
 *
 * For a vector v of size N: result[i] = v[i]
 *
 * @tparam T Input vector tag type
 * @tparam V Output vector type (Vec<Half<T>>)
 * @param t Input vector tag
 * @param v Input vector of size N
 * @return Vector of size N/2 containing lower half elements
 *
 * @note For multi-word vectors, this extracts words from the lower half
 *       without additional processing. For single-word vectors, delegates
 *       to the architecture-specific word::lower implementation.
 *
 * @example
 *   Tag<float32_t, 8> t;
 *   auto v = load(t, {0, 1, 2, 3, 4, 5, 6, 7});
 *   auto u = lower(t, v);  // u = [0, 1, 2, 3]
 */
template <TLV_DECL_TAG(T), typename V = Vec<Half<T>>>
VECOPS_VFUNC V lower(T t, Vec<T> v) {
  using namespace details;
  constexpr nint_t NWi = num_words(t);
  if constexpr (NWi > 1) {
    static_assert(NWi == 2 * num_words(Vec2Tag<V>{}));
    return vmap(Half<T>{}, [&]<nint_t I>(auto tt) { return get_word<I>(t, v); });
  } else {
    return word::lower(t, v);
  }
}

/**
 * @brief Extract elements at even indices from a vector.
 *
 * Returns a vector of half the length containing elements at even positions
 * (indices 0, 2, 4, ...) from the input vector.
 *
 * For a vector v of size N: result[i] = v[2*i]
 *
 * @tparam T Input vector tag type
 * @tparam V Output vector type (Vec<Half<T>>)
 * @param t Input vector tag
 * @param v Input vector of size N
 * @return Vector of size N/2 containing even-indexed elements
 *
 * @pre Input vector must have at least 2 elements.
 *
 * @note For multi-word vectors, this processes pairs of adjacent words
 *       together to extract even elements. For single-word vectors,
 *       delegates to the architecture-specific word::even implementation.
 *
 * @example
 *   Tag<float32_t, 8> t;
 *   auto v = load(t, {0, 1, 2, 3, 4, 5, 6, 7});
 *   auto u = even(t, v);  // u = [0, 2, 4, 6]
 */
template <TLV_DECL_TAG(T), typename V = Vec<Half<T>>>
VECOPS_VFUNC V even(T t, Vec<T> v) {
  using namespace details;
  static_assert(size(t) >= 2, "Insufficient elements");
  constexpr nint_t NWi = num_words(t);
  if constexpr (NWi > 1) {
    static_assert(NWi == 2 * num_words(Vec2Tag<V>{}));
    constexpr Twice<WordOf<T>> t2;
    V v_o;
    foreach<NWi / 2>([&]<nint_t I>{
      auto u = word::even(t2, Vec<decltype(t2)>{get_word<2 * I>(t, v), get_word<2 * I + 1>(t, v)});
      v_o = set_word<I>(Half<T>{}, v_o, u);
    });
    return v_o;
  } else {
    return word::even(t, v);
  }
}

/**
 * @brief Extract elements at odd indices from a vector.
 *
 * Returns a vector of half the length containing elements at odd positions
 * (indices 1, 3, 5, ...) from the input vector.
 *
 * For a vector v of size N: result[i] = v[2*i + 1]
 *
 * @tparam T Input vector tag type
 * @tparam V Output vector type (Vec<Half<T>>)
 * @param t Input vector tag
 * @param v Input vector of size N
 * @return Vector of size N/2 containing odd-indexed elements
 *
 * @pre Input vector must have at least 2 elements.
 *
 * @note For multi-word vectors, this processes pairs of adjacent words
 *       together to extract odd elements. For single-word vectors,
 *       delegates to the architecture-specific word::odd implementation.
 *
 * @example
 *   Tag<float32_t, 8> t;
 *   auto v = load(t, {0, 1, 2, 3, 4, 5, 6, 7});
 *   auto u = odd(t, v);  // u = [1, 3, 5, 7]
 */
template <TLV_DECL_TAG(T), typename V = Vec<Half<T>>>
VECOPS_VFUNC V odd(T t, Vec<T> v) {
  using namespace details;
  static_assert(size(t) >= 2, "Insufficient elements");
  constexpr nint_t NWi = num_words(t);
  if constexpr (NWi > 1) {
    static_assert(NWi == 2 * num_words(Vec2Tag<V>{}));
    constexpr Twice<WordOf<T>> t2;
    V v_o;
    foreach<NWi / 2>([&]<nint_t I>{
      auto u = word::odd(t2, Vec<decltype(t2)>{get_word<2 * I>(t, v), get_word<2 * I + 1>(t, v)});
      v_o = set_word<I>(Half<T>{}, v_o, u);
    });
    return v_o;
  } else {
    return word::odd(t, v);
  }
}

/**
 * @brief Concatenate two half-length vectors into a full-length vector.
 *
 * Combines two vectors of half the target size into a single vector of
 * the target size. The lower vector's elements occupy the lower indices,
 * and the higher vector's elements occupy the upper indices.
 *
 * Result: result[i] = v_lo[i] for i < N/2, result[i] = v_hi[i - N/2] for i >= N/2
 *
 * @tparam T Output vector tag type
 * @tparam V Input vector type (Vec<Half<T>>)
 * @param t Output vector tag
 * @param v_lo Lower half vector (placed at lower indices)
 * @param v_hi Upper half vector (placed at upper indices)
 * @return Concatenated vector of size N
 *
 * @note This is the inverse operation of upper() and lower() combined:
 *       concat(t, lower(t, v), upper(t, v)) == v
 *
 * @example
 *   Tag<float32_t, 8> t;
 *   Tag<float32_t, 4> th;
 *   auto lo = load(th, {0, 1, 2, 3});
 *   auto hi = load(th, {4, 5, 6, 7});
 *   auto v = concat(t, lo, hi);  // v = [0, 1, 2, 3, 4, 5, 6, 7]
 */
template <TLV_DECL_TAG(T), typename V = Vec<Half<T>>, TL_IF(is_vec<V>)>
VECOPS_VFUNC Vec<T> concat(T t, V v_lo, V v_hi) {
  using namespace details;
  using Ti = Vec2Tag<V>;
  constexpr nint_t NWo = num_words(t);
  if constexpr (NWo > 1) {
    static_assert(NWo == 2 * num_words(Ti{}));
    Vec<T> v_o;
    foreach<NWo / 2>([&]<nint_t I>{
      v_o = set_word<I>(t, v_o, get_word<I>(Ti(), v_lo));
    });
    foreach<NWo / 2>([&]<nint_t I>{
      v_o = set_word<I + NWo / 2>(t, v_o, get_word<I>(Ti(), v_hi));
    });
    return v_o;
  } else {
    return word::concat(t, v_lo, v_hi);
  }
}

/* ********************************************************************** */
//                     Mask Lower / Upper / Concat                          //
/* ********************************************************************** */

/**
 * @brief Extract the lower half of a mask.
 *
 * Returns a mask of half the length containing the lower (low-indexed)
 * predicate bits of the input mask.
 *
 * For a mask m of size N: result[i] = m[i]
 *
 * @tparam T Input tag type
 * @param t Input tag
 * @param m Input mask of size N
 * @return Mask of size N/2 containing lower half bits
 */
template <TLV_DECL_TAG(T), typename M = Mask<Half<T>>>
VECOPS_VFUNC M lower(T t, Mask<T> m) {
  using namespace details;
  constexpr nint_t NWi = num_words(t);
  if constexpr (NWi > 1) {
    static_assert(NWi == 2 * num_words(Half<T>{}));
    M r;
    constexpr nint_t NWh = NWi / 2;
    foreach<NWh>([&]<nint_t I>{
      r = set_word_mask<I>(Half<T>{}, r, get_word_mask<I>(t, m));
    });
    return r;
  } else {
    return word::lower(t, m);
  }
}

/**
 * @brief Extract the upper half of a mask.
 *
 * Returns a mask of half the length containing the upper (high-indexed)
 * predicate bits of the input mask.
 *
 * For a mask m of size N: result[i] = m[i + N/2]
 *
 * @tparam T Input tag type
 * @param t Input tag
 * @param m Input mask of size N
 * @return Mask of size N/2 containing upper half bits
 */
template <TLV_DECL_TAG(T), typename M = Mask<Half<T>>>
VECOPS_VFUNC M upper(T t, Mask<T> m) {
  using namespace details;
  constexpr nint_t NWi = num_words(t);
  if constexpr (NWi > 1) {
    static_assert(NWi == 2 * num_words(Half<T>{}));
    M r;
    constexpr nint_t NWh = NWi / 2;
    foreach<NWh>([&]<nint_t I>{
      r = set_word_mask<I>(Half<T>{}, r, get_word_mask<I + NWh>(t, m));
    });
    return r;
  } else {
    return word::upper(t, m);
  }
}

/**
 * @brief Concatenate two half-length masks into a full-length mask.
 *
 * Combines two masks of half the target size into a single mask of
 * the target size. The lower mask's bits occupy the lower indices,
 * and the higher mask's bits occupy the upper indices.
 *
 * Result: result[i] = m_lo[i] for i < N/2, result[i] = m_hi[i - N/2] for i >= N/2
 *
 * @tparam T Output tag type
 * @tparam V Input mask type (Mask<Half<T>>)
 * @param t Output tag
 * @param m_lo Lower half mask (placed at lower indices)
 * @param m_hi Upper half mask (placed at upper indices)
 * @return Concatenated mask of size N
 *
 * @note This is the inverse operation of upper() and lower() combined:
 *       concat(t, lower(t, m), upper(t, m)) == m
 */
template <TLV_DECL_TAG(T), typename V = Mask<Half<T>>, TL_IF(is_mask<V>)>
VECOPS_VFUNC Mask<T> concat(T t, V m_lo, V m_hi) {
  using namespace details;
  constexpr nint_t NWo = num_words(t);
  if constexpr (NWo > 1) {
    static_assert(NWo == 2 * num_words(Half<T>{}));
    Mask<T> r;
    constexpr nint_t NWh = NWo / 2;
    Half<T> th;
    foreach<NWh>([&]<nint_t I>{
      r = set_word_mask<I>(t, r, get_word_mask<I>(th, m_lo));
    });
    foreach<NWh>([&]<nint_t I>{
      r = set_word_mask<I + NWh>(t, r, get_word_mask<I>(th, m_hi));
    });
    return r;
  } else {
    return word::concat(t, m_lo, m_hi);
  }
}

/* ********************************************************************** */
//                             Reductions
/* ********************************************************************** */

template <TLV_DECL_TAG(T)>
VECOPS_VFUNC TypeOf<T> reduce_add(T t, Vec<T> v) {
  if constexpr (is_word_vec(t)) {
    return word::reduce_add(t, v);
  } else {
    constexpr Half<T> th;
    return vec::reduce_add(
        th, vec::add(vec::lower(t, v), vec::upper(t, v)));
  }
}

template <TLV_DECL_TAG(T)>
VECOPS_VFUNC TypeOf<T> reduce_add(T t, Vec<T> v, Mask<T> m) {
  if constexpr (is_word_vec(t)) {
    return word::reduce_add(t, v, m);
  } else {
    constexpr Half<T> th;
    const auto m_lo = vec::lower(t, m);
    const auto m_hi = vec::upper(t, m);
    const auto both = vec::bit_and(th, m_lo, m_hi);
    const auto either = vec::bit_or(th, m_lo, m_hi);
    const auto v_lo = vec::lower(t, v);
    const auto v_hi = vec::upper(t, v);
    auto folded = vec::blend(v_hi, m_lo, v_lo);
    folded = vec::blend(
        folded, both, vec::add(v_lo, v_hi));
    return vec::reduce_add(th, folded, either);
  }
}

template <TLV_DECL_TAG(T)>
VECOPS_VFUNC TypeOf<T> reduce_max(T t, Vec<T> v) {
  if constexpr (is_word_vec(t)) {
    return word::reduce_max(t, v);
  } else {
    constexpr Half<T> th;
    return vec::reduce_max(
        th, vec::max(vec::lower(t, v), vec::upper(t, v)));
  }
}

template <TLV_DECL_TAG(T)>
VECOPS_VFUNC TypeOf<T> reduce_max(T t, Vec<T> v, Mask<T> m) {
  if constexpr (is_word_vec(t)) {
    return word::reduce_max(t, v, m);
  } else {
    constexpr Half<T> th;
    const auto m_lo = vec::lower(t, m);
    const auto m_hi = vec::upper(t, m);
    const auto both = vec::bit_and(th, m_lo, m_hi);
    const auto either = vec::bit_or(th, m_lo, m_hi);
    const auto v_lo = vec::lower(t, v);
    const auto v_hi = vec::upper(t, v);
    auto folded = vec::blend(v_hi, m_lo, v_lo);
    folded = vec::blend(
        folded, both, vec::max(v_lo, v_hi));
    return vec::reduce_max(th, folded, either);
  }
}

template <TLV_DECL_TAG(T)>
VECOPS_VFUNC TypeOf<T> reduce_min(T t, Vec<T> v) {
  if constexpr (is_word_vec(t)) {
    return word::reduce_min(t, v);
  } else {
    constexpr Half<T> th;
    return vec::reduce_min(
        th, vec::min(vec::lower(t, v), vec::upper(t, v)));
  }
}

template <TLV_DECL_TAG(T)>
VECOPS_VFUNC TypeOf<T> reduce_min(T t, Vec<T> v, Mask<T> m) {
  if constexpr (is_word_vec(t)) {
    return word::reduce_min(t, v, m);
  } else {
    constexpr Half<T> th;
    const auto m_lo = vec::lower(t, m);
    const auto m_hi = vec::upper(t, m);
    const auto both = vec::bit_and(th, m_lo, m_hi);
    const auto either = vec::bit_or(th, m_lo, m_hi);
    const auto v_lo = vec::lower(t, v);
    const auto v_hi = vec::upper(t, v);
    auto folded = vec::blend(v_hi, m_lo, v_lo);
    folded = vec::blend(
        folded, both, vec::min(v_lo, v_hi));
    return vec::reduce_min(th, folded, either);
  }
}

/**
 * @brief Extract even-indexed elements from two vectors and concatenate.
 *
 * Extracts elements at even indices from both input vectors and concatenates
 * them into a single output vector. The result has the same total element
 * count as each input vector.
 *
 * For input vectors v_lo and v_hi of size N:
 *   result[i] = v_lo[2*i]     for i < N/2
 *   result[i] = v_hi[2*(i-N/2)] for i >= N/2
 *
 * @tparam T Vector tag type
 * @param t Vector tag
 * @param v_lo First input vector (lower half of result comes from its even elements)
 * @param v_hi Second input vector (upper half of result comes from its even elements)
 * @return Vector of size N containing even elements from both inputs
 *
 * @note This operation is commonly used after local_interleave_lower to undo
 *       the interleaving. Combined with concat_odd, these two operations can
 *       deinterleave a pair of interleaved vectors.
 *
 * @example
 *   Tag<float32_t, 8> t;
 *   auto lo = load(t, {0, 1, 2, 3, 4, 5, 6, 7});
 *   auto hi = load(t, {8, 9, 10, 11, 12, 13, 14, 15});
 *   auto v = concat_even(t, lo, hi);  // v = [0, 2, 4, 6, 8, 10, 12, 14]
 */
template <TLV_DECL_TAG(T)>
VECOPS_VFUNC Vec<T> concat_even(T t, Vec<T> v_lo, Vec<T> v_hi) {
  using namespace details;
  constexpr nint_t NWo = num_words(t);
  if constexpr (NWo > 1) {
    Half<T> th;
    auto u_lo = vec::even(t, v_lo);
    auto u_hi = vec::even(t, v_hi);
    Vec<T> v_o;
    foreach<NWo / 2>([&]<nint_t I>{
      v_o = set_word<I>(t, v_o, get_word<I>(th, u_lo));
    });
    foreach<NWo / 2>([&]<nint_t I>{
      v_o = set_word<NWo / 2 + I>(t, v_o, get_word<I>(th, u_hi));
    });
    return v_o;
  } else {
    return word::concat_even(t, v_lo, v_hi);
  }
}

/**
 * @brief Extract odd-indexed elements from two vectors and concatenate.
 *
 * Extracts elements at odd indices from both input vectors and concatenates
 * them into a single output vector. The result has the same total element
 * count as each input vector.
 *
 * For input vectors v_lo and v_hi of size N:
 *   result[i] = v_lo[2*i + 1]     for i < N/2
 *   result[i] = v_hi[2*(i-N/2) + 1] for i >= N/2
 *
 * @tparam T Vector tag type
 * @param t Vector tag
 * @param v_lo First input vector (lower half of result comes from its odd elements)
 * @param v_hi Second input vector (upper half of result comes from its odd elements)
 * @return Vector of size N containing odd elements from both inputs
 *
 * @note This operation is commonly used after local_interleave_upper to undo
 *       the interleaving. Combined with concat_even, these two operations can
 *       deinterleave a pair of interleaved vectors.
 *
 * @example
 *   Tag<float32_t, 8> t;
 *   auto lo = load(t, {0, 1, 2, 3, 4, 5, 6, 7});
 *   auto hi = load(t, {8, 9, 10, 11, 12, 13, 14, 15});
 *   auto v = concat_odd(t, lo, hi);  // v = [1, 3, 5, 7, 9, 11, 13, 15]
 */
template <TLV_DECL_TAG(T)>
VECOPS_VFUNC Vec<T> concat_odd(T t, Vec<T> v_lo, Vec<T> v_hi) {
  using namespace details;
  constexpr nint_t NWo = num_words(t);
  if constexpr (NWo > 1) {
    Half<T> th;
    auto u_lo = vec::odd(t, v_lo);
    auto u_hi = vec::odd(t, v_hi);
    Vec<T> v_o;
    foreach<NWo / 2>([&]<nint_t I>{
      v_o = set_word<I>(t, v_o, get_word<I>(th, u_lo));
    });
    foreach<NWo / 2>([&]<nint_t I>{
      v_o = set_word<NWo / 2 + I>(t, v_o, get_word<I>(th, u_hi));
    });
    return v_o;
  } else {
    return word::concat_odd(t, v_lo, v_hi);
  }
}

/**
 * @brief Interleave elements from the lower half of each 16-byte block.
 *
 * Performs element-wise interleaving within each 16-byte block, using only
 * elements from the lower 8 bytes of each input vector. The operation is
 * local to each 16-byte block (no cross-block data movement).
 *
 * For float32 x 8 (two 16-byte blocks):
 *   Input:  a = [a7, a6, a5, a4, a3, a2, a1, a0], b = [b7, b6, b5, b4, b3, b2, b1, b0]
 *   Result: [b5, a5, b4, a4, b1, a1, b0, a0]
 *
 * More generally, for each 16-byte block (M = 16 / sizeof(element)):
 *   result[2*i]   = a[i]     for i < M/2
 *   result[2*i+1] = b[i]     for i < M/2
 *
 * @tparam V Input/output vector type
 * @tparam T Vector tag type
 * @param a First input vector
 * @param b Second input vector
 * @return Interleaved vector containing lower-half elements from both inputs
 *
 * @note This operation stays within each 16-byte block, making it efficient
 *       on x86 architectures. Use concat_even(a, b) to deinterleave the result.
 *
 * @see local_interleave_upper for interleaving upper-half elements
 * @see interleave for full-vector interleaving (may be slower due to cross-block movement)
 */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
VECOPS_VFUNC V local_interleave_lower(V a, V b) {
  using namespace details;
  constexpr T t;
  return vmap(
      t, [&](auto tt, auto&& aa, auto&& bb) { return word::local_interleave_lower(aa, bb); },
      ShardVec(t, a), ShardVec(t, b)
  );
}

/**
 * @brief Interleave elements from the upper half of each 16-byte block.
 *
 * Performs element-wise interleaving within each 16-byte block, using only
 * elements from the upper 8 bytes of each input vector. The operation is
 * local to each 16-byte block (no cross-block data movement).
 *
 * For float32 x 8 (two 16-byte blocks):
 *   Input:  a = [a7, a6, a5, a4, a3, a2, a1, a0], b = [b7, b6, b5, b4, b3, b2, b1, b0]
 *   Result: [b7, a7, b6, a6, b3, a3, b2, a2]
 *
 * More generally, for each 16-byte block (M = 16 / sizeof(element)):
 *   result[2*i]   = a[M/2 + i]     for i < M/2
 *   result[2*i+1] = b[M/2 + i]     for i < M/2
 *
 * @tparam V Input/output vector type
 * @tparam T Vector tag type
 * @param a First input vector
 * @param b Second input vector
 * @return Interleaved vector containing upper-half elements from both inputs
 *
 * @note This operation stays within each 16-byte block, making it efficient
 *       on x86 architectures. Use concat_odd(a, b) to deinterleave the result.
 *
 * @see local_interleave_lower for interleaving lower-half elements
 * @see interleave for full-vector interleaving (may be slower due to cross-block movement)
 */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
VECOPS_VFUNC V local_interleave_upper(V a, V b) {
  using namespace details;
  constexpr T t;
  return vmap(
      t, [&](auto tt, auto&& aa, auto&& bb) { return word::local_interleave_upper(aa, bb); },
      ShardVec(t, a), ShardVec(t, b)
  );
}

/**
 * @brief Interleave elements from two half-length vectors into a full vector.
 *
 * Performs full interleaving of two half-length vectors, producing a vector
 * of twice the length where elements alternate between the two inputs.
 *
 * Result: [v_hi[N/2-1], v_lo[N/2-1], ..., v_hi[1], v_lo[1], v_hi[0], v_lo[0]]
 *
 * More formally: result[2*i] = v_lo[i], result[2*i+1] = v_hi[i]
 *
 * @tparam T Output vector tag type
 * @tparam V Input vector type (Vec<Half<T>>)
 * @param t Output vector tag
 * @param v_lo First input vector (placed at even indices in result)
 * @param v_hi Second input vector (placed at odd indices in result)
 * @return Interleaved vector of size N
 *
 * @warning This operation may involve cross-block data movement on x86 AVX/AVX2,
 *          which can result in performance penalties compared to local_interleave_*
 *          operations. For better performance on these architectures, consider
 *          using local_interleave_lower/local_interleave_upper instead.
 *
 * @note This is the inverse operation of even() and odd() combined:
 *       interleave(t, even(t, v), odd(t, v)) == v
 *
 * @example
 *   Tag<float32_t, 8> t;
 *   Tag<float32_t, 4> th;
 *   auto lo = load(th, {0, 1, 2, 3});
 *   auto hi = load(th, {4, 5, 6, 7});
 *   auto v = interleave(t, lo, hi);  // v = [4, 0, 5, 1, 6, 2, 7, 3] (MSB to LSB)
 */
template <TLV_DECL_TAG(T), typename V = Vec<Half<T>>>
VECOPS_VFUNC Vec<T> interleave(T t, V v_lo, V v_hi) {
  using namespace details;
  using Ti = Vec2Tag<V>;
  using TWo = WordOf<T>;
  constexpr nint_t NWo = num_words(t);
  if constexpr (num_words(t) > 1) {
    static_assert(NWo == 2 * num_words(Ti{}));
    Vec<T> v_o;
    foreach<NWo / 2>([&]<nint_t I>{
      auto vi_lo = get_word<I>(Ti(), v_lo);
      auto vi_hi = get_word<I>(Ti(), v_hi);
      v_o = set_word<2 * I>(t, v_o, word::interleave(
          TWo(), word::lower(TWo(), vi_lo), word::lower(TWo(), vi_hi)
      ));
      v_o = set_word<2 * I + 1>(t, v_o, word::interleave(
          TWo(), word::upper(TWo(), vi_lo), word::upper(TWo(), vi_hi)
      ));
    });
    return v_o;
  } else {
    return word::interleave(t, v_lo, v_hi);
  }
}

/**
 * @brief Interleave even-indexed elements from two vectors.
 *
 * Takes elements at even indices from both input vectors and interleaves them.
 * The result has half the length of each input.
 *
 * For vectors a and b of size N:
 *   result[2*i]   = a[2*i]
 *   result[2*i+1] = b[2*i]
 *   for i in [0, N/2)
 *
 * @tparam V Input/output vector type
 * @tparam T Vector tag type
 * @param a First input vector
 * @param b Second input vector
 * @return Vector of size N/2 containing interleaved even elements
 *
 * @note This is the inverse operation of concat_even for deinterleaving.
 *       Combined with interleave_odd, these can separate interleaved data.
 *
 * @example
 *   Tag<float32_t, 8> t;
 *   auto a = load(t, {0, 1, 2, 3, 4, 5, 6, 7});
 *   auto b = load(t, {8, 9, 10, 11, 12, 13, 14, 15});
 *   auto v = interleave_even(a, b);  // v = [8, 0, 10, 2, 12, 4, 14, 6]
 */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
VECOPS_VFUNC V interleave_even(V a, V b) {
  using namespace details;
  constexpr T t;
  return vmap(
      t, [&](auto tt, auto&& aa, auto&& bb) { return word::interleave_even(aa, bb); },
      ShardVec(t, a), ShardVec(t, b)
  );
}

/**
 * @brief Interleave odd-indexed elements from two vectors.
 *
 * Takes elements at odd indices from both input vectors and interleaves them.
 * The result has half the length of each input.
 *
 * For vectors a and b of size N:
 *   result[2*i]   = a[2*i + 1]
 *   result[2*i+1] = b[2*i + 1]
 *   for i in [0, N/2)
 *
 * @tparam V Input/output vector type
 * @tparam T Vector tag type
 * @param a First input vector
 * @param b Second input vector
 * @return Vector of size N/2 containing interleaved odd elements
 *
 * @note This is the inverse operation of concat_odd for deinterleaving.
 *       Combined with interleave_even, these can separate interleaved data.
 *
 * @example
 *   Tag<float32_t, 8> t;
 *   auto a = load(t, {0, 1, 2, 3, 4, 5, 6, 7});
 *   auto b = load(t, {8, 9, 10, 11, 12, 13, 14, 15});
 *   auto v = interleave_odd(a, b);  // v = [9, 1, 11, 3, 13, 5, 15, 7]
 */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
VECOPS_VFUNC V interleave_odd(V a, V b) {
  using namespace details;
  constexpr T t;
  return vmap(
      t, [&](auto tt, auto&& aa, auto&& bb) { return word::interleave_odd(aa, bb); },
      ShardVec(t, a), ShardVec(t, b)
  );
}


/* ************************************************************************** */
//                       Data type & size conversions                         //
/* ************************************************************************** */

namespace details {

template <typename To, typename Ti>
inline constexpr bool same_vector_bytes =
    To::is_runtime_size == Ti::is_runtime_size &&
    (To::is_runtime_size ? To::POW2 == Ti::POW2 : To::Bytes == Ti::Bytes);

template <typename To, typename Vi, typename... Options>
consteval bool valid_conversion() {
  if constexpr (!(policy_details::is_conversion_option<Options> && ...)) {
    return false;
  } else if constexpr (
      policy_details::count_options<
          policy_details::IsLayout, Options...> > 1 ||
      policy_details::count_options<
          policy_details::IsValuePolicy, Options...> > 1 ||
      policy_details::count_options<
          policy_details::IsPopulation, Options...> > 1) {
    return false;
  } else {
    using Ti = Vec2Tag<Vi>;
    using Ei = TypeOf<Ti>;
    using Eo = TypeOf<To>;
    using Layout = policy_details::selected_option_t<
        policy_details::IsLayout, cvt::ordered_t, Options...>;
    using ValuePolicy = policy_details::selected_option_t<
        policy_details::IsValuePolicy, cvt::saturate_t, Options...>;
    constexpr int population_count =
        policy_details::count_options<
            policy_details::IsPopulation, Options...>;

    if constexpr (policy_details::is_wrap<ValuePolicy> &&
                  !(std::is_integral_v<Ei> && std::is_integral_v<Eo> &&
                    sizeof(Eo) < sizeof(Ei))) {
      return false;
    } else if constexpr (policy_details::is_ordered<Layout> ||
                         policy_details::is_unordered<Layout>) {
      return population_count == 0;
    } else {
      constexpr int ratio = sizeof(Ei) < sizeof(Eo)
          ? int(sizeof(Eo) / sizeof(Ei))
          : int(sizeof(Ei) / sizeof(Eo));
      constexpr int phase = Layout::phase;
      return same_vector_bytes<To, Ti> &&
             (ratio == 2 || ratio == 4 || ratio == 8) &&
             (phase == 0 || (phase == 1 && ratio == 2)) &&
             (sizeof(Ei) > sizeof(Eo) || population_count == 0);
    }
  }
}

template <typename... Options>
consteval void validate_conversion_options() {
  static_assert((policy_details::is_conversion_option<Options> && ...),
                "unsupported convert option");
  static_assert(
      policy_details::count_options<policy_details::IsLayout, Options...> <= 1,
      "convert accepts at most one layout option");
  static_assert(
      policy_details::count_options<
          policy_details::IsValuePolicy, Options...> <= 1,
      "convert accepts at most one value policy");
  static_assert(
      policy_details::count_options<
          policy_details::IsPopulation, Options...> <= 1,
      "convert accepts at most one population option");
}

template <TLV_DECL_TAG(To), TLV_DECL_VEC(Vi)>
VECOPS_VFUNC Vec<To> convert_ordered_saturate(To to, Vi vi) {
  constexpr Vec2Tag<Vi> ti;
  using Ei = TypeOf<Vec2Tag<Vi>>;
  using Eo = TypeOf<To>;
  if constexpr (sizeof(Ei) < sizeof(Eo)) {
    if constexpr (num_words(ti) > 1 && num_words(to) > 1) {
      constexpr Half<To> th;
      auto lo = convert_ordered_saturate(th, vec::lower(ti, vi));
      auto hi = convert_ordered_saturate(th, vec::upper(ti, vi));
      return vec::concat(to, lo, hi);
    } else {
      return word::promote(to, vi);
    }
  } else if constexpr (sizeof(Ei) > sizeof(Eo)) {
    if constexpr (num_words(ti) > 1 && num_words(to) > 1) {
      constexpr Half<To> th;
      auto lo = convert_ordered_saturate(th, vec::lower(ti, vi));
      auto hi = convert_ordered_saturate(th, vec::upper(ti, vi));
      return vec::concat(to, lo, hi);
    } else {
      return word::demote(to, vi);
    }
  } else {
    return vmap(
        to, [](auto tt, auto vv) { return word::convert(tt, vv); },
        ShardVec(ti, vi));
  }
}

template <TLV_DECL_TAG(To), TLV_DECL_VEC(Vi)>
VECOPS_VFUNC Vec<To> convert_ordered_wrap(To to, Vi vi) {
  constexpr Vec2Tag<Vi> ti;
  using Ei = TypeOf<Vec2Tag<Vi>>;
  using Eo = TypeOf<To>;
  static_assert(
      std::is_integral_v<Ei> && std::is_integral_v<Eo> &&
      sizeof(Eo) < sizeof(Ei),
      "cvt::wrap/truncate requires integer-to-integer narrowing");
  VECOPS_ASSERT(
      size(to) <= size(ti), "insufficient input lanes for convert");
  auto output = vec::zeros(to);
  for (nint_t i = 0; i < size(to); ++i) {
    output = vec::set(
        to, output, i, vecops::wrap_convert<Eo>(vec::get(ti, vi, i)));
  }
  return output;
}

template <int Phase, TLV_DECL_TAG(To), TLV_DECL_VEC(Vi), typename Population>
VECOPS_VFUNC Vec<To> convert_lane_saturate(
    To to, Vi vi, Population&& population) {
  constexpr Vec2Tag<Vi> ti;
  using Ei = TypeOf<Vec2Tag<Vi>>;
  using Eo = TypeOf<To>;
  constexpr int ratio = sizeof(Ei) < sizeof(Eo)
      ? int(sizeof(Eo) / sizeof(Ei))
      : int(sizeof(Ei) / sizeof(Eo));
  static_assert(
      same_vector_bytes<To, decltype(ti)> &&
      (ratio == 2 || ratio == 4 || ratio == 8),
      "cvt::lane requires equal vector bytes and a 2x/4x/8x width ratio");
  static_assert(Phase == 0 || (Phase == 1 && ratio == 2),
                "only lane<0>, and lane<1> for 2x conversion, are supported");
  if constexpr (sizeof(Ei) < sizeof(Eo)) {
    if constexpr (Phase == 0) {
      return vmap(
          to, [](auto tt, auto vv) { return word::promote_even(tt, vv); },
          ShardVec(ti, vi));
    } else {
      return vmap(
          to, [](auto tt, auto vv) { return word::promote_odd(tt, vv); },
          ShardVec(ti, vi));
    }
  } else {
    if constexpr (policy_details::is_zero<Population>) {
      if constexpr (Phase == 0) {
        return vmap(
            to,
            [](auto tt, auto vv) {
              return word::demote_even(tt, vv);
            },
            ShardVec(ti, vi));
      } else {
        return vmap(
            to,
            [](auto tt, auto vv) {
              return word::demote_odd(tt, vv);
            },
            ShardVec(ti, vi));
      }
    } else {
      auto fallback = population_value(
          to, std::forward<Population>(population));
      if constexpr (Phase == 0) {
        return vmap(
            to,
            [](auto tt, auto vv, auto ff) {
              return word::demote_even(tt, vv, ff);
            },
            ShardVec(ti, vi), ShardVec(to, fallback));
      } else {
        return vmap(
            to,
            [](auto tt, auto vv, auto ff) {
              return word::demote_odd(tt, vv, ff);
            },
            ShardVec(ti, vi), ShardVec(to, fallback));
      }
    }
  }
}

template <int Phase, TLV_DECL_TAG(To), TLV_DECL_VEC(Vi), typename Population>
VECOPS_VFUNC Vec<To> convert_lane_wrap(
    To to, Vi vi, Population&& population) {
  constexpr Vec2Tag<Vi> ti;
  using Ei = TypeOf<Vec2Tag<Vi>>;
  using Eo = TypeOf<To>;
  static_assert(
      std::is_integral_v<Ei> && std::is_integral_v<Eo> &&
      sizeof(Eo) < sizeof(Ei),
      "cvt::wrap/truncate requires integer-to-integer narrowing");
  constexpr int ratio = int(sizeof(Ei) / sizeof(Eo));
  static_assert(
      same_vector_bytes<To, decltype(ti)> &&
      (ratio == 2 || ratio == 4 || ratio == 8),
      "cvt::lane requires equal vector bytes and a 2x/4x/8x width ratio");
  static_assert(Phase == 0 || (Phase == 1 && ratio == 2),
                "only lane<0>, and lane<1> for 2x conversion, are supported");
  auto output = population_value(
      to, std::forward<Population>(population));
  for (nint_t i = 0; i < size(ti); ++i) {
    output = vec::set(
        to, output, ratio * i + Phase,
        vecops::wrap_convert<Eo>(vec::get(ti, vi, i)));
  }
  return output;
}

template <int Levels, TLV_DECL_TAG(To), TLV_DECL_TAG(Ti), typename ValuePolicy>
VECOPS_VFUNC Vec<To> convert_unordered_widen(
    To to, Ti ti, Vec<Ti> vi, ValuePolicy policy) {
#if defined(CPU_CAPABILITY_SVE)
  if constexpr (Levels > 0) {
    constexpr Half<To> to_half;
    Vec<Half<To>> lo;
    Vec<Half<To>> hi;
    if constexpr (Levels == 1) {
      lo = convert_lane_saturate<0>(to_half, vi, opt::zero);
      hi = convert_lane_saturate<1>(to_half, vi, opt::zero);
    } else {
      constexpr Half<Ti> ti_half;
      lo = convert_unordered_widen<Levels - 1>(
          to_half, ti_half, word::even(ti, vi), policy);
      hi = convert_unordered_widen<Levels - 1>(
          to_half, ti_half, word::odd(ti, vi), policy);
    }
    return vec::concat(to, lo, hi);
  }
#endif
  if constexpr (policy_details::is_wrap<ValuePolicy>) {
    return convert_ordered_wrap(to, vi);
  } else {
    return convert_ordered_saturate(to, vi);
  }
}

template <int Levels, TLV_DECL_TAG(To), TLV_DECL_TAG(Ti), typename ValuePolicy>
VECOPS_VFUNC Vec<To> convert_unordered_narrow(
    To to, Ti ti, Vec<Ti> vi, ValuePolicy policy) {
#if defined(CPU_CAPABILITY_SVE)
  if constexpr (Levels > 0) {
    if constexpr (
        Levels == 1 &&
        !std::is_same_v<TypeOf<To>, bfloat16_t>) {
      auto packed = convert_lane_saturate<0>(
          to, vec::lower(ti, vi), opt::zero);
      return convert_lane_saturate<1>(
          to, vec::upper(ti, vi), opt::merge(packed));
    } else {
      constexpr Half<To> to_half;
      constexpr Half<Ti> ti_half;
      auto lo = convert_unordered_narrow<Levels - 1>(
          to_half, ti_half, vec::lower(ti, vi), policy);
      auto hi = convert_unordered_narrow<Levels - 1>(
          to_half, ti_half, vec::upper(ti, vi), policy);
      return vec::interleave(to, lo, hi);
    }
  }
#endif
  if constexpr (policy_details::is_wrap<ValuePolicy>) {
    return convert_ordered_wrap(to, vi);
  } else {
    return convert_ordered_saturate(to, vi);
  }
}

} // namespace details

/**
 * @brief Convert vector elements with orthogonal value/layout/population policy.
 */
template <TLV_DECL_TAG(To), TLV_DECL_VEC(Vi), typename... Options>
  requires (details::valid_conversion<To, Vi, Options...>())
VECOPS_VFUNC Vec<To> convert(To to, Vi vi, Options&&... options) {
  details::validate_conversion_options<Options...>();
  constexpr int population_count =
      policy_details::count_options<
          policy_details::IsPopulation, Options...>;
  auto&& layout = policy_details::select_option<
      policy_details::IsLayout>(
      cvt::ordered, std::forward<Options>(options)...);
  auto&& value_policy = policy_details::select_option<
      policy_details::IsValuePolicy>(
      cvt::saturate, std::forward<Options>(options)...);
  auto&& population = policy_details::select_option<
      policy_details::IsPopulation>(
      opt::zero, std::forward<Options>(options)...);

  constexpr Vec2Tag<Vi> ti;
  using Ei = TypeOf<Vec2Tag<Vi>>;
  using Eo = TypeOf<To>;
  if constexpr (policy_details::is_wrap<decltype(value_policy)>) {
    static_assert(
        std::is_integral_v<Ei> && std::is_integral_v<Eo> &&
        sizeof(Eo) < sizeof(Ei),
        "cvt::wrap/truncate requires integer-to-integer narrowing");
  }

  if constexpr (policy_details::is_ordered<decltype(layout)>) {
    static_assert(population_count == 0,
                  "ordered convert does not accept population options");
    if constexpr (policy_details::is_wrap<decltype(value_policy)>) {
      return details::convert_ordered_wrap(to, vi);
    } else {
      return details::convert_ordered_saturate(to, vi);
    }
  } else if constexpr (policy_details::is_lane<decltype(layout)>) {
    constexpr int phase =
        policy_details::remove_cvref_t<decltype(layout)>::phase;
    constexpr int ratio = sizeof(Ei) < sizeof(Eo)
        ? int(sizeof(Eo) / sizeof(Ei))
        : int(sizeof(Ei) / sizeof(Eo));
    static_assert(
        phase == 0 || (phase == 1 && ratio == 2),
        "only lane<0>, and lane<1> for 2x conversion, are supported");
    if constexpr (sizeof(Ei) < sizeof(Eo)) {
      static_assert(population_count == 0,
                    "widening lane convert does not accept population options");
    }
    if constexpr (policy_details::is_wrap<decltype(value_policy)>) {
      return details::convert_lane_wrap<phase>(
          to, vi, std::forward<decltype(population)>(population));
    } else {
      return details::convert_lane_saturate<phase>(
          to, vi, std::forward<decltype(population)>(population));
    }
  } else {
    static_assert(population_count == 0,
                  "unordered convert does not accept population options");
    if constexpr (sizeof(Ei) == sizeof(Eo)) {
      return details::convert_ordered_saturate(to, vi);
    } else {
      constexpr int ratio = sizeof(Ei) < sizeof(Eo)
          ? int(sizeof(Eo) / sizeof(Ei))
          : int(sizeof(Ei) / sizeof(Eo));
      static_assert(
          ratio == 2 || ratio == 4 || ratio == 8,
          "unordered convert requires a 2x/4x/8x width ratio");
      if constexpr (sizeof(Ei) < sizeof(Eo)) {
        return details::convert_unordered_widen<log2_floor(ratio)>(
            to, ti, vi, value_policy);
      } else {
        return details::convert_unordered_narrow<log2_floor(ratio)>(
            to, ti, vi, value_policy);
      }
    }
  }
}

/* ************************************************************************** */
//                              Mask conversion                              //
/* ************************************************************************** */

template <TLV_DECL_TAG(To), TLV_DECL_TAG(Ti)>
VECOPS_VFUNC Mask<To> convert(To to, Ti ti, Mask<Ti> mi) {
  using Ei = TypeOf<Ti>;
  using Eo = TypeOf<To>;
  if constexpr (num_words(ti) > 1 || num_words(to) > 1) {
    constexpr Half<To> to_half;
    constexpr Half<Ti> ti_half;
    auto lo = vec::convert(to_half, ti_half, vec::lower(ti, mi));
    auto hi = vec::convert(to_half, ti_half, vec::upper(ti, mi));
    return vec::concat(to, lo, hi);
  } else if constexpr (sizeof(Ei) < sizeof(Eo)) {
    return word::promote(to, ti, mi);
  } else if constexpr (sizeof(Ei) > sizeof(Eo)) {
    return word::demote(to, ti, mi);
  } else {
    return word::convert(to, ti, mi);
  }
}

/* ************************************************************************** */
//                Conversion fused with consecutive memory access           //
/* ************************************************************************** */

namespace details {

template <typename... Options>
consteval void validate_memory_conversion_options() {
  static_assert(
      ((policy_details::is_memory_conversion_option<Options> ||
        policy_details::is_layout<Options> ||
        policy_details::is_value_policy<Options>) && ...),
      "unsupported load_convert/store_convert option");
  static_assert(
      policy_details::count_options<policy_details::IsLayout, Options...> <= 1,
      "memory conversion accepts at most one layout option");
  static_assert(
      policy_details::count_options<
          policy_details::IsValuePolicy, Options...> <= 1,
      "memory conversion accepts at most one value policy");
  static_assert(
      policy_details::count_options<policy_details::IsAlignment, Options...> <= 1,
      "memory conversion accepts at most one alignment option");
  static_assert(
      policy_details::count_options<
          policy_details::IsTemporality, Options...> <= 1,
      "memory conversion accepts at most one temporal option");
  static_assert(
      policy_details::count_options<
          policy_details::IsPacking, Options...> <= 1,
      "memory conversion accepts at most one packing option");
  static_assert(
      policy_details::count_options<policy_details::IsActive, Options...> <= 1,
      "opt::masked and opt::first are mutually exclusive");
  static_assert(
      policy_details::count_options<
          policy_details::IsPopulation, Options...> <= 1,
      "memory conversion accepts at most one population option");
}

template <bool IsStore, typename Ei, typename Eo, typename... Options>
consteval bool valid_memory_conversion() {
  if constexpr (!(
      (policy_details::is_memory_conversion_option<Options> ||
       policy_details::is_layout<Options> ||
       policy_details::is_value_policy<Options>) && ...)) {
    return false;
  } else if constexpr (
      policy_details::count_options<
          policy_details::IsLayout, Options...> > 1 ||
      policy_details::count_options<
          policy_details::IsValuePolicy, Options...> > 1 ||
      policy_details::count_options<
          policy_details::IsAlignment, Options...> > 1 ||
      policy_details::count_options<
          policy_details::IsTemporality, Options...> > 1 ||
      policy_details::count_options<
          policy_details::IsPacking, Options...> > 1 ||
      policy_details::count_options<
          policy_details::IsActive, Options...> > 1 ||
      policy_details::count_options<
          policy_details::IsPopulation, Options...> > 1) {
    return false;
  } else {
    using Layout = policy_details::selected_option_t<
        policy_details::IsLayout, cvt::ordered_t, Options...>;
    using ValuePolicy = policy_details::selected_option_t<
        policy_details::IsValuePolicy, cvt::saturate_t, Options...>;
    using Packing = policy_details::selected_option_t<
        policy_details::IsPacking, mem::packed_t, Options...>;
    using Alignment = policy_details::selected_option_t<
        policy_details::IsAlignment, mem::unaligned_t, Options...>;
    using Temporality = policy_details::selected_option_t<
        policy_details::IsTemporality, mem::temporal_t, Options...>;
    constexpr int active_count =
        policy_details::count_options<
            policy_details::IsActive, Options...>;
    constexpr int population_count =
        policy_details::count_options<
            policy_details::IsPopulation, Options...>;
    constexpr bool ordered_layout =
        policy_details::is_ordered<Layout>;
    constexpr bool unordered_layout =
        policy_details::is_unordered<Layout>;
    constexpr bool valid_population =
        population_count == 0 ||
        (!IsStore && ordered_layout && active_count == 1);
    constexpr bool valid_wrap =
        !policy_details::is_wrap<ValuePolicy> ||
        (std::is_integral_v<Ei> && std::is_integral_v<Eo> &&
         sizeof(Eo) < sizeof(Ei));
    constexpr bool valid_packing =
        policy_details::is_packed<Packing> ||
        (IsStore && ordered_layout &&
         policy_details::is_saturate<ValuePolicy> &&
         policy_details::is_unaligned<Alignment> &&
         policy_details::is_temporal<Temporality>);
    return (ordered_layout || unordered_layout) &&
           valid_population && valid_wrap && valid_packing;
  }
}

template <TLV_DECL_TAG(To), typename Ei>
VECOPS_VFUNC Vec<To> fused_load_saturate(To to, const Ei* p) {
  constexpr Rebind<Ei, To> ti;
  if constexpr (is_word_vec(to) || is_word_vec(ti)) {
    if constexpr (sizeof(Ei) < sizeof(TypeOf<To>)) {
      return word::promote_loadu(to, p);
    } else if constexpr (sizeof(Ei) > sizeof(TypeOf<To>)) {
      return word::demote_loadu(to, p);
    } else {
      return word::convert_loadu(to, p);
    }
  } else if constexpr (sizeof(Ei) != sizeof(TypeOf<To>)) {
    constexpr Half<To> th;
    const auto lo = fused_load_saturate(th, p);
    const auto hi =
        fused_load_saturate(th, p + size(th));
    return vec::concat(to, lo, hi);
  } else {
    return vmap(
        to,
        [](auto tt, const Ei* pp) VECOPS_INLINE_LAMBDA {
          if constexpr (sizeof(Ei) < sizeof(TypeOf<decltype(tt)>)) {
            return word::promote_loadu(tt, pp);
          } else if constexpr (sizeof(Ei) >
                               sizeof(TypeOf<decltype(tt)>)) {
            return word::demote_loadu(tt, pp);
          } else {
            return word::convert_loadu(tt, pp);
          }
        },
        StepPointer(p, word_size(to)));
  }
}

template <TLV_DECL_TAG(To), typename Ei>
VECOPS_VFUNC Vec<To> fused_load_saturate(
    To to, const Ei* p, Mask<To> m) {
  constexpr Rebind<Ei, To> ti;
  if constexpr (is_word_vec(to) || is_word_vec(ti)) {
    if constexpr (sizeof(Ei) < sizeof(TypeOf<To>)) {
      return word::promote_loadu(to, p, m);
    } else if constexpr (sizeof(Ei) > sizeof(TypeOf<To>)) {
      return word::demote_loadu(to, p, m);
    } else {
      return word::convert_loadu(to, p, m);
    }
  } else if constexpr (sizeof(Ei) != sizeof(TypeOf<To>)) {
    constexpr Half<To> th;
    const auto m_lo = vec::lower(to, m);
    const auto m_hi = vec::upper(to, m);
    const auto lo = fused_load_saturate(th, p, m_lo);
    const auto hi =
        fused_load_saturate(th, p + size(th), m_hi);
    return vec::concat(to, lo, hi);
  } else {
    return vmap(
        to,
        [](auto tt, const Ei* pp,
           auto mm) VECOPS_INLINE_LAMBDA {
          if constexpr (sizeof(Ei) < sizeof(TypeOf<decltype(tt)>)) {
            return word::promote_loadu(tt, pp, mm);
          } else if constexpr (sizeof(Ei) >
                               sizeof(TypeOf<decltype(tt)>)) {
            return word::demote_loadu(tt, pp, mm);
          } else {
            return word::convert_loadu(tt, pp, mm);
          }
        },
        StepPointer(p, word_size(to)), ShardMask(to, m));
  }
}

template <TLV_DECL_TAG(To), typename Ei>
VECOPS_VFUNC Vec<To> fused_load_saturate(
    To to, const Ei* p, Mask<To> m, Vec<To> default_v) {
  constexpr Rebind<Ei, To> ti;
  if constexpr (is_word_vec(to) || is_word_vec(ti)) {
    if constexpr (sizeof(Ei) < sizeof(TypeOf<To>)) {
      return word::promote_loadu(to, p, m, default_v);
    } else if constexpr (sizeof(Ei) > sizeof(TypeOf<To>)) {
      return word::demote_loadu(to, p, m, default_v);
    } else {
      return word::convert_loadu(to, p, m, default_v);
    }
  } else if constexpr (sizeof(Ei) != sizeof(TypeOf<To>)) {
    constexpr Half<To> th;
    const auto m_lo = vec::lower(to, m);
    const auto m_hi = vec::upper(to, m);
    const auto v_lo = vec::lower(to, default_v);
    const auto v_hi = vec::upper(to, default_v);
    const auto lo =
        fused_load_saturate(th, p, m_lo, v_lo);
    const auto hi = fused_load_saturate(
        th, p + size(th), m_hi, v_hi);
    return vec::concat(to, lo, hi);
  } else {
    return vmap(
        to,
        [](auto tt, const Ei* pp, auto mm,
           auto dd) VECOPS_INLINE_LAMBDA {
          if constexpr (sizeof(Ei) < sizeof(TypeOf<decltype(tt)>)) {
            return word::promote_loadu(tt, pp, mm, dd);
          } else if constexpr (sizeof(Ei) >
                               sizeof(TypeOf<decltype(tt)>)) {
            return word::demote_loadu(tt, pp, mm, dd);
          } else {
            return word::convert_loadu(tt, pp, mm, dd);
          }
        },
        StepPointer(p, word_size(to)),
        ShardMask(to, m), ShardVec(to, default_v));
  }
}

template <TLV_DECL_TAG(Ti), typename Eo>
VECOPS_VFUNC void fused_store_saturate(
    Ti ti, Eo* p, Vec<Ti> vi) {
  constexpr Rebind<Eo, Ti> to;
  constexpr bool sve_bf16_path =
#if defined(CPU_CAPABILITY_SVE) && defined(__ARM_FEATURE_SVE_BF16) && \
    !defined(VECOPS_PRESERVE_SUBNORMALS)
      std::is_same_v<TypeOf<Ti>, float32_t> &&
      std::is_same_v<Eo, bfloat16_t>;
#else
      false;
#endif
  if constexpr (sve_bf16_path) {
    word::demote_storeu(ti, p, vi);
  } else if constexpr (is_word_vec(ti) || is_word_vec(to)) {
    if constexpr (sizeof(TypeOf<Ti>) < sizeof(Eo)) {
      word::promote_storeu(ti, p, vi);
    } else if constexpr (sizeof(TypeOf<Ti>) > sizeof(Eo)) {
      word::demote_storeu(ti, p, vi);
    } else {
      word::convert_storeu(ti, p, vi);
    }
  } else if constexpr (sizeof(TypeOf<Ti>) != sizeof(Eo)) {
    constexpr Half<Ti> th;
    const Vec<Half<Ti>> v_lo = vec::lower(ti, vi);
    const Vec<Half<Ti>> v_hi = vec::upper(ti, vi);
    fused_store_saturate(th, p, v_lo);
    fused_store_saturate(th, p + size(th), v_hi);
  } else {
    vmap(
        ti,
        [](auto tt, Eo* pp, auto vv) VECOPS_INLINE_LAMBDA {
          if constexpr (sizeof(TypeOf<decltype(tt)>) < sizeof(Eo)) {
            word::promote_storeu(tt, pp, vv);
          } else if constexpr (sizeof(TypeOf<decltype(tt)>) > sizeof(Eo)) {
            word::demote_storeu(tt, pp, vv);
          } else {
            word::convert_storeu(tt, pp, vv);
          }
        },
        StepPointer(p, word_size(ti)), ShardVec(ti, vi));
  }
}

template <TLV_DECL_TAG(Ti), typename Eo>
VECOPS_VFUNC void fused_store_saturate(
    Ti ti, Eo* p, Mask<Ti> m, Vec<Ti> vi) {
  constexpr Rebind<Eo, Ti> to;
  constexpr bool sve_bf16_path =
#if defined(CPU_CAPABILITY_SVE) && defined(__ARM_FEATURE_SVE_BF16) && \
    !defined(VECOPS_PRESERVE_SUBNORMALS)
      std::is_same_v<TypeOf<Ti>, float32_t> &&
      std::is_same_v<Eo, bfloat16_t>;
#else
      false;
#endif
  if constexpr (sve_bf16_path) {
    word::demote_storeu(ti, p, m, vi);
  } else if constexpr (is_word_vec(ti) || is_word_vec(to)) {
    if constexpr (sizeof(TypeOf<Ti>) < sizeof(Eo)) {
      word::promote_storeu(ti, p, m, vi);
    } else if constexpr (sizeof(TypeOf<Ti>) > sizeof(Eo)) {
      word::demote_storeu(ti, p, m, vi);
    } else {
      word::convert_storeu(ti, p, m, vi);
    }
  } else if constexpr (sizeof(TypeOf<Ti>) != sizeof(Eo)) {
    constexpr Half<Ti> th;
    const Mask<Half<Ti>> m_lo = vec::lower(ti, m);
    const Mask<Half<Ti>> m_hi = vec::upper(ti, m);
    const Vec<Half<Ti>> v_lo = vec::lower(ti, vi);
    const Vec<Half<Ti>> v_hi = vec::upper(ti, vi);
    fused_store_saturate(th, p, m_lo, v_lo);
    fused_store_saturate(
        th, p + size(th), m_hi, v_hi);
  } else {
    vmap(
        ti,
        [](auto tt, Eo* pp, auto mm, auto vv) VECOPS_INLINE_LAMBDA {
          if constexpr (sizeof(TypeOf<decltype(tt)>) < sizeof(Eo)) {
            word::promote_storeu(tt, pp, mm, vv);
          } else if constexpr (sizeof(TypeOf<decltype(tt)>) > sizeof(Eo)) {
            word::demote_storeu(tt, pp, mm, vv);
          } else {
            word::convert_storeu(tt, pp, mm, vv);
          }
        },
        StepPointer(p, word_size(ti)),
        ShardMask(ti, m), ShardVec(ti, vi));
  }
}

template <TLV_DECL_TAG(Ti), typename Eo>
inline constexpr bool prefer_native_full_width_split_store =
#if defined(CPU_CAPABILITY_SVE) && defined(__ARM_FEATURE_SVE_BF16) && \
    !defined(VECOPS_PRESERVE_SUBNORMALS)
    std::is_same_v<TypeOf<Ti>, float32_t> &&
    std::is_same_v<Eo, bfloat16_t> && num_words(Ti{}) > 1;
#else
    false;
#endif

template <TLV_DECL_TAG(Ti), typename Eo>
VECOPS_VFUNC void fused_store_saturate_split(
    Ti ti, Eo* p, Vec<Ti> vi) {
  if constexpr (prefer_native_full_width_split_store<Ti, Eo>) {
    // A multiword F32 -> BF16 conversion contains at least one complete
    // native output vector. On the 512-bit SVE target this is a 64-byte
    // store. Never split it into multiple 32-byte narrowing stores: those
    // are substantially slower for streaming LayerNorm workloads.
    fused_store_saturate(ti, p, vi);
  } else if constexpr (is_word_vec(ti)) {
    if constexpr (sizeof(TypeOf<Ti>) < sizeof(Eo)) {
      word::promote_storeu(ti, p, vi);
    } else if constexpr (sizeof(TypeOf<Ti>) > sizeof(Eo)) {
      word::demote_storeu(ti, p, vi);
    } else {
      word::convert_storeu(ti, p, vi);
    }
  } else {
    constexpr Half<Ti> th;
    fused_store_saturate_split(th, p, vec::lower(ti, vi));
    fused_store_saturate_split(
        th, p + size(th), vec::upper(ti, vi));
  }
}

template <TLV_DECL_TAG(Ti), typename Eo>
VECOPS_VFUNC void fused_store_saturate_split(
    Ti ti, Eo* p, Mask<Ti> m, Vec<Ti> vi) {
  if constexpr (prefer_native_full_width_split_store<Ti, Eo>) {
    // Keep masked multiword stores on the same full-width backend as the
    // unmasked route; the backend derives the output-side predicates.
    fused_store_saturate(ti, p, m, vi);
  } else if constexpr (is_word_vec(ti)) {
    if constexpr (sizeof(TypeOf<Ti>) < sizeof(Eo)) {
      word::promote_storeu(ti, p, m, vi);
    } else if constexpr (sizeof(TypeOf<Ti>) > sizeof(Eo)) {
      word::demote_storeu(ti, p, m, vi);
    } else {
      word::convert_storeu(ti, p, m, vi);
    }
  } else {
    constexpr Half<Ti> th;
    fused_store_saturate_split(
        th, p, vec::lower(ti, m), vec::lower(ti, vi));
    fused_store_saturate_split(
        th, p + size(th), vec::upper(ti, m), vec::upper(ti, vi));
  }
}

} // namespace details

/**
 * @brief Conversion fused with a consecutive load.
 *
 * For an unordered conversion, opt::masked uses the memory-side mask.
 */
template <TLV_DECL_TAG(To), typename Ei, typename... Options>
  requires (
      is_element_type<Ei> &&
      details::valid_memory_conversion<
          false, Ei, TypeOf<To>, Options...>())
VECOPS_VFUNC Vec<To> load_convert(
    To to, const Ei* p, Options&&... options) {
  details::validate_memory_conversion_options<Options...>();
  auto&& layout = policy_details::select_option<
      policy_details::IsLayout>(
      cvt::ordered, std::forward<Options>(options)...);
  static_assert(policy_details::is_ordered<decltype(layout)> ||
                policy_details::is_unordered<decltype(layout)>,
                "load_convert supports ordered/unordered layouts only");

  constexpr int active_count =
      policy_details::count_options<policy_details::IsActive, Options...>;
  constexpr int population_count =
      policy_details::count_options<
          policy_details::IsPopulation, Options...>;
  static_assert(active_count != 0 || population_count == 0,
                "load_convert population requires opt::masked or opt::first");

  auto&& alignment = policy_details::select_option<
      policy_details::IsAlignment>(
      mem::unaligned, std::forward<Options>(options)...);
  auto&& temporality = policy_details::select_option<
      policy_details::IsTemporality>(
      mem::temporal, std::forward<Options>(options)...);
  auto&& value_policy = policy_details::select_option<
      policy_details::IsValuePolicy>(
      cvt::saturate, std::forward<Options>(options)...);
  auto&& population = policy_details::select_option<
      policy_details::IsPopulation>(
      opt::zero, std::forward<Options>(options)...);

  constexpr Rebind<Ei, To> ti;
  Mask<To> output_mask;
  if constexpr (
      active_count != 0 &&
      policy_details::is_ordered<decltype(layout)>) {
    auto&& active = policy_details::select_option<
        policy_details::IsActive>(
        opt::first(size(to)), std::forward<Options>(options)...);
    if constexpr (policy_details::is_masked<decltype(active)>) {
      static_assert(std::is_same_v<
                        policy_details::remove_cvref_t<decltype(active.value)>,
                        Mask<To>>,
                    "load_convert mask must match the output tag");
      output_mask = active.value;
    } else {
      VECOPS_ASSERT(
          0 <= active.value && active.value <= size(to),
          "load_convert count %zd !in 0..%zd", active.value, size(to));
      output_mask = vec::mwhilelt(to, 0, active.value);
    }
  }

  constexpr bool native_path =
      policy_details::is_ordered<decltype(layout)> &&
      policy_details::is_unaligned<decltype(alignment)> &&
      policy_details::is_temporal<decltype(temporality)> &&
      policy_details::is_saturate<decltype(value_policy)>;
  if constexpr (native_path) {
    if constexpr (active_count == 0) {
      return details::fused_load_saturate(to, p);
    } else {
      if constexpr (policy_details::is_zero<decltype(population)>) {
        return details::fused_load_saturate(to, p, output_mask);
      } else {
        return details::fused_load_saturate(
            to, p, output_mask,
            details::population_value(to, population));
      }
    }
  } else {
    Vec<Rebind<Ei, To>> loaded;
    if constexpr (active_count == 0) {
      if constexpr (policy_details::is_aligned<decltype(alignment)>) {
        loaded = vec::load(ti, p, mem::aligned);
      } else {
        loaded = vec::load(ti, p);
      }
      return vec::convert(to, loaded, layout, value_policy);
    } else {
      Mask<Rebind<Ei, To>> input_mask;
      if constexpr (policy_details::is_unordered<decltype(layout)>) {
        auto&& active = policy_details::select_option<
            policy_details::IsActive>(
            opt::first(size(to)), std::forward<Options>(options)...);
        if constexpr (policy_details::is_masked<decltype(active)>) {
          static_assert(std::is_same_v<
                            policy_details::remove_cvref_t<
                                decltype(active.value)>,
                            Mask<Rebind<Ei, To>>>,
                        "unordered load_convert mask must match the "
                        "memory-side tag");
          input_mask = active.value;
        } else {
          VECOPS_ASSERT(
              0 <= active.value && active.value <= size(ti),
              "load_convert count %zd !in 0..%zd",
              active.value, size(ti));
          input_mask = vec::mwhilelt(ti, 0, active.value);
        }
      } else {
        input_mask = vec::convert(ti, to, output_mask);
      }
      if constexpr (policy_details::is_aligned<decltype(alignment)>) {
        loaded = vec::load(
            ti, p, mem::aligned, opt::masked(input_mask));
      } else {
        loaded = vec::load(ti, p, opt::masked(input_mask));
      }
      auto converted = vec::convert(to, loaded, layout, value_policy);
      if constexpr (policy_details::is_unordered<decltype(layout)> ||
                    policy_details::is_zero<decltype(population)>) {
        return converted;
      } else {
        return vec::blend(
            details::population_value(to, population),
            output_mask, converted);
      }
    }
  }
}

/**
 * @brief Conversion fused with a consecutive store.
 *
 * For an unordered conversion, opt::masked uses the memory-side mask.
 */
template <TLV_DECL_TAG(Ti), typename Eo, typename... Options>
  requires (
      is_element_type<Eo> &&
      details::valid_memory_conversion<
          true, TypeOf<Ti>, Eo, Options...>())
VECOPS_VFUNC void store_convert(
    Ti ti, Eo* p, Vec<Ti> vi, Options&&... options) {
  details::validate_memory_conversion_options<Options...>();
  static_assert(
      policy_details::count_options<
          policy_details::IsPopulation, Options...> == 0,
      "store_convert does not accept population options");
  auto&& layout = policy_details::select_option<
      policy_details::IsLayout>(
      cvt::ordered, std::forward<Options>(options)...);
  static_assert(policy_details::is_ordered<decltype(layout)> ||
                policy_details::is_unordered<decltype(layout)>,
                "store_convert supports ordered/unordered layouts only");

  auto&& alignment = policy_details::select_option<
      policy_details::IsAlignment>(
      mem::unaligned, std::forward<Options>(options)...);
  auto&& temporality = policy_details::select_option<
      policy_details::IsTemporality>(
      mem::temporal, std::forward<Options>(options)...);
  auto&& packing = policy_details::select_option<
      policy_details::IsPacking>(
      mem::packed, std::forward<Options>(options)...);
  auto&& value_policy = policy_details::select_option<
      policy_details::IsValuePolicy>(
      cvt::saturate, std::forward<Options>(options)...);
  constexpr int active_count =
      policy_details::count_options<policy_details::IsActive, Options...>;
  constexpr Rebind<Eo, Ti> to;

  Mask<Ti> input_mask;
  if constexpr (
      active_count != 0 &&
      policy_details::is_ordered<decltype(layout)>) {
    auto&& active = policy_details::select_option<
        policy_details::IsActive>(
        opt::first(size(ti)), std::forward<Options>(options)...);
    if constexpr (policy_details::is_masked<decltype(active)>) {
      static_assert(std::is_same_v<
                        policy_details::remove_cvref_t<decltype(active.value)>,
                        Mask<Ti>>,
                    "store_convert mask must match the input tag");
      input_mask = active.value;
    } else {
      VECOPS_ASSERT(
          0 <= active.value && active.value <= size(ti),
          "store_convert count %zd !in 0..%zd", active.value, size(ti));
      input_mask = vec::mwhilelt(ti, 0, active.value);
    }
  }

  constexpr bool native_path =
      policy_details::is_ordered<decltype(layout)> &&
      policy_details::is_unaligned<decltype(alignment)> &&
      policy_details::is_temporal<decltype(temporality)> &&
      policy_details::is_saturate<decltype(value_policy)>;
  if constexpr (native_path) {
    if constexpr (active_count == 0) {
      if constexpr (policy_details::is_split<decltype(packing)>)
        details::fused_store_saturate_split(ti, p, vi);
      else
        details::fused_store_saturate(ti, p, vi);
    } else {
      if constexpr (policy_details::is_split<decltype(packing)>)
        details::fused_store_saturate_split(ti, p, input_mask, vi);
      else
        details::fused_store_saturate(ti, p, input_mask, vi);
    }
  } else {
    auto converted = vec::convert(to, vi, layout, value_policy);
    if constexpr (active_count == 0) {
      if constexpr (policy_details::is_aligned<decltype(alignment)>) {
        vec::store(to, p, converted, mem::aligned);
      } else {
        vec::store(to, p, converted);
      }
    } else {
      Mask<Rebind<Eo, Ti>> output_mask;
      if constexpr (policy_details::is_unordered<decltype(layout)>) {
        auto&& active = policy_details::select_option<
            policy_details::IsActive>(
            opt::first(size(ti)), std::forward<Options>(options)...);
        if constexpr (policy_details::is_masked<decltype(active)>) {
          static_assert(std::is_same_v<
                            policy_details::remove_cvref_t<
                                decltype(active.value)>,
                            Mask<Rebind<Eo, Ti>>>,
                        "unordered store_convert mask must match the "
                        "memory-side tag");
          output_mask = active.value;
        } else {
          VECOPS_ASSERT(
              0 <= active.value && active.value <= size(to),
              "store_convert count %zd !in 0..%zd",
              active.value, size(to));
          output_mask = vec::mwhilelt(to, 0, active.value);
        }
      } else {
        output_mask = vec::convert(to, ti, input_mask);
      }
      if constexpr (policy_details::is_aligned<decltype(alignment)>) {
        vec::store(
            to, p, converted, mem::aligned, opt::masked(output_mask));
      } else {
        vec::store(to, p, converted, opt::masked(output_mask));
      }
    }
  }
}

/**
 * @brief Reinterpret cast vector bits to a different type.
 *
 * Reinterprets the bit pattern of the input vector as a different type
 * without any conversion. If output is smaller, high elements are discarded.
 * If output is larger, high elements are filled with undefined values.
 *
 * @tparam To Target type tag
 * @param t Target vector tag
 * @param v Input vector
 * @return Reinterpreted vector
 */
template <TLV_DECL_TAG(To), TLV_DECL_VEC(Vi)>
VECOPS_VFUNC Vec<To> bitcast(To t, Vi v) {
  using namespace details;
  using     Ti       = Vec2Tag<Vi>;
  constexpr Ti   t_i;                  constexpr To   t_o;
  constexpr auto NWi = num_words(t_i); constexpr auto NWo = num_words(t_o);
  using     TWi      = WordOf<Ti>;     using     TWo      = WordOf<To>;

  static_assert(!(is_scalable(t_i) ^ is_scalable(t_o)));
  constexpr nint_t n_cpy_iter = std::min(NWi, NWo);

  Vec<To> v_o;
  foreach<n_cpy_iter>([&]<nint_t I>{
    v_o = set_word<I>(t_o, v_o, word::bitcast(TWo(), get_word<I>(t_i, v)));
  });
  return v_o;
}

} // namespace vecops::vec

#endif //VECOPS_VEC_H
