// Select the real public hardware atom for the paired native baseline.
#pragma once
#include "vecops/matmul/Atom.h"

template<class A, class B>
constexpr auto benchmark_atom_type() {
#if defined(HAS_SME)
  if constexpr (std::same_as<A, vecops::bfloat16_t>)
    return std::type_identity<vecops::matmul::SME_BF16F32>{};
  else if constexpr (std::same_as<A, vecops::float16_t>)
    return std::type_identity<vecops::matmul::SME_F16F32>{};
  else if constexpr (std::same_as<A, vecops::float32_t>)
    return std::type_identity<vecops::matmul::SME_F32F32>{};
#if defined(HAS_SME_F64F64)
  else if constexpr (std::same_as<A, vecops::float64_t>)
    return std::type_identity<vecops::matmul::SME_F64F64>{};
#endif
  else
    return std::type_identity<vecops::matmul::SME_I8I32<A, B>>{};
#else
  if constexpr (std::same_as<A, vecops::bfloat16_t>)
    return std::type_identity<vecops::matmul::AMX_BF16F32>{};
  else if constexpr (std::same_as<A, vecops::float16_t>)
    return std::type_identity<vecops::matmul::AMX_F16F32>{};
  else
    return std::type_identity<vecops::matmul::AMX_I8I32<A, B>>{};
#endif
}

template<class A, class B>
using BenchmarkAtom = typename decltype(benchmark_atom_type<A, B>())::type;
