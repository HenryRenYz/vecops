// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
//
//

#include "vecops/platform/Features.h"

#include <gtest/gtest.h>

#include <type_traits>
#include <utility>
#include <vector>

#include "MatmulTestArch.h"
#include "vecops/ops/Matmul.h"

namespace {

using namespace vecops;
using namespace vecops::meta;
using namespace vecops::tensor;

#if defined(ARCH_X86_FAMILY)
using PrimaryAtom = ::vecops::matmul::AMX_BF16F32;
using MixedAtom = ::vecops::matmul::AMX_I8I32<int8_t, uint8_t>;
#else
using PrimaryAtom = ::vecops::matmul::SME_BF16F32;
using MixedAtom = ::vecops::matmul::SME_I8I32<int8_t, uint8_t>;
#endif

template <typename Atom>
using SwapDisabledConfig = ops::MatmulConfig<
    Atom, ::vecops::matmul::family_selection::Automatic,
    kernel::matmul_policy::Automatic,
    ::vecops::matmul::GenericTiledTuning<>,
    platform::SystemCacheInfoProvider, false>;

template <bool Transposed, typename T, nint_t Rows, nint_t Columns>
constexpr auto matrix(T* data) {
  if constexpr (Transposed) {
    return make_tensor(
        data, make_layout(
                  make_shape(cint<Rows>, cint<Columns>),
                  make_strides(cint<1>, cint<Rows>)));
  } else {
    return make_tensor(
        data, make_layout(
                  make_shape(cint<Rows>, cint<Columns>),
                  make_strides(cint<Columns>, cint<1>)));
  }
}

template <bool AT, bool BT, bool CT, typename Config,
          nint_t M = 64, nint_t N = 128, nint_t K = 256,
          typename MemoryA = typename Config::Atom::TA,
          typename MemoryB = typename Config::Atom::TB>
struct OrientationProblem {
  using Atom = typename Config::Atom;
  using TA = typename Atom::TA;
  using TB = typename Atom::TB;
  using Acc = typename Atom::TAcc;
  using ATensor =
      decltype(matrix<AT, MemoryA, M, K>(static_cast<MemoryA*>(nullptr)));
  using ConventionalBTensor =
      decltype(matrix<BT, MemoryB, K, N>(static_cast<MemoryB*>(nullptr)));
  using BTensor = decltype(tensor::transpose_view<0, 1>(
      std::declval<ConventionalBTensor>()));
  using CTensor =
      decltype(matrix<CT, Acc, M, N>(static_cast<Acc*>(nullptr)));
  using ASpec = decltype(tensor::input<TA>(std::declval<ATensor>()));
  using BSpec = decltype(tensor::input<TB>(std::declval<BTensor>()));
  using CInputSpec = decltype(tensor::input<Acc>(
      std::declval<CTensor>(), tensor::zeros_transform<Acc, Acc>));
  using COutputSpec =
      decltype(tensor::output<Acc>(std::declval<CTensor>()));
  using Invocation = decltype(
      ::vecops::matmul::details::make_matmul_invocation(
          std::declval<const Config&>(), cint<M>, cint<N>, cint<K>,
          std::declval<ASpec>(), std::declval<BSpec>(),
          std::declval<CInputSpec>(), std::declval<COutputSpec>()));
};

template <bool AT, bool BT, bool CT, typename Config,
          nint_t M = 64, nint_t N = 128, nint_t K = 256,
          typename MemoryA = typename Config::Atom::TA,
          typename MemoryB = typename Config::Atom::TB>
consteval bool orientation_decision() {
  using Invocation = typename OrientationProblem<
      AT, BT, CT, Config, M, N, K, MemoryA, MemoryB>::Invocation;
  return Invocation::swaps_ab;
}

using DirectionalPacking = ::vecops::matmul::MatmulPackingTuning<
    ::vecops::matmul::packing_policy::InsideOnly,
    ::vecops::matmul::packing_policy::OutsideOnly>;
using DirectionalPackingConfig = ops::MatmulConfigWithPacking<
    PrimaryAtom, DirectionalPacking>;

using UnswappedDirectionalInvocation = typename OrientationProblem<
    false, false, false, DirectionalPackingConfig>::Invocation;
static_assert(!UnswappedDirectionalInvocation::swaps_ab);
static_assert(std::same_as<
    typename UnswappedDirectionalInvocation::APackingPolicy,
    ::vecops::matmul::packing_policy::InsideOnly>);
static_assert(std::same_as<
    typename UnswappedDirectionalInvocation::BPackingPolicy,
    ::vecops::matmul::packing_policy::OutsideOnly>);

#if defined(ARCH_X86_FAMILY)
// Policies name logical operands and therefore follow them through the AMX
// orientation exchange: physical A receives logical B's policy and vice
// versa.
using SwappedDirectionalInvocation = typename OrientationProblem<
    true, true, false, DirectionalPackingConfig>::Invocation;
static_assert(SwappedDirectionalInvocation::swaps_ab);
static_assert(std::same_as<
    typename SwappedDirectionalInvocation::APackingPolicy,
    ::vecops::matmul::packing_policy::OutsideOnly>);
static_assert(std::same_as<
    typename SwappedDirectionalInvocation::BPackingPolicy,
    ::vecops::matmul::packing_policy::InsideOnly>);

using SwappedRequiredPacking = ::vecops::matmul::MatmulPackingTuning<
    ::vecops::matmul::packing_policy::InsideOnly,
    ::vecops::matmul::packing_policy::RequireOutside>;
using SwappedRequiredConfig = ops::MatmulConfigWithPacking<
    PrimaryAtom, SwappedRequiredPacking>;
using SwappedRequiredInvocation = typename OrientationProblem<
    true, true, false, SwappedRequiredConfig>::Invocation;
static_assert(SwappedRequiredInvocation::swaps_ab);
static_assert(std::same_as<
    typename SwappedRequiredInvocation::APackingPolicy,
    ::vecops::matmul::packing_policy::RequireOutside>);
static_assert(std::same_as<
    typename SwappedRequiredInvocation::BPackingPolicy,
    ::vecops::matmul::packing_policy::InsideOnly>);
#endif

TEST(MatmulOrientationTest, ArchitecturePolicyClassifiesEightLayouts) {
  using Enabled = ops::MatmulConfig<PrimaryAtom>;
  using Disabled = SwapDisabledConfig<PrimaryAtom>;
  static_assert(!orientation_decision<false, false, false, Enabled>());
  static_assert(!orientation_decision<true, false, false, Enabled>());
#if defined(ARCH_X86_FAMILY)
  static_assert(!orientation_decision<false, true, false, Enabled>());
  static_assert(!orientation_decision<false, false, true, Enabled>());
  static_assert(orientation_decision<true, true, false, Enabled>());
  static_assert(!orientation_decision<true, false, true, Enabled>());
  static_assert(!orientation_decision<
      true, false, true, Enabled, 128, 64, 256>());
  static_assert(orientation_decision<false, true, true, Enabled>());
  static_assert(orientation_decision<true, true, true, Enabled>());
  static_assert(orientation_decision<
      true, true, true, Enabled, 64, 128, 256, float32_t, float32_t>());
  static_assert(orientation_decision<
      true, true, false, Enabled, 64, 128, 16>());
  static_assert(orientation_decision<
      true, true, false, Enabled, 64, 128, 32>());
  static_assert(orientation_decision<
      true, true, false, Enabled, 4, 8, 32>());
  static_assert(!orientation_decision<
      true, true, false, Enabled, 4, 32, 32>());
#else
  static_assert(!orientation_decision<false, true, false, Enabled>());
  static_assert(!orientation_decision<false, false, true, Enabled>());
  static_assert(!orientation_decision<true, true, false, Enabled>());
  static_assert(!orientation_decision<true, false, true, Enabled>());
  static_assert(!orientation_decision<false, true, true, Enabled>());
  static_assert(!orientation_decision<true, true, true, Enabled>());
  static_assert(!orientation_decision<
      true, true, true, Enabled, 64, 128, 256, float32_t, float32_t>());
#endif
  static_assert(!orientation_decision<false, false, false, Disabled>());
  static_assert(!orientation_decision<true, true, true, Disabled>());
  static_assert(!orientation_decision<
      false, false, false, Enabled, 8, 1, 128>());
  static_assert(!orientation_decision<
      true, true, true, Enabled, 1, 8, 128>());
  static_assert(std::is_same_v<
      typename OrientationProblem<
          false, false, false, Enabled>::Invocation,
      typename OrientationProblem<
          false, false, false, Disabled>::Invocation>);
#if defined(ARCH_X86_FAMILY)
  using SwappedMixed = ::vecops::matmul::AMX_I8I32<uint8_t, int8_t>;
#else
  using SwappedMixed = ::vecops::matmul::SME_I8I32<uint8_t, int8_t>;
#endif
  static_assert(
      std::is_same_v<typename MixedAtom::SwappedAtom, SwappedMixed>);
}

template <typename T>
T input_value(nint_t value, nint_t modulus) {
  if constexpr (std::is_unsigned_v<T>) {
    return static_cast<T>(value % modulus);
  } else if constexpr (std::is_integral_v<T>) {
    return static_cast<T>(value % modulus - modulus / 2);
  } else {
    return static_cast<T>(
        static_cast<float>(value % modulus - modulus / 2) / 16.0f);
  }
}

template <typename Atom, bool AT, bool BT, bool CT, bool EnableSwap,
          nint_t M = 17, nint_t N = 23, nint_t K = 128>
void check_product() {
  using TA = typename Atom::TA;
  using TB = typename Atom::TB;
  using Acc = typename Atom::TAcc;
  std::vector<TA> a(static_cast<std::size_t>(M * K));
  std::vector<TB> b(static_cast<std::size_t>(K * N));
  std::vector<Acc> c(static_cast<std::size_t>(M * N), Acc{});
  auto a_tensor = matrix<AT, TA, M, K>(a.data());
  auto conventional_b = matrix<BT, TB, K, N>(b.data());
  auto b_tensor = tensor::transpose_view<0, 1>(conventional_b);
  auto c_tensor = matrix<CT, Acc, M, N>(c.data());
  for (nint_t i = 0; i < M; ++i)
    for (nint_t kk = 0; kk < K; ++kk)
      a_tensor(i, kk) = input_value<TA>(i * 3 + kk * 5, 13);
  for (nint_t kk = 0; kk < K; ++kk)
    for (nint_t j = 0; j < N; ++j)
      conventional_b(kk, j) = input_value<TB>(kk * 7 + j * 3, 11);

  using Config = std::conditional_t<
      EnableSwap, ops::MatmulConfig<Atom>, SwapDisabledConfig<Atom>>;
  auto operation = ops::matmul(Config{});
  kernel::Workspace storage(operation.required_workspace(
      cint<M>, cint<N>, cint<K>, a_tensor, b_tensor, c_tensor));
  auto workspace = storage.view();
  operation(
      workspace, cint<M>, cint<N>, cint<K>,
      a_tensor, b_tensor, c_tensor);

  for (nint_t i = 0; i < M; ++i) {
    for (nint_t j = 0; j < N; ++j) {
      Acc expected{};
      for (nint_t kk = 0; kk < K; ++kk)
        expected += static_cast<Acc>(a_tensor(i, kk)) *
            static_cast<Acc>(conventional_b(kk, j));
      if constexpr (std::is_floating_point_v<Acc>)
        EXPECT_NEAR(c_tensor(i, j), expected, 4.0e-4f);
      else
        EXPECT_EQ(c_tensor(i, j), expected);
    }
  }
}

TEST(MatmulOrientationTest, AllLayoutsProduceTheReferenceResult) {
  ASSERT_TRUE(vecops::test::matmul::MatmulTestArchTraits::enable());
  check_product<PrimaryAtom, false, false, false, true>();
  check_product<PrimaryAtom, true,  false, false, true>();
  check_product<PrimaryAtom, false, true,  false, true>();
  check_product<PrimaryAtom, false, false, true,  true>();
  check_product<PrimaryAtom, true,  true,  false, true>();
  check_product<PrimaryAtom, true,  false, true,  true>();
  check_product<PrimaryAtom, false, true,  true,  true>();
  check_product<PrimaryAtom, true,  true,  true,  true>();

  // Exercise the unswapped vertical-C implementation for every input layout.
  check_product<PrimaryAtom, false, false, true, false>();
  check_product<PrimaryAtom, true,  false, true, false>();
  check_product<PrimaryAtom, false, true,  true, false>();
  check_product<PrimaryAtom, true,  true,  true, false>();

  // Attention-oriented tails: short K and N smaller than one accumulator tile.
  check_product<PrimaryAtom, false, false, true, false, 19, 4, 16>();
  check_product<PrimaryAtom, false, false, true, false, 19, 8, 16>();
  check_product<PrimaryAtom, true, true, true, true, 32, 32, 16>();
#if !defined(ARCH_X86_FAMILY) || defined(HAS_AMX_INT8)
  check_product<MixedAtom, true, true, true, true>();
  check_product<MixedAtom, true, true, true, false>();
#endif
}

TEST(MatmulOrientationTest, SwappedProblemPreservesFusedCTransforms) {
  ASSERT_TRUE(vecops::test::matmul::MatmulTestArchTraits::enable());
  constexpr nint_t M = 17;
  constexpr nint_t N = 23;
  constexpr nint_t K = 128;
  using Atom = PrimaryAtom;
  using TA = typename Atom::TA;
  using TB = typename Atom::TB;
  using Acc = typename Atom::TAcc;
  std::vector<TA> a(static_cast<std::size_t>(M * K));
  std::vector<TB> b(static_cast<std::size_t>(K * N));
  std::vector<Acc> c(static_cast<std::size_t>(M * N));
  auto a_tensor = matrix<true, TA, M, K>(a.data());
  auto conventional_b = matrix<true, TB, K, N>(b.data());
  auto b_tensor = tensor::transpose_view<0, 1>(conventional_b);
  auto c_tensor = matrix<true, Acc, M, N>(c.data());
  for (nint_t i = 0; i < M; ++i) {
    for (nint_t kk = 0; kk < K; ++kk) {
      a_tensor(i, kk) = static_cast<TA>(
          static_cast<float>((i * 3 + kk * 5) % 13 - 6) / 16.0f);
    }
  }
  for (nint_t kk = 0; kk < K; ++kk) {
    for (nint_t j = 0; j < N; ++j) {
      conventional_b(kk, j) = static_cast<TB>(
          static_cast<float>((kk * 7 + j * 3) % 11 - 5) / 16.0f);
    }
  }
  for (nint_t i = 0; i < M; ++i) {
    for (nint_t j = 0; j < N; ++j) {
      c_tensor(i, j) = static_cast<Acc>(
          static_cast<float>((i + j) % 7 - 3) / 8.0f);
    }
  }
  auto scale = tensor::make_elementwise_vec_transform<Acc, Acc>(
      [](auto tag, auto value) VECOPS_KERNEL_LAMBDA {
        return vec::mul(value, vec::fill(tag, Acc{2}));
      });
  auto c_input = tensor::input<Acc>(c_tensor);
  auto c_output = tensor::output<Acc>(c_tensor, scale);
  using Config = ops::MatmulConfig<Atom>;
  using Invocation = decltype(
      ::vecops::matmul::details::make_matmul_invocation(
          std::declval<const Config&>(), cint<M>, cint<N>, cint<K>,
          std::declval<decltype(tensor::input<TA>(a_tensor))>(),
          std::declval<decltype(tensor::input<TB>(b_tensor))>(),
          std::declval<decltype(c_input)>(),
          std::declval<decltype(c_output)>()));
#if defined(ARCH_X86_FAMILY)
  static_assert(Invocation::swaps_ab);
#else
  // SME keeps non-zero CInput in its original orientation so broadcast and
  // vertical initialization retain their dedicated paths.
  static_assert(!Invocation::swaps_ab);
#endif

  auto operation = ops::matmul(Config{});
  kernel::Workspace storage(operation.required_workspace(
      cint<M>, cint<N>, cint<K>, a_tensor, b_tensor, c_input, c_output));
  auto workspace = storage.view();
  operation(
      workspace, cint<M>, cint<N>, cint<K>,
      a_tensor, b_tensor, c_input, c_output);
  for (nint_t i = 0; i < M; ++i) {
    for (nint_t j = 0; j < N; ++j) {
      Acc expected = static_cast<Acc>(
          static_cast<float>((i + j) % 7 - 3) / 8.0f);
      for (nint_t kk = 0; kk < K; ++kk) {
        expected += static_cast<Acc>(a_tensor(i, kk)) *
            static_cast<Acc>(conventional_b(kk, j));
      }
      EXPECT_NEAR(c_tensor(i, j), Acc{2} * expected, 8.0e-4f);
    }
  }
}

TEST(MatmulOrientationTest, VerticalOutputPreservesBiasAndOutputTransform) {
  ASSERT_TRUE(vecops::test::matmul::MatmulTestArchTraits::enable());
  constexpr nint_t M = 19;
  constexpr nint_t N = 7;
  constexpr nint_t K = 16;
  using Atom = PrimaryAtom;
  using TA = typename Atom::TA;
  using TB = typename Atom::TB;
  using Acc = typename Atom::TAcc;
  std::vector<TA> a(static_cast<std::size_t>(M * K));
  std::vector<TB> b(static_cast<std::size_t>(K * N));
  std::vector<Acc> bias(static_cast<std::size_t>(N));
  std::vector<Acc> c(static_cast<std::size_t>(M * N));
  auto a_tensor = matrix<false, TA, M, K>(a.data());
  auto conventional_b = matrix<false, TB, K, N>(b.data());
  auto b_tensor = tensor::transpose_view<0, 1>(conventional_b);
  auto bias_tensor = tensor::make_tensor(
      bias.data(), tensor::make_layout(
                       tensor::make_shape(cint<M>, cint<N>),
                       tensor::make_strides(cint<0>, cint<1>)));
  auto c_tensor = matrix<true, Acc, M, N>(c.data());
  for (nint_t i = 0; i < M; ++i)
    for (nint_t kk = 0; kk < K; ++kk)
      a_tensor(i, kk) = input_value<TA>(i * 3 + kk * 5, 13);
  for (nint_t kk = 0; kk < K; ++kk)
    for (nint_t j = 0; j < N; ++j)
      conventional_b(kk, j) = input_value<TB>(kk * 7 + j * 3, 11);
  for (nint_t j = 0; j < N; ++j)
    bias[j] = static_cast<Acc>(static_cast<float>(j - 3) / 8.0f);

  auto scale = tensor::make_elementwise_vec_transform<Acc, Acc>(
      [](auto tag, auto value) VECOPS_KERNEL_LAMBDA {
        return vec::mul(value, vec::fill(tag, Acc{2}));
      });
  auto c_input = tensor::input<Acc>(bias_tensor);
  auto c_output = tensor::output<Acc>(c_tensor, scale);
  using Config = SwapDisabledConfig<Atom>;
  auto operation = ops::matmul(Config{});
  kernel::Workspace storage(operation.required_workspace(
      cint<M>, cint<N>, cint<K>, a_tensor, b_tensor, c_input, c_output));
  auto workspace = storage.view();
  operation(
      workspace, cint<M>, cint<N>, cint<K>,
      a_tensor, b_tensor, c_input, c_output);

  for (nint_t i = 0; i < M; ++i) {
    for (nint_t j = 0; j < N; ++j) {
      Acc expected = bias[j];
      for (nint_t kk = 0; kk < K; ++kk) {
        expected += static_cast<Acc>(a_tensor(i, kk)) *
            static_cast<Acc>(conventional_b(kk, j));
      }
      EXPECT_NEAR(c_tensor(i, j), Acc{2} * expected, 8.0e-4f);
    }
  }
}

} // namespace
