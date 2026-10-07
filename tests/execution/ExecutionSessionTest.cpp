// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <type_traits>

#if defined(HAS_AMX_TILE) && defined(__linux__)
#include <sys/syscall.h>
#include <unistd.h>
#endif

#include "vecops/execution/ExecutionSession.h"

namespace {

using namespace vecops;

struct FakeStatefulGemm {
#if defined(HAS_SME_FA64)
  using ResourceRequirements = execution::details::ResourceSet<
      execution::details::arm::StreamingZA>;
#elif defined(HAS_AMX_TILE)
  using ResourceRequirements = execution::details::ResourceSet<
      execution::details::x86::Tiles>;
#else
  using ResourceRequirements = execution::details::ResourceSet<>;
#endif

#if defined(HAS_AMX_TILE)
  inline static constexpr auto tile_configuration = [] {
    execution::details::x86::TileConfiguration result{};
    result.column_bytes[0] = 64;
    result.rows[0] = 16;
    return result;
  }();
#endif

  template <execution::ExecutionScope Scope>
  VECOPS_ALWAYS_INLINE void operator()(
      Scope& scope, const float* a, const float* b, float* c) const {
    auto scalar_body = [&]() VECOPS_INLINE_LAMBDA {
      VECOPS_UNROLL
      for (int m = 0; m < 2; ++m) {
        VECOPS_UNROLL
        for (int n = 0; n < 2; ++n) {
          float sum = 0.0f;
          VECOPS_UNROLL
          for (int k = 0; k < 3; ++k) {
            sum += a[m * 3 + k] * b[k * 2 + n];
          }
          c[m * 2 + n] = sum;
        }
      }
    };

#if defined(HAS_AMX_TILE)
    scope.with_configuration(
        tile_configuration,
        [&](auto& configured) VECOPS_INLINE_LAMBDA {
          static_assert(std::same_as<
              typename std::remove_cvref_t<
                  decltype(configured)>::ActiveConfiguration,
              execution::details::x86::TileConfiguration>);
          scalar_body();
        });
#else
    (void)scope;
    scalar_body();
#endif
  }
};

static_assert(execution::ExecutionScope<ExecutionSession>);
#if defined(HAS_SME_FA64)
static_assert(!execution::details::has_resource_v<
    execution::details::arm::StreamingZA,
    ExecutionSession::ActiveResources>);
#elif defined(HAS_AMX_TILE)
static_assert(!execution::details::has_resource_v<
    execution::details::x86::Tiles,
    ExecutionSession::ActiveResources>);
#endif

#if defined(HAS_AMX_TILE) && defined(__linux__)
bool request_amx_permission() {
  constexpr unsigned long ArchRequestXcompPerm = 0x1023;
  constexpr unsigned long XfeatureTileData = 18;
  return syscall(
      SYS_arch_prctl, ArchRequestXcompPerm, XfeatureTileData) == 0;
}
#endif

TEST(ExecutionSessionTest, StatefulScalarGemmProducesExpectedValues) {
#if defined(HAS_AMX_TILE) && defined(__linux__)
  ASSERT_TRUE(request_amx_permission());
#endif
  const std::array<float, 6> a{1, 2, 3, 4, 5, 6};
  const std::array<float, 6> b{7, 8, 9, 10, 11, 12};
  std::array<float, 4> c{};
  FakeStatefulGemm gemm;
  ExecutionSession execution{};
  execution.with_region(
      gemm, [&](auto& region) VECOPS_INLINE_LAMBDA_NOEXCEPT {
#if defined(HAS_SME_FA64)
        static_assert(execution::has_resource_v<
            execution::details::arm::StreamingZA, decltype(region)>);
#elif defined(HAS_AMX_TILE)
        static_assert(execution::has_resource_v<
            execution::details::x86::Tiles, decltype(region)>);
#endif
        gemm(region, a.data(), b.data(), c.data());
      });
  EXPECT_FLOAT_EQ(c[0], 58.0f);
  EXPECT_FLOAT_EQ(c[1], 64.0f);
  EXPECT_FLOAT_EQ(c[2], 139.0f);
  EXPECT_FLOAT_EQ(c[3], 154.0f);
}

extern "C" VECOPS_NOINLINE void fake_stateful_gemm_probe(
    const float* a, const float* b, float* c) {
  FakeStatefulGemm gemm;
  ExecutionSession execution{};
  execution.with_region(
      gemm, [&](auto& region) VECOPS_INLINE_LAMBDA_NOEXCEPT {
        // Model an operator-level resource scope nested inside the outer
        // region. Its requirements are already active and must not emit a
        // second AMX/streaming transition.
        region.with_resources(
            gemm, [&](auto& operation) VECOPS_INLINE_LAMBDA {
              gemm(operation, a, b, c);
            });
      });
}

} // namespace
