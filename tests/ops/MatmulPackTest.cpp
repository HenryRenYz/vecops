// @vecops-target-shards-x86: 9
// @vecops-target-shards-ARM: 6

#include "vecops/Features.h"

#if defined(ARCH_X86_FAMILY)

#if defined(VECOPS_TARGET_SHARD_ACTIVE)

#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <limits>
#include <type_traits>
#include <vector>

#include "TestUtils.h"
#include "vecops/matmul/Atom.h"
#include "vecops/kernel/Workspace.h"
#include "vecops/ops/MatmulPack.h"

namespace {

static_assert(VECOPS_TARGET_SHARD_COUNT == 9);

using namespace vecops;
using namespace vecops::meta;
using namespace vecops::tensor;

template <typename T>
struct PackedStorage {
  kernel::Workspace owner;
  kernel::WorkspaceView view;
  T* data;

  explicit PackedStorage(nint_t elements)
      : owner(elements * static_cast<nint_t>(sizeof(T)) + 64),
        view(owner.view()),
        data(static_cast<T*>(view.allocate(
            elements * static_cast<nint_t>(sizeof(T)), 64))) {}
};

template <typename Atom, gemm::Operand Side>
void check_direct_pack(nint_t spatial, nint_t k, nint_t padding) {
  using Packing = gemm::packing_t<Atom, Side>;
  using T = typename Packing::Element;
  const nint_t stride_k = 1;
  const nint_t stride_spatial = k + padding;
  std::vector<T> input(static_cast<std::size_t>(
      spatial * stride_spatial));
  for (nint_t s = 0; s < spatial; ++s) {
    for (nint_t kk = 0; kk < k; ++kk) {
      input[static_cast<std::size_t>(s * stride_spatial + kk)] =
          test_utils::get_test_value<T>(static_cast<int>(s * 131 + kk));
    }
  }
  auto input_layout = make_layout(
      make_shape(Any{spatial}, Any{k}),
      make_strides(Any{stride_spatial}, cint<stride_k>));
  auto input_tensor = make_tensor(input.data(), input_layout);
  auto output_layout = ops::matmul_packed_layout<Atom, Side>(input_layout);
  PackedStorage<T> storage(numel(output_layout));
  std::fill_n(storage.data, numel(output_layout),
              test_utils::get_test_value<T>(77));
  auto output_tensor = make_tensor(storage.data, output_layout);
  auto operation = ops::make_matmul_pack<Atom, Side>(
      input_tensor, output_tensor);
  static_assert(std::same_as<
      typename decltype(operation)::ResourceRequirements,
      typename execution::details::current_backend_t::DefaultRequirements>);
  ExecutionSession execution{};
  operation(execution);

  const T* packed = storage.data;
  if constexpr (Side == gemm::Operand::A) {
    for (nint_t sp = 0; sp < ceil_div(spatial, Packing::Panel); ++sp) {
      for (nint_t kt = 0; kt < ceil_div(k, Packing::KTile); ++kt) {
        for (nint_t lane = 0; lane < Packing::Panel; ++lane) {
          for (nint_t ki = 0; ki < Packing::KTile; ++ki) {
            const nint_t s = sp * Packing::Panel + lane;
            const nint_t kk = kt * Packing::KTile + ki;
            const T expected = (s < spatial && kk < k)
                ? input[static_cast<std::size_t>(
                      s * stride_spatial + kk)]
                : T{};
            EXPECT_TRUE(test_utils::values_equal(expected, *packed))
                << "s=" << s << " k=" << kk;
            ++packed;
          }
        }
      }
    }
  } else {
    for (nint_t sp = 0; sp < ceil_div(spatial, Packing::Panel); ++sp) {
      for (nint_t kt = 0; kt < ceil_div(k, Packing::KTile); ++kt) {
        for (nint_t kg = 0; kg < Packing::KTile / Packing::KPack; ++kg) {
          for (nint_t lane = 0; lane < Packing::Panel; ++lane) {
            for (nint_t ki = 0; ki < Packing::KPack; ++ki) {
              const nint_t s = sp * Packing::Panel + lane;
              const nint_t kk = kt * Packing::KTile +
                  kg * Packing::KPack + ki;
              const T expected = (s < spatial && kk < k)
                  ? input[static_cast<std::size_t>(
                        s * stride_spatial + kk)]
                  : T{};
              EXPECT_TRUE(test_utils::values_equal(expected, *packed))
                  << "s=" << s << " k=" << kk;
              ++packed;
            }
          }
        }
      }
    }
  }
  EXPECT_EQ(packed, storage.data + numel(output_layout));
}

template <typename InputOperand, typename InputLayout>
void check_compensated_b_pack(
    const InputOperand& input_operand, const InputLayout& input_layout,
    const std::vector<int8_t>& logical_input,
    nint_t n, nint_t k, int32_t a_zero_point) {
  using Atom = gemm::AMX_I8I32<uint8_t, int8_t>;
  using Packing = gemm::packing_t<Atom, gemm::Operand::B>;
  auto output_layout = ops::matmul_packed_layout<
      Atom, gemm::Operand::B>(input_layout);
  PackedStorage<int8_t> storage(numel(output_layout));
  auto output_tensor = make_tensor(storage.data, output_layout);
  constexpr int32_t Canary = 0x13579bdf;
  std::vector<int32_t> compensation(
      static_cast<std::size_t>(n + 1), Canary);
  auto compensation_tensor = make_tensor(
      compensation.data(),
      make_layout(make_shape(Any{n})));
  auto operation = ops::make_matmul_pack_b_compensated<Atom>(
      input_operand, output_tensor, compensation_tensor, a_zero_point);
  static_assert(std::same_as<
      typename decltype(operation)::ResourceRequirements,
      typename execution::details::current_backend_t::DefaultRequirements>);
  EXPECT_EQ(operation.required_workspace(), 0);
  ExecutionSession execution{};
  operation(execution);

  const int8_t* packed = storage.data;
  for (nint_t sp = 0; sp < ceil_div(n, Packing::Panel); ++sp) {
    for (nint_t kt = 0; kt < ceil_div(k, Packing::KTile); ++kt) {
      for (nint_t kg = 0; kg < Packing::KTile / Packing::KPack; ++kg) {
        for (nint_t lane = 0; lane < Packing::Panel; ++lane) {
          for (nint_t ki = 0; ki < Packing::KPack; ++ki) {
            const nint_t row = sp * Packing::Panel + lane;
            const nint_t kk = kt * Packing::KTile +
                kg * Packing::KPack + ki;
            const int8_t expected = row < n && kk < k
                ? logical_input[static_cast<std::size_t>(row * k + kk)]
                : int8_t{};
            EXPECT_EQ(*packed++, expected)
                << "n=" << row << " k=" << kk;
          }
        }
      }
    }
  }
  EXPECT_EQ(packed, storage.data + numel(output_layout));
  for (nint_t row = 0; row < n; ++row) {
    int32_t sum = 0;
    for (nint_t kk = 0; kk < k; ++kk)
      sum += logical_input[static_cast<std::size_t>(row * k + kk)];
    EXPECT_EQ(compensation[static_cast<std::size_t>(row)],
              -a_zero_point * sum)
        << "n=" << row;
  }
  EXPECT_EQ(compensation[static_cast<std::size_t>(n)], Canary);
}

template <gemm::Operand Side, nint_t StrideK = 2,
          typename Atom = gemm::AMX_BF16F32>
void check_strided_conversion_and_transform() {
  using T = typename gemm::packing_t<Atom, Side>::Element;
  constexpr nint_t Spatial = 5;
  constexpr nint_t K = 9;
  constexpr nint_t StrideSpatial = K * StrideK + 3;
  std::vector<float32_t> input(
      static_cast<std::size_t>(Spatial * StrideSpatial), -999.0f);
  for (nint_t s = 0; s < Spatial; ++s) {
    for (nint_t k = 0; k < K; ++k) {
      input[static_cast<std::size_t>(s * StrideSpatial + k * StrideK)] =
          static_cast<float32_t>(s * 17 + k) + 0.25f;
    }
  }
  auto input_layout = make_layout(
      make_shape(cint<Spatial>, cint<K>),
      make_strides(cint<StrideSpatial>, cint<StrideK>));
  auto input_tensor = make_tensor(input.data(), input_layout);
  auto transform = make_elementwise_vec_transform<float32_t, float32_t>(
      []<vec::VectorTag Tag>(Tag tag, vec::Vec<Tag> value) {
        return vec::add(tag, value, vec::fill(tag, 1.0f));
      });
  auto input_spec = tensor::input<T>(input_tensor, transform);
  auto output_layout = ops::matmul_packed_layout<Atom, Side>(input_layout);
  PackedStorage<T> storage(numel(output_layout));
  auto output_tensor = make_tensor(storage.data, output_layout);
  ExecutionSession execution{};
  ops::matmul_pack<Atom, Side>(
      execution, input_spec, output_tensor);

  using Packing = gemm::packing_t<Atom, Side>;
  const T* packed = storage.data;
  const nint_t padded_k = ceil_div(K, Packing::KTile) * Packing::KTile;
  for (nint_t sp = 0; sp < ceil_div(Spatial, Packing::Panel); ++sp) {
    if constexpr (Side == gemm::Operand::A) {
      for (nint_t kt = 0; kt < ceil_div(K, Packing::KTile); ++kt) {
        for (nint_t lane = 0; lane < Packing::Panel; ++lane) {
          for (nint_t ki = 0; ki < Packing::KTile; ++ki) {
            const nint_t s = sp * Packing::Panel + lane;
            const nint_t k = kt * Packing::KTile + ki;
            T expected{};
            if (s < Spatial && k < K) {
              expected = static_cast<T>(input[static_cast<std::size_t>(
                  s * StrideSpatial + k * StrideK)] + 1.0f);
            }
            EXPECT_TRUE(test_utils::values_equal(expected, *packed++));
          }
        }
      }
    } else {
      for (nint_t kg = 0; kg < padded_k / Packing::KPack; ++kg) {
        for (nint_t lane = 0; lane < Packing::Panel; ++lane) {
          for (nint_t ki = 0; ki < Packing::KPack; ++ki) {
            const nint_t s = sp * Packing::Panel + lane;
            const nint_t k = kg * Packing::KPack + ki;
            T expected{};
            if (s < Spatial && k < K) {
              expected = static_cast<T>(input[static_cast<std::size_t>(
                  s * StrideSpatial + k * StrideK)] + 1.0f);
            }
            EXPECT_TRUE(test_utils::values_equal(expected, *packed++));
          }
        }
      }
    }
  }
  EXPECT_EQ(packed, storage.data + numel(output_layout));
}

using AMXI8 = gemm::AMX_I8I32<int8_t, uint8_t>;
using AMXA = decltype(gemm::packed_layout<AMXI8, gemm::Operand::A>(
    make_layout(make_shape(cint<17>, cint<65>))));
using AMXB = decltype(gemm::packed_layout<AMXI8, gemm::Operand::B>(
    make_layout(make_shape(cint<17>, cint<65>))));
static_assert(AMXA::Ndim == 4 && AMXB::Ndim == 5);
static_assert(gemm::is_packed_layout<AMXI8, gemm::Operand::A, AMXA>());
static_assert(gemm::is_packed_layout<AMXI8, gemm::Operand::B, AMXB>());

#if VECOPS_TARGET_SHARD_INDEX == 0
TEST(MatmulPackTest, BF16AndF16AFullAndTails) {
  check_direct_pack<gemm::AMX_BF16F32, gemm::Operand::A>(16, 32, 0);
  check_direct_pack<gemm::AMX_BF16F32, gemm::Operand::A>(17, 33, 3);
  check_direct_pack<gemm::AMX_F16F32, gemm::Operand::A>(16, 32, 0);
  check_direct_pack<gemm::AMX_F16F32, gemm::Operand::A>(1, 1, 5);
}

#elif VECOPS_TARGET_SHARD_INDEX == 1
TEST(MatmulPackTest, BF16AndF16BFullAndTails) {
  check_direct_pack<gemm::AMX_BF16F32, gemm::Operand::B>(16, 32, 0);
  check_direct_pack<gemm::AMX_BF16F32, gemm::Operand::B>(19, 35, 4);
  check_direct_pack<gemm::AMX_F16F32, gemm::Operand::B>(16, 32, 0);
  check_direct_pack<gemm::AMX_F16F32, gemm::Operand::B>(1, 1, 2);
}

#elif VECOPS_TARGET_SHARD_INDEX == 2
TEST(MatmulPackTest, AllInt8SignednessCombinations) {
  check_direct_pack<gemm::AMX_I8I32<int8_t, int8_t>, gemm::Operand::A>(
      16, 64, 0);
  check_direct_pack<gemm::AMX_I8I32<int8_t, int8_t>, gemm::Operand::B>(
      16, 64, 0);
  check_direct_pack<gemm::AMX_I8I32<int8_t, int8_t>, gemm::Operand::A>(
      17, 65, 1);
  check_direct_pack<gemm::AMX_I8I32<int8_t, int8_t>, gemm::Operand::B>(
      19, 67, 2);
  check_direct_pack<gemm::AMX_I8I32<int8_t, uint8_t>, gemm::Operand::A>(
      21, 69, 3);
  check_direct_pack<gemm::AMX_I8I32<int8_t, uint8_t>, gemm::Operand::B>(
      23, 71, 4);
  check_direct_pack<gemm::AMX_I8I32<uint8_t, int8_t>, gemm::Operand::A>(
      25, 73, 5);
  check_direct_pack<gemm::AMX_I8I32<uint8_t, int8_t>, gemm::Operand::B>(
      27, 75, 6);
  check_direct_pack<gemm::AMX_I8I32<uint8_t, uint8_t>, gemm::Operand::A>(
      29, 77, 7);
  check_direct_pack<gemm::AMX_I8I32<uint8_t, uint8_t>, gemm::Operand::B>(
      31, 79, 8);
}

#elif VECOPS_TARGET_SHARD_INDEX == 3
TEST(MatmulPackTest, StridedConversionAndTransformStayVectorized) {
  check_strided_conversion_and_transform<gemm::Operand::A>();
  check_strided_conversion_and_transform<gemm::Operand::B>();
  check_strided_conversion_and_transform<
      gemm::Operand::A, 2, gemm::AMX_F16F32>();
  check_strided_conversion_and_transform<
      gemm::Operand::B, 2, gemm::AMX_F16F32>();
}

#elif VECOPS_TARGET_SHARD_INDEX == 4
TEST(MatmulPackTest, RowMajorBConversionAndTransformMatchesPackedLayout) {
  check_strided_conversion_and_transform<gemm::Operand::B, 1>();
}

#elif VECOPS_TARGET_SHARD_INDEX == 5
TEST(MatmulPackTest, CompensatedBPackFusesDirectAndTransformedColumnSums) {
  constexpr int32_t ZeroPointA = 3;
  {
    constexpr nint_t N = 33;
    constexpr nint_t K = 129;
    constexpr nint_t Stride = K + 3;
    std::vector<int8_t> input(
        static_cast<std::size_t>(N * Stride), int8_t{-99});
    std::vector<int8_t> logical(static_cast<std::size_t>(N * K));
    for (nint_t n = 0; n < N; ++n) {
      for (nint_t k = 0; k < K; ++k) {
        const auto value = static_cast<int8_t>((n * 17 + k * 3) % 61 - 30);
        input[static_cast<std::size_t>(n * Stride + k)] = value;
        logical[static_cast<std::size_t>(n * K + k)] = value;
      }
    }
    auto layout = make_layout(
        make_shape(cint<N>, cint<K>),
        make_strides(cint<Stride>, cint<1>));
    auto input_tensor = make_tensor(input.data(), layout);
    check_compensated_b_pack(
        input_tensor, layout, logical, N, K, ZeroPointA);
  }
  {
    constexpr nint_t N = 19;
    constexpr nint_t K = 67;
    std::vector<float32_t> input(static_cast<std::size_t>(N * K));
    std::vector<int8_t> logical(static_cast<std::size_t>(N * K));
    for (nint_t i = 0; i < N * K; ++i) {
      const auto quantized = static_cast<int8_t>((i * 5) % 63 - 31);
      input[static_cast<std::size_t>(i)] =
          static_cast<float32_t>(quantized) * 0.25f;
      logical[static_cast<std::size_t>(i)] = quantized;
    }
    auto layout = make_layout(make_shape(Any{N}, Any{K}));
    auto input_tensor = make_tensor(input.data(), layout);
    auto transform = make_elementwise_vec_transform<float32_t, float32_t>(
        []<vec::VectorTag Tag>(Tag tag, vec::Vec<Tag> value) {
          return vec::mul(tag, value, vec::fill(tag, float32_t{4}));
        });
    auto input_spec = tensor::input<int8_t>(input_tensor, transform);
    check_compensated_b_pack(
        input_spec, layout, logical, N, K, ZeroPointA);
  }
}

#elif VECOPS_TARGET_SHARD_INDEX == 6
TEST(MatmulPackTest, CompensatedBPackHandlesZeroExtents) {
  using Atom = gemm::AMX_I8I32<uint8_t, int8_t>;
  ExecutionSession execution{};
  {
    constexpr nint_t N = 17;
    auto input_layout = make_layout(make_shape(cint<N>, cint<0>));
    auto output_layout = ops::matmul_packed_layout<
        Atom, gemm::Operand::B>(input_layout);
    int8_t* pointer = nullptr;
    std::vector<int32_t> compensation(N, 7);
    auto compensation_tensor = make_tensor(
        compensation.data(), make_layout(make_shape(cint<N>)));
    ops::matmul_pack_b_compensated<Atom>(
        execution, make_tensor(pointer, input_layout),
        make_tensor(pointer, output_layout), compensation_tensor, 3);
    EXPECT_TRUE(std::all_of(
        compensation.begin(), compensation.end(),
        [](int32_t value) { return value == 0; }));
  }
  {
    auto input_layout = make_layout(make_shape(cint<0>, cint<67>));
    auto output_layout = ops::matmul_packed_layout<
        Atom, gemm::Operand::B>(input_layout);
    int8_t* pointer = nullptr;
    int32_t compensation = 11;
    auto compensation_tensor = make_tensor(
        &compensation, make_layout(make_shape(cint<0>)));
    ops::matmul_pack_b_compensated<Atom>(
        execution, make_tensor(pointer, input_layout),
        make_tensor(pointer, output_layout), compensation_tensor, 3);
    EXPECT_EQ(compensation, 11);
  }
}

#elif VECOPS_TARGET_SHARD_INDEX == 7
TEST(MatmulPackTest, ZeroExtentDoesNotAccessStorage) {
  auto input_layout = make_layout(make_shape(cint<0>, cint<7>));
  auto output_layout = ops::matmul_packed_layout<
      gemm::AMX_BF16F32, gemm::Operand::A>(input_layout);
  EXPECT_EQ(numel(output_layout), 0);
  bfloat16_t* pointer = nullptr;
  auto input = make_tensor(pointer, input_layout);
  auto output = make_tensor(pointer, output_layout);
  ExecutionSession execution{};
  ops::matmul_pack<gemm::AMX_BF16F32, gemm::Operand::A>(
      execution, input, output);

  auto zero_k_layout = make_layout(make_shape(cint<7>, cint<0>));
  auto zero_k_packed_layout = ops::matmul_packed_layout<
      gemm::AMX_BF16F32, gemm::Operand::B>(zero_k_layout);
  EXPECT_EQ(numel(zero_k_packed_layout), 0);
  auto zero_k_input = make_tensor(pointer, zero_k_layout);
  auto zero_k_output = make_tensor(pointer, zero_k_packed_layout);
  ops::matmul_pack<gemm::AMX_BF16F32, gemm::Operand::B>(
      execution, zero_k_input, zero_k_output);
}

#elif VECOPS_TARGET_SHARD_INDEX == 8

#if defined(VECOPS_DEBUG)
TEST(MatmulPackDeathTest, RejectsInvalidOutputAndAliasing) {
  using Atom = gemm::AMX_BF16F32;
  auto input_layout = make_layout(make_shape(cint<3>, cint<5>));
  auto output_layout = ops::matmul_packed_layout<
      Atom, gemm::Operand::A>(input_layout);
  PackedStorage<bfloat16_t> storage(numel(output_layout) + 1);
  auto input = make_tensor(storage.data, input_layout);

  auto wrong_layout = make_layout(make_shape(cint<1>));
  auto wrong_output = make_tensor(storage.data, wrong_layout);
  EXPECT_DEATH(
      (ops::make_matmul_pack<Atom, gemm::Operand::A>(
          input, wrong_output)),
      "packed output layout");

  auto misaligned_output = make_tensor(storage.data + 1, output_layout);
  EXPECT_DEATH(
      (ops::make_matmul_pack<Atom, gemm::Operand::A>(
          input, misaligned_output)),
      "64-byte aligned");

  auto aliased_output = make_tensor(storage.data, output_layout);
  EXPECT_DEATH(
      (ops::make_matmul_pack<Atom, gemm::Operand::A>(
          input, aliased_output)),
      "in-place");
}

TEST(MatmulPackDeathTest, RejectsInvalidCompensationOutput) {
  using Atom = gemm::AMX_I8I32<uint8_t, int8_t>;
  auto input_layout = make_layout(make_shape(cint<3>, cint<5>));
  auto output_layout = ops::matmul_packed_layout<
      Atom, gemm::Operand::B>(input_layout);
  std::vector<int8_t> input(15);
  PackedStorage<int8_t> storage(numel(output_layout));
  std::vector<int32_t> compensation(6);
  auto input_tensor = make_tensor(input.data(), input_layout);
  auto output_tensor = make_tensor(storage.data, output_layout);

  auto wrong_extent = make_tensor(
      compensation.data(), make_layout(make_shape(cint<2>)));
  EXPECT_DEATH(
      (ops::make_matmul_pack_b_compensated<Atom>(
          input_tensor, output_tensor, wrong_extent, 3)),
      "extent must equal N");

  auto strided = make_tensor(
      compensation.data(),
      make_layout(make_shape(cint<3>), make_strides(cint<2>)));
  EXPECT_DEATH(
      (ops::make_matmul_pack_b_compensated<Atom>(
          input_tensor, output_tensor, strided, 3)),
      "must be contiguous");

  auto aliased = make_tensor(
      reinterpret_cast<int32_t*>(storage.data),
      make_layout(make_shape(cint<3>)));
  EXPECT_DEATH(
      (ops::make_matmul_pack_b_compensated<Atom>(
          input_tensor, output_tensor, aliased, 3)),
      "must not alias");
}

TEST(MatmulPackDeathTest, RejectsNegativeExtents) {
  EXPECT_DEATH(
      ({
        auto input_layout = make_layout(make_shape(Any{-1}, Any{7}));
        (void)ops::matmul_packed_layout<
            gemm::AMX_BF16F32, gemm::Operand::A>(input_layout);
      }),
      "non-negative");

  EXPECT_DEATH(
      ({
        constexpr nint_t Limit = std::numeric_limits<nint_t>::max();
        auto input_layout = make_layout(make_shape(Any{Limit}, Any{Limit}));
        (void)ops::matmul_packed_layout<
            gemm::AMX_BF16F32, gemm::Operand::A>(input_layout);
      }),
      "overflows");
}
#endif

#endif // VECOPS_TARGET_SHARD_INDEX

} // namespace

#endif // VECOPS_TARGET_SHARD_ACTIVE

#else

#if defined(VECOPS_TARGET_SHARD_ACTIVE)

#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <vector>

#include "TestUtils.h"
#include "vecops/execution/details/arm/Resources.h"
#include "vecops/matmul/Atom.h"
#include "vecops/kernel/Workspace.h"
#include "vecops/ops/MatmulPack.h"

namespace {

static_assert(VECOPS_TARGET_SHARD_COUNT == 6);

using namespace vecops;
using namespace vecops::meta;
using namespace vecops::tensor;

#if VECOPS_TARGET_SHARD_INDEX == 0
TEST(MatmulPackTest, PackedLayoutPreservesStreamingMetadata) {
  using Atom = gemm::SME_F32F32;
  using Packing = gemm::packing_t<Atom, gemm::Operand::A>;
  const auto input = make_layout(make_shape(cint<33>, cint<70>));
  const auto packed = ops::matmul_packed_layout<
      Atom, gemm::Operand::A>(input);
  using Layout = decltype(packed);

#if defined(HAS_FIXED_STREAMING_SVE_BITS)
  constexpr nint_t Panel =
      2 * FIXED_STREAMING_SVE_BITS / 8 / sizeof(float32_t);
  static_assert(std::same_as<size_type_t<0, Layout>,
                             Const<ceil_div(33, Panel)>>);
  static_assert(std::same_as<size_type_t<2, Layout>, Const<Panel>>);
#else
  static_assert(std::same_as<size_type_t<0, Layout>, Any>);
  static_assert(std::same_as<size_type_t<2, Layout>,
                             Dynamic<8, 8, 128>>);
#endif
  static_assert(std::same_as<size_type_t<1, Layout>, Const<70>>);
  static_assert(std::same_as<size_type_t<3, Layout>, Const<1>>);
  EXPECT_EQ(packed.shape()[2],
            static_cast<nint_t>(Packing::panel()));
}
#endif

template <typename T>
struct PackedStorage {
  kernel::Workspace owner;
  kernel::WorkspaceView view;
  T* data;
  explicit PackedStorage(nint_t elements)
      : owner(elements * static_cast<nint_t>(sizeof(T)) + 64),
        view(owner.view()),
        data(static_cast<T*>(view.allocate(
            elements * static_cast<nint_t>(sizeof(T)), 64))) {}
};

template <typename Atom, gemm::Operand Side>
void check_direct_pack(nint_t spatial, nint_t k) {
  using Packing = gemm::packing_t<Atom, Side>;
  using T = typename Packing::Element;
  std::vector<T> input(static_cast<std::size_t>(spatial * k));
  for (nint_t s = 0; s < spatial; ++s) {
    for (nint_t kk = 0; kk < k; ++kk) {
      input[static_cast<std::size_t>(s * k + kk)] =
          test_utils::get_test_value<T>(static_cast<int>(s * 97 + kk));
    }
  }
  auto input_layout = make_layout(make_shape(Any{spatial}, Any{k}));
  auto input_tensor = make_tensor(input.data(), input_layout);
  auto output_layout = ops::matmul_packed_layout<Atom, Side>(input_layout);
  PackedStorage<T> storage(numel(output_layout));
  auto output_tensor = make_tensor(storage.data, output_layout);
  auto operation = ops::make_matmul_pack<Atom, Side>(
      input_tensor, output_tensor);
  static_assert(!execution::details::has_resource_v<
      execution::details::arm::StreamingZA,
      typename decltype(operation)::ResourceRequirements>);
  ExecutionSession execution{};
  operation(execution);

  const nint_t panel = output_layout.shape()[2];
  const T* packed = storage.data;
  for (nint_t sp = 0; sp < ceil_div(spatial, panel); ++sp) {
    for (nint_t kg = 0; kg < ceil_div(k, Packing::KPack); ++kg) {
      for (nint_t lane = 0; lane < panel; ++lane) {
        for (nint_t ki = 0; ki < Packing::KPack; ++ki) {
          const nint_t s = sp * panel + lane;
          const nint_t kk = kg * Packing::KPack + ki;
          const T expected = (s < spatial && kk < k)
              ? input[static_cast<std::size_t>(s * k + kk)]
              : T{};
          EXPECT_TRUE(test_utils::values_equal(expected, *packed++))
              << "s=" << s << " k=" << kk;
        }
      }
    }
  }
  EXPECT_EQ(packed, storage.data + numel(output_layout));
}

template <gemm::Operand Side>
void check_vector_fallback() {
  using Atom = gemm::SME_BF16F32;
  using T = vecops::bfloat16_t;
  constexpr nint_t Spatial = 7;
  constexpr nint_t K = 11;
  constexpr nint_t StrideK = 2;
  constexpr nint_t StrideSpatial = K * StrideK + 5;
  std::vector<float32_t> input(
      static_cast<std::size_t>(Spatial * StrideSpatial), -999.0f);
  for (nint_t s = 0; s < Spatial; ++s) {
    for (nint_t k = 0; k < K; ++k) {
      input[static_cast<std::size_t>(s * StrideSpatial + k * StrideK)] =
          static_cast<float32_t>(s * 19 + k) + 0.5f;
    }
  }
  auto input_layout = make_layout(
      make_shape(cint<Spatial>, cint<K>),
      make_strides(cint<StrideSpatial>, cint<StrideK>));
  auto input_tensor = make_tensor(input.data(), input_layout);
  auto transform = make_elementwise_vec_transform<float32_t, float32_t>(
      []<vec::VectorTag Tag>(Tag tag, vec::Vec<Tag> value)
          VECOPS_KERNEL_LAMBDA {
        return vec::add(tag, value, vec::fill(tag, 1.0f));
      });
  auto input_spec = tensor::input<T>(input_tensor, transform);
  auto output_layout = ops::matmul_packed_layout<Atom, Side>(input_layout);
  PackedStorage<T> storage(numel(output_layout));
  auto output_tensor = make_tensor(storage.data, output_layout);
  auto operation = ops::make_matmul_pack<Atom, Side>(
      input_spec, output_tensor);
  static_assert(std::same_as<
      typename decltype(operation)::ResourceRequirements,
      typename execution::details::current_backend_t::DefaultRequirements>);
  ExecutionSession execution{};
  operation(execution);

  using Packing = gemm::packing_t<Atom, Side>;
  const nint_t panel = output_layout.shape()[2];
  const T* packed = storage.data;
  for (nint_t sp = 0; sp < ceil_div(Spatial, panel); ++sp) {
    for (nint_t kg = 0; kg < ceil_div(K, Packing::KPack); ++kg) {
      for (nint_t lane = 0; lane < panel; ++lane) {
        for (nint_t ki = 0; ki < Packing::KPack; ++ki) {
          const nint_t s = sp * panel + lane;
          const nint_t k = kg * Packing::KPack + ki;
          T expected{};
          if (s < Spatial && k < K) {
            expected = static_cast<T>(input[static_cast<std::size_t>(
                s * StrideSpatial + k * StrideK)] + 1.0f);
          }
          EXPECT_TRUE(test_utils::values_equal(expected, *packed++));
        }
      }
    }
  }
  EXPECT_EQ(packed, storage.data + numel(output_layout));
}

template <gemm::Operand Side, bool Transform>
void check_unit_stride_optimized_input() {
  using Atom = gemm::SME_F32F32;
  using T = vecops::float32_t;
  using Memory = std::conditional_t<Transform, T, vecops::float16_t>;
  constexpr nint_t Spatial = 35;
  constexpr nint_t K = 67;
  std::vector<Memory> input(
      static_cast<std::size_t>(Spatial * K));
  for (nint_t s = 0; s < Spatial; ++s) {
    for (nint_t k = 0; k < K; ++k) {
      const T value = static_cast<T>(s * 19 + k) + 0.5f;
      input[static_cast<std::size_t>(s * K + k)] =
          static_cast<Memory>(value);
    }
  }
  auto input_layout = make_layout(
      make_shape(cint<Spatial>, cint<K>));
  auto input_tensor = make_tensor(input.data(), input_layout);
  const auto input_operand = [&] {
    if constexpr (Transform) {
      auto transform = make_elementwise_vec_transform<T, T>(
          []<vec::VectorTag Tag>(Tag tag, vec::Vec<Tag> value)
              VECOPS_KERNEL_LAMBDA {
            return vec::add(tag, value, vec::fill(tag, T{1.0f}));
          });
      return tensor::input<T>(input_tensor, transform);
    } else {
      return tensor::input<T>(input_tensor);
    }
  }();
  auto output_layout = ops::matmul_packed_layout<Atom, Side>(input_layout);
  PackedStorage<T> storage(numel(output_layout));
  auto operation = ops::make_matmul_pack<Atom, Side>(
      input_operand, make_tensor(storage.data, output_layout));
  using Expected = std::conditional_t<
      Transform,
      kernel::matmul_pack_implementation::SMEStagedTransform,
      kernel::matmul_pack_implementation::SMEStagedFP16ToFP32>;
  static_assert(std::same_as<
      typename decltype(operation)::Implementation, Expected>);
  static_assert(std::same_as<
      typename decltype(operation)::ResourceRequirements,
      typename execution::details::current_backend_t::DefaultRequirements>);
  ExecutionSession execution{};
  operation(execution);

  using Packing = gemm::packing_t<Atom, Side>;
  const nint_t panel = output_layout.shape()[2];
  const T* packed = storage.data;
  for (nint_t sp = 0; sp < ceil_div(Spatial, panel); ++sp) {
    for (nint_t kg = 0; kg < ceil_div(K, Packing::KPack); ++kg) {
      for (nint_t lane = 0; lane < panel; ++lane) {
        for (nint_t ki = 0; ki < Packing::KPack; ++ki) {
          const nint_t s = sp * panel + lane;
          const nint_t k = kg * Packing::KPack + ki;
          T expected{};
          if (s < Spatial && k < K) {
            expected = static_cast<T>(
                input[static_cast<std::size_t>(s * K + k)]);
            if constexpr (Transform) expected += T{1.0f};
          }
          EXPECT_TRUE(test_utils::values_equal(expected, *packed++));
        }
      }
    }
  }
}

template <typename Atom, typename Memory, gemm::Operand Side>
void check_postprocess_conversion() {
  using T = typename gemm::packing_t<Atom, Side>::Element;
  constexpr nint_t Spatial = 35;
  constexpr nint_t K = 67;
  std::vector<Memory> input(static_cast<std::size_t>(Spatial * K));
  for (nint_t s = 0; s < Spatial; ++s) {
    for (nint_t k = 0; k < K; ++k) {
      input[static_cast<std::size_t>(s * K + k)] =
          static_cast<Memory>(
              static_cast<float32_t>((s * 19 + k) % 31) - 15.5f);
    }
  }
  auto input_layout = make_layout(make_shape(cint<Spatial>, cint<K>));
  auto output_layout = ops::matmul_packed_layout<Atom, Side>(input_layout);
  PackedStorage<T> storage(numel(output_layout));
  auto operation = ops::make_matmul_pack<Atom, Side>(
      tensor::input<T>(make_tensor(input.data(), input_layout)),
      make_tensor(storage.data, output_layout));
  static_assert(std::same_as<
      typename decltype(operation)::Implementation,
      kernel::matmul_pack_implementation::SMEPostprocess>);
  static_assert(!execution::details::has_resource_v<
      execution::details::arm::StreamingZA,
      typename decltype(operation)::ResourceRequirements>);
  ExecutionSession execution{};
  operation(execution);

  using Packing = gemm::packing_t<Atom, Side>;
  const nint_t panel = output_layout.shape()[2];
  const T* packed = storage.data;
  for (nint_t sp = 0; sp < ceil_div(Spatial, panel); ++sp) {
    for (nint_t kg = 0; kg < ceil_div(K, Packing::KPack); ++kg) {
      for (nint_t lane = 0; lane < panel; ++lane) {
        for (nint_t ki = 0; ki < Packing::KPack; ++ki) {
          const nint_t s = sp * panel + lane;
          const nint_t k = kg * Packing::KPack + ki;
          const T expected = s < Spatial && k < K
              ? static_cast<T>(input[static_cast<std::size_t>(s * K + k)])
              : T{};
          EXPECT_TRUE(test_utils::values_equal(expected, *packed++));
        }
      }
    }
  }
}

template <gemm::Operand Side>
void check_fp32_to_int8_postprocess() {
  using Atom = gemm::SME_I8I32<int8_t, uint8_t>;
  using T = typename gemm::packing_t<Atom, Side>::Element;
  constexpr nint_t Spatial = 35;
  constexpr nint_t K = 67;
  std::vector<float32_t> input(static_cast<std::size_t>(Spatial * K));
  for (nint_t i = 0; i < Spatial * K; ++i) {
    const int low = std::is_unsigned_v<T> ? 0 : -3;
    const int q = low + static_cast<int>((i * 5 + 3) % 7);
    input[static_cast<std::size_t>(i)] = static_cast<float32_t>(q) * 0.25f;
  }
  auto input_layout = make_layout(make_shape(cint<Spatial>, cint<K>));
  auto input_tensor = make_tensor(input.data(), input_layout);
  auto quantize = make_elementwise_vec_transform<float32_t, float32_t>(
      [](auto tag, auto value) VECOPS_KERNEL_LAMBDA {
        return vec::mul(tag, value, vec::fill(tag, 4.0f));
      });
  auto output_layout = ops::matmul_packed_layout<Atom, Side>(input_layout);
  PackedStorage<T> storage(numel(output_layout));
  auto operation = ops::make_matmul_pack<Atom, Side>(
      tensor::input<T>(input_tensor, quantize),
      make_tensor(storage.data, output_layout));
  static_assert(std::same_as<
      typename decltype(operation)::Implementation,
      kernel::matmul_pack_implementation::SMEPostprocess>);
  ExecutionSession execution{};
  operation(execution);

  using Packing = gemm::packing_t<Atom, Side>;
  const nint_t panel = output_layout.shape()[2];
  const T* packed = storage.data;
  for (nint_t sp = 0; sp < ceil_div(Spatial, panel); ++sp) {
    for (nint_t kg = 0; kg < ceil_div(K, Packing::KPack); ++kg) {
      for (nint_t lane = 0; lane < panel; ++lane) {
        for (nint_t ki = 0; ki < Packing::KPack; ++ki) {
          const nint_t s = sp * panel + lane;
          const nint_t k = kg * Packing::KPack + ki;
          const T expected = s < Spatial && k < K
              ? static_cast<T>(
                    input[static_cast<std::size_t>(s * K + k)] * 4.0f)
              : T{};
          EXPECT_EQ(expected, *packed++);
        }
      }
    }
  }
}

template <gemm::Operand Side>
void check_bf16_staged_transform() {
  using Atom = gemm::SME_BF16F32;
  using T = vecops::bfloat16_t;
  constexpr nint_t Spatial = 35;
  constexpr nint_t K = 67;
  std::vector<T> input(static_cast<std::size_t>(Spatial * K));
  for (nint_t s = 0; s < Spatial; ++s) {
    for (nint_t k = 0; k < K; ++k) {
      input[static_cast<std::size_t>(s * K + k)] =
          static_cast<T>(static_cast<float32_t>(s * 19 + k) + 0.5f);
    }
  }
  auto input_layout = make_layout(make_shape(cint<Spatial>, cint<K>));
  auto transform = make_elementwise_vec_transform<T, T>(
      []<vec::VectorTag Tag>(Tag tag, vec::Vec<Tag> value)
          VECOPS_KERNEL_LAMBDA {
        return vec::add(tag, value, vec::fill(tag, T{1.0f}));
      });
  auto output_layout = ops::matmul_packed_layout<Atom, Side>(input_layout);
  PackedStorage<T> storage(numel(output_layout));
  auto operation = ops::make_matmul_pack<Atom, Side>(
      tensor::input<T>(make_tensor(input.data(), input_layout), transform),
      make_tensor(storage.data, output_layout));
  static_assert(std::same_as<
      typename decltype(operation)::Implementation,
      kernel::matmul_pack_implementation::SMEStagedTransform>);
  ExecutionSession execution{};
  operation(execution);

  using Packing = gemm::packing_t<Atom, Side>;
  const nint_t panel = output_layout.shape()[2];
  const T* packed = storage.data;
  for (nint_t sp = 0; sp < ceil_div(Spatial, panel); ++sp) {
    for (nint_t kg = 0; kg < ceil_div(K, Packing::KPack); ++kg) {
      for (nint_t lane = 0; lane < panel; ++lane) {
        for (nint_t ki = 0; ki < Packing::KPack; ++ki) {
          const nint_t s = sp * panel + lane;
          const nint_t k = kg * Packing::KPack + ki;
          T expected{};
          if (s < Spatial && k < K) {
            expected = static_cast<T>(
                input[static_cast<std::size_t>(s * K + k)] + T{1.0f});
          }
          EXPECT_TRUE(test_utils::values_equal(expected, *packed++));
        }
      }
    }
  }
}

#if defined(HAS_SME_F64F64)
template <gemm::Operand Side, nint_t Spatial, nint_t K,
          bool ConstShape, nint_t InnerStride = 1,
          nint_t RowPadding = 0>
void check_fp32_to_f64_pack() {
  using Atom = gemm::SME_F64F64;
  using T = float64_t;
  constexpr nint_t RowStride = K * InnerStride + RowPadding;
  constexpr nint_t InputElements =
      (Spatial - 1) * RowStride + (K - 1) * InnerStride + 1;
  std::vector<float32_t> input(
      static_cast<std::size_t>(InputElements), -999.0f);
  for (nint_t s = 0; s < Spatial; ++s) {
    for (nint_t k = 0; k < K; ++k) {
      const int centered =
          static_cast<int>(((s * K + k) * 7 + 3) % 17) - 8;
      input[static_cast<std::size_t>(
          s * RowStride + k * InnerStride)] =
          static_cast<float32_t>(centered) * 0.125f;
    }
  }
  const auto input_layout = [&] {
    if constexpr (ConstShape) {
      return make_layout(
          make_shape(cint<Spatial>, cint<K>),
          make_strides(cint<RowStride>, cint<InnerStride>));
    } else {
      return make_layout(
          make_shape(Any{Spatial}, Any{K}),
          make_strides(Any{RowStride}, cint<InnerStride>));
    }
  }();
  auto output_layout = ops::matmul_packed_layout<Atom, Side>(input_layout);
  PackedStorage<T> storage(numel(output_layout));
  auto operation = ops::make_matmul_pack<Atom, Side>(
      tensor::input<T>(make_tensor(input.data(), input_layout)),
      make_tensor(storage.data, output_layout));
  using Expected = std::conditional_t<
      InnerStride == 1,
      kernel::matmul_pack_implementation::SMEFP32ToFP64,
      kernel::matmul_pack_implementation::Vector>;
  static_assert(std::same_as<
      typename decltype(operation)::Implementation, Expected>);
  ExecutionSession execution{};
  operation(execution);

  using Packing = gemm::packing_t<Atom, Side>;
  const nint_t panel = output_layout.shape()[2];
  const T* packed = storage.data;
  for (nint_t sp = 0; sp < ceil_div(Spatial, panel); ++sp) {
    for (nint_t kg = 0; kg < ceil_div(K, Packing::KPack); ++kg) {
      for (nint_t lane = 0; lane < panel; ++lane) {
        const nint_t s = sp * panel + lane;
        const nint_t k = kg * Packing::KPack;
        const T expected = s < Spatial && k < K
            ? static_cast<T>(input[static_cast<std::size_t>(
                  s * RowStride + k * InnerStride)])
            : T{};
        EXPECT_TRUE(test_utils::values_equal(expected, *packed++));
      }
    }
  }
  EXPECT_EQ(packed, storage.data + numel(output_layout));
}

template <gemm::Operand Side, bool ConstShape>
void check_fp32_to_f64_pack_boundaries() {
  check_fp32_to_f64_pack<Side, 32, 16, ConstShape>();
  check_fp32_to_f64_pack<Side, 35, 16, ConstShape>();
  check_fp32_to_f64_pack<Side, 32, 17, ConstShape>();
  check_fp32_to_f64_pack<Side, 35, 17, ConstShape>();
}
#endif

template <typename Atom, gemm::Operand Side>
void check_pipelined_direct_pack() {
  using Packing = gemm::packing_t<Atom, Side>;
  using T = typename Packing::Element;
  const nint_t panel = static_cast<nint_t>(Packing::panel());
  const nint_t word_chunk = panel * nint_t{sizeof(uint32_t)} /
      (2 * nint_t{sizeof(T)});
  check_direct_pack<Atom, Side>(
      2 * panel + 3, 5 * word_chunk + Packing::KPack - 1);
}

#if VECOPS_TARGET_SHARD_INDEX == 1
TEST(MatmulPackTest, DirectAAndBAllElementWidths) {
  check_direct_pack<gemm::SME_F32F32, gemm::Operand::A>(33, 70);
  check_direct_pack<gemm::SME_F32F32, gemm::Operand::B>(35, 9);
  check_direct_pack<gemm::SME_BF16F32, gemm::Operand::A>(31, 67);
  check_direct_pack<gemm::SME_BF16F32, gemm::Operand::B>(33, 69);
  check_direct_pack<gemm::SME_F16F32, gemm::Operand::A>(27, 63);
  check_direct_pack<gemm::SME_F16F32, gemm::Operand::B>(29, 65);
  check_direct_pack<
      gemm::SME_I8I32<int8_t, int8_t>, gemm::Operand::A>(35, 67);
  check_direct_pack<
      gemm::SME_I8I32<int8_t, int8_t>, gemm::Operand::B>(37, 69);
  check_direct_pack<
      gemm::SME_I8I32<int8_t, uint8_t>, gemm::Operand::A>(39, 71);
  check_direct_pack<
      gemm::SME_I8I32<int8_t, uint8_t>, gemm::Operand::B>(41, 73);
  check_direct_pack<
      gemm::SME_I8I32<uint8_t, int8_t>, gemm::Operand::A>(43, 75);
  check_direct_pack<
      gemm::SME_I8I32<uint8_t, int8_t>, gemm::Operand::B>(45, 77);
  check_direct_pack<
      gemm::SME_I8I32<uint8_t, uint8_t>, gemm::Operand::A>(47, 79);
  check_direct_pack<
      gemm::SME_I8I32<uint8_t, uint8_t>, gemm::Operand::B>(49, 81);
#if defined(HAS_SME_F64F64)
  check_direct_pack<gemm::SME_F64F64, gemm::Operand::A>(19, 17);
  check_direct_pack<gemm::SME_F64F64, gemm::Operand::B>(21, 9);
#endif
}

#elif VECOPS_TARGET_SHARD_INDEX == 2
TEST(MatmulPackTest, FullPanelsPipelineAcrossKChunks) {
  check_pipelined_direct_pack<gemm::SME_F32F32, gemm::Operand::A>();
  check_pipelined_direct_pack<gemm::SME_F32F32, gemm::Operand::B>();
  check_pipelined_direct_pack<gemm::SME_BF16F32, gemm::Operand::A>();
  check_pipelined_direct_pack<gemm::SME_BF16F32, gemm::Operand::B>();
  check_pipelined_direct_pack<gemm::SME_F16F32, gemm::Operand::A>();
  check_pipelined_direct_pack<gemm::SME_F16F32, gemm::Operand::B>();
  check_pipelined_direct_pack<
      gemm::SME_I8I32<int8_t, int8_t>, gemm::Operand::A>();
  check_pipelined_direct_pack<
      gemm::SME_I8I32<int8_t, int8_t>, gemm::Operand::B>();
  check_pipelined_direct_pack<
      gemm::SME_I8I32<int8_t, uint8_t>, gemm::Operand::A>();
  check_pipelined_direct_pack<
      gemm::SME_I8I32<int8_t, uint8_t>, gemm::Operand::B>();
  check_pipelined_direct_pack<
      gemm::SME_I8I32<uint8_t, int8_t>, gemm::Operand::A>();
  check_pipelined_direct_pack<
      gemm::SME_I8I32<uint8_t, int8_t>, gemm::Operand::B>();
  check_pipelined_direct_pack<
      gemm::SME_I8I32<uint8_t, uint8_t>, gemm::Operand::A>();
  check_pipelined_direct_pack<
      gemm::SME_I8I32<uint8_t, uint8_t>, gemm::Operand::B>();
}

#elif VECOPS_TARGET_SHARD_INDEX == 3
TEST(MatmulPackTest, StridedTransformUsesVectorFallback) {
  check_vector_fallback<gemm::Operand::A>();
  check_vector_fallback<gemm::Operand::B>();
}

#elif VECOPS_TARGET_SHARD_INDEX == 4
TEST(MatmulPackTest, UnitStrideConversionAndTransformUseOptimizedPaths) {
  check_unit_stride_optimized_input<gemm::Operand::A, false>();
  check_unit_stride_optimized_input<gemm::Operand::B, false>();
  check_unit_stride_optimized_input<gemm::Operand::A, true>();
  check_unit_stride_optimized_input<gemm::Operand::B, true>();
  check_postprocess_conversion<
      gemm::SME_BF16F32, float32_t, gemm::Operand::A>();
  check_postprocess_conversion<
      gemm::SME_BF16F32, float32_t, gemm::Operand::B>();
  check_postprocess_conversion<
      gemm::SME_F16F32, float32_t, gemm::Operand::A>();
  check_postprocess_conversion<
      gemm::SME_F16F32, float32_t, gemm::Operand::B>();
  check_postprocess_conversion<
      gemm::SME_F32F32, vecops::bfloat16_t, gemm::Operand::A>();
  check_postprocess_conversion<
      gemm::SME_F32F32, vecops::bfloat16_t, gemm::Operand::B>();
  check_fp32_to_int8_postprocess<gemm::Operand::A>();
  check_fp32_to_int8_postprocess<gemm::Operand::B>();
  check_bf16_staged_transform<gemm::Operand::A>();
  check_bf16_staged_transform<gemm::Operand::B>();
#if defined(HAS_SME_F64F64)
  check_fp32_to_f64_pack_boundaries<gemm::Operand::A, true>();
  check_fp32_to_f64_pack_boundaries<gemm::Operand::A, false>();
  check_fp32_to_f64_pack_boundaries<gemm::Operand::B, true>();
  check_fp32_to_f64_pack_boundaries<gemm::Operand::B, false>();
  check_fp32_to_f64_pack<gemm::Operand::A, 35, 17, false, 1, 3>();
  check_fp32_to_f64_pack<gemm::Operand::B, 35, 17, false, 1, 3>();
  check_fp32_to_f64_pack<gemm::Operand::A, 35, 17, true, 2, 3>();
  check_fp32_to_f64_pack<gemm::Operand::B, 35, 17, true, 2, 3>();
#endif
}

#elif VECOPS_TARGET_SHARD_INDEX == 5
TEST(MatmulPackTest, ZeroExtentDoesNotAccessStorage) {
  auto input_layout = make_layout(make_shape(cint<7>, cint<0>));
  auto output_layout = ops::matmul_packed_layout<
      gemm::SME_F32F32, gemm::Operand::B>(input_layout);
  EXPECT_EQ(numel(output_layout), 0);
  float32_t* pointer = nullptr;
  auto input = make_tensor(pointer, input_layout);
  auto output = make_tensor(pointer, output_layout);
  ExecutionSession execution{};
  ops::matmul_pack<gemm::SME_F32F32, gemm::Operand::B>(
      execution, input, output);
}

#endif // VECOPS_TARGET_SHARD_INDEX

} // namespace

#endif // VECOPS_TARGET_SHARD_ACTIVE

#endif
