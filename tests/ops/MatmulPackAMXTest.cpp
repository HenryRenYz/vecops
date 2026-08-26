#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <limits>
#include <type_traits>
#include <vector>

#include "TestUtils.h"
#include "vecops/gemm/Atoms.h"
#include "vecops/kernel/Workspace.h"
#include "vecops/ops/MatmulPack.h"

namespace {

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

template <gemm::Operand Side>
void check_strided_conversion_and_transform() {
  using Atom = gemm::AMX_BF16F32;
  using T = bfloat16_t;
  constexpr nint_t Spatial = 5;
  constexpr nint_t K = 9;
  constexpr nint_t StrideK = 2;
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

TEST(MatmulPackAMXTest, BF16AndF16AFullAndTails) {
  check_direct_pack<gemm::AMX_BF16F32, gemm::Operand::A>(16, 32, 0);
  check_direct_pack<gemm::AMX_BF16F32, gemm::Operand::A>(17, 33, 3);
  check_direct_pack<gemm::AMX_F16F32, gemm::Operand::A>(1, 1, 5);
}

TEST(MatmulPackAMXTest, BF16AndF16BFullAndTails) {
  check_direct_pack<gemm::AMX_BF16F32, gemm::Operand::B>(16, 32, 0);
  check_direct_pack<gemm::AMX_BF16F32, gemm::Operand::B>(19, 35, 4);
  check_direct_pack<gemm::AMX_F16F32, gemm::Operand::B>(1, 1, 2);
}

TEST(MatmulPackAMXTest, AllInt8SignednessCombinations) {
  check_direct_pack<gemm::AMX_I8I32<int8_t, int8_t>, gemm::Operand::A>(
      17, 65, 1);
  check_direct_pack<gemm::AMX_I8I32<int8_t, uint8_t>, gemm::Operand::B>(
      17, 65, 2);
  check_direct_pack<gemm::AMX_I8I32<uint8_t, int8_t>, gemm::Operand::A>(
      3, 7, 3);
  check_direct_pack<gemm::AMX_I8I32<uint8_t, uint8_t>, gemm::Operand::B>(
      5, 9, 4);
}

TEST(MatmulPackAMXTest, StridedConversionAndTransformStayVectorized) {
  check_strided_conversion_and_transform<gemm::Operand::A>();
  check_strided_conversion_and_transform<gemm::Operand::B>();
}

TEST(MatmulPackAMXTest, ZeroExtentDoesNotAccessStorage) {
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
}

#if defined(VECOPS_DEBUG)
TEST(MatmulPackAMXDeathTest, RejectsInvalidOutputAndAliasing) {
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

TEST(MatmulPackAMXDeathTest, RejectsNegativeExtents) {
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

} // namespace
