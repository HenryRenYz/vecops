#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <vector>

#include "TestUtils.h"
#include "vecops/execution/details/arm/Resources.h"
#include "vecops/gemm/Atoms.h"
#include "vecops/kernel/Workspace.h"
#include "vecops/ops/MatmulPack.h"

namespace {

using namespace vecops;
using namespace vecops::meta;
using namespace vecops::tensor;

TEST(MatmulPackSMETest, PackedLayoutPreservesStreamingMetadata) {
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

template <gemm::Operand Side>
void check_fp32_to_bf16_postprocess() {
  using Atom = gemm::SME_BF16F32;
  using T = vecops::bfloat16_t;
  constexpr nint_t Spatial = 35;
  constexpr nint_t K = 67;
  std::vector<float32_t> input(static_cast<std::size_t>(Spatial * K));
  for (nint_t s = 0; s < Spatial; ++s) {
    for (nint_t k = 0; k < K; ++k) {
      input[static_cast<std::size_t>(s * K + k)] =
          static_cast<float32_t>(s * 19 + k) + 0.5f;
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

TEST(MatmulPackSMETest, DirectAAndBAllElementWidths) {
  check_direct_pack<gemm::SME_F32F32, gemm::Operand::A>(33, 70);
  check_direct_pack<gemm::SME_F32F32, gemm::Operand::B>(35, 9);
  check_direct_pack<gemm::SME_BF16F32, gemm::Operand::A>(31, 67);
  check_direct_pack<gemm::SME_F16F32, gemm::Operand::B>(29, 65);
  check_direct_pack<gemm::SME_I8I32<int8_t, uint8_t>, gemm::Operand::B>(
      37, 69);
}

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

TEST(MatmulPackSMETest, FullPanelsPipelineAcrossKChunks) {
  check_pipelined_direct_pack<gemm::SME_F32F32, gemm::Operand::A>();
  check_pipelined_direct_pack<gemm::SME_F32F32, gemm::Operand::B>();
  check_pipelined_direct_pack<gemm::SME_BF16F32, gemm::Operand::A>();
  check_pipelined_direct_pack<gemm::SME_BF16F32, gemm::Operand::B>();
  check_pipelined_direct_pack<gemm::SME_F16F32, gemm::Operand::A>();
  check_pipelined_direct_pack<gemm::SME_F16F32, gemm::Operand::B>();
  using Int8Atom = gemm::SME_I8I32<int8_t, uint8_t>;
  check_pipelined_direct_pack<Int8Atom, gemm::Operand::A>();
  check_pipelined_direct_pack<Int8Atom, gemm::Operand::B>();
}

TEST(MatmulPackSMETest, StridedTransformUsesVectorFallback) {
  check_vector_fallback<gemm::Operand::A>();
  check_vector_fallback<gemm::Operand::B>();
}

TEST(MatmulPackSMETest, UnitStrideConversionAndTransformUseOptimizedPaths) {
  check_unit_stride_optimized_input<gemm::Operand::A, false>();
  check_unit_stride_optimized_input<gemm::Operand::B, false>();
  check_unit_stride_optimized_input<gemm::Operand::A, true>();
  check_unit_stride_optimized_input<gemm::Operand::B, true>();
  check_fp32_to_bf16_postprocess<gemm::Operand::A>();
  check_fp32_to_bf16_postprocess<gemm::Operand::B>();
  check_bf16_staged_transform<gemm::Operand::A>();
  check_bf16_staged_transform<gemm::Operand::B>();
}

TEST(MatmulPackSMETest, ZeroExtentDoesNotAccessStorage) {
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

} // namespace
