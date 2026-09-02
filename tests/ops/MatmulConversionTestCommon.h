//
// Copyright (c) vecops contributors.
//

#ifndef VECOPS_TESTS_OPS_MATMUL_CONVERSION_TEST_COMMON_H
#define VECOPS_TESTS_OPS_MATMUL_CONVERSION_TEST_COMMON_H

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <type_traits>
#include <vector>

#include "vecops/ops/Matmul.h"
#include "vecops/ops/MatmulPack.h"
#include "vecops/matmul/Quantization.h"

namespace vecops::test::matmul {

template <typename Config,
          meta::ValueInput M, meta::ValueInput N, meta::ValueInput K,
          tensor::InputOperand A, tensor::InputOperand B,
          tensor::OutputOperand C>
VECOPS_INLINE auto make_test_matmul_invocation(
    const Config& config, M&& m, N&& n, K&& k,
    A&& a, B&& b, C&& c) {
  using Atom = typename Config::Atom;
  auto c_output = tensor::as_output_spec<typename Atom::TAcc>(
      std::forward<C>(c));
  using Memory = typename decltype(c_output)::MemoryElement;
  auto c_input = tensor::input<typename Atom::TAcc>(
      c_output.tensor(),
      tensor::zeros_transform<typename Atom::TAcc, Memory>);
  return ::vecops::matmul::details::make_matmul_invocation(
      config, std::forward<M>(m), std::forward<N>(n),
      std::forward<K>(k), std::forward<A>(a), std::forward<B>(b),
      std::move(c_input), std::move(c_output));
}

template <typename Config,
          meta::ValueInput M, meta::ValueInput N, meta::ValueInput K,
          tensor::InputOperand A, tensor::InputOperand B,
          tensor::InputOperand CInput, tensor::OutputOperand COutput>
VECOPS_INLINE auto make_test_matmul_invocation(
    const Config& config, M&& m, N&& n, K&& k,
    A&& a, B&& b, CInput&& c_input, COutput&& c_output) {
  return ::vecops::matmul::details::make_matmul_invocation(
      config, std::forward<M>(m), std::forward<N>(n),
      std::forward<K>(k), std::forward<A>(a), std::forward<B>(b),
      std::forward<CInput>(c_input), std::forward<COutput>(c_output));
}

enum class ExtentMode {
  Dynamic,
  Const,
};

template <typename T>
T conversion_value(nint_t index, int seed) {
  if constexpr (std::is_integral_v<T>) {
    const int low = std::is_unsigned_v<T> ? 0 : -3;
    return static_cast<T>(low + static_cast<int>((index * 5 + seed) % 7));
  } else {
    const int centered = static_cast<int>((index * 7 + seed) % 17) - 8;
    return static_cast<T>(static_cast<float>(centered) / 16.0f);
  }
}

template <typename T>
bool conversion_values_equal(T expected, T actual) {
  if constexpr (std::is_integral_v<T>) {
    return expected == actual;
  } else {
    const double e = static_cast<double>(expected);
    const double a = static_cast<double>(actual);
    const double tolerance = sizeof(T) <= 2 ? 2.0e-2
                            : sizeof(T) == 4 ? 8.0e-4
                                             : 2.0e-10;
    return std::abs(a - e) <=
        tolerance * std::max({1.0, std::abs(a), std::abs(e)});
  }
}

template <typename Atom, typename MemoryA, typename MemoryB, typename MemoryC,
          typename MExtent, typename NExtent, typename KExtent>
void check_conversion(
    MExtent m_extent, NExtent n_extent, KExtent k_extent,
    bool expect_workspace = false) {
  using TA = typename Atom::TA;
  using TB = typename Atom::TB;
  using Acc = typename Atom::TAcc;
  const nint_t m = static_cast<nint_t>(m_extent);
  const nint_t n = static_cast<nint_t>(n_extent);
  const nint_t k = static_cast<nint_t>(k_extent);
  std::vector<MemoryA> a(static_cast<std::size_t>(m * k));
  std::vector<MemoryB> b(static_cast<std::size_t>(n * k));
  std::vector<MemoryC> c(static_cast<std::size_t>(m * n));
  for (nint_t i = 0; i < m * k; ++i) a[static_cast<std::size_t>(i)] =
      conversion_value<MemoryA>(i, 3);
  for (nint_t i = 0; i < n * k; ++i) b[static_cast<std::size_t>(i)] =
      conversion_value<MemoryB>(i, 11);

  auto at = tensor::make_tensor(
      a.data(), tensor::make_layout(
                    tensor::make_shape(m_extent, k_extent)));
  auto bt = tensor::make_tensor(
      b.data(), tensor::make_layout(
                    tensor::make_shape(n_extent, k_extent)));
  auto ct = tensor::make_tensor(
      c.data(), tensor::make_layout(
                    tensor::make_shape(m_extent, n_extent)));
  auto operation = make_test_matmul_invocation(ops::MatmulConfig<Atom>{},
      m_extent, n_extent, k_extent,
      tensor::input<TA>(at), tensor::input<TB>(bt),
      tensor::output<Acc>(ct));
  if (expect_workspace) {
    EXPECT_GT(
        operation.required_workspace(),
        kernel::matmul_implementation::scratch_bytes<
            typename decltype(operation)::Implementation>());
  }
  kernel::Workspace storage(operation.required_workspace());
  auto workspace = storage.view();
  operation(workspace);

  for (nint_t row = 0; row < m; ++row) {
    for (nint_t col = 0; col < n; ++col) {
      Acc expected{};
      for (nint_t kk = 0; kk < k; ++kk) {
        const TA av = static_cast<TA>(
            a[static_cast<std::size_t>(row * k + kk)]);
        const TB bv = static_cast<TB>(
            b[static_cast<std::size_t>(col * k + kk)]);
        expected += static_cast<Acc>(av) * static_cast<Acc>(bv);
      }
      const MemoryC converted = static_cast<MemoryC>(expected);
      EXPECT_TRUE(conversion_values_equal(
          converted, c[static_cast<std::size_t>(row * n + col)]))
          << "row=" << row << " col=" << col;
    }
  }
}

template <typename Atom, bool BroadcastB,
          typename BatchExtent, typename MExtent,
          typename NExtent, typename KExtent>
void check_batched_native(
    BatchExtent batch_extent, MExtent m_extent,
    NExtent n_extent, KExtent k_extent,
    bool expect_workspace = false) {
  using TA = typename Atom::TA;
  using TB = typename Atom::TB;
  using Acc = typename Atom::TAcc;
  const nint_t batch = static_cast<nint_t>(batch_extent);
  const nint_t m = static_cast<nint_t>(m_extent);
  const nint_t n = static_cast<nint_t>(n_extent);
  const nint_t k = static_cast<nint_t>(k_extent);
  const nint_t b_batches = BroadcastB ? 1 : batch;
  std::vector<TA> a(static_cast<std::size_t>(batch * m * k));
  std::vector<TB> b(static_cast<std::size_t>(b_batches * n * k));
  std::vector<Acc> c(static_cast<std::size_t>(batch * m * n));
  for (nint_t i = 0; i < batch * m * k; ++i)
    a[static_cast<std::size_t>(i)] = conversion_value<TA>(i, 3);
  for (nint_t i = 0; i < b_batches * n * k; ++i)
    b[static_cast<std::size_t>(i)] = conversion_value<TB>(i, 11);

  auto at = tensor::make_tensor(
      a.data(), tensor::make_layout(tensor::make_shape(
                    batch_extent, m_extent, k_extent)));
  auto bl = [&] {
    if constexpr (BroadcastB) {
      return tensor::make_layout(
          tensor::make_shape(
              batch_extent, n_extent, k_extent),
          tensor::make_strides(
              meta::cint<0>, k_extent, meta::cint<1>));
    } else {
      return tensor::make_layout(tensor::make_shape(
          batch_extent, n_extent, k_extent));
    }
  }();
  auto bt = tensor::make_tensor(b.data(), bl);
  auto ct = tensor::make_tensor(
      c.data(), tensor::make_layout(tensor::make_shape(
                    batch_extent, m_extent, n_extent)));
  auto operation = make_test_matmul_invocation(ops::MatmulConfig<Atom>{},
      m_extent, n_extent, k_extent, at, bt, ct);
  if (expect_workspace) {
    EXPECT_GT(operation.required_workspace(), 0);
  } else {
    EXPECT_EQ(operation.required_workspace(),
              kernel::matmul_implementation::scratch_bytes<
                  typename decltype(operation)::Implementation>());
  }
  kernel::Workspace storage(operation.required_workspace());
  auto workspace = storage.view();
  operation(workspace);

  for (nint_t bi = 0; bi < batch; ++bi) {
    const nint_t b_batch = BroadcastB ? 0 : bi;
    for (nint_t row = 0; row < m; ++row) {
      for (nint_t col = 0; col < n; ++col) {
        Acc expected{};
        for (nint_t kk = 0; kk < k; ++kk) {
          const TA av = a[static_cast<std::size_t>(
              (bi * m + row) * k + kk)];
          const TB bv = b[static_cast<std::size_t>(
              (b_batch * n + col) * k + kk)];
          expected += static_cast<Acc>(av) * static_cast<Acc>(bv);
        }
        const auto actual = c[static_cast<std::size_t>(
            (bi * m + row) * n + col)];
        EXPECT_TRUE(conversion_values_equal(expected, actual))
            << "batch=" << bi << " row=" << row << " col=" << col;
      }
    }
  }
}

template <typename Atom>
void check_batched_packed_b(
    nint_t batch, nint_t m, nint_t n, nint_t k) {
  using TA = typename Atom::TA;
  using TB = typename Atom::TB;
  using Acc = typename Atom::TAcc;
  std::vector<TA> a(static_cast<std::size_t>(batch * m * k));
  std::vector<TB> b(static_cast<std::size_t>(n * k));
  std::vector<Acc> c(static_cast<std::size_t>(batch * m * n));
  for (nint_t i = 0; i < batch * m * k; ++i)
    a[static_cast<std::size_t>(i)] = conversion_value<TA>(i, 3);
  for (nint_t i = 0; i < n * k; ++i)
    b[static_cast<std::size_t>(i)] = conversion_value<TB>(i, 11);

  auto at = tensor::make_tensor(
      a.data(), tensor::make_layout(tensor::make_shape(
                    meta::Any{batch}, meta::Any{m}, meta::Any{k})));
  auto bt = tensor::make_tensor(
      b.data(), tensor::make_layout(
                    tensor::make_shape(meta::Any{n}, meta::Any{k})));
  auto packed_layout = ::vecops::matmul::packed_layout<
      Atom, ::vecops::matmul::Operand::B>(bt.layout());
  const nint_t packed_bytes = tensor::numel(packed_layout) *
      static_cast<nint_t>(sizeof(TB));
  kernel::Workspace packed_storage(packed_bytes + 64);
  auto packed_workspace = packed_storage.view();
  auto* packed = static_cast<TB*>(
      packed_workspace.allocate(packed_bytes, 64));
  auto packed_tensor = tensor::make_tensor(packed, packed_layout);
  ExecutionSession pack_execution{};
  ::vecops::matmul::details::run_matmul_pack<Atom, ::vecops::matmul::Operand::B>(
      pack_execution, bt, packed_tensor);

  auto ct = tensor::make_tensor(
      c.data(), tensor::make_layout(tensor::make_shape(
                    meta::Any{batch}, meta::Any{m}, meta::Any{n})));
  auto operation = make_test_matmul_invocation(ops::MatmulConfig<Atom>{},
      m, n, k, at, packed_tensor, ct);
  kernel::Workspace storage(operation.required_workspace());
  auto workspace = storage.view();
  operation(workspace);

  for (nint_t bi = 0; bi < batch; ++bi) {
    for (nint_t row = 0; row < m; ++row) {
      for (nint_t col = 0; col < n; ++col) {
        Acc expected{};
        for (nint_t kk = 0; kk < k; ++kk) {
          expected += static_cast<Acc>(a[static_cast<std::size_t>(
                          (bi * m + row) * k + kk)]) *
              static_cast<Acc>(
                  b[static_cast<std::size_t>(col * k + kk)]);
        }
        const auto actual = c[static_cast<std::size_t>(
            (bi * m + row) * n + col)];
        EXPECT_TRUE(conversion_values_equal(expected, actual))
            << "batch=" << bi << " row=" << row << " col=" << col;
      }
    }
  }
}

template <typename Atom,
          typename BatchExtent, typename MExtent,
          typename NExtent, typename KExtent>
void check_batched_quantized_shared_b(
    BatchExtent batch_extent, MExtent m_extent,
    NExtent n_extent, KExtent k_extent) {
  using TA = typename Atom::TA;
  using TB = typename Atom::TB;
  using Acc = typename Atom::TAcc;
  static_assert(std::is_integral_v<TA> && std::is_integral_v<TB>);
  static_assert(std::same_as<Acc, int32_t>);
  const nint_t batch = static_cast<nint_t>(batch_extent);
  const nint_t m = static_cast<nint_t>(m_extent);
  const nint_t n = static_cast<nint_t>(n_extent);
  const nint_t k = static_cast<nint_t>(k_extent);
  std::vector<float32_t> a(static_cast<std::size_t>(batch * m * k));
  std::vector<float32_t> b(static_cast<std::size_t>(n * k));
  std::vector<float32_t> c(static_cast<std::size_t>(batch * m * n));
  for (nint_t i = 0; i < batch * m * k; ++i)
    a[static_cast<std::size_t>(i)] =
        static_cast<float32_t>(conversion_value<TA>(i, 3)) * 0.25f;
  for (nint_t i = 0; i < n * k; ++i)
    b[static_cast<std::size_t>(i)] =
        static_cast<float32_t>(conversion_value<TB>(i, 11)) * 0.25f;

  auto at = tensor::make_tensor(
      a.data(), tensor::make_layout(tensor::make_shape(
                    batch_extent, m_extent, k_extent)));
  auto bt = tensor::make_tensor(
      b.data(),
      tensor::make_layout(
          tensor::make_shape(
              batch_extent, n_extent, k_extent),
          tensor::make_strides(
              meta::cint<0>, k_extent, meta::cint<1>)));
  auto ct = tensor::make_tensor(
      c.data(), tensor::make_layout(tensor::make_shape(
                    batch_extent, m_extent, n_extent)));
  auto quantize = tensor::make_elementwise_vec_transform<
      float32_t, float32_t>(
      [](auto tag, auto value) VECOPS_KERNEL_LAMBDA {
        return vec::mul(tag, value, vec::fill(tag, 4.0f));
      });
  auto dequantize = tensor::make_elementwise_vec_transform<
      float32_t, float32_t>(
      [](auto tag, auto value) VECOPS_KERNEL_LAMBDA {
        return vec::mul(tag, value, vec::fill(tag, 0.125f));
      });
  auto operation = make_test_matmul_invocation(ops::MatmulConfig<Atom>{},
      m_extent, n_extent, k_extent,
      tensor::input<TA>(at, quantize),
      tensor::input<TB>(bt, quantize),
      tensor::output<Acc>(ct, dequantize));
  auto a_matrix_layout = tensor::make_layout(
      tensor::make_shape(m_extent, k_extent));
  auto b_matrix_layout = tensor::make_layout(
      tensor::make_shape(n_extent, k_extent));
  const auto packed_a_layout = ::vecops::matmul::packed_layout<
      Atom, ::vecops::matmul::Operand::A>(a_matrix_layout);
  const auto packed_b_layout = ::vecops::matmul::packed_layout<
      Atom, ::vecops::matmul::Operand::B>(b_matrix_layout);
  const nint_t expected_workspace =
      tensor::numel(packed_a_layout) * static_cast<nint_t>(sizeof(TA)) +
      tensor::numel(packed_b_layout) * static_cast<nint_t>(sizeof(TB)) + 126;
  EXPECT_EQ(operation.required_workspace(), expected_workspace);
  kernel::Workspace storage(operation.required_workspace());
  auto workspace = storage.view();
  operation(workspace);

  for (nint_t bi = 0; bi < batch; ++bi) {
    for (nint_t row = 0; row < m; ++row) {
      for (nint_t col = 0; col < n; ++col) {
        Acc expected{};
        for (nint_t kk = 0; kk < k; ++kk) {
          const TA av = static_cast<TA>(
              a[static_cast<std::size_t>((bi * m + row) * k + kk)] * 4);
          const TB bv = static_cast<TB>(
              b[static_cast<std::size_t>(col * k + kk)] * 4);
          expected += static_cast<Acc>(av) * static_cast<Acc>(bv);
        }
        const float32_t reference = static_cast<float32_t>(expected) * 0.125f;
        const auto actual = c[static_cast<std::size_t>(
            (bi * m + row) * n + col)];
        EXPECT_TRUE(conversion_values_equal(reference, actual))
            << "batch=" << bi << " row=" << row << " col=" << col;
      }
    }
  }
}

template <typename Atom, ::vecops::matmul::Operand Side>
void check_mixed_packing(nint_t m, nint_t n, nint_t k) {
  using TA = typename Atom::TA;
  using TB = typename Atom::TB;
  using Acc = typename Atom::TAcc;
  using Packed = typename ::vecops::matmul::packing_t<Atom, Side>::Element;
  std::vector<TA> a(static_cast<std::size_t>(m * k));
  std::vector<TB> b(static_cast<std::size_t>(n * k));
  std::vector<Acc> c(static_cast<std::size_t>(m * n));
  for (nint_t i = 0; i < m * k; ++i)
    a[static_cast<std::size_t>(i)] = conversion_value<TA>(i, 3);
  for (nint_t i = 0; i < n * k; ++i)
    b[static_cast<std::size_t>(i)] = conversion_value<TB>(i, 11);

  auto al = tensor::make_layout(
      tensor::make_shape(meta::Any{m}, meta::Any{k}));
  auto bl = tensor::make_layout(
      tensor::make_shape(meta::Any{n}, meta::Any{k}));
  auto at = tensor::make_tensor(a.data(), al);
  auto bt = tensor::make_tensor(b.data(), bl);
  const auto& source_layout = Side == ::vecops::matmul::Operand::A ? al : bl;
  auto packed_layout = ::vecops::matmul::packed_layout<Atom, Side>(source_layout);
  const nint_t packed_bytes =
      tensor::numel(packed_layout) * static_cast<nint_t>(sizeof(Packed));
  kernel::Workspace packed_storage(packed_bytes + 64);
  auto packed_workspace = packed_storage.view();
  auto* packed = static_cast<Packed*>(
      packed_workspace.allocate(packed_bytes, 64));
  auto packed_tensor = tensor::make_tensor(packed, packed_layout);
  ExecutionSession execution{};
  if constexpr (Side == ::vecops::matmul::Operand::A)
    ::vecops::matmul::details::run_matmul_pack<Atom, Side>(execution, at, packed_tensor);
  else
    ::vecops::matmul::details::run_matmul_pack<Atom, Side>(execution, bt, packed_tensor);

  auto ct = tensor::make_tensor(
      c.data(), tensor::make_layout(
                    tensor::make_shape(meta::Any{m}, meta::Any{n})));
  auto operation = [&] {
    if constexpr (Side == ::vecops::matmul::Operand::A)
      return make_test_matmul_invocation(ops::MatmulConfig<Atom>{}, m, n, k, packed_tensor, bt, ct);
    else
      return make_test_matmul_invocation(ops::MatmulConfig<Atom>{}, m, n, k, at, packed_tensor, ct);
  }();
  kernel::Workspace storage(operation.required_workspace());
  auto workspace = storage.view();
  operation(workspace);

  for (nint_t row = 0; row < m; ++row) {
    for (nint_t col = 0; col < n; ++col) {
      Acc expected{};
      for (nint_t kk = 0; kk < k; ++kk) {
        expected += static_cast<Acc>(
            a[static_cast<std::size_t>(row * k + kk)]) *
            static_cast<Acc>(
                b[static_cast<std::size_t>(col * k + kk)]);
      }
      const auto actual = c[static_cast<std::size_t>(row * n + col)];
      EXPECT_TRUE(conversion_values_equal(expected, actual))
          << "side=" << (Side == ::vecops::matmul::Operand::A ? "A" : "B")
          << " row=" << row << " col=" << col;
    }
  }
}

template <typename Atom>
void check_bias_relu(nint_t m, nint_t n, nint_t k) {
  using TA = typename Atom::TA;
  using TB = typename Atom::TB;
  using Acc = typename Atom::TAcc;
  static_assert(std::is_floating_point_v<Acc>);
  std::vector<TA> a(static_cast<std::size_t>(m * k));
  std::vector<TB> b(static_cast<std::size_t>(n * k));
  std::vector<Acc> bias(static_cast<std::size_t>(n));
  std::vector<Acc> c(static_cast<std::size_t>(m * n));
  for (nint_t i = 0; i < m * k; ++i)
    a[static_cast<std::size_t>(i)] = conversion_value<TA>(i, 3);
  for (nint_t i = 0; i < n * k; ++i)
    b[static_cast<std::size_t>(i)] = conversion_value<TB>(i, 11);
  for (nint_t i = 0; i < n; ++i)
    bias[static_cast<std::size_t>(i)] =
        static_cast<Acc>(static_cast<double>((i * 3) % 11 - 5) / 8.0);

  auto at = tensor::make_tensor(
      a.data(), tensor::make_layout(
                    tensor::make_shape(meta::Any{m}, meta::Any{k})));
  auto bt = tensor::make_tensor(
      b.data(), tensor::make_layout(
                    tensor::make_shape(meta::Any{n}, meta::Any{k})));
  auto bias_tensor = tensor::make_tensor(
      bias.data(),
      tensor::make_layout(
          tensor::make_shape(meta::Any{m}, meta::Any{n}),
          tensor::make_strides(meta::cint<0>, meta::cint<1>)));
  auto ct = tensor::make_tensor(
      c.data(), tensor::make_layout(
                    tensor::make_shape(meta::Any{m}, meta::Any{n})));
  auto relu = tensor::make_elementwise_vec_transform<Acc, Acc>(
      [](auto tag, auto value) VECOPS_KERNEL_LAMBDA {
        return vec::max(tag, value, vec::zeros(tag));
      });
  auto operation = make_test_matmul_invocation(ops::MatmulConfig<Atom>{},
      m, n, k, at, bt, tensor::input<Acc>(bias_tensor),
      tensor::output<Acc>(ct, relu));
  kernel::Workspace storage(operation.required_workspace());
  auto workspace = storage.view();
  operation(workspace);

  for (nint_t row = 0; row < m; ++row) {
    for (nint_t col = 0; col < n; ++col) {
      Acc expected = bias[static_cast<std::size_t>(col)];
      for (nint_t kk = 0; kk < k; ++kk) {
        expected += static_cast<Acc>(
            a[static_cast<std::size_t>(row * k + kk)]) *
            static_cast<Acc>(
                b[static_cast<std::size_t>(col * k + kk)]);
      }
      expected = std::max(expected, Acc{});
      EXPECT_TRUE(conversion_values_equal(
          expected, c[static_cast<std::size_t>(row * n + col)]))
          << "row=" << row << " col=" << col;
    }
  }
}

template <typename Atom, bool DirectB = false,
          typename MExtent, typename NExtent, typename KExtent>
void check_asymmetric_quantized(
    MExtent m_extent, NExtent n_extent, KExtent k_extent,
    bool expect_workspace = false) {
  using TA = typename Atom::TA;
  using TB = typename Atom::TB;
  using Acc = typename Atom::TAcc;
  static_assert(std::same_as<TA, uint8_t>);
  static_assert(std::same_as<TB, int8_t>);
  static_assert(std::same_as<Acc, int32_t>);
  constexpr Acc ZeroPointA = 3;
  const nint_t m = static_cast<nint_t>(m_extent);
  const nint_t n = static_cast<nint_t>(n_extent);
  const nint_t k = static_cast<nint_t>(k_extent);
  std::vector<float32_t> a(static_cast<std::size_t>(m * k));
  using MemoryB = std::conditional_t<DirectB, TB, float32_t>;
  std::vector<MemoryB> b(static_cast<std::size_t>(n * k));
  std::vector<Acc> correction(static_cast<std::size_t>(n));
  std::vector<float32_t> c(static_cast<std::size_t>(m * n));
  for (nint_t i = 0; i < m * k; ++i)
    a[static_cast<std::size_t>(i)] = conversion_value<float32_t>(i, 3);
  for (nint_t i = 0; i < n * k; ++i) {
    if constexpr (DirectB) {
      b[static_cast<std::size_t>(i)] = conversion_value<TB>(i, 11);
    } else {
      b[static_cast<std::size_t>(i)] = conversion_value<float32_t>(i, 11);
    }
  }
  for (nint_t col = 0; col < n; ++col) {
    Acc sum{};
    for (nint_t kk = 0; kk < k; ++kk) {
      if constexpr (DirectB) {
        sum += static_cast<Acc>(
            b[static_cast<std::size_t>(col * k + kk)]);
      } else {
        sum += static_cast<Acc>(
            b[static_cast<std::size_t>(col * k + kk)] * 4);
      }
    }
    correction[static_cast<std::size_t>(col)] = -ZeroPointA * sum;
  }

  auto at = tensor::make_tensor(
      a.data(), tensor::make_layout(
                    tensor::make_shape(m_extent, k_extent)));
  auto bt = tensor::make_tensor(
      b.data(), tensor::make_layout(
                    tensor::make_shape(n_extent, k_extent)));
  auto correction_tensor = tensor::make_tensor(
      correction.data(),
      tensor::make_layout(
          tensor::make_shape(m_extent, n_extent),
          tensor::make_strides(meta::cint<0>, meta::cint<1>)));
  auto ct = tensor::make_tensor(
      c.data(), tensor::make_layout(
                    tensor::make_shape(m_extent, n_extent)));
  auto quantize_a = tensor::make_elementwise_vec_transform<
      float32_t, float32_t>(
      [](auto tag, auto value) VECOPS_KERNEL_LAMBDA {
        return vec::add(
            tag, vec::mul(tag, value, vec::fill(tag, 4.0f)),
            vec::fill(tag, 3.0f));
      });
  auto quantize_b = tensor::make_elementwise_vec_transform<
      float32_t, float32_t>(
      [](auto tag, auto value) VECOPS_KERNEL_LAMBDA {
        return vec::mul(tag, value, vec::fill(tag, 4.0f));
      });
  auto dequantize = tensor::make_elementwise_vec_transform<
      float32_t, float32_t>(
      [](auto tag, auto value) VECOPS_KERNEL_LAMBDA {
        return vec::mul(tag, value, vec::fill(tag, 0.0625f));
      });
  const auto b_operand = [&] {
    if constexpr (DirectB) {
      return tensor::input<TB>(bt);
    } else {
      return tensor::input<TB>(bt, quantize_b);
    }
  }();
  auto operation = make_test_matmul_invocation(ops::MatmulConfig<Atom>{},
      m_extent, n_extent, k_extent,
      tensor::input<TA>(at, quantize_a),
      b_operand,
      tensor::input<Acc>(correction_tensor),
      tensor::output<Acc>(ct, dequantize));
  if (expect_workspace) {
    EXPECT_GT(
        operation.required_workspace(),
        kernel::matmul_implementation::scratch_bytes<
            typename decltype(operation)::Implementation>());
  }
  kernel::Workspace storage(operation.required_workspace());
  auto workspace = storage.view();
  operation(workspace);

  for (nint_t row = 0; row < m; ++row) {
    for (nint_t col = 0; col < n; ++col) {
      Acc expected = correction[static_cast<std::size_t>(col)];
      for (nint_t kk = 0; kk < k; ++kk) {
        const auto qa = static_cast<TA>(
            a[static_cast<std::size_t>(row * k + kk)] * 4 + ZeroPointA);
        const auto qb = [&] {
          if constexpr (DirectB) {
            return static_cast<TB>(
                b[static_cast<std::size_t>(col * k + kk)]);
          } else {
            return static_cast<TB>(
                b[static_cast<std::size_t>(col * k + kk)] * 4);
          }
        }();
        expected += static_cast<Acc>(qa) * static_cast<Acc>(qb);
      }
      const float32_t reference = static_cast<float32_t>(expected) * 0.0625f;
      EXPECT_TRUE(conversion_values_equal(
          reference, c[static_cast<std::size_t>(row * n + col)]))
          << "row=" << row << " col=" << col;
    }
  }
}

template <typename Atom,
          typename MExtent, typename NExtent, typename KExtent>
void check_runtime_and_per_column_scale(
    MExtent m_extent, NExtent n_extent, KExtent k_extent) {
  using TA = typename Atom::TA;
  using TB = typename Atom::TB;
  using Acc = typename Atom::TAcc;
  static_assert(std::is_floating_point_v<Acc>);
  const nint_t m = static_cast<nint_t>(m_extent);
  const nint_t n = static_cast<nint_t>(n_extent);
  const nint_t k = static_cast<nint_t>(k_extent);
  std::vector<TA> a(static_cast<std::size_t>(m * k));
  std::vector<TB> b(static_cast<std::size_t>(n * k));
  std::vector<Acc> dynamic_output(static_cast<std::size_t>(m * n));
  std::vector<Acc> column_output(static_cast<std::size_t>(m * n));
  std::vector<Acc> column_scales(static_cast<std::size_t>(n + 64));
  Acc dynamic_scale = static_cast<Acc>(0.75);
  for (nint_t i = 0; i < m * k; ++i)
    a[static_cast<std::size_t>(i)] = conversion_value<TA>(i, 3);
  for (nint_t i = 0; i < n * k; ++i)
    b[static_cast<std::size_t>(i)] = conversion_value<TB>(i, 11);
  for (nint_t col = 0; col < n + 64; ++col) {
    column_scales[static_cast<std::size_t>(col)] =
        static_cast<Acc>(0.5) + static_cast<Acc>(col % 7) *
            static_cast<Acc>(0.0625);
  }

  auto at = tensor::make_tensor(
      a.data(), tensor::make_layout(
                    tensor::make_shape(m_extent, k_extent)));
  auto bt = tensor::make_tensor(
      b.data(), tensor::make_layout(
                    tensor::make_shape(n_extent, k_extent)));
  auto dynamic_tensor = tensor::make_tensor(
      dynamic_output.data(), tensor::make_layout(
          tensor::make_shape(m_extent, n_extent)));
  auto column_tensor = tensor::make_tensor(
      column_output.data(), tensor::make_layout(
          tensor::make_shape(m_extent, n_extent)));
  auto dynamic_transform = tensor::make_elementwise_vec_transform<Acc, Acc>(
      [scale = &dynamic_scale](auto tag, auto value) VECOPS_KERNEL_LAMBDA {
        return vec::mul(tag, value, vec::fill(tag, *scale));
      });
  auto column_transform = tensor::make_lane_local_vec_transform<Acc, Acc>(
      [scales = column_scales.data()](
          auto tag, auto value,
          const auto& context) VECOPS_KERNEL_LAMBDA {
        const auto coordinate = context.lane_coord(0);
        const nint_t column = coordinate[coordinate.size() - 1];
        return vec::mul(tag, value, vec::load(tag, scales + column));
      });
  auto dynamic_operation = make_test_matmul_invocation(ops::MatmulConfig<Atom>{},
      m_extent, n_extent, k_extent, at, bt,
      tensor::output<Acc>(dynamic_tensor, dynamic_transform));
  auto column_operation = make_test_matmul_invocation(ops::MatmulConfig<Atom>{},
      m_extent, n_extent, k_extent, at, bt,
      tensor::output<Acc>(column_tensor, column_transform));
  kernel::Workspace dynamic_storage(dynamic_operation.required_workspace());
  auto dynamic_workspace = dynamic_storage.view();
  dynamic_operation(dynamic_workspace);
  kernel::Workspace column_storage(column_operation.required_workspace());
  auto column_workspace = column_storage.view();
  column_operation(column_workspace);

  for (nint_t row = 0; row < m; ++row) {
    for (nint_t col = 0; col < n; ++col) {
      Acc dot{};
      for (nint_t kk = 0; kk < k; ++kk) {
        dot += static_cast<Acc>(a[static_cast<std::size_t>(row * k + kk)]) *
            static_cast<Acc>(b[static_cast<std::size_t>(col * k + kk)]);
      }
      const auto index = static_cast<std::size_t>(row * n + col);
      EXPECT_TRUE(conversion_values_equal(
          dot * dynamic_scale, dynamic_output[index]))
          << "dynamic row=" << row << " col=" << col;
      EXPECT_TRUE(conversion_values_equal(
          dot * column_scales[static_cast<std::size_t>(col)],
          column_output[index]))
          << "per-column row=" << row << " col=" << col;
    }
  }
}

template <typename Atom>
void check_bias_clamp(nint_t m, nint_t n, nint_t k) {
  using TA = typename Atom::TA;
  using TB = typename Atom::TB;
  using Acc = typename Atom::TAcc;
  static_assert(std::is_floating_point_v<Acc>);
  std::vector<TA> a(static_cast<std::size_t>(m * k));
  std::vector<TB> b(static_cast<std::size_t>(n * k));
  std::vector<Acc> bias(static_cast<std::size_t>(n));
  std::vector<Acc> c(static_cast<std::size_t>(m * n));
  for (nint_t i = 0; i < m * k; ++i)
    a[static_cast<std::size_t>(i)] = conversion_value<TA>(i, 3);
  for (nint_t i = 0; i < n * k; ++i)
    b[static_cast<std::size_t>(i)] = conversion_value<TB>(i, 11);
  for (nint_t col = 0; col < n; ++col) {
    bias[static_cast<std::size_t>(col)] =
        static_cast<Acc>(static_cast<double>((col * 3) % 11 - 5) / 8.0);
  }
  auto at = tensor::make_tensor(
      a.data(), tensor::make_layout(
                    tensor::make_shape(meta::Any{m}, meta::Any{k})));
  auto bt = tensor::make_tensor(
      b.data(), tensor::make_layout(
                    tensor::make_shape(meta::Any{n}, meta::Any{k})));
  auto bias_tensor = tensor::make_tensor(
      bias.data(), tensor::make_layout(
                       tensor::make_shape(meta::Any{m}, meta::Any{n}),
                       tensor::make_strides(meta::cint<0>, meta::cint<1>)));
  auto ct = tensor::make_tensor(
      c.data(), tensor::make_layout(
                    tensor::make_shape(meta::Any{m}, meta::Any{n})));
  auto clamp = tensor::make_elementwise_vec_transform<Acc, Acc>(
      [](auto tag, auto value) VECOPS_KERNEL_LAMBDA {
        return vec::min(
            tag, vec::max(tag, value, vec::fill(tag, Acc{-0.25f})),
            vec::fill(tag, Acc{0.25f}));
      });
  auto operation = make_test_matmul_invocation(ops::MatmulConfig<Atom>{},
      m, n, k, at, bt, tensor::input<Acc>(bias_tensor),
      tensor::output<Acc>(ct, clamp));
  kernel::Workspace storage(operation.required_workspace());
  auto workspace = storage.view();
  operation(workspace);
  for (nint_t row = 0; row < m; ++row) {
    for (nint_t col = 0; col < n; ++col) {
      Acc expected = bias[static_cast<std::size_t>(col)];
      for (nint_t kk = 0; kk < k; ++kk) {
        expected +=
            static_cast<Acc>(a[static_cast<std::size_t>(row * k + kk)]) *
            static_cast<Acc>(b[static_cast<std::size_t>(col * k + kk)]);
      }
      expected = std::clamp(expected, Acc{-0.25f}, Acc{0.25f});
      EXPECT_TRUE(conversion_values_equal(
          expected, c[static_cast<std::size_t>(row * n + col)]))
          << "row=" << row << " col=" << col;
    }
  }
}

template <typename Atom,
          typename MExtent, typename NExtent, typename KExtent>
void check_dual_asymmetric_quantized(
    MExtent m_extent, NExtent n_extent, KExtent k_extent,
    bool expect_workspace = false) {
  using TA = typename Atom::TA;
  using TB = typename Atom::TB;
  using Acc = typename Atom::TAcc;
  static_assert(std::same_as<TA, uint8_t> && std::same_as<TB, uint8_t>);
  static_assert(std::same_as<Acc, int32_t>);
  constexpr Acc ZeroPointA = 3;
  constexpr Acc ZeroPointB = 5;
  const nint_t m = static_cast<nint_t>(m_extent);
  const nint_t n = static_cast<nint_t>(n_extent);
  const nint_t k = static_cast<nint_t>(k_extent);
  std::vector<float32_t> a(static_cast<std::size_t>(m * k));
  std::vector<float32_t> b(static_cast<std::size_t>(n * k));
  std::vector<Acc> correction(static_cast<std::size_t>(m * n));
  std::vector<float32_t> c(static_cast<std::size_t>(m * n));
  for (nint_t i = 0; i < m * k; ++i)
    a[static_cast<std::size_t>(i)] =
        static_cast<float32_t>((i * 5 + 3) % 7) * 0.25f;
  for (nint_t i = 0; i < n * k; ++i)
    b[static_cast<std::size_t>(i)] =
        static_cast<float32_t>((i * 3 + 1) % 7) * 0.25f;
  for (nint_t row = 0; row < m; ++row) {
    Acc sum_a{};
    for (nint_t kk = 0; kk < k; ++kk)
      sum_a += static_cast<Acc>(
          a[static_cast<std::size_t>(row * k + kk)] * 4 + ZeroPointA);
    for (nint_t col = 0; col < n; ++col) {
      Acc sum_b{};
      for (nint_t kk = 0; kk < k; ++kk)
        sum_b += static_cast<Acc>(
            b[static_cast<std::size_t>(col * k + kk)] * 4 + ZeroPointB);
      correction[static_cast<std::size_t>(row * n + col)] =
          -ZeroPointA * sum_b - ZeroPointB * sum_a +
          static_cast<Acc>(k) * ZeroPointA * ZeroPointB;
    }
  }
  auto at = tensor::make_tensor(
      a.data(), tensor::make_layout(
                    tensor::make_shape(m_extent, k_extent)));
  auto bt = tensor::make_tensor(
      b.data(), tensor::make_layout(
                    tensor::make_shape(n_extent, k_extent)));
  auto correction_tensor = tensor::make_tensor(
      correction.data(), tensor::make_layout(
          tensor::make_shape(m_extent, n_extent)));
  auto ct = tensor::make_tensor(
      c.data(), tensor::make_layout(
                    tensor::make_shape(m_extent, n_extent)));
  auto quantize_a = tensor::make_elementwise_vec_transform<
      float32_t, float32_t>(
      [](auto tag, auto value) VECOPS_KERNEL_LAMBDA {
        return vec::add(
            tag, vec::mul(tag, value, vec::fill(tag, 4.0f)),
            vec::fill(tag, 3.0f));
      });
  auto quantize_b = tensor::make_elementwise_vec_transform<
      float32_t, float32_t>(
      [](auto tag, auto value) VECOPS_KERNEL_LAMBDA {
        return vec::add(
            tag, vec::mul(tag, value, vec::fill(tag, 4.0f)),
            vec::fill(tag, 5.0f));
      });
  auto dequantize = tensor::make_elementwise_vec_transform<
      float32_t, float32_t>(
      [](auto tag, auto value) VECOPS_KERNEL_LAMBDA {
        return vec::mul(tag, value, vec::fill(tag, 0.0625f));
      });
  auto operation = make_test_matmul_invocation(ops::MatmulConfig<Atom>{},
      m_extent, n_extent, k_extent,
      tensor::input<TA>(at, quantize_a),
      tensor::input<TB>(bt, quantize_b),
      tensor::input<Acc>(correction_tensor),
      tensor::output<Acc>(ct, dequantize));
  if (expect_workspace) {
    EXPECT_GT(
        operation.required_workspace(),
        kernel::matmul_implementation::scratch_bytes<
            typename decltype(operation)::Implementation>());
  }
  kernel::Workspace storage(operation.required_workspace());
  auto workspace = storage.view();
  operation(workspace);
  for (nint_t row = 0; row < m; ++row) {
    for (nint_t col = 0; col < n; ++col) {
      Acc expected = correction[static_cast<std::size_t>(row * n + col)];
      for (nint_t kk = 0; kk < k; ++kk) {
        const auto qa = static_cast<TA>(
            a[static_cast<std::size_t>(row * k + kk)] * 4 + ZeroPointA);
        const auto qb = static_cast<TB>(
            b[static_cast<std::size_t>(col * k + kk)] * 4 + ZeroPointB);
        expected += static_cast<Acc>(qa) * static_cast<Acc>(qb);
      }
      const float32_t reference = static_cast<float32_t>(expected) * 0.0625f;
      EXPECT_TRUE(conversion_values_equal(
          reference, c[static_cast<std::size_t>(row * n + col)]))
          << "row=" << row << " col=" << col;
    }
  }
}

template <typename Atom, typename MemoryA, typename MemoryB,
          typename MemoryC>
void check_mixed_bias_relu(nint_t m, nint_t n, nint_t k) {
  using TA = typename Atom::TA;
  using TB = typename Atom::TB;
  using Acc = typename Atom::TAcc;
  static_assert(std::is_floating_point_v<Acc>);
  std::vector<MemoryA> a(static_cast<std::size_t>(m * k));
  std::vector<MemoryB> b(static_cast<std::size_t>(n * k));
  std::vector<float32_t> bias(static_cast<std::size_t>(n));
  std::vector<MemoryC> c(static_cast<std::size_t>(m * n));
  for (nint_t i = 0; i < m * k; ++i)
    a[static_cast<std::size_t>(i)] = conversion_value<MemoryA>(i, 3);
  for (nint_t i = 0; i < n * k; ++i)
    b[static_cast<std::size_t>(i)] = conversion_value<MemoryB>(i, 11);
  for (nint_t col = 0; col < n; ++col)
    bias[static_cast<std::size_t>(col)] =
        static_cast<float32_t>((col * 3) % 11 - 5) / 8.0f;

  auto al = tensor::make_layout(
      tensor::make_shape(meta::Any{m}, meta::Any{k}));
  auto bl = tensor::make_layout(
      tensor::make_shape(meta::Any{n}, meta::Any{k}));
  auto cl = tensor::make_layout(
      tensor::make_shape(meta::Any{m}, meta::Any{n}));
  auto bias_layout = tensor::make_layout(
      tensor::make_shape(meta::Any{m}, meta::Any{n}),
      tensor::make_strides(meta::cint<0>, meta::cint<1>));
  auto relu = tensor::make_elementwise_vec_transform<Acc, Acc>(
      [](auto tag, auto value) VECOPS_KERNEL_LAMBDA {
        return vec::max(tag, value, vec::zeros(tag));
      });
  auto operation = make_test_matmul_invocation(ops::MatmulConfig<Atom>{},
      m, n, k,
      tensor::input<TA>(tensor::make_tensor(a.data(), al)),
      tensor::input<TB>(tensor::make_tensor(b.data(), bl)),
      tensor::input<Acc>(tensor::make_tensor(bias.data(), bias_layout)),
      tensor::output<Acc>(tensor::make_tensor(c.data(), cl), relu));
  kernel::Workspace storage(operation.required_workspace());
  auto workspace = storage.view();
  operation(workspace);

  for (nint_t row = 0; row < m; ++row) {
    for (nint_t col = 0; col < n; ++col) {
      Acc expected = static_cast<Acc>(bias[static_cast<std::size_t>(col)]);
      for (nint_t kk = 0; kk < k; ++kk) {
        expected += static_cast<Acc>(static_cast<TA>(
            a[static_cast<std::size_t>(row * k + kk)])) *
            static_cast<Acc>(static_cast<TB>(
                b[static_cast<std::size_t>(col * k + kk)]));
      }
      const MemoryC converted = static_cast<MemoryC>(
          std::max(expected, Acc{}));
      EXPECT_TRUE(conversion_values_equal(
          converted, c[static_cast<std::size_t>(row * n + col)]))
          << "row=" << row << " col=" << col;
    }
  }
}

template <typename Atom>
void check_quantized_a_direct_b(nint_t m, nint_t n, nint_t k) {
  using TA = typename Atom::TA;
  using TB = typename Atom::TB;
  using Acc = typename Atom::TAcc;
  static_assert(std::is_integral_v<TA> && std::is_integral_v<TB>);
  static_assert(std::same_as<Acc, int32_t>);
  std::vector<float32_t> a(static_cast<std::size_t>(m * k));
  std::vector<TB> b(static_cast<std::size_t>(n * k));
  std::vector<float32_t> c(static_cast<std::size_t>(m * n));
  for (nint_t i = 0; i < m * k; ++i)
    a[static_cast<std::size_t>(i)] =
        static_cast<float32_t>(conversion_value<TA>(i, 3)) * 0.25f;
  for (nint_t i = 0; i < n * k; ++i)
    b[static_cast<std::size_t>(i)] = conversion_value<TB>(i, 11);
  auto al = tensor::make_layout(
      tensor::make_shape(meta::Any{m}, meta::Any{k}));
  auto bl = tensor::make_layout(
      tensor::make_shape(meta::Any{n}, meta::Any{k}));
  auto cl = tensor::make_layout(
      tensor::make_shape(meta::Any{m}, meta::Any{n}));
  auto quantize = tensor::make_elementwise_vec_transform<
      float32_t, float32_t>(
      [](auto tag, auto value) VECOPS_KERNEL_LAMBDA {
        return vec::mul(tag, value, vec::fill(tag, 4.0f));
      });
  auto dequantize = tensor::make_elementwise_vec_transform<
      float32_t, float32_t>(
      [](auto tag, auto value) VECOPS_KERNEL_LAMBDA {
        return vec::mul(tag, value, vec::fill(tag, 0.125f));
      });
  auto operation = make_test_matmul_invocation(ops::MatmulConfig<Atom>{},
      m, n, k,
      tensor::input<TA>(tensor::make_tensor(a.data(), al), quantize),
      tensor::input<TB>(tensor::make_tensor(b.data(), bl)),
      tensor::output<Acc>(tensor::make_tensor(c.data(), cl), dequantize));
  kernel::Workspace storage(operation.required_workspace());
  auto workspace = storage.view();
  operation(workspace);
  for (nint_t row = 0; row < m; ++row) {
    for (nint_t col = 0; col < n; ++col) {
      Acc expected{};
      for (nint_t kk = 0; kk < k; ++kk) {
        const TA av = static_cast<TA>(
            a[static_cast<std::size_t>(row * k + kk)] * 4.0f);
        expected += static_cast<Acc>(av) * static_cast<Acc>(
            b[static_cast<std::size_t>(col * k + kk)]);
      }
      EXPECT_TRUE(conversion_values_equal(
          static_cast<float32_t>(expected) * 0.125f,
          c[static_cast<std::size_t>(row * n + col)]))
          << "row=" << row << " col=" << col;
    }
  }
}

template <typename Atom>
void check_native_integer_accumulate(nint_t m, nint_t n, nint_t k) {
  using TA = typename Atom::TA;
  using TB = typename Atom::TB;
  using Acc = typename Atom::TAcc;
  static_assert(std::is_integral_v<TA> && std::is_integral_v<TB>);
  static_assert(std::same_as<Acc, int32_t>);
  std::vector<TA> a(static_cast<std::size_t>(m * k));
  std::vector<TB> b(static_cast<std::size_t>(n * k));
  std::vector<Acc> input_c(static_cast<std::size_t>(m * n));
  std::vector<Acc> output_c(static_cast<std::size_t>(m * n));
  for (nint_t i = 0; i < m * k; ++i)
    a[static_cast<std::size_t>(i)] = conversion_value<TA>(i, 3);
  for (nint_t i = 0; i < n * k; ++i)
    b[static_cast<std::size_t>(i)] = conversion_value<TB>(i, 11);
  for (nint_t i = 0; i < m * n; ++i)
    input_c[static_cast<std::size_t>(i)] = static_cast<Acc>(i % 7 - 3);
  auto al = tensor::make_layout(
      tensor::make_shape(meta::Any{m}, meta::Any{k}));
  auto bl = tensor::make_layout(
      tensor::make_shape(meta::Any{n}, meta::Any{k}));
  auto cl = tensor::make_layout(
      tensor::make_shape(meta::Any{m}, meta::Any{n}));
  auto operation = make_test_matmul_invocation(ops::MatmulConfig<Atom>{},
      m, n, k, tensor::make_tensor(a.data(), al),
      tensor::make_tensor(b.data(), bl),
      tensor::make_tensor(input_c.data(), cl),
      tensor::make_tensor(output_c.data(), cl));
  kernel::Workspace storage(operation.required_workspace());
  auto workspace = storage.view();
  operation(workspace);
  for (nint_t row = 0; row < m; ++row) {
    for (nint_t col = 0; col < n; ++col) {
      const auto index = static_cast<std::size_t>(row * n + col);
      Acc expected = input_c[index];
      for (nint_t kk = 0; kk < k; ++kk) {
        expected += static_cast<Acc>(
            a[static_cast<std::size_t>(row * k + kk)]) *
            static_cast<Acc>(
                b[static_cast<std::size_t>(col * k + kk)]);
      }
      EXPECT_EQ(expected, output_c[index])
          << "row=" << row << " col=" << col;
    }
  }
}

template <typename Atom>
void check_relu_and_sigmoid(nint_t m, nint_t n, nint_t k) {
  using TA = typename Atom::TA;
  using TB = typename Atom::TB;
  using Acc = typename Atom::TAcc;
  static_assert(std::is_floating_point_v<Acc>);
  std::vector<TA> a(static_cast<std::size_t>(m * k));
  std::vector<TB> b(static_cast<std::size_t>(n * k));
  std::vector<Acc> relu_output(static_cast<std::size_t>(m * n));
  std::vector<Acc> sigmoid_output(static_cast<std::size_t>(m * n));
  for (nint_t i = 0; i < m * k; ++i)
    a[static_cast<std::size_t>(i)] = conversion_value<TA>(i, 3);
  for (nint_t i = 0; i < n * k; ++i)
    b[static_cast<std::size_t>(i)] = conversion_value<TB>(i, 11);
  auto al = tensor::make_layout(
      tensor::make_shape(meta::Any{m}, meta::Any{k}));
  auto bl = tensor::make_layout(
      tensor::make_shape(meta::Any{n}, meta::Any{k}));
  auto cl = tensor::make_layout(
      tensor::make_shape(meta::Any{m}, meta::Any{n}));
  auto relu = tensor::make_elementwise_vec_transform<Acc, Acc>(
      [](auto tag, auto value) VECOPS_KERNEL_LAMBDA {
        return vec::max(tag, value, vec::zeros(tag));
      });
  auto sigmoid = tensor::make_elementwise_vec_transform<Acc, Acc>(
      [](auto tag, auto value) VECOPS_KERNEL_LAMBDA {
        const auto one = vec::fill(tag, Acc{1});
        return vec::div(
            tag, one,
            vec::add(tag, one, vec::exp(tag, vec::neg(tag, value))));
      });
  auto relu_operation = make_test_matmul_invocation(ops::MatmulConfig<Atom>{},
      m, n, k, tensor::make_tensor(a.data(), al),
      tensor::make_tensor(b.data(), bl),
      tensor::output<Acc>(
          tensor::make_tensor(relu_output.data(), cl), relu));
  auto sigmoid_operation = make_test_matmul_invocation(ops::MatmulConfig<Atom>{},
      m, n, k, tensor::make_tensor(a.data(), al),
      tensor::make_tensor(b.data(), bl),
      tensor::output<Acc>(
          tensor::make_tensor(sigmoid_output.data(), cl), sigmoid));
  kernel::Workspace relu_storage(relu_operation.required_workspace());
  auto relu_workspace = relu_storage.view();
  relu_operation(relu_workspace);
  kernel::Workspace sigmoid_storage(sigmoid_operation.required_workspace());
  auto sigmoid_workspace = sigmoid_storage.view();
  sigmoid_operation(sigmoid_workspace);
  for (nint_t row = 0; row < m; ++row) {
    for (nint_t col = 0; col < n; ++col) {
      Acc dot{};
      for (nint_t kk = 0; kk < k; ++kk) {
        dot += static_cast<Acc>(
            a[static_cast<std::size_t>(row * k + kk)]) *
            static_cast<Acc>(
                b[static_cast<std::size_t>(col * k + kk)]);
      }
      const auto index = static_cast<std::size_t>(row * n + col);
      EXPECT_TRUE(conversion_values_equal(
          std::max(dot, Acc{}), relu_output[index]));
      const auto expected_sigmoid = static_cast<Acc>(
          1.0 / (1.0 + std::exp(-static_cast<double>(dot))));
      EXPECT_TRUE(conversion_values_equal(
          expected_sigmoid, sigmoid_output[index]));
    }
  }
}

template <typename Atom>
void check_silu_and_bias_silu(nint_t m, nint_t n, nint_t k) {
  using TA = typename Atom::TA;
  using TB = typename Atom::TB;
  using Acc = typename Atom::TAcc;
  static_assert(std::is_floating_point_v<Acc>);
  std::vector<TA> a(static_cast<std::size_t>(m * k));
  std::vector<TB> b(static_cast<std::size_t>(n * k));
  std::vector<Acc> bias(static_cast<std::size_t>(n));
  std::vector<Acc> silu_output(static_cast<std::size_t>(m * n));
  std::vector<Acc> bias_silu_output(static_cast<std::size_t>(m * n));
  for (nint_t i = 0; i < m * k; ++i)
    a[static_cast<std::size_t>(i)] = conversion_value<TA>(i, 3);
  for (nint_t i = 0; i < n * k; ++i)
    b[static_cast<std::size_t>(i)] = conversion_value<TB>(i, 11);
  for (nint_t col = 0; col < n; ++col)
    bias[static_cast<std::size_t>(col)] =
        static_cast<Acc>(static_cast<double>(col % 7 - 3) / 8.0);
  auto al = tensor::make_layout(
      tensor::make_shape(meta::Any{m}, meta::Any{k}));
  auto bl = tensor::make_layout(
      tensor::make_shape(meta::Any{n}, meta::Any{k}));
  auto cl = tensor::make_layout(
      tensor::make_shape(meta::Any{m}, meta::Any{n}));
  auto bias_layout = tensor::make_layout(
      tensor::make_shape(meta::Any{m}, meta::Any{n}),
      tensor::make_strides(meta::cint<0>, meta::cint<1>));
  auto silu = tensor::make_elementwise_vec_transform<Acc, Acc>(
      [](auto tag, auto value) VECOPS_KERNEL_LAMBDA {
        const auto one = vec::fill(tag, Acc{1});
        const auto sigmoid = vec::div(
            tag, one,
            vec::add(tag, one, vec::exp(tag, vec::neg(tag, value))));
        return vec::mul(tag, value, sigmoid);
      });
  auto silu_operation = make_test_matmul_invocation(ops::MatmulConfig<Atom>{},
      m, n, k, tensor::make_tensor(a.data(), al),
      tensor::make_tensor(b.data(), bl),
      tensor::output<Acc>(
          tensor::make_tensor(silu_output.data(), cl), silu));
  auto bias_silu_operation = make_test_matmul_invocation(ops::MatmulConfig<Atom>{},
      m, n, k, tensor::make_tensor(a.data(), al),
      tensor::make_tensor(b.data(), bl),
      tensor::input<Acc>(tensor::make_tensor(bias.data(), bias_layout)),
      tensor::output<Acc>(
          tensor::make_tensor(bias_silu_output.data(), cl), silu));
  kernel::Workspace silu_storage(silu_operation.required_workspace());
  auto silu_workspace = silu_storage.view();
  silu_operation(silu_workspace);
  kernel::Workspace bias_storage(
      bias_silu_operation.required_workspace());
  auto bias_workspace = bias_storage.view();
  bias_silu_operation(bias_workspace);
  for (nint_t row = 0; row < m; ++row) {
    for (nint_t col = 0; col < n; ++col) {
      Acc dot{};
      for (nint_t kk = 0; kk < k; ++kk) {
        dot += static_cast<Acc>(
            a[static_cast<std::size_t>(row * k + kk)]) *
            static_cast<Acc>(
                b[static_cast<std::size_t>(col * k + kk)]);
      }
      const auto index = static_cast<std::size_t>(row * n + col);
      const auto silu_reference = [](Acc value) {
        const double x = static_cast<double>(value);
        return static_cast<Acc>(x / (1.0 + std::exp(-x)));
      };
      EXPECT_TRUE(conversion_values_equal(
          silu_reference(dot), silu_output[index]));
      EXPECT_TRUE(conversion_values_equal(
          silu_reference(dot + bias[static_cast<std::size_t>(col)]),
          bias_silu_output[index]));
    }
  }
}

template <typename Atom>
void check_integer_bias_per_column_quantization(
    nint_t m, nint_t n, nint_t k) {
  using TA = typename Atom::TA;
  using TB = typename Atom::TB;
  using Acc = typename Atom::TAcc;
  static_assert(std::is_integral_v<TA> && std::is_integral_v<TB>);
  static_assert(std::same_as<Acc, int32_t>);
  std::vector<float32_t> a(static_cast<std::size_t>(m * k));
  std::vector<TB> b(static_cast<std::size_t>(n * k));
  std::vector<Acc> bias(static_cast<std::size_t>(n));
  std::vector<float32_t> scales(static_cast<std::size_t>(n + 64));
  std::vector<float32_t> dequantized(static_cast<std::size_t>(m * n));
  std::vector<uint8_t> requantized(static_cast<std::size_t>(m * n));
  for (nint_t i = 0; i < m * k; ++i)
    a[static_cast<std::size_t>(i)] =
        static_cast<float32_t>(conversion_value<TA>(i, 3)) * 0.25f;
  for (nint_t i = 0; i < n * k; ++i)
    b[static_cast<std::size_t>(i)] = conversion_value<TB>(i, 11);
  for (nint_t col = 0; col < n; ++col)
    bias[static_cast<std::size_t>(col)] =
        static_cast<Acc>(col % 7 - 3);
  for (nint_t col = 0; col < n + 64; ++col)
    scales[static_cast<std::size_t>(col)] =
        0.015625f + static_cast<float32_t>(col % 7) * 0.00390625f;
  int32_t output_zero_point = 7;
  auto al = tensor::make_layout(
      tensor::make_shape(meta::Any{m}, meta::Any{k}));
  auto bl = tensor::make_layout(
      tensor::make_shape(meta::Any{n}, meta::Any{k}));
  auto cl = tensor::make_layout(
      tensor::make_shape(meta::Any{m}, meta::Any{n}));
  auto bias_layout = tensor::make_layout(
      tensor::make_shape(meta::Any{m}, meta::Any{n}),
      tensor::make_strides(meta::cint<0>, meta::cint<1>));
  auto quantize_a = tensor::make_elementwise_vec_transform<
      float32_t, float32_t>(
      [](auto tag, auto value) VECOPS_KERNEL_LAMBDA {
        return vec::mul(tag, value, vec::fill(tag, 4.0f));
      });
  auto dequantize = tensor::make_lane_local_vec_transform<
      float32_t, float32_t>(
      [column_scales = scales.data()](
          auto tag, auto value,
          const auto& context) VECOPS_KERNEL_LAMBDA {
        const auto coordinate = context.lane_coord(0);
        const nint_t column = coordinate[coordinate.size() - 1];
        return vec::mul(
            tag, value, vec::load(tag, column_scales + column));
      });
  auto requantize = tensor::make_lane_local_vec_transform<
      float32_t, float32_t>(
      [column_scales = scales.data(),
       zero_point = &output_zero_point](
          auto tag, auto value,
          const auto& context) VECOPS_KERNEL_LAMBDA {
        const auto coordinate = context.lane_coord(0);
        const nint_t column = coordinate[coordinate.size() - 1];
        return vec::add(
            tag,
            vec::mul(tag, value, vec::load(tag, column_scales + column)),
            vec::fill(tag, static_cast<float32_t>(*zero_point)));
      });
  const auto a_operand = tensor::input<TA>(
      tensor::make_tensor(a.data(), al), quantize_a);
  const auto b_operand = tensor::input<TB>(
      tensor::make_tensor(b.data(), bl));
  const auto bias_operand = tensor::input<Acc>(
      tensor::make_tensor(bias.data(), bias_layout));
  auto dequant_operation = make_test_matmul_invocation(ops::MatmulConfig<Atom>{},
      m, n, k, a_operand, b_operand, bias_operand,
      tensor::output<Acc>(
          tensor::make_tensor(dequantized.data(), cl), dequantize));
  auto requant_operation = make_test_matmul_invocation(ops::MatmulConfig<Atom>{},
      m, n, k, a_operand, b_operand, bias_operand,
      tensor::output<Acc>(
          tensor::make_tensor(requantized.data(), cl), requantize));
  kernel::Workspace dequant_storage(
      dequant_operation.required_workspace());
  auto dequant_workspace = dequant_storage.view();
  dequant_operation(dequant_workspace);
  kernel::Workspace requant_storage(
      requant_operation.required_workspace());
  auto requant_workspace = requant_storage.view();
  requant_operation(requant_workspace);
  for (nint_t row = 0; row < m; ++row) {
    for (nint_t col = 0; col < n; ++col) {
      Acc expected = bias[static_cast<std::size_t>(col)];
      for (nint_t kk = 0; kk < k; ++kk) {
        const TA qa = static_cast<TA>(
            a[static_cast<std::size_t>(row * k + kk)] * 4.0f);
        expected += static_cast<Acc>(qa) * static_cast<Acc>(
            b[static_cast<std::size_t>(col * k + kk)]);
      }
      const auto index = static_cast<std::size_t>(row * n + col);
      const float32_t scaled = static_cast<float32_t>(expected) *
          scales[static_cast<std::size_t>(col)];
      EXPECT_TRUE(conversion_values_equal(scaled, dequantized[index]));
      const float32_t shifted = scaled + output_zero_point;
      const auto expected_u8 = static_cast<uint8_t>(std::clamp(
          shifted, 0.0f,
          static_cast<float32_t>(std::numeric_limits<uint8_t>::max())));
      EXPECT_EQ(expected_u8, requantized[index]);
    }
  }
}

template <typename Atom, bool BroadcastA, bool BroadcastB>
void check_batched_runtime_per_row_column_quantization(
    nint_t batch, nint_t m, nint_t n, nint_t k) {
  using TA = typename Atom::TA;
  using TB = typename Atom::TB;
  using Acc = typename Atom::TAcc;
  static_assert(std::same_as<TA, uint8_t>);
  static_assert(std::same_as<TB, int8_t>);
  static_assert(std::same_as<Acc, int32_t>);
  const nint_t a_batches = BroadcastA ? 1 : batch;
  const nint_t b_batches = BroadcastB ? 1 : batch;
  const nint_t row_scale_batch_stride = BroadcastA ? 0 : m;
  std::vector<float32_t> a(
      static_cast<std::size_t>(a_batches * m * k));
  std::vector<TB> b(static_cast<std::size_t>(b_batches * n * k));
  std::vector<Acc> correction(
      static_cast<std::size_t>(b_batches * n));
  std::vector<float32_t> row_quant_multipliers(
      static_cast<std::size_t>(a_batches * m));
  std::vector<float32_t> row_dequant_scales(
      static_cast<std::size_t>(a_batches * m));
  std::vector<float32_t> column_scales(
      static_cast<std::size_t>(n + 64));
  std::vector<float32_t> c(
      static_cast<std::size_t>(batch * m * n));
  int32_t input_zero_point = 7;

  for (nint_t scale_index = 0; scale_index < a_batches * m;
       ++scale_index) {
    const float32_t multiplier =
        static_cast<float32_t>(4 << (scale_index % 3));
    row_quant_multipliers[static_cast<std::size_t>(scale_index)] =
        multiplier;
    row_dequant_scales[static_cast<std::size_t>(scale_index)] =
        1.0f / multiplier;
  }
  for (nint_t bi = 0; bi < a_batches; ++bi) {
    for (nint_t row = 0; row < m; ++row) {
      const nint_t scale_index = bi * m + row;
      const float32_t scale =
          row_dequant_scales[static_cast<std::size_t>(scale_index)];
      for (nint_t kk = 0; kk < k; ++kk) {
        const int logical =
            static_cast<int>((bi * m * k + row * k + kk) % 7) - 3;
        a[static_cast<std::size_t>((bi * m + row) * k + kk)] =
            static_cast<float32_t>(logical) * scale;
      }
    }
  }
  for (nint_t i = 0; i < b_batches * n * k; ++i)
    b[static_cast<std::size_t>(i)] = conversion_value<TB>(i, 11);
  for (nint_t bi = 0; bi < b_batches; ++bi) {
    for (nint_t col = 0; col < n; ++col) {
      Acc sum_b{};
      for (nint_t kk = 0; kk < k; ++kk)
        sum_b += static_cast<Acc>(b[static_cast<std::size_t>(
            (bi * n + col) * k + kk)]);
      correction[static_cast<std::size_t>(bi * n + col)] =
          -static_cast<Acc>(input_zero_point) * sum_b;
    }
  }
  for (nint_t col = 0; col < n + 64; ++col)
    column_scales[static_cast<std::size_t>(col)] =
        0.03125f + static_cast<float32_t>(col % 5) * 0.0078125f;

  const auto a_layout = [&] {
    if constexpr (BroadcastA) {
      return tensor::make_layout(
          tensor::make_shape(
              meta::Any{batch}, meta::Any{m}, meta::Any{k}),
          tensor::make_strides(
              meta::cint<0>, meta::Any{k}, meta::cint<1>));
    } else {
      return tensor::make_layout(tensor::make_shape(
          meta::Any{batch}, meta::Any{m}, meta::Any{k}));
    }
  }();
  const auto b_layout = [&] {
    if constexpr (BroadcastB) {
      return tensor::make_layout(
          tensor::make_shape(
              meta::Any{batch}, meta::Any{n}, meta::Any{k}),
          tensor::make_strides(
              meta::cint<0>, meta::Any{k}, meta::cint<1>));
    } else {
      return tensor::make_layout(tensor::make_shape(
          meta::Any{batch}, meta::Any{n}, meta::Any{k}));
    }
  }();
  const auto c_layout = tensor::make_layout(tensor::make_shape(
      meta::Any{batch}, meta::Any{m}, meta::Any{n}));
  const auto correction_layout = [&] {
    if constexpr (BroadcastB) {
      return tensor::make_layout(
          tensor::make_shape(
              meta::Any{batch}, meta::Any{m}, meta::Any{n}),
          tensor::make_strides(
              meta::cint<0>, meta::cint<0>, meta::cint<1>));
    } else {
      return tensor::make_layout(
          tensor::make_shape(
              meta::Any{batch}, meta::Any{m}, meta::Any{n}),
          tensor::make_strides(
              meta::Any{n}, meta::cint<0>, meta::cint<1>));
    }
  }();
  auto quantize_a = ::vecops::matmul::
      make_runtime_per_row_asymmetric_quantize_transform({
          row_quant_multipliers.data(), &input_zero_point,
          row_scale_batch_stride});
  auto dequantize = ::vecops::matmul::
      make_runtime_per_row_column_dequantize_transform({
          row_dequant_scales.data(), column_scales.data(),
          row_scale_batch_stride});
  auto operation = make_test_matmul_invocation(ops::MatmulConfig<Atom>{},
      m, n, k,
      tensor::input<TA>(tensor::make_tensor(a.data(), a_layout), quantize_a),
      tensor::input<TB>(tensor::make_tensor(b.data(), b_layout)),
      tensor::input<Acc>(
          tensor::make_tensor(correction.data(), correction_layout)),
      tensor::output<Acc>(
          tensor::make_tensor(c.data(), c_layout), dequantize));
  kernel::Workspace storage(operation.required_workspace());
  auto workspace = storage.view();
  operation(workspace);

  for (nint_t bi = 0; bi < batch; ++bi) {
    const nint_t a_batch = BroadcastA ? 0 : bi;
    const nint_t b_batch = BroadcastB ? 0 : bi;
    for (nint_t row = 0; row < m; ++row) {
      const nint_t scale_index = a_batch * m + row;
      for (nint_t col = 0; col < n; ++col) {
        Acc expected = correction[static_cast<std::size_t>(
            b_batch * n + col)];
        for (nint_t kk = 0; kk < k; ++kk) {
          const auto av = a[static_cast<std::size_t>(
              (a_batch * m + row) * k + kk)];
          const TA qa = static_cast<TA>(
              av * row_quant_multipliers[
                       static_cast<std::size_t>(scale_index)] +
              input_zero_point);
          const TB bv = b[static_cast<std::size_t>(
              (b_batch * n + col) * k + kk)];
          expected += static_cast<Acc>(qa) * static_cast<Acc>(bv);
        }
        const float32_t reference = static_cast<float32_t>(expected) *
            row_dequant_scales[static_cast<std::size_t>(scale_index)] *
            column_scales[static_cast<std::size_t>(col)];
        const auto actual = c[static_cast<std::size_t>(
            (bi * m + row) * n + col)];
        EXPECT_TRUE(conversion_values_equal(reference, actual))
            << "batch=" << bi << " row=" << row << " col=" << col
            << " broadcast_a=" << BroadcastA
            << " broadcast_b=" << BroadcastB;
      }
    }
  }
}

template <typename Atom, ExtentMode Mode,
          nint_t M, nint_t N, nint_t K>
std::vector<typename Atom::TAcc> run_native_extent_case() {
  using TA = typename Atom::TA;
  using TB = typename Atom::TB;
  using Acc = typename Atom::TAcc;
  std::vector<TA> a(static_cast<std::size_t>(M * K));
  std::vector<TB> b(static_cast<std::size_t>(N * K));
  std::vector<Acc> c(static_cast<std::size_t>(M * N));
  for (nint_t i = 0; i < M * K; ++i) {
    a[static_cast<std::size_t>(i)] = conversion_value<TA>(i, 3);
  }
  for (nint_t i = 0; i < N * K; ++i) {
    b[static_cast<std::size_t>(i)] = conversion_value<TB>(i, 11);
  }

  const auto execute = [&]<typename MExtent, typename NExtent,
                            typename KExtent>(
                           MExtent m_extent, NExtent n_extent,
                           KExtent k_extent) {
    auto at = tensor::make_tensor(
        a.data(), tensor::make_layout(
                      tensor::make_shape(m_extent, k_extent)));
    auto bt = tensor::make_tensor(
        b.data(), tensor::make_layout(
                      tensor::make_shape(n_extent, k_extent)));
    auto ct = tensor::make_tensor(
        c.data(), tensor::make_layout(
                      tensor::make_shape(m_extent, n_extent)));
    auto operation = make_test_matmul_invocation(ops::MatmulConfig<Atom>{},
        m_extent, n_extent, k_extent, at, bt, ct);
    kernel::Workspace storage(operation.required_workspace());
    auto workspace = storage.view();
    operation(workspace);
  };
  if constexpr (Mode == ExtentMode::Const) {
    execute(meta::cint<M>, meta::cint<N>, meta::cint<K>);
  } else {
    execute(meta::Any{M}, meta::Any{N}, meta::Any{K});
  }
  return c;
}

template <typename Atom, nint_t M, nint_t N, nint_t K>
void check_native_extent_pair() {
  using Acc = typename Atom::TAcc;
  const auto dynamic =
      run_native_extent_case<Atom, ExtentMode::Dynamic, M, N, K>();
  const auto constant =
      run_native_extent_case<Atom, ExtentMode::Const, M, N, K>();
  ASSERT_EQ(dynamic.size(), constant.size());
  for (std::size_t i = 0; i < dynamic.size(); ++i) {
    if constexpr (std::is_floating_point_v<Acc>) {
      EXPECT_TRUE(conversion_values_equal(dynamic[i], constant[i]))
          << "extent pair mismatch at index=" << i
          << " shape=" << M << "x" << N << "x" << K;
    } else {
      EXPECT_EQ(dynamic[i], constant[i])
          << "extent pair mismatch at index=" << i
          << " shape=" << M << "x" << N << "x" << K;
    }
  }
}

template <typename Atom, bool BroadcastB, ExtentMode Mode,
          nint_t Batch, nint_t M, nint_t N, nint_t K>
std::vector<typename Atom::TAcc> run_batched_native_extent_case() {
  using TA = typename Atom::TA;
  using TB = typename Atom::TB;
  using Acc = typename Atom::TAcc;
  constexpr nint_t BBatches = BroadcastB ? 1 : Batch;
  std::vector<TA> a(static_cast<std::size_t>(Batch * M * K));
  std::vector<TB> b(static_cast<std::size_t>(BBatches * N * K));
  std::vector<Acc> c(static_cast<std::size_t>(Batch * M * N));
  for (nint_t i = 0; i < Batch * M * K; ++i) {
    a[static_cast<std::size_t>(i)] = conversion_value<TA>(i, 5);
  }
  for (nint_t i = 0; i < BBatches * N * K; ++i) {
    b[static_cast<std::size_t>(i)] = conversion_value<TB>(i, 13);
  }

  const auto execute = [&]<typename BatchExtent, typename MExtent,
                            typename NExtent, typename KExtent>(
                           BatchExtent batch_extent, MExtent m_extent,
                           NExtent n_extent, KExtent k_extent) {
    auto at = tensor::make_tensor(
        a.data(), tensor::make_layout(tensor::make_shape(
                      batch_extent, m_extent, k_extent)));
    auto bt = [&] {
      if constexpr (BroadcastB) {
        return tensor::make_tensor(
            b.data(), tensor::make_layout(
                          tensor::make_shape(
                              batch_extent, n_extent, k_extent),
                          tensor::make_strides(
                              meta::cint<0>, k_extent, meta::cint<1>)));
      } else {
        return tensor::make_tensor(
            b.data(), tensor::make_layout(tensor::make_shape(
                          batch_extent, n_extent, k_extent)));
      }
    }();
    auto ct = tensor::make_tensor(
        c.data(), tensor::make_layout(tensor::make_shape(
                      batch_extent, m_extent, n_extent)));
    auto operation = make_test_matmul_invocation(ops::MatmulConfig<Atom>{},
        m_extent, n_extent, k_extent, at, bt, ct);
    kernel::Workspace storage(operation.required_workspace());
    auto workspace = storage.view();
    operation(workspace);
  };
  if constexpr (Mode == ExtentMode::Const) {
    execute(
        meta::cint<Batch>, meta::cint<M>, meta::cint<N>, meta::cint<K>);
  } else {
    execute(
        meta::Any{Batch}, meta::Any{M}, meta::Any{N}, meta::Any{K});
  }
  return c;
}

template <typename Atom, bool BroadcastB,
          nint_t Batch, nint_t M, nint_t N, nint_t K>
void check_batched_native_extent_pair() {
  using Acc = typename Atom::TAcc;
  const auto dynamic = run_batched_native_extent_case<
      Atom, BroadcastB, ExtentMode::Dynamic, Batch, M, N, K>();
  const auto constant = run_batched_native_extent_case<
      Atom, BroadcastB, ExtentMode::Const, Batch, M, N, K>();
  ASSERT_EQ(dynamic.size(), constant.size());
  for (std::size_t i = 0; i < dynamic.size(); ++i) {
    if constexpr (std::is_floating_point_v<Acc>) {
      EXPECT_TRUE(conversion_values_equal(dynamic[i], constant[i]))
          << "batch extent pair mismatch at index=" << i;
    } else {
      EXPECT_EQ(dynamic[i], constant[i])
          << "batch extent pair mismatch at index=" << i;
    }
  }
}

} // namespace vecops::test::matmul

#endif // VECOPS_TESTS_OPS_MATMUL_CONVERSION_TEST_COMMON_H
