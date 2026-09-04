#include "ProviderCommon.h"

#include <kupl_mma.h>

#include <algorithm>
#include <stdexcept>
#include <vector>

#include "vecops/ops/Transpose.h"
#include "vecops/tensor/Tensor.h"

namespace vecops::bench::matmul_otherlibs {

using namespace ::kupl::tensor;

constexpr nint_t ceil_to(nint_t value, nint_t multiple) {
  return (value + multiple - 1) / multiple * multiple;
}

template <nint_t LogicalK>
class KUPLRunner {
public:
  static constexpr nint_t PhysicalK = ceil_to(LogicalK, 2);
  static constexpr int KGroups = static_cast<int>(PhysicalK / 2);

  KUPLRunner(const Case& c, Buffers& buffers, Operation op)
      : c_(c), buffers_(buffers), op_(op),
        row_tiles_(ceil_to(c.n, 16) / 16),
        col_tiles_(ceil_to(c.m, 64) / 64),
        packed_a_(checked_elements(c.batch, row_tiles_ * 16, PhysicalK)),
        packed_b_(checked_elements(c.batch, col_tiles_ * 64, PhysicalK)),
        tile_c_(16 * 64) {
    if (c.k != LogicalK) {
      throw std::invalid_argument("KUPL compile-time K does not match case");
    }
    std::fill(packed_a_.begin(), packed_a_.end(), bfloat16_t{0.0f});
    std::fill(packed_b_.begin(), packed_b_.end(), bfloat16_t{0.0f});
  }

  void operator()() {
    pack_inputs();
    for (nint_t batch = 0; batch < c_.batch; ++batch) {
      for (nint_t mt = 0; mt < row_tiles_; ++mt) {
        for (nint_t nt = 0; nt < col_tiles_; ++nt) {
          prepare_c_tile(batch, mt, nt);
          run_tile(batch, mt, nt);
          finish_c_tile(batch, mt, nt);
        }
      }
    }
  }

  double padding_ratio() const {
    const double physical = static_cast<double>(c_.batch) *
        static_cast<double>(row_tiles_ * 16) *
        static_cast<double>(col_tiles_ * 64) *
        static_cast<double>(PhysicalK);
    const double logical = static_cast<double>(c_.batch) *
        static_cast<double>(c_.n) * static_cast<double>(c_.m) *
        static_cast<double>(c_.k);
    return physical / logical;
  }

private:
  void pack_inputs() {
    using namespace ::vecops;
    using namespace ::vecops::tensor;
    auto transpose = ops::transpose(ops::TransposeConfig<bfloat16_t>{});
    ExecutionSession execution{};
    for (nint_t batch = 0; batch < c_.batch; ++batch) {
      const auto x_base = checked_elements(batch, c_.n, c_.k);
      const auto w_base = checked_elements(batch, c_.m, c_.k);
      const auto pa_base = checked_elements(batch, row_tiles_ * 16, PhysicalK);
      const auto pb_base = checked_elements(batch, col_tiles_ * 64, PhysicalK);
      for (nint_t mt = 0; mt < row_tiles_; ++mt) {
        const nint_t rows = std::min<nint_t>(16, c_.n - mt * 16);
        auto input_layout = make_layout(
            make_shape(meta::Any{rows}, meta::Any{c_.k}),
            make_strides(meta::Any{c_.k}, meta::cint<1>));
        auto output_layout = make_layout(
            make_shape(meta::Any{c_.k}, meta::Any{rows}),
            make_strides(meta::cint<16>, meta::cint<1>));
        auto input = make_tensor(
            buffers_.x.data() + x_base +
                static_cast<std::size_t>(mt * 16 * c_.k),
            input_layout);
        auto output = make_tensor(
            packed_a_.data() + pa_base +
                static_cast<std::size_t>(mt * 16 * PhysicalK),
            output_layout);
        transpose(execution, input, output);
      }
      for (nint_t nt = 0; nt < col_tiles_; ++nt) {
        const nint_t cols = std::min<nint_t>(64, c_.m - nt * 64);
        auto input_layout = make_layout(
            make_shape(meta::Any{cols}, meta::Any{c_.k}),
            make_strides(meta::Any{c_.k}, meta::cint<1>));
        auto output_layout = make_layout(
            make_shape(meta::Any{c_.k}, meta::Any{cols}),
            make_strides(meta::cint<64>, meta::cint<1>));
        auto input = make_tensor(
            buffers_.weight.data() + w_base +
                static_cast<std::size_t>(nt * 64 * c_.k),
            input_layout);
        auto output = make_tensor(
            packed_b_.data() + pb_base +
                static_cast<std::size_t>(nt * 64 * PhysicalK),
            output_layout);
        transpose(execution, input, output);
      }
    }
  }

  void prepare_c_tile(nint_t batch, nint_t mt, nint_t nt) {
    std::fill(tile_c_.begin(), tile_c_.end(), 0.0f);
    for (nint_t row = 0; row < 16; ++row) {
      const nint_t logical_row = mt * 16 + row;
      if (logical_row >= c_.n) continue;
      for (nint_t col = 0; col < 64; ++col) {
        const nint_t logical_col = nt * 64 + col;
        if (logical_col >= c_.m) continue;
        float value = 0.0f;
        if (op_ == Operation::GemmAdd) {
          value = buffers_.initial[
              checked_elements(batch, c_.n, c_.m) +
              static_cast<std::size_t>(logical_row * c_.m + logical_col)];
        } else if (op_ == Operation::Bias ||
                   op_ == Operation::BiasRelu ||
                   op_ == Operation::BiasSilu) {
          value = buffers_.bias[static_cast<std::size_t>(logical_col)];
        }
        tile_c_[static_cast<std::size_t>(row * 64 + col)] = value;
      }
    }
  }

  void run_tile(nint_t batch, nint_t mt, nint_t nt) {
    const auto pa_base = checked_elements(batch, row_tiles_ * 16, PhysicalK) +
        static_cast<std::size_t>(mt * 16 * PhysicalK);
    const auto pb_base = checked_elements(batch, col_tiles_ * 64, PhysicalK) +
        static_cast<std::size_t>(nt * 64 * PhysicalK);

    auto shape_a = make_shape(Int<16>{}, Int<PhysicalK>{});
    auto shape_b = make_shape(Int<PhysicalK>{}, Int<64>{});
    auto shape_c = make_shape(Int<16>{}, Int<64>{});
    auto layout_a = make_layout(
        shape_a, make_stride(Int<1>{}, Int<16>{}));
    auto layout_b = make_layout(
        shape_b, make_stride(Int<64>{}, Int<1>{}));
    auto layout_c = make_layout(shape_c, make_stride(Int<64>{}, Int<1>{}));
    auto tiled_mma = make_tiled_mma(
        Ops<KP36_16x64x2_BF16BF16F32>{},
        make_shape(Int<1>{}, Int<1>{}, Int<KGroups>{}));
    auto tiled_store = make_tiled_store(
        Ops<KP36_16x64_F32_STORE>{}, make_shape(Int<1>{}, Int<1>{}));

    auto a = make_tensor(
        reinterpret_cast<::bfloat16_t*>(packed_a_.data() + pa_base), layout_a);
    auto b = make_tensor(
        reinterpret_cast<::bfloat16_t*>(packed_b_.data() + pb_base), layout_b);
    auto c = make_tensor(tile_c_.data(), layout_c);
    mma(tiled_mma, c, a, b, c);
    store(tiled_store, c);
  }

  void finish_c_tile(nint_t batch, nint_t mt, nint_t nt) {
    const auto y_base = checked_elements(batch, c_.n, c_.m);
    for (nint_t row = 0; row < 16; ++row) {
      const nint_t logical_row = mt * 16 + row;
      if (logical_row >= c_.n) continue;
      for (nint_t col = 0; col < 64; ++col) {
        const nint_t logical_col = nt * 64 + col;
        if (logical_col >= c_.m) continue;
        float value = tile_c_[static_cast<std::size_t>(row * 64 + col)];
        if (op_ == Operation::BiasRelu) value = std::max(value, 0.0f);
        if (op_ == Operation::BiasSilu) {
          value = value / (1.0f + std::exp(-value));
        }
        buffers_.output[y_base + static_cast<std::size_t>(
            logical_row * c_.m + logical_col)] = value;
      }
    }
  }

  const Case& c_;
  Buffers& buffers_;
  Operation op_;
  nint_t row_tiles_;
  nint_t col_tiles_;
  std::vector<bfloat16_t> packed_a_;
  std::vector<bfloat16_t> packed_b_;
  std::vector<float> tile_c_;
};

template <nint_t K>
void register_kupl_case(const Case& c) {
  for (Operation op : {Operation::Gemm, Operation::GemmAdd,
                       Operation::Bias, Operation::BiasRelu,
                       Operation::BiasSilu}) {
    if ((c.operations & op_bit(op)) == 0) continue;
    const auto epilogue =
        op == Operation::BiasRelu || op == Operation::BiasSilu
            ? "separate"
            : "native_accumulate";
    const auto name = benchmark_name(
        "KUPL-MMA", c, op, "NA", "raw_e2e", epilogue);
    auto* registered = benchmark::RegisterBenchmark(
        name.c_str(), [c, op](benchmark::State& state) {
          Buffers buffers(c);
          KUPLRunner<K> runner(c, buffers, op);
          buffers.prepare_output(op);
          runner();
          std::string error;
          if (!verify_samples(c, op, buffers, &error)) {
            state.SkipWithError(error);
            return;
          }
          buffers.prepare_output(op);
          for (auto _ : state) {
            runner();
            benchmark::DoNotOptimize(buffers.output.data());
            benchmark::ClobberMemory();
          }
          set_counters(state, c, false, true, false);
          state.counters["physical_k"] = benchmark::Counter(
              static_cast<double>(KUPLRunner<K>::PhysicalK));
          state.counters["padding_ratio"] =
              benchmark::Counter(runner.padding_ratio());
          state.SetLabel(
              "KP36_16x64x2_BF16BF16F32+vecops_transpose_pack");
        });
    registered->Unit(benchmark::kMicrosecond);
    configure_comparison_benchmark(registered, 0.1, 7);
  }
}

void register_kupl_cases() {
  register_kupl_case<256>(CoreCases[0]);
  register_kupl_case<769>(CoreCases[1]);
  register_kupl_case<576>(CoreCases[2]);
  register_kupl_case<4096>(CoreCases[3]);
  register_kupl_case<4096>(CoreCases[4]);
  register_kupl_case<4096>(CoreCases[5]);
  register_kupl_case<4096>(CoreCases[6]);
  register_kupl_case<4096>(CoreCases[7]);
  register_kupl_case<4096>(CoreCases[8]);
  register_kupl_case<4096>(CoreCases[9]);
  register_kupl_case<11008>(CoreCases[10]);
  register_kupl_case<11008>(CoreCases[11]);
  register_kupl_case<128>(CoreCases[12]);
  register_kupl_case<128>(AF3Cases[0]);
  register_kupl_case<128>(AF3Cases[1]);
  register_kupl_case<128>(AF3Cases[2]);
  register_kupl_case<128>(AF3Cases[3]);
  register_kupl_case<128>(AF3Cases[4]);
  register_kupl_case<128>(AF3Cases[5]);
  register_kupl_case<128>(AF3Cases[6]);
  register_kupl_case<32>(AF3Cases[7]);
  register_kupl_case<32>(AF3Cases[8]);
  register_kupl_case<32>(AF3Cases[9]);
  register_kupl_case<32>(AF3Cases[10]);
  register_kupl_case<128>(AF3Cases[11]);
  register_kupl_case<512>(AF3Cases[12]);
  register_kupl_case<1536>(AF3Cases[13]);
  register_kupl_case<2048>(AF3Cases[14]);
  register_kupl_case<128>(AF3Cases[15]);
  register_kupl_case<512>(AF3Cases[16]);
  register_kupl_case<1536>(AF3Cases[17]);
}

} // namespace vecops::bench::matmul_otherlibs

int main(int argc, char** argv) {
  using namespace vecops::bench::matmul_otherlibs;
  register_kupl_cases();
  return run_registered_benchmarks(argc, argv, "matmul_otherlibs_kupl");
}
