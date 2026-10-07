// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
#include "ProviderCommon.h"

#include <libxsmm.h>

#include <stdexcept>
#include <vector>

namespace vecops::bench::matmul_otherlibs {

class LIBXSMMRunner {
public:
  LIBXSMMRunner(const Case& c, Buffers& buffers, Operation op)
      : c_(c), buffers_(buffers), op_(op),
        physical_k_(c.k + c.k % 2),
        packed_weight_(checked_elements(c.batch, c.m, physical_k_)),
        packed_x_(c.k % 2 != 0
                      ? checked_elements(c.batch, c.n, physical_k_)
                      : 0),
        uses_odd_k_fallback_(c.k % 2 != 0) {
    const auto shape = libxsmm_create_gemm_shape(
        static_cast<libxsmm_blasint>(c.m),
        static_cast<libxsmm_blasint>(c.n),
        static_cast<libxsmm_blasint>(physical_k_),
        static_cast<libxsmm_blasint>(c.m),
        static_cast<libxsmm_blasint>(physical_k_),
        static_cast<libxsmm_blasint>(c.m),
        LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16,
        LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32);
    libxsmm_bitfield flags = LIBXSMM_GEMM_FLAG_NONE;
    if (op != Operation::GemmAdd) flags |= LIBXSMM_GEMM_FLAG_BETA_0;
#if defined(ARCH_X86_FAMILY)
    const int native_arch = libxsmm_get_target_archid();
    if (uses_odd_k_fallback_) {
      // Current SPR/AMX code generation divides by an empty tile block for
      // odd K (e.g. the representative K=769 tail). CPX is the highest
      // non-AMX BF16 target and handles the same logical shape directly.
      libxsmm_set_target_archid(LIBXSMM_X86_AVX512_CPX);
    }
#endif
    kernel_ = libxsmm_dispatch_gemm(
        shape, flags, LIBXSMM_GEMM_PREFETCH_NONE);
#if defined(ARCH_X86_FAMILY)
    if (uses_odd_k_fallback_) libxsmm_set_target_archid(native_arch);
#endif
    if (kernel_ == nullptr) {
      throw std::runtime_error("LIBXSMM could not dispatch this GEMM shape");
    }
  }

  bool uses_odd_k_fallback() const { return uses_odd_k_fallback_; }
  double padding_ratio() const {
    return static_cast<double>(physical_k_) / static_cast<double>(c_.k);
  }

  void operator()() {
    pack_inputs();
    for (nint_t batch = 0; batch < c_.batch; ++batch) {
      libxsmm_gemm_param params{};
      const auto x_offset = uses_odd_k_fallback_
          ? checked_elements(batch, c_.n, physical_k_)
          : checked_elements(batch, c_.n, c_.k);
      const auto w_offset = checked_elements(batch, c_.m, physical_k_);
      const auto y_offset = checked_elements(batch, c_.n, c_.m);
      params.a.primary = packed_weight_.data() + w_offset;
      params.b.primary = uses_odd_k_fallback_
          ? static_cast<void*>(packed_x_.data() + x_offset)
          : static_cast<void*>(buffers_.x.data() + x_offset);
      params.c.primary = buffers_.output.data() + y_offset;
      kernel_(&params);
    }
    if (op_ == Operation::Bias || op_ == Operation::BiasRelu ||
        op_ == Operation::BiasSilu) {
      for (nint_t batch = 0; batch < c_.batch; ++batch) {
        const auto y_offset = checked_elements(batch, c_.n, c_.m);
        for (nint_t row = 0; row < c_.n; ++row) {
          for (nint_t col = 0; col < c_.m; ++col) {
            const auto index = y_offset +
                static_cast<std::size_t>(row * c_.m + col);
            buffers_.output[index] = apply_epilogue(
                buffers_.output[index], op_,
                buffers_.bias[static_cast<std::size_t>(col)]);
          }
        }
      }
    }
  }

private:
  void pack_inputs() {
    // LIBXSMM consumes column-major A. Convert the public row-major W[M,K]
    // for every raw_e2e invocation; odd K also needs a zero-padded X copy.
    for (nint_t batch = 0; batch < c_.batch; ++batch) {
      const auto source_base = checked_elements(batch, c_.m, c_.k);
      const auto packed_base = checked_elements(batch, c_.m, physical_k_);
      for (nint_t row = 0; row < c_.m; ++row) {
        for (nint_t kk = 0; kk < physical_k_; ++kk) {
          packed_weight_[packed_base +
                         static_cast<std::size_t>(row + kk * c_.m)] =
              kk < c_.k
                  ? buffers_.weight[source_base +
                                    static_cast<std::size_t>(row * c_.k + kk)]
                  : bfloat16_t{0.0f};
        }
      }
      if (uses_odd_k_fallback_) {
        const auto x_source_base = checked_elements(batch, c_.n, c_.k);
        const auto x_packed_base = checked_elements(batch, c_.n, physical_k_);
        for (nint_t row = 0; row < c_.n; ++row) {
          for (nint_t kk = 0; kk < physical_k_; ++kk) {
            packed_x_[x_packed_base +
                      static_cast<std::size_t>(row * physical_k_ + kk)] =
                kk < c_.k
                    ? buffers_.x[x_source_base +
                                 static_cast<std::size_t>(row * c_.k + kk)]
                    : bfloat16_t{0.0f};
          }
        }
      }
    }
  }

  const Case& c_;
  Buffers& buffers_;
  Operation op_;
  nint_t physical_k_;
  std::vector<bfloat16_t> packed_weight_;
  std::vector<bfloat16_t> packed_x_;
  bool uses_odd_k_fallback_;
  libxsmm_gemmfunction kernel_ = nullptr;
};

} // namespace vecops::bench::matmul_otherlibs

int main(int argc, char** argv) {
  using namespace vecops::bench::matmul_otherlibs;
  libxsmm_init();
  register_external_cases(
      "LIBXSMM", [](benchmark::State& state, const Case& c, Operation op) {
        try {
          run_external_benchmark_factory(
              state, c, op,
              [&state, &c](Buffers& buffers, Operation selected) {
                auto runner = std::make_shared<LIBXSMMRunner>(
                    c, buffers, selected);
                if (runner->uses_odd_k_fallback()) {
#if defined(ARCH_X86_FAMILY)
                  state.SetLabel("CPX odd-K fallback (SPR generator SIGFPE)");
                  state.counters["isa_fallback"] = benchmark::Counter(1.0);
#else
                  state.SetLabel("odd-K zero padding");
                  state.counters["isa_fallback"] = benchmark::Counter(0.0);
#endif
                } else {
                  state.counters["isa_fallback"] = benchmark::Counter(0.0);
                }
                state.counters["padding_ratio"] = benchmark::Counter(
                    runner->padding_ratio());
                return [runner = std::move(runner)]() mutable { (*runner)(); };
              },
              false, true, false);
        } catch (const std::exception& error) {
          state.SkipWithError(error.what());
        }
      },
      0.1, 7, "raw_e2e", false, false, false);
  const int result =
      run_registered_benchmarks(argc, argv, "matmul_otherlibs_libxsmm");
  libxsmm_finalize();
  return result;
}
