#include "ProviderCommon.h"

#include <cblas.h>

#include <charconv>
#include <cstdlib>
#include <cstdint>
#include <stdexcept>
#include <string_view>
#include <type_traits>

namespace vecops::bench::matmul_otherlibs {

static_assert(sizeof(bfloat16_t) == sizeof(bfloat16));
static_assert(std::is_trivially_copyable_v<bfloat16_t>);

int openblas_repetitions() {
  constexpr int DefaultRepetitions = 7;
  const char* text = std::getenv("VECOPS_OPENBLAS_REPETITIONS");
  if (text == nullptr || *text == '\0') return DefaultRepetitions;
  int repetitions = 0;
  const std::string_view value{text};
  const auto [end, error] =
      std::from_chars(value.data(), value.data() + value.size(), repetitions);
  if (error != std::errc{} || end != value.data() + value.size() ||
      repetitions <= 0) {
    throw std::runtime_error(
        "VECOPS_OPENBLAS_REPETITIONS must be a positive integer");
  }
  return repetitions;
}

void run_openblas(const Case& c, Buffers& buffers, Operation op) {
  const float beta = op == Operation::GemmAdd ? 1.0f : 0.0f;
#if defined(__aarch64__)
  const nint_t physical_k = c.k + c.k % 2;
  static thread_local std::vector<bfloat16_t> padded_x;
  static thread_local std::vector<bfloat16_t> padded_weight;
  if (physical_k != c.k) {
    padded_x.assign(checked_elements(c.batch, c.n, physical_k),
                    bfloat16_t{0.0f});
    padded_weight.assign(checked_elements(c.batch, c.m, physical_k),
                         bfloat16_t{0.0f});
    for (nint_t batch = 0; batch < c.batch; ++batch) {
      for (nint_t row = 0; row < c.n; ++row) {
        std::copy_n(
            buffers.x.data() + checked_elements(batch, c.n, c.k) +
                static_cast<std::size_t>(row * c.k),
            static_cast<std::size_t>(c.k),
            padded_x.data() + checked_elements(batch, c.n, physical_k) +
                static_cast<std::size_t>(row * physical_k));
      }
      for (nint_t row = 0; row < c.m; ++row) {
        std::copy_n(
            buffers.weight.data() + checked_elements(batch, c.m, c.k) +
                static_cast<std::size_t>(row * c.k),
            static_cast<std::size_t>(c.k),
            padded_weight.data() +
                checked_elements(batch, c.m, physical_k) +
                static_cast<std::size_t>(row * physical_k));
      }
    }
  }
#else
  const nint_t physical_k = c.k;
#endif
  for (nint_t batch = 0; batch < c.batch; ++batch) {
    const auto x_offset = checked_elements(batch, c.n, physical_k);
    const auto w_offset = checked_elements(batch, c.m, physical_k);
    const auto y_offset = checked_elements(batch, c.n, c.m);
#if defined(__aarch64__)
    const bfloat16_t* x = physical_k == c.k ? buffers.x.data() : padded_x.data();
    const bfloat16_t* weight =
        physical_k == c.k ? buffers.weight.data() : padded_weight.data();
#else
    const bfloat16_t* x = buffers.x.data();
    const bfloat16_t* weight = buffers.weight.data();
#endif
    cblas_sbgemm(
        CblasRowMajor, CblasNoTrans, CblasTrans,
        static_cast<blasint>(c.n), static_cast<blasint>(c.m),
        static_cast<blasint>(physical_k), 1.0f,
        reinterpret_cast<const bfloat16*>(x + x_offset),
        static_cast<blasint>(physical_k),
        reinterpret_cast<const bfloat16*>(weight + w_offset),
        static_cast<blasint>(physical_k), beta,
        buffers.output.data() + y_offset, static_cast<blasint>(c.m));
  }
  if (op == Operation::Bias || op == Operation::BiasRelu ||
      op == Operation::BiasSilu) {
    for (nint_t batch = 0; batch < c.batch; ++batch) {
      const auto y_offset = checked_elements(batch, c.n, c.m);
      for (nint_t row = 0; row < c.n; ++row) {
        for (nint_t col = 0; col < c.m; ++col) {
          const auto index = y_offset + static_cast<std::size_t>(row * c.m + col);
          buffers.output[index] = apply_epilogue(
              buffers.output[index], op,
              buffers.bias[static_cast<std::size_t>(col)]);
        }
      }
    }
  }
}

} // namespace vecops::bench::matmul_otherlibs

int main(int argc, char** argv) {
  using namespace vecops::bench::matmul_otherlibs;
  openblas_set_num_threads(1);
  register_external_cases(
      "OpenBLAS", [](benchmark::State& state, const Case& c, Operation op) {
        run_external_benchmark(
            state, c, op,
            [&c](Buffers& buffers, Operation selected) {
              run_openblas(c, buffers, selected);
            });
#if defined(__aarch64__)
        state.counters["padding_ratio"] = benchmark::Counter(
            static_cast<double>(c.k + c.k % 2) / static_cast<double>(c.k));
#endif
        state.SetLabel(VECOPS_OPENBLAS_BUILD_TARGET);
      }, 0.1, openblas_repetitions());
  return run_registered_benchmarks(argc, argv, "matmul_otherlibs_openblas");
}
