// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
#include "ProviderCommon.h"

// ACL's public NEGEMM restricts BF16 GEMM output to BF16.  The arm_gemm
// layer underneath it is the ACL component that exposes BF16 x BF16 -> FP32.
#include "arm_common/bfloat.hpp"
#include "arm_gemm/arm_gemm.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace vecops::bench::matmul_otherlibs {

class AlignedBytes {
public:
  explicit AlignedBytes(std::size_t bytes, std::size_t alignment = 64)
      : storage_(bytes + alignment), alignment_(alignment) {}

  void* data() {
    const auto address = reinterpret_cast<std::uintptr_t>(storage_.data());
    const auto aligned = (address + alignment_ - 1) & ~(alignment_ - 1);
    return reinterpret_cast<void*>(aligned);
  }

private:
  std::vector<std::byte> storage_;
  std::size_t alignment_;
};

inline arm_gemm::Activation acl_activation(Operation op) {
  return op == Operation::BiasRelu
      ? arm_gemm::Activation{arm_gemm::Activation::Type::ReLU}
      : arm_gemm::Activation{};
}

class ACLRunner {
public:
  ACLRunner(const Case& c, Buffers& buffers, Operation op)
      : c_(c), buffers_(buffers), op_(op),
        transposed_weight_(buffers.weight.size()),
        args_(
            &arm_compute::CPUInfo::get(),
            static_cast<unsigned int>(c.n),
            static_cast<unsigned int>(c.m),
            static_cast<unsigned int>(c.k), 1, 1,
            static_cast<unsigned int>(c.batch), false, acl_activation(op), 1,
            false, false, op == Operation::GemmAdd, nullptr),
        gemm_(arm_gemm::gemm<
              arm_gemm::bfloat16, arm_gemm::bfloat16, float>(args_)),
        prepacked_b_(gemm_ ? gemm_->get_B_pretransposed_array_size() : 0),
        workspace_(gemm_ ? gemm_->get_working_size() : 0) {
    static_assert(sizeof(arm_gemm::bfloat16) == sizeof(bfloat16_t));
    static_assert(alignof(arm_gemm::bfloat16) <= alignof(bfloat16_t));
    if (!gemm_) throw std::runtime_error("ACL arm_gemm found no BF16 kernel");
    if (gemm_->get_working_size() != 0) {
      gemm_->set_working_space(workspace_.data());
    }
    gemm_->set_nthreads(1);
    set_arrays();
    config_ = gemm_->get_config();
  }

  void operator()() {
    prepare_weights();
    set_arrays();
    const auto window = gemm_->get_window_size();
    const arm_gemm::ndcoord_t work{
        {0, window.get_size(0)}, {0, window.get_size(1)},
        {0, window.get_size(2)}, {0, window.get_size(3)},
        {0, window.get_size(4)}, {0, window.get_size(5)}};
    const arm_gemm::ndcoord_t locator{
        {0, 1}, {0, 1}, {0, 1}, {0, 1}, {0, 1}, {0, 1}};
    gemm_->execute(work, locator, 0);
    if (op_ == Operation::BiasSilu) {
      for (float& value : buffers_.output) {
        value = value / (1.0f + std::exp(-value));
      }
    }
  }

  const arm_gemm::GemmConfig& selected_config() const { return config_; }

private:
  void prepare_weights() {
    // arm_gemm's public B view is KxM and its selected kernel may require a
    // second opaque pretranspose. Both data transformations are timed in the
    // common raw_e2e track; kernel/configuration selection remains setup.
    for (nint_t batch = 0; batch < c_.batch; ++batch) {
      const auto base = checked_elements(batch, c_.m, c_.k);
      for (nint_t col = 0; col < c_.m; ++col) {
        for (nint_t kk = 0; kk < c_.k; ++kk) {
          transposed_weight_[base + static_cast<std::size_t>(kk * c_.m + col)] =
              buffers_.weight[
                  base + static_cast<std::size_t>(col * c_.k + kk)];
        }
      }
    }
    if (gemm_->B_is_pretransposed()) {
      gemm_->pretranspose_B_array(
          prepacked_b_.data(), weights(), static_cast<int>(c_.m),
          static_cast<int>(c_.m * c_.k), false);
      gemm_->set_pretransposed_B_data(prepacked_b_.data());
    }
  }

  const arm_gemm::bfloat16* activations() const {
    return reinterpret_cast<const arm_gemm::bfloat16*>(buffers_.x.data());
  }

  const arm_gemm::bfloat16* weights() const {
    return reinterpret_cast<const arm_gemm::bfloat16*>(
        transposed_weight_.data());
  }

  void set_arrays() {
    const float* bias =
        op_ == Operation::Bias || op_ == Operation::BiasRelu ||
                op_ == Operation::BiasSilu
            ? buffers_.bias.data()
            : nullptr;
    gemm_->set_arrays(
        activations(), static_cast<int>(c_.k), 0,
        static_cast<int>(c_.n * c_.k), weights(),
        static_cast<int>(c_.m), static_cast<int>(c_.m * c_.k),
        buffers_.output.data(), static_cast<int>(c_.m), 0,
        static_cast<int>(c_.n * c_.m), bias, 0);
  }

  const Case& c_;
  Buffers& buffers_;
  Operation op_;
  std::vector<bfloat16_t> transposed_weight_;
  arm_gemm::GemmArgs args_;
  arm_gemm::UniqueGemmCommon<
      arm_gemm::bfloat16, arm_gemm::bfloat16, float> gemm_;
  AlignedBytes prepacked_b_;
  AlignedBytes workspace_;
  arm_gemm::GemmConfig config_;
};

void register_acl_cases() {
  register_external_cases(
      "ACL", [](benchmark::State& state, const Case& c, Operation op) {
        try {
          run_external_benchmark_factory(
              state, c, op,
              [&state, &c](Buffers& buffers, Operation operation) {
                auto runner = std::make_shared<ACLRunner>(c, buffers, operation);
                const auto& config = runner->selected_config();
                state.SetLabel(config.filter);
                state.counters["selected_kc"] = benchmark::Counter(
                    static_cast<double>(config.inner_block_size));
                state.counters["selected_outer_block"] = benchmark::Counter(
                    static_cast<double>(config.outer_block_size));
                return [runner = std::move(runner)]() mutable { (*runner)(); };
              },
              true, true, false);
        } catch (const std::exception& error) {
          state.SkipWithError(error.what());
        }
      },
      0.1, 7, "raw_e2e", true, true, false);
}

} // namespace vecops::bench::matmul_otherlibs

int main(int argc, char** argv) {
  using namespace vecops::bench::matmul_otherlibs;
  register_acl_cases();
  return run_registered_benchmarks(argc, argv, "matmul_otherlibs_acl");
}
