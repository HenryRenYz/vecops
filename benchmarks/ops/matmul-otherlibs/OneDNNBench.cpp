#include "OneDNNRunner.h"

namespace vecops::bench::matmul_otherlibs {

} // namespace vecops::bench::matmul_otherlibs

int main(int argc, char** argv) {
  using namespace vecops::bench::matmul_otherlibs;
  register_external_cases(
      "oneDNN", [](benchmark::State& state, const Case& c, Operation op) {
        run_external_benchmark_factory(
            state, c, op,
            [&state, &c](Buffers& buffers, Operation selected) {
              auto runner = std::make_shared<OneDNNRunner>(
                  c, buffers, selected);
              state.SetLabel(runner->implementation());
              state.counters["weight_reorder_required"] = benchmark::Counter(
                  runner->weights_reorder_required() ? 1.0 : 0.0);
              state.counters["prepared_weight_bytes"] = benchmark::Counter(
                  static_cast<double>(runner->prepared_weight_bytes()));
              return [runner = std::move(runner)]() mutable { (*runner)(); };
            },
            c.batch > 1, true, false);
      }, 0.1, 7, "raw_e2e", true, false, false);
  return run_registered_benchmarks(argc, argv, "matmul_otherlibs_onednn");
}
