#include <algorithm>
#include <vector>

#include <benchmark/benchmark.h>

#include "vecops/tensor/DataAccess.h"

using namespace vecops;
using namespace vecops::tensor;

namespace {

using Tag = vec::ScalableTag<float32_t, 0>;
constexpr nint_t kElements = 16384;

template <bool Cursor, bool Convert, bool Strided, bool Prefetch>
void data_access_load(benchmark::State& state) {
  using Memory = std::conditional_t<Convert, float64_t, float32_t>;
  constexpr nint_t LogicalStride = Strided ? 2 : 1;
  std::vector<Memory> storage(
      static_cast<std::size_t>(kElements * LogicalStride));
  for (nint_t i = 0; i < kElements; ++i) {
    storage[static_cast<std::size_t>(i * LogicalStride)] =
        static_cast<Memory>(i % 101);
  }
  auto tensor = make_tensor<1>(storage.data(), {kElements * LogicalStride});
  auto spec = input<float32_t>(tensor);
  using Policy = InputAccessPolicy<0, 1, AccessPlan::direct>;
  kernel::Workspace workspace_storage(0);
  auto workspace = workspace_storage.view();
  auto access = bind(spec, Policy{}, workspace);
  Tag tag{};
  const nint_t lanes = vec::size(tag);

  for (auto _ : state) {
    auto accumulator = vec::zeros(tag);
    if constexpr (Cursor) {
      nint_t offset = 0;
      if constexpr (Strided) {
        auto cursor = access.scan(
            tag, coord(0), axis<0>, kElements,
            vec::strided(LogicalStride));
        while (cursor.has_full()) {
          if constexpr (Prefetch) {
            if (offset + 4 * lanes < kElements)
              vec::prefetch(
                  tag, storage.data() +
                           (offset + 4 * lanes) * LogicalStride);
          }
          accumulator = vec::add(accumulator, cursor.load_full());
          cursor.advance_full();
          offset += lanes;
        }
      } else {
        auto cursor = access.scan(
            tag, coord(0), axis<0>, kElements,
            ContiguousLaneMapping{});
        while (cursor.has_full()) {
          if constexpr (Prefetch) {
            if (offset + 4 * lanes < kElements)
              vec::prefetch(tag, storage.data() + offset + 4 * lanes);
          }
          accumulator = vec::add(accumulator, cursor.load_full());
          cursor.advance_full();
          offset += lanes;
        }
      }
    } else {
      for (nint_t offset = 0; offset + lanes <= kElements;
           offset += lanes) {
        if constexpr (Prefetch) {
          if (offset + 4 * lanes < kElements)
            vec::prefetch(
                tag, storage.data() +
                         (offset + 4 * lanes) * LogicalStride);
        }
        if constexpr (Strided) {
          accumulator = vec::add(
              accumulator,
              access.load(
                  tag, coord(offset * LogicalStride), axis<0>,
                  vec::opt::unmasked, vec::strided(LogicalStride)));
        } else {
          accumulator = vec::add(
              accumulator,
              access.load(
                  tag, coord(offset), axis<0>, vec::opt::unmasked));
        }
      }
    }
    benchmark::DoNotOptimize(vec::reduce_add(tag, accumulator));
  }
  state.SetItemsProcessed(state.iterations() * kElements);
}

template <bool Convert, bool Strided, bool Prefetch>
void raw_load(benchmark::State& state) {
  using Memory = std::conditional_t<Convert, float64_t, float32_t>;
  constexpr nint_t LogicalStride = Strided ? 2 : 1;
  std::vector<Memory> storage(
      static_cast<std::size_t>(kElements * LogicalStride), Memory{1});
  Tag tag{};
  const nint_t lanes = vec::size(tag);
  for (auto _ : state) {
    auto accumulator = vec::zeros(tag);
    const Memory* pointer = storage.data();
    nint_t remaining = kElements;
    while (remaining >= lanes) {
      if constexpr (Prefetch) {
        if (remaining > 4 * lanes) {
          vec::prefetch(tag, pointer + 4 * lanes * LogicalStride);
        }
      }
      if constexpr (Strided) {
        accumulator = vec::add(
            accumulator,
            vec::load_convert(
                tag, pointer, vec::opt::unmasked,
                vec::strided(LogicalStride)));
      } else {
        accumulator = vec::add(
            accumulator,
            vec::load_convert(tag, pointer, vec::opt::unmasked));
      }
      pointer += lanes * LogicalStride;
      remaining -= lanes;
    }
    benchmark::DoNotOptimize(vec::reduce_add(tag, accumulator));
  }
  state.SetItemsProcessed(state.iterations() * kElements);
}

template <bool DataAccess>
void indexed_load(benchmark::State& state) {
  using IndexTag = vec::Rebind<int32_t, Tag>;
  std::vector<float32_t> storage(static_cast<std::size_t>(kElements), 1.0f);
  Tag tag{};
  IndexTag index_tag{};
  const nint_t lanes = vec::size(tag);
  auto indices = vec::zeros(index_tag);
  for (nint_t lane = 0; lane < lanes; ++lane) {
    indices = vec::set(
        index_tag, indices, lane,
        static_cast<int32_t>(lanes - lane - 1));
  }
  auto spec = input<float32_t>(make_tensor<1>(storage.data(), {kElements}));
  kernel::Workspace workspace_storage(0);
  auto workspace = workspace_storage.view();
  auto access = bind(
      spec, InputAccessPolicy<0, 1, AccessPlan::direct>{}, workspace);
  for (auto _ : state) {
    auto accumulator = vec::zeros(tag);
    for (nint_t offset = 0; offset + lanes <= kElements; offset += lanes) {
      if constexpr (DataAccess) {
        accumulator = vec::add(
            accumulator,
            access.load(
                tag, coord(offset), vec::opt::unmasked,
                vec::indexed(indices)));
      } else {
        accumulator = vec::add(
            accumulator,
            vec::load_convert(
                tag, storage.data() + offset, vec::opt::unmasked,
                vec::indexed(indices)));
      }
    }
    benchmark::DoNotOptimize(vec::reduce_add(tag, accumulator));
  }
  state.SetItemsProcessed(state.iterations() * kElements);
}

template <bool DataAccess>
void converting_store(benchmark::State& state) {
  std::vector<float64_t> storage(static_cast<std::size_t>(kElements));
  auto spec = output<float32_t>(
      make_tensor<1>(storage.data(), {kElements}));
  kernel::Workspace workspace_storage(0);
  auto workspace = workspace_storage.view();
  auto access = bind(
      spec, OutputAccessPolicy<0, AccessPlan::direct>{}, workspace);
  Tag tag{};
  const nint_t lanes = vec::size(tag);
  const auto value = vec::fill(tag, 3.25f);
  for (auto _ : state) {
    for (nint_t offset = 0; offset + lanes <= kElements; offset += lanes) {
      if constexpr (DataAccess) {
        access.store(
            tag, coord(offset), value, vec::opt::unmasked);
      } else {
        vec::store_convert(
            tag, storage.data() + offset, value, vec::opt::unmasked);
      }
    }
    benchmark::ClobberMemory();
  }
  access.commit();
  state.SetItemsProcessed(state.iterations() * kElements);
}

BENCHMARK_TEMPLATE(raw_load, false, false, false);
BENCHMARK_TEMPLATE(data_access_load, false, false, false, false);
BENCHMARK_TEMPLATE(data_access_load, true, false, false, false);
BENCHMARK_TEMPLATE(raw_load, true, false, false);
BENCHMARK_TEMPLATE(data_access_load, false, true, false, false);
BENCHMARK_TEMPLATE(data_access_load, true, true, false, false);
BENCHMARK_TEMPLATE(raw_load, false, true, false);
BENCHMARK_TEMPLATE(data_access_load, false, false, true, false);
BENCHMARK_TEMPLATE(data_access_load, true, false, true, false);
BENCHMARK_TEMPLATE(raw_load, false, false, true);
BENCHMARK_TEMPLATE(data_access_load, true, false, false, true);
BENCHMARK_TEMPLATE(indexed_load, false);
BENCHMARK_TEMPLATE(indexed_load, true);
BENCHMARK_TEMPLATE(converting_store, false);
BENCHMARK_TEMPLATE(converting_store, true);

} // namespace

BENCHMARK_MAIN();
