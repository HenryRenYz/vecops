// @vecops-target-shards: 6
//
// Copyright (c) vecops contributors.
//
// Isolated end-to-end experiment for consuming the existing SME PackedAB ABI
// with ordinary-SVE BFMMLA/I8MM instructions.  This is deliberately kept out
// of the production SME backend and does not define a second public pack ABI.

#include "MatmulBenchCommon.h"

#include <arm_sve.h>

#include <array>
#include <string>
#include <type_traits>
#include <utility>

#include "vecops/gemm/details/sme/Atoms.h"

#if !defined(__ARM_FEATURE_SVE_BF16)
#error "MatmulMMLAExperimentBench requires SVE BF16 matrix multiply"
#endif

#if !defined(__ARM_FEATURE_SVE_MATMUL_INT8)
#error "MatmulMMLAExperimentBench requires SVE I8MM"
#endif

namespace vecops::bench::matmul::mmla_experiment {
template <int Shard>
void register_mmla_experiment_shard();
}

#if defined(VECOPS_TARGET_SHARD_ACTIVE)

namespace vecops::bench::matmul::mmla_experiment {

using S8S8 = gemm::SME_I8I32<int8_t, int8_t>;
using U8U8 = gemm::SME_I8I32<uint8_t, uint8_t>;

template <typename Atom>
struct Traits;

template <>
struct Traits<gemm::SME_BF16F32> {
  using Element = bfloat16_t;
  using Acc = float32_t;
  using InputVec = svbfloat16_t;
  using AccVec = svfloat32_t;

  static VECOPS_ALWAYS_INLINE InputVec load(const Element* pointer) {
    const auto bits = svld1_u16(
        svptrue_b16(), reinterpret_cast<const uint16_t*>(pointer));
    return svreinterpret_bf16_u16(bits);
  }

  static VECOPS_ALWAYS_INLINE InputVec zero_input() {
    return svreinterpret_bf16_u16(svdup_u16(0));
  }

  static VECOPS_ALWAYS_INLINE InputVec zip_groups(
      InputVec group0, InputVec group1) {
    return svreinterpret_bf16_u32(svzip1_u32(
        svreinterpret_u32_bf16(group0),
        svreinterpret_u32_bf16(group1)));
  }

  template <int Segment>
  static VECOPS_ALWAYS_INLINE InputVec broadcast_segment(InputVec value) {
    return svdupq_lane_bf16(value, Segment);
  }

  static VECOPS_ALWAYS_INLINE AccVec zero_acc() { return svdup_f32(0); }

  static VECOPS_ALWAYS_INLINE AccVec mmla(
      AccVec acc, InputVec a, InputVec b) {
    return svbfmmla_f32(acc, a, b);
  }

  static VECOPS_ALWAYS_INLINE void store_contiguous(
      AccVec value, Acc* output, nint_t count) {
    svst1_f32(
        svwhilelt_b32(uint64_t{0}, static_cast<uint64_t>(count)),
        output, value);
  }

  static VECOPS_ALWAYS_INLINE void store_row_pair(
      AccVec value, Acc* row0, Acc* row1,
      nint_t columns, bool has_row1) {
    const auto bits = svreinterpret_u64_f32(value);
    const auto first = svreinterpret_f32_u64(svuzp1_u64(bits, bits));
    const auto second = svreinterpret_f32_u64(svuzp2_u64(bits, bits));
    const auto pg = svwhilelt_b32(
        uint64_t{0}, static_cast<uint64_t>(columns));
    svst1_f32(pg, row0, first);
    if (has_row1) svst1_f32(pg, row1, second);
  }
};

template <>
struct Traits<S8S8> {
  using Element = int8_t;
  using Acc = int32_t;
  using InputVec = svint8_t;
  using AccVec = svint32_t;

  static VECOPS_ALWAYS_INLINE InputVec load(const Element* pointer) {
    return svld1_s8(svptrue_b8(), pointer);
  }

  static VECOPS_ALWAYS_INLINE InputVec zero_input() { return svdup_s8(0); }

  static VECOPS_ALWAYS_INLINE InputVec zip_groups(
      InputVec group0, InputVec group1) {
    return svreinterpret_s8_u32(svzip1_u32(
        svreinterpret_u32_s8(group0), svreinterpret_u32_s8(group1)));
  }

  template <int Segment>
  static VECOPS_ALWAYS_INLINE InputVec broadcast_segment(InputVec value) {
    return svdupq_lane_s8(value, Segment);
  }

  static VECOPS_ALWAYS_INLINE AccVec zero_acc() { return svdup_s32(0); }

  static VECOPS_ALWAYS_INLINE AccVec mmla(
      AccVec acc, InputVec a, InputVec b) {
    return svmmla_s32(acc, a, b);
  }

  static VECOPS_ALWAYS_INLINE void store_contiguous(
      AccVec value, Acc* output, nint_t count) {
    svst1_s32(
        svwhilelt_b32(uint64_t{0}, static_cast<uint64_t>(count)),
        output, value);
  }

  static VECOPS_ALWAYS_INLINE void store_row_pair(
      AccVec value, Acc* row0, Acc* row1,
      nint_t columns, bool has_row1) {
    const auto bits = svreinterpret_u64_s32(value);
    const auto first = svreinterpret_s32_u64(svuzp1_u64(bits, bits));
    const auto second = svreinterpret_s32_u64(svuzp2_u64(bits, bits));
    const auto pg = svwhilelt_b32(
        uint64_t{0}, static_cast<uint64_t>(columns));
    svst1_s32(pg, row0, first);
    if (has_row1) svst1_s32(pg, row1, second);
  }
};

template <>
struct Traits<U8U8> {
  using Element = uint8_t;
  using Acc = int32_t;
  using InputVec = svuint8_t;
  using AccVec = svuint32_t;

  static VECOPS_ALWAYS_INLINE InputVec load(const Element* pointer) {
    return svld1_u8(svptrue_b8(), pointer);
  }

  static VECOPS_ALWAYS_INLINE InputVec zero_input() { return svdup_u8(0); }

  static VECOPS_ALWAYS_INLINE InputVec zip_groups(
      InputVec group0, InputVec group1) {
    return svreinterpret_u8_u32(svzip1_u32(
        svreinterpret_u32_u8(group0), svreinterpret_u32_u8(group1)));
  }

  template <int Segment>
  static VECOPS_ALWAYS_INLINE InputVec broadcast_segment(InputVec value) {
    return svdupq_lane_u8(value, Segment);
  }

  static VECOPS_ALWAYS_INLINE AccVec zero_acc() { return svdup_u32(0); }

  static VECOPS_ALWAYS_INLINE AccVec mmla(
      AccVec acc, InputVec a, InputVec b) {
    return svmmla_u32(acc, a, b);
  }

  static VECOPS_ALWAYS_INLINE void store_contiguous(
      AccVec value, Acc* output, nint_t count) {
    svst1_s32(
        svwhilelt_b32(uint64_t{0}, static_cast<uint64_t>(count)),
        output, svreinterpret_s32_u32(value));
  }

  static VECOPS_ALWAYS_INLINE void store_row_pair(
      AccVec value, Acc* row0, Acc* row1,
      nint_t columns, bool has_row1) {
    const auto bits = svreinterpret_u64_u32(value);
    const auto first = svreinterpret_s32_u64(svuzp1_u64(bits, bits));
    const auto second = svreinterpret_s32_u64(svuzp2_u64(bits, bits));
    const auto pg = svwhilelt_b32(
        uint64_t{0}, static_cast<uint64_t>(columns));
    svst1_s32(pg, row0, first);
    if (has_row1) svst1_s32(pg, row1, second);
  }
};

// TODO: If the experiment wins end-to-end, move only the stabilized operation
// shape into a backend-local helper.  Keep widening/MMLA semantics out of the
// public vec API until mixed-sign and variable-VL behavior are settled.
template <typename Atom>
VECOPS_NOINLINE VECOPS_FUNCTION_ALIGN(64) void packed_ab_mmla(
    const typename Atom::TA* packed_a,
    const typename Atom::TB* packed_b,
    typename Atom::TAcc* output,
    nint_t m, nint_t n, nint_t logical_k, nint_t panel) {
  using Op = Traits<Atom>;
  constexpr nint_t KPack =
      gemm::packing_t<Atom, gemm::Operand::A>::KPack;
  const nint_t groups = ceil_div(logical_k, KPack);
  const nint_t group_stride = panel * KPack;

  // The vertical form makes one accumulator naturally row-major for 8x2.
  if (n == 2 && m > 2) {
    auto acc = Op::zero_acc();
    VECOPS_LOOP_ALIGN(64) for (nint_t group = 0; group < groups; group += 2) {
      const auto a0 = Op::load(packed_a + group * group_stride);
      const auto b0 = Op::load(packed_b + group * group_stride);
      const auto a1 = group + 1 < groups
          ? Op::load(packed_a + (group + 1) * group_stride)
          : Op::zero_input();
      const auto b1 = group + 1 < groups
          ? Op::load(packed_b + (group + 1) * group_stride)
          : Op::zero_input();
      const auto a = Op::zip_groups(a0, a1);
      const auto b = Op::zip_groups(b0, b1);
      acc = Op::mmla(acc, a, Op::template broadcast_segment<0>(b));
    }
    Op::store_contiguous(acc, output, m * n);
    return;
  }

  auto acc0 = Op::zero_acc();
  auto acc1 = Op::zero_acc();
  VECOPS_LOOP_ALIGN(64) for (nint_t group = 0; group < groups; group += 2) {
    const auto a0 = Op::load(packed_a + group * group_stride);
    const auto b0 = Op::load(packed_b + group * group_stride);
    const auto a1 = group + 1 < groups
        ? Op::load(packed_a + (group + 1) * group_stride)
        : Op::zero_input();
    const auto b1 = group + 1 < groups
        ? Op::load(packed_b + (group + 1) * group_stride)
        : Op::zero_input();
    const auto a = Op::zip_groups(a0, a1);
    const auto b = Op::zip_groups(b0, b1);
    acc0 = Op::mmla(acc0, Op::template broadcast_segment<0>(a), b);
    if (m > 2) {
      acc1 = Op::mmla(acc1, Op::template broadcast_segment<1>(a), b);
    }
  }
  Op::store_row_pair(
      acc0, output, output + n, n, m > 1);
  if (m > 2) {
    Op::store_row_pair(
        acc1, output + 2 * n, output + 3 * n, n, m > 3);
  }
}

template <typename Atom>
inline const char* type_name();

template <>
inline const char* type_name<gemm::SME_BF16F32>() { return "bf16"; }

template <>
inline const char* type_name<S8S8>() { return "s8s8"; }

template <>
inline const char* type_name<U8U8>() { return "u8u8"; }

template <typename Atom, typename MExtent, typename NExtent, typename KExtent>
void run_mmla_with_extents(
    benchmark::State& state, const MatmulCase& test_case,
    MExtent m, NExtent n, KExtent k) {
  using TA = typename Atom::TA;
  using TB = typename Atom::TB;
  using Acc = typename Atom::TAcc;

  std::vector<TA> a(
      static_cast<std::size_t>(test_case.m * test_case.k));
  std::vector<TB> b(
      static_cast<std::size_t>(test_case.n * test_case.k));
  std::vector<Acc> c(
      static_cast<std::size_t>(test_case.m * test_case.n), Acc{});
  fill_input(a, 3);
  fill_input(b, 11);

  const auto a_layout = make_layout(make_shape(m, k));
  const auto b_layout = make_layout(make_shape(n, k));
  const auto packed_a_layout =
      ops::matmul_packed_layout<Atom, gemm::Operand::A>(a_layout);
  const auto packed_b_layout =
      ops::matmul_packed_layout<Atom, gemm::Operand::B>(b_layout);
  const nint_t packed_a_elements = numel(packed_a_layout);
  const nint_t packed_b_elements = numel(packed_b_layout);
  const nint_t packed_a_bytes = packed_a_elements * nint_t{sizeof(TA)};
  const nint_t packed_b_bytes = packed_b_elements * nint_t{sizeof(TB)};
  kernel::Workspace packed_a_storage(packed_a_bytes + 64);
  kernel::Workspace packed_b_storage(packed_b_bytes + 64);
  auto packed_a_workspace = packed_a_storage.view();
  auto packed_b_workspace = packed_b_storage.view();
  auto* packed_a = static_cast<TA*>(
      packed_a_workspace.allocate(packed_a_bytes, 64));
  auto* packed_b = static_cast<TB*>(
      packed_b_workspace.allocate(packed_b_bytes, 64));
  auto a_tensor = make_tensor(a.data(), a_layout);
  auto b_tensor = make_tensor(b.data(), b_layout);
  auto packed_a_tensor = make_tensor(packed_a, packed_a_layout);
  auto packed_b_tensor = make_tensor(packed_b, packed_b_layout);
  ExecutionSession execution{};
  ops::matmul_pack<Atom, gemm::Operand::A>(
      execution, a_tensor, packed_a_tensor);
  ops::matmul_pack<Atom, gemm::Operand::B>(
      execution, b_tensor, packed_b_tensor);

  const nint_t panel = packed_a_layout.shape()[2];
  if (static_cast<nint_t>(svcntb()) != 2 * panel) {
    state.SkipWithError(
        "experiment currently requires ordinary SVE VL equal to SME SVL");
    return;
  }

  packed_ab_mmla<Atom>(
      packed_a, packed_b, c.data(),
      test_case.m, test_case.n, test_case.k, panel);
  if (!verify_samples<Atom>(
          a, b, c, test_case.m, test_case.n, test_case.k)) {
    state.SkipWithError("MMLA experiment verification failed");
    return;
  }

  for (auto _ : state) {
    benchmark::DoNotOptimize(packed_a);
    benchmark::DoNotOptimize(packed_b);
    packed_ab_mmla<Atom>(
        packed_a, packed_b, c.data(),
        test_case.m, test_case.n, test_case.k, panel);
    benchmark::DoNotOptimize(c.data());
    benchmark::ClobberMemory();
  }

  const int64_t mnk = static_cast<int64_t>(
      test_case.m * test_case.n * test_case.k);
  state.SetItemsProcessed(state.iterations() * mnk);
  state.counters["FLOP/s"] = benchmark::Counter(
      static_cast<double>(2 * mnk),
      benchmark::Counter::kIsIterationInvariantRate);
  state.counters["m"] = benchmark::Counter(double(test_case.m));
  state.counters["n"] = benchmark::Counter(double(test_case.n));
  state.counters["k"] = benchmark::Counter(double(test_case.k));
  state.counters["packed_elements"] = benchmark::Counter(
      double(packed_a_elements + packed_b_elements));
}

template <typename Atom, ExtentMode Extents>
std::string name(const char* engine, const MatmulCase& test_case) {
  return "MatmulMMLAExperiment/engine:" + std::string(engine) +
      "/shape:" + std::to_string(test_case.m) + "x" +
      std::to_string(test_case.n) + "x" + std::to_string(test_case.k) +
      "/dtype:" + type_name<Atom>() + "/input:packed_ab/extent:" +
      extent_mode_name<Extents>() + "/arch:" + VECOPS_BENCH_ARCH_CODE;
}

template <typename Atom, ExtentMode Extents,
          nint_t M, nint_t N, nint_t K>
void register_experiment_case() {
  const MatmulCase test_case{
      "mmla_experiment", "packed_ab", M, N, K,
      mode_bit(InputMode::PackedAB)};
  auto* mopa = benchmark::RegisterBenchmark(
      name<Atom, Extents>("mopa", test_case).c_str(),
      [test_case](benchmark::State& state) {
        if constexpr (Extents == ExtentMode::Dynamic) {
          run_case<Atom, InputMode::PackedAB>(state, test_case);
        } else {
          run_fixed_case<Atom, InputMode::PackedAB, M, N, K>(
              state, test_case);
        }
      })->Unit(benchmark::kNanosecond);
  ::vecops::bench::configure_registered_benchmark(
      mopa, 0.02, 5)->ReportAggregatesOnly(true);

  auto* mmla = benchmark::RegisterBenchmark(
      name<Atom, Extents>("mmla_current_pack", test_case).c_str(),
      [test_case](benchmark::State& state) {
        if constexpr (Extents == ExtentMode::Dynamic) {
          run_mmla_with_extents<Atom>(
              state, test_case, Any{M}, Any{N}, Any{K});
        } else {
          run_mmla_with_extents<Atom>(
              state, test_case, cint<M>, cint<N>, cint<K>);
        }
      })->Unit(benchmark::kNanosecond);
  ::vecops::bench::configure_registered_benchmark(
      mmla, 0.02, 5)->ReportAggregatesOnly(true);
}

inline constexpr std::array<std::array<nint_t, 2>, 6> ExperimentShapes{{
      {{2, 2}}, {{2, 4}}, {{4, 2}},
      {{2, 8}}, {{8, 2}}, {{4, 4}},
}};
inline constexpr std::array<nint_t, 8> ExperimentKValues{{
      64, 65, 256, 257, 512, 513, 1024, 1025,
}};

template <typename Atom, ExtentMode Extents, std::size_t Index>
void register_experiment_index() {
  constexpr auto Shape = ExperimentShapes[Index / ExperimentKValues.size()];
  constexpr nint_t K = ExperimentKValues[Index % ExperimentKValues.size()];
  register_experiment_case<Atom, Extents, Shape[0], Shape[1], K>();
}

template <typename Atom, ExtentMode Extents, std::size_t... I>
void register_type(std::index_sequence<I...>) {
  (register_experiment_index<Atom, Extents, I>(), ...);
}

template <int Shard>
void register_mmla_experiment_shard() {
  static_assert(0 <= Shard && Shard < 6);
  constexpr ExtentMode Extents = (Shard % 2) == 0
      ? ExtentMode::Dynamic
      : ExtentMode::Const;
  constexpr std::size_t CaseCount =
      ExperimentShapes.size() * ExperimentKValues.size();
  if constexpr (Shard / 2 == 0) {
    register_type<gemm::SME_BF16F32, Extents>(
        std::make_index_sequence<CaseCount>{});
  } else if constexpr (Shard / 2 == 1) {
    register_type<S8S8, Extents>(std::make_index_sequence<CaseCount>{});
  } else {
    register_type<U8U8, Extents>(std::make_index_sequence<CaseCount>{});
  }
}

} // namespace vecops::bench::matmul::mmla_experiment

template void
vecops::bench::matmul::mmla_experiment::register_mmla_experiment_shard<
    VECOPS_TARGET_SHARD_INDEX>();

#else

int main(int argc, char** argv) {
  []<std::size_t... I>(std::index_sequence<I...>) {
    (vecops::bench::matmul::mmla_experiment::
         register_mmla_experiment_shard<static_cast<int>(I)>(), ...);
  }(std::make_index_sequence<6>{});
  return vecops::bench::matmul::run_benchmarks(
      argc, argv, "matmul_mmla_experiment");
}

#endif
