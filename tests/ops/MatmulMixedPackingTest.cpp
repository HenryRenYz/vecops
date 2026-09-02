// @vecops-target-shards: 8

#include <gtest/gtest.h>

#include <cstdint>
#include <sys/syscall.h>
#include <unistd.h>

#include "MatmulConversionTestCommon.h"

using namespace vecops;

template <typename Atom, nint_t M, nint_t N, nint_t K>
void run_mixed_packing_atom();

template <typename Atom, nint_t M, nint_t N, nint_t K>
void run_both_packed_atom();

#if defined(VECOPS_TARGET_SHARD_ACTIVE)

static_assert(VECOPS_TARGET_SHARD_COUNT == 8);

template <typename Atom, nint_t M, nint_t N, nint_t K>
void run_mixed_packing_atom() {
  test::matmul::check_mixed_packing<Atom, ::vecops::matmul::Operand::A>(M, N, K);
  test::matmul::check_mixed_packing<Atom, ::vecops::matmul::Operand::B>(M, N, K);
}

template <typename Atom, meta::ValueType KExtent>
void check_both_packed(
    nint_t m, nint_t n, KExtent k_extent) {
  using TA = typename Atom::TA;
  using TB = typename Atom::TB;
  using Acc = typename Atom::TAcc;
  const nint_t k = static_cast<nint_t>(k_extent);
  std::vector<TA> a(static_cast<std::size_t>(m * k));
  std::vector<TB> b(static_cast<std::size_t>(n * k));
  std::vector<Acc> c(static_cast<std::size_t>(m * n), Acc{});
  for (nint_t i = 0; i < m * k; ++i)
    a[static_cast<std::size_t>(i)] =
        test::matmul::conversion_value<TA>(i, 3);
  for (nint_t i = 0; i < n * k; ++i)
    b[static_cast<std::size_t>(i)] =
        test::matmul::conversion_value<TB>(i, 11);

  auto al = tensor::make_layout(
      tensor::make_shape(meta::Any{m}, k_extent));
  auto bl = tensor::make_layout(
      tensor::make_shape(meta::Any{n}, k_extent));
  auto at = tensor::make_tensor(a.data(), al);
  auto bt = tensor::make_tensor(b.data(), bl);
  auto apl = ops::matmul_packed_layout<Atom, ::vecops::matmul::Operand::A>(al);
  auto bpl = ops::matmul_packed_layout<Atom, ::vecops::matmul::Operand::B>(bl);
  static_assert(::vecops::matmul::is_packed_layout<
      Atom, ::vecops::matmul::Operand::A, decltype(apl)>());
  static_assert(::vecops::matmul::is_packed_layout<
      Atom, ::vecops::matmul::Operand::B, decltype(bpl)>());

  const nint_t a_bytes = tensor::numel(apl) *
      static_cast<nint_t>(sizeof(TA));
  const nint_t b_bytes = tensor::numel(bpl) *
      static_cast<nint_t>(sizeof(TB));
  kernel::Workspace a_storage(a_bytes + 64);
  kernel::Workspace b_storage(b_bytes + 64);
  auto a_workspace = a_storage.view();
  auto b_workspace = b_storage.view();
  auto* ap = static_cast<TA*>(a_workspace.allocate(a_bytes, 64));
  auto* bp = static_cast<TB*>(b_workspace.allocate(b_bytes, 64));
  auto apt = tensor::make_tensor(ap, apl);
  auto bpt = tensor::make_tensor(bp, bpl);
  ExecutionSession pack_execution{};
  ops::matmul_pack_details::run_matmul_pack<Atom, ::vecops::matmul::Operand::A>(pack_execution, at, apt);
  ops::matmul_pack_details::run_matmul_pack<Atom, ::vecops::matmul::Operand::B>(pack_execution, bt, bpt);

  auto ct = tensor::make_tensor(
      c.data(), tensor::make_layout(
                    tensor::make_shape(meta::Any{m}, meta::Any{n})));
  auto operation = ops::matmul_details::make_matmul_invocation(ops::MatmulConfig<Atom>{},
      meta::Any{m}, meta::Any{n}, k_extent, apt, bpt, ct);
  static_assert(std::same_as<
      typename decltype(operation)::KExtentType, KExtent>);
  kernel::Workspace operation_storage(operation.required_workspace());
  auto operation_workspace = operation_storage.view();
  operation(operation_workspace);

  for (nint_t row = 0; row < m; ++row) {
    for (nint_t col = 0; col < n; ++col) {
      Acc expected{};
      for (nint_t kk = 0; kk < k; ++kk) {
        expected += static_cast<Acc>(a[static_cast<std::size_t>(
                        row * k + kk)]) *
            static_cast<Acc>(b[static_cast<std::size_t>(col * k + kk)]);
      }
      const Acc actual = c[static_cast<std::size_t>(row * n + col)];
      EXPECT_TRUE(test::matmul::conversion_values_equal(expected, actual))
          << "both packed row=" << row << " col=" << col
          << " k=" << k;
    }
  }
}

template <typename Atom, nint_t M, nint_t N, nint_t K>
void run_both_packed_atom() {
  constexpr nint_t KR = decltype(Atom::K_R)::value;
  if constexpr (KR == 1) {
    check_both_packed<Atom>(M, N, meta::Any{K});
  } else {
    using BoundedK = meta::Dynamic<1, KR, K + KR>;
    static_assert(!BoundedK::aligns(KR));
    static_assert(BoundedK::conforms(K));
    check_both_packed<Atom>(M, N, BoundedK{K});
  }
}

#if defined(ARCH_X86_FAMILY)

#if VECOPS_TARGET_SHARD_INDEX == 0
template void run_mixed_packing_atom<::vecops::matmul::AMX_BF16F32, 19, 21, 65>();
template void run_both_packed_atom<::vecops::matmul::AMX_BF16F32, 19, 21, 65>();
#elif VECOPS_TARGET_SHARD_INDEX == 1
#if defined(HAS_AMX_FP16)
template void run_mixed_packing_atom<::vecops::matmul::AMX_F16F32, 19, 21, 65>();
template void run_both_packed_atom<::vecops::matmul::AMX_F16F32, 19, 21, 65>();
#endif
#elif VECOPS_TARGET_SHARD_INDEX == 2
#if defined(HAS_AMX_INT8)
template void run_mixed_packing_atom<
    ::vecops::matmul::AMX_I8I32<int8_t, int8_t>, 19, 21, 65>();
template void run_both_packed_atom<
    ::vecops::matmul::AMX_I8I32<int8_t, int8_t>, 19, 21, 65>();
#endif
#elif VECOPS_TARGET_SHARD_INDEX == 3
#if defined(HAS_AMX_INT8)
template void run_mixed_packing_atom<
    ::vecops::matmul::AMX_I8I32<int8_t, uint8_t>, 19, 21, 65>();
template void run_both_packed_atom<
    ::vecops::matmul::AMX_I8I32<int8_t, uint8_t>, 19, 21, 65>();
#endif
#elif VECOPS_TARGET_SHARD_INDEX == 4
#if defined(HAS_AMX_INT8)
template void run_mixed_packing_atom<
    ::vecops::matmul::AMX_I8I32<uint8_t, int8_t>, 19, 21, 65>();
template void run_both_packed_atom<
    ::vecops::matmul::AMX_I8I32<uint8_t, int8_t>, 19, 21, 65>();
#endif
#elif VECOPS_TARGET_SHARD_INDEX == 5
#if defined(HAS_AMX_INT8)
template void run_mixed_packing_atom<
    ::vecops::matmul::AMX_I8I32<uint8_t, uint8_t>, 19, 21, 65>();
template void run_both_packed_atom<
    ::vecops::matmul::AMX_I8I32<uint8_t, uint8_t>, 19, 21, 65>();
#endif
#endif

#else

#if VECOPS_TARGET_SHARD_INDEX == 0
template void run_mixed_packing_atom<::vecops::matmul::SME_BF16F32, 19, 21, 17>();
template void run_both_packed_atom<::vecops::matmul::SME_BF16F32, 19, 21, 17>();
#elif VECOPS_TARGET_SHARD_INDEX == 1
template void run_mixed_packing_atom<::vecops::matmul::SME_F16F32, 19, 21, 17>();
template void run_both_packed_atom<::vecops::matmul::SME_F16F32, 19, 21, 17>();
#elif VECOPS_TARGET_SHARD_INDEX == 2
template void run_mixed_packing_atom<::vecops::matmul::SME_F32F32, 19, 21, 17>();
template void run_both_packed_atom<::vecops::matmul::SME_F32F32, 19, 21, 17>();
#elif VECOPS_TARGET_SHARD_INDEX == 3
#if defined(HAS_SME_F64F64)
template void run_mixed_packing_atom<::vecops::matmul::SME_F64F64, 11, 13, 17>();
template void run_both_packed_atom<::vecops::matmul::SME_F64F64, 11, 13, 17>();
#endif
#elif VECOPS_TARGET_SHARD_INDEX == 4
template void run_mixed_packing_atom<
    ::vecops::matmul::SME_I8I32<int8_t, int8_t>, 19, 21, 33>();
template void run_both_packed_atom<
    ::vecops::matmul::SME_I8I32<int8_t, int8_t>, 19, 21, 33>();
#elif VECOPS_TARGET_SHARD_INDEX == 5
template void run_mixed_packing_atom<
    ::vecops::matmul::SME_I8I32<int8_t, uint8_t>, 19, 21, 33>();
template void run_both_packed_atom<
    ::vecops::matmul::SME_I8I32<int8_t, uint8_t>, 19, 21, 33>();
#elif VECOPS_TARGET_SHARD_INDEX == 6
template void run_mixed_packing_atom<
    ::vecops::matmul::SME_I8I32<uint8_t, int8_t>, 19, 21, 33>();
template void run_both_packed_atom<
    ::vecops::matmul::SME_I8I32<uint8_t, int8_t>, 19, 21, 33>();
#else
template void run_mixed_packing_atom<
    ::vecops::matmul::SME_I8I32<uint8_t, uint8_t>, 19, 21, 33>();
template void run_both_packed_atom<
    ::vecops::matmul::SME_I8I32<uint8_t, uint8_t>, 19, 21, 33>();
#endif

#endif

#else

#if defined(ARCH_X86_FAMILY)

bool enable_amx() {
  constexpr long ArchRequestXcompPerm = 0x1023;
  constexpr long XfeatureTileData = 18;
  return syscall(
      SYS_arch_prctl, ArchRequestXcompPerm, XfeatureTileData) == 0;
}

TEST(MatmulMixedPackingTest, EveryAMXAtomSupportsEitherPackedOperand) {
  ASSERT_TRUE(enable_amx());
  run_mixed_packing_atom<::vecops::matmul::AMX_BF16F32, 19, 21, 65>();
#if defined(HAS_AMX_FP16)
  run_mixed_packing_atom<::vecops::matmul::AMX_F16F32, 19, 21, 65>();
#endif
#if defined(HAS_AMX_INT8)
  run_mixed_packing_atom<
      ::vecops::matmul::AMX_I8I32<int8_t, int8_t>, 19, 21, 65>();
  run_mixed_packing_atom<
      ::vecops::matmul::AMX_I8I32<int8_t, uint8_t>, 19, 21, 65>();
  run_mixed_packing_atom<
      ::vecops::matmul::AMX_I8I32<uint8_t, int8_t>, 19, 21, 65>();
  run_mixed_packing_atom<
      ::vecops::matmul::AMX_I8I32<uint8_t, uint8_t>, 19, 21, 65>();
#endif
}

TEST(MatmulMixedPackingTest, EveryAMXAtomSupportsBothPackedOperands) {
  ASSERT_TRUE(enable_amx());
  run_both_packed_atom<::vecops::matmul::AMX_BF16F32, 19, 21, 65>();
#if defined(HAS_AMX_FP16)
  run_both_packed_atom<::vecops::matmul::AMX_F16F32, 19, 21, 65>();
#endif
#if defined(HAS_AMX_INT8)
  run_both_packed_atom<
      ::vecops::matmul::AMX_I8I32<int8_t, int8_t>, 19, 21, 65>();
  run_both_packed_atom<
      ::vecops::matmul::AMX_I8I32<int8_t, uint8_t>, 19, 21, 65>();
  run_both_packed_atom<
      ::vecops::matmul::AMX_I8I32<uint8_t, int8_t>, 19, 21, 65>();
  run_both_packed_atom<
      ::vecops::matmul::AMX_I8I32<uint8_t, uint8_t>, 19, 21, 65>();
#endif
}

#else

TEST(MatmulMixedPackingTest, EverySMEAtomSupportsEitherPackedOperand) {
  run_mixed_packing_atom<::vecops::matmul::SME_BF16F32, 19, 21, 17>();
  run_mixed_packing_atom<::vecops::matmul::SME_F16F32, 19, 21, 17>();
  run_mixed_packing_atom<::vecops::matmul::SME_F32F32, 19, 21, 17>();
#if defined(HAS_SME_F64F64)
  run_mixed_packing_atom<::vecops::matmul::SME_F64F64, 11, 13, 17>();
#endif
  run_mixed_packing_atom<
      ::vecops::matmul::SME_I8I32<int8_t, int8_t>, 19, 21, 33>();
  run_mixed_packing_atom<
      ::vecops::matmul::SME_I8I32<int8_t, uint8_t>, 19, 21, 33>();
  run_mixed_packing_atom<
      ::vecops::matmul::SME_I8I32<uint8_t, int8_t>, 19, 21, 33>();
  run_mixed_packing_atom<
      ::vecops::matmul::SME_I8I32<uint8_t, uint8_t>, 19, 21, 33>();
}

TEST(MatmulMixedPackingTest, EverySMEAtomSupportsBothPackedOperands) {
  run_both_packed_atom<::vecops::matmul::SME_BF16F32, 19, 21, 17>();
  run_both_packed_atom<::vecops::matmul::SME_F16F32, 19, 21, 17>();
  run_both_packed_atom<::vecops::matmul::SME_F32F32, 19, 21, 17>();
#if defined(HAS_SME_F64F64)
  run_both_packed_atom<::vecops::matmul::SME_F64F64, 11, 13, 17>();
#endif
  run_both_packed_atom<
      ::vecops::matmul::SME_I8I32<int8_t, int8_t>, 19, 21, 33>();
  run_both_packed_atom<
      ::vecops::matmul::SME_I8I32<int8_t, uint8_t>, 19, 21, 33>();
  run_both_packed_atom<
      ::vecops::matmul::SME_I8I32<uint8_t, int8_t>, 19, 21, 33>();
  run_both_packed_atom<
      ::vecops::matmul::SME_I8I32<uint8_t, uint8_t>, 19, 21, 33>();
}

#endif

#endif
