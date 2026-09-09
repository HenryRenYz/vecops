#include <array>
#include <cstdint>

#include <gtest/gtest.h>

#include "vecops/kernel/Workspace.h"

namespace {

using namespace vecops;

struct alignas(128) OverAligned {
  std::uint64_t value;
};

TEST(WorkspaceTest, TypedAllocationUsesDefaultVectorAlignment) {
  kernel::Workspace storage(256);
  auto workspace = storage.view();

  int* values = workspace.allocate<int>(4);
  EXPECT_EQ(reinterpret_cast<std::uintptr_t>(values) % vec::DEFAULT_ALIGNMENT,
            0);

  OverAligned* over_aligned = workspace.allocate<OverAligned>(1);
  EXPECT_EQ(reinterpret_cast<std::uintptr_t>(over_aligned) % alignof(OverAligned),
            0);
}

TEST(WorkspaceTest, CanAllocateAccountsForCurrentAddressAlignment) {
  alignas(64) std::array<std::byte, 128> storage{};
  kernel::WorkspaceView workspace(storage.data() + 1, 127);
  EXPECT_TRUE(workspace.can_allocate(63, 64));
  EXPECT_TRUE(workspace.can_allocate(64, 64));
  EXPECT_FALSE(workspace.can_allocate(65, 64));
  EXPECT_NE(workspace.allocate(64, 64), nullptr);
  EXPECT_FALSE(workspace.can_allocate(1, 64));
}

TEST(WorkspaceTest, AllocatesContiguousTensorAndPreservesShapeMetadata) {
  kernel::Workspace storage(256);
  auto workspace = storage.view();
  const auto shape = tensor::make_shape(meta::cint<2>, meta::Any{3});

  auto values = workspace.allocate_tensor<float>(shape);

  static_assert(decltype(values)::Ndim == 2);
  static_assert(std::same_as<
      tensor::size_type_t<0, typename decltype(values)::Layout>,
      meta::Const<2>>);
  EXPECT_EQ(tensor::size<0>(values), 2);
  EXPECT_EQ(tensor::size<1>(values), 3);
  EXPECT_EQ(tensor::stride<0>(values), 3);
  EXPECT_EQ(tensor::stride<1>(values), 1);
  EXPECT_TRUE(values.is_contiguous());
  EXPECT_EQ(reinterpret_cast<std::uintptr_t>(values.data()) %
                vec::DEFAULT_ALIGNMENT,
            0);
  EXPECT_LE(workspace.used(), kernel::WorkspaceView::tensor_bytes<float>(shape));
}

} // namespace
