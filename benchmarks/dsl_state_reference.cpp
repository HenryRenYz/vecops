// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
// Reference recurrence with explicit vector locals and no capture boundary.
#include "vecops/Kernel.h"
#include "vecops/vec/Vec.h"
#include <algorithm>
#include <bit>

template <class Tag>
VECOPS_ALWAYS_INLINE void state_step(vecops::vec::Vec<Tag>& x) {
  const auto old = x;
  x = vecops::vec::add(Tag{}, x, vecops::vec::fill(Tag{}, 1.0f));
  x = vecops::vec::add(Tag{}, x, vecops::vec::mul(Tag{}, old, vecops::vec::fill(Tag{}, 0.25f)));
}

// Build standalone locals recursively, then carry every piece through the
// same loop. Keeping all chains live matches the DSL schedule: comparing
// against one complete recurrence loop per piece would handicap C++.
template <vecops::nint_t Offset, vecops::nint_t End, class... Tags>
VECOPS_ALWAYS_INLINE void state_reference(const float* a, float* c, vecops::nint_t count,
                                          vecops::vec::Vec<Tags>&... values) {
  if constexpr (Offset < End) {
    constexpr auto native = vecops::vec::size(vecops::vec::ScalableTag<float>{});
    constexpr auto lanes =
      std::bit_floor(static_cast<std::make_unsigned_t<vecops::nint_t>>(std::min(End - Offset, native)));
    using Tag = vecops::vec::FixedTag<float, lanes>;
    auto x = vecops::vec::load(Tag{}, a + Offset);
    state_reference<Offset + lanes, End, Tags..., Tag>(a, c, count, values..., x);
    vecops::vec::store(Tag{}, c + Offset, x);
  } else {
    for (vecops::nint_t i = 0; i < count; ++i) {
      (state_step<Tags>(values), ...);
    }
  }
}

void __kernel__(vecops::tensor::TensorLike auto A, vecops::tensor::TensorLike auto C, vecops::nint_t count,
                vecops::nint_t repeat) {
  for (vecops::nint_t iteration = 0; iteration < repeat; ++iteration) {
    state_reference<0, vecops::spec::Width>(A.data(), C.data(), count);
    asm volatile("" ::: "memory");
  }
}
