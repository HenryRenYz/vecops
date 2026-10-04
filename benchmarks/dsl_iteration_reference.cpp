// Same algorithm as the DSL, using native Vec locals and an inlined helper.
#include "vecops/Kernel.h"
#include "vecops/vec/Vec.h"
#include <algorithm>

using Tag = vecops::vec::ScalableTag<float>;

VECOPS_ALWAYS_INLINE auto affine(vecops::vec::Vec<Tag> x) {
  return vecops::vec::add(Tag{}, vecops::vec::mul(Tag{}, x, vecops::vec::fill(Tag{}, 2.0f)),
                          vecops::vec::fill(Tag{}, 1.0f));
}

void __kernel__(vecops::tensor::TensorLike auto A, vecops::tensor::TensorLike auto B, vecops::tensor::TensorLike auto C,
                vecops::tensor::TensorLike auto R, vecops::nint_t repeat) {
  constexpr auto N = vecops::spec::N;
  const auto width = vecops::vec::size(Tag{});
  for (vecops::nint_t iteration = 0; iteration < repeat; ++iteration) {
    if constexpr (vecops::spec::Kind == 0) {
      for (vecops::nint_t i = 0; i < N; i += width) {
        auto active = std::min<vecops::nint_t>(width, N - i);
        auto x = affine(vecops::vec::load(Tag{}, A.data() + i, vecops::vec::opt::first(active)));
        vecops::vec::store(Tag{}, B.data() + i, x, vecops::vec::opt::first(active));
        auto y = vecops::vec::sub(Tag{}, vecops::vec::mul(Tag{}, x, vecops::vec::fill(Tag{}, 3.0f)),
                                  vecops::vec::fill(Tag{}, 1.0f));
        vecops::vec::store(Tag{}, C.data() + i, y, vecops::vec::opt::first(active));
      }
    } else if constexpr (vecops::spec::Kind == 1) {
      auto s0 = vecops::vec::zeros(Tag{}), s1 = s0, q0 = s0, q1 = s0;
      vecops::nint_t i = 0;
      for (; N - i >= 2 * width; i += 2 * width) {
        auto x = vecops::vec::load(Tag{}, A.data() + i);
        auto y = vecops::vec::load(Tag{}, A.data() + i + width);
        s0 = vecops::vec::add(Tag{}, s0, x);
        s1 = vecops::vec::add(Tag{}, s1, y);
        q0 = vecops::vec::add(Tag{}, q0, vecops::vec::mul(Tag{}, x, x));
        q1 = vecops::vec::add(Tag{}, q1, vecops::vec::mul(Tag{}, y, y));
      }
      for (; i < N; i += width) {
        auto x =
          vecops::vec::load(Tag{}, A.data() + i, vecops::vec::opt::first(std::min<vecops::nint_t>(width, N - i)));
        s0 = vecops::vec::add(Tag{}, s0, x);
        q0 = vecops::vec::add(Tag{}, q0, vecops::vec::mul(Tag{}, x, x));
      }
      R.data()[0] = vecops::vec::reduce_add(Tag{}, vecops::vec::add(Tag{}, s0, s1));
      R.data()[1] = vecops::vec::reduce_add(Tag{}, vecops::vec::add(Tag{}, q0, q1));
    } else {
      float sum = 0;
      for (vecops::nint_t i = 0; i < N; ++i) {
        sum += A.data()[i] * 2;
        C.data()[i] = sum * 3 + 1;
      }
    }
    asm volatile("" ::: "memory");
  }
}
