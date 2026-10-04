// FP32 reference: x86 public Matmul has AMX atoms, not an FP32 atom. Use
// a named-register 4-row SIMD leaf with the same sequential K recurrence.
#include "vecops/Kernel.h"
#include "vecops/vec/Vec.h"

using Tag = vecops::vec::ScalableTag<float>;

VECOPS_ALWAYS_INLINE auto input_value(auto value) {
  if constexpr (vecops::spec::InputScale == 2)
    return value * 2 + 1;
  else
    return value;
}

VECOPS_ALWAYS_INLINE auto output_value(vecops::vec::Vec<Tag> value) {
  if constexpr (vecops::spec::Epilogue == 1)
    return vecops::vec::max(Tag{}, value, vecops::vec::zeros(Tag{}));
  else if constexpr (vecops::spec::Epilogue == 2)
    return vecops::vec::min(Tag{}, vecops::vec::max(Tag{}, value, vecops::vec::fill(Tag{}, -0.25f)),
                            vecops::vec::fill(Tag{}, 0.25f));
  else if constexpr (vecops::spec::Epilogue == 3)
    return vecops::vec::mul(Tag{}, value, vecops::vec::fill(Tag{}, 0.5f));
  else if constexpr (vecops::spec::Epilogue == 4)
    return vecops::vec::div(
      Tag{}, value,
      vecops::vec::add(Tag{}, vecops::vec::fill(Tag{}, 1.0f), vecops::vec::exp(Tag{}, vecops::vec::neg(Tag{}, value))));
  else
    return value;
}

VECOPS_ALWAYS_INLINE auto load_b(auto B, vecops::nint_t inner, vecops::nint_t col, vecops::nint_t active) {
  auto pointer = B.data() + inner * B.stride(0) + col * B.stride(1);
  if constexpr (vecops::spec::RowB)
    return vecops::vec::load(Tag{}, pointer, vecops::vec::opt::first(active));
  else
    return vecops::vec::load(Tag{}, pointer, vecops::vec::opt::strided(vecops::tensor::stride<1>(B)),
                             vecops::vec::opt::first(active));
}

void __kernel__(vecops::tensor::TensorLike auto A, vecops::tensor::TensorLike auto B, vecops::tensor::TensorLike auto C,
                vecops::tensor::TensorLike auto D, std::int64_t repeat) {
  const auto m = A.size(0), n = B.size(1), k = A.size(1), width = vecops::vec::size(Tag{});
  for (std::int64_t iteration = 0; iteration < repeat; ++iteration) {
    for (std::int64_t row = 0; row < m; row += 4) {
      for (std::int64_t col = 0; col < n; col += width) {
        auto active = std::min<vecops::nint_t>(width, n - col);
        auto c0 = vecops::vec::zeros(Tag{}), c1 = c0, c2 = c0, c3 = c0;
        if constexpr (vecops::spec::Prologue != 0) {
          auto prior_row = vecops::spec::Prologue == 2 ? 0 : row;
          c0 = vecops::vec::load(Tag{}, D.data() + prior_row * D.stride(0) + col, vecops::vec::opt::first(active));
          if (row + 1 < m)
            c1 = vecops::vec::load(Tag{},
                                   D.data() + (prior_row + (vecops::spec::Prologue == 2 ? 0 : 1)) * D.stride(0) + col,
                                   vecops::vec::opt::first(active));
          if (row + 2 < m)
            c2 = vecops::vec::load(Tag{},
                                   D.data() + (prior_row + (vecops::spec::Prologue == 2 ? 0 : 2)) * D.stride(0) + col,
                                   vecops::vec::opt::first(active));
          if (row + 3 < m)
            c3 = vecops::vec::load(Tag{},
                                   D.data() + (prior_row + (vecops::spec::Prologue == 2 ? 0 : 3)) * D.stride(0) + col,
                                   vecops::vec::opt::first(active));
        }
        for (std::int64_t inner = 0; inner < k; ++inner) {
          auto b = load_b(B, inner, col, active);
          if constexpr (vecops::spec::InputScale == 2)
            b = vecops::vec::add(Tag{}, vecops::vec::mul(Tag{}, b, vecops::vec::fill(Tag{}, 2.0f)),
                                 vecops::vec::fill(Tag{}, 1.0f));
          auto a0 = vecops::vec::fill(Tag{}, input_value(A.data()[row * A.stride(0) + inner * A.stride(1)]));
          c0 = vecops::vec::fmadd(Tag{}, a0, b, c0);
          if (row + 1 < m)
            c1 = vecops::vec::fmadd(
              Tag{}, vecops::vec::fill(Tag{}, input_value(A.data()[(row + 1) * A.stride(0) + inner * A.stride(1)])), b,
              c1);
          if (row + 2 < m)
            c2 = vecops::vec::fmadd(
              Tag{}, vecops::vec::fill(Tag{}, input_value(A.data()[(row + 2) * A.stride(0) + inner * A.stride(1)])), b,
              c2);
          if (row + 3 < m)
            c3 = vecops::vec::fmadd(
              Tag{}, vecops::vec::fill(Tag{}, input_value(A.data()[(row + 3) * A.stride(0) + inner * A.stride(1)])), b,
              c3);
        }
        vecops::vec::store(Tag{}, C.data() + row * C.stride(0) + col, output_value(c0),
                           vecops::vec::opt::first(active));
        if (row + 1 < m)
          vecops::vec::store(Tag{}, C.data() + (row + 1) * C.stride(0) + col, output_value(c1),
                             vecops::vec::opt::first(active));
        if (row + 2 < m)
          vecops::vec::store(Tag{}, C.data() + (row + 2) * C.stride(0) + col, output_value(c2),
                             vecops::vec::opt::first(active));
        if (row + 3 < m)
          vecops::vec::store(Tag{}, C.data() + (row + 3) * C.stride(0) + col, output_value(c3),
                             vecops::vec::opt::first(active));
      }
    }
    asm volatile("" ::: "memory");
  }
}
