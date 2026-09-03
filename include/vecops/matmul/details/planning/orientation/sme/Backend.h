//
// Copyright (c) vecops contributors.
//

#ifndef VECOPS_MATMUL_DETAILS_ORIENTATION_SME_BACKEND_H
#define VECOPS_MATMUL_DETAILS_ORIENTATION_SME_BACKEND_H

namespace vecops::matmul::details::orientation {

/** Compile-time SME orientation selector for rank-two operands. */
template <>
struct Backend<::vecops::matmul::SMEKernelKind> {
  template <typename Atom, meta::ValueType M, meta::ValueType N,
            meta::ValueType K, typename ASpec, typename BSpec,
            typename CInputSpec, typename COutputSpec>
  static consteval bool swap_ab() {
    (void)sizeof(M);
    (void)sizeof(N);
    (void)sizeof(K);
    if constexpr (!swappable_rank2_input_v<ASpec> ||
                  !swappable_rank2_input_v<BSpec> ||
                  !swappable_rank2_c_input_v<CInputSpec> ||
                  !swappable_rank2_c_output_v<COutputSpec> ||
                  ::vecops::matmul::is_packed_layout<
                      Atom, ::vecops::matmul::Operand::A,
                      typename ASpec::InputLayout>() ||
                  ::vecops::matmul::is_packed_layout<
                      Atom, ::vecops::matmul::Operand::B,
                      typename BSpec::InputLayout>()) {
      return false;
    } else {
      // SME's raw loaders and packed formats are symmetric in A/B.  Until a
      // direct vertical C epilogue exists, transposing the problem converts a
      // statically column-contiguous output into the horizontal fast path.
      return column_contiguous_v<typename COutputSpec::OutputLayout> &&
          !row_contiguous_v<typename COutputSpec::OutputLayout>;
    }
  }
};

} // namespace vecops::matmul::details::orientation

#endif // VECOPS_MATMUL_DETAILS_ORIENTATION_SME_BACKEND_H
