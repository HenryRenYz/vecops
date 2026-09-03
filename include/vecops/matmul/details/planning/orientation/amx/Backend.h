//
// Copyright (c) vecops contributors.
//

#ifndef VECOPS_MATMUL_DETAILS_ORIENTATION_AMX_BACKEND_H
#define VECOPS_MATMUL_DETAILS_ORIENTATION_AMX_BACKEND_H

namespace vecops::matmul::details::orientation {

/** Compile-time AMX orientation selector for rank-two operands. */
template <>
struct Backend<::vecops::matmul::AMXKernelKind> {
  template <typename Atom, meta::ValueType M, meta::ValueType N,
            meta::ValueType K, typename ASpec, typename BSpec,
            typename CInputSpec, typename COutputSpec>
  static consteval bool swap_ab() {
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
      constexpr bool ARow = row_contiguous_v<typename ASpec::InputLayout>;
      constexpr bool AColumn =
          column_contiguous_v<typename ASpec::InputLayout>;
      constexpr bool BRow = row_contiguous_v<typename BSpec::InputLayout>;
      constexpr bool BColumn =
          column_contiguous_v<typename BSpec::InputLayout>;
      constexpr bool CColumn =
          column_contiguous_v<typename COutputSpec::OutputLayout> &&
          !row_contiguous_v<typename COutputSpec::OutputLayout>;
      constexpr bool CRow =
          row_contiguous_v<typename COutputSpec::OutputLayout>;

      // (A=T, B=R) becomes AMX's natural raw orientation (A=R, B=T).
      // With transposed C this dominates on all three operands.
      if constexpr (AColumn && BRow && CColumn) return true;

      // Swapping preserves the two input layout classes while making C
      // contiguous.  The equal-layout forms also preserve native element
      // types; mixed-sign atoms are transposed by SwappedAtom.
      if constexpr (CColumn && AColumn && BColumn) return true;

      // When both inputs are K-contiguous, online packing makes the B side
      // more expensive.  Swap only when metadata proves that it places the
      // smaller spatial extent on B while also making C contiguous.
      if constexpr (CColumn && ARow && BRow &&
                    provably_at_least_v<N, M>)
        return true;

      // For the opposite raw layout, swapping fixes both AMX input roles but
      // turns a contiguous C into a transposed store.  One complete AMX K
      // step is the measured break-even point once M is large enough and not
      // disproportionately smaller than N.  Tiny-M/wide-N shapes retain the
      // vector-friendly direct path because its store advantage dominates.
      constexpr nint_t KThreshold =
          std::remove_cvref_t<decltype(Atom::K_R)>::value;
      if constexpr (CRow && AColumn && BRow &&
                    provably_at_least_value_v<M, 4> &&
                    provably_scaled_at_least_v<M, N, 2> &&
                    provably_at_least_value_v<K, KThreshold>)
        return true;

      return false;
    }
  }
};

} // namespace vecops::matmul::details::orientation

#endif // VECOPS_MATMUL_DETAILS_ORIENTATION_AMX_BACKEND_H
