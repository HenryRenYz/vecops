// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
//
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
      // Direct vertical ZA stores remove the former C-layout asymmetry.  Raw,
      // converted and online-packed probes are now neutral within noise, so
      // preserve the caller's orientation and avoid an extra specialization.
      // A future asymmetric narrow-N leaf can re-enable swap from its own
      // capability profile rather than from output layout alone.
      return false;
    }
  }
};

} // namespace vecops::matmul::details::orientation

#endif // VECOPS_MATMUL_DETAILS_ORIENTATION_SME_BACKEND_H
