//
// Copyright (c) vecops contributors.
//

#ifndef VECOPS_OPS_DETAILS_MATMUL_PACK_SELECTION_H
#define VECOPS_OPS_DETAILS_MATMUL_PACK_SELECTION_H

#include <type_traits>

#include "vecops/gemm/Packing.h"
#include "vecops/kernel/details/matmul_pack/Backend.h"

namespace vecops::ops::matmul_pack_details {

template <gemm::Atom Atom, gemm::Operand Side,
          typename InputSpec, typename OutputSpec>
using SelectedImplementation = std::conditional_t<
    kernel::matmul_pack_details::Backend<
        typename gemm::packing_t<Atom, Side>::FormatType,
        kernel::matmul_pack_implementation::SME>::template eligible<
            InputSpec, OutputSpec>,
    kernel::matmul_pack_implementation::SME,
    std::conditional_t<
        kernel::matmul_pack_details::Backend<
            typename gemm::packing_t<Atom, Side>::FormatType,
            kernel::matmul_pack_implementation::SMEFP32ToFP64>::
            template eligible<InputSpec, OutputSpec>,
        kernel::matmul_pack_implementation::SMEFP32ToFP64,
        std::conditional_t<
            kernel::matmul_pack_details::Backend<
                typename gemm::packing_t<Atom, Side>::FormatType,
                kernel::matmul_pack_implementation::SMEStagedTransform>::
                template eligible<InputSpec, OutputSpec>,
            kernel::matmul_pack_implementation::SMEStagedTransform,
            std::conditional_t<
                kernel::matmul_pack_details::Backend<
                    typename gemm::packing_t<Atom, Side>::FormatType,
                    kernel::matmul_pack_implementation::SMEStagedFP16ToFP32>::
                    template eligible<InputSpec, OutputSpec>,
                kernel::matmul_pack_implementation::SMEStagedFP16ToFP32,
                std::conditional_t<
                    kernel::matmul_pack_details::Backend<
                        typename gemm::packing_t<Atom, Side>::FormatType,
                        kernel::matmul_pack_implementation::SMEPostprocess>::
                        template eligible<InputSpec, OutputSpec>,
                    kernel::matmul_pack_implementation::SMEPostprocess,
                    kernel::matmul_pack_implementation::Vector>>>>>;

} // namespace vecops::ops::matmul_pack_details

#endif // VECOPS_OPS_DETAILS_MATMUL_PACK_SELECTION_H
