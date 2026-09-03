//
// Copyright (c) vecops contributors.
//

#ifndef VECOPS_MATMUL_DETAILS_SCHEDULER_H
#define VECOPS_MATMUL_DETAILS_SCHEDULER_H

#include "vecops/matmul/Packing.h"
#include "vecops/matmul/details/kernel/Kernel.h"
#include "vecops/tensor/DataAccess.h"

/**
 * @file vecops/matmul/details/tiled/Scheduler.h
 * @brief Schedule exactly one MC x NC x KC block onto the backend Tile2D
 *        layer (the innermost layer of the generic tiled matmul).
 */

namespace vecops::matmul::details {

/** Schedule exactly one MC x NC x KC block on the backend Tile2D layer. */
template <typename AtomT, typename SchedulerPolicy, typename Implementation>
struct Scheduler {
  template <execution::ExecutionScope Scope,
            meta::ValueType M, meta::ValueType N, meta::ValueType K,
            tensor::InputSpecLike ASpec, tensor::InputSpecLike BSpec,
            tensor::InputSpecLike CInputSpec,
            tensor::OutputSpecLike COutputSpec>
  VECOPS_ALWAYS_INLINE static void run(
      Scope& scope, M m, N n, K k,
      const ASpec& a, const BSpec& b,
      const CInputSpec& c_input, const COutputSpec& c_output,
      void* scratch) {
    // All four operands are consumed along their last axis (K for A/B, N for
    // C) with a unit stride and a direct memory plan. That is exactly the
    // row-major shape the backends' fast paths are written against (SME
    // strided u32 gathers, ZA row loads, AMX row loads), so the tiler hands
    // the backend the layout it wants instead of a generic one.
    using APolicy = tensor::InputAccessPolicy<
        ASpec::InputTensor::Ndim - 1, 1, tensor::AccessPlan::direct>;
    using BPolicy = tensor::InputAccessPolicy<
        BSpec::InputTensor::Ndim - 1, 1, tensor::AccessPlan::direct>;
    using CInputPolicy = tensor::InputAccessPolicy<
        CInputSpec::InputTensor::Ndim - 1, 1, tensor::AccessPlan::direct>;
    using COutputPolicy = tensor::OutputAccessPolicy<
        COutputSpec::OutputTensor::Ndim - 1, tensor::AccessPlan::direct>;
    // Packed B keeps its packed layout; the flag tells the configuration
    // layer this leaf already consumes the packed input format.
    constexpr bool PackedB = is_packed_layout<
        AtomT, Operand::B, typename BSpec::InputLayout>();
    kernel::with_matmul_configuration<
        AtomT, SchedulerPolicy, PackedB>(
            scope, m, n, Implementation{},
            [&](auto& configured) VECOPS_INLINE_LAMBDA_NOEXCEPT {
              kernel::with_operands(
                  configured,
                  tensor::operand(a, APolicy{}),
                  tensor::operand(b, BPolicy{}),
                  tensor::operand(c_input, CInputPolicy{}),
                  tensor::operand(c_output, COutputPolicy{}),
                  [&](auto& a_access, auto& b_access,
                      auto& c_input_access, auto& c_output_access)
                      VECOPS_KERNEL_LAMBDA {
                    kernel::matmul_bound_configured<AtomT, SchedulerPolicy>(
                        configured, m, n, k, a_access, b_access,
                        c_input_access, c_output_access, scratch,
                        Implementation{});
                    // Flush the output session while the kernel's accesses
                    // are still alive. A no-op for direct storage; mandatory
                    // for transformed/materialized outputs, whose deferred
                    // write-back must not outlive this scope.
                    c_output_access.commit();
                  });
            });
  }
};

} // namespace vecops::matmul::details

#endif // VECOPS_MATMUL_DETAILS_SCHEDULER_H
