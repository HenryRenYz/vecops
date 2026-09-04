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
template <typename AtomT, typename SchedulerPolicy, typename Implementation,
          kernel::loop::Tile2DTraversalOrder Traversal =
              kernel::loop::Tile2DTraversalOrder::m_major>
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
    // Bind operands before acquiring matrix state, then use the self-contained
    // backend entry.  This lets whole-problem vector/dot leaves serve both
    // unsplit blocks and split-K accumulator phases; only the General fallback
    // opens TILECFG or StreamingZA.
    kernel::with_operands(
        scope,
        tensor::operand(a, APolicy{}),
        tensor::operand(b, BPolicy{}),
        tensor::operand(c_input, CInputPolicy{}),
        tensor::operand(c_output, COutputPolicy{}),
        [&](auto& a_access, auto& b_access,
            auto& c_input_access, auto& c_output_access)
            VECOPS_KERNEL_LAMBDA {
          if constexpr (Traversal ==
                        kernel::loop::Tile2DTraversalOrder::m_major) {
            kernel::matmul_bound<
                AtomT, SchedulerPolicy, true,
                ::vecops::matmul::details::AutomaticFamilyDispatch>(
                    scope, m, n, k, a_access, b_access,
                    c_input_access, c_output_access, scratch,
                    Implementation{});
          } else {
            kernel::matmul_bound_n_major<
                AtomT, SchedulerPolicy, true,
                ::vecops::matmul::details::AutomaticFamilyDispatch>(
                    scope, m, n, k, a_access, b_access,
                    c_input_access, c_output_access, scratch,
                    Implementation{});
          }
          // Flush transformed/materialized outputs before their access session
          // goes out of scope. Direct storage commits as a no-op.
          c_output_access.commit();
        });
  }

  /**
   * Split-K form. Both original C and accumulator endpoints are bound once;
   * Route selects which pair the backend uses without changing its template
   * identity across first/middle/last K blocks.
   */
  template <execution::ExecutionScope Scope,
            meta::ValueType M, meta::ValueType N, meta::ValueType K,
            tensor::InputSpecLike ASpec, tensor::InputSpecLike BSpec,
            tensor::InputSpecLike CInputSpec,
            tensor::OutputSpecLike COutputSpec,
            tensor::InputSpecLike AccInputSpec,
            tensor::OutputSpecLike AccOutputSpec,
            typename Route>
  VECOPS_ALWAYS_INLINE static void run_phased(
      Scope& scope, M m, N n, K k,
      const ASpec& a, const BSpec& b,
      const CInputSpec& c_input, const COutputSpec& c_output,
      const AccInputSpec& acc_input, const AccOutputSpec& acc_output,
      Route route, void* scratch) {
    using APolicy = tensor::InputAccessPolicy<
        ASpec::InputTensor::Ndim - 1, 1, tensor::AccessPlan::direct>;
    using BPolicy = tensor::InputAccessPolicy<
        BSpec::InputTensor::Ndim - 1, 1, tensor::AccessPlan::direct>;
    using CInputPolicy = tensor::InputAccessPolicy<
        CInputSpec::InputTensor::Ndim - 1, 1,
        tensor::AccessPlan::direct>;
    using COutputPolicy = tensor::OutputAccessPolicy<
        COutputSpec::OutputTensor::Ndim - 1,
        tensor::AccessPlan::direct>;
    using AccInputPolicy = tensor::InputAccessPolicy<
        AccInputSpec::InputTensor::Ndim - 1, 1,
        tensor::AccessPlan::direct>;
    using AccOutputPolicy = tensor::OutputAccessPolicy<
        AccOutputSpec::OutputTensor::Ndim - 1,
        tensor::AccessPlan::direct>;
    kernel::with_operands(
        scope,
        tensor::operand(a, APolicy{}),
        tensor::operand(b, BPolicy{}),
        [&](auto& a_access, auto& b_access) VECOPS_KERNEL_LAMBDA {
          kernel::with_operands(
              scope,
              tensor::operand(c_input, CInputPolicy{}),
              tensor::operand(c_output, COutputPolicy{}),
              tensor::operand(acc_input, AccInputPolicy{}),
              tensor::operand(acc_output, AccOutputPolicy{}),
              [&](auto& c_input_access, auto& c_output_access,
                  auto& acc_input_access, auto& acc_output_access)
                  VECOPS_KERNEL_LAMBDA {
                if constexpr (Traversal ==
                              kernel::loop::Tile2DTraversalOrder::m_major) {
                  kernel::matmul_bound_phased<AtomT, SchedulerPolicy>(
                      scope, m, n, k, a_access, b_access,
                      c_input_access, c_output_access,
                      acc_input_access, acc_output_access,
                      route, scratch, Implementation{});
                } else {
                  kernel::matmul_bound_phased_n_major<
                      AtomT, SchedulerPolicy>(
                          scope, m, n, k, a_access, b_access,
                          c_input_access, c_output_access,
                          acc_input_access, acc_output_access,
                          route, scratch, Implementation{});
                }
                // Both are direct sessions today; marking both complete keeps
                // ownership correct while Route decides which one was written.
                c_output_access.commit();
                acc_output_access.commit();
              });
        });
  }
};

} // namespace vecops::matmul::details

#endif // VECOPS_MATMUL_DETAILS_SCHEDULER_H
