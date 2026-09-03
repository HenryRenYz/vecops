//
// Copyright (c) vecops contributors.
//

#ifndef VECOPS_OPS_MATMUL_PACK_H
#define VECOPS_OPS_MATMUL_PACK_H

/**
 * @file vecops/ops/MatmulPack.h
 * @brief Public entry point for Matmul packing operators.
 *
 * Matrix-multiply hardware (Intel AMX tiles, Arm SME ZA) consumes operands
 * in blocked on-chip layouts, not the plain row-major `[M,K]` / `[N,K]`
 * tensors users hold. **Packing** is the data-movement step that rearranges
 * one logical operand into that blocked layout: it is a pure permutation
 * (plus zero-padding of the trailing partial block), so a packed operand is
 * bit-identical in content to its source and can be reused across any
 * number of matmuls that share the operand.
 *
 * `MatmulPack<Config>` packs one operand — which one is fixed by the
 * config's `side` (A or B; A is `[M,K]`, B is `[N,K]`, matching the
 * `C = A*B^T` convention of ops/Matmul.h). The destination must already be
 * a tensor over `matmul::packed_layout<Atom, Side>(input_layout)`; passing
 * such a packed tensor as A or B to `Matmul` lets the planner skip the
 * packing stage entirely (precomputed weights are the canonical use).
 *
 * `MatmulPackBCompensated<Config>` is the quantization-oriented variant for
 * asymmetric-A × signed-B integer matmul (u8 × s8 → i32): besides packing B
 * exactly like `MatmulPack`, it emits an `int32_t[N]` sidecar of
 * `-a_zero_point * sum_k(B[n,k])`, which can be broadcast directly as the
 * C input of the subsequent matmul to correct for A's zero-point offset.
 * It is a separate operator (not a nullable option on `MatmulPack`) so the
 * ordinary packing path keeps the same template instantiation and generated
 * instructions.
 *
 * ## Usage
 *
 * @code
 * #include "vecops/ops/MatmulPack.h"
 * using namespace vecops;
 *
 * auto packer = ops::matmul_pack(
 *     ops::MatmulPackConfig<matmul::AMX_BF16F32,
 *                           matmul::Operand::B>{});
 * auto packed_layout =
 *     matmul::packed_layout<matmul::AMX_BF16F32, matmul::Operand::B>(b_layout);
 * // allocate output over packed_layout, then:
 * packer(scope, b_tensor, packed_b_tensor);
 *
 * // Compensated variant (u8 A x s8 B atoms only):
 * auto compensated = ops::matmul_pack_b_compensated(
 *     ops::MatmulPackBCompensatedConfig<matmul::AMX_I8I32<uint8_t, int8_t>>{
 *         .a_zero_point = zp});
 * compensated(scope, b_tensor, packed_b_tensor, compensation_vector);
 * @endcode
 *
 * ## Pitfalls
 *
 * - The output layout must be obtained from
 *   `matmul::packed_layout<Atom, Side>(input_layout)`; a mismatched (e.g.
 *   row-major) output is rejected at compile time or by assertion.
 * - Packed layouts are per-atom and per-side: a B-pack for one atom cannot
 *   feed a matmul configured with another atom, and an A-pack is not a
 *   B-pack.
 * - Packing needs the execution resources of its backend (SME variants use
 *   ZA); use the scope overloads or the temporary-session workspace
 *   overloads as with `Matmul`.
 * - `MatmulPackBCompensated` additionally requires the atom's types to be
 *   exactly `TA=uint8_t`, `TB=int8_t`, `TAcc=int32_t`, and the
 *   compensation output to be a native `int32_t` rank-1 tensor of length N.
 */

#include <cstdint>
#include <utility>

#include "vecops/execution/ExecutionSession.h"
#include "vecops/matmul/details/packing/Plan.h"

namespace vecops::ops {

/**
 * @brief Configuration for `MatmulPack`: which operand of the matmul to pack.
 *
 * @tparam AtomT  Hardware atom whose `Packing<Side>` format defines the
 *                blocked output layout.
 * @tparam SideV  The operand being packed: `Operand::A` for the `[M,K]`
 *                operand, `Operand::B` for the `[N,K]` operand. The packed
 *                result may only be fed back as that same side of a matmul
 *                configured with the same atom.
 */
template <::vecops::matmul::Atom AtomT, ::vecops::matmul::Operand SideV>
struct MatmulPackConfig {
  using Atom = AtomT;
  static constexpr ::vecops::matmul::Operand side = SideV;
};

/**
 * @brief Configuration for `MatmulPackBCompensated`.
 *
 * @tparam AtomT Hardware atom; must have `TA=uint8_t`, `TB=int8_t`,
 *               `TAcc=int32_t` (asserted at instantiation).
 */
template <::vecops::matmul::Atom AtomT>
struct MatmulPackBCompensatedConfig {
  using Atom = AtomT;
  /**
   * @brief Zero-point of the asymmetrically-quantized A operand.
   *
   * The compensation output stores `-a_zero_point * sum_k(B[n,k])` per
   * output column; broadcasting it as the C input of the u8×s8 matmul
   * completes the correction for A's zero-point offset. Zero (symmetric A)
   * makes the compensation all-zero but still emits it.
   */
  int32_t a_zero_point = 0;
};

/**
 * @brief Reusable operator that packs one matmul operand into its blocked
 * hardware layout.
 *
 * The object stores configuration only; source and destination tensors are
 * passed per call. The packed output can be reused across any number of
 * matmul invocations that share the operand (weights being the canonical
 * case), letting the planner skip re-packing.
 *
 * @tparam Config A `MatmulPackConfig` (atom + side) bundle.
 */
template <typename Config>
class MatmulPack {
public:
  /** @brief The stored configuration bundle. */
  const Config config;

  /**
   * @brief Construct a packer from a configuration bundle.
   * @param cfg Configuration to store (moved in).
   */
  VECOPS_INLINE constexpr explicit MatmulPack(Config cfg = {})
      : config(std::move(cfg)) {}

  /**
   * @brief Workspace requirement for one packing call.
   * @tparam Input   Source operand (rank-2 logical tensor).
   * @tparam Output  Destination over the packed layout.
   * @param input    The logical operand to read.
   * @param output   The packed tensor to fill.
   * @return Workspace size in bytes.
   */
  template <tensor::InputOperand Input, tensor::OutputOperand Output>
  VECOPS_INLINE nint_t required_workspace(
      const Input& input, const Output& output) const {
    return ::vecops::matmul::details::matmul_pack_workspace_bytes<
        typename Config::Atom, Config::side>(input, output);
  }

  /**
   * @brief Pack one operand inside an execution scope.
   * @tparam Scope   An `execution::ExecutionScope` with the pack backend's
   *                 resources activated.
   * @tparam Input   Source operand (rank-2 logical tensor).
   * @tparam Output  Destination over the packed layout.
   * @param scope    Active execution scope to run under.
   * @param input    The logical operand to read.
   * @param output   The packed tensor to fill.
   */
  template <execution::ExecutionScope Scope,
            tensor::InputOperand Input, tensor::OutputOperand Output>
  VECOPS_INLINE void operator()(
      Scope& scope, Input&& input, Output&& output) const {
    ::vecops::matmul::details::run_matmul_pack<
        typename Config::Atom, Config::side>(
            scope, std::forward<Input>(input),
            std::forward<Output>(output));
  }

  /**
   * @brief Pack one operand against a raw workspace view.
   *
   * Wraps `workspace` in a temporary `ExecutionSession` and forwards to the
   * scope-based overload.
   *
   * @tparam Input   Source operand (rank-2 logical tensor).
   * @tparam Output  Destination over the packed layout.
   * @param workspace View satisfying `required_workspace(...)`.
   * @param input    The logical operand to read.
   * @param output   The packed tensor to fill.
   */
  template <tensor::InputOperand Input, tensor::OutputOperand Output>
  VECOPS_INLINE void operator()(
      kernel::WorkspaceView& workspace, Input&& input, Output&& output) const {
    ExecutionSession execution{workspace};
    (*this)(execution, std::forward<Input>(input),
            std::forward<Output>(output));
  }
};

/**
 * @brief Reusable operator: pack signed-byte B and generate the asymmetric-A
 * column correction in one pass.
 *
 * Produces the same packed B as `MatmulPack` for `Operand::B`, plus a
 * rank-1 `int32_t[N]` compensation vector holding
 * `-a_zero_point * sum_k(B[n,k])` that can be broadcast as the C input of
 * the subsequent u8×s8→i32 matmul.
 *
 * @tparam Config A `MatmulPackBCompensatedConfig` bundle.
 */
template <typename Config>
class MatmulPackBCompensated {
public:
  /** @brief The stored configuration bundle. */
  const Config config;

  /**
   * @brief Construct from a configuration bundle.
   * @param cfg Configuration to store (moved in).
   */
  VECOPS_INLINE constexpr explicit MatmulPackBCompensated(Config cfg = {})
      : config(std::move(cfg)) {}

  /**
   * @brief Workspace requirement for one compensated packing call.
   * @tparam Input               Source operand (rank-2 signed-byte tensor).
   * @tparam Output              Destination over the packed B layout.
   * @tparam CompensationOutput  Rank-1 native `int32_t[N]` output.
   * @return Workspace size in bytes (currently always 0; kept for API
   *         uniformity with the other pack operators).
   */
  template <tensor::InputOperand Input, tensor::OutputOperand Output,
            tensor::OutputOperand CompensationOutput>
  VECOPS_INLINE nint_t required_workspace(
      const Input& input, const Output& output,
      const CompensationOutput& compensation) const {
    return ::vecops::matmul::details::
        matmul_pack_b_compensated_workspace_bytes<typename Config::Atom>(
            input, output, compensation);
  }

  /**
   * @brief Pack B and emit the compensation vector inside an execution
   * scope.
   * @tparam Scope               An `execution::ExecutionScope` with the
   *                             pack backend's resources activated.
   * @tparam Input               Source operand (rank-2 signed-byte tensor).
   * @tparam Output              Destination over the packed B layout.
   * @tparam CompensationOutput  Rank-1 native `int32_t[N]` output.
   * @param scope                Active execution scope to run under.
   * @param input                The logical B operand to read.
   * @param output               The packed B tensor to fill.
   * @param compensation         The correction vector to fill.
   */
  template <execution::ExecutionScope Scope,
            tensor::InputOperand Input, tensor::OutputOperand Output,
            tensor::OutputOperand CompensationOutput>
  VECOPS_INLINE void operator()(
      Scope& scope, Input&& input, Output&& output,
      CompensationOutput&& compensation) const {
    ::vecops::matmul::details::run_matmul_pack_b_compensated<
        typename Config::Atom>(
            scope, std::forward<Input>(input),
            std::forward<Output>(output),
            std::forward<CompensationOutput>(compensation),
            config.a_zero_point);
  }

  /**
   * @brief Pack B and emit the compensation vector against a raw workspace
   * view.
   *
   * Wraps `workspace` in a temporary `ExecutionSession` and forwards to the
   * scope-based overload.
   *
   * @tparam Input               Source operand (rank-2 signed-byte tensor).
   * @tparam Output              Destination over the packed B layout.
   * @tparam CompensationOutput  Rank-1 native `int32_t[N]` output.
   * @param workspace            View satisfying `required_workspace(...)`.
   * @param input                The logical B operand to read.
   * @param output               The packed B tensor to fill.
   * @param compensation         The correction vector to fill.
   */
  template <tensor::InputOperand Input, tensor::OutputOperand Output,
            tensor::OutputOperand CompensationOutput>
  VECOPS_INLINE void operator()(
      kernel::WorkspaceView& workspace, Input&& input, Output&& output,
      CompensationOutput&& compensation) const {
    ExecutionSession execution{workspace};
    (*this)(execution, std::forward<Input>(input),
            std::forward<Output>(output),
            std::forward<CompensationOutput>(compensation));
  }
};

/**
 * @brief Factory: build a `MatmulPack` operator from a configuration.
 * @tparam Config The configuration bundle type.
 * @param config  Configuration to store inside the operator.
 * @return A `MatmulPack<Config>` value.
 */
template <typename Config>
VECOPS_INLINE constexpr auto matmul_pack(Config config) {
  return MatmulPack<Config>{std::move(config)};
}

/**
 * @brief Factory: build a `MatmulPackBCompensated` operator from a
 * configuration.
 * @tparam Config The configuration bundle type.
 * @param config  Configuration to store inside the operator.
 * @return A `MatmulPackBCompensated<Config>` value.
 */
template <typename Config>
VECOPS_INLINE constexpr auto matmul_pack_b_compensated(Config config) {
  return MatmulPackBCompensated<Config>{std::move(config)};
}

} // namespace vecops::ops

#endif // VECOPS_OPS_MATMUL_PACK_H
