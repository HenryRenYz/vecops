//
// Copyright (c) vecops contributors.
//

#ifndef VECOPS_OPS_MATMUL_PACK_H
#define VECOPS_OPS_MATMUL_PACK_H

/**
 * @file MatmulPack.h
 * @brief Public entry point for Matmul packing operators.
 */

#include <cstdint>
#include <utility>

#include "vecops/execution/ExecutionSession.h"
#include "vecops/matmul/details/packing/Plan.h"

namespace vecops::ops {

template <::vecops::matmul::Atom AtomT, ::vecops::matmul::Operand SideV>
struct MatmulPackConfig {
  using Atom = AtomT;
  static constexpr ::vecops::matmul::Operand side = SideV;
};

template <::vecops::matmul::Atom AtomT>
struct MatmulPackBCompensatedConfig {
  using Atom = AtomT;
  int32_t a_zero_point = 0;
};

template <typename Config>
class MatmulPack {
public:
  const Config config;

  VECOPS_INLINE constexpr explicit MatmulPack(Config cfg = {})
      : config(std::move(cfg)) {}

  template <tensor::InputOperand Input, tensor::OutputOperand Output>
  VECOPS_INLINE nint_t required_workspace(
      const Input& input, const Output& output) const {
    return ::vecops::matmul::details::matmul_pack_workspace_bytes<
        typename Config::Atom, Config::side>(input, output);
  }

  template <execution::ExecutionScope Scope,
            tensor::InputOperand Input, tensor::OutputOperand Output>
  VECOPS_INLINE void operator()(
      Scope& scope, Input&& input, Output&& output) const {
    ::vecops::matmul::details::run_matmul_pack<
        typename Config::Atom, Config::side>(
            scope, std::forward<Input>(input),
            std::forward<Output>(output));
  }

  template <tensor::InputOperand Input, tensor::OutputOperand Output>
  VECOPS_INLINE void operator()(
      kernel::WorkspaceView& workspace, Input&& input, Output&& output) const {
    ExecutionSession execution{workspace};
    (*this)(execution, std::forward<Input>(input),
            std::forward<Output>(output));
  }
};

template <typename Config>
class MatmulPackBCompensated {
public:
  const Config config;

  VECOPS_INLINE constexpr explicit MatmulPackBCompensated(Config cfg = {})
      : config(std::move(cfg)) {}

  template <tensor::InputOperand Input, tensor::OutputOperand Output,
            tensor::OutputOperand CompensationOutput>
  VECOPS_INLINE nint_t required_workspace(
      const Input& input, const Output& output,
      const CompensationOutput& compensation) const {
    return ::vecops::matmul::details::
        matmul_pack_b_compensated_workspace_bytes<typename Config::Atom>(
            input, output, compensation);
  }

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

template <typename Config>
VECOPS_INLINE constexpr auto matmul_pack(Config config) {
  return MatmulPack<Config>{std::move(config)};
}

template <typename Config>
VECOPS_INLINE constexpr auto matmul_pack_b_compensated(Config config) {
  return MatmulPackBCompensated<Config>{std::move(config)};
}

} // namespace vecops::ops

#endif // VECOPS_OPS_MATMUL_PACK_H
