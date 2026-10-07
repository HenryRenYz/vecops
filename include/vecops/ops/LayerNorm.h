// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
#ifndef VECOPS_OPS_LAYERNORM_H
#define VECOPS_OPS_LAYERNORM_H

/**
 * @file vecops/ops/LayerNorm.h
 * @brief Public entry point for the LayerNorm operator.
 *
 * `layer_norm` normalizes the last dimension of an N-D operand and
 * optionally applies an affine transform:
 *
 *     y_i = gamma_i * (x_i - mean) / sqrt(var + eps) + beta_i
 *
 * where mean/var are computed over the last dimension of each row and
 * gamma/beta are optional rank-one operands of that length (pass
 * `tensor::nullopt` to skip either). The operator object stores only the
 * runtime config (`LayerNormConfig`: compute type, `eps`); shapes and
 * operands are passed per call.
 *
 * ## Usage
 *
 * @code
 * #include "vecops/ops/LayerNorm.h"
 * using namespace vecops;
 * using namespace vecops::tensor;
 *
 * auto x = make_tensor<2>(x_data, {rows, n});
 * auto y = make_tensor<2>(y_data, {rows, n});
 * auto g = make_tensor<1>(g_data, {n});
 * auto b = make_tensor<1>(b_data, {n});
 *
 * // One-shot form (self-allocated workspace + ExecutionSession):
 * ops::layer_norm()(input<float32_t>(x), input<float32_t>(g),
 *                   input<float32_t>(b), output<float32_t>(y));
 *
 * // Without affine parameters:
 * ops::layer_norm()(input<float32_t>(x), output<float32_t>(y));
 * @endcode
 *
 * Workspace-form and ExecutionScope-form overloads mirror `softmax`:
 * query `required_workspace(in, g, b, out)`, or pass an active scope as
 * the first argument to reuse an already-active hardware region.
 *
 * ## Pitfalls
 *
 * - The normalized dimension is always the **last** one; gamma/beta must
 *   match it exactly (assertion).
 * - Input and output shapes must match exactly (assertion).
 * - Variance uses the single-pass `E[x^2] - E[x]^2` formula: inputs with a
 *   large mean and tiny variance can lose precision to cancellation.
 * - The one-shot form allocates per call; prefer the workspace/scope forms
 *   in hot loops.
 */

#include "vecops/ops/details/layernorm/Operation.h"

#endif // VECOPS_OPS_LAYERNORM_H
