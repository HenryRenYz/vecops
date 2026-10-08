// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
#ifndef VECOPS_OPS_SOFTMAX_H
#define VECOPS_OPS_SOFTMAX_H

/**
 * @file vecops/ops/Softmax.h
 * @brief Public entry point for the Softmax operator.
 *
 * `softmax` normalizes the last dimension of an N-D operand. Every prefix
 * coordinate addresses one independent row:
 *
 *     y_i = exp(x_i - max(x)) / sum_j exp(x_j - max(x))
 *
 * The operator object stores only compile-time configuration
 * (`SoftmaxConfig`: compute type, exp accuracy, online toggle); all shapes
 * and operands are passed per call, so one object serves any rank and any
 * number of problem sizes.
 *
 * ## Usage
 *
 * @code
 * #include "vecops/ops/Softmax.h"
 * using namespace vecops;
 * using namespace vecops::tensor;
 *
 * auto x = make_tensor<2>(x_data, {rows, n});
 * auto y = make_tensor<2>(y_data, {rows, n});
 *
 * // One-shot form: workspace and ExecutionSession are created internally.
 * ops::softmax()(input<float32_t>(x), output<float32_t>(y));
 *
 * // Caller-owned workspace form (reuse one buffer across many calls).
 * auto op = ops::softmax();
 * auto x_spec = input<float32_t>(x);
 * auto y_spec = output<float32_t>(y);
 * kernel::Workspace storage(op.required_workspace(x_spec, y_spec));
 * auto view = storage.view();
 * op(view, x_spec, y_spec);
 * @endcode
 *
 * A third form takes an `ExecutionScope&` first argument and reuses an
 * already-active hardware region; nested calls under a compatible scope
 * compile to no additional resource transition.
 *
 * ## Pitfalls
 *
 * - The normalized dimension is always the **last** one; there is no axis
 *   parameter. Transpose first if another axis must be normalized.
 * - Input and output shapes must match exactly (assertion).
 * - The one-shot form allocates workspace and an `ExecutionSession` per
 *   call; prefer the workspace or scope forms in hot loops.
 * - `required_workspace()` is shape-dependent: re-query it after shape
 *   changes, and size the buffer from the largest planned call.
 */

#include "vecops/ops/details/softmax/Operation.h"

#endif // VECOPS_OPS_SOFTMAX_H
