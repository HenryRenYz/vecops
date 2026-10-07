// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
#ifndef VECOPS_TENSOR_OPTIONAL_OPERAND_H
#define VECOPS_TENSOR_OPTIONAL_OPERAND_H

#include <concepts>
#include <type_traits>

namespace vecops::tensor {

/**
 * @brief Compile-time sentinel for an omitted operator operand.
 *
 * `nullopt_t` is deliberately not a Tensor, Spec, or DataAccess session.  It
 * carries no Layout and does not participate in tensor traversal; individual
 * operators decide which operand positions accept it and what omission means.
 */
struct nullopt_t final {};

inline constexpr nullopt_t nullopt{};

template <typename T>
inline constexpr bool is_nullopt_v =
    std::same_as<std::remove_cvref_t<T>, nullopt_t>;

} // namespace vecops::tensor

#endif // VECOPS_TENSOR_OPTIONAL_OPERAND_H
