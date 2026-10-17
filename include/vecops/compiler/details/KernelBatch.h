// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
// Copyright (c) vecops contributors.
#ifndef VECOPS_COMPILER_DETAILS_KERNEL_BATCH_H
#define VECOPS_COMPILER_DETAILS_KERNEL_BATCH_H

#include <cstddef>
#include <map>
#include <memory>
#include <span>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "vecops/runtime/Operator.h"

namespace vecops::compiler::details {

/** Cold-build plan within one identical compiler configuration.
 *
 * Callers must match/bind every input call before constructing this plan.
 * Runtime Dynamic observations deliberately do not select compiled artifacts;
 * their relationships have already been checked by each call's binding. The
 * result retains the input-to-artifact map for provider adoption, including
 * different providers/cache namespaces requesting the same executable.
 */
struct DeduplicatedKernelBatch {
  std::vector<runtime::BoundKernelRecipe> recipes;
  std::vector<std::size_t> input_to_artifact;
};

/** Combine only exact source-artifact identities, preserving all validators.
 *
 * Source path, declarative interface, binding key and logical parallelism are
 * defensive identity components in addition to the provider's artifact key.
 * Each representative retains every independent alias validator owner. Shared
 * owners are checked once at each compiler boundary, before building and before
 * publishing; different owners are never merged by source/artifact identity.
 * No runtime argument/relationship validation may be replaced by this helper.
 */
[[nodiscard]] inline runtime::Result<DeduplicatedKernelBatch>
deduplicate_kernel_batch(std::span<const runtime::BoundKernelRecipe> inputs) {
  using Source = runtime::BoundKernelRecipe::SourceInstantiation;
  using Identity = std::tuple<std::string, std::string, std::string, std::string, std::string, std::string, nint_t>;
  DeduplicatedKernelBatch result;
  result.recipes.reserve(inputs.size());
  result.input_to_artifact.reserve(inputs.size());
  std::map<Identity, std::size_t> artifacts;
  std::vector<std::vector<std::shared_ptr<const Source>>> aliases;
  for (const auto& input : inputs) {
    if (input.source == nullptr || input.source->definition == nullptr)
      return runtime::Status(runtime::StatusCode::InvalidArgument,
                             "bound recipe does not carry a source kernel instantiation");
    const auto& source = *input.source;
    Identity identity{input.recipe_id,
                      input.artifact_key,
                      input.specialization_key,
                      source.kernel_file.generic_string(),
                      source.definition->canonical(),
                      source.binding.specialization_key,
                      source.parallelism};
    const auto [position, inserted] = artifacts.try_emplace(std::move(identity), result.recipes.size());
    const auto artifact = position->second;
    if (inserted) {
      result.recipes.push_back(input);
      aliases.emplace_back();
    }
    result.input_to_artifact.push_back(artifact);
    aliases[artifact].push_back(input.source);
  }
  for (std::size_t index = 0; index < aliases.size(); ++index) {
    if (aliases[index].size() == 1)
      continue;
    auto source = std::make_shared<Source>(*result.recipes[index].source);
    source->validate_inputs = runtime::BuildInputValidator{};
    for (const auto& alias : aliases[index])
      source->validate_inputs.append(alias->validate_inputs);
    result.recipes[index].source = std::move(source);
  }
  return result;
}

} // namespace vecops::compiler::details

#endif // VECOPS_COMPILER_DETAILS_KERNEL_BATCH_H
