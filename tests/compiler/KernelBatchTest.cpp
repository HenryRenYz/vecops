// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>

#include <array>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "vecops/compiler/details/KernelBatch.h"

namespace {

using namespace vecops::runtime;
using vecops::compiler::details::deduplicate_kernel_batch;

std::shared_ptr<const KernelDef> definition(std::string name = "test::batch-dynamic") {
  const auto batch = DimensionDef::named_dynamic("B", 1, 0, 1024);
  return std::make_shared<const KernelDef>(
    std::move(name),
    std::vector<KernelParameterDef>{
      TensorDef{"input", {batch, "D"}, {"D", 1}, TensorDTypeDef("IOType")},
      TensorDef{"output", {batch, "D"}, {"D", 1}, TensorDTypeDef("IOType"), false, TensorAccess::Output}},
    std::map<std::string, SpecializationType, std::less<>>{{"D", SpecializationType::ConstInt},
                                                           {"IOType", SpecializationType::DType}});
}

KernelCall call(std::int64_t rows, std::int64_t columns, DType dtype = DType::Float32, std::int64_t output_rows = -1) {
  static float data = 0;
  return KernelCall(make_arguments(
    TensorView{
      .data = &data, .dtype = dtype, .sizes = {rows, columns}, .strides = {columns, 1}, .flags = VECOPS_TENSOR_READ},
    TensorView{.data = &data,
               .dtype = dtype,
               .sizes = {output_rows < 0 ? rows : output_rows, columns},
               .strides = {columns, 1},
               .flags = VECOPS_TENSOR_WRITE}));
}

BoundKernelRecipe recipe(std::int64_t rows, std::int64_t columns = 4, DType dtype = DType::Float32,
                         std::function<bool()> validate = {}) {
  SourceKernelRecipe source("batch-source", "batch.cpp", definition(), "source-fingerprint", {}, std::move(validate));
  auto bound = source.bind(call(rows, columns, dtype));
  EXPECT_TRUE(bound.ok()) << bound.status().message();
  return std::move(bound).value();
}

TEST(KernelBatchTest, DynamicObservationsShareOneArtifactAndKeepInputOrder) {
  const std::array inputs{recipe(2), recipe(19), recipe(3, 8), recipe(23), recipe(5, 8)};
  ASSERT_EQ(inputs[0].specialization_key, inputs[1].specialization_key);
  auto result = deduplicate_kernel_batch(inputs);
  ASSERT_TRUE(result.ok()) << result.status().message();
  EXPECT_EQ(result.value().recipes.size(), 2);
  EXPECT_EQ(result.value().input_to_artifact, (std::vector<std::size_t>{0, 0, 1, 0, 1}));
  EXPECT_EQ(result.value().recipes[0].specialization_key, inputs[0].specialization_key);
  EXPECT_EQ(result.value().recipes[1].specialization_key, inputs[2].specialization_key);
}

TEST(KernelBatchTest, ConstantAndDTypeSpecializationsRemainDistinct) {
  const std::array inputs{recipe(2), recipe(19, 8), recipe(23, 4, DType::Float64)};
  auto result = deduplicate_kernel_batch(inputs);
  ASSERT_TRUE(result.ok());
  EXPECT_EQ(result.value().recipes.size(), 3);
  EXPECT_EQ(result.value().input_to_artifact, (std::vector<std::size_t>{0, 1, 2}));
}

TEST(KernelBatchTest, SourceSchemaAndParallelismRemainDefensiveIdentityComponents) {
  const auto original = recipe(2);
  std::vector<BoundKernelRecipe> inputs(8, original);
  inputs[1].recipe_id += "-other";
  inputs[2].artifact_key += "-other";
  inputs[3].specialization_key += "-other";
  for (std::size_t index = 4; index < inputs.size(); ++index) {
    auto source = std::make_shared<BoundKernelRecipe::SourceInstantiation>(*original.source);
    if (index == 4)
      source->kernel_file = "other.cpp";
    else if (index == 5)
      source->definition = definition("test::other-schema");
    else if (index == 6)
      source->binding.specialization_key += "-other";
    else
      source->parallelism = 7;
    inputs[index].source = std::move(source);
  }
  auto result = deduplicate_kernel_batch(inputs);
  ASSERT_TRUE(result.ok());
  EXPECT_EQ(result.value().recipes.size(), inputs.size());
}

TEST(KernelBatchTest, EveryAliasIsValidatedAtBothCompilerBoundaries) {
  std::array<int, 3> counts{};
  const std::array inputs{recipe(2, 4, DType::Float32, [&] { return ++counts[0] > 0; }),
                          recipe(19, 4, DType::Float32, [&] { return ++counts[1] > 0; }),
                          recipe(23, 4, DType::Float32, [&] { return ++counts[2] > 0; })};
  auto result = deduplicate_kernel_batch(inputs);
  ASSERT_TRUE(result.ok());
  ASSERT_EQ(result.value().recipes.size(), 1);
  const auto& validate = result.value().recipes[0].source->validate_inputs;
  ASSERT_TRUE(validate);
  EXPECT_TRUE(validate()); // Before compilation.
  EXPECT_TRUE(validate()); // Before publication.
  EXPECT_EQ(counts, (std::array{2, 2, 2}));
  EXPECT_NE(result.value().recipes[0].source, inputs[0].source);
  EXPECT_TRUE(inputs[0].source->validate_inputs());
  EXPECT_EQ(counts, (std::array{3, 2, 2})); // Original callbacks were not mutated.
}

TEST(KernelBatchTest, FailingAliasDoesNotShortCircuitOtherValidation) {
  std::array<int, 3> counts{};
  const std::array inputs{recipe(2, 4, DType::Float32,
                                 [&] {
                                   ++counts[0];
                                   return false;
                                 }),
                          recipe(19, 4, DType::Float32,
                                 [&] {
                                   ++counts[1];
                                   return true;
                                 }),
                          recipe(23, 4, DType::Float32, [&] {
                            ++counts[2];
                            return true;
                          })};
  auto result = deduplicate_kernel_batch(inputs);
  ASSERT_TRUE(result.ok());
  EXPECT_FALSE(result.value().recipes[0].source->validate_inputs());
  EXPECT_EQ(counts, (std::array{1, 1, 1}));
}

TEST(KernelBatchTest, AliasChangeBetweenBuildAndPublicationIsDetected) {
  int first = 0;
  int second = 0;
  const std::array inputs{recipe(2, 4, DType::Float32,
                                 [&] {
                                   ++first;
                                   return true;
                                 }),
                          recipe(19, 4, DType::Float32, [&] { return ++second == 1; })};
  auto result = deduplicate_kernel_batch(inputs);
  ASSERT_TRUE(result.ok());
  EXPECT_TRUE(result.value().recipes[0].source->validate_inputs());
  EXPECT_FALSE(result.value().recipes[0].source->validate_inputs());
  EXPECT_EQ(first, 2);
  EXPECT_EQ(second, 2);
}

TEST(KernelBatchTest, AggregatedValidationRetainsAliasLifetimeAndAllowsMissingCallback) {
  int count = 0;
  std::vector inputs{recipe(2), recipe(19, 4, DType::Float32, [&] { return ++count > 0; })};
  const std::weak_ptr alias = inputs[1].source;
  auto result = deduplicate_kernel_batch(inputs);
  ASSERT_TRUE(result.ok());
  inputs.clear();
  EXPECT_FALSE(alias.expired());
  EXPECT_TRUE(result.value().recipes[0].source->validate_inputs());
  EXPECT_EQ(count, 1);
}

TEST(KernelBatchTest, InvalidDynamicAliasStillFailsBindingBeforeDeduplication) {
  SourceKernelRecipe source("batch-source", "batch.cpp", definition(), "source-fingerprint");
  ASSERT_TRUE(source.bind(call(2, 4)).ok());
  auto invalid = source.bind(call(19, 4, DType::Float32, 23));
  EXPECT_FALSE(invalid.ok());
  EXPECT_EQ(invalid.status().code(), StatusCode::InvalidArgument);
}

TEST(KernelBatchTest, EmptyAndMissingSourceInputsAreHandled) {
  auto empty = deduplicate_kernel_batch({});
  ASSERT_TRUE(empty.ok());
  EXPECT_TRUE(empty.value().recipes.empty());
  EXPECT_TRUE(empty.value().input_to_artifact.empty());
  const std::array inputs{BoundKernelRecipe{}};
  auto missing = deduplicate_kernel_batch(inputs);
  EXPECT_FALSE(missing.ok());
  EXPECT_EQ(missing.status().code(), StatusCode::InvalidArgument);
  auto invalid = recipe(2);
  invalid.source = std::make_shared<BoundKernelRecipe::SourceInstantiation>();
  const std::array no_definition{invalid};
  EXPECT_FALSE(deduplicate_kernel_batch(no_definition).ok());
}

} // namespace
