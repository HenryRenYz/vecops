/**
 * @file Operator.cpp
 * @brief KernelCall normalization, ordered recipe dispatch, and execution.
 *
 * The source-recipe implementation binds all metadata before a provider can
 * load or compile an artifact. In particular `match` is metadata-only and
 * build-free, whereas provider resolution is the intentionally side-effectful
 * boundary. This file does not allocate tensor storage or workspace.
 */

#include "vecops/runtime/Operator.h"

#include <fstream>
#include <iomanip>
#include <numeric>
#include <sstream>

namespace vecops::runtime {
namespace {

OperatorSchema logical_schema(const KernelDef& definition) {
  std::vector<ParameterSchema> parameters;
  parameters.reserve(definition.parameters().size());
  for (const auto& parameter : definition.parameters()) {
    if (const auto* tensor = std::get_if<TensorDef>(&parameter)) {
      TensorConstraint constraint;
      constraint.rank = static_cast<std::uint32_t>(tensor->shape.size());
      constraint.device_type = tensor->device_type;
      constraint.device_index = tensor->device_index;
      constraint.optional = tensor->optional;
      constraint.required_flags = tensor->readable() && tensor->writable() ? VECOPS_TENSOR_READ | VECOPS_TENSOR_WRITE
                                  : tensor->writable()                     ? VECOPS_TENSOR_WRITE
                                                                           : VECOPS_TENSOR_READ;
      if (tensor->dtype.dtype())
        constraint.dtype_mask = dtype_bit(*tensor->dtype.dtype());
      constraint.sizes.reserve(tensor->shape.size());
      constraint.strides.reserve(tensor->strides.size());
      auto convert = [](const DimensionDef& dimension) {
        if (dimension.kind() == DimensionDef::Kind::Fixed)
          return DimensionConstraint::constant(dimension.fixed_value());
        if (dimension.kind() == DimensionDef::Kind::Dynamic && !dimension.alignment().is_symbol() &&
            !dimension.lower_bound().is_symbol() && !dimension.upper_bound().is_symbol())
          return DimensionConstraint::dynamic(dimension.lower_bound().fixed_value(),
                                              dimension.upper_bound().fixed_value(),
                                              dimension.alignment().fixed_value());
        return DimensionConstraint::any();
      };
      for (const auto& dimension : tensor->shape)
        constraint.sizes.push_back(convert(dimension));
      for (const auto& dimension : tensor->strides)
        constraint.strides.push_back(convert(dimension));
      parameters.push_back(ParameterSchema{tensor->name, std::move(constraint)});
    } else {
      const auto& scalar = std::get<ValueDef>(parameter);
      parameters.push_back(ParameterSchema{scalar.name, ScalarConstraint{dtype_bit(scalar.dtype)}});
    }
  }
  return OperatorSchema(definition.name(), std::move(parameters));
}

std::string fingerprint_file(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input)
    return {};
  std::uint64_t hash = UINT64_C(14695981039346656037);
  char buffer[64 * 1024];
  while (input) {
    input.read(buffer, sizeof(buffer));
    for (std::streamsize index = 0; index < input.gcount(); ++index) {
      hash ^= static_cast<unsigned char>(buffer[index]);
      hash *= UINT64_C(1099511628211);
    }
  }
  std::ostringstream output;
  output << std::hex << std::setw(16) << std::setfill('0') << hash;
  return output.str();
}

} // namespace

SourceKernelRecipe::SourceKernelRecipe(std::string id, std::filesystem::path kernel_file,
                                       std::shared_ptr<const KernelDef> definition, std::string source_fingerprint,
                                       SpecializationValues default_values)
  : id_(std::move(id))
  , kernel_file_(std::move(kernel_file))
  , definition_(std::move(definition))
  , source_fingerprint_(source_fingerprint.empty() ? fingerprint_file(kernel_file_) : std::move(source_fingerprint))
  , default_values_(std::move(default_values)) {
}

Status SourceKernelRecipe::match(const KernelCall& call) const {
  if (definition_ == nullptr || id_.empty() || kernel_file_.empty())
    return Status(StatusCode::InternalError, "source kernel recipe is incomplete");
  auto values = default_values_;
  for (const auto& [name, value] : call.values())
    values.insert_or_assign(name, value);
  auto binding = bind_kernel_call(*definition_, KernelCall(call.arguments(), std::move(values)));
  if (!binding)
    return Status(StatusCode::NotApplicable, binding.status().message());
  return Status::success();
}

Result<BoundKernelRecipe> SourceKernelRecipe::bind(const KernelCall& call) const {
  auto values = default_values_;
  for (const auto& [name, value] : call.values())
    values.insert_or_assign(name, value);
  auto binding = bind_kernel_call(*definition_, KernelCall(call.arguments(), std::move(values)));
  if (!binding)
    return binding.status();
  auto source = std::make_shared<BoundKernelRecipe::SourceInstantiation>(
    BoundKernelRecipe::SourceInstantiation{kernel_file_, definition_, std::move(binding).value()});
  return BoundKernelRecipe{id_, source->binding.specialization_key,
                           id_ + ";source=" + source_fingerprint_ + ";" + definition_->canonical() + ";" +
                             source->binding.specialization_key,
                           std::move(source)};
}

Result<std::vector<std::size_t>>
OrderedDispatchPolicy::candidates(const KernelCall&,
                                  std::span<const std::shared_ptr<const KernelRecipe>> recipes) const {
  std::vector<std::size_t> result(recipes.size());
  std::iota(result.begin(), result.end(), std::size_t{0});
  return result;
}

Operator::Operator(OperatorSchema schema, std::vector<std::shared_ptr<const KernelRecipe>> recipes,
                   std::shared_ptr<const DispatchPolicy> dispatch_policy, std::shared_ptr<ExecutableProvider> provider)
  : schema_(std::move(schema))
  , recipes_(std::move(recipes))
  , dispatch_policy_(std::move(dispatch_policy))
  , provider_(std::move(provider)) {
}

Operator::Operator(KernelDef definition, std::vector<std::shared_ptr<const KernelRecipe>> recipes,
                   std::shared_ptr<const DispatchPolicy> dispatch_policy, std::shared_ptr<ExecutableProvider> provider)
  : schema_(logical_schema(definition))
  , kernel_definition_(std::make_shared<KernelDef>(std::move(definition)))
  , recipes_(std::move(recipes))
  , dispatch_policy_(std::move(dispatch_policy))
  , provider_(std::move(provider)) {
}

Result<KernelCall> Operator::normalize(const KernelCall& call) const {
  if (kernel_definition_ == nullptr) {
    auto status = schema_.validate(call.arguments());
    if (!status.ok())
      return status;
    return call;
  }
  auto binding = bind_kernel_call(*kernel_definition_, call);
  if (!binding)
    return binding.status();
  auto status = schema_.validate(binding.value().arguments);
  if (!status.ok())
    return status;
  return KernelCall(binding.value().arguments, binding.value().values);
}

Result<std::shared_ptr<Executable>> Operator::resolve(const ArgumentMetadata& arguments) const {
  return resolve(KernelCall(arguments));
}

Result<std::shared_ptr<Executable>> Operator::resolve(const KernelCall& call) const {
  auto normalized = normalize(call);
  if (!normalized)
    return normalized.status();
  return resolve_normalized(normalized.value());
}

Result<std::shared_ptr<Executable>> Operator::resolve_normalized(const KernelCall& call) const {
  if (dispatch_policy_ == nullptr || provider_ == nullptr)
    return Status(StatusCode::InternalError, "operator has no dispatch policy or executable provider");
  auto candidates = dispatch_policy_->candidates(call, recipes_);
  if (!candidates)
    return candidates.status();
  Status last_status(StatusCode::NotFound, "no kernel recipe accepted this invocation");
  for (const auto index : candidates.value()) {
    if (index >= recipes_.size() || recipes_[index] == nullptr)
      return Status(StatusCode::InternalError, "dispatch policy returned an invalid recipe index");
    const auto& recipe = recipes_[index];
    auto status = recipe->match(call);
    if (!status.ok()) {
      if (status.code() == StatusCode::NotApplicable) {
        last_status = status;
        continue;
      }
      return status;
    }
    auto bound = recipe->bind(call);
    if (!bound) {
      if (bound.status().code() == StatusCode::NotApplicable) {
        last_status = bound.status();
        continue;
      }
      return bound.status();
    }
    if (bound.value().recipe_id.empty())
      bound.value().recipe_id = std::string(recipe->id());
    auto executable = provider_->resolve(bound.value());
    if (executable) {
      if (executable.value() == nullptr)
        return Status(StatusCode::InternalError, "executable provider returned a null executable");
      if (executable.value()->operator_name() != schema_.name())
        return Status(StatusCode::AbiMismatch, "resolved executable belongs to a different operator");
      if (executable.value()->specialization_key() != bound.value().specialization_key)
        return Status(StatusCode::AbiMismatch, "resolved executable has a different specialization key");
      return executable;
    }
    if (executable.status().code() == StatusCode::NotFound || executable.status().code() == StatusCode::NotApplicable) {
      last_status = executable.status();
      continue;
    }
    return executable.status();
  }
  return last_status;
}

Status Operator::invoke(const ArgumentMetadata& arguments, void* workspace, std::uint64_t workspace_size,
                        const VecopsExecutionContext* context) const {
  auto normalized = normalize(KernelCall(arguments));
  if (!normalized)
    return normalized.status();
  auto executable = resolve_normalized(normalized.value());
  if (!executable)
    return executable.status();
  return executable.value()->invoke(normalized.value().arguments(), workspace, workspace_size, context);
}

Status Operator::invoke(const KernelCall& call, const VecopsExecutionContext* context, void* workspace,
                        std::uint64_t workspace_size) const {
  auto normalized = normalize(call);
  if (!normalized)
    return normalized.status();
  auto executable = resolve_normalized(normalized.value());
  if (!executable)
    return executable.status();
  return executable.value()->invoke(normalized.value().arguments(), workspace, workspace_size, context);
}

} // namespace vecops::runtime
