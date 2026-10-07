// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
/**
 * @file Schema.cpp
 * @brief Metadata-only validation for logical operator schemas.
 */

#include "vecops/runtime/Schema.h"

#include <limits>
#include <sstream>

namespace vecops::runtime {
namespace {

Status invalid(std::string message) {
  return Status(StatusCode::InvalidArgument, std::move(message));
}

bool dtype_accepted(DType dtype, std::uint64_t mask) {
  return (dtype_bit(dtype) & mask) != 0;
}

} // namespace

DimensionConstraint DimensionConstraint::any() {
  return {};
}

DimensionConstraint DimensionConstraint::constant(std::int64_t value) {
  return {.kind = Kind::Const, .value = value};
}

DimensionConstraint DimensionConstraint::dynamic(std::int64_t lower_bound, std::int64_t upper_bound,
                                                 std::int64_t alignment) {
  return {
    .kind = Kind::Dynamic,
    .lower_bound = lower_bound,
    .upper_bound = upper_bound,
    .alignment = alignment,
  };
}

bool DimensionConstraint::accepts(std::int64_t candidate) const {
  switch (kind) {
  case Kind::Any:
    return true;
  case Kind::Const:
    return candidate == value;
  case Kind::Dynamic:
    return candidate >= lower_bound && candidate <= upper_bound && alignment > 0 && candidate % alignment == 0;
  }
  return false;
}

Status validate_tensor(const TensorView& tensor, const TensorConstraint& constraint, std::string_view parameter_name) {
  if (!dtype_accepted(tensor.dtype, constraint.dtype_mask)) {
    return invalid(std::string(parameter_name) + " has an unsupported dtype");
  }
  if ((constraint.device_type != 0 && tensor.device_type != constraint.device_type) ||
      (constraint.device_index >= 0 && tensor.device_index != constraint.device_index)) {
    return invalid(std::string(parameter_name) + " is on the wrong device");
  }
  if (tensor.sizes.size() != tensor.strides.size()) {
    return invalid(std::string(parameter_name) + " has different size and stride ranks");
  }
  if (tensor.sizes.size() > std::numeric_limits<std::uint32_t>::max()) {
    return invalid(std::string(parameter_name) + " rank exceeds the ABI limit");
  }
  for (const auto size : tensor.sizes) {
    if (size < 0) {
      return invalid(std::string(parameter_name) + " has a negative size");
    }
  }
  if (constraint.rank && tensor.sizes.size() != *constraint.rank) {
    return invalid(std::string(parameter_name) + " has the wrong rank");
  }
  if (!constraint.sizes.empty() && constraint.sizes.size() != tensor.sizes.size()) {
    return invalid(std::string(parameter_name) + " has no size constraint for its rank");
  }
  if (!constraint.strides.empty() && constraint.strides.size() != tensor.strides.size()) {
    return invalid(std::string(parameter_name) + " has no stride constraint for its rank");
  }
  for (std::size_t index = 0; index < constraint.sizes.size(); ++index) {
    if (!constraint.sizes[index].accepts(tensor.sizes[index])) {
      std::ostringstream stream;
      stream << parameter_name << " size[" << index << "] violates schema";
      return invalid(stream.str());
    }
  }
  for (std::size_t index = 0; index < constraint.strides.size(); ++index) {
    if (!constraint.strides[index].accepts(tensor.strides[index])) {
      std::ostringstream stream;
      stream << parameter_name << " stride[" << index << "] violates schema";
      return invalid(stream.str());
    }
  }
  if ((tensor.flags & constraint.required_flags) != constraint.required_flags) {
    return invalid(std::string(parameter_name) + " does not satisfy the required access flags");
  }
  return Status::success();
}

OperatorSchema::OperatorSchema(std::string name, std::vector<ParameterSchema> parameters,
                               RelationValidator relation_validator)
  : name_(std::move(name))
  , parameters_(std::move(parameters))
  , relation_validator_(std::move(relation_validator)) {
}

Status OperatorSchema::validate(const ArgumentMetadata& arguments) const {
  if (arguments.size() != parameters_.size()) {
    std::ostringstream stream;
    stream << name_ << " expects " << parameters_.size() << " arguments, got " << arguments.size();
    return invalid(stream.str());
  }
  for (std::size_t index = 0; index < parameters_.size(); ++index) {
    const auto& parameter = parameters_[index];
    const auto& value = arguments[index];
    if (const auto* tensor_constraint = std::get_if<TensorConstraint>(&parameter.constraint)) {
      if (std::holds_alternative<std::monostate>(value)) {
        if (!tensor_constraint->optional) {
          return invalid(parameter.name + " is not optional");
        }
        continue;
      }
      const auto* tensor = std::get_if<TensorView>(&value);
      if (tensor == nullptr)
        return invalid(parameter.name + " must be a tensor");
      auto status = validate_tensor(*tensor, *tensor_constraint, parameter.name);
      if (!status.ok())
        return status;
    } else {
      if (const auto* scalar = std::get_if<Scalar>(&value)) {
        const auto& constraint = std::get<ScalarConstraint>(parameter.constraint);
        if (!dtype_accepted(scalar->dtype, constraint.dtype_mask)) {
          return invalid(parameter.name + " has an unsupported scalar dtype");
        }
      } else {
        return invalid(parameter.name + " must be a scalar");
      }
    }
  }
  if (relation_validator_)
    return relation_validator_(arguments.values());
  return Status::success();
}

} // namespace vecops::runtime
