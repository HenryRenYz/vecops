// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
/**
 * @file Schema.h
 * @brief Framework-neutral logical operator schemas and argument validation.
 *
 * Schemas own parameter names and constraints but never tensor storage.
 * Validation is synchronous, read-only with respect to tensor memory, and
 * checks metadata only.  Callers are responsible for keeping argument metadata
 * and its external tensor storage alive throughout any later invocation.
 */
#ifndef VECOPS_RUNTIME_SCHEMA_H
#define VECOPS_RUNTIME_SCHEMA_H

#include <cstdint>
#include <functional>
#include <optional>
#include <span>
#include <string>
#include <variant>
#include <vector>

#include "vecops/runtime/Argument.h"
#include "vecops/runtime/Status.h"

namespace vecops::runtime {

/**
 * @brief Rule constraining one tensor extent or stride.
 *
 * Default construction accepts every value.  `dynamic` rules use inclusive
 * bounds and require divisibility by a positive alignment.  The type has value
 * semantics and is safe for concurrent immutable access.
 */
struct DimensionConstraint {
  /** @brief Constraint interpretation. */
  enum class Kind { Any, Const, Dynamic };

  /** @brief Selected rule kind. */
  Kind kind = Kind::Any;
  /** @brief Exact value used by `Kind::Const`. */
  std::int64_t value = 0;
  /** @brief Inclusive dynamic lower bound. */
  std::int64_t lower_bound = 0;
  /** @brief Inclusive dynamic upper bound. */
  std::int64_t upper_bound = INT64_MAX;
  /** @brief Positive dynamic divisibility requirement. */
  std::int64_t alignment = 1;

  /** @brief Create an unconstrained dimension rule. */
  static DimensionConstraint any();
  /**
   * @brief Create a rule accepting only @p value.
   * @param value Required dimension value.
   */
  static DimensionConstraint constant(std::int64_t value);
  /**
   * @brief Create a bounded, aligned dynamic rule.
   * @param lower_bound Inclusive accepted minimum.
   * @param upper_bound Inclusive accepted maximum.
   * @param alignment Required positive divisor.
   * @return Dynamic rule; invalid bounds or non-positive alignment are rejected
   * by validation rather than normalized.
   */
  static DimensionConstraint dynamic(std::int64_t lower_bound = 0, std::int64_t upper_bound = INT64_MAX,
                                     std::int64_t alignment = 1);
  /**
   * @brief Test whether @p candidate satisfies this rule.
   * @param candidate Size or stride value to test.
   * @return True when the selected rule accepts it.
   *
   * Pure and thread-safe for immutable constraints.
   */
  [[nodiscard]] bool accepts(std::int64_t candidate) const;
};

/**
 * @brief Metadata requirements for one tensor parameter.
 *
 * Empty `sizes` and `strides` leave the corresponding axes unconstrained;
 * populated vectors are interpreted positionally.  `device_type == 0` and
 * `device_index == -1` are wildcards.  This object owns only its constraint
 * vectors and has no relationship to a tensor allocation.
 */
struct TensorConstraint {
  /** @brief Bit mask of accepted `DType` values. */
  std::uint64_t dtype_mask = all_dtypes();
  /** @brief Required device type, or zero for any. */
  std::uint32_t device_type = 0;
  /** @brief Required device ordinal, or -1 for any. */
  std::int32_t device_index = -1;
  /** @brief Required rank, absent for any rank. */
  std::optional<std::uint32_t> rank;
  /** @brief Positional shape constraints when provided. */
  std::vector<DimensionConstraint> sizes;
  /** @brief Positional stride constraints when provided. */
  std::vector<DimensionConstraint> strides;
  /** @brief Required `VecopsTensorFlags` access bits. */
  std::uint64_t required_flags = VECOPS_TENSOR_READ;
  /** @brief Whether a `std::monostate` may occupy this parameter position. */
  bool optional = false;
};

/** @brief Metadata requirements for one scalar parameter. */
struct ScalarConstraint {
  /** @brief Bit mask of accepted scalar `DType` values. */
  std::uint64_t dtype_mask = all_dtypes();
};

/**
 * @brief Named positional parameter in an `OperatorSchema`.
 *
 * `constraint` determines whether the corresponding runtime value must be a
 * tensor or scalar.  `name` is used in diagnostics and is copied by the owning
 * schema.
 */
struct ParameterSchema {
  /** @brief Human-readable parameter name. */
  std::string name;
  /** @brief Tensor or scalar eligibility rule. */
  std::variant<TensorConstraint, ScalarConstraint> constraint;
};

/**
 * @brief Logical, framework-neutral operator schema.
 *
 * The schema owns its name, parameter vector, and optional relation validator.
 * The validator observes type-erased values after per-parameter checks pass;
 * it must not retain references or mutate tensor storage.  Immutable schemas
 * can be validated concurrently as long as the supplied arguments are not
 * concurrently mutated.
 */
class OperatorSchema {
public:
  /**
   * @brief Cross-parameter validation callback.
   *
   * The span is valid only for the duration of `validate`; returning status is
   * the only supported communication channel.  The callback must be fast,
   * synchronous, side-effect free, and safe for any concurrency promised by
   * the schema's owner.
   */
  using RelationValidator = std::function<Status(std::span<const Value>)>;

  /**
   * @brief Construct an owning schema.
   * @param name Logical operator name used for diagnostics and dispatch.
   * @param parameters Ordered tensor/scalar parameter definitions.
   * @param relation_validator Optional cross-parameter check run after local validation.
   */
  OperatorSchema(std::string name, std::vector<ParameterSchema> parameters, RelationValidator relation_validator = {});

  /** @brief Return the owned logical operator name. */
  [[nodiscard]] const std::string& name() const {
    return name_;
  }
  /** @brief Return ordered immutable parameter definitions. */
  [[nodiscard]] std::span<const ParameterSchema> parameters() const {
    return parameters_;
  }
  /**
   * @brief Validate ordered arguments against this schema.
   * @param arguments Metadata to inspect without accessing tensor elements.
   * @return Success or an explanatory invalid-argument status.
   *
   * Does not invoke a kernel or change metadata/tensor storage.  The relation
   * validator, if any, runs synchronously after all local checks succeed.
   */
  [[nodiscard]] Status validate(const ArgumentMetadata& arguments) const;

private:
  std::string name_;
  std::vector<ParameterSchema> parameters_;
  RelationValidator relation_validator_;
};

/**
 * @brief Validate one tensor view against a tensor constraint.
 * @param tensor Non-owning metadata to inspect.
 * @param constraint Eligibility rule to apply.
 * @param parameter_name Name included in diagnostics.
 * @return Success or an invalid-argument status.
 *
 * This is a metadata-only, side-effect-free operation and does not dereference
 * `tensor.data`.
 */
Status validate_tensor(const TensorView& tensor, const TensorConstraint& constraint, std::string_view parameter_name);

} // namespace vecops::runtime

#endif // VECOPS_RUNTIME_SCHEMA_H
