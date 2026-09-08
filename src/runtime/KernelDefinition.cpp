/**
 * @file KernelDefinition.cpp
 * @brief Two-phase declarative kernel-call inference and validation.
 *
 * Phase one observes every named specialization and runtime Dynamic value.
 * Phase two resolves Dynamic constraint expressions from the complete
 * environment and validates the call. Keeping these phases separate permits
 * Dynamic bounds to reference a Const value inferred from another tensor
 * without making parameter order a semantic constraint.
 */

#include "vecops/runtime/KernelDefinition.h"

#include <bit>
#include <cctype>
#include <iomanip>
#include <set>
#include <sstream>

namespace vecops::runtime {
namespace {

Status invalid(std::string message) {
  return Status(StatusCode::InvalidArgument, std::move(message));
}

bool identifier(std::string_view name) {
  if (name.empty() || !(std::isalpha(static_cast<unsigned char>(name.front())) || name.front() == '_'))
    return false;
  for (const unsigned char character : name.substr(1))
    if (!(std::isalnum(character) || character == '_'))
      return false;
  static const std::set<std::string_view> keywords{"alignas",   "alignof",  "and",      "asm",
                                                   "auto",      "bool",     "break",    "case",
                                                   "catch",     "char",     "class",    "const",
                                                   "constexpr", "continue", "default",  "delete",
                                                   "do",        "double",   "else",     "enum",
                                                   "explicit",  "export",   "extern",   "false",
                                                   "float",     "for",      "friend",   "goto",
                                                   "if",        "inline",   "int",      "long",
                                                   "namespace", "new",      "noexcept", "not",
                                                   "nullptr",   "operator", "or",       "private",
                                                   "protected", "public",   "register", "reinterpret_cast",
                                                   "requires",  "return",   "short",    "signed",
                                                   "sizeof",    "static",   "struct",   "switch",
                                                   "template",  "this",     "throw",    "true",
                                                   "try",       "typedef",  "typename", "union",
                                                   "unsigned",  "using",    "virtual",  "void",
                                                   "volatile",  "while"};
  return !keywords.contains(name);
}

bool valid_dtype(DType dtype) {
  return dtype != DType::Invalid && dtype_bit(dtype) != 0;
}

std::string dtype_name(DType dtype) {
  switch (dtype) {
  case DType::Bool:
    return "bool";
  case DType::Int8:
    return "i8";
  case DType::UInt8:
    return "u8";
  case DType::Int16:
    return "i16";
  case DType::UInt16:
    return "u16";
  case DType::Int32:
    return "i32";
  case DType::UInt32:
    return "u32";
  case DType::Int64:
    return "i64";
  case DType::UInt64:
    return "u64";
  case DType::Float16:
    return "f16";
  case DType::BFloat16:
    return "bf16";
  case DType::Float32:
    return "f32";
  case DType::Float64:
    return "f64";
  case DType::Invalid:
    return "invalid";
  }
  return "invalid";
}

std::string value_text(const SpecializationValue& value) {
  if (const auto* integer = std::get_if<std::int64_t>(&value))
    return std::to_string(*integer);
  return dtype_name(std::get<DType>(value));
}

Status bind_named(const KernelDef& definition, SpecializationValues& values, std::string_view name,
                  SpecializationValue observed, std::string_view location) {
  const auto declared = definition.values().find(name);
  if (declared == definition.values().end())
    return invalid(std::string(location) + " references undeclared specialization value '" + std::string(name) + "'");
  const bool correct_type = declared->second == SpecializationType::ConstInt
                              ? std::holds_alternative<std::int64_t>(observed)
                              : std::holds_alternative<DType>(observed);
  if (!correct_type)
    return invalid("specialization value '" + std::string(name) + "' has the wrong type");
  const auto existing = values.find(name);
  if (existing == values.end()) {
    values.emplace(std::string(name), observed);
    return Status::success();
  }
  if (existing->second != observed)
    return invalid("specialization value '" + std::string(name) + "' is " + value_text(existing->second) + ", but " +
                   std::string(location) + " observed " + value_text(observed));
  return Status::success();
}

Status validate_expr(const KernelDef& definition, const ConstExpr& expression, std::string_view location) {
  if (!expression.is_symbol())
    return Status::success();
  const auto found = definition.values().find(expression.symbol());
  if (found == definition.values().end() || found->second != SpecializationType::ConstInt)
    return invalid(std::string(location) + " references undeclared or non-ConstInt symbol '" +
                   std::string(expression.symbol()) + "'");
  return Status::success();
}

Result<std::int64_t> resolve_expr(const ConstExpr& expression, const SpecializationValues& values,
                                  std::string_view location) {
  if (!expression.is_symbol())
    return expression.fixed_value();
  const auto found = values.find(expression.symbol());
  if (found == values.end())
    return invalid(std::string(location) + " requires unresolved symbol '" + std::string(expression.symbol()) + "'");
  if (const auto* value = std::get_if<std::int64_t>(&found->second))
    return *value;
  return invalid(std::string(location) + " requires an integer symbol");
}

Status observe_dimension(const KernelDef& definition, const DimensionDef& dimension, std::int64_t observed,
                         SpecializationValues& values, std::vector<SpecializationValue>& anonymous,
                         std::map<std::string, std::int64_t, std::less<>>& dynamic_values, std::string_view location) {
  if (dimension.kind() == DimensionDef::Kind::Symbol)
    return bind_named(definition, values, dimension.symbol(), observed, location);
  if (dimension.kind() == DimensionDef::Kind::Const)
    anonymous.emplace_back(observed);
  if (dimension.kind() == DimensionDef::Kind::Dynamic && !dimension.symbol().empty()) {
    const auto [position, inserted] = dynamic_values.emplace(dimension.symbol(), observed);
    if (!inserted && position->second != observed)
      return invalid("runtime dimension '" + std::string(dimension.symbol()) + "' is " +
                     std::to_string(position->second) + ", but " + std::string(location) + " observed " +
                     std::to_string(observed));
  }
  return Status::success();
}

Status validate_dimension(const DimensionDef& dimension, std::int64_t observed, const SpecializationValues& values,
                          std::string_view location) {
  if (dimension.kind() == DimensionDef::Kind::Fixed) {
    if (observed != dimension.fixed_value())
      return invalid(std::string(location) + " must equal " + std::to_string(dimension.fixed_value()));
    return Status::success();
  }
  if (dimension.kind() != DimensionDef::Kind::Dynamic)
    return Status::success();

  auto alignment = resolve_expr(dimension.alignment(), values, std::string(location) + " alignment");
  if (!alignment)
    return alignment.status();
  auto lower = resolve_expr(dimension.lower_bound(), values, std::string(location) + " lower bound");
  if (!lower)
    return lower.status();
  auto upper = resolve_expr(dimension.upper_bound(), values, std::string(location) + " upper bound");
  if (!upper)
    return upper.status();
  if (alignment.value() <= 0 || !std::has_single_bit(static_cast<std::uint64_t>(alignment.value())))
    return invalid(std::string(location) + " alignment must be a positive power of two");
  if (lower.value() > upper.value())
    return invalid(std::string(location) + " lower bound exceeds its upper bound");
  if (observed < lower.value() || observed > upper.value() || observed % alignment.value() != 0)
    return invalid(std::string(location) + " violates Dynamic<" + std::to_string(alignment.value()) + ", " +
                   std::to_string(lower.value()) + ", " + std::to_string(upper.value()) + ">");
  return Status::success();
}

} // namespace

DimensionDef::DimensionDef(std::string symbol)
  : kind_(Kind::Symbol)
  , symbol_(std::move(symbol)) {
}
DimensionDef::DimensionDef(const char* symbol)
  : DimensionDef(std::string(symbol == nullptr ? "" : symbol)) {
}

DimensionDef DimensionDef::constant() {
  return DimensionDef(Kind::Const);
}

DimensionDef DimensionDef::dynamic(ConstExpr alignment, ConstExpr lower_bound, ConstExpr upper_bound) {
  DimensionDef result(Kind::Dynamic);
  result.alignment_ = std::move(alignment);
  result.lower_bound_ = std::move(lower_bound);
  result.upper_bound_ = std::move(upper_bound);
  return result;
}

DimensionDef DimensionDef::named_dynamic(std::string symbol, ConstExpr alignment, ConstExpr lower_bound,
                                         ConstExpr upper_bound) {
  DimensionDef result = dynamic(std::move(alignment), std::move(lower_bound), std::move(upper_bound));
  result.symbol_ = std::move(symbol);
  return result;
}

DimensionDef DimensionDef::any() {
  return dynamic();
}

KernelDef::KernelDef(std::string name, std::vector<KernelParameterDef> parameters,
                     std::map<std::string, SpecializationType, std::less<>> values)
  : name_(std::move(name))
  , parameters_(std::move(parameters))
  , values_(std::move(values)) {
}

Status KernelDef::validate() const {
  if (name_.empty())
    return invalid("kernel definition name must not be empty");
  std::set<std::string, std::less<>> parameter_names;
  std::map<std::string, std::string, std::less<>> dynamic_symbols;
  for (const auto& parameter : parameters_) {
    const auto& parameter_name =
      std::visit([](const auto& item) -> const std::string& { return item.name; }, parameter);
    if (parameter_name.empty())
      return invalid("kernel parameter names must not be empty");
    if (!parameter_names.insert(parameter_name).second)
      return invalid("duplicate kernel parameter '" + parameter_name + "'");
    if (const auto* tensor = std::get_if<TensorDef>(&parameter)) {
      if (tensor->shape.size() != tensor->strides.size())
        return invalid("tensor '" + tensor->name + "' shape and stride ranks differ");
      if (tensor->optional && tensor->writable())
        return invalid("writable tensor '" + tensor->name + "' cannot be optional");
      if (tensor->dtype.dtype() && !valid_dtype(*tensor->dtype.dtype()))
        return invalid("tensor '" + tensor->name + "' has an invalid fixed dtype");
      if (!tensor->dtype.symbol().empty()) {
        const auto found = values_.find(tensor->dtype.symbol());
        if (found == values_.end() || found->second != SpecializationType::DType)
          return invalid("tensor '" + tensor->name + "' references undeclared or non-dtype symbol '" +
                         std::string(tensor->dtype.symbol()) + "'");
      }
      auto check_dimensions = [&](std::span<const DimensionDef> dimensions, std::string_view part) -> Status {
        for (std::size_t axis = 0; axis < dimensions.size(); ++axis) {
          const auto& dimension = dimensions[axis];
          const auto location = "tensor '" + tensor->name + "' " + std::string(part) + "[" + std::to_string(axis) + "]";
          if (dimension.kind() == DimensionDef::Kind::Symbol) {
            const auto found = values_.find(dimension.symbol());
            if (found == values_.end() || found->second != SpecializationType::ConstInt)
              return invalid(location + " references undeclared or non-ConstInt symbol '" +
                             std::string(dimension.symbol()) + "'");
          } else if (dimension.kind() == DimensionDef::Kind::Dynamic) {
            if (auto status = validate_expr(*this, dimension.alignment(), location + " alignment"); !status.ok())
              return status;
            if (auto status = validate_expr(*this, dimension.lower_bound(), location + " lower bound"); !status.ok())
              return status;
            if (auto status = validate_expr(*this, dimension.upper_bound(), location + " upper bound"); !status.ok())
              return status;
            if (!dimension.alignment().is_symbol() && !dimension.lower_bound().is_symbol() &&
                !dimension.upper_bound().is_symbol()) {
              const auto alignment = dimension.alignment().fixed_value();
              if (alignment <= 0 || !std::has_single_bit(static_cast<std::uint64_t>(alignment)))
                return invalid(location + " alignment must be a positive power of two");
              if (dimension.lower_bound().fixed_value() > dimension.upper_bound().fixed_value())
                return invalid(location + " lower bound exceeds its upper bound");
            }
            if (!dimension.symbol().empty()) {
              if (!identifier(dimension.symbol()))
                return invalid(location + " has invalid runtime dimension symbol '" + std::string(dimension.symbol()) +
                               "'");
              if (values_.contains(dimension.symbol()))
                return invalid(location + " runtime dimension symbol '" + std::string(dimension.symbol()) +
                               "' conflicts with a specialization value");
              const auto expression_text = [](const ConstExpr& value) {
                return value.is_symbol() ? "@" + std::string(value.symbol())
                                         : "#" + std::to_string(value.fixed_value());
              };
              const auto declaration = expression_text(dimension.alignment()) + "," +
                                       expression_text(dimension.lower_bound()) + "," +
                                       expression_text(dimension.upper_bound());
              const auto [position, inserted] = dynamic_symbols.emplace(std::string(dimension.symbol()), declaration);
              if (!inserted && position->second != declaration)
                return invalid("runtime dimension symbol '" + std::string(dimension.symbol()) +
                               "' has conflicting Dynamic declarations");
            }
          }
        }
        return Status::success();
      };
      if (auto status = check_dimensions(tensor->shape, "shape"); !status.ok())
        return status;
      if (auto status = check_dimensions(tensor->strides, "stride"); !status.ok())
        return status;
    } else {
      const auto& scalar = std::get<ValueDef>(parameter);
      if (!valid_dtype(scalar.dtype))
        return invalid("scalar '" + scalar.name + "' has an invalid dtype");
      if (scalar.default_value && scalar.default_value->dtype != scalar.dtype)
        return invalid("scalar '" + scalar.name + "' default dtype does not match its declaration");
    }
  }
  for (const auto& [name, type] : values_) {
    (void)type;
    if (name.empty())
      return invalid("specialization value names must not be empty");
    if (!identifier(name))
      return invalid("specialization value '" + name + "' is not a valid C++ identifier");
  }
  return Status::success();
}

std::string KernelDef::canonical() const {
  auto expression = [](const ConstExpr& value) {
    return value.is_symbol() ? "@" + std::string(value.symbol()) : "#" + std::to_string(value.fixed_value());
  };
  auto dimension = [&](const DimensionDef& value) {
    switch (value.kind()) {
    case DimensionDef::Kind::Fixed:
      return "F" + std::to_string(value.fixed_value());
    case DimensionDef::Kind::Symbol:
      return "S" + std::string(value.symbol());
    case DimensionDef::Kind::Const:
      return std::string("C");
    case DimensionDef::Kind::Dynamic:
      return "D" + (value.symbol().empty() ? std::string{} : "@" + std::string(value.symbol())) + "(" +
             expression(value.alignment()) + "," + expression(value.lower_bound()) + "," +
             expression(value.upper_bound()) + ")";
    }
    return std::string("?");
  };
  std::ostringstream output;
  output << "name:" << name_;
  for (const auto& [name, type] : values_)
    output << ";value:" << name << ':' << (type == SpecializationType::ConstInt ? 'i' : 't');
  for (const auto& parameter : parameters_) {
    if (const auto* tensor = std::get_if<TensorDef>(&parameter)) {
      output << ";tensor:" << tensor->name << ":optional=" << tensor->optional
             << ":access=" << static_cast<int>(tensor->access) << ":device=" << tensor->device_type << ':'
             << tensor->device_index << ":dtype=";
      if (tensor->dtype.dtype())
        output << static_cast<std::uint32_t>(*tensor->dtype.dtype());
      else if (!tensor->dtype.symbol().empty())
        output << '@' << tensor->dtype.symbol();
      else
        output << '?';
      output << ":shape=";
      for (const auto& item : tensor->shape)
        output << dimension(item) << ',';
      output << ":strides=";
      for (const auto& item : tensor->strides)
        output << dimension(item) << ',';
    } else {
      const auto& scalar = std::get<ValueDef>(parameter);
      output << ";scalar:" << scalar.name << ":dtype=" << static_cast<std::uint32_t>(scalar.dtype) << ":default=";
      if (!scalar.default_value) {
        output << '?';
      } else {
        switch (scalar.dtype) {
        case DType::Bool:
        case DType::UInt8:
        case DType::UInt16:
        case DType::UInt32:
        case DType::UInt64:
          output << scalar.default_value->value.u64;
          break;
        case DType::Int8:
        case DType::Int16:
        case DType::Int32:
        case DType::Int64:
          output << scalar.default_value->value.i64;
          break;
        case DType::Float16:
        case DType::BFloat16:
        case DType::Float32:
        case DType::Float64:
          output << std::hexfloat << scalar.default_value->value.f64 << std::defaultfloat;
          break;
        case DType::Invalid:
          output << '?';
          break;
        }
      }
    }
  }
  return output.str();
}

Result<BoundKernel> bind_kernel_call(const KernelDef& definition, const KernelCall& call) {
  if (auto status = definition.validate(); !status.ok())
    return status;
  SpecializationValues values = call.values();
  for (const auto& [name, value] : values) {
    const auto declared = definition.values().find(name);
    if (declared == definition.values().end())
      return invalid("unknown specialization value '" + name + "'");
    const bool integer = std::holds_alternative<std::int64_t>(value);
    if ((declared->second == SpecializationType::ConstInt) != integer)
      return invalid("specialization value '" + name + "' has the wrong type");
    if (const auto* dtype = std::get_if<DType>(&value); dtype && !valid_dtype(*dtype))
      return invalid("specialization value '" + name + "' has an invalid dtype");
  }

  std::vector<Value> normalized(call.arguments().values().begin(), call.arguments().values().end());
  if (normalized.size() > definition.parameters().size())
    return invalid("kernel '" + definition.name() + "' received too many arguments");
  while (normalized.size() < definition.parameters().size()) {
    const auto& parameter = definition.parameters()[normalized.size()];
    if (const auto* tensor = std::get_if<TensorDef>(&parameter); tensor && tensor->optional)
      normalized.emplace_back(std::monostate{});
    else if (const auto* scalar = std::get_if<ValueDef>(&parameter); scalar && scalar->default_value)
      normalized.emplace_back(*scalar->default_value);
    else
      return invalid("kernel '" + definition.name() + "' is missing argument '" +
                     std::visit([](const auto& item) { return item.name; }, parameter) + "'");
  }

  // Phase one: infer all named symbols before resolving Dynamic expressions.
  std::vector<SpecializationValue> anonymous;
  std::map<std::string, std::int64_t, std::less<>> dynamic_values;
  for (std::size_t index = 0; index < definition.parameters().size(); ++index) {
    const auto* tensor_def = std::get_if<TensorDef>(&definition.parameters()[index]);
    if (tensor_def == nullptr)
      continue;
    const auto& argument = normalized[index];
    if (std::holds_alternative<std::monostate>(argument)) {
      if (!tensor_def->optional)
        return invalid("tensor '" + tensor_def->name + "' is not optional");
      anonymous.emplace_back(std::int64_t{0});
      continue;
    }
    const auto* tensor = std::get_if<TensorView>(&argument);
    if (tensor == nullptr)
      return invalid("parameter '" + tensor_def->name + "' must be a tensor");
    if (tensor_def->optional)
      anonymous.emplace_back(std::int64_t{1});
    if (tensor->sizes.size() != tensor->strides.size() || tensor->sizes.size() != tensor_def->shape.size())
      return invalid("tensor '" + tensor_def->name + "' has the wrong rank");
    if (!tensor_def->dtype.symbol().empty()) {
      auto status = bind_named(definition, values, tensor_def->dtype.symbol(), tensor->dtype,
                               "tensor '" + tensor_def->name + "' dtype");
      if (!status.ok())
        return status;
    } else if (tensor_def->dtype.anonymous()) {
      anonymous.emplace_back(tensor->dtype);
    }
    for (std::size_t axis = 0; axis < tensor->sizes.size(); ++axis) {
      auto status =
        observe_dimension(definition, tensor_def->shape[axis], tensor->sizes[axis], values, anonymous, dynamic_values,
                          "tensor '" + tensor_def->name + "' shape[" + std::to_string(axis) + "]");
      if (!status.ok())
        return status;
      status =
        observe_dimension(definition, tensor_def->strides[axis], tensor->strides[axis], values, anonymous,
                          dynamic_values, "tensor '" + tensor_def->name + "' stride[" + std::to_string(axis) + "]");
      if (!status.ok())
        return status;
    }
  }

  for (const auto& [name, type] : definition.values()) {
    (void)type;
    if (!values.contains(name))
      return invalid("specialization value '" + name + "' cannot be inferred and must be supplied");
  }

  // Phase two: validate metadata using the complete symbol environment.
  for (std::size_t index = 0; index < definition.parameters().size(); ++index) {
    const auto& parameter = definition.parameters()[index];
    const auto& argument = normalized[index];
    if (const auto* tensor_def = std::get_if<TensorDef>(&parameter)) {
      if (std::holds_alternative<std::monostate>(argument))
        continue;
      const auto& tensor = std::get<TensorView>(argument);
      if (!valid_dtype(tensor.dtype))
        return invalid("tensor '" + tensor_def->name + "' has an invalid dtype");
      if (tensor_def->device_type != 0 && tensor.device_type != tensor_def->device_type)
        return invalid("tensor '" + tensor_def->name + "' is on the wrong device type");
      if (tensor_def->device_index >= 0 && tensor.device_index != tensor_def->device_index)
        return invalid("tensor '" + tensor_def->name + "' is on the wrong device index");
      const auto required_flags = tensor_def->readable() && tensor_def->writable()
                                    ? VECOPS_TENSOR_READ | VECOPS_TENSOR_WRITE
                                  : tensor_def->writable() ? VECOPS_TENSOR_WRITE
                                                           : VECOPS_TENSOR_READ;
      if ((tensor.flags & required_flags) != required_flags)
        return invalid("tensor '" + tensor_def->name + "' has incompatible access flags");
      for (const auto size : tensor.sizes) {
        if (size < 0)
          return invalid("tensor '" + tensor_def->name + "' has a negative size");
      }
      if (tensor_def->dtype.dtype() && tensor.dtype != *tensor_def->dtype.dtype())
        return invalid("tensor '" + tensor_def->name + "' has the wrong dtype");
      for (std::size_t axis = 0; axis < tensor.sizes.size(); ++axis) {
        auto status = validate_dimension(tensor_def->shape[axis], tensor.sizes[axis], values,
                                         "tensor '" + tensor_def->name + "' shape[" + std::to_string(axis) + "]");
        if (!status.ok())
          return status;
        status = validate_dimension(tensor_def->strides[axis], tensor.strides[axis], values,
                                    "tensor '" + tensor_def->name + "' stride[" + std::to_string(axis) + "]");
        if (!status.ok())
          return status;
      }
    } else {
      const auto& scalar_def = std::get<ValueDef>(parameter);
      const auto* scalar = std::get_if<Scalar>(&argument);
      if (scalar == nullptr)
        return invalid("parameter '" + scalar_def.name + "' must be a scalar");
      if (scalar->dtype != scalar_def.dtype)
        return invalid("scalar '" + scalar_def.name + "' has the wrong dtype");
    }
  }

  std::ostringstream key;
  key << definition.name();
  for (const auto& [name, value] : values)
    key << ';' << name << '=' << value_text(value);
  for (std::size_t index = 0; index < anonymous.size(); ++index)
    key << ";$" << index << '=' << value_text(anonymous[index]);
  return BoundKernel{
    .arguments = ArgumentMetadata(std::move(normalized)),
    .values = std::move(values),
    .anonymous_values = std::move(anonymous),
    .specialization_key = std::move(key).str(),
  };
}

} // namespace vecops::runtime
