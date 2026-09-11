/**
 * @file KernelCompiler.cpp
 * @brief Generate `vecops::spec` headers and stable ABI adapters for source kernels.
 *
 * This translation unit turns a fully bound KernelDef into a self-contained
 * C++ adapter project. It intentionally consumes `BoundKernel`, never raw
 * user metadata: named and anonymous compile-time values have already been
 * inferred and checked before code generation begins. Generated source owns
 * only its adapter objects; user `__kernel__` controls workspace allocation.
 */

#include "vecops/compiler/Compiler.h"
#include "vecops/runtime/Operator.h"

#include <atomic>
#include <fstream>
#include <iomanip>
#include <map>
#include <set>
#include <sstream>
#include <system_error>

namespace vecops::compiler {
namespace {

namespace fs = std::filesystem;
using runtime::BoundKernel;
using runtime::ConstExpr;
using runtime::DType;
using runtime::DimensionDef;
using runtime::KernelDef;
using runtime::SpecializationType;
using runtime::TensorDef;
using runtime::TensorView;
using runtime::ValueDef;

std::atomic<std::uint64_t> next_attempt{0};

runtime::Status invalid(std::string message) {
  return runtime::Status(runtime::StatusCode::InvalidArgument, std::move(message));
}

std::string cpp_string(std::string_view value) {
  std::ostringstream output;
  output << '"';
  for (const unsigned char character : value) {
    switch (character) {
    case '\\':
      output << "\\\\";
      break;
    case '"':
      output << "\\\"";
      break;
    case '\n':
      output << "\\n";
      break;
    case '\r':
      output << "\\r";
      break;
    case '\t':
      output << "\\t";
      break;
    default:
      if (character < 0x20 || character >= 0x7f)
        output << "\\x" << std::hex << std::setw(2) << std::setfill('0') << static_cast<unsigned>(character)
               << std::dec;
      else
        output << static_cast<char>(character);
    }
  }
  output << '"';
  return output.str();
}

std::string cpp_type(DType dtype) {
  switch (dtype) {
  case DType::Bool:
    return "bool";
  case DType::Int8:
    return "::vecops::int8_t";
  case DType::UInt8:
    return "::vecops::uint8_t";
  case DType::Int16:
    return "::vecops::int16_t";
  case DType::UInt16:
    return "::vecops::uint16_t";
  case DType::Int32:
    return "::vecops::int32_t";
  case DType::UInt32:
    return "::vecops::uint32_t";
  case DType::Int64:
    return "::vecops::int64_t";
  case DType::UInt64:
    return "::vecops::uint64_t";
  case DType::Float16:
    return "::vecops::float16_t";
  case DType::BFloat16:
    return "::vecops::bfloat16_t";
  case DType::Float32:
    return "::vecops::float32_t";
  case DType::Float64:
    return "::vecops::float64_t";
  case DType::Invalid:
    return "void";
  }
  return "void";
}

std::string read_file(const fs::path& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input)
    return {};
  return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}

std::uint64_t hash_text(std::uint64_t hash, std::string_view text) {
  for (const unsigned char byte : text) {
    hash ^= byte;
    hash *= UINT64_C(1099511628211);
  }
  return hash;
}

std::string hex_hash(std::uint64_t value) {
  std::ostringstream output;
  output << std::hex << std::setw(16) << std::setfill('0') << value;
  return output.str();
}

std::int64_t resolve(const ConstExpr& expression, const BoundKernel& binding) {
  if (!expression.is_symbol())
    return expression.fixed_value();
  return std::get<std::int64_t>(binding.values.find(expression.symbol())->second);
}

std::int64_t resolved_dimension(const DimensionDef& definition, std::int64_t observed, const BoundKernel& binding) {
  if (definition.kind() == DimensionDef::Kind::Fixed)
    return definition.fixed_value();
  if (definition.kind() == DimensionDef::Kind::Symbol)
    return std::get<std::int64_t>(binding.values.find(definition.symbol())->second);
  return observed;
}

DType resolved_dtype(const TensorDef& definition, const TensorView& tensor, const BoundKernel& binding) {
  if (definition.dtype.dtype())
    return *definition.dtype.dtype();
  if (!definition.dtype.symbol().empty())
    return std::get<DType>(binding.values.find(definition.dtype.symbol())->second);
  return tensor.dtype;
}

std::string meta_value(const DimensionDef& definition, std::int64_t observed, const BoundKernel& binding,
                       std::string_view runtime_value) {
  if (definition.kind() != DimensionDef::Kind::Dynamic)
    return "::vecops::meta::cint<" + std::to_string(resolved_dimension(definition, observed, binding)) + ">";
  const auto alignment = resolve(definition.alignment(), binding);
  const auto lower = resolve(definition.lower_bound(), binding);
  const auto upper = resolve(definition.upper_bound(), binding);
  if (alignment == 1 && lower == INT64_MIN && upper == INT64_MAX)
    return "::vecops::meta::Any{" + std::to_string(observed) + "}";
  const auto lower_text = lower == INT64_MIN ? "::vecops::meta::kLoInf" : std::to_string(lower);
  const auto upper_text = upper == INT64_MAX ? "::vecops::meta::kHiInf" : std::to_string(upper);
  return "::vecops::meta::Dynamic<" + std::to_string(alignment) + ", " + lower_text + ", " + upper_text + ">{" +
         std::string(runtime_value) + "}";
}

std::string dynamic_meta_type(const DimensionDef& definition, const BoundKernel& binding) {
  const auto alignment = resolve(definition.alignment(), binding);
  const auto lower = resolve(definition.lower_bound(), binding);
  const auto upper = resolve(definition.upper_bound(), binding);
  if (alignment == 1 && lower == INT64_MIN && upper == INT64_MAX)
    return "::vecops::meta::Any";
  const auto lower_text = lower == INT64_MIN ? "::vecops::meta::kLoInf" : std::to_string(lower);
  const auto upper_text = upper == INT64_MAX ? "::vecops::meta::kHiInf" : std::to_string(upper);
  return "::vecops::meta::Dynamic<" + std::to_string(alignment) + ", " + lower_text + ", " + upper_text + ">";
}

std::string generate_spec(const KernelDef& definition, const BoundKernel& binding) {
  std::ostringstream output;
  output << "#pragma once\n#include \"vecops/CoreTypes.h\"\n#include \"vecops/Meta.h\"\n\n"
            "namespace vecops::spec {\n";
  for (const auto& [name, type] : definition.values()) {
    const auto& value = binding.values.at(name);
    if (type == SpecializationType::ConstInt)
      output << "inline constexpr auto " << name << " = ::vecops::meta::cint<" << std::get<std::int64_t>(value)
             << ">;\n";
    else
      output << "using " << name << " = " << cpp_type(std::get<DType>(value)) << ";\n";
  }
  std::set<std::string, std::less<>> emitted_dynamic_symbols;
  for (const auto& parameter : definition.parameters()) {
    const auto* tensor = std::get_if<TensorDef>(&parameter);
    if (tensor == nullptr)
      continue;
    for (const auto& dimension : tensor->shape) {
      if (dimension.kind() == DimensionDef::Kind::Dynamic && !dimension.symbol().empty() &&
          emitted_dynamic_symbols.insert(std::string(dimension.symbol())).second)
        output << "using " << dimension.symbol() << " = " << dynamic_meta_type(dimension, binding) << ";\n";
    }
    for (const auto& dimension : tensor->strides) {
      if (dimension.kind() == DimensionDef::Kind::Dynamic && !dimension.symbol().empty() &&
          emitted_dynamic_symbols.insert(std::string(dimension.symbol())).second)
        output << "using " << dimension.symbol() << " = " << dynamic_meta_type(dimension, binding) << ";\n";
    }
  }
  output << "} // namespace vecops::spec\n";
  return output.str();
}

std::string scalar_value(const ValueDef& definition, std::size_t index) {
  const auto value = "call->values[" + std::to_string(index) + "].value.scalar.value.";
  switch (definition.dtype) {
  case DType::Bool:
    return "(" + value + "u64 != 0)";
  case DType::Int8:
  case DType::Int16:
  case DType::Int32:
  case DType::Int64:
    return "static_cast<" + cpp_type(definition.dtype) + ">(" + value + "i64)";
  case DType::UInt8:
  case DType::UInt16:
  case DType::UInt32:
  case DType::UInt64:
    return "static_cast<" + cpp_type(definition.dtype) + ">(" + value + "u64)";
  case DType::Float16:
  case DType::BFloat16:
  case DType::Float32:
  case DType::Float64:
    return "static_cast<" + cpp_type(definition.dtype) + ">(" + value + "f64)";
  case DType::Invalid:
    return "{}";
  }
  return "{}";
}

void emit_constraints(std::ostringstream& output, std::string_view name, std::span<const DimensionDef> definitions,
                      std::span<const std::int64_t> observed, const BoundKernel& binding) {
  if (definitions.empty())
    return;
  output << "static constexpr VecopsDimensionConstraint " << name << "[] = {\n";
  for (std::size_t axis = 0; axis < definitions.size(); ++axis) {
    const auto& definition = definitions[axis];
    if (definition.kind() == DimensionDef::Kind::Dynamic) {
      const auto lower = resolve(definition.lower_bound(), binding);
      const auto upper = resolve(definition.upper_bound(), binding);
      const auto lower_text = lower == INT64_MIN ? "INT64_MIN" : std::to_string(lower);
      const auto upper_text = upper == INT64_MAX ? "INT64_MAX" : std::to_string(upper);
      output << "  {sizeof(VecopsDimensionConstraint), VECOPS_DIM_DYNAMIC, 0, " << lower_text << ", " << upper_text
             << ", " << resolve(definition.alignment(), binding) << "},\n";
    } else {
      output << "  {sizeof(VecopsDimensionConstraint), VECOPS_DIM_CONST, "
             << resolved_dimension(definition, observed[axis], binding) << ", 0, 0, 1},\n";
    }
  }
  output << "};\n";
}

std::string generate_adapter(const fs::path& kernel_file, const KernelDef& definition, const BoundKernel& binding) {
  std::ostringstream output;
  output << "#include \"vecops_spec.h\"\n"
            "#include \"vecops/runtime/KernelAbi.h\"\n"
            "#include \"vecops/tensor/Tensor.h\"\n"
            "#include <algorithm>\n#include <cstddef>\n#include <cstring>\n#include <exception>\n#include <limits>\n"
            "#include <utility>\n\n"
         << "#include " << cpp_string(fs::absolute(kernel_file).lexically_normal().string())
         << "\n#include \"vecops/execution/WorkspaceContext.h\"\n\n"
            "namespace {\n"
            "void set_error(VecopsError* error, int code, const char* message) {\n"
            "  if (!error || error->struct_size < sizeof(VecopsError)) return;\n"
            "  error->code = code; const auto size = std::strlen(message); error->message_required = size + 1;\n"
            "  if (!error->message || error->message_capacity == 0) return;\n"
            "  const auto copied = std::min(size, error->message_capacity - 1);\n"
            "  std::memcpy(error->message, message, copied); error->message[copied] = '\\0';\n"
            "}\n"
            "template <typename Setup, typename... Args>\n"
            "void invoke_source_kernel(const VecopsCall* call, const char* recipe, Setup&& setup, Args&&... args) {\n"
            "  if constexpr (requires(::vecops::execution::WorkspaceContext& workspace) {\n"
            "                  __kernel__(workspace, std::forward<Args>(args)...); }) {\n"
            "    if (call->context && call->context->struct_size >= sizeof(VecopsExecutionContext) &&\n"
            "        (call->context->flags & VECOPS_EXECUTION_CONTEXT_FLAG_WORKSPACE_CONTEXT) != 0 &&\n"
            "        call->context->user_data) {\n"
            "      auto& workspace = *static_cast<::vecops::execution::WorkspaceContext*>(\n"
            "        call->context->user_data);\n"
            "      auto kernel_scope = workspace.serial_scope(recipe);\n"
            "      std::forward<Setup>(setup)(workspace);\n"
            "      __kernel__(workspace, std::forward<Args>(args)...);\n"
            "      return;\n"
            "    }\n"
            "    if (call->workspace != nullptr || call->workspace_size != 0) {\n"
            "      ::vecops::execution::WorkspaceContext workspace(\n"
            "        recipe, call->workspace, static_cast<::vecops::nint_t>(call->workspace_size));\n"
            "      std::forward<Setup>(setup)(workspace);\n"
            "      __kernel__(workspace, std::forward<Args>(args)...);\n"
            "      return;\n"
            "    }\n"
            "    static thread_local ::vecops::execution::WorkspaceReplayCache replay_cache;\n"
            "    replay_cache.invoke(recipe, setup, [&](auto& prepared_workspace) {\n"
            "      __kernel__(prepared_workspace, std::forward<Args>(args)...);\n"
            "    });\n"
            "  } else {\n"
            "    static_assert(requires { __kernel__(std::forward<Args>(args)...); },\n"
            "                  \"__kernel__ parameters do not match the KernelDef\");\n"
            "    __kernel__(std::forward<Args>(args)...);\n"
            "  }\n"
            "}\n";

  std::size_t tensor_ordinal = 0;
  for (std::size_t index = 0; index < definition.parameters().size(); ++index) {
    const auto* tensor_def = std::get_if<TensorDef>(&definition.parameters()[index]);
    if (tensor_def == nullptr)
      continue;
    if (const auto* tensor = std::get_if<TensorView>(&binding.arguments[index])) {
      emit_constraints(output, "sizes_" + std::to_string(tensor_ordinal), tensor_def->shape, tensor->sizes, binding);
      emit_constraints(output, "strides_" + std::to_string(tensor_ordinal), tensor_def->strides, tensor->strides,
                       binding);
    }
    ++tensor_ordinal;
  }

  output << "static const VecopsParameterDescriptor parameters[] = {\n";
  tensor_ordinal = 0;
  for (std::size_t index = 0; index < definition.parameters().size(); ++index) {
    if (const auto* tensor_def = std::get_if<TensorDef>(&definition.parameters()[index])) {
      const auto* tensor = std::get_if<TensorView>(&binding.arguments[index]);
      if (tensor == nullptr) {
        output << "  {sizeof(VecopsParameterDescriptor), VECOPS_VALUE_TENSOR, " << cpp_string(tensor_def->name)
               << ", {.tensor={sizeof(VecopsTensorParameter), 0, " << tensor_def->device_type << ", "
               << tensor_def->device_index << ", -1, 1, sizeof(VecopsDimensionConstraint), nullptr, nullptr, "
               << (tensor_def->readable() && tensor_def->writable() ? "VECOPS_TENSOR_READ | VECOPS_TENSOR_WRITE"
                   : tensor_def->writable()                         ? "VECOPS_TENSOR_WRITE"
                                                                    : "VECOPS_TENSOR_READ")
               << "}}},\n";
      } else {
        const auto dtype_mask = runtime::dtype_bit(resolved_dtype(*tensor_def, *tensor, binding));
        const auto rank = tensor_def->shape.size();
        const auto size_pointer = rank == 0 ? "nullptr" : "sizes_" + std::to_string(tensor_ordinal);
        const auto stride_pointer = rank == 0 ? "nullptr" : "strides_" + std::to_string(tensor_ordinal);
        output << "  {sizeof(VecopsParameterDescriptor), VECOPS_VALUE_TENSOR, " << cpp_string(tensor_def->name)
               << ", {.tensor={sizeof(VecopsTensorParameter), " << dtype_mask << "ULL, " << tensor_def->device_type
               << ", " << tensor_def->device_index << ", " << rank << ", 0, sizeof(VecopsDimensionConstraint), "
               << size_pointer << ", " << stride_pointer << ", "
               << (tensor_def->readable() && tensor_def->writable() ? "VECOPS_TENSOR_READ | VECOPS_TENSOR_WRITE"
                   : tensor_def->writable()                         ? "VECOPS_TENSOR_WRITE"
                                                                    : "VECOPS_TENSOR_READ")
               << "}}},\n";
      }
      ++tensor_ordinal;
    } else {
      const auto& scalar = std::get<ValueDef>(definition.parameters()[index]);
      output << "  {sizeof(VecopsParameterDescriptor), VECOPS_VALUE_SCALAR, " << cpp_string(scalar.name)
             << ", {.scalar={sizeof(VecopsScalarParameter), " << runtime::dtype_bit(scalar.dtype) << "ULL}}},\n";
    }
  }
  output << "};\n\n"
            "int32_t workspace(const VecopsCall*, uint64_t* size, VecopsError*) {\n"
            "  if (!size) return VECOPS_STATUS_INVALID_ARGUMENT; *size = 0; return VECOPS_STATUS_OK;\n"
            "}\n"
            "int32_t run(const VecopsCall* call, VecopsError* error) {\n"
            "  if (!call || call->struct_size < sizeof(VecopsCall) || call->num_values != "
         << definition.parameters().size()
         << " || !call->values) { set_error(error, VECOPS_STATUS_INVALID_ARGUMENT, \"invalid call frame\"); return "
            "VECOPS_STATUS_INVALID_ARGUMENT; }\n"
            "  try {\n";

  std::map<std::string, std::string, std::less<>> runtime_dimensions;
  for (std::size_t index = 0; index < definition.parameters().size(); ++index) {
    const auto* tensor_def = std::get_if<TensorDef>(&definition.parameters()[index]);
    if (tensor_def == nullptr || !std::holds_alternative<TensorView>(binding.arguments[index]))
      continue;
    auto emit_relations = [&](std::span<const DimensionDef> dimensions, std::string_view member) {
      for (std::size_t axis = 0; axis < dimensions.size(); ++axis) {
        const auto& dimension = dimensions[axis];
        if (dimension.kind() != DimensionDef::Kind::Dynamic || dimension.symbol().empty())
          continue;
        const auto expression = "call->values[" + std::to_string(index) + "].value.tensor." + std::string(member) +
                                "[" + std::to_string(axis) + "]";
        const auto [position, inserted] = runtime_dimensions.emplace(std::string(dimension.symbol()), expression);
        if (!inserted)
          output << "    if (" << expression << " != " << position->second
                 << ") { set_error(error, VECOPS_STATUS_INVALID_ARGUMENT, "
                 << cpp_string("runtime dimension '" + std::string(dimension.symbol()) + "' mismatch")
                 << "); return VECOPS_STATUS_INVALID_ARGUMENT; }\n";
      }
    };
    emit_relations(tensor_def->shape, "sizes");
    emit_relations(tensor_def->strides, "strides");
  }

  std::vector<std::string> axis_observations;
  std::set<std::string, std::less<>> observed_symbols;
  auto integer_text = [](std::int64_t value) {
    if (value == INT64_MIN)
      return std::string{"INT64_MIN"};
    if (value == INT64_MAX)
      return std::string{"INT64_MAX"};
    return std::to_string(value);
  };
  for (std::size_t index = 0; index < definition.parameters().size(); ++index) {
    const auto* tensor_def = std::get_if<TensorDef>(&definition.parameters()[index]);
    if (tensor_def == nullptr || !std::holds_alternative<TensorView>(binding.arguments[index]))
      continue;
    auto observe_dimensions = [&](std::span<const DimensionDef> dimensions, std::string_view member) {
      for (std::size_t axis = 0; axis < dimensions.size(); ++axis) {
        const auto& dimension = dimensions[axis];
        if (dimension.kind() != DimensionDef::Kind::Dynamic)
          continue;
        if (!dimension.symbol().empty() && !observed_symbols.emplace(dimension.symbol()).second)
          continue;
        const auto expression = "call->values[" + std::to_string(index) + "].value.tensor." + std::string(member) +
                                "[" + std::to_string(axis) + "]";
        axis_observations.push_back(
          "      workspace.observe_axis({::vecops::execution::AxisContract::Kind::Dynamic, 0, " +
          std::to_string(resolve(dimension.alignment(), binding)) + ", " +
          integer_text(resolve(dimension.lower_bound(), binding)) + ", " +
          integer_text(resolve(dimension.upper_bound(), binding)) + ", " + expression + ", " +
          integer_text(resolve(dimension.upper_bound(), binding)) + "});\n");
      }
    };
    observe_dimensions(tensor_def->shape, "sizes");
    observe_dimensions(tensor_def->strides, "strides");
  }

  std::vector<std::string> arguments;
  for (std::size_t index = 0; index < definition.parameters().size(); ++index) {
    if (const auto* tensor_def = std::get_if<TensorDef>(&definition.parameters()[index])) {
      const auto* tensor = std::get_if<TensorView>(&binding.arguments[index]);
      if (tensor == nullptr) {
        arguments.emplace_back("::vecops::tensor::nullopt");
        continue;
      }
      const auto local = "arg_" + std::to_string(index);
      const auto view = "view_" + std::to_string(index);
      output << "    const auto& " << view << " = call->values[" << index << "].value.tensor;\n"
             << "    auto " << local << " = ::vecops::tensor::make_tensor(reinterpret_cast<"
             << (tensor_def->writable() ? "" : "const ") << cpp_type(resolved_dtype(*tensor_def, *tensor, binding))
             << "*>(static_cast<std::byte*>(" << view << ".data) + " << view << ".byte_offset),\n"
             << "      ::vecops::tensor::make_shape(";
      for (std::size_t axis = 0; axis < tensor_def->shape.size(); ++axis) {
        if (axis != 0)
          output << ", ";
        output << meta_value(tensor_def->shape[axis], tensor->sizes[axis], binding,
                             view + ".sizes[" + std::to_string(axis) + "]");
      }
      output << "), ::vecops::tensor::make_strides(";
      for (std::size_t axis = 0; axis < tensor_def->strides.size(); ++axis) {
        if (axis != 0)
          output << ", ";
        output << meta_value(tensor_def->strides[axis], tensor->strides[axis], binding,
                             view + ".strides[" + std::to_string(axis) + "]");
      }
      output << "));\n";
      arguments.push_back(local);
    } else {
      arguments.push_back(scalar_value(std::get<ValueDef>(definition.parameters()[index]), index));
    }
  }
  output << "    if (call->workspace_size > static_cast<uint64_t>(std::numeric_limits<::vecops::nint_t>::max())) {\n"
            "      set_error(error, VECOPS_STATUS_INVALID_ARGUMENT, \"workspace exceeds the host address space\");\n"
            "      return VECOPS_STATUS_INVALID_ARGUMENT;\n"
            "    }\n"
         << "    invoke_source_kernel(call, " << cpp_string(definition.name()) << ", [&](auto& workspace) {\n";
  for (const auto& observation : axis_observations)
    output << observation;
  output << "    }";
  for (std::size_t index = 0; index < arguments.size(); ++index) {
    output << ", " << arguments[index];
  }
  output
    << ");\n    return VECOPS_STATUS_OK;\n"
       "  } catch (const std::exception& exception) {\n"
       "    set_error(error, VECOPS_STATUS_EXECUTION_ERROR, exception.what()); return VECOPS_STATUS_EXECUTION_ERROR;\n"
       "  } catch (...) {\n"
       "    set_error(error, VECOPS_STATUS_EXECUTION_ERROR, \"kernel threw an unknown exception\"); return "
       "VECOPS_STATUS_EXECUTION_ERROR;\n"
       "  }\n"
       "}\n"
       "const VecopsKernelDescriptorV1 descriptor = {\n"
       "  sizeof(VecopsKernelDescriptorV1), VECOPS_KERNEL_ABI_VERSION_MAJOR, VECOPS_KERNEL_ABI_VERSION_MINOR,\n  "
    << cpp_string(definition.name()) << ", " << cpp_string(binding.specialization_key) << ", "
    << definition.parameters().size()
    << ", sizeof(VecopsParameterDescriptor), parameters, workspace, run, 0\n};\n"
       "} // namespace\n\n"
       "extern \"C\" VECOPS_RUNTIME_EXPORT const VecopsKernelDescriptorV1* vecops_kernel_query_v1() {\n"
       "  return &descriptor;\n"
       "}\n";
  return output.str();
}

runtime::Status write_file(const fs::path& path, const std::string& contents) {
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  if (!output)
    return invalid("cannot write generated file: " + path.string());
  output << contents;
  output.close();
  if (!output)
    return invalid("failed while writing generated file: " + path.string());
  return runtime::Status::success();
}

} // namespace

runtime::Result<std::shared_ptr<runtime::Executable>> Compiler::compile_kernel(const fs::path& kernel_file,
                                                                               const runtime::KernelDef& definition,
                                                                               const runtime::KernelCall& call) const {
  if (!kernel_config_)
    return invalid("compile_kernel requires Compiler(KernelCompilerConfig)");
  if (!fs::is_regular_file(kernel_file))
    return invalid("kernel source does not exist: " + kernel_file.string());
  if (kernel_config_->work_directory.empty())
    return invalid("KernelCompilerConfig.work_directory is required");

  auto binding = runtime::bind_kernel_call(definition, call);
  if (!binding)
    return binding.status();
  const auto source = read_file(kernel_file);
  std::uint64_t fingerprint = UINT64_C(14695981039346656037);
  fingerprint = hash_text(fingerprint, definition.canonical());
  fingerprint = hash_text(fingerprint, source);
  auto definition_owner = std::make_shared<runtime::KernelDef>(definition);
  auto instantiation = std::make_shared<runtime::BoundKernelRecipe::SourceInstantiation>(
    runtime::BoundKernelRecipe::SourceInstantiation{kernel_file, definition_owner, std::move(binding).value()});
  runtime::BoundKernelRecipe recipe{"direct-source-kernel", instantiation->binding.specialization_key,
                                    "direct-source-kernel;source=" + hex_hash(fingerprint) + ";" +
                                      instantiation->binding.specialization_key,
                                    std::move(instantiation)};
  return compile_kernel(recipe);
}

runtime::Result<std::shared_ptr<runtime::Executable>>
Compiler::compile_kernel(const runtime::BoundKernelRecipe& recipe) const {
  if (recipe.source == nullptr || recipe.source->definition == nullptr)
    return invalid("bound recipe does not carry a source kernel instantiation");
  const std::array recipes{recipe};
  auto results = compile_kernels(recipes);
  return std::move(results.front());
}

std::vector<runtime::Result<std::shared_ptr<runtime::Executable>>>
Compiler::compile_kernels(std::span<const runtime::BoundKernelRecipe> recipes, std::size_t parallel_jobs) const {
  using ExecutableResult = runtime::Result<std::shared_ptr<runtime::Executable>>;
  std::vector<ExecutableResult> results;
  if (recipes.empty())
    return results;

  auto fail_all = [&](runtime::Status status) {
    results.clear();
    results.reserve(recipes.size());
    for (std::size_t index = 0; index < recipes.size(); ++index)
      results.emplace_back(status);
    return results;
  };
  if (!kernel_config_)
    return fail_all(invalid("compile_kernels requires Compiler(KernelCompilerConfig)"));
  if (kernel_config_->work_directory.empty())
    return fail_all(invalid("KernelCompilerConfig.work_directory is required"));

  std::uint64_t batch_fingerprint = UINT64_C(14695981039346656037);
  for (const auto& recipe : recipes) {
    if (recipe.source == nullptr || recipe.source->definition == nullptr)
      return fail_all(invalid("bound recipe does not carry a source kernel instantiation"));
    if (!fs::is_regular_file(recipe.source->kernel_file))
      return fail_all(invalid("kernel source does not exist: " + recipe.source->kernel_file.string()));
    batch_fingerprint = hash_text(batch_fingerprint, recipe.artifact_key);
  }

  std::error_code error;
  fs::create_directories(kernel_config_->work_directory, error);
  if (error)
    return fail_all(invalid("cannot create compiler work directory: " + error.message()));
  fs::path attempt;
  const auto batch_identity = hex_hash(batch_fingerprint);
  for (std::size_t retry = 0; retry < 1024; ++retry) {
    error.clear();
    const auto candidate =
      kernel_config_->work_directory /
      ("kernel-batch-" + batch_identity + "-" + std::to_string(next_attempt.fetch_add(1, std::memory_order_relaxed)));
    if (fs::create_directory(candidate, error)) {
      attempt = candidate;
      break;
    }
    if (error && error != std::errc::file_exists)
      return fail_all(invalid("cannot create compiler batch directory: " + error.message()));
  }
  if (attempt.empty())
    return fail_all(invalid("cannot allocate a unique compiler batch directory"));

  KernelBuildBatchRequest batch;
  batch.generated_source_directory = attempt / "project";
  batch.build_directory = attempt / "build";
  batch.tasks.reserve(recipes.size());
  for (std::size_t index = 0; index < recipes.size(); ++index) {
    const auto& recipe = recipes[index];
    const auto& source = *recipe.source;
    std::uint64_t fingerprint = UINT64_C(14695981039346656037);
    fingerprint = hash_text(fingerprint, source.binding.specialization_key);
    fingerprint = hash_text(fingerprint, source.definition->canonical());
    fingerprint = hash_text(fingerprint, read_file(source.kernel_file));
    const auto identity = hex_hash(fingerprint);

    const auto input_directory = attempt / "tasks" / std::to_string(index) / "input";
    fs::create_directories(input_directory, error);
    if (error)
      return fail_all(invalid("cannot create generated input directory: " + error.message()));
    auto status = write_file(input_directory / "vecops_spec.h", generate_spec(*source.definition, source.binding));
    if (!status.ok())
      return fail_all(std::move(status));
    status = write_file(input_directory / "vecops_adapter.cpp",
                        generate_adapter(source.kernel_file, *source.definition, source.binding));
    if (!status.ok())
      return fail_all(std::move(status));

    KernelBuildRequest request;
    request.target_name = "vecops_kernel_" + identity + "_" + std::to_string(index);
    request.output_name = "vecops_kernel_" + identity;
    request.target_arch = kernel_config_->target_arch;
    request.sdk = kernel_config_->sdk;
    request.toolchain = kernel_config_->toolchain;
    if (parallel_jobs != 0)
      request.toolchain.parallel_jobs = parallel_jobs;
    request.sources = {input_directory / "vecops_adapter.cpp"};
    request.include_directories = kernel_config_->include_directories;
    request.include_directories.push_back(input_directory);
    request.compile_definitions = kernel_config_->compile_definitions;
    request.compile_options = kernel_config_->compile_options;
    request.link_directories = kernel_config_->link_directories;
    request.link_libraries = kernel_config_->link_libraries;
    request.link_options = kernel_config_->link_options;
    request.generated_source_directory = batch.generated_source_directory / "tasks" / std::to_string(index);
    request.build_directory = batch.build_directory / "tasks" / std::to_string(index);
    request.artifact_directory = attempt / "artifacts" / std::to_string(index);
    batch.tasks.push_back(std::move(request));
  }

  auto built = compile_batch(batch);
  results.reserve(recipes.size());
  for (std::size_t index = 0; index < recipes.size(); ++index) {
    if (index >= built.tasks.size() || !built.tasks[index].success || !built.tasks[index].kernel_library) {
      std::string message = "kernel batch compilation failed";
      if (index < built.tasks.size() && !built.tasks[index].error.empty())
        message += ": " + built.tasks[index].error;
      else if (!built.error.empty())
        message += ": " + built.error;
      if (built.configure && !built.configure->output.empty())
        message += "\n[configure]\n" + built.configure->output;
      if (built.build && !built.build->output.empty())
        message += "\n[build]\n" + built.build->output;
      results.emplace_back(runtime::Status(runtime::StatusCode::ExecutionError, std::move(message)));
      continue;
    }
    auto executable = runtime::Executable::load(*built.tasks[index].kernel_library);
    if (!executable) {
      results.emplace_back(executable.status());
      continue;
    }
    const auto& source = *recipes[index].source;
    if (executable.value()->operator_name() != source.definition->name() ||
        executable.value()->specialization_key() != source.binding.specialization_key) {
      results.emplace_back(
        runtime::Status(runtime::StatusCode::AbiMismatch, "generated kernel identity does not match its binding"));
      continue;
    }
    results.emplace_back(std::move(executable).value());
  }
  return results;
}

} // namespace vecops::compiler
