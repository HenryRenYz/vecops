/**
 * @file Executable.cpp
 * @brief Validation, dynamic loading, workspace queries, and kernel execution.
 *
 * Artifact metadata is validated before it is exposed to callers. Public
 * invocation performs specialization checks again before crossing the C ABI.
 */

#include "vecops/runtime/Executable.h"

#include <array>
#include <cstring>
#include <limits>
#include <sstream>

#if defined(__unix__) || defined(__APPLE__)
#  include <dlfcn.h>
#endif

namespace vecops::runtime {
namespace {

constexpr std::size_t kErrorCapacity = 1024;

template <typename T>
const T& strided_element(const T* data, std::uint32_t stride, std::size_t index) {
  // ABI descriptor arrays may use a larger, append-only record stride.  Avoid
  // pointer arithmetic in T units so a v1 runtime reads only each v1 prefix.
  const auto* bytes = reinterpret_cast<const std::byte*>(data);
  return *reinterpret_cast<const T*>(bytes + stride * index);
}

Status make_status(StatusCode code, std::string message) {
  return Status(code, std::move(message));
}

Status validate_descriptor(const VecopsKernelDescriptorV1* descriptor) {
  if (descriptor == nullptr) {
    return make_status(StatusCode::AbiMismatch, "kernel query returned a null descriptor");
  }
  if (descriptor->struct_size < sizeof(VecopsKernelDescriptorV1)) {
    return make_status(StatusCode::AbiMismatch, "kernel descriptor is smaller than ABI v1");
  }
  if (descriptor->abi_version_major != VECOPS_KERNEL_ABI_VERSION_MAJOR ||
      descriptor->abi_version_minor > VECOPS_KERNEL_ABI_VERSION_MINOR) {
    return make_status(StatusCode::AbiMismatch, "kernel ABI version is not supported by this runtime");
  }
  if (descriptor->operator_name == nullptr || descriptor->specialization_key == nullptr) {
    return make_status(StatusCode::AbiMismatch, "kernel descriptor has no operator or specialization name");
  }
  if (descriptor->num_parameters != 0 && descriptor->parameters == nullptr) {
    return make_status(StatusCode::AbiMismatch, "kernel descriptor has no parameter array");
  }
  if (descriptor->num_parameters != 0 && (descriptor->parameter_stride < sizeof(VecopsParameterDescriptor) ||
                                          descriptor->parameter_stride % alignof(VecopsParameterDescriptor) != 0)) {
    return make_status(StatusCode::AbiMismatch, "kernel has an invalid parameter stride");
  }
  if (descriptor->run == nullptr) {
    return make_status(StatusCode::AbiMismatch, "kernel descriptor has no run entry point");
  }
  for (std::uint32_t index = 0; index < descriptor->num_parameters; ++index) {
    const auto& parameter = strided_element(descriptor->parameters, descriptor->parameter_stride, index);
    if (parameter.struct_size < sizeof(VecopsParameterDescriptor) || parameter.name == nullptr) {
      return make_status(StatusCode::AbiMismatch, "kernel has an invalid parameter descriptor");
    }
    if (parameter.kind == VECOPS_VALUE_TENSOR) {
      const auto& tensor = parameter.constraint.tensor;
      if (tensor.struct_size < sizeof(VecopsTensorParameter) || tensor.rank < -1 ||
          (tensor.rank > 0 && tensor.sizes == nullptr) ||
          (tensor.rank > 0 && (tensor.dimension_constraint_stride < sizeof(VecopsDimensionConstraint) ||
                               tensor.dimension_constraint_stride % alignof(VecopsDimensionConstraint) != 0)) ||
          (tensor.rank == -1 && (tensor.sizes != nullptr || tensor.strides != nullptr))) {
        return make_status(StatusCode::AbiMismatch, "kernel has an invalid tensor constraint");
      }
      for (std::int32_t axis = 0; axis < tensor.rank; ++axis) {
        const auto& size = strided_element(tensor.sizes, tensor.dimension_constraint_stride, axis);
        if (size.struct_size < sizeof(VecopsDimensionConstraint) || size.kind > VECOPS_DIM_DYNAMIC ||
            (size.kind == VECOPS_DIM_DYNAMIC && size.alignment <= 0)) {
          return make_status(StatusCode::AbiMismatch, "kernel has an invalid size constraint");
        }
        if (tensor.strides != nullptr) {
          const auto& stride = strided_element(tensor.strides, tensor.dimension_constraint_stride, axis);
          if (stride.struct_size < sizeof(VecopsDimensionConstraint) || stride.kind > VECOPS_DIM_DYNAMIC ||
              (stride.kind == VECOPS_DIM_DYNAMIC && stride.alignment <= 0)) {
            return make_status(StatusCode::AbiMismatch, "kernel has an invalid stride constraint");
          }
        }
      }
    } else if (parameter.kind == VECOPS_VALUE_SCALAR) {
      if (parameter.constraint.scalar.struct_size < sizeof(VecopsScalarParameter)) {
        return make_status(StatusCode::AbiMismatch, "kernel has an invalid scalar constraint");
      }
    } else {
      return make_status(StatusCode::AbiMismatch, "kernel parameter has an unknown value kind");
    }
  }
  return Status::success();
}

bool accepts_dimension(const VecopsDimensionConstraint& constraint, std::int64_t value) {
  if (constraint.struct_size < sizeof(VecopsDimensionConstraint))
    return false;
  switch (constraint.kind) {
  case VECOPS_DIM_ANY:
    return true;
  case VECOPS_DIM_CONST:
    return value == constraint.value;
  case VECOPS_DIM_DYNAMIC:
    return constraint.alignment > 0 && value >= constraint.lower_bound && value <= constraint.upper_bound &&
           value % constraint.alignment == 0;
  default:
    return false;
  }
}

Status validate_against_descriptor(const ArgumentMetadata& arguments, const VecopsKernelDescriptorV1& descriptor) {
  if (arguments.size() != descriptor.num_parameters) {
    return make_status(StatusCode::InvalidArgument, "argument count does not match the kernel specialization");
  }
  for (std::size_t index = 0; index < arguments.size(); ++index) {
    const auto& value = arguments[index];
    const auto& parameter = strided_element(descriptor.parameters, descriptor.parameter_stride, index);
    if (std::holds_alternative<std::monostate>(value)) {
      if (parameter.kind == VECOPS_VALUE_TENSOR && parameter.constraint.tensor.optional != 0) {
        continue;
      }
      return make_status(StatusCode::InvalidArgument, std::string(parameter.name) + " is not optional");
    }
    if (parameter.kind == VECOPS_VALUE_TENSOR) {
      const auto* tensor = std::get_if<TensorView>(&value);
      if (tensor == nullptr) {
        return make_status(StatusCode::InvalidArgument, std::string(parameter.name) + " must be a tensor");
      }
      const auto& constraint = parameter.constraint.tensor;
      if ((dtype_bit(tensor->dtype) & constraint.dtype_mask) == 0) {
        return make_status(StatusCode::InvalidArgument, std::string(parameter.name) + " dtype mismatch");
      }
      if ((constraint.device_type != 0 && tensor->device_type != constraint.device_type) ||
          (constraint.device_index >= 0 && tensor->device_index != constraint.device_index)) {
        return make_status(StatusCode::InvalidArgument, std::string(parameter.name) + " device mismatch");
      }
      if (tensor->sizes.size() != tensor->strides.size() ||
          tensor->sizes.size() > std::numeric_limits<std::uint32_t>::max() ||
          (constraint.rank >= 0 && tensor->sizes.size() != static_cast<std::size_t>(constraint.rank))) {
        return make_status(StatusCode::InvalidArgument, std::string(parameter.name) + " rank mismatch");
      }
      bool has_elements = true;
      for (const auto size : tensor->sizes) {
        if (size < 0) {
          return make_status(StatusCode::InvalidArgument, std::string(parameter.name) + " has a negative size");
        }
        if (size == 0)
          has_elements = false;
      }
      if (has_elements && tensor->data == nullptr) {
        return make_status(StatusCode::InvalidArgument, std::string(parameter.name) + " has no data");
      }
      const auto required_access = constraint.required_flags & (VECOPS_TENSOR_READ | VECOPS_TENSOR_WRITE);
      if ((tensor->flags & required_access) != required_access) {
        return make_status(StatusCode::InvalidArgument, std::string(parameter.name) + " access mismatch");
      }
      for (std::size_t axis = 0; axis < tensor->sizes.size(); ++axis) {
        if (constraint.sizes != nullptr &&
            !accepts_dimension(strided_element(constraint.sizes, constraint.dimension_constraint_stride, axis),
                               tensor->sizes[axis])) {
          return make_status(StatusCode::InvalidArgument, std::string(parameter.name) + " shape mismatch");
        }
        if (constraint.strides != nullptr &&
            !accepts_dimension(strided_element(constraint.strides, constraint.dimension_constraint_stride, axis),
                               tensor->strides[axis])) {
          return make_status(StatusCode::InvalidArgument, std::string(parameter.name) + " stride mismatch");
        }
      }
    } else {
      const auto* scalar = std::get_if<Scalar>(&value);
      if (scalar == nullptr || (dtype_bit(scalar->dtype) & parameter.constraint.scalar.dtype_mask) == 0) {
        return make_status(StatusCode::InvalidArgument, std::string(parameter.name) + " scalar mismatch");
      }
    }
  }
  return Status::success();
}

Status kernel_status(std::int32_t result, const VecopsError& error, std::string_view fallback) {
  if (result == VECOPS_STATUS_OK)
    return Status::success();
  std::string message(fallback);
  if (error.message != nullptr && error.message[0] != '\0')
    message = error.message;
  StatusCode code = StatusCode::ExecutionError;
  if (result >= VECOPS_STATUS_INVALID_ARGUMENT && result <= VECOPS_STATUS_INTERNAL_ERROR) {
    code = static_cast<StatusCode>(result);
  }
  return make_status(code, std::move(message));
}

} // namespace

LoadedModule::~LoadedModule() {
#if defined(__unix__) || defined(__APPLE__)
  if (handle_ != nullptr)
    dlclose(handle_);
#endif
}

Result<std::shared_ptr<LoadedModule>> LoadedModule::load(const std::filesystem::path& library_path) {
#if defined(__unix__) || defined(__APPLE__)
  dlerror();
  void* handle = dlopen(library_path.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (handle == nullptr) {
    const char* error = dlerror();
    return make_status(StatusCode::LoadError, "failed to load " + library_path.string() + ": " +
                                                (error == nullptr ? "unknown dynamic-loader error" : error));
  }
  dlerror();
  auto* query = reinterpret_cast<VecopsKernelQueryFnV1>(dlsym(handle, VECOPS_KERNEL_QUERY_SYMBOL_V1));
  const char* symbol_error = dlerror();
  if (symbol_error != nullptr || query == nullptr) {
    std::string message = "kernel artifact does not export ";
    message += VECOPS_KERNEL_QUERY_SYMBOL_V1;
    if (symbol_error != nullptr)
      message += std::string(": ") + symbol_error;
    dlclose(handle);
    return make_status(StatusCode::AbiMismatch, std::move(message));
  }
  const auto* descriptor = query();
  auto descriptor_status = validate_descriptor(descriptor);
  if (!descriptor_status.ok()) {
    dlclose(handle);
    return descriptor_status;
  }
  return std::shared_ptr<LoadedModule>(new LoadedModule(library_path, handle, descriptor));
#else
  (void)library_path;
  return make_status(StatusCode::LoadError, "dynamic loading is not implemented on this platform");
#endif
}

Result<std::shared_ptr<Executable>> Executable::load(const std::filesystem::path& library_path) {
  auto module = LoadedModule::load(library_path);
  if (!module)
    return module.status();
  return std::make_shared<Executable>(std::move(module).value());
}

std::string_view Executable::operator_name() const {
  return module_->descriptor().operator_name;
}

std::string_view Executable::specialization_key() const {
  return module_->descriptor().specialization_key;
}

Status Executable::can_invoke(const ArgumentMetadata& arguments) const {
  return validate_against_descriptor(arguments, module_->descriptor());
}

Result<std::uint64_t> Executable::workspace_size(const ArgumentMetadata& arguments,
                                                 const VecopsExecutionContext* context) const {
  auto status = can_invoke(arguments);
  if (!status.ok())
    return status;
  const auto workspace = module_->descriptor().workspace;
  if (workspace == nullptr)
    return std::uint64_t{0};
  CallFrame frame(arguments, nullptr, 0, context);
  std::uint64_t size = 0;
  std::array<char, kErrorCapacity> message{};
  VecopsError error{
    .struct_size = sizeof(VecopsError),
    .code = VECOPS_STATUS_OK,
    .message = message.data(),
    .message_capacity = message.size(),
    .message_required = 0,
  };
  const auto result = workspace(&frame.abi(), &size, &error);
  message.back() = '\0';
  status = kernel_status(result, error, "kernel workspace query failed");
  if (!status.ok())
    return status;
  return size;
}

Status Executable::invoke(const ArgumentMetadata& arguments, void* workspace, std::uint64_t workspace_size,
                          const VecopsExecutionContext* context) const {
  auto required = this->workspace_size(arguments, context);
  if (!required)
    return required.status();
  if (required.value() > workspace_size || (required.value() != 0 && workspace == nullptr)) {
    std::ostringstream stream;
    stream << "kernel requires " << required.value() << " workspace bytes, but " << workspace_size << " were supplied";
    return make_status(StatusCode::InvalidArgument, stream.str());
  }
  CallFrame frame(arguments, workspace, workspace_size, context);
  return invoke_unchecked(frame.abi());
}

Status Executable::invoke_unchecked(const VecopsCall& call) const {
  std::array<char, kErrorCapacity> message{};
  VecopsError error{
    .struct_size = sizeof(VecopsError),
    .code = VECOPS_STATUS_OK,
    .message = message.data(),
    .message_capacity = message.size(),
    .message_required = 0,
  };
  const auto result = module_->descriptor().run(&call, &error);
  message.back() = '\0';
  return kernel_status(result, error, "kernel execution failed");
}

} // namespace vecops::runtime
