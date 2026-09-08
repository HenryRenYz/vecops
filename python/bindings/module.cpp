/**
 * @file module.cpp
 * @brief pybind11 adapters for the compiler, runtime, and NumPy CPU arrays.
 *
 * Python owners remain alive for every non-owning tensor view. Synchronous
 * compiler and executable calls release the GIL and never call back into
 * Python from their C++ hot paths.
 */

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <cstdint>
#include <limits>
#include <atomic>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl/filesystem.h>

#include "vecops/compiler/Compiler.h"
#include "vecops/runtime/Executable.h"
#include "vecops/runtime/OperatorBridgeAbi.h"
#include "vecops/runtime/Provider.h"

namespace py = pybind11;

namespace {

using vecops::runtime::ArgumentMetadata;
using vecops::runtime::DType;
using vecops::runtime::Executable;
using vecops::runtime::KernelCall;
using vecops::runtime::KernelDef;
using vecops::runtime::Scalar;
using vecops::runtime::Status;
using vecops::runtime::StatusCode;
using vecops::runtime::TensorView;
using vecops::runtime::Value;

namespace compiler = vecops::compiler;

/** Raises the Python exception appropriate for a failed runtime status. */
[[noreturn]] void throw_status(const Status& status) {
  if (status.code() == StatusCode::InvalidArgument || status.code() == StatusCode::NotApplicable) {
    throw py::value_error(status.message());
  }
  if (status.code() == StatusCode::LoadError || status.code() == StatusCode::NotFound) {
    throw py::import_error(status.message());
  }
  throw std::runtime_error(status.message());
}

/** Maps a native-endian NumPy dtype to the matching runtime dtype. */
DType numpy_dtype(const py::dtype& dtype) {
  if (!py::cast<bool>(dtype.attr("isnative"))) {
    throw py::type_error("vecops only supports native-endian NumPy dtypes");
  }

  const auto kind = py::cast<std::string>(dtype.attr("kind"));
  const auto item_size = py::cast<py::ssize_t>(dtype.attr("itemsize"));
  if (kind == "b" && item_size == 1)
    return DType::Bool;
  if (kind == "i") {
    if (item_size == 1)
      return DType::Int8;
    if (item_size == 2)
      return DType::Int16;
    if (item_size == 4)
      return DType::Int32;
    if (item_size == 8)
      return DType::Int64;
  }
  if (kind == "u") {
    if (item_size == 1)
      return DType::UInt8;
    if (item_size == 2)
      return DType::UInt16;
    if (item_size == 4)
      return DType::UInt32;
    if (item_size == 8)
      return DType::UInt64;
  }
  if (kind == "f") {
    if (item_size == 2)
      return DType::Float16;
    if (item_size == 4)
      return DType::Float32;
    if (item_size == 8)
      return DType::Float64;
  }
  throw py::type_error("unsupported NumPy dtype: " + py::cast<std::string>(dtype.attr("name")));
}

/** Returns the ABI byte width for a NumPy-representable runtime dtype. */
std::size_t dtype_size(DType dtype) {
  switch (dtype) {
  case DType::Bool:
  case DType::Int8:
  case DType::UInt8:
    return 1;
  case DType::Int16:
  case DType::UInt16:
  case DType::Float16:
  case DType::BFloat16:
    return 2;
  case DType::Int32:
  case DType::UInt32:
  case DType::Float32:
    return 4;
  case DType::Int64:
  case DType::UInt64:
  case DType::Float64:
    return 8;
  case DType::Invalid:
    return 0;
  }
  return 0;
}

/**
 * Owns a NumPy array alongside its non-owning runtime TensorView.
 *
 * Keeping the Python owner here guarantees that storage remains alive for every
 * workspace query and synchronous invocation made through PyArgumentMetadata.
 */
class PyTensorView {
public:
  explicit PyTensorView(py::object array, bool writable)
    : owner_(std::move(array)) {
    if (!py::isinstance<py::array>(owner_)) {
      // TODO: Add device-aware adapters when vecops supports non-CPU tensors.
      throw py::type_error("expected a NumPy ndarray backed by CPU memory");
    }

    const auto numpy_array = py::reinterpret_borrow<py::array>(owner_);
    if (writable && !numpy_array.writeable()) {
      throw py::value_error("a writable vecops TensorView requires a writable NumPy ndarray");
    }
    if (writable && (numpy_array.flags() & py::array::c_style) == 0) {
      throw py::value_error("a writable vecops TensorView requires a C-contiguous NumPy ndarray");
    }

    const auto info = numpy_array.request();
    if (info.itemsize <= 0) {
      throw py::value_error("NumPy ndarray has an invalid element size");
    }

    view_.data = info.ptr;
    view_.device_type = VECOPS_DEVICE_CPU;
    view_.device_index = 0;
    view_.dtype = numpy_dtype(numpy_array.dtype());
    if (static_cast<std::size_t>(info.itemsize) != dtype_size(view_.dtype)) {
      throw py::type_error("NumPy dtype item size does not match the vecops ABI dtype width");
    }
    view_.flags = VECOPS_TENSOR_READ | (writable ? VECOPS_TENSOR_WRITE : 0);
    view_.sizes.reserve(static_cast<std::size_t>(info.ndim));
    view_.strides.reserve(static_cast<std::size_t>(info.ndim));
    for (py::ssize_t axis = 0; axis < info.ndim; ++axis) {
      const auto byte_stride = info.strides[static_cast<std::size_t>(axis)];
      if (byte_stride < 0 || byte_stride % info.itemsize != 0) {
        throw py::value_error("vecops does not support negative or non-element-aligned NumPy strides");
      }
      const auto element_stride = byte_stride / info.itemsize;
      if (writable && info.shape[static_cast<std::size_t>(axis)] > 1 && element_stride == 0) {
        throw py::value_error("a writable vecops TensorView cannot have overlapping broadcast strides");
      }
      view_.sizes.push_back(info.shape[static_cast<std::size_t>(axis)]);
      view_.strides.push_back(element_stride);
    }
  }

  /** Returns the framework-neutral view passed to the runtime. */
  [[nodiscard]] const TensorView& view() const {
    return view_;
  }

  /** Returns the ndarray that owns the view's storage. */
  [[nodiscard]] const py::object& owner() const {
    return owner_;
  }

private:
  py::object owner_;
  TensorView view_;
};

/** Storage-free tensor metadata used to bind or precompile a specialization. */
class PyTensorMeta {
public:
  PyTensorMeta(DType dtype, std::vector<std::int64_t> sizes, std::vector<std::int64_t> strides, std::uint64_t flags,
               std::uint32_t device_type, std::int32_t device_index) {
    if (sizes.size() != strides.size())
      throw py::value_error("TensorMeta sizes and strides must have equal lengths");
    view_.dtype = dtype;
    view_.sizes = std::move(sizes);
    view_.strides = std::move(strides);
    view_.flags = flags;
    view_.device_type = device_type;
    view_.device_index = device_index;
  }

  [[nodiscard]] const TensorView& view() const {
    return view_;
  }

private:
  TensorView view_;
};

/** Owns Python value wrappers and their erased runtime metadata. */
class PyArgumentMetadata {
public:
  explicit PyArgumentMetadata(py::iterable values)
    : values_(py::tuple(values)) {
    std::vector<Value> erased;
    erased.reserve(values_.size());
    for (const auto item : values_) {
      if (item.is_none()) {
        erased.emplace_back(std::monostate{});
      } else if (py::isinstance<PyTensorView>(item)) {
        erased.emplace_back(py::cast<const PyTensorView&>(item).view());
      } else if (py::isinstance<PyTensorMeta>(item)) {
        erased.emplace_back(py::cast<const PyTensorMeta&>(item).view());
      } else if (py::isinstance<Scalar>(item)) {
        erased.emplace_back(py::cast<const Scalar&>(item));
      } else if (py::isinstance<py::bool_>(item)) {
        erased.emplace_back(Scalar::unsigned_integer(py::cast<bool>(item) ? 1 : 0, DType::Bool));
      } else if (py::isinstance<py::int_>(item)) {
        erased.emplace_back(Scalar::signed_integer(py::cast<std::int64_t>(item), DType::Int64));
      } else if (py::isinstance<py::float_>(item)) {
        erased.emplace_back(Scalar::floating(py::cast<double>(item), DType::Float64));
      } else {
        throw py::type_error("arguments must contain TensorView, Scalar, bool, int, float, or None values");
      }
    }
    metadata_ = ArgumentMetadata(std::move(erased));
  }

  /** Returns the erased runtime metadata. */
  [[nodiscard]] const ArgumentMetadata& metadata() const {
    return metadata_;
  }

  /** Returns the original wrappers retained for storage ownership. */
  [[nodiscard]] const py::tuple& values() const {
    return values_;
  }

private:
  py::tuple values_;
  ArgumentMetadata metadata_;
};

vecops::runtime::ConstExpr const_expr(py::handle value) {
  if (py::isinstance<py::int_>(value))
    return py::cast<std::int64_t>(value);
  if (py::isinstance<py::str>(value))
    return py::cast<std::string>(value);
  if (py::isinstance<vecops::runtime::ConstExpr>(value))
    return py::cast<const vecops::runtime::ConstExpr&>(value);
  throw py::type_error("expected an integer or ConstInt symbol name");
}

vecops::runtime::DimensionDef dimension_def(py::handle value) {
  if (py::isinstance<py::int_>(value))
    return vecops::runtime::DimensionDef(py::cast<std::int64_t>(value));
  if (py::isinstance<py::str>(value))
    return vecops::runtime::DimensionDef(py::cast<std::string>(value));
  if (py::isinstance<vecops::runtime::DimensionDef>(value))
    return py::cast<const vecops::runtime::DimensionDef&>(value);
  throw py::type_error("dimensions must be integers, symbol names, Const, or Dynamic declarations");
}

std::vector<vecops::runtime::DimensionDef> dimensions(py::iterable values) {
  std::vector<vecops::runtime::DimensionDef> result;
  for (const auto value : values)
    result.push_back(dimension_def(value));
  return result;
}

vecops::runtime::TensorDTypeDef tensor_dtype(py::handle value) {
  if (value.is_none())
    return {};
  if (py::isinstance<DType>(value))
    return vecops::runtime::TensorDTypeDef(py::cast<DType>(value));
  if (py::isinstance<py::str>(value))
    return vecops::runtime::TensorDTypeDef(py::cast<std::string>(value));
  if (py::isinstance<vecops::runtime::TensorDTypeDef>(value))
    return py::cast<const vecops::runtime::TensorDTypeDef&>(value);
  throw py::type_error("tensor dtype must be None, DType, or a dtype symbol name");
}

Scalar python_scalar(py::handle value, DType dtype) {
  switch (dtype) {
  case DType::Bool:
    if (!py::isinstance<py::bool_>(value))
      throw py::type_error("boolean default required");
    return Scalar::unsigned_integer(py::cast<bool>(value) ? 1 : 0, dtype);
  case DType::Int8:
  case DType::Int16:
  case DType::Int32:
  case DType::Int64:
    if (!py::isinstance<py::int_>(value))
      throw py::type_error("integer default required");
    return Scalar::signed_integer(py::cast<std::int64_t>(value), dtype);
  case DType::UInt8:
  case DType::UInt16:
  case DType::UInt32:
  case DType::UInt64:
    if (!py::isinstance<py::int_>(value))
      throw py::type_error("integer default required");
    return Scalar::unsigned_integer(py::cast<std::uint64_t>(value), dtype);
  case DType::Float16:
  case DType::BFloat16:
  case DType::Float32:
  case DType::Float64:
    if (!py::isinstance<py::float_>(value) && !py::isinstance<py::int_>(value))
      throw py::type_error("floating-point default required");
    return Scalar::floating(py::cast<double>(value), dtype);
  case DType::Invalid:
    throw py::value_error("invalid scalar dtype");
  }
  throw py::value_error("invalid scalar dtype");
}

vecops::runtime::SpecializationValues specialization_values(py::dict values) {
  vecops::runtime::SpecializationValues result;
  for (const auto& [key, value] : values) {
    const auto name = py::cast<std::string>(key);
    if (py::isinstance<DType>(value))
      result.emplace(name, py::cast<DType>(value));
    else if (py::isinstance<py::int_>(value))
      result.emplace(name, py::cast<std::int64_t>(value));
    else
      throw py::type_error("specialization values must be int or DType");
  }
  return result;
}

class PyKernelCall {
public:
  PyKernelCall(PyArgumentMetadata arguments, py::dict values)
    : arguments_(std::move(arguments))
    , call_(arguments_.metadata(), specialization_values(std::move(values))) {
  }

  PyKernelCall(py::iterable arguments, py::dict values)
    : PyKernelCall(PyArgumentMetadata(std::move(arguments)), std::move(values)) {
  }

  [[nodiscard]] const PyArgumentMetadata& arguments() const {
    return arguments_;
  }
  [[nodiscard]] const KernelCall& call() const {
    return call_;
  }

private:
  PyArgumentMetadata arguments_;
  KernelCall call_;
};

class PyOperator {
public:
  PyOperator(std::filesystem::path kernel_file, KernelDef definition,
             std::shared_ptr<compiler::Compiler> compiler_instance, vecops::runtime::ArtifactCacheMode cache_mode,
             std::filesystem::path cache_directory, std::string namespace_key, std::string recipe_id,
             std::string source_fingerprint)
    : definition_(std::make_shared<KernelDef>(std::move(definition))) {
    add_operator(std::move(kernel_file), definition_, std::move(compiler_instance), cache_mode,
                 std::move(cache_directory), std::move(namespace_key), std::move(recipe_id),
                 std::move(source_fingerprint), {});
    register_handle();
  }

  PyOperator(std::vector<std::filesystem::path> kernel_files, std::vector<KernelDef> definitions,
             std::shared_ptr<compiler::Compiler> compiler_instance, vecops::runtime::ArtifactCacheMode cache_mode,
             std::filesystem::path cache_directory, std::string namespace_key, std::vector<std::string> recipe_ids,
             std::vector<std::string> source_fingerprints,
             std::vector<vecops::runtime::SpecializationValues> default_values) {
    if (kernel_files.empty() || kernel_files.size() != definitions.size() || kernel_files.size() != recipe_ids.size() ||
        kernel_files.size() != source_fingerprints.size() || kernel_files.size() != default_values.size())
      throw py::value_error("operator recipe lists must be non-empty and have equal lengths");
    definition_ = std::make_shared<KernelDef>(definitions.front());
    state_ = std::make_shared<State>();
    for (std::size_t index = 0; index < kernel_files.size(); ++index) {
      auto definition = std::make_shared<KernelDef>(std::move(definitions[index]));
      add_operator(std::move(kernel_files[index]), std::move(definition), compiler_instance, cache_mode,
                   cache_directory, namespace_key, std::move(recipe_ids[index]), std::move(source_fingerprints[index]),
                   std::move(default_values[index]));
    }
    register_handle();
  }

  ~PyOperator() {
    std::lock_guard lock(operator_registry_mutex);
    operator_registry.erase(handle_);
  }

  [[nodiscard]] Status invoke(const PyKernelCall& call) const {
    Status last(StatusCode::NotApplicable, "no registered recipe accepted the invocation");
    for (std::size_t index = 0; index < state_->operator_instances.size(); ++index) {
      auto values = state_->default_values[index];
      for (const auto& [name, value] : call.call().values())
        values.insert_or_assign(name, value);
      auto status = state_->operator_instances[index]->invoke(KernelCall(call.call().arguments(), std::move(values)));
      if (status.ok())
        return status;
      if (status.code() != StatusCode::InvalidArgument && status.code() != StatusCode::NotApplicable &&
          status.code() != StatusCode::NotFound)
        return status;
      last = std::move(status);
    }
    return last;
  }

  [[nodiscard]] Status prepare(const PyKernelCall& call) const {
    Status last(StatusCode::NotApplicable, "no registered recipe accepted the specialization");
    for (std::size_t index = 0; index < state_->operator_instances.size(); ++index) {
      auto values = state_->default_values[index];
      for (const auto& [name, value] : call.call().values())
        values.insert_or_assign(name, value);
      auto result = state_->operator_instances[index]->resolve(KernelCall(call.call().arguments(), std::move(values)));
      if (result)
        return Status::success();
      if (result.status().code() != StatusCode::InvalidArgument &&
          result.status().code() != StatusCode::NotApplicable && result.status().code() != StatusCode::NotFound)
        return result.status();
      last = result.status();
    }
    return last;
  }
  [[nodiscard]] std::uint64_t handle() const {
    return handle_;
  }

  py::object invoke_values(py::args arguments, py::kwargs values) const {
    if (arguments.size() > definition_->parameters().size())
      throw py::type_error("too many positional operator arguments");
    py::dict remaining(values);
    py::list wrapped;
    py::list outputs;
    for (std::size_t index = 0; index < definition_->parameters().size(); ++index) {
      const auto& parameter = definition_->parameters()[index];
      const auto& name = std::visit([](const auto& item) -> const std::string& { return item.name; }, parameter);
      const py::str key(name);
      py::object argument;
      if (index < arguments.size()) {
        if (remaining.contains(key))
          throw py::type_error("multiple values for operator argument '" + name + "'");
        argument = py::reinterpret_borrow<py::object>(arguments[index]);
      } else if (remaining.contains(key)) {
        argument = py::reinterpret_borrow<py::object>(remaining[key]);
        remaining.attr("pop")(key);
      } else if (const auto* tensor = std::get_if<vecops::runtime::TensorDef>(&parameter); tensor && tensor->optional) {
        argument = py::none();
      } else if (const auto* scalar = std::get_if<vecops::runtime::ValueDef>(&parameter);
                 scalar && scalar->default_value) {
        argument = py::cast(*scalar->default_value);
      } else {
        throw py::type_error("missing required operator argument '" + name + "'");
      }
      if (const auto* tensor = std::get_if<vecops::runtime::TensorDef>(&parameter)) {
        if (argument.is_none() || py::isinstance<PyTensorView>(argument)) {
          wrapped.append(argument);
        } else {
          auto view = PyTensorView(argument, tensor->writable());
          wrapped.append(py::cast(std::move(view)));
        }
        if (tensor->writable())
          outputs.append(argument);
      } else {
        const auto& scalar = std::get<vecops::runtime::ValueDef>(parameter);
        if (py::isinstance<Scalar>(argument))
          wrapped.append(argument);
        else
          wrapped.append(py::cast(python_scalar(argument, scalar.dtype)));
      }
    }
    PyKernelCall call(wrapped, std::move(remaining));
    const auto status = [&]() {
      py::gil_scoped_release release;
      return invoke(call);
    }();
    if (!status.ok())
      throw_status(status);
    if (outputs.empty())
      return py::none();
    if (outputs.size() == 1)
      return outputs[0].cast<py::object>();
    return py::tuple(outputs);
  }

private:
  void add_operator(std::filesystem::path kernel_file, std::shared_ptr<const KernelDef> definition,
                    std::shared_ptr<compiler::Compiler> compiler_instance,
                    vecops::runtime::ArtifactCacheMode cache_mode, std::filesystem::path cache_directory,
                    std::string namespace_key, std::string recipe_id, std::string source_fingerprint,
                    vecops::runtime::SpecializationValues default_values) {
    if (state_ == nullptr)
      state_ = std::make_shared<State>();
    auto recipe = std::make_shared<vecops::runtime::SourceKernelRecipe>(
      std::move(recipe_id), std::move(kernel_file), definition, std::move(source_fingerprint), default_values);
    vecops::runtime::ArtifactProviderConfig provider_config;
    provider_config.mode = cache_mode;
    provider_config.cache_directory = std::move(cache_directory);
    provider_config.namespace_key = std::move(namespace_key);
    if (cache_mode != vecops::runtime::ArtifactCacheMode::CacheOnly) {
      if (compiler_instance == nullptr)
        throw py::value_error("ReadWrite and CompileOnly operators require a Compiler");
      provider_config.build = [compiler_instance =
                                 std::move(compiler_instance)](const vecops::runtime::BoundKernelRecipe& bound) {
        return compiler_instance->compile_kernel(bound);
      };
    }
    auto provider = std::make_shared<vecops::runtime::ArtifactExecutableProvider>(std::move(provider_config));
    state_->operator_instances.push_back(std::make_unique<vecops::runtime::Operator>(
      *definition, std::vector<std::shared_ptr<const vecops::runtime::KernelRecipe>>{std::move(recipe)},
      std::make_shared<vecops::runtime::OrderedDispatchPolicy>(), std::move(provider)));
    state_->default_values.push_back(std::move(default_values));
  }

  void register_handle() {
    handle_ = next_operator_handle.fetch_add(1, std::memory_order_relaxed);
    std::lock_guard lock(operator_registry_mutex);
    operator_registry.emplace(handle_, state_);
  }

  struct State {
    std::vector<std::unique_ptr<vecops::runtime::Operator>> operator_instances;
    std::vector<vecops::runtime::SpecializationValues> default_values;
  };
  static std::atomic<std::uint64_t> next_operator_handle;
  static std::mutex operator_registry_mutex;
  static std::map<std::uint64_t, std::weak_ptr<State>> operator_registry;

  friend Status invoke_registered_operator(std::uint64_t, const VecopsCall*, std::uint32_t,
                                           const VecopsSpecializationArgument*);
  std::shared_ptr<const KernelDef> definition_;
  std::shared_ptr<State> state_;
  std::uint64_t handle_ = 0;
};

std::atomic<std::uint64_t> PyOperator::next_operator_handle{1};
std::mutex PyOperator::operator_registry_mutex;
std::map<std::uint64_t, std::weak_ptr<PyOperator::State>> PyOperator::operator_registry;

Status invoke_registered_operator(std::uint64_t handle, const VecopsCall* call, std::uint32_t num_specs,
                                  const VecopsSpecializationArgument* specs) {
  if (call == nullptr || call->struct_size < sizeof(VecopsCall) || (call->num_values != 0 && call->values == nullptr))
    return Status(StatusCode::InvalidArgument, "framework bridge received an invalid call");
  if (num_specs != 0 && specs == nullptr)
    return Status(StatusCode::InvalidArgument, "framework bridge received no specialization array");
  std::shared_ptr<PyOperator::State> state;
  {
    std::lock_guard lock(PyOperator::operator_registry_mutex);
    const auto found = PyOperator::operator_registry.find(handle);
    if (found != PyOperator::operator_registry.end())
      state = found->second.lock();
  }
  if (state == nullptr)
    return Status(StatusCode::NotFound, "framework operator handle is no longer registered");

  std::vector<Value> values;
  values.reserve(call->num_values);
  for (std::uint32_t index = 0; index < call->num_values; ++index) {
    const auto& value = call->values[index];
    if (value.struct_size < sizeof(VecopsValue))
      return Status(StatusCode::InvalidArgument, "framework bridge value is truncated");
    if (value.kind == VECOPS_VALUE_NONE) {
      values.emplace_back(std::monostate{});
    } else if (value.kind == VECOPS_VALUE_TENSOR) {
      const auto& tensor = value.value.tensor;
      if (tensor.struct_size < sizeof(VecopsTensorView) ||
          (tensor.rank != 0 && (tensor.sizes == nullptr || tensor.strides == nullptr)))
        return Status(StatusCode::InvalidArgument, "framework bridge tensor metadata is invalid");
      std::vector<std::int64_t> sizes;
      std::vector<std::int64_t> strides;
      if (tensor.rank != 0) {
        sizes.assign(tensor.sizes, tensor.sizes + tensor.rank);
        strides.assign(tensor.strides, tensor.strides + tensor.rank);
      }
      values.emplace_back(TensorView{
        .data = tensor.data,
        .byte_offset = tensor.byte_offset,
        .device_type = tensor.device_type,
        .device_index = tensor.device_index,
        .dtype = static_cast<DType>(tensor.dtype),
        .sizes = std::move(sizes),
        .strides = std::move(strides),
        .flags = tensor.flags,
      });
    } else if (value.kind == VECOPS_VALUE_SCALAR) {
      const auto& scalar = value.value.scalar;
      if (scalar.struct_size < sizeof(VecopsScalar))
        return Status(StatusCode::InvalidArgument, "framework bridge scalar metadata is invalid");
      values.emplace_back(Scalar{static_cast<DType>(scalar.dtype), scalar.value});
    } else {
      return Status(StatusCode::InvalidArgument, "framework bridge value has an unknown kind");
    }
  }
  vecops::runtime::SpecializationValues specialization;
  for (std::uint32_t index = 0; index < num_specs; ++index) {
    const auto& spec = specs[index];
    if (spec.struct_size < sizeof(VecopsSpecializationArgument) || spec.name == nullptr)
      return Status(StatusCode::InvalidArgument, "framework bridge specialization value is invalid");
    if (spec.kind == VECOPS_SPECIALIZATION_CONST_INT)
      specialization.emplace(spec.name, spec.value.integer);
    else if (spec.kind == VECOPS_SPECIALIZATION_DTYPE)
      specialization.emplace(spec.name, static_cast<DType>(spec.value.dtype));
    else
      return Status(StatusCode::InvalidArgument, "framework bridge specialization kind is invalid");
  }
  KernelCall kernel_call(ArgumentMetadata(std::move(values)), std::move(specialization));
  Status last(StatusCode::NotApplicable, "no registered recipe accepted the framework invocation");
  for (std::size_t index = 0; index < state->operator_instances.size(); ++index) {
    auto values = state->default_values[index];
    for (const auto& [name, value] : kernel_call.values())
      values.insert_or_assign(name, value);
    auto result = state->operator_instances[index]->invoke(KernelCall(kernel_call.arguments(), std::move(values)));
    if (result.ok())
      return result;
    if (result.code() != StatusCode::InvalidArgument && result.code() != StatusCode::NotApplicable &&
        result.code() != StatusCode::NotFound)
      return result;
    last = std::move(result);
  }
  return last;
}

/** Invokes an executable synchronously with automatically managed workspace. */
Status invoke(const Executable& executable, const PyArgumentMetadata& arguments) {
  const auto required = executable.workspace_size(arguments.metadata());
  if (!required)
    return required.status();
  if (required.value() > std::numeric_limits<std::size_t>::max()) {
    return Status(StatusCode::InvalidArgument, "kernel workspace size exceeds the host address space");
  }

  std::vector<std::byte> workspace(static_cast<std::size_t>(required.value()));
  return executable.invoke(arguments.metadata(), workspace.empty() ? nullptr : workspace.data(), workspace.size());
}

/** Joins the configure and build logs while preserving their stage labels. */
std::string build_log(const compiler::BuildResult& result) {
  std::string log;
  if (result.configure) {
    log += "[configure]\n";
    log += result.configure->output;
  }
  if (result.build_and_install) {
    if (!log.empty() && log.back() != '\n')
      log += '\n';
    log += "[build_and_install]\n";
    log += result.build_and_install->output;
  }
  return log;
}

} // namespace

extern "C" VECOPS_RUNTIME_EXPORT int32_t
vecops_operator_bridge_invoke_v1(std::uint64_t handle, const VecopsCall* call, std::uint32_t num_specialization_values,
                                 const VecopsSpecializationArgument* specialization_values, VecopsError* error) {
  const auto result = invoke_registered_operator(handle, call, num_specialization_values, specialization_values);
  if (result.ok())
    return VECOPS_STATUS_OK;
  if (error != nullptr && error->struct_size >= sizeof(VecopsError)) {
    error->code = static_cast<std::int32_t>(result.code());
    error->message_required = result.message().size() + 1;
    if (error->message != nullptr && error->message_capacity != 0) {
      const auto copied = std::min(result.message().size(), error->message_capacity - 1);
      std::memcpy(error->message, result.message().data(), copied);
      error->message[copied] = '\0';
    }
  }
  return static_cast<std::int32_t>(result.code());
}

PYBIND11_MODULE(_C, module) {
  module.doc() = R"doc(Expert-level bindings for the vecops C++ runtime.

Most applications should use :mod:`vecops`, whose Python wrappers normalize
dtypes, derive schemas, and own framework integration. This module mirrors C++
value types for diagnostics, explicit build control, and wrapper implementation;
its constructor-level interfaces intentionally expose native concepts.)doc";

  py::enum_<DType>(module, "DType", "Element types understood by the vecops runtime.")
    .value("invalid", DType::Invalid)
    .value("bool_", DType::Bool)
    .value("int8", DType::Int8)
    .value("uint8", DType::UInt8)
    .value("int16", DType::Int16)
    .value("uint16", DType::UInt16)
    .value("int32", DType::Int32)
    .value("uint32", DType::UInt32)
    .value("int64", DType::Int64)
    .value("uint64", DType::UInt64)
    .value("float16", DType::Float16)
    .value("bfloat16", DType::BFloat16)
    .value("float32", DType::Float32)
    .value("float64", DType::Float64);

  py::class_<PyTensorView>(module, "TensorView", "A CPU NumPy array and its vecops tensor metadata.")
    .def(py::init<py::object, bool>(), py::arg("array"), py::arg("writable") = false,
         "Create a view while retaining the ndarray for the lifetime of the view.")
    .def_static(
      "from_numpy", [](py::object array, bool writable) { return PyTensorView(std::move(array), writable); },
      py::arg("array"), py::arg("writable") = false)
    .def_property_readonly("array", &PyTensorView::owner)
    .def_property_readonly("dtype", [](const PyTensorView& self) { return self.view().dtype; })
    .def_property_readonly("sizes", [](const PyTensorView& self) { return self.view().sizes; })
    .def_property_readonly("strides", [](const PyTensorView& self) { return self.view().strides; })
    .def_property_readonly("device_type", [](const PyTensorView& self) { return self.view().device_type; })
    .def_property_readonly("device_index", [](const PyTensorView& self) { return self.view().device_index; })
    .def_property_readonly("byte_offset", [](const PyTensorView& self) { return self.view().byte_offset; })
    .def_property_readonly("writable", [](const PyTensorView& self) { return self.view().writable(); });

  py::class_<PyTensorMeta>(module, "TensorMeta", "Storage-free tensor metadata for specialization binding.")
    .def(py::init<DType, std::vector<std::int64_t>, std::vector<std::int64_t>, std::uint64_t, std::uint32_t,
                  std::int32_t>(),
         py::arg("dtype"), py::arg("sizes"), py::arg("strides"),
         py::arg("flags") = static_cast<std::uint64_t>(VECOPS_TENSOR_READ),
         py::arg("device_type") = static_cast<std::uint32_t>(VECOPS_DEVICE_CPU), py::arg("device_index") = 0)
    .def_property_readonly("dtype", [](const PyTensorMeta& self) { return self.view().dtype; })
    .def_property_readonly("sizes", [](const PyTensorMeta& self) { return self.view().sizes; })
    .def_property_readonly("strides", [](const PyTensorMeta& self) { return self.view().strides; })
    .def_property_readonly("writable", [](const PyTensorMeta& self) { return self.view().writable(); });

  py::class_<Scalar>(module, "Scalar", "A typed scalar runtime argument.")
    .def_static("signed_integer", &Scalar::signed_integer, py::arg("value"), py::arg("dtype") = DType::Int64)
    .def_static("unsigned_integer", &Scalar::unsigned_integer, py::arg("value"), py::arg("dtype") = DType::UInt64)
    .def_static("floating", &Scalar::floating, py::arg("value"), py::arg("dtype") = DType::Float64)
    .def_readwrite("dtype", &Scalar::dtype)
    .def_property_readonly("value", [](const Scalar& scalar) -> py::object {
      switch (scalar.dtype) {
      case DType::Bool:
        return py::bool_(scalar.value.u64 != 0);
      case DType::Int8:
      case DType::Int16:
      case DType::Int32:
      case DType::Int64:
        return py::int_(scalar.value.i64);
      case DType::UInt8:
      case DType::UInt16:
      case DType::UInt32:
      case DType::UInt64:
        return py::int_(scalar.value.u64);
      case DType::Float16:
      case DType::BFloat16:
      case DType::Float32:
      case DType::Float64:
        return py::float_(scalar.value.f64);
      case DType::Invalid:
        return py::none();
      }
      return py::none();
    });

  py::class_<PyArgumentMetadata>(module, "ArgumentMetadata", "Ordered runtime arguments with retained owners.")
    .def(py::init<py::iterable>(), py::arg("values"))
    .def_property_readonly("values", &PyArgumentMetadata::values)
    .def("__len__", [](const PyArgumentMetadata& self) { return self.metadata().size(); });

  py::enum_<vecops::runtime::SpecializationType>(module, "SpecializationType",
                                                 "Kind of named value emitted in generated vecops::spec.")
    .value("const_int", vecops::runtime::SpecializationType::ConstInt)
    .value("dtype", vecops::runtime::SpecializationType::DType);
  module.attr("ConstInt") = py::cast(vecops::runtime::SpecializationType::ConstInt);
  module.attr("DTypeValue") = py::cast(vecops::runtime::SpecializationType::DType);

  py::class_<vecops::runtime::ConstExpr>(module, "ConstExpr",
                                         "Fixed integer or named ConstInt expression for Dynamic constraints.")
    .def(py::init<std::int64_t>())
    .def(py::init<std::string>())
    .def_property_readonly("is_symbol", &vecops::runtime::ConstExpr::is_symbol)
    .def_property_readonly("fixed_value", &vecops::runtime::ConstExpr::fixed_value)
    .def_property_readonly("symbol", &vecops::runtime::ConstExpr::symbol);

  py::enum_<vecops::runtime::DimensionDef::Kind>(module, "DimensionKind",
                                                 "Specialization/validation mode of one shape or stride coordinate.")
    .value("fixed", vecops::runtime::DimensionDef::Kind::Fixed)
    .value("symbol", vecops::runtime::DimensionDef::Kind::Symbol)
    .value("const", vecops::runtime::DimensionDef::Kind::Const)
    .value("dynamic", vecops::runtime::DimensionDef::Kind::Dynamic);

  py::class_<vecops::runtime::DimensionDef>(module, "DimensionDef",
                                            "Raw C++ declarative dimension rule; prefer vecops.Dynamic publicly.")
    .def(py::init<std::int64_t>())
    .def(py::init<std::string>())
    .def_static("constant", &vecops::runtime::DimensionDef::constant)
    .def_static(
      "dynamic",
      [](py::object alignment, py::object lower, py::object upper) {
        return vecops::runtime::DimensionDef::dynamic(const_expr(alignment), const_expr(lower), const_expr(upper));
      },
      py::arg("alignment") = 1, py::arg("lower_bound") = std::numeric_limits<std::int64_t>::min(),
      py::arg("upper_bound") = std::numeric_limits<std::int64_t>::max())
    .def_static(
      "named_dynamic",
      [](std::string symbol, py::object alignment, py::object lower, py::object upper) {
        return vecops::runtime::DimensionDef::named_dynamic(std::move(symbol), const_expr(alignment), const_expr(lower),
                                                            const_expr(upper));
      },
      py::arg("symbol"), py::arg("alignment") = 1, py::arg("lower_bound") = std::numeric_limits<std::int64_t>::min(),
      py::arg("upper_bound") = std::numeric_limits<std::int64_t>::max())
    .def_static("any", &vecops::runtime::DimensionDef::any)
    .def_property_readonly("kind", &vecops::runtime::DimensionDef::kind)
    .def_property_readonly("fixed_value", &vecops::runtime::DimensionDef::fixed_value)
    .def_property_readonly("symbol", &vecops::runtime::DimensionDef::symbol)
    .def_property_readonly("alignment", &vecops::runtime::DimensionDef::alignment,
                           py::return_value_policy::reference_internal)
    .def_property_readonly("lower_bound", &vecops::runtime::DimensionDef::lower_bound,
                           py::return_value_policy::reference_internal)
    .def_property_readonly("upper_bound", &vecops::runtime::DimensionDef::upper_bound,
                           py::return_value_policy::reference_internal);
  module.attr("Const") = py::cast(vecops::runtime::DimensionDef::constant());
  module.attr("Any") = py::cast(vecops::runtime::DimensionDef::any());
  module.def(
    "Dynamic",
    [](py::object alignment, py::object lower, py::object upper) {
      return vecops::runtime::DimensionDef::dynamic(const_expr(alignment), const_expr(lower), const_expr(upper));
    },
    py::arg("alignment") = 1, py::arg("lower_bound") = std::numeric_limits<std::int64_t>::min(),
    py::arg("upper_bound") = std::numeric_limits<std::int64_t>::max());

  py::class_<vecops::runtime::TensorDTypeDef>(module, "TensorDTypeDef",
                                              "Fixed, named, or anonymous tensor dtype specialization rule.")
    .def(py::init<>())
    .def(py::init<DType>())
    .def(py::init<std::string>())
    .def_property_readonly("dtype", &vecops::runtime::TensorDTypeDef::dtype)
    .def_property_readonly("symbol", &vecops::runtime::TensorDTypeDef::symbol)
    .def_property_readonly("anonymous", &vecops::runtime::TensorDTypeDef::anonymous);

  py::enum_<vecops::runtime::TensorAccess>(module, "TensorAccess",
                                           "Required access mode of one out-style tensor parameter.")
    .value("input", vecops::runtime::TensorAccess::Input)
    .value("output", vecops::runtime::TensorAccess::Output)
    .value("inout", vecops::runtime::TensorAccess::InOut);

  py::class_<vecops::runtime::TensorDef>(module, "TensorDef",
                                         "Raw tensor declaration; prefer vecops.TensorDef or typing annotations.")
    .def(py::init([](std::string name, py::iterable shape, py::iterable strides, py::object dtype, bool optional,
                     vecops::runtime::TensorAccess access, bool output, std::uint32_t device_type,
                     std::int32_t device_index) {
           if (output) {
             if (access != vecops::runtime::TensorAccess::Input)
               throw py::value_error("TensorDef cannot specify both output=True and a non-input access");
             access = vecops::runtime::TensorAccess::Output;
           }
           return vecops::runtime::TensorDef{std::move(name),     dimensions(shape), dimensions(strides),
                                             tensor_dtype(dtype), optional,          access,
                                             device_type,         device_index};
         }),
         py::arg("name"), py::arg("shape"), py::arg("strides"), py::arg("dtype") = py::none(),
         py::arg("optional") = false, py::arg("access") = vecops::runtime::TensorAccess::Input,
         py::arg("output") = false, py::arg("device_type") = static_cast<std::uint32_t>(VECOPS_DEVICE_CPU),
         py::arg("device_index") = -1)
    .def_readwrite("name", &vecops::runtime::TensorDef::name)
    .def_readwrite("shape", &vecops::runtime::TensorDef::shape)
    .def_readwrite("strides", &vecops::runtime::TensorDef::strides)
    .def_readwrite("dtype", &vecops::runtime::TensorDef::dtype)
    .def_readwrite("optional", &vecops::runtime::TensorDef::optional)
    .def_readwrite("access", &vecops::runtime::TensorDef::access)
    .def_property(
      "output", [](const vecops::runtime::TensorDef& self) { return self.writable(); },
      [](vecops::runtime::TensorDef& self, bool value) {
        self.access = value ? vecops::runtime::TensorAccess::Output : vecops::runtime::TensorAccess::Input;
      })
    .def_readwrite("device_type", &vecops::runtime::TensorDef::device_type)
    .def_readwrite("device_index", &vecops::runtime::TensorDef::device_index);

  py::class_<vecops::runtime::ValueDef>(module, "ValueDef",
                                        "Raw runtime scalar declaration; prefer vecops.ValueDef publicly.")
    .def(py::init([](std::string name, DType dtype, py::object default_value) {
           vecops::runtime::ValueDef result{.name = std::move(name), .dtype = dtype};
           if (!default_value.is_none())
             result.default_value = python_scalar(default_value, dtype);
           return result;
         }),
         py::arg("name"), py::arg("dtype"), py::arg("default") = py::none())
    .def_readwrite("name", &vecops::runtime::ValueDef::name)
    .def_readwrite("dtype", &vecops::runtime::ValueDef::dtype)
    .def_readwrite("default", &vecops::runtime::ValueDef::default_value);

  py::class_<KernelDef>(module, "KernelDef",
                        "Raw C++ kernel schema; values are names visible from generated vecops::spec.")
    .def(py::init<std::string, std::vector<vecops::runtime::KernelParameterDef>,
                  std::map<std::string, vecops::runtime::SpecializationType, std::less<>>>(),
         py::arg("name"), py::arg("inputs"),
         py::arg("values") = std::map<std::string, vecops::runtime::SpecializationType, std::less<>>{})
    .def_property_readonly("name", &KernelDef::name)
    .def_property_readonly("inputs",
                           [](const KernelDef& self) {
                             return std::vector<vecops::runtime::KernelParameterDef>(self.parameters().begin(),
                                                                                     self.parameters().end());
                           })
    .def_property_readonly("values", &KernelDef::values)
    .def("validate", [](const KernelDef& self) {
      auto status = self.validate();
      if (!status.ok())
        throw_status(status);
    });

  py::class_<PyKernelCall>(module, "KernelCall",
                           "Raw positional metadata and explicit specialization values for one binding.")
    .def(py::init<PyArgumentMetadata, py::dict>(), py::arg("arguments"), py::arg("values") = py::dict())
    .def(py::init<py::iterable, py::dict>(), py::arg("arguments"), py::arg("values") = py::dict())
    .def_property_readonly("arguments", &PyKernelCall::arguments, py::return_value_policy::reference_internal)
    .def_property_readonly("values", [](const PyKernelCall& self) { return self.call().values(); });
  module.attr("KernelShapes") = module.attr("KernelCall");

  module.def(
    "bind_kernel_call",
    [](const KernelDef& definition, const PyKernelCall& call) {
      auto result = vecops::runtime::bind_kernel_call(definition, call.call());
      if (!result)
        throw_status(result.status());
      py::dict output;
      output["values"] = py::cast(result.value().values);
      output["specialization_key"] = result.value().specialization_key;
      return output;
    },
    py::arg("definition"), py::arg("call"),
    "Bind metadata, infer all named values, and return the specialization identity without execution.");

  py::enum_<vecops::runtime::ArtifactCacheMode>(module, "ArtifactCacheMode",
                                                "Persistent artifact-cache side-effect policy.")
    .value("cache_only", vecops::runtime::ArtifactCacheMode::CacheOnly)
    .value("read_write", vecops::runtime::ArtifactCacheMode::ReadWrite)
    .value("compile_only", vecops::runtime::ArtifactCacheMode::CompileOnly);

  py::class_<Executable, std::shared_ptr<Executable>>(module, "Executable", "A validated vecops kernel artifact.")
    .def_static(
      "load",
      [](const std::string& library_path) {
        auto executable = Executable::load(library_path);
        if (!executable)
          throw_status(executable.status());
        return std::move(executable).value();
      },
      py::arg("library_path"))
    .def_property_readonly("operator_name", &Executable::operator_name)
    .def_property_readonly("specialization_key", &Executable::specialization_key)
    .def(
      "can_invoke",
      [](const Executable& self, const PyArgumentMetadata& arguments) {
        py::gil_scoped_release release;
        return self.can_invoke(arguments.metadata()).ok();
      },
      py::arg("arguments"))
    .def(
      "workspace_size",
      [](const Executable& self, const PyArgumentMetadata& arguments) {
        auto result = [&]() {
          py::gil_scoped_release release;
          return self.workspace_size(arguments.metadata());
        }();
        if (!result)
          throw_status(result.status());
        return result.value();
      },
      py::arg("arguments"))
    .def(
      "invoke",
      [](const Executable& self, const PyArgumentMetadata& arguments) {
        const auto status = [&]() {
          py::gil_scoped_release release;
          return invoke(self, arguments);
        }();
        if (!status.ok())
          throw_status(status);
      },
      py::arg("arguments"));

  py::enum_<compiler::SdkLayout>(module, "SdkLayout", "How generated projects consume the vecops SDK.")
    .value("auto", compiler::SdkLayout::Auto)
    .value("source_tree", compiler::SdkLayout::SourceTree)
    .value("package", compiler::SdkLayout::Package);

  py::enum_<compiler::BuildStage>(module, "BuildStage", "Last stage reached by a kernel build.")
    .value("none", compiler::BuildStage::None)
    .value("validate", compiler::BuildStage::Validate)
    .value("generate", compiler::BuildStage::Generate)
    .value("configure", compiler::BuildStage::Configure)
    .value("build_and_install", compiler::BuildStage::BuildAndInstall)
    .value("complete", compiler::BuildStage::Complete);

  py::class_<compiler::SdkSpec>(module, "SdkSpec", "Location and layout of a vecops SDK.")
    .def(py::init<>())
    .def_readwrite("path", &compiler::SdkSpec::path)
    .def_readwrite("layout", &compiler::SdkSpec::layout);

  py::class_<compiler::ToolchainSpec>(module, "ToolchainSpec", "Native tools and environment for a kernel build.")
    .def(py::init<>())
    .def_readwrite("cmake_program", &compiler::ToolchainSpec::cmake_program)
    .def_readwrite("c_compiler", &compiler::ToolchainSpec::c_compiler)
    .def_readwrite("cxx_compiler", &compiler::ToolchainSpec::cxx_compiler)
    .def_readwrite("toolchain_file", &compiler::ToolchainSpec::toolchain_file)
    .def_readwrite("generator", &compiler::ToolchainSpec::generator)
    .def_readwrite("build_type", &compiler::ToolchainSpec::build_type)
    .def_readwrite("parallel_jobs", &compiler::ToolchainSpec::parallel_jobs)
    .def_readwrite("environment", &compiler::ToolchainSpec::environment);

  py::class_<compiler::KernelCompilerConfig>(module, "KernelCompilerConfig",
                                             "Reusable raw source-kernel compiler policy; prefer vecops.Compiler.")
    .def(py::init<>())
    .def_readwrite("sdk", &compiler::KernelCompilerConfig::sdk)
    .def_readwrite("toolchain", &compiler::KernelCompilerConfig::toolchain)
    .def_readwrite("work_directory", &compiler::KernelCompilerConfig::work_directory)
    .def_readwrite("target_arch", &compiler::KernelCompilerConfig::target_arch)
    .def_readwrite("include_directories", &compiler::KernelCompilerConfig::include_directories)
    .def_readwrite("compile_definitions", &compiler::KernelCompilerConfig::compile_definitions)
    .def_readwrite("compile_options", &compiler::KernelCompilerConfig::compile_options)
    .def_readwrite("link_directories", &compiler::KernelCompilerConfig::link_directories)
    .def_readwrite("link_libraries", &compiler::KernelCompilerConfig::link_libraries)
    .def_readwrite("link_options", &compiler::KernelCompilerConfig::link_options);

  py::class_<compiler::KernelBuildRequest>(module, "KernelBuildRequest", "Complete description of one kernel build.")
    .def(py::init<>())
    .def_readwrite("target_name", &compiler::KernelBuildRequest::target_name)
    .def_readwrite("output_name", &compiler::KernelBuildRequest::output_name)
    .def_readwrite("target_arch", &compiler::KernelBuildRequest::target_arch)
    .def_readwrite("sdk", &compiler::KernelBuildRequest::sdk)
    .def_readwrite("toolchain", &compiler::KernelBuildRequest::toolchain)
    .def_readwrite("sources", &compiler::KernelBuildRequest::sources)
    .def_readwrite("include_directories", &compiler::KernelBuildRequest::include_directories)
    .def_readwrite("compile_definitions", &compiler::KernelBuildRequest::compile_definitions)
    .def_readwrite("compile_options", &compiler::KernelBuildRequest::compile_options)
    .def_readwrite("link_directories", &compiler::KernelBuildRequest::link_directories)
    .def_readwrite("link_libraries", &compiler::KernelBuildRequest::link_libraries)
    .def_readwrite("link_options", &compiler::KernelBuildRequest::link_options)
    .def_readwrite("generated_source_directory", &compiler::KernelBuildRequest::generated_source_directory)
    .def_readwrite("build_directory", &compiler::KernelBuildRequest::build_directory)
    .def_readwrite("artifact_directory", &compiler::KernelBuildRequest::artifact_directory)
    .def_readwrite("cmake_cache_variables", &compiler::KernelBuildRequest::cmake_cache_variables)
    .def_readwrite("reuse_build_directory", &compiler::KernelBuildRequest::reuse_build_directory)
    .def_readwrite("reuse_generated_source_directory", &compiler::KernelBuildRequest::reuse_generated_source_directory);

  py::class_<compiler::CommandResult>(module, "CommandResult", "Captured result of one native build command.")
    .def_readonly("arguments", &compiler::CommandResult::arguments)
    .def_readonly("exit_code", &compiler::CommandResult::exit_code)
    .def_readonly("log_path", &compiler::CommandResult::log_path)
    .def_readonly("output", &compiler::CommandResult::output);

  py::class_<compiler::BuildResult>(module, "BuildResult", "Structured result of a C++ Compiler invocation.")
    .def_readonly("success", &compiler::BuildResult::success)
    .def_readonly("stage", &compiler::BuildResult::stage)
    .def_readonly("error", &compiler::BuildResult::error)
    .def_readonly("configure", &compiler::BuildResult::configure)
    .def_readonly("build_and_install", &compiler::BuildResult::build_and_install)
    .def_readonly("artifact_directory", &compiler::BuildResult::artifact_directory)
    .def_readonly("artifact_files", &compiler::BuildResult::artifact_files)
    .def_readonly("kernel_library", &compiler::BuildResult::kernel_library)
    .def_property_readonly("log", &build_log)
    .def_property_readonly("artifact", [](const compiler::BuildResult& result) { return result.kernel_library; });

  py::class_<compiler::Compiler, std::shared_ptr<compiler::Compiler>>(
    module, "Compiler", "C++ kernel compiler without Python subprocess callbacks.")
    .def(py::init<>())
    .def(py::init<compiler::KernelCompilerConfig>(), py::arg("config"))
    .def(
      "compile",
      [](const compiler::Compiler& self, const compiler::KernelBuildRequest& request) {
        py::gil_scoped_release release;
        return self.compile(request);
      },
      py::arg("request"))
    .def(
      "compile_kernel",
      [](const compiler::Compiler& self, const std::filesystem::path& kernel_file, const KernelDef& definition,
         const PyKernelCall& call) {
        auto result = [&]() {
          py::gil_scoped_release release;
          return self.compile_kernel(kernel_file, definition, call.call());
        }();
        if (!result)
          throw_status(result.status());
        return std::move(result).value();
      },
      py::arg("kernel_file"), py::arg("kernel_def"), py::arg("call"));

  py::class_<PyOperator>(module, "Operator",
                         "Raw native source operator; public vecops.Operator adds dtype and call conveniences.")
    .def(py::init<std::filesystem::path, KernelDef, std::shared_ptr<compiler::Compiler>,
                  vecops::runtime::ArtifactCacheMode, std::filesystem::path, std::string, std::string, std::string>(),
         py::arg("kernel_file"), py::arg("kernel_def"), py::arg("compiler") = nullptr,
         py::arg("cache_mode") = vecops::runtime::ArtifactCacheMode::CacheOnly,
         py::arg("cache_directory") = std::filesystem::path{}, py::arg("namespace_key") = std::string{},
         py::arg("recipe_id") = std::string("source-kernel-v1"), py::arg("source_fingerprint") = std::string{})
    .def_static(
      "_group",
      [](std::vector<std::filesystem::path> kernel_files, std::vector<KernelDef> definitions,
         std::shared_ptr<compiler::Compiler> compiler_instance, vecops::runtime::ArtifactCacheMode cache_mode,
         std::filesystem::path cache_directory, std::string namespace_key, std::vector<std::string> recipe_ids,
         std::vector<std::string> source_fingerprints, std::vector<py::dict> defaults) {
        std::vector<vecops::runtime::SpecializationValues> default_values;
        default_values.reserve(defaults.size());
        for (auto& item : defaults)
          default_values.push_back(specialization_values(std::move(item)));
        return std::make_unique<PyOperator>(std::move(kernel_files), std::move(definitions),
                                            std::move(compiler_instance), cache_mode, std::move(cache_directory),
                                            std::move(namespace_key), std::move(recipe_ids),
                                            std::move(source_fingerprints), std::move(default_values));
      },
      py::arg("kernel_files"), py::arg("kernel_defs"), py::arg("compiler"), py::arg("cache_mode"),
      py::arg("cache_directory"), py::arg("namespace_key"), py::arg("recipe_ids"), py::arg("source_fingerprints"),
      py::arg("default_values"),
      "Create an ordered native group. This is an internal primitive for framework registration.")
    .def_property_readonly("_handle", &PyOperator::handle)
    .def(
      "invoke",
      [](const PyOperator& self, const PyKernelCall& call) {
        const auto status = [&]() {
          py::gil_scoped_release release;
          return self.invoke(call);
        }();
        if (!status.ok())
          throw_status(status);
      },
      py::arg("call"))
    .def(
      "prepare",
      [](const PyOperator& self, const PyKernelCall& call) {
        const auto status = [&]() {
          py::gil_scoped_release release;
          return self.prepare(call);
        }();
        if (!status.ok())
          throw_status(status);
      },
      py::arg("call"))
    .def("__call__", &PyOperator::invoke_values,
         "Invoke with positional NumPy tensor/scalar arguments and specialization keyword arguments.");
}
