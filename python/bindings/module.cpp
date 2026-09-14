/**
 * @file module.cpp
 * @brief pybind11 adapters for the compiler, runtime, and NumPy CPU arrays.
 *
 * Python owners remain alive for every non-owning tensor view. Synchronous
 * compiler and executable calls release the GIL and never call back into
 * Python from their C++ hot paths.
 */

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <iterator>
#include <limits>
#include <atomic>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl/filesystem.h>

#include "vecops/compiler/Compiler.h"
#include "vecops/execution/Parallel.h"
#include "vecops/kernel/Workspace.h"
#include "vecops/runtime/Executable.h"
#include "vecops/runtime/OperatorBridgeAbi.h"
#include "vecops/runtime/Provider.h"
#if defined(VECOPS_PYTHON_HAS_MEMORY)
#include "vecops/execution/MemoryWorkspaceSession.h"
#include "vecops/memory/Memory.h"
#endif

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

/** Metadata identity used by the allocation-free framework hot path. */
struct BridgeValueSignature {
  std::uint32_t kind = VECOPS_VALUE_NONE;
  std::uint32_t dtype = VECOPS_DTYPE_INVALID;
  std::uint32_t device_type = 0;
  std::int32_t device_index = 0;
  std::uint64_t flags = 0;
  std::vector<std::int64_t> sizes;
  std::vector<std::int64_t> strides;
};

struct BridgeSpecializationSignature {
  std::string name;
  std::uint32_t kind = 0;
  std::int64_t integer = 0;
  std::uint32_t dtype = VECOPS_DTYPE_INVALID;
};

struct BridgeCallSignature {
  std::vector<BridgeValueSignature> values;
  std::vector<BridgeSpecializationSignature> specializations;
  vecops::nint_t parallelism = 1;

  [[nodiscard]] static vecops::nint_t call_parallelism(const VecopsCall& call) {
    const VecopsThreadPoolV1* pool = nullptr;
    vecops::nint_t requested = 0;
    if (call.context != nullptr) {
      if (call.context->struct_size >=
          offsetof(VecopsExecutionContext, requested_threads) + sizeof(call.context->requested_threads))
        requested = static_cast<vecops::nint_t>(call.context->requested_threads);
      if (call.context->struct_size >=
          offsetof(VecopsExecutionContext, thread_pool) + sizeof(call.context->thread_pool))
        pool = call.context->thread_pool;
    }
    const auto available = vecops::execution::max_parallelism(pool);
    return requested > 0 ? std::min(requested, available) : available;
  }

  [[nodiscard]] static std::optional<BridgeCallSignature> capture(const VecopsCall& call, std::uint32_t num_specs,
                                                                  const VecopsSpecializationArgument* specs) {
    BridgeCallSignature result;
    result.parallelism = call_parallelism(call);
    result.values.reserve(call.num_values);
    for (std::uint32_t index = 0; index < call.num_values; ++index) {
      const auto& value = call.values[index];
      if (value.struct_size < sizeof(VecopsValue))
        return std::nullopt;
      BridgeValueSignature captured;
      captured.kind = value.kind;
      if (value.kind == VECOPS_VALUE_TENSOR) {
        const auto& tensor = value.value.tensor;
        if (tensor.struct_size < sizeof(VecopsTensorView) ||
            (tensor.rank != 0 && (tensor.sizes == nullptr || tensor.strides == nullptr)))
          return std::nullopt;
        captured.dtype = tensor.dtype;
        captured.device_type = tensor.device_type;
        captured.device_index = tensor.device_index;
        captured.flags = tensor.flags;
        if (tensor.rank != 0) {
          captured.sizes.assign(tensor.sizes, tensor.sizes + tensor.rank);
          captured.strides.assign(tensor.strides, tensor.strides + tensor.rank);
        }
      } else if (value.kind == VECOPS_VALUE_SCALAR) {
        if (value.value.scalar.struct_size < sizeof(VecopsScalar))
          return std::nullopt;
        captured.dtype = value.value.scalar.dtype;
      } else if (value.kind != VECOPS_VALUE_NONE) {
        return std::nullopt;
      }
      result.values.push_back(std::move(captured));
    }
    result.specializations.reserve(num_specs);
    for (std::uint32_t index = 0; index < num_specs; ++index) {
      const auto& spec = specs[index];
      if (spec.struct_size < sizeof(VecopsSpecializationArgument) || spec.name == nullptr)
        return std::nullopt;
      BridgeSpecializationSignature captured;
      captured.name = spec.name;
      captured.kind = spec.kind;
      if (spec.kind == VECOPS_SPECIALIZATION_CONST_INT)
        captured.integer = spec.value.integer;
      else if (spec.kind == VECOPS_SPECIALIZATION_DTYPE)
        captured.dtype = spec.value.dtype;
      else
        return std::nullopt;
      result.specializations.push_back(std::move(captured));
    }
    return result;
  }

  [[nodiscard]] bool matches(const VecopsCall& call, std::uint32_t num_specs,
                             const VecopsSpecializationArgument* specs) const {
    if (parallelism != call_parallelism(call) || call.num_values != values.size() ||
        num_specs != specializations.size())
      return false;
    for (std::uint32_t index = 0; index < call.num_values; ++index) {
      const auto& value = call.values[index];
      const auto& expected = values[index];
      if (value.struct_size < sizeof(VecopsValue) || value.kind != expected.kind)
        return false;
      if (value.kind == VECOPS_VALUE_TENSOR) {
        const auto& tensor = value.value.tensor;
        if (tensor.struct_size < sizeof(VecopsTensorView) || tensor.dtype != expected.dtype ||
            tensor.device_type != expected.device_type || tensor.device_index != expected.device_index ||
            tensor.flags != expected.flags || tensor.rank != expected.sizes.size() ||
            (tensor.rank != 0 && (tensor.sizes == nullptr || tensor.strides == nullptr)))
          return false;
        bool has_elements = true;
        for (std::uint32_t axis = 0; axis < tensor.rank; ++axis) {
          if (tensor.sizes[axis] != expected.sizes[axis] || tensor.strides[axis] != expected.strides[axis])
            return false;
          has_elements = has_elements && tensor.sizes[axis] != 0;
        }
        if (has_elements && tensor.data == nullptr)
          return false;
      } else if (value.kind == VECOPS_VALUE_SCALAR) {
        if (value.value.scalar.struct_size < sizeof(VecopsScalar) || value.value.scalar.dtype != expected.dtype)
          return false;
      }
    }
    for (std::uint32_t index = 0; index < num_specs; ++index) {
      const auto& spec = specs[index];
      const auto& expected = specializations[index];
      if (spec.struct_size < sizeof(VecopsSpecializationArgument) || spec.name == nullptr ||
          spec.kind != expected.kind || expected.name != spec.name)
        return false;
      if ((spec.kind == VECOPS_SPECIALIZATION_CONST_INT && spec.value.integer != expected.integer) ||
          (spec.kind == VECOPS_SPECIALIZATION_DTYPE && spec.value.dtype != expected.dtype))
        return false;
    }
    return true;
  }
};

#if defined(VECOPS_PYTHON_HAS_MEMORY)
const VecopsWorkspaceArenaProvider* current_memory_workspace_provider();
#endif

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
  struct BatchItem {
    const PyOperator* operation = nullptr;
    const PyKernelCall* call = nullptr;
  };

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
#if defined(VECOPS_PYTHON_HAS_MEMORY)
    VecopsExecutionContext memory_context{};
    const VecopsExecutionContext* context = nullptr;
    if (const auto* provider = current_memory_workspace_provider(); provider != nullptr) {
      memory_context.struct_size = sizeof(VecopsExecutionContext);
      memory_context.workspace_provider = provider;
      context = &memory_context;
    }
#endif
    Status last(StatusCode::NotApplicable, "no registered recipe accepted the invocation");
    for (std::size_t index = 0; index < state_->operator_instances.size(); ++index) {
      auto values = state_->default_values[index];
      for (const auto& [name, value] : call.call().values())
        values.insert_or_assign(name, value);
#if defined(VECOPS_PYTHON_HAS_MEMORY)
      auto status = state_->operator_instances[index]->invoke(
        KernelCall(call.call().arguments(), std::move(values)), context);
#else
      auto status = state_->operator_instances[index]->invoke(
        KernelCall(call.call().arguments(), std::move(values)));
#endif
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

  /** Prepare many cache misses through one CMake build per compiler. */
  [[nodiscard]] static Status prepare_batch(std::span<const BatchItem> items, std::size_t parallel_jobs,
                                            vecops::nint_t execution_parallelism);

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
      provider_config.build = [compiler_instance](const vecops::runtime::BoundKernelRecipe& bound) {
        return compiler_instance->compile_kernel(bound);
      };
    }
    auto provider = std::make_shared<vecops::runtime::ArtifactExecutableProvider>(std::move(provider_config));
    state_->operator_instances.push_back(std::make_unique<vecops::runtime::Operator>(
      *definition, std::vector<std::shared_ptr<const vecops::runtime::KernelRecipe>>{recipe},
      std::make_shared<vecops::runtime::OrderedDispatchPolicy>(), provider));
    state_->recipes.push_back(std::move(recipe));
    state_->providers.push_back(std::move(provider));
    state_->compilers.push_back(std::move(compiler_instance));
    state_->default_values.push_back(std::move(default_values));
  }

  void register_handle() {
    handle_ = next_operator_handle.fetch_add(1, std::memory_order_relaxed);
    std::lock_guard lock(operator_registry_mutex);
    operator_registry.emplace(handle_, state_);
  }

  struct State {
    std::vector<std::unique_ptr<vecops::runtime::Operator>> operator_instances;
    std::vector<std::shared_ptr<const vecops::runtime::SourceKernelRecipe>> recipes;
    std::vector<std::shared_ptr<vecops::runtime::ArtifactExecutableProvider>> providers;
    std::vector<std::shared_ptr<compiler::Compiler>> compilers;
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

Status PyOperator::prepare_batch(std::span<const BatchItem> items, std::size_t parallel_jobs,
                                 vecops::nint_t execution_parallelism) {
  struct Planned {
    std::shared_ptr<compiler::Compiler> compiler_instance;
    std::shared_ptr<vecops::runtime::ArtifactExecutableProvider> provider;
    vecops::runtime::BoundKernelRecipe recipe;
  };
  std::vector<Planned> planned;
  planned.reserve(items.size());

  for (const auto& item : items) {
    if (item.operation == nullptr || item.call == nullptr)
      return Status(StatusCode::InvalidArgument, "batch preparation received a null request");
    const auto& operation = *item.operation;
    Status last(StatusCode::NotApplicable, "no registered recipe accepted the specialization");
    bool satisfied = false;
    for (std::size_t index = 0; index < operation.state_->recipes.size(); ++index) {
      auto values = operation.state_->default_values[index];
      for (const auto& [name, value] : item.call->call().values())
        values.insert_or_assign(name, value);
      KernelCall call(item.call->call().arguments(), std::move(values));
      auto match = operation.state_->recipes[index]->match(call);
      if (!match.ok()) {
        if (match.code() == StatusCode::NotApplicable) {
          last = std::move(match);
          continue;
        }
        return match;
      }
      auto bound = operation.state_->recipes[index]->bind(call);
      if (!bound) {
        if (bound.status().code() == StatusCode::NotApplicable) {
          last = bound.status();
          continue;
        }
        return bound.status();
      }
      bound.value() = vecops::runtime::specialize_parallelism(
        std::move(bound).value(), execution_parallelism);
      auto cached = operation.state_->providers[index]->lookup(bound.value());
      if (cached) {
        if (bound.value().source != nullptr && bound.value().source->definition != nullptr &&
            cached.value()->operator_name() != bound.value().source->definition->name())
          return Status(StatusCode::AbiMismatch, "cached batch artifact belongs to a different operator");
        satisfied = true;
        break;
      }
      if (cached.status().code() != StatusCode::NotFound)
        return cached.status();
      if (operation.state_->providers[index]->mode() == vecops::runtime::ArtifactCacheMode::CacheOnly) {
        last = cached.status();
        continue;
      }
      if (operation.state_->compilers[index] == nullptr)
        return Status(StatusCode::InternalError, "batch kernel request has no compiler");
      planned.push_back(
        Planned{operation.state_->compilers[index], operation.state_->providers[index], std::move(bound).value()});
      satisfied = true;
      break;
    }
    if (!satisfied)
      return last;
  }

  std::map<std::string, std::vector<std::size_t>, std::less<>> groups;
  for (std::size_t index = 0; index < planned.size(); ++index) {
    const auto key = planned[index].compiler_instance->batch_key();
    if (key.empty())
      return Status(StatusCode::InternalError, "kernel batch compiler has no source-kernel configuration");
    groups[key].push_back(index);
  }
  for (const auto& [compiler_key, indices] : groups) {
    (void)compiler_key;
    std::vector<vecops::runtime::BoundKernelRecipe> recipes;
    recipes.reserve(indices.size());
    for (const auto index : indices)
      recipes.push_back(planned[index].recipe);
    auto compiled = planned[indices.front()].compiler_instance->compile_kernels(recipes, parallel_jobs);
    if (compiled.size() != indices.size())
      return Status(StatusCode::InternalError, "kernel batch compiler returned the wrong result count");
    for (std::size_t offset = 0; offset < indices.size(); ++offset) {
      if (!compiled[offset])
        return compiled[offset].status();
      auto& plan = planned[indices[offset]];
      auto adopted = plan.provider->adopt(plan.recipe, std::move(compiled[offset]).value());
      if (!adopted)
        return adopted.status();
    }
  }
  return Status::success();
}

#if defined(VECOPS_PYTHON_HAS_MEMORY)

void set_memory_provider_error(VecopsError* error, std::int32_t code, std::string_view message) {
  if (error == nullptr || error->struct_size < sizeof(VecopsError))
    return;
  error->code = code;
  error->message_required = message.size() + 1;
  if (error->message == nullptr || error->message_capacity == 0)
    return;
  const auto copied = std::min(message.size(), error->message_capacity - 1);
  std::memcpy(error->message, message.data(), copied);
  error->message[copied] = '\0';
}

struct BridgeMemoryWorkspaceProvider {
  explicit BridgeMemoryWorkspaceProvider(std::shared_ptr<vecops::execution::WorkspaceArenaProvider> provider)
    : provider(std::move(provider)) {
    static std::atomic<std::uint64_t> next_identity{1};
    abi = {sizeof(VecopsWorkspaceArenaProvider),
           0,
           next_identity.fetch_add(1, std::memory_order_relaxed),
           this,
           capacity,
           allocate,
           release_arena,
           retain,
           release_context,
           VECOPS_WORKSPACE_ARENA_PROVIDER_FLAG_SHARED_ARENAS};
  }

  static std::uint64_t capacity(void* context, std::uint32_t tier) {
    auto& self = *static_cast<BridgeMemoryWorkspaceProvider*>(context);
    const auto logical_tier = tier == VECOPS_WORKSPACE_TIER_FAST ? vecops::execution::WorkspaceTier::Fast
                                                                 : vecops::execution::WorkspaceTier::Slow;
    return static_cast<std::uint64_t>(self.provider->capacity(logical_tier));
  }

  static std::int32_t allocate(void* context, std::uint32_t tier, std::uint64_t bytes, std::uint64_t alignment,
                               VecopsWorkspaceArena* result, VecopsError* error) {
    if (result == nullptr || result->struct_size < sizeof(VecopsWorkspaceArena)) {
      set_memory_provider_error(error, VECOPS_STATUS_INVALID_ARGUMENT, "workspace arena result is truncated");
      return VECOPS_STATUS_INVALID_ARGUMENT;
    }
    if (bytes > static_cast<std::uint64_t>(std::numeric_limits<vecops::nint_t>::max()) ||
        alignment > static_cast<std::uint64_t>(std::numeric_limits<vecops::nint_t>::max())) {
      set_memory_provider_error(error, VECOPS_STATUS_INVALID_ARGUMENT,
                                "workspace arena request exceeds the host address space");
      return VECOPS_STATUS_INVALID_ARGUMENT;
    }
    try {
      auto& self = *static_cast<BridgeMemoryWorkspaceProvider*>(context);
      const auto logical_tier = tier == VECOPS_WORKSPACE_TIER_FAST ? vecops::execution::WorkspaceTier::Fast
                                                                   : vecops::execution::WorkspaceTier::Slow;
      auto arena = self.provider->allocate(logical_tier, static_cast<vecops::nint_t>(bytes),
                                           static_cast<vecops::nint_t>(alignment));
      std::shared_ptr<void>* owner = nullptr;
      if (arena.owner)
        owner = new std::shared_ptr<void>(std::move(arena.owner));
      *result = {sizeof(VecopsWorkspaceArena), 0, arena.data, static_cast<std::uint64_t>(arena.capacity), owner};
      return VECOPS_STATUS_OK;
    } catch (const std::exception& exception) {
      set_memory_provider_error(error, VECOPS_STATUS_EXECUTION_ERROR, exception.what());
      return VECOPS_STATUS_EXECUTION_ERROR;
    } catch (...) {
      set_memory_provider_error(error, VECOPS_STATUS_EXECUTION_ERROR, "workspace arena allocation failed");
      return VECOPS_STATUS_EXECUTION_ERROR;
    }
  }

  static void release_arena(void*, void* owner) {
    delete static_cast<std::shared_ptr<void>*>(owner);
  }

  static void retain(void* context) {
    static_cast<BridgeMemoryWorkspaceProvider*>(context)->references.fetch_add(1, std::memory_order_relaxed);
  }

  static void release_context(void* context) {
    auto* self = static_cast<BridgeMemoryWorkspaceProvider*>(context);
    if (self->references.fetch_sub(1, std::memory_order_acq_rel) == 1)
      delete self;
  }

  std::atomic<std::uint64_t> references{1};
  std::shared_ptr<vecops::execution::WorkspaceArenaProvider> provider;
  VecopsWorkspaceArenaProvider abi{};
};

thread_local std::vector<BridgeMemoryWorkspaceProvider*> memory_workspace_sessions;

const VecopsWorkspaceArenaProvider* current_memory_workspace_provider() {
  return memory_workspace_sessions.empty() ? nullptr : &memory_workspace_sessions.back()->abi;
}

class PyMemoryWorkspaceSession {
public:
  PyMemoryWorkspaceSession(vecops::memory::MemorySystem memory, std::optional<vecops::memory::CpuDomainId> domain,
                           std::uint64_t fast_capacity, std::optional<std::uint64_t> slow_capacity,
                           bool allow_fast_fallback, bool use_large_pages) {
    constexpr auto limit = static_cast<std::uint64_t>(std::numeric_limits<vecops::nint_t>::max());
    if (fast_capacity > limit || (slow_capacity.has_value() && slow_capacity.value() > limit))
      throw py::value_error("workspace capacity exceeds the host address space");
    vecops::execution::MemoryWorkspaceSessionConfig config{
      .domain = domain.has_value() ? vecops::memory::CpuDomainSelector::specific(domain.value())
                                   : vecops::memory::CpuDomainSelector::current(),
      .fast_capacity = static_cast<vecops::nint_t>(fast_capacity),
      .slow_capacity = static_cast<vecops::nint_t>(slow_capacity.value_or(fast_capacity)),
      .allow_fast_fallback = allow_fast_fallback,
      .use_large_pages = use_large_pages,
    };
    session_ = std::make_shared<vecops::execution::MemoryWorkspaceSession>(std::move(memory), config);
    state_ = new BridgeMemoryWorkspaceProvider(session_);
  }

  PyMemoryWorkspaceSession(const PyMemoryWorkspaceSession&) = delete;
  PyMemoryWorkspaceSession& operator=(const PyMemoryWorkspaceSession&) = delete;

  ~PyMemoryWorkspaceSession() {
    close_noexcept();
    BridgeMemoryWorkspaceProvider::release_context(state_);
  }

  PyMemoryWorkspaceSession& enter() {
    if (entered_)
      throw std::runtime_error("memory workspace session cannot be re-entered");
    memory_workspace_sessions.push_back(state_);
    active_ = true;
    entered_ = true;
    return *this;
  }

  bool exit(const py::object&, const py::object&, const py::object&) {
    if (!active_ || memory_workspace_sessions.empty() || memory_workspace_sessions.back() != state_)
      throw std::runtime_error("memory workspace sessions must exit in LIFO order");
    memory_workspace_sessions.pop_back();
    active_ = false;
    session_->close();
    return false;
  }

  void close() {
    if (active_) {
      if (memory_workspace_sessions.empty() || memory_workspace_sessions.back() != state_)
        throw std::runtime_error("memory workspace sessions must close in LIFO order");
      memory_workspace_sessions.pop_back();
      active_ = false;
    }
    entered_ = true;
    session_->close();
  }

  [[nodiscard]] bool closed() const noexcept {
    return session_->closed();
  }

  [[nodiscard]] std::uint64_t fast_capacity() const noexcept {
    return static_cast<std::uint64_t>(session_->fast_capacity());
  }

  [[nodiscard]] std::uint64_t slow_capacity() const noexcept {
    return static_cast<std::uint64_t>(session_->slow_capacity());
  }

  [[nodiscard]] vecops::memory::CpuDomainId domain() const noexcept {
    return session_->domain();
  }

private:
  void close_noexcept() noexcept {
    if (active_) {
      const auto found = std::find(memory_workspace_sessions.rbegin(), memory_workspace_sessions.rend(), state_);
      if (found != memory_workspace_sessions.rend())
        memory_workspace_sessions.erase(std::next(found).base());
    }
    active_ = false;
    session_->close();
  }

  std::shared_ptr<vecops::execution::MemoryWorkspaceSession> session_;
  BridgeMemoryWorkspaceProvider* state_ = nullptr;
  bool active_ = false;
  bool entered_ = false;
};

#endif

Status invoke_registered_operator(std::uint64_t handle, const VecopsCall* call, std::uint32_t num_specs,
                                  const VecopsSpecializationArgument* specs) {
  if (call == nullptr || call->struct_size < sizeof(VecopsCall) || (call->num_values != 0 && call->values == nullptr))
    return Status(StatusCode::InvalidArgument, "framework bridge received an invalid call");
  if (num_specs != 0 && specs == nullptr)
    return Status(StatusCode::InvalidArgument, "framework bridge received no specialization array");

#if defined(VECOPS_PYTHON_HAS_MEMORY)
  VecopsCall routed_call{};
  VecopsExecutionContext routed_context{};
  const bool has_explicit_provider =
    call->context != nullptr && call->context->struct_size >=
      offsetof(VecopsExecutionContext, workspace_provider) + sizeof(call->context->workspace_provider) &&
    call->context->workspace_provider != nullptr;
  if (!has_explicit_provider && call->workspace == nullptr && call->workspace_size == 0) {
    if (const auto* provider = current_memory_workspace_provider(); provider != nullptr) {
      if (call->context != nullptr) {
        const auto copied = std::min<std::size_t>(call->context->struct_size, sizeof(VecopsExecutionContext));
        std::memcpy(&routed_context, call->context, copied);
      }
      routed_context.struct_size = sizeof(VecopsExecutionContext);
      routed_context.workspace_provider = provider;
      routed_call = *call;
      routed_call.context = &routed_context;
      call = &routed_call;
    }
  }
#endif

  // Generated Torch bridges use no external context or workspace.  Retain a
  // small per-thread set of exact metadata specializations so alternating AF3
  // operators can bypass registry locking, schema binding, provider lookup,
  // and owning metadata conversion after one successful slow invocation.
  struct HotEntry {
    std::uint64_t handle = 0;
    std::weak_ptr<PyOperator::State> state;
    std::shared_ptr<Executable> executable;
    BridgeCallSignature signature;
    std::uint64_t last_use = 0;
  };
  constexpr std::size_t hot_capacity = 8;
  static thread_local std::array<std::optional<HotEntry>, hot_capacity> hot_entries;
  static thread_local std::uint64_t hot_clock = 0;
  static const bool hot_path_enabled = [] {
    const char* value = std::getenv("VECOPS_TORCH_PREPARED_CALL");
    return value == nullptr || std::strcmp(value, "0") != 0;
  }();
  const bool provider_context = call->context != nullptr &&
    ((call->context->struct_size >=
        offsetof(VecopsExecutionContext, workspace_provider) + sizeof(call->context->workspace_provider) &&
      call->context->workspace_provider != nullptr) ||
     (call->context->struct_size >=
        offsetof(VecopsExecutionContext, thread_pool) + sizeof(call->context->thread_pool) &&
      call->context->thread_pool != nullptr));
  const bool cacheable = hot_path_enabled && (call->context == nullptr || provider_context) &&
                         call->workspace == nullptr && call->workspace_size == 0;
  if (cacheable) {
    for (auto& slot : hot_entries) {
      if (!slot || slot->handle != handle)
        continue;
      auto owner = slot->state.lock();
      if (!owner) {
        slot.reset();
        continue;
      }
      if (slot->signature.matches(*call, num_specs, specs)) {
        slot->last_use = ++hot_clock;
        return slot->executable->invoke_prevalidated(*call);
      }
    }
  }

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
    auto executable = state->operator_instances[index]->resolve(
      KernelCall(kernel_call.arguments(), std::move(values)), call->context);
    if (executable) {
      auto result =
        executable.value()->invoke(kernel_call.arguments(), call->workspace, call->workspace_size, call->context);
      if (!result.ok())
        return result;
      if (cacheable) {
        auto signature = BridgeCallSignature::capture(*call, num_specs, specs);
        if (signature) {
          std::size_t replace = 0;
          for (std::size_t candidate = 0; candidate < hot_entries.size(); ++candidate) {
            if (!hot_entries[candidate]) {
              replace = candidate;
              break;
            }
            if (hot_entries[candidate]->last_use < hot_entries[replace]->last_use)
              replace = candidate;
          }
          hot_entries[replace].emplace(HotEntry{handle, state, executable.value(), std::move(*signature), ++hot_clock});
        }
      }
      return result;
    }
    if (executable.status().code() != StatusCode::InvalidArgument &&
        executable.status().code() != StatusCode::NotApplicable && executable.status().code() != StatusCode::NotFound)
      return executable.status();
    last = executable.status();
  }
  return last;
}

/** Invokes an executable synchronously with automatically managed workspace. */
Status invoke(const Executable& executable, const PyArgumentMetadata& arguments) {
  const auto required = executable.workspace_size(arguments.metadata());
  if (!required)
    return required.status();
  if (required.value() > std::numeric_limits<std::size_t>::max() ||
      required.value() >
        static_cast<std::uint64_t>(std::numeric_limits<vecops::nint_t>::max() - vecops::vec::DEFAULT_ALIGNMENT)) {
    return Status(StatusCode::InvalidArgument, "kernel workspace size exceeds the host address space");
  }

  vecops::kernel::Workspace storage(static_cast<vecops::nint_t>(required.value()));
  auto view = storage.view();
  void* workspace = required.value() == 0
                      ? nullptr
                      : view.allocate(static_cast<vecops::nint_t>(required.value()), vecops::vec::DEFAULT_ALIGNMENT);
  return executable.invoke(arguments.metadata(), workspace, required.value());
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

#if defined(VECOPS_PYTHON_HAS_MEMORY)
namespace {

void bind_memory(py::module_& module) {
  namespace memory = vecops::memory;

  py::enum_<memory::BackendPreference>(module, "MemoryBackend")
    .value("auto", memory::BackendPreference::Auto)
    .value("system", memory::BackendPreference::System)
    .value("hwloc", memory::BackendPreference::Hwloc);
  py::enum_<memory::MemoryKind>(module, "MemoryKind")
    .value("unknown", memory::MemoryKind::Unknown)
    .value("dram", memory::MemoryKind::DRAM)
    .value("hbm", memory::MemoryKind::HBM)
    .value("cxl", memory::MemoryKind::CXL)
    .value("pmem", memory::MemoryKind::PMEM);
  py::enum_<memory::PlacementIntent>(module, "MemoryPlacement")
    .value("default", memory::PlacementIntent::Default)
    .value("high_bandwidth", memory::PlacementIntent::HighBandwidth)
    .value("low_latency", memory::PlacementIntent::LowLatency)
    .value("exact_target", memory::PlacementIntent::ExactTarget);
  py::enum_<memory::FallbackPolicy>(module, "MemoryFallback")
    .value("none", memory::FallbackPolicy::None)
    .value("to_default", memory::FallbackPolicy::ToDefault);
  py::enum_<memory::RankingObjective>(module, "MemoryRanking")
    .value("bandwidth", memory::RankingObjective::Bandwidth)
    .value("latency", memory::RankingObjective::Latency);

  py::class_<memory::TargetOverride>(module, "MemoryTargetOverride")
    .def(py::init<>())
    .def_readwrite("os_numa_id", &memory::TargetOverride::os_numa_id)
    .def_readwrite("kind", &memory::TargetOverride::kind)
    .def_readwrite("max_managed_bytes", &memory::TargetOverride::max_managed_bytes)
    .def_readwrite("min_free_bytes", &memory::TargetOverride::min_free_bytes);
  py::class_<memory::PathOverride>(module, "MemoryPathOverride")
    .def(py::init<>())
    .def_readwrite("initiator_os_numa_id", &memory::PathOverride::initiator_os_numa_id)
    .def_readwrite("target_os_numa_id", &memory::PathOverride::target_os_numa_id)
    .def_readwrite("bandwidth_mib_s", &memory::PathOverride::bandwidth_mib_s)
    .def_readwrite("latency_ns", &memory::PathOverride::latency_ns);
  py::class_<memory::MemoryConfig>(module, "MemoryConfig")
    .def(py::init<>())
    .def_readwrite("backend", &memory::MemoryConfig::backend)
    .def_readwrite("target_overrides", &memory::MemoryConfig::target_overrides)
    .def_readwrite("path_overrides", &memory::MemoryConfig::path_overrides);
  py::class_<memory::AllocationRequest>(module, "MemoryAllocationRequest")
    .def(py::init<>())
    .def_readwrite("bytes", &memory::AllocationRequest::bytes)
    .def_property(
      "domain",
      [](const memory::AllocationRequest& self) -> py::object {
        if (self.domain.kind == memory::CpuDomainSelector::Kind::Current)
          return py::none();
        return py::int_(self.domain.id);
      },
      [](memory::AllocationRequest& self, py::object value) {
        self.domain = value.is_none() ? memory::CpuDomainSelector::current()
                                      : memory::CpuDomainSelector::specific(py::cast<memory::CpuDomainId>(value));
      })
    .def_readwrite("intent", &memory::AllocationRequest::intent)
    .def_readwrite("objective_rank", &memory::AllocationRequest::objective_rank)
    .def_readwrite("exact_os_numa_id", &memory::AllocationRequest::exact_os_numa_id)
    .def_readwrite("fallback", &memory::AllocationRequest::fallback)
    .def_readwrite("alignment", &memory::AllocationRequest::alignment)
    .def_readwrite("use_large_pages", &memory::AllocationRequest::use_large_pages);

  py::class_<memory::Allocation>(module, "MemoryAllocation", py::buffer_protocol())
    .def_buffer([](memory::Allocation& self) {
      return py::buffer_info(self.data(), 1, py::format_descriptor<std::uint8_t>::format(), 1,
                             {static_cast<py::ssize_t>(self.size())}, {1});
    })
    .def_property_readonly("size", &memory::Allocation::size)
    .def_property_readonly("large_page_bytes", &memory::Allocation::large_page_bytes)
    .def_property_readonly("regular_page_bytes", &memory::Allocation::regular_page_bytes)
    .def_property_readonly("target_id", [](const memory::Allocation& self) -> py::object {
      return self.target().has_value() ? py::cast(self.target().value()) : py::none();
    });

  py::class_<PyMemoryWorkspaceSession>(module, "MemoryWorkspaceSession")
    .def("__enter__", &PyMemoryWorkspaceSession::enter, py::return_value_policy::reference_internal)
    .def("__exit__", &PyMemoryWorkspaceSession::exit)
    .def("close", &PyMemoryWorkspaceSession::close)
    .def_property_readonly("closed", &PyMemoryWorkspaceSession::closed)
    .def_property_readonly("domain", &PyMemoryWorkspaceSession::domain)
    .def_property_readonly("fast_capacity", &PyMemoryWorkspaceSession::fast_capacity)
    .def_property_readonly("slow_capacity", &PyMemoryWorkspaceSession::slow_capacity);

  py::class_<memory::MemorySystem>(module, "MemorySystem")
    .def_static("discover", &memory::MemorySystem::discover, py::arg("config") = memory::MemoryConfig{})
    .def("allocate", &memory::MemorySystem::allocate, py::arg("request"))
    .def("current_cpu_domain", &memory::MemorySystem::current_cpu_domain)
    .def("describe", &memory::MemorySystem::describe)
    .def(
      "workspace_session",
      [](memory::MemorySystem self, std::uint64_t fast_capacity, std::optional<std::uint64_t> slow_capacity,
         std::optional<memory::CpuDomainId> domain, bool allow_fast_fallback, bool use_large_pages) {
        return std::make_unique<PyMemoryWorkspaceSession>(std::move(self), domain, fast_capacity, slow_capacity,
                                                          allow_fast_fallback, use_large_pages);
      },
      py::arg("fast_capacity"), py::arg("slow_capacity") = py::none(), py::arg("domain") = py::none(),
      py::arg("allow_fast_fallback") = true, py::arg("use_large_pages") = true)
    .def("topology",
         [](const memory::MemorySystem& self) {
           const auto& topology = self.topology();
           py::dict result;
           result["backend"] = topology.backend;
           py::list domains;
           for (const auto& domain : topology.cpu_domains) {
             py::dict item;
             item["id"] = domain.id;
             item["os_numa_id"] = domain.os_numa_id.has_value() ? py::cast(domain.os_numa_id.value()) : py::none();
             item["cpu_ids"] = domain.cpu_ids;
             domains.append(std::move(item));
           }
           result["cpu_domains"] = std::move(domains);
           py::list targets;
           for (const auto& target : topology.memory_targets) {
             py::dict item;
             item["id"] = target.id;
             item["os_numa_id"] = target.os_numa_id;
             item["kind"] = memory::to_string(target.kind);
             item["capacity_bytes"] = target.capacity_bytes;
             targets.append(std::move(item));
           }
           result["memory_targets"] = std::move(targets);
           py::list paths;
           for (const auto& path : topology.memory_paths) {
             py::dict item;
             item["initiator"] = path.initiator;
             item["target"] = path.target;
             item["bandwidth_mib_s"] =
               path.bandwidth_mib_s.has_value() ? py::cast(path.bandwidth_mib_s.value()) : py::none();
             item["latency_ns"] = path.latency_ns.has_value() ? py::cast(path.latency_ns.value()) : py::none();
             item["exact_locality"] = path.exact_locality;
             paths.append(std::move(item));
           }
           result["memory_paths"] = std::move(paths);
           return result;
         })
    .def(
      "tiers",
      [](const memory::MemorySystem& self, memory::CpuDomainId domain, memory::RankingObjective objective) {
        py::list result;
        for (const auto& rank : self.tiers(domain, objective).ranks) {
          py::dict item;
          item["rank"] = rank.rank;
          item["targets"] = rank.targets;
          item["value"] =
            rank.representative_value.has_value() ? py::cast(rank.representative_value.value()) : py::none();
          result.append(std::move(item));
        }
        return result;
      },
      py::arg("domain"), py::arg("objective"))
    .def("stats", [](const memory::MemorySystem& self) {
      py::list result;
      for (const auto& stats : self.stats()) {
        py::dict item;
        item["target"] = stats.target;
        item["os_numa_id"] = stats.os_numa_id;
        item["managed_bytes"] = stats.managed_bytes;
        item["peak_managed_bytes"] = stats.peak_managed_bytes;
        item["allocation_count"] = stats.allocation_count;
        item["failed_allocation_count"] = stats.failed_allocation_count;
        item["fallback_count"] = stats.fallback_count;
        item["managed_large_page_bytes"] = stats.managed_large_page_bytes;
        item["peak_managed_large_page_bytes"] = stats.peak_managed_large_page_bytes;
        item["os_free_bytes"] = stats.os_free_bytes.has_value() ? py::cast(stats.os_free_bytes.value()) : py::none();
        item["os_large_page_free_bytes"] = stats.os_large_page_free_bytes.has_value()
                                                   ? py::cast(stats.os_large_page_free_bytes.value())
                                                   : py::none();
        item["budget_remaining_bytes"] =
          stats.budget_remaining_bytes.has_value() ? py::cast(stats.budget_remaining_bytes.value()) : py::none();
        result.append(std::move(item));
      }
      return result;
    });
}

} // namespace
#endif

PYBIND11_MODULE(_C, module) {
  module.doc() = R"doc(Expert-level bindings for the vecops C++ runtime.

Most applications should use :mod:`vecops`, whose Python wrappers normalize
dtypes, derive schemas, and own framework integration. This module mirrors C++
value types for diagnostics, explicit build control, and wrapper implementation;
its constructor-level interfaces intentionally expose native concepts.)doc";
  module.attr("build_api_version") = 5;
#if defined(VECOPS_PYTHON_HAS_MEMORY)
  bind_memory(module);
#endif

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
    .def_readwrite("link_runtime", &compiler::KernelBuildRequest::link_runtime)
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

  py::class_<compiler::KernelBuildBatchRequest>(module, "KernelBuildBatchRequest",
                                                "Targets compiled by one generated CMake project.")
    .def(py::init<>())
    .def_readwrite("tasks", &compiler::KernelBuildBatchRequest::tasks)
    .def_readwrite("generated_source_directory", &compiler::KernelBuildBatchRequest::generated_source_directory)
    .def_readwrite("build_directory", &compiler::KernelBuildBatchRequest::build_directory);

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

  py::class_<compiler::BuildBatchResult>(module, "BuildBatchResult", "Structured result of one CMake build graph.")
    .def_readonly("success", &compiler::BuildBatchResult::success)
    .def_readonly("stage", &compiler::BuildBatchResult::stage)
    .def_readonly("error", &compiler::BuildBatchResult::error)
    .def_readonly("configure", &compiler::BuildBatchResult::configure)
    .def_readonly("build", &compiler::BuildBatchResult::build)
    .def_readonly("tasks", &compiler::BuildBatchResult::tasks);

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
      "compile_batch",
      [](const compiler::Compiler& self, const compiler::KernelBuildBatchRequest& request) {
        py::gil_scoped_release release;
        return self.compile_batch(request);
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

  module.def(
    "prepare_batch",
    [](py::iterable requests, std::size_t parallel_jobs, vecops::nint_t execution_parallelism) {
      if (execution_parallelism <= 0)
        throw py::value_error("execution_parallelism must be positive");
      std::vector<PyOperator::BatchItem> items;
      for (const auto request : requests) {
        const auto tuple = py::cast<py::tuple>(request);
        if (tuple.size() != 2)
          throw py::value_error("prepare_batch entries must be (Operator, KernelCall) pairs");
        items.push_back(
          PyOperator::BatchItem{&py::cast<const PyOperator&>(tuple[0]), &py::cast<const PyKernelCall&>(tuple[1])});
      }
      const auto status = [&]() {
        py::gil_scoped_release release;
        return PyOperator::prepare_batch(items, parallel_jobs, execution_parallelism);
      }();
      if (!status.ok())
        throw_status(status);
    },
    py::arg("requests"), py::arg("parallel_jobs") = 0, py::arg("execution_parallelism") = 1,
    "Prepare native operator cache misses through shared CMake batches.");
}
