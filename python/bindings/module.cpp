// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
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
#include <chrono>
#include <thread>
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
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl/filesystem.h>

#include "vecops/compiler/Compiler.h"
#include "vecops/compiler/details/KernelBatch.h"
#include "vecops/execution/Parallel.h"
#include "vecops/execution/BufferPlan.h"
#include "vecops/execution/ResourceWorkspace.h"
#include "vecops/execution/Grid.h"
#include "vecops/execution/FloatControl.h"
#include "vecops/kernel/Workspace.h"
#include "vecops/runtime/Executable.h"
#include "vecops/runtime/OperatorBridgeAbi.h"
#include "vecops/runtime/Provider.h"
#include "ContentWatch.h"
#if defined(VECOPS_PYTHON_HAS_MEMORY)
#  include "MemoryModelBindings.h"
#  include "vecops/memory/Model.h"
#  include "MemoryRuntimeBindings.h"
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
  // NumPy extension dtypes (e.g. ml_dtypes) use kind V for bfloat16.
  // Match the exact descriptor name, not arbitrary two-byte void records.
  if (item_size == 2 && py::cast<std::string>(dtype.attr("name")) == "bfloat16")
    return DType::BFloat16;
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
 * Element-aligned signed strides are preserved, including overlapping views;
 * the ndarray remains responsible for proving that its reachable storage is
 * valid. Only an empty C-contiguous ndarray whose entire stride tuple is zero
 * receives the legacy dense-stride canonicalization. Explicit empty layouts
 * are preserved even though NumPy marks them both C- and F-contiguous.
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
    // Use ndarray metadata directly: bfloat16 has no PEP-3118 format and
    // requesting a generic Python buffer would reject its valid storage.
    const auto item_size = numpy_array.itemsize();
    if (item_size <= 0) {
      throw py::value_error("NumPy ndarray has an invalid element size");
    }

    view_.data = const_cast<void*>(numpy_array.data());
    view_.device_type = VECOPS_DEVICE_CPU;
    view_.device_index = 0;
    view_.dtype = numpy_dtype(numpy_array.dtype());
    if (static_cast<std::size_t>(item_size) != dtype_size(view_.dtype)) {
      throw py::type_error("NumPy dtype item size does not match the vecops ABI dtype width");
    }
    view_.flags = VECOPS_TENSOR_READ | (writable ? VECOPS_TENSOR_WRITE : 0);
    view_.sizes.reserve(static_cast<std::size_t>(numpy_array.ndim()));
    view_.strides.reserve(static_cast<std::size_t>(numpy_array.ndim()));
    for (py::ssize_t axis = 0; axis < numpy_array.ndim(); ++axis) {
      const auto byte_stride = numpy_array.strides(axis);
      if (byte_stride % item_size != 0) {
        throw py::value_error("vecops does not support non-element-aligned NumPy strides");
      }
      const auto element_stride = byte_stride / item_size;
      view_.sizes.push_back(numpy_array.shape(axis));
      view_.strides.push_back(element_stride);
    }
    if (numpy_array.size() == 0 && (numpy_array.flags() & py::array::c_style) &&
        std::ranges::all_of(view_.strides, [](const auto stride) { return stride == 0; })) {
      // The old buffer adapter canonicalized contiguous empty arrays. NumPy
      // reports zero byte strides for them; keep the dense ABI convention so
      // a length-zero vector still satisfies its declared unit lane stride.
      // Contiguity flags cannot distinguish explicit layouts on empty axes:
      // do not overwrite a user-provided nonzero signed stride tuple.
      vecops::nint_t dense = 1;
      for (py::ssize_t axis = numpy_array.ndim(); axis-- > 0;) {
        view_.strides[static_cast<std::size_t>(axis)] = dense;
        if (axis && __builtin_mul_overflow(dense, view_.sizes[static_cast<std::size_t>(axis)], &dense))
          throw py::value_error("empty NumPy dense stride overflows int64");
      }
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
  bool operator==(const BridgeValueSignature&) const = default;
  std::uint32_t kind = VECOPS_VALUE_NONE;
  std::uint32_t dtype = VECOPS_DTYPE_INVALID;
  std::uint32_t device_type = 0;
  std::int32_t device_index = 0;
  std::uint64_t flags = 0;
  std::vector<std::int64_t> sizes;
  std::vector<std::int64_t> strides;
};

struct BridgeSpecializationSignature {
  bool operator==(const BridgeSpecializationSignature&) const = default;
  std::string name;
  std::uint32_t kind = 0;
  std::int64_t integer = 0;
  std::uint32_t dtype = VECOPS_DTYPE_INVALID;
};

struct BridgeCallSignature {
  bool operator==(const BridgeCallSignature&) const = default;
  std::vector<BridgeValueSignature> values;
  std::vector<BridgeSpecializationSignature> specializations;
  vecops::nint_t parallelism = 1;
  bool runtime_parallelism = false;

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
                                                                  const VecopsSpecializationArgument* specs,
                                                                  bool runtime_parallelism = false) {
    BridgeCallSignature result;
    result.runtime_parallelism = runtime_parallelism;
    result.parallelism = runtime_parallelism ? 1 : call_parallelism(call);
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
    if ((!runtime_parallelism && parallelism != call_parallelism(call)) || call.num_values != values.size() ||
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

// The repeating AF3 trunk visits more than eight distinct source signatures.
// A smaller cache repeatedly rebuilds KernelInstances instead of binding them.
constexpr std::size_t bridge_hot_max_capacity = 128;
std::size_t bridge_hot_capacity() {
  static const auto value = [] {
    const char* text = std::getenv("VECOPS_TORCH_HOT_CAPACITY");
    if (!text || !*text) return bridge_hot_max_capacity;
    char* end = nullptr;
    const auto parsed = std::strtoul(text, &end, 10);
    if (*end || parsed == 0 || parsed > bridge_hot_max_capacity)
      throw std::invalid_argument("VECOPS_TORCH_HOT_CAPACITY must be in [1,128]");
    return static_cast<std::size_t>(parsed);
  }();
  return value;
}
bool bridge_hot_stats_enabled() {
  static const bool value = [] {
    const char* text = std::getenv("VECOPS_TORCH_HOT_STATS");
    return text && std::strcmp(text, "0") != 0;
  }();
  return value;
}
struct BridgeHotStats {
  std::uint64_t hits = 0, misses = 0, prepares = 0, evictions = 0, context_misses = 0;
};
thread_local BridgeHotStats bridge_hot_stats;

struct BridgePlanContext {
  const VecopsThreadPoolV1* pool = nullptr;
  std::uint64_t workspace_identity = 0;
  std::uint32_t requested_threads = 0;
  bool operator==(const BridgePlanContext&) const = default;
  static BridgePlanContext capture(const VecopsCall& call) {
    BridgePlanContext result;
    const auto* context = call.context;
    if (!context) return result;
    if (context->struct_size >= offsetof(VecopsExecutionContext, thread_pool) + sizeof(context->thread_pool))
      result.pool = context->thread_pool;
    if (context->struct_size >= offsetof(VecopsExecutionContext, requested_threads) + sizeof(context->requested_threads))
      result.requested_threads = context->requested_threads;
    if (context->struct_size >= offsetof(VecopsExecutionContext, workspace_provider) + sizeof(context->workspace_provider) &&
        context->workspace_provider)
      result.workspace_identity = context->workspace_provider->identity;
    return result;
  }
};

const VecopsAllocationObserverV1* active_allocation_observer();

struct ManagedEagerPlan {
  std::mutex mutex;
  std::thread::id owner_thread=std::this_thread::get_id();
  bool selected=false;
  uint64_t handle;
  BridgeCallSignature signature;
  BridgePlanContext context;
  std::shared_ptr<vecops::runtime::PreparedKernelLease> lease;
  void close() { std::lock_guard lock(mutex); selected=false;lease.reset(); }
  void unselect() {std::lock_guard lock(mutex);selected=false;}
};
thread_local std::map<uint64_t,std::vector<std::weak_ptr<ManagedEagerPlan>>> managed_eager_plans;
std::mutex prepared_origins_mutex;
std::map<const vecops::runtime::PreparedKernelLease*,
         std::pair<std::weak_ptr<vecops::runtime::PreparedKernelLease>,uint64_t>> prepared_origins;
void remember_origin(const std::shared_ptr<vecops::runtime::PreparedKernelLease>& lease,uint64_t handle) {
  std::lock_guard lock(prepared_origins_mutex);
  for(auto it=prepared_origins.begin();it!=prepared_origins.end();)
    if(it->second.first.expired()) it=prepared_origins.erase(it); else ++it;
  prepared_origins[lease.get()]={lease,handle};
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

struct BridgeHotEntry;
class PyOperator {
public:
  struct BatchItem {
    const PyOperator* operation = nullptr;
    const PyKernelCall* call = nullptr;
  };

  PyOperator(std::filesystem::path kernel_file, KernelDef definition,
             std::shared_ptr<compiler::Compiler> compiler_instance, vecops::runtime::ArtifactCacheMode cache_mode,
             std::filesystem::path cache_directory, std::string namespace_key, std::string recipe_id,
             std::string source_fingerprint, std::function<bool()> validate_inputs = {})
    : definition_(std::make_shared<KernelDef>(std::move(definition))) {
    add_operator(std::move(kernel_file), definition_, std::move(compiler_instance), cache_mode,
                 std::move(cache_directory), std::move(namespace_key), std::move(recipe_id),
                 std::move(source_fingerprint), {}, std::move(validate_inputs));
    register_handle();
  }

  ~PyOperator() {
    std::lock_guard lock(operator_registry_mutex);
    operator_registry.erase(handle_);
  }

  [[nodiscard]] Status invoke(const PyKernelCall& call) const {
    VecopsExecutionContext memory_context{};
    memory_context.struct_size = sizeof(VecopsExecutionContext);
    memory_context.thread_pool = vecops_cpu_thread_pool_v1();
    memory_context.allocation_observer = active_allocation_observer();
    Status last(StatusCode::NotApplicable, "no registered recipe accepted the invocation");
    for (std::size_t index = 0; index < state_->operator_instances.size(); ++index) {
      auto values = state_->default_values[index];
      for (const auto& [name, value] : call.call().values())
        values.insert_or_assign(name, value);
      auto status = state_->operator_instances[index]->invoke(KernelCall(call.call().arguments(), std::move(values)),
                                                              &memory_context);
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

  /** Prepare each unique artifact once per compiler, adopting it into every provider. */
  [[nodiscard]] static Status prepare_batch(std::span<const BatchItem> items, std::size_t parallel_jobs,
                                            vecops::nint_t execution_parallelism);
  static std::shared_ptr<vecops::runtime::PreparedKernelLease> prepare_grid_plan(
      std::uint64_t handle, ArgumentMetadata arguments, vecops::runtime::SpecializationValues specialization,
      const VecopsThreadPoolV1* pool, bool describe_only = false) {
    std::shared_ptr<State> state;
    {
      std::lock_guard lock(operator_registry_mutex);
      auto found = operator_registry.find(handle);
      if (found != operator_registry.end()) state = found->second.lock();
    }
    if (!state) throw std::invalid_argument("native operator is no longer registered");
    VecopsExecutionContext context{};
    context.struct_size = sizeof(context);
    context.thread_pool = pool ? pool : vecops_cpu_thread_pool_v1();
    context.defer_buffer_binding = describe_only;
    context.requested_threads = context.thread_pool->max_parallelism(context.thread_pool->context);
    for (std::size_t i = 0; i < state->operator_instances.size(); ++i) {
      auto values = state->default_values[i];
      for (const auto& [name, value] : specialization) values.insert_or_assign(name, value);
      auto executable = state->operator_instances[i]->resolve(KernelCall(arguments, std::move(values)), &context);
      if (!executable) {
        if (executable.status().code() == StatusCode::NotApplicable || executable.status().code() == StatusCode::NotFound)
          continue;
        throw_status(executable.status());
      }
      if (!executable.value()->module()->plan_descriptor()) return {};
      vecops::runtime::CallFrame frame(arguments, nullptr, 0, &context);
      auto prepared = executable.value()->prepare_prevalidated(frame.abi());
      if (!prepared) throw_status(prepared.status());
      auto lease = prepared.value()->export_abi();
      remember_origin(lease,handle);
      return lease;
    }
    return {};
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
                    vecops::runtime::SpecializationValues default_values, std::function<bool()> validate_inputs = {}) {
    if (state_ == nullptr)
      state_ = std::make_shared<State>();
    auto recipe = std::make_shared<vecops::runtime::SourceKernelRecipe>(std::move(recipe_id), std::move(kernel_file),
                                                                        definition, std::move(source_fingerprint),
                                                                        default_values, std::move(validate_inputs));
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

  friend struct BridgeHotEntry;
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
      bound.value() = vecops::runtime::specialize_parallelism(std::move(bound).value(), execution_parallelism);
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
    // Every call has passed match/bind above, including Dynamic relations.
    // Only the final cold artifact identities are shared, never call checks.
    auto unique = compiler::details::deduplicate_kernel_batch(recipes);
    if (!unique)
      return unique.status();
    auto& batch = unique.value();
    auto compiled = planned[indices.front()].compiler_instance->compile_kernels(batch.recipes, parallel_jobs);
    if (compiled.size() != batch.recipes.size())
      return Status(StatusCode::InternalError, "kernel batch compiler returned the wrong result count");
    for (const auto& executable : compiled)
      if (!executable)
        return executable.status();
    for (std::size_t offset = 0; offset < indices.size(); ++offset) {
      auto& plan = planned[indices[offset]];
      // Copy the shared executable ownership: aliases may use independent
      // providers, namespaces or cache roots, all of which must be populated.
      auto adopted = plan.provider->adopt(plan.recipe, compiled[batch.input_to_artifact[offset]].value());
      if (!adopted)
        return adopted.status();
    }
  }
  return Status::success();
}

struct BridgeHotEntry {
  std::uint64_t handle = 0;
  std::weak_ptr<PyOperator::State> state;
  std::shared_ptr<Executable> executable;
  BridgeCallSignature signature;
  std::uint64_t last_use = 0;
  BridgePlanContext context;
  std::shared_ptr<vecops::runtime::PreparedKernel> prepared;
};
static thread_local std::array<std::optional<BridgeHotEntry>, bridge_hot_max_capacity> hot_entries;
static thread_local std::uint64_t hot_clock = 0;

thread_local std::shared_ptr<vecops::execution::WorkspaceArenaProvider> ambient_workspace_provider;
thread_local std::shared_ptr<const VecopsWorkspaceArenaProvider> ambient_workspace_abi;
void clear_eager_workspace_cache() {
  for(auto& entry:hot_entries) entry.reset();
  ambient_workspace_abi.reset();
  ambient_workspace_provider.reset();
}
const VecopsWorkspaceArenaProvider* current_workspace_abi() {
  auto provider=vecops::execution::default_workspace_arena_provider();
  if(provider!=ambient_workspace_provider) {
    auto abi=vecops::execution::workspace_arena_provider_to_abi(provider);
    ambient_workspace_provider=std::move(provider);
    ambient_workspace_abi=std::move(abi);
  }
  return ambient_workspace_abi.get();
}
thread_local std::uint64_t runtime_scope_top=0;
std::atomic<std::uint64_t> runtime_scope_sequence{1};
class NativeRuntimeScope {
  std::thread::id thread=std::this_thread::get_id();
  std::uint64_t id=0,previous_id=0;
  std::shared_ptr<vecops::execution::WorkspaceArenaProvider> previous;
public:
  explicit NativeRuntimeScope(std::shared_ptr<vecops::memory::model::MemoryRuntime> runtime) {
    auto provider=vecops::execution::resource_workspace_arena_provider(std::move(runtime));
    clear_eager_workspace_cache();
    previous=vecops::execution::exchange_workspace_arena_provider(std::move(provider));
    previous_id=runtime_scope_top;
    id=runtime_scope_sequence.fetch_add(1);
    runtime_scope_top=id;
  }
  NativeRuntimeScope(const NativeRuntimeScope&)=delete;
  NativeRuntimeScope& operator=(const NativeRuntimeScope&)=delete;
  void close() {
    if(!id) return;
    if(thread!=std::this_thread::get_id() || runtime_scope_top!=id)
      throw std::runtime_error("memory runtime scopes must close on their entering thread in LIFO order");
    clear_eager_workspace_cache();
    vecops::execution::exchange_workspace_arena_provider(std::move(previous));
    runtime_scope_top=previous_id;id=0;
  }
  ~NativeRuntimeScope() {
    if(id && thread==std::this_thread::get_id() && runtime_scope_top==id) close();
  }
};

Status invoke_registered_operator(std::uint64_t handle, const VecopsCall* call, std::uint32_t num_specs,
                                  const VecopsSpecializationArgument* specs) {
  if (call == nullptr || call->struct_size < sizeof(VecopsCall) || (call->num_values != 0 && call->values == nullptr))
    return Status(StatusCode::InvalidArgument, "framework bridge received an invalid call");
  if (num_specs != 0 && specs == nullptr)
    return Status(StatusCode::InvalidArgument, "framework bridge received no specialization array");


  // An application-prepared eager signature uses the exact instance whose
  // buffers were included in its model memory program. No second instance or
  // allocator is consulted, even on the first real invocation.
  if(auto it=managed_eager_plans.find(handle);it!=managed_eager_plans.end()) {
    bool active=false;
    for(auto record=it->second.begin();record!=it->second.end();) {
      auto plan=record->lock();
      if(!plan) { record=it->second.erase(record); continue; }
      ++record;
      std::lock_guard lock(plan->mutex);
      if(!plan->lease || !plan->selected) continue;
      active=true;
      if(!(plan->context==BridgePlanContext::capture(*call)) || !plan->signature.matches(*call,num_specs,specs))
        continue;
      const auto& abi=plan->lease->abi();
      std::array<char,2048> message{};VecopsError error{sizeof(error),0,message.data(),message.size(),0};
      auto code=abi.bind(abi.context,call,&error);
      if(code==VECOPS_STATUS_OK) code=abi.replay(abi.context,&error);
      return code==VECOPS_STATUS_OK ? Status::success() : Status(StatusCode::ExecutionError,message.data());
    }
    if(active) return Status(StatusCode::InvalidArgument,"managed eager operator metadata/context was not prepared");
  }

  // Carry the selected authority across the generated-DSO ABI boundary.
  // Managed leases above already carry explicit planned bindings.
  VecopsExecutionContext resource_context{};
  VecopsCall resource_call=*call;
  if(!call->context || call->context->struct_size<offsetof(VecopsExecutionContext,workspace_provider)+sizeof(void*) ||
     !call->context->workspace_provider) {
    if(call->context)
      std::memcpy(&resource_context,call->context,std::min<std::size_t>(call->context->struct_size,sizeof(resource_context)));
    resource_context.struct_size=sizeof(resource_context);
    resource_context.workspace_provider=current_workspace_abi();
    resource_call.context=&resource_context;
    call=&resource_call;
  }

  // Generated Torch bridges use no external context or workspace.  Retain a
  // small per-thread set of exact metadata specializations so alternating AF3
  // operators can bypass registry locking, schema binding, provider lookup,
  // and owning metadata conversion after one successful slow invocation.
  static const bool hot_path_enabled = [] {
    const char* value = std::getenv("VECOPS_TORCH_PREPARED_CALL");
    return value == nullptr || std::strcmp(value, "0") != 0;
  }();
  const bool provider_context =
    call->context != nullptr && ((call->context->struct_size >= offsetof(VecopsExecutionContext, workspace_provider) +
                                                                  sizeof(call->context->workspace_provider) &&
                                  call->context->workspace_provider != nullptr) ||
                                 (call->context->struct_size >= offsetof(VecopsExecutionContext, thread_pool) +
                                                                  sizeof(call->context->thread_pool) &&
                                  call->context->thread_pool != nullptr));
  const bool cacheable = hot_path_enabled && (call->context == nullptr || provider_context) &&
                         call->workspace == nullptr && call->workspace_size == 0;
  if (cacheable) {
    for (std::size_t i = 0; i < bridge_hot_capacity(); ++i) {
      auto& slot = hot_entries[i];
      if (!slot || slot->handle != handle)
        continue;
      auto owner = slot->state.lock();
      if (!owner) {
        slot.reset();
        continue;
      }
      if (slot->signature.matches(*call, num_specs, specs)) {
        slot->last_use = ++hot_clock;
        if (slot->prepared) {
          if (!(slot->context == BridgePlanContext::capture(*call))) {
            if (bridge_hot_stats_enabled()) ++bridge_hot_stats.context_misses;
            slot.reset();
            continue;
          }
          auto bound = slot->prepared->bind(*call);
          if (bound.ok()) {
            if (bridge_hot_stats_enabled()) ++bridge_hot_stats.hits;
            return slot->prepared->replay();
          }
          // Equal tensor metadata can occur in a new workspace or provider
          // context. Retire the old instance and prepare in the current one.
          if (bound.code() == StatusCode::InvalidArgument || bound.code() == StatusCode::NotApplicable) {
            if (bridge_hot_stats_enabled()) ++bridge_hot_stats.context_misses;
            slot.reset();
            continue;
          }
          return bound;
        }
        if (bridge_hot_stats_enabled()) ++bridge_hot_stats.hits;
        return slot->executable->invoke_prevalidated(*call);
      }
    }
  }
  if (cacheable && bridge_hot_stats_enabled()) ++bridge_hot_stats.misses;

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
    auto executable =
      state->operator_instances[index]->resolve(KernelCall(kernel_call.arguments(), std::move(values)), call->context);
    if (executable) {
      std::shared_ptr<vecops::runtime::PreparedKernel> prepared;
      if (executable.value()->module()->plan_descriptor()) {
        if (bridge_hot_stats_enabled()) ++bridge_hot_stats.prepares;
        auto plan = executable.value()->prepare_prevalidated(*call);
        if (!plan) return plan.status();
        prepared = std::move(plan).value();
      }
      auto result = prepared ? prepared->replay() :
        executable.value()->invoke(kernel_call.arguments(), call->workspace, call->workspace_size, call->context);
      if (!result.ok())
        return result;
      if (cacheable) {
        auto signature = BridgeCallSignature::capture(*call, num_specs, specs,
                                                      state->recipes[index]->definition().runtime_parallelism());
        if (signature) {
          std::size_t replace = 0;
          for (std::size_t candidate = 0; candidate < bridge_hot_capacity(); ++candidate) {
            if (!hot_entries[candidate]) {
              replace = candidate;
              break;
            }
            if (hot_entries[candidate]->last_use < hot_entries[replace]->last_use)
              replace = candidate;
          }
          if (bridge_hot_stats_enabled() && hot_entries[replace]) ++bridge_hot_stats.evictions;
          hot_entries[replace].emplace(BridgeHotEntry{handle, state, executable.value(), std::move(*signature), ++hot_clock,
                                               BridgePlanContext::capture(*call),
                                               std::move(prepared)});
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
  VecopsExecutionContext context{};
  context.struct_size=sizeof(context);
  context.workspace_provider=current_workspace_abi();
  return executable.invoke(arguments.metadata(), workspace, required.value(), &context);
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
  Status result;
  try {
    result=invoke_registered_operator(handle, call, num_specialization_values, specialization_values);
  } catch(const std::exception& failure) {
    result=Status(StatusCode::ExecutionError,failure.what());
  } catch(...) {
    result=Status(StatusCode::ExecutionError,"native operator bridge failed");
  }
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

namespace {

/** Native-side workspace allocation collector (no GIL in the callback). */
struct WorkspaceAllocationCollector {
  struct Record {
    std::string recipe;
    std::string site;
    std::uint64_t scope_hash = 0;
    std::uint64_t bytes = 0;
    std::uint64_t alignment = 0;
    std::uint64_t replicas = 0;
    std::int32_t domain = 0;
    std::int32_t placement = 0;
    std::int32_t tier = 0;
    double estimated_traffic_bytes = 0.0;
  };

  std::mutex mutex;
  std::vector<Record> records;
  std::atomic<std::size_t> dropped{0}; // capacity and allocation failures
  std::atomic<bool> enabled{false};
  static constexpr std::size_t capacity = 1u << 20;
  struct MemoryEvent {
    std::uint64_t timestamp_ns, worker, allocation_id, bytes;
    std::uint32_t action;
    std::string site;
  };
  std::vector<MemoryEvent> memory_events;
  std::map<const void*, std::uint64_t> allocation_ids;
  std::uint64_t next_allocation_id = 1;

  static void on_memory(void* context, const VecopsMemoryEventV1* record) noexcept {
    auto* self = static_cast<WorkspaceAllocationCollector*>(context);
    if (self == nullptr || !self->enabled.load(std::memory_order_relaxed) || record == nullptr ||
        record->struct_size < sizeof(*record))
      return;
    try {
      std::lock_guard guard(self->mutex);
      if (self->memory_events.size() >= capacity) {
        ++self->dropped;
        return;
      }
      std::uint64_t id = 0;
      if (record->action < 2) {
        auto& stored = self->allocation_ids[record->allocation];
        if (record->action == 0 || !stored)
          stored = self->next_allocation_id++;
        id = stored;
      }
      const auto timestamp =
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
          .count();
      self->memory_events.push_back(MemoryEvent{static_cast<std::uint64_t>(timestamp),
                                                std::hash<std::thread::id>{}(std::this_thread::get_id()), id,
                                                record->bytes, record->action, record->site ? record->site : ""});
      if (record->action == 1)
        self->allocation_ids.erase(record->allocation);
    } catch (...) { /* Diagnostics cannot throw across the C ABI. */
      ++self->dropped;
    }
  }

  static void on_allocation(void* context, const VecopsWorkspaceAllocationV1* record) noexcept {
    auto* self = static_cast<WorkspaceAllocationCollector*>(context);
    if (self == nullptr || !self->enabled.load(std::memory_order_relaxed))
      return;
    if (record == nullptr || record->struct_size < sizeof(*record))
      return;
    try {
      std::lock_guard<std::mutex> guard(self->mutex);
      if (self->records.size() >= capacity) {
        ++self->dropped;
        return;
      }
      self->records.push_back(Record{
        record->recipe != nullptr ? record->recipe : "",
        record->site != nullptr ? record->site : "",
        record->scope_hash,
        record->bytes,
        record->alignment,
        record->replicas,
        record->domain,
        record->placement,
        record->tier,
        record->estimated_traffic_bytes,
      });
    } catch (...) {
      ++self->dropped;
    }
  }

  VecopsAllocationObserverV1 observer() noexcept {
    VecopsAllocationObserverV1 observer{};
    observer.struct_size = sizeof(observer);
    observer.user_context = this;
    observer.on_workspace_allocation = &WorkspaceAllocationCollector::on_allocation;
    observer.on_memory_event = &WorkspaceAllocationCollector::on_memory;
    return observer;
  }
};

WorkspaceAllocationCollector g_workspace_collector;
const VecopsAllocationObserverV1* active_allocation_observer() {
  static const auto observer = g_workspace_collector.observer();
  return g_workspace_collector.enabled.load(std::memory_order_relaxed) ? &observer : nullptr;
}

} // namespace

ArgumentMetadata memory_operator_arguments(const py::list& records) {
  std::vector<Value> arguments; arguments.reserve(records.size());
  for(const auto item:records) {
    if(item.is_none()) {arguments.emplace_back(std::monostate{});continue;}
    auto record=py::cast<py::dict>(item);auto dtype=py::cast<DType>(record["dtype"]);
    if(py::cast<std::string>(record["kind"])=="tensor")
      arguments.emplace_back(TensorView{.data=reinterpret_cast<void*>(py::cast<std::uintptr_t>(record["data"])),
        .device_index=record.contains("device_index") ? py::cast<int32_t>(record["device_index"]) : 0,
        .dtype=dtype,.sizes=py::cast<std::vector<int64_t>>(record["sizes"]),
        .strides=py::cast<std::vector<int64_t>>(record["strides"]),.flags=py::cast<uint64_t>(record["flags"])});
    else arguments.emplace_back(python_scalar(record["value"],dtype));
  }
  return ArgumentMetadata(std::move(arguments));
}

// Physical binding ownership crosses the exported prepared C ABI as well as
// Python. The record, rather than the Python wrapper, survives foreign leases.
struct PreparedPythonOwner {
  std::atomic<unsigned> refs{1};
  PyObject* value;
  explicit PreparedPythonOwner(py::handle object) : value(object.ptr()) { Py_INCREF(value); }
  static void retain(void* p) { static_cast<PreparedPythonOwner*>(p)->refs.fetch_add(1,std::memory_order_relaxed); }
  static void release(void* p) {
    auto* self=static_cast<PreparedPythonOwner*>(p);
    if (self->refs.fetch_sub(1,std::memory_order_acq_rel)!=1) return;
    if (Py_IsInitialized()) { py::gil_scoped_acquire gil; Py_DECREF(self->value); }
    delete self;
  }
};

PYBIND11_MODULE(_C, module) {
#if defined(VECOPS_PYTHON_HAS_MEMORY)
  auto memory = module.def_submodule("memory", "Generic memory resources and native runtime.");
  bind_memory_model_types(memory);
  bind_memory_runtime(memory);
  memory.def("default_workspace_runtime", &vecops::execution::default_workspace_runtime);
  py::class_<NativeRuntimeScope>(memory,"_WorkspaceRuntimeScope")
    .def(py::init<std::shared_ptr<vecops::memory::model::MemoryRuntime>>())
    .def("close", &NativeRuntimeScope::close);
#endif
  module.def("_clear_eager_workspace_cache",&clear_eager_workspace_cache);
  module.doc() = R"doc(Expert-level bindings for the vecops C++ runtime.

Most applications should use :mod:`vecops`, whose Python wrappers normalize
dtypes, derive schemas, and own framework integration. This module mirrors C++
value types for diagnostics, explicit build control, and wrapper implementation;
its constructor-level interfaces intentionally expose native concepts.)doc";
  module.attr("build_api_version") = 11;
  py::class_<vecops::runtime::PreparedKernelLease, std::shared_ptr<vecops::runtime::PreparedKernelLease>>(
      module, "PreparedKernelLease")
    .def_property_readonly("abi_address", [](const vecops::runtime::PreparedKernelLease& self) {
      return reinterpret_cast<std::uintptr_t>(&self.abi());
    })
    .def_property_readonly("workers", [](const vecops::runtime::PreparedKernelLease& self) { return self.abi().workers; })
    .def_property_readonly("grids", [](const vecops::runtime::PreparedKernelLease& self) { return self.abi().grids; })
    .def_property_readonly("inplace_contracts", [](const vecops::runtime::PreparedKernelLease& self) {
      std::vector<std::pair<uint32_t, uint32_t>> result;
      const auto& abi = self.abi();
      if (abi.struct_size >= sizeof(VecopsPreparedKernelV1)) {
        for (uint32_t i = 0; i < abi.inplace_count; ++i)
          result.emplace_back(abi.inplace[i].input, abi.inplace[i].output);
      }
      return result;
    })
    .def_property_readonly("buffer_requirements", [](const vecops::runtime::PreparedKernelLease& self) {
      const auto& abi=self.abi();
      if (abi.struct_size < sizeof(abi) || !abi.activate_buffers || !abi.suspend_buffers)
        throw std::runtime_error("artifact has no explicit buffer binding contract");
      py::list result;
      for (uint32_t i=0; i<abi.buffer_count; ++i) {
        const auto& b=abi.buffers[i]; py::dict row;
        row["name"]=b.name; row["bytes"]=b.bytes; row["alignment"]=b.alignment;
        row["replicas"]=b.replicas; row["replica_stride"]=b.replica_stride;
        row["domain"]=b.domain == static_cast<uint32_t>(vecops::execution::BufferDomain::Global) ? "global" : "worker_local";
        row["traffic_bytes"]=b.estimated_traffic_bytes; row["retained"]=bool(b.retained);
        py::list times;
        for (size_t j=0;j<b.lifetime_count;++j) times.append(py::make_tuple(b.lifetimes[j].begin,b.lifetimes[j].end));
        row["lifetimes"]=std::move(times); result.append(std::move(row));
      }
      return result;
    })
    .def("activate_buffers", [](vecops::runtime::PreparedKernelLease& self, py::dict values, py::object owner) {
      if(owner.is_none()) throw py::value_error("physical buffer owner is required");
      const auto& abi=self.abi();
      if(abi.struct_size<sizeof(abi) || !abi.activate_buffers)
        throw std::runtime_error("artifact has no explicit buffer activation contract");
      std::vector<std::string> names; names.reserve(values.size());
      std::vector<VecopsKernelBufferBindingV1> bindings; bindings.reserve(values.size());
      for (auto [name,value] : values) {
        names.push_back(py::cast<std::string>(name));
        auto row=py::cast<py::dict>(value);
        bindings.push_back({names.back().c_str(),reinterpret_cast<void*>(py::cast<std::uintptr_t>(row["address"])),
          py::cast<uint64_t>(row["bytes"]),py::cast<uint64_t>(row["replica_stride"])});
      }
      auto* lifetime=new PreparedPythonOwner(owner);
      const VecopsBufferOwnerV1 token{lifetime,&PreparedPythonOwner::retain,&PreparedPythonOwner::release};
      std::array<char,2048> message{};
      VecopsError error{sizeof(error),0,message.data(),message.size(),0};
      int32_t status;
      {
        py::gil_scoped_release release;
        status=abi.activate_buffers(abi.context,bindings.data(),bindings.size(),&token,&error);
      }
      PreparedPythonOwner::release(lifetime);
      if(status != VECOPS_STATUS_OK) throw std::runtime_error(message.data());
    })
    .def("suspend_buffers", [](vecops::runtime::PreparedKernelLease& self) {
      const auto& abi=self.abi();
      if(abi.struct_size<sizeof(abi) || !abi.suspend_buffers)
        throw std::runtime_error("artifact has no explicit buffer suspension contract");
      std::array<char,2048> message{};
      VecopsError error{sizeof(error),0,message.data(),message.size(),0};
      int32_t status;
      {
        py::gil_scoped_release release;
        status=abi.suspend_buffers(abi.context,&error);
      }
      if(status != VECOPS_STATUS_OK) throw std::runtime_error(message.data());
    });
  module.def("prepare_operator_plan", [](std::uint64_t handle, py::list records, py::dict values,
                                         std::uintptr_t pool, bool describe_only) {
    auto arguments=memory_operator_arguments(records);
    auto specialization = specialization_values(std::move(values));
    py::gil_scoped_release release;
    return PyOperator::prepare_grid_plan(handle, std::move(arguments), std::move(specialization),
                                         reinterpret_cast<const VecopsThreadPoolV1*>(pool), describe_only);
  }, py::arg("handle"), py::arg("records"), py::arg("values"), py::arg("pool"),
     py::arg("describe_only") = false);
  py::class_<ManagedEagerPlan,std::shared_ptr<ManagedEagerPlan>>(module,"ManagedEagerPlan")
    .def("close",[](ManagedEagerPlan& self) {py::gil_scoped_release release;self.close();})
    .def("unselect",&ManagedEagerPlan::unselect)
    .def("select",[](ManagedEagerPlan& self) {
      if(self.owner_thread!=std::this_thread::get_id())
        throw std::runtime_error("eager plan selection requires its preparation thread");
      auto it=managed_eager_plans.find(self.handle);
      if(it==managed_eager_plans.end()) throw std::runtime_error("eager plan is not registered");
      {std::lock_guard lock(self.mutex);if(!self.lease) throw std::runtime_error("eager plan is closed");}
      for(auto& weak:it->second) {
        auto other=weak.lock();
        if(!other) continue;
        std::lock_guard lock(other->mutex);
        if(other->context==self.context && other->signature==self.signature)
          other->selected=other.get()==&self;
      }
    });
  module.def("register_prepared_operator_plan",[](std::shared_ptr<vecops::runtime::PreparedKernelLease> lease,
      uint64_t handle,py::list records,py::dict values,uintptr_t pool) {
    {
      std::lock_guard lock(prepared_origins_mutex);
      auto it=prepared_origins.find(lease.get());
      if(it==prepared_origins.end() || it->second.first.lock()!=lease || it->second.second!=handle)
        throw py::value_error("prepared plan belongs to another operator");
    }
    auto arguments=memory_operator_arguments(records);
    std::vector<std::string> names; names.reserve(values.size());
    std::vector<VecopsSpecializationArgument> specs;specs.reserve(values.size());
    for(auto [name,value]:values) {
      names.push_back(py::cast<std::string>(name));
      VecopsSpecializationArgument item{}; item.struct_size=sizeof(item);item.name=names.back().c_str();
      if(py::isinstance<py::int_>(value)) {item.kind=VECOPS_SPECIALIZATION_CONST_INT;item.value.integer=py::cast<int64_t>(value);}
      else {item.kind=VECOPS_SPECIALIZATION_DTYPE;item.value.dtype=static_cast<uint32_t>(py::cast<DType>(value));}
      specs.push_back(item);
    }
    VecopsExecutionContext context{};context.struct_size=sizeof(context);
    context.thread_pool=reinterpret_cast<const VecopsThreadPoolV1*>(pool);
    if(!context.thread_pool) throw py::value_error("eager registration requires a thread pool");
    context.requested_threads=context.thread_pool->max_parallelism(context.thread_pool->context);
    vecops::runtime::CallFrame frame(arguments,nullptr,0,&context);
    auto signature=BridgeCallSignature::capture(frame.abi(),specs.size(),specs.data());
    if(!signature) throw py::value_error("invalid eager metadata signature");
    auto plan=std::make_shared<ManagedEagerPlan>();plan->handle=handle;plan->signature=std::move(*signature);
    plan->context=BridgePlanContext::capture(frame.abi());plan->lease=std::move(lease);
    auto& installed=managed_eager_plans[handle];
    std::erase_if(installed,[](const auto& previous){return previous.expired();});
    installed.push_back(plan);
    return plan;
  });
  module.def("bridge_hot_cache_stats", [] {
    py::dict result;
    result["enabled"] = bridge_hot_stats_enabled(); result["capacity"] = bridge_hot_capacity();
    result["hits"] = bridge_hot_stats.hits; result["misses"] = bridge_hot_stats.misses;
    result["prepares"] = bridge_hot_stats.prepares; result["evictions"] = bridge_hot_stats.evictions;
    result["context_misses"] = bridge_hot_stats.context_misses;
    return result;
  }, "Main-calling-thread source-bridge cache counters; enable before use with VECOPS_TORCH_HOT_STATS=1.");
  module.def("reset_bridge_hot_cache_stats", [] { bridge_hot_stats = {}; },
             "Reset this thread's counters without discarding prepared entries.");
  module.def(
    "content_watch_directory", [](const std::filesystem::path& path) { return content_watch.add(path); },
    py::arg("path"));
  module.def("content_watch_epoch", [] { return content_watch.epoch(); });
  module.def("cpu_thread_limit", &vecops_cpu_thread_limit, py::arg("threads"),
             "Set the calling thread's direct CPU executor budget and return its previous value.");
  module.def("grid_workers_available", &vecops::execution::grid_workers_available, py::arg("capacity"),
             py::arg("requested"), py::arg("programs"), py::arg("program_cycles"), py::arg("worker_cycles"),
             "Metadata-only access to the actual native grid worker selector.");
  module.def(
    "cpu_thread_pool_capacity",
    [] {
      const auto* pool = vecops_cpu_thread_pool_v1();
      return pool->max_parallelism(pool->context);
    },
    "Return the direct CPU executor's current effective capacity without creating workers.");
  module.def(
    "cpu_thread_pool_address", [] { return reinterpret_cast<std::uintptr_t>(vecops_cpu_thread_pool_v1()); },
    "Return the stable adaptive CPU executor ABI address.");
  module.def("flush_subnormals_supported", &vecops::execution::flush_subnormals_supported,
             "Whether this build can set hardware subnormal flushing.");
  module.def("current_thread_flush_subnormals", &vecops::execution::current_thread_flush_subnormals,
             "Return the calling thread's hardware subnormal flush mode.");
  module.def("set_current_thread_flush_subnormals", &vecops::execution::set_current_thread_flush_subnormals,
             py::arg("enabled"), "Set the calling thread's hardware subnormal flush mode.");
  module.def(
    "native_thread_pool_address",
    [](std::uint32_t threads) { return reinterpret_cast<std::uintptr_t>(vecops_native_thread_pool_v1(threads)); },
    py::arg("threads"), "Return the process-local persistent thread-pool ABI address.");
  module.def(
    "native_thread_pool_begin_active", [](std::uint32_t threads) { vecops_native_thread_pool_begin_active(threads); },
    py::arg("threads"), "Keep native workers active across a bounded sequence of submissions.");
  module.def(
    "native_thread_pool_end_active", [] { vecops_native_thread_pool_end_active(); },
    "End one bounded native-worker active region.");

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

  using DimensionExpr = vecops::runtime::DimensionExpr;
  py::class_<DimensionExpr>(module, "DimensionExpr", "Postfix integer expression for exact tensor dimensions.")
    .def_static("literal", &DimensionExpr::literal, py::arg("value"))
    .def_static("const_symbol", &DimensionExpr::const_symbol, py::arg("symbol"))
    .def_static("runtime_symbol", &DimensionExpr::runtime_symbol, py::arg("symbol"))
    .def_static("add",
                [](DimensionExpr left, DimensionExpr right) {
                  return DimensionExpr::binary(DimensionExpr::Op::Add, std::move(left), std::move(right));
                })
    .def_static("sub",
                [](DimensionExpr left, DimensionExpr right) {
                  return DimensionExpr::binary(DimensionExpr::Op::Sub, std::move(left), std::move(right));
                })
    .def_static("mul",
                [](DimensionExpr left, DimensionExpr right) {
                  return DimensionExpr::binary(DimensionExpr::Op::Mul, std::move(left), std::move(right));
                })
    .def_static("floordiv",
                [](DimensionExpr left, DimensionExpr right) {
                  return DimensionExpr::binary(DimensionExpr::Op::FloorDiv, std::move(left), std::move(right));
                })
    .def_static("mod",
                [](DimensionExpr left, DimensionExpr right) {
                  return DimensionExpr::binary(DimensionExpr::Op::Mod, std::move(left), std::move(right));
                })
    .def_static("minimum",
                [](DimensionExpr left, DimensionExpr right) {
                  return DimensionExpr::binary(DimensionExpr::Op::Min, std::move(left), std::move(right));
                })
    .def_static("maximum", [](DimensionExpr left, DimensionExpr right) {
      return DimensionExpr::binary(DimensionExpr::Op::Max, std::move(left), std::move(right));
    });

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
    .value("dynamic", vecops::runtime::DimensionDef::Kind::Dynamic)
    .value("expression", vecops::runtime::DimensionDef::Kind::Expression);

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
    .def_static(
      "expression",
      [](vecops::runtime::DimensionExpr expression) {
        return vecops::runtime::DimensionDef::expression(std::move(expression));
      },
      py::arg("expression"))
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
    .def_property("runtime_parallelism", &KernelDef::runtime_parallelism, &KernelDef::set_runtime_parallelism)
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

  py::class_<compiler::BuildEvent>(module, "BuildEvent")
    .def_readonly("phase", &compiler::BuildEvent::phase)
    .def_readonly("state", &compiler::BuildEvent::state)
    .def_readonly("log_path", &compiler::BuildEvent::log_path)
    .def_readonly("elapsed_seconds", &compiler::BuildEvent::elapsed_seconds)
    .def_readonly("exit_code", &compiler::BuildEvent::exit_code);
  py::class_<compiler::ToolchainSpec>(module, "ToolchainSpec", "Native tools and environment for a kernel build.")
    .def(py::init<>())
    .def_readwrite("cmake_program", &compiler::ToolchainSpec::cmake_program)
    .def_readwrite("c_compiler", &compiler::ToolchainSpec::c_compiler)
    .def_readwrite("cxx_compiler", &compiler::ToolchainSpec::cxx_compiler)
    .def_readwrite("toolchain_file", &compiler::ToolchainSpec::toolchain_file)
    .def_readwrite("generator", &compiler::ToolchainSpec::generator)
    .def_readwrite("build_type", &compiler::ToolchainSpec::build_type)
    .def_readwrite("parallel_jobs", &compiler::ToolchainSpec::parallel_jobs)
    .def_readwrite("environment", &compiler::ToolchainSpec::environment)
    .def_readwrite("output_mode", &compiler::ToolchainSpec::output_mode)
    .def_readwrite("observer", &compiler::ToolchainSpec::observer);

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
                  vecops::runtime::ArtifactCacheMode, std::filesystem::path, std::string, std::string, std::string,
                  std::function<bool()>>(),
         py::arg("kernel_file"), py::arg("kernel_def"), py::arg("compiler") = nullptr,
         py::arg("cache_mode") = vecops::runtime::ArtifactCacheMode::CacheOnly,
         py::arg("cache_directory") = std::filesystem::path{}, py::arg("namespace_key") = std::string{},
         py::arg("recipe_id") = std::string("source-kernel-v1"), py::arg("source_fingerprint") = std::string{},
         py::arg("validate_inputs") = std::function<bool()>{})
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

  module.def(
    "set_workspace_allocation_tracking",
    [](bool enabled) {
      g_workspace_collector.enabled.store(enabled, std::memory_order_relaxed);
      if (enabled) {
        std::lock_guard<std::mutex> guard(g_workspace_collector.mutex);
        g_workspace_collector.records.clear();
        g_workspace_collector.dropped = 0;
        g_workspace_collector.memory_events.clear();
        g_workspace_collector.allocation_ids.clear();
        g_workspace_collector.next_allocation_id = 1;
      }
      const auto observer = g_workspace_collector.observer();
      return py::make_tuple(reinterpret_cast<std::uintptr_t>(observer.on_workspace_allocation),
                            reinterpret_cast<std::uintptr_t>(observer.user_context),
                            static_cast<std::uintptr_t>(sizeof(VecopsAllocationObserverV1)));
    },
    py::arg("enabled") = true,
    "Enable the native workspace allocation collector and return "
    "(callback_address, user_context_address, observer_struct_size) for "
    "installing into bridge DSOs via their exported setter.");

  module.def(
    "drain_workspace_allocations",
    []() {
      std::vector<WorkspaceAllocationCollector::Record> drained;
      std::size_t dropped = 0;
      {
        std::lock_guard<std::mutex> guard(g_workspace_collector.mutex);
        drained.swap(g_workspace_collector.records);
        dropped = g_workspace_collector.dropped.exchange(0);
      }
      py::list result;
      for (const auto& record : drained) {
        py::dict item;
        item["recipe"] = record.recipe;
        item["site"] = record.site;
        item["scope_hash"] = record.scope_hash;
        item["bytes"] = record.bytes;
        item["alignment"] = record.alignment;
        item["replicas"] = record.replicas;
        item["domain"] = record.domain;
        item["placement"] = record.placement;
        item["tier"] = record.tier;
        item["estimated_traffic_bytes"] = record.estimated_traffic_bytes;
        result.append(std::move(item));
      }
      if (dropped != 0) {
        py::dict item;
        item["dropped"] = dropped;
        result.append(std::move(item));
      }
      return result;
    },
    "Return and clear the collected workspace allocation records.");
  module.def("memory_event_callback_address",
             [] { return reinterpret_cast<std::uintptr_t>(&WorkspaceAllocationCollector::on_memory); });
  module.def("drain_memory_events", [] {
    std::vector<WorkspaceAllocationCollector::MemoryEvent> events;
    {
      std::lock_guard guard(g_workspace_collector.mutex);
      events.swap(g_workspace_collector.memory_events);
    }
    py::list result;
    for (const auto& event : events) {
      py::dict item;
      item["timestamp_ns"] = event.timestamp_ns;
      item["worker"] = event.worker;
      item["allocation_id"] = event.allocation_id;
      item["bytes"] = event.bytes;
      item["released"] = event.action == 1;
      item["action"] = event.action;
      item["site"] = event.site;
      result.append(item);
    }
    return result;
  });
}
