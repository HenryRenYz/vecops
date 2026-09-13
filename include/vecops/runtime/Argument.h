/**
 * @file Argument.h
 * @brief Type-erased, owning C++ argument metadata for runtime invocations.
 *
 * The C++ objects in this header own only metadata such as shape and stride
 * vectors.  They never own tensor storage.  `CallFrame` converts that
 * metadata to the C ABI for one workspace or run call; it must therefore
 * remain alive while the ABI call is in progress.
 *
 * `ArgumentMetadata` and `TensorView` are ordinary value types.  Concurrent
 * reads are safe when callers do not mutate the same object concurrently.
 * The pointed-to tensor storage follows the access and synchronization rules
 * of the framework that owns it.
 */
#ifndef VECOPS_RUNTIME_ARGUMENT_H
#define VECOPS_RUNTIME_ARGUMENT_H

#include <concepts>
#include <cstdint>
#include <span>
#include <string_view>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "vecops/CoreTypes.h"
#include "vecops/runtime/CallAbi.h"

namespace vecops::tensor {
template <typename T, typename TShape, typename TStrides, typename TBinding>
class Tensor;
}

namespace vecops::runtime {

/**
 * @brief Runtime element type corresponding to `VecopsDType`.
 *
 * Values intentionally have the same representation as the stable C ABI
 * enumeration.  `Invalid` is a sentinel and is not a valid tensor or scalar
 * element type.
 */
enum class DType : std::uint32_t {
  Invalid = VECOPS_DTYPE_INVALID,
  Bool = VECOPS_DTYPE_BOOL,
  Int8 = VECOPS_DTYPE_I8,
  UInt8 = VECOPS_DTYPE_U8,
  Int16 = VECOPS_DTYPE_I16,
  UInt16 = VECOPS_DTYPE_U16,
  Int32 = VECOPS_DTYPE_I32,
  UInt32 = VECOPS_DTYPE_U32,
  Int64 = VECOPS_DTYPE_I64,
  UInt64 = VECOPS_DTYPE_U64,
  Float16 = VECOPS_DTYPE_F16,
  BFloat16 = VECOPS_DTYPE_BF16,
  Float32 = VECOPS_DTYPE_F32,
  Float64 = VECOPS_DTYPE_F64,
};

/**
 * @brief Return the ABI dtype-mask bit for @p dtype.
 *
 * @param dtype Concrete runtime element type.
 * @return Its one-hot bit, or zero for `DType::Invalid` and unrecognized
 * values.
 *
 * This function is pure and thread-safe.
 */
constexpr std::uint64_t dtype_bit(DType dtype) {
  const auto value = static_cast<std::uint32_t>(dtype);
  return value < static_cast<std::uint32_t>(DType::Bool) || value > static_cast<std::uint32_t>(DType::Float64)
           ? 0
           : (std::uint64_t{1} << (value - 1));
}

/**
 * @brief Return a mask containing every concrete dtype supported by the ABI.
 *
 * @return The bitwise union of all values returned by `dtype_bit` for valid
 * dtypes.
 *
 * This function is pure and thread-safe.
 */
constexpr std::uint64_t all_dtypes() {
  return (std::uint64_t{1} << 13) - 1;
}

/**
 * @brief Map an exact C++ scalar type to its runtime dtype.
 *
 * The primary template evaluates to `DType::Invalid`; use it in a
 * `static_assert` when an API accepts only ABI-representable element types.
 * CV-qualified types should be normalized with `std::remove_cv_t` first.
 */
template <typename T>
inline constexpr DType dtype_of = DType::Invalid;
template <>
inline constexpr DType dtype_of<bool> = DType::Bool;
template <>
inline constexpr DType dtype_of<int8_t> = DType::Int8;
template <>
inline constexpr DType dtype_of<uint8_t> = DType::UInt8;
template <>
inline constexpr DType dtype_of<int16_t> = DType::Int16;
template <>
inline constexpr DType dtype_of<uint16_t> = DType::UInt16;
template <>
inline constexpr DType dtype_of<int32_t> = DType::Int32;
template <>
inline constexpr DType dtype_of<uint32_t> = DType::UInt32;
template <>
inline constexpr DType dtype_of<int64_t> = DType::Int64;
template <>
inline constexpr DType dtype_of<uint64_t> = DType::UInt64;
template <>
inline constexpr DType dtype_of<float16_t> = DType::Float16;
template <>
inline constexpr DType dtype_of<bfloat16_t> = DType::BFloat16;
template <>
inline constexpr DType dtype_of<float32_t> = DType::Float32;
template <>
inline constexpr DType dtype_of<float64_t> = DType::Float64;

/**
 * @brief Compile-time mapping from a concrete `DType` to its C++ scalar type.
 *
 * The primary template deliberately has no member type.  Consequently,
 * `dtype_to_type_t<DType::Invalid>` and unknown enum values fail at compile
 * time instead of producing a placeholder type.
 *
 * @tparam dtype A concrete ABI dtype.
 */
template <DType dtype>
struct dtype_to_type {
  static_assert(dtype != dtype, "dtype_to_type requires a concrete, supported DType");
};

/** @brief Specialization mapping `DType::Bool` to `bool`. */
template <>
struct dtype_to_type<DType::Bool> {
  using type = bool;
};
/** @brief Specialization mapping `DType::Int8` to `int8_t`. */
template <>
struct dtype_to_type<DType::Int8> {
  using type = int8_t;
};
/** @brief Specialization mapping `DType::UInt8` to `uint8_t`. */
template <>
struct dtype_to_type<DType::UInt8> {
  using type = uint8_t;
};
/** @brief Specialization mapping `DType::Int16` to `int16_t`. */
template <>
struct dtype_to_type<DType::Int16> {
  using type = int16_t;
};
/** @brief Specialization mapping `DType::UInt16` to `uint16_t`. */
template <>
struct dtype_to_type<DType::UInt16> {
  using type = uint16_t;
};
/** @brief Specialization mapping `DType::Int32` to `int32_t`. */
template <>
struct dtype_to_type<DType::Int32> {
  using type = int32_t;
};
/** @brief Specialization mapping `DType::UInt32` to `uint32_t`. */
template <>
struct dtype_to_type<DType::UInt32> {
  using type = uint32_t;
};
/** @brief Specialization mapping `DType::Int64` to `int64_t`. */
template <>
struct dtype_to_type<DType::Int64> {
  using type = int64_t;
};
/** @brief Specialization mapping `DType::UInt64` to `uint64_t`. */
template <>
struct dtype_to_type<DType::UInt64> {
  using type = uint64_t;
};
/** @brief Specialization mapping `DType::Float16` to `float16_t`. */
template <>
struct dtype_to_type<DType::Float16> {
  using type = float16_t;
};
/** @brief Specialization mapping `DType::BFloat16` to `bfloat16_t`. */
template <>
struct dtype_to_type<DType::BFloat16> {
  using type = bfloat16_t;
};
/** @brief Specialization mapping `DType::Float32` to `float32_t`. */
template <>
struct dtype_to_type<DType::Float32> {
  using type = float32_t;
};
/** @brief Specialization mapping `DType::Float64` to `float64_t`. */
template <>
struct dtype_to_type<DType::Float64> {
  using type = float64_t;
};

/**
 * @brief Alias for the C++ scalar type represented by @p dtype.
 *
 * @tparam dtype A concrete ABI dtype; `Invalid` and unknown values are
 * ill-formed.
 */
template <DType dtype>
using dtype_to_type_t = typename dtype_to_type<dtype>::type;

/**
 * @brief Non-owning tensor metadata accepted by the C++ runtime.
 *
 * `data` points to the first allocation byte and `byte_offset` locates the
 * logical element at index zero.  Sizes and strides are measured in elements.
 * The vectors own their metadata, but not `data`; the storage must outlive
 * each validation, workspace, and invocation that uses this view.  `flags`
 * declare the caller's permitted access and do not synchronize concurrent
 * readers or writers.
 */
struct TensorView {
  /** @brief Base address of externally-owned tensor storage, or null for an empty tensor. */
  void* data = nullptr;
  /** @brief Byte displacement from `data` to logical element zero. */
  std::uint64_t byte_offset = 0;
  /** @brief `VecopsDeviceType` numeric value identifying the storage device. */
  std::uint32_t device_type = VECOPS_DEVICE_CPU;
  /** @brief Device ordinal; meaningful only with a non-generic device type. */
  std::int32_t device_index = 0;
  /** @brief Element representation of the storage. */
  DType dtype = DType::Invalid;
  /** @brief Extent of each axis, measured in elements and required to be non-negative. */
  std::vector<std::int64_t> sizes;
  /** @brief Per-axis element strides; must have the same length as `sizes`. */
  std::vector<std::int64_t> strides;
  /** @brief `VecopsTensorFlags` access declaration for this view. */
  std::uint64_t flags = VECOPS_TENSOR_READ;

  /**
   * @brief Report whether the view declares write access.
   * @return True when `VECOPS_TENSOR_WRITE` is set in `flags`.
   *
   * Pure and thread-safe provided the caller does not concurrently mutate
   * this object.
   */
  [[nodiscard]] bool writable() const {
    return (flags & VECOPS_TENSOR_WRITE) != 0;
  }
};

/**
 * @brief Scalar value encoded in the ABI's three-way storage union.
 *
 * The active union member is selected by `dtype`; factory functions write the
 * matching signed, unsigned, or floating representation.  Construction does
 * not range-check a requested narrow dtype, so callers must supply a value
 * representable by that dtype.
 */
struct Scalar {
  /** @brief Concrete scalar dtype; must not be `Invalid`. */
  DType dtype = DType::Invalid;
  /** @brief ABI storage whose active member follows `dtype`. */
  VecopsScalarStorage value{};

  /** @brief Construct a scalar encoded through `VecopsScalarStorage::i64`.
   * @param value Signed value to encode.
   * @param dtype Signed integer dtype represented by @p value.
   * @return A scalar with signed storage initialized.
   */
  static Scalar signed_integer(std::int64_t value, DType dtype = DType::Int64);
  /** @brief Construct a scalar encoded through `VecopsScalarStorage::u64`.
   * @param value Unsigned value to encode.
   * @param dtype Unsigned integer or boolean dtype represented by @p value.
   * @return A scalar with unsigned storage initialized.
   */
  static Scalar unsigned_integer(std::uint64_t value, DType dtype = DType::UInt64);
  /** @brief Construct a scalar encoded through `VecopsScalarStorage::f64`.
   * @param value Floating-point value to encode.
   * @param dtype Floating-point dtype represented by @p value.
   * @return A scalar with floating storage initialized.
   */
  static Scalar floating(double value, DType dtype = DType::Float64);
};

/** @brief One positional runtime argument: absent, tensor, or scalar. */
using Value = std::variant<std::monostate, TensorView, Scalar>;

/**
 * @brief Ordered, owning metadata for a logical operator invocation.
 *
 * Value ordering must agree with the associated `OperatorSchema` or kernel
 * descriptor.  Copies duplicate metadata but retain the same external tensor
 * addresses.  References and spans returned from this object are invalidated
 * by non-const access followed by a vector reallocation.
 */
class ArgumentMetadata {
public:
  /** @brief Construct an empty argument sequence. */
  ArgumentMetadata() = default;
  /**
   * @brief Take ownership of ordered argument values.
   * @param values Values in schema parameter order.
   */
  explicit ArgumentMetadata(std::vector<Value> values)
    : values_(std::move(values)) {
  }

  /** @brief Return an immutable view of all ordered values. */
  [[nodiscard]] std::span<const Value> values() const {
    return values_;
  }
  /** @brief Return a mutable view of all ordered values. */
  [[nodiscard]] std::span<Value> values() {
    return values_;
  }
  /** @brief Return the number of positional values. */
  [[nodiscard]] std::size_t size() const {
    return values_.size();
  }
  /**
   * @brief Return the value at @p index without bounds checking.
   * @param index Positional argument index; must be less than `size()`.
   * @return Immutable reference into this metadata object.
   */
  [[nodiscard]] const Value& operator[](std::size_t index) const {
    return values_[index];
  }

private:
  std::vector<Value> values_;
};

/**
 * @brief Materialize an `ArgumentMetadata` instance as a transient ABI call.
 *
 * The frame copies ABI records and owns their outer array, while tensor shape
 * and stride pointers refer to @p arguments.  Thus both the frame and the
 * source `ArgumentMetadata` must outlive every use of `abi()`.  It neither
 * owns nor accesses tensor data.  Construct a separate frame per concurrent
 * call unless the caller provides external synchronization.
 */
class CallFrame {
public:
  /**
   * @brief Build an ABI call frame from ordered C++ metadata.
   * @param arguments Source metadata that must outlive this frame.
   * @param workspace Caller-owned workspace, or null when no workspace is used.
   * @param workspace_size Size of @p workspace in bytes.
   * @param context Optional non-owning execution context valid for the ABI call.
   */
  explicit CallFrame(const ArgumentMetadata& arguments, void* workspace = nullptr, std::uint64_t workspace_size = 0,
                     const VecopsExecutionContext* context = nullptr);

  /**
   * @brief Return the ABI call record backed by this frame.
   * @return Reference valid until this frame is destroyed.
   */
  [[nodiscard]] const VecopsCall& abi() const {
    return call_;
  }

private:
  std::vector<VecopsValue> values_;
  VecopsCall call_{};
};

/** @brief Preserve an already type-erased runtime value. */
inline Value erase_value(Value value) {
  return value;
}
/** @brief Represent an omitted optional argument. */
inline Value erase_value(std::monostate value) {
  return value;
}
/** @brief Copy a tensor view into a type-erased runtime value. */
inline Value erase_value(TensorView value) {
  return value;
}
/** @brief Copy a scalar into a type-erased runtime value. */
inline Value erase_value(Scalar value) {
  return value;
}

/** @brief Type-erase a native boolean as an ABI boolean scalar. */
inline Value erase_value(bool value) {
  return Scalar::unsigned_integer(value ? 1 : 0, DType::Bool);
}

/** @brief Type-erase a supported native signed integer scalar. */
template <std::signed_integral T>
  requires(dtype_of<std::remove_cv_t<T>> != DType::Invalid)
inline Value erase_value(T value) {
  return Scalar::signed_integer(static_cast<std::int64_t>(value), dtype_of<std::remove_cv_t<T>>);
}

/** @brief Type-erase a supported native unsigned integer scalar. */
template <std::unsigned_integral T>
  requires(!std::same_as<std::remove_cv_t<T>, bool> && dtype_of<std::remove_cv_t<T>> != DType::Invalid)
inline Value erase_value(T value) {
  return Scalar::unsigned_integer(static_cast<std::uint64_t>(value), dtype_of<std::remove_cv_t<T>>);
}

/** @brief Type-erase a supported native floating-point scalar. */
template <std::floating_point T>
  requires(dtype_of<std::remove_cv_t<T>> != DType::Invalid)
inline Value erase_value(T value) {
  return Scalar::floating(static_cast<double>(value), dtype_of<std::remove_cv_t<T>>);
}

/**
 * @brief Convert a typed `tensor::Tensor` to non-owning runtime metadata.
 * @tparam T Tensor element type; its unqualified form must have a concrete `dtype_of` mapping.
 * @tparam Shape Compile-time shape descriptor.
 * @tparam Strides Compile-time stride descriptor.
 * @param tensor Source non-owning typed tensor view.
 * @return Runtime view with copied shape/stride vectors and the same data address.
 *
 * No element data is copied or accessed.  The tensor storage must outlive the
 * returned `TensorView` and every invocation using it.  Const source elements
 * become read-only; non-const elements declare read/write access.
 */
template <typename T, typename Shape, typename Strides, typename Binding>
TensorView erase_tensor(const tensor::Tensor<T, Shape, Strides, Binding>& tensor) {
  using Element = std::remove_cv_t<T>;
  static_assert(dtype_of<Element> != DType::Invalid, "Unsupported vecops runtime tensor element type");
  TensorView result;
  result.data = const_cast<Element*>(tensor.data());
  result.dtype = dtype_of<Element>;
  result.flags = VECOPS_TENSOR_READ;
  if constexpr (!std::is_const_v<T>)
    result.flags |= VECOPS_TENSOR_WRITE;
  result.sizes.reserve(tensor.Ndim);
  result.strides.reserve(tensor.Ndim);
  for (int index = 0; index < tensor.Ndim; ++index) {
    result.sizes.push_back(tensor.size(index));
    result.strides.push_back(tensor.stride(index));
  }
  return result;
}

/**
 * @brief Type-erase a typed `tensor::Tensor` as a runtime value.
 * @tparam T Tensor element type.
 * @tparam Shape Compile-time shape descriptor.
 * @tparam Strides Compile-time stride descriptor.
 * @param tensor Source tensor view.
 * @return Tensor alternative containing copied metadata and a non-owning data pointer.
 */
template <typename T, typename Shape, typename Strides, typename Binding>
Value erase_value(const tensor::Tensor<T, Shape, Strides, Binding>& tensor) {
  return erase_tensor(tensor);
}

/**
 * @brief Type-erase ordered arguments into owning metadata.
 * @tparam Values Supported tensor, scalar, runtime-value, or omitted-value types.
 * @param values Positional values in the target schema order.
 * @return Metadata owning its value alternatives and shape/stride vectors.
 *
 * Tensor allocations remain externally owned and must outlive a later call.
 * This helper performs no validation, allocation of tensor storage, or device
 * synchronization.
 */
template <typename... Values>
ArgumentMetadata make_arguments(Values&&... values) {
  std::vector<Value> erased;
  erased.reserve(sizeof...(Values));
  (erased.push_back(erase_value(std::forward<Values>(values))), ...);
  return ArgumentMetadata(std::move(erased));
}

} // namespace vecops::runtime

#endif // VECOPS_RUNTIME_ARGUMENT_H
