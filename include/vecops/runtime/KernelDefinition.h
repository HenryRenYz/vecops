/**
 * @file KernelDefinition.h
 * @brief Declarative source-kernel signatures and metadata specialization.
 *
 * `KernelDef` describes ordered tensor/scalar parameters plus the named
 * compile-time values visible in generated kernel source as `vecops::spec`.
 * `bind_kernel_call` is metadata-only: it infers symbols from tensor metadata,
 * checks explicitly supplied values, validates constraints, and produces one
 * deterministic specialization identity.
 */
#ifndef VECOPS_RUNTIME_KERNEL_DEFINITION_H
#define VECOPS_RUNTIME_KERNEL_DEFINITION_H

#include <concepts>
#include <cstdint>
#include <functional>
#include <map>
#include <optional>
#include <span>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "vecops/runtime/Argument.h"
#include "vecops/runtime/Status.h"

namespace vecops::runtime {

/** @brief Concrete compile-time value supplied to or inferred for a kernel. */
using SpecializationValue = std::variant<std::int64_t, DType>;
/** @brief Deterministically ordered environment of named specialization values. */
using SpecializationValues = std::map<std::string, SpecializationValue, std::less<>>;

/** @brief Kind of named value emitted into the generated `vecops::spec` namespace. */
enum class SpecializationType {
  /** A `meta::Const` integer visible as `inline constexpr auto`. */
  ConstInt,
  /** An element type visible as a generated C++ type alias. */
  DType,
};

/**
 * @brief Fixed integer or reference to a named `ConstInt` specialization value.
 *
 * Dynamic alignment and bounds accept this type so their declaration can be
 * fixed or use a symbol inferred elsewhere in the same call.  A string is
 * always a symbol, not a numeric expression; KernelDef validation verifies it
 * names a declared `ConstInt` value.
 */
class ConstExpr {
public:
  /** @brief Store a fixed integer expression. */
  template <std::integral Integer>
  ConstExpr(Integer value)
    : value_(static_cast<std::int64_t>(value)) {
  }
  /** @brief Store a named `ConstInt` reference. */
  ConstExpr(std::string symbol)
    : symbol_(std::move(symbol)) {
  }
  ConstExpr(const char* symbol)
    : symbol_(symbol == nullptr ? "" : symbol) {
  }

  /** @brief Test whether this expression is a name rather than a fixed value. */
  [[nodiscard]] bool is_symbol() const {
    return symbol_.has_value();
  }
  /** @brief Return the fixed value; meaningful only when `is_symbol()` is false. */
  [[nodiscard]] std::int64_t fixed_value() const {
    return value_;
  }
  /** @brief Return the referenced symbol, or an empty view for a fixed value. */
  [[nodiscard]] std::string_view symbol() const {
    return symbol_ ? std::string_view(*symbol_) : std::string_view{};
  }

private:
  std::int64_t value_ = 0;
  std::optional<std::string> symbol_;
};

/**
 * @brief Declarative rule for one tensor shape or stride coordinate.
 *
 * Fixed dimensions validate an exact integer. Symbols infer/check a named
 * `ConstInt`; anonymous `Const` values participate in the specialization key
 * without becoming visible in `vecops::spec`; Dynamic values remain runtime
 * metadata subject to their fixed or symbolic constraint expressions. A
 * Dynamic may itself carry a name used to enforce equality across tensor
 * metadata without specializing on the observed runtime value.
 */
class DimensionDef {
public:
  /** @brief How this coordinate is specialized or validated. */
  enum class Kind {
    /** Require a literal integer equal to `fixed_value()`. */
    Fixed,
    /** Infer/check a named ConstInt specialization. */
    Symbol,
    /** Infer an unnamed ConstInt specialization. */
    Const,
    /** Keep a runtime coordinate constrained by alignment and bounds. */
    Dynamic,
  };

  template <std::integral Integer>
  DimensionDef(Integer value)
    : kind_(Kind::Fixed)
    , fixed_value_(static_cast<std::int64_t>(value)) {
  }
  /** @brief Infer/check the named `ConstInt` @p symbol. */
  DimensionDef(std::string symbol);
  DimensionDef(const char* symbol);

  /** @brief Infer an anonymous compile-time integer. */
  static DimensionDef constant();
  /**
   * @brief Keep a runtime coordinate with alignment and inclusive bounds.
   * @param alignment Fixed or named positive power-of-two alignment.
   * @param lower_bound Fixed or named inclusive lower bound.
   * @param upper_bound Fixed or named inclusive upper bound.
   */
  static DimensionDef dynamic(ConstExpr alignment = 1, ConstExpr lower_bound = INT64_MIN,
                              ConstExpr upper_bound = INT64_MAX);
  /**
   * @brief Keep a named runtime coordinate with Dynamic constraints.
   * @param symbol Equality name shared by every occurrence in one call.
   * @note The observed value is not a specialization value or cache-key input.
   */
  static DimensionDef named_dynamic(std::string symbol, ConstExpr alignment = 1, ConstExpr lower_bound = INT64_MIN,
                                    ConstExpr upper_bound = INT64_MAX);
  /** @brief Shorthand for an unconstrained Dynamic coordinate. */
  static DimensionDef any();

  /** @brief Return this declaration's interpretation. */
  [[nodiscard]] Kind kind() const {
    return kind_;
  }
  /** @brief Return the fixed value; meaningful only for `Kind::Fixed`. */
  [[nodiscard]] std::int64_t fixed_value() const {
    return fixed_value_;
  }
  /** @brief Return the Const symbol or optional named-Dynamic equality symbol. */
  [[nodiscard]] std::string_view symbol() const {
    return symbol_;
  }
  [[nodiscard]] const ConstExpr& alignment() const {
    return alignment_;
  }
  [[nodiscard]] const ConstExpr& lower_bound() const {
    return lower_bound_;
  }
  [[nodiscard]] const ConstExpr& upper_bound() const {
    return upper_bound_;
  }

private:
  explicit DimensionDef(Kind kind)
    : kind_(kind) {
  }
  Kind kind_ = Kind::Dynamic;
  std::int64_t fixed_value_ = 0;
  std::string symbol_;
  ConstExpr alignment_{1};
  ConstExpr lower_bound_{INT64_MIN};
  ConstExpr upper_bound_{INT64_MAX};
};

/**
 * @brief Rule selecting a fixed, named, or anonymous-specialized tensor dtype.
 *
 * Default construction creates an anonymous dtype: it is inferred from the
 * actual tensor and contributes to the specialization key but has no name in
 * generated source.  A string creates a named `DType` specialization visible
 * through `vecops::spec`.
 */
class TensorDTypeDef {
public:
  /** @brief Create an anonymous dtype specialization. */
  TensorDTypeDef() = default;
  /** @brief Require one fixed ABI dtype. */
  TensorDTypeDef(DType dtype)
    : dtype_(dtype) {
  }
  /** @brief Infer/check the named `DType` specialization @p symbol. */
  TensorDTypeDef(std::string symbol)
    : symbol_(std::move(symbol)) {
  }
  TensorDTypeDef(const char* symbol)
    : symbol_(symbol == nullptr ? "" : symbol) {
  }

  /** @brief Return the fixed dtype when this is a fixed-dtype declaration. */
  [[nodiscard]] const std::optional<DType>& dtype() const {
    return dtype_;
  }
  /** @brief Return the named dtype symbol, or an empty view otherwise. */
  [[nodiscard]] std::string_view symbol() const {
    return symbol_;
  }
  /** @brief Test whether dtype is inferred anonymously from an actual tensor. */
  [[nodiscard]] bool anonymous() const {
    return !dtype_ && symbol_.empty();
  }

private:
  std::optional<DType> dtype_;
  std::string symbol_;
};

/**
 * @brief Tensor storage access required by a declarative kernel parameter.
 *
 * Outputs are out-style arguments allocated by the caller.  Optional tensors
 * are restricted to `Input` because an absent mutable argument is invalid.
 */
enum class TensorAccess {
  /** Read-only kernel input. */
  Input,
  /** Write-only caller-allocated output. */
  Output,
  /** Readable and writable caller-allocated argument. */
  InOut,
};

/**
 * @brief One tensor parameter of a declarative kernel signature.
 *
 * Shape and stride rules have identical rank and are evaluated positionally.
 * `device_index == -1` accepts any device ordinal.  This structure owns only
 * its declarative metadata; tensor storage comes from a later `KernelCall`.
 */
struct TensorDef {
  std::string name;
  std::vector<DimensionDef> shape;
  std::vector<DimensionDef> strides;
  TensorDTypeDef dtype;
  bool optional = false;
  TensorAccess access = TensorAccess::Input;
  std::uint32_t device_type = VECOPS_DEVICE_CPU;
  std::int32_t device_index = -1;

  /** @brief Test whether a caller must grant/read access. */
  [[nodiscard]] bool readable() const {
    return access != TensorAccess::Output;
  }
  /** @brief Test whether a caller must grant/write access. */
  [[nodiscard]] bool writable() const {
    return access != TensorAccess::Input;
  }
};

/**
 * @brief One runtime scalar parameter and its optional default.
 *
 * Defaults are inserted when a trailing scalar argument is omitted during
 * `bind_kernel_call`.  The default's dtype must exactly equal `dtype`.
 */
struct ValueDef {
  std::string name;
  DType dtype = DType::Invalid;
  std::optional<Scalar> default_value;

  /** @brief Create a scalar declaration from an ABI-representable C++ type. */
  template <typename T>
  static ValueDef typed(std::string name, std::optional<T> default_value = std::nullopt) {
    static_assert(dtype_of<std::remove_cv_t<T>> != DType::Invalid, "unsupported kernel scalar type");
    ValueDef result{.name = std::move(name), .dtype = dtype_of<std::remove_cv_t<T>>};
    if (default_value)
      result.default_value = std::get<Scalar>(erase_value(*default_value));
    return result;
  }
};

/** @brief One ordered declarative parameter, tensor or scalar. */
using KernelParameterDef = std::variant<TensorDef, ValueDef>;

/**
 * @brief Complete source-kernel interface and named specialization vocabulary.
 *
 * Kernel source must define `void __kernel__(...)` with parameters matching
 * this declaration, optionally preceded by `WorkspaceContext&`. The generated
 * adapter selects the signature at compile time. The context-aware form can
 * allocate dynamically, trace a logical plan, or replay a bound plan without
 * duplicating workspace-size arithmetic.
 * Every name used from `vecops::spec`, including shape, stride, Dynamic-bound,
 * and dtype references, is declared in `values`.
 * `validate()` reports declaration errors without touching an invocation.
 */
class KernelDef {
public:
  /** @brief Take ownership of a name, ordered parameter definitions, and value vocabulary. */
  KernelDef(std::string name, std::vector<KernelParameterDef> parameters,
            std::map<std::string, SpecializationType, std::less<>> values = {});

  /** @brief Return the stable logical kernel name. */
  [[nodiscard]] const std::string& name() const {
    return name_;
  }
  /** @brief Return ordered immutable tensor/scalar declarations. */
  [[nodiscard]] std::span<const KernelParameterDef> parameters() const {
    return parameters_;
  }
  /** @brief Return the ordered named `vecops::spec` vocabulary. */
  [[nodiscard]] const std::map<std::string, SpecializationType, std::less<>>& values() const {
    return values_;
  }
  /** @brief Validate identifiers, dimensions, dtype declarations, and defaults. */
  [[nodiscard]] Status validate() const;
  /** @brief Stable serialization used as an artifact-cache identity component. */
  [[nodiscard]] std::string canonical() const;

private:
  std::string name_;
  std::vector<KernelParameterDef> parameters_;
  std::map<std::string, SpecializationType, std::less<>> values_;
};

using KernelDefinition = KernelDef;

/**
 * @brief One invocation's positional metadata and explicitly supplied values.
 *
 * A call may omit trailing optional tensors and defaulted scalars.  Its map
 * needs only values not inferred from the provided tensors; supplied inferred
 * values are instead checked for equality during binding.
 */
class KernelCall {
public:
  /** @brief Construct an empty call, useful only for an empty KernelDef. */
  KernelCall() = default;
  /** @brief Take ownership of positional metadata and explicitly supplied values. */
  explicit KernelCall(ArgumentMetadata arguments, SpecializationValues values = {})
    : arguments_(std::move(arguments))
    , values_(std::move(values)) {
  }

  /** @brief Type-erase ordered C++ values and construct a KernelCall. */
  template <typename... Values>
  static KernelCall from_values(SpecializationValues specialization_values, Values&&... values) {
    return KernelCall(make_arguments(std::forward<Values>(values)...), std::move(specialization_values));
  }

  /** @brief Return the ordered non-owning tensor/scalar metadata. */
  [[nodiscard]] const ArgumentMetadata& arguments() const {
    return arguments_;
  }
  /** @brief Return the values explicitly supplied by the caller. */
  [[nodiscard]] const SpecializationValues& values() const {
    return values_;
  }

private:
  ArgumentMetadata arguments_;
  SpecializationValues values_;
};

/**
 * @brief Fully normalized and inferred call consumed by recipes and compilers.
 *
 * `arguments` includes inserted optional/default values. `values` is complete
 * for every name declared by KernelDef, and `anonymous_values` is ordered by
 * declaration traversal so that unnamed specializations remain cache-safe.
 */
struct BoundKernel {
  /** @brief Normalized positional arguments including default/optional insertions. */
  ArgumentMetadata arguments;
  /** @brief Complete named specialization environment after inference. */
  SpecializationValues values;
  /** Anonymous Const/dtype/optional-presence values in declaration order. */
  std::vector<SpecializationValue> anonymous_values;
  /** @brief Deterministic identity used by descriptors and artifact caches. */
  std::string specialization_key;
};

/**
 * @brief Infer and validate one call against a declarative source-kernel signature.
 *
 * Binding first merges explicit values with tensor-derived named values, then
 * validates all fixed/dynamic constraints after the symbol environment is
 * complete. It reads no tensor data or storage address, so metadata-only calls
 * can prepare an artifact before an allocation exists.
 */
[[nodiscard]] Result<BoundKernel> bind_kernel_call(const KernelDef& definition, const KernelCall& call);

using KernelShapes = KernelCall;

} // namespace vecops::runtime

#endif // VECOPS_RUNTIME_KERNEL_DEFINITION_H
