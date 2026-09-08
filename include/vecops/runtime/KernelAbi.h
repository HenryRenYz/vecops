/**
 * @file KernelAbi.h
 * @brief C ABI exported by independently compiled kernel artifact DSOs.
 *
 * An artifact exports exactly `vecops_kernel_query_v1`, which returns an
 * immutable descriptor valid while its DSO remains loaded.  The descriptor
 * makes no C++ ABI promises: all function pointers accept `CallAbi.h` records
 * and report failure with a `VecopsStatusCode`.  The loading runtime checks
 * record sizes and the major version before invoking any callback.
 */
#ifndef VECOPS_RUNTIME_KERNEL_ABI_H
#define VECOPS_RUNTIME_KERNEL_ABI_H

#include "vecops/runtime/CallAbi.h"

#ifdef __cplusplus
extern "C" {
#endif

/** Major version accepted by this runtime; a mismatch is never compatible. */
#define VECOPS_KERNEL_ABI_VERSION_MAJOR 1u
/** Minor version implemented by this runtime; newer compatible minors may be accepted. */
#define VECOPS_KERNEL_ABI_VERSION_MINOR 0u
/** Exported v1 descriptor-query symbol name. */
#define VECOPS_KERNEL_QUERY_SYMBOL_V1 "vecops_kernel_query_v1"

typedef enum VecopsDimensionKind {
  /** No runtime constraint beyond the tensor's own well-formed metadata. */
  VECOPS_DIM_ANY = 0,
  /** Require equality with `VecopsDimensionConstraint::value`. */
  VECOPS_DIM_CONST = 1,
  /** Require inclusive bounds and divisibility by `alignment`. */
  VECOPS_DIM_DYNAMIC = 2
} VecopsDimensionKind;

/**
 * @brief Constraint for one extent or stride coordinate.
 *
 * `value` is used by `VECOPS_DIM_CONST`; `lower_bound`, `upper_bound`, and
 * positive power-of-two `alignment` are used by `VECOPS_DIM_DYNAMIC`. Bounds
 * are inclusive.  The descriptor owns the record for its whole DSO lifetime.
 */
typedef struct VecopsDimensionConstraint {
  /** Size of this record known to the descriptor consumer. */
  uint32_t struct_size;
  /** `VecopsDimensionKind` selecting the active rule fields. */
  uint32_t kind;
  /** Required coordinate for `VECOPS_DIM_CONST`. */
  int64_t value;
  /** Inclusive lower bound for `VECOPS_DIM_DYNAMIC`. */
  int64_t lower_bound;
  /** Inclusive upper bound for `VECOPS_DIM_DYNAMIC`. */
  int64_t upper_bound;
  /** Positive power-of-two divisor for `VECOPS_DIM_DYNAMIC`. */
  int64_t alignment;
} VecopsDimensionConstraint;

/**
 * @brief Eligibility and access requirements for one tensor parameter.
 *
 * `sizes` and `strides` each point to `rank` constraints.  `dtype_mask` uses
 * bit `(dtype - 1)` for each accepted `VecopsDType`; zero therefore accepts no
 * dtype.  `optional` permits `VECOPS_VALUE_NONE`, and `required_flags` is the
 * subset of tensor flags that the caller must grant.
 */
typedef struct VecopsTensorParameter {
  /** Size of this record known to the descriptor consumer. */
  uint32_t struct_size;
  /** Bitset of accepted `VecopsDType` values. */
  uint64_t dtype_mask;
  /** Required device type, or zero for any device type. */
  uint32_t device_type;
  /** Required device ordinal, or -1 for any ordinal. */
  int32_t device_index;
  /** Required tensor rank. */
  int32_t rank;
  /** Non-zero when `VECOPS_VALUE_NONE` is accepted. */
  uint32_t optional;
  /** Byte stride between adjacent dimension constraints. */
  uint32_t dimension_constraint_stride;
  /** Descriptor-owned shape constraint array. */
  const VecopsDimensionConstraint* sizes;
  /** Descriptor-owned stride constraint array. */
  const VecopsDimensionConstraint* strides;
  /** Tensor access bits the caller must grant. */
  uint64_t required_flags;
} VecopsTensorParameter;

/** @brief Dtype eligibility for one scalar parameter, using the tensor mask encoding. */
typedef struct VecopsScalarParameter {
  /** Size of this record known to the descriptor consumer. */
  uint32_t struct_size;
  /** Bitset of accepted scalar dtypes. */
  uint64_t dtype_mask;
} VecopsScalarParameter;

/**
 * @brief Named positional descriptor entry.
 *
 * `kind` is a `VecopsValueKind` and selects `constraint.tensor` or
 * `constraint.scalar`; descriptor builders must not use `VECOPS_VALUE_NONE`
 * here.
 */
typedef struct VecopsParameterDescriptor {
  /** Size of this record known to the descriptor consumer. */
  uint32_t struct_size;
  /** `VECOPS_VALUE_TENSOR` or `VECOPS_VALUE_SCALAR`. */
  uint32_t kind;
  /** Descriptor-owned nul-terminated diagnostic parameter name. */
  const char* name;
  union {
    /** Tensor eligibility rule when `kind == VECOPS_VALUE_TENSOR`. */
    VecopsTensorParameter tensor;
    /** Scalar eligibility rule when `kind == VECOPS_VALUE_SCALAR`. */
    VecopsScalarParameter scalar;
  } constraint;
} VecopsParameterDescriptor;

/**
 * @brief Callback that reports temporary external workspace in bytes.
 *
 * It receives validated metadata and must write `*workspace_size` on success.
 * It must not write tensor storage.  Compiler-generated source kernels always
 * return zero because their `Workspace` allocation is internal.
 */
typedef int32_t (*VecopsWorkspaceFn)(const VecopsCall*, uint64_t*, VecopsError*);
/** @brief Callback that synchronously executes a validated call frame. */
typedef int32_t (*VecopsRunFn)(const VecopsCall*, VecopsError*);

/**
 * @brief Immutable v1 artifact descriptor returned from the query symbol.
 *
 * `parameter_stride` permits a future ABI to extend each parameter record;
 * v1 uses `sizeof(VecopsParameterDescriptor)`.  `workspace` may be null only
 * when workspace is always zero; `run` is required.  The strings and arrays
 * are owned by the DSO and remain valid until it is unloaded.
 */
typedef struct VecopsKernelDescriptorV1 {
  /** Size of this descriptor record. */
  uint32_t struct_size;
  /** Artifact ABI major version. */
  uint16_t abi_version_major;
  /** Artifact ABI minor version. */
  uint16_t abi_version_minor;
  /** Descriptor-owned logical operator name. */
  const char* operator_name;
  /** Descriptor-owned exact specialization identity. */
  const char* specialization_key;
  /** Number of positional entries in `parameters`. */
  uint32_t num_parameters;
  /** Byte stride between adjacent parameter records. */
  uint32_t parameter_stride;
  /** Descriptor-owned ordered parameter array. */
  const VecopsParameterDescriptor* parameters;
  /** Optional external-workspace query callback. */
  VecopsWorkspaceFn workspace;
  /** Required synchronous execution callback. */
  VecopsRunFn run;
  /** Reserved artifact flags; v1 consumers require unknown bits to be zero. */
  uint64_t flags;
} VecopsKernelDescriptorV1;

/** @brief Function type of `vecops_kernel_query_v1`. */
typedef const VecopsKernelDescriptorV1* (*VecopsKernelQueryFnV1)(void);

/**
 * @brief Return this DSO's immutable v1 descriptor.
 *
 * Artifact authors export this exact C symbol.  It must be safe to call after
 * the loader maps the DSO and must return either a valid static descriptor or
 * null; it must not return a temporary object.
 */
VECOPS_RUNTIME_EXPORT const VecopsKernelDescriptorV1* vecops_kernel_query_v1(void);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // VECOPS_RUNTIME_KERNEL_ABI_H
