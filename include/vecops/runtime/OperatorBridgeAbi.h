/**
 * @file OperatorBridgeAbi.h
 * @brief C ABI used by generated framework adapters to invoke a process-local Operator.
 *
 * Unlike `KernelAbi.h`, this protocol is not exported by kernel artifacts.
 * It joins a generated Torch/NumPy extension to the currently loaded vecops
 * Python module using an opaque process-local operator handle.  A handle is
 * invalid once its owning Python operator is destroyed, so bridge calls must
 * be synchronous and must not cache it beyond that lifetime.
 */
#ifndef VECOPS_RUNTIME_OPERATOR_BRIDGE_ABI_H
#define VECOPS_RUNTIME_OPERATOR_BRIDGE_ABI_H

#include "vecops/runtime/CallAbi.h"

#ifdef __cplusplus
extern "C" {
#endif

/** Major version of the process-local bridge protocol. */
#define VECOPS_OPERATOR_BRIDGE_ABI_VERSION_MAJOR 1u
/** Compatible minor version of the process-local bridge protocol. */
#define VECOPS_OPERATOR_BRIDGE_ABI_VERSION_MINOR 0u

typedef enum VecopsSpecializationValueKind {
  /** A named `vecops::spec` compile-time integer. */
  VECOPS_SPECIALIZATION_CONST_INT = 1,
  /** A named `vecops::spec` dtype alias. */
  VECOPS_SPECIALIZATION_DTYPE = 2
} VecopsSpecializationValueKind;

/**
 * @brief One named specialization passed by a generated framework bridge.
 *
 * `name` is a nul-terminated symbol name declared by the KernelDef and the
 * active union member follows `kind`.  The bridge retains the string and the
 * array through `vecops_operator_bridge_invoke_v1`.
 */
typedef struct VecopsSpecializationArgument {
  /** Size of this record known to the bridge. */
  uint32_t struct_size;
  /** `VecopsSpecializationValueKind` selecting `value`. */
  uint32_t kind;
  /** Caller-owned nul-terminated KernelDef specialization name. */
  const char* name;
  union {
    /** Active integer payload for `VECOPS_SPECIALIZATION_CONST_INT`. */
    int64_t integer;
    /** Active ABI dtype for `VECOPS_SPECIALIZATION_DTYPE`. */
    uint32_t dtype;
  } value;
} VecopsSpecializationArgument;

/**
 * @brief Invoke a Python-owned runtime Operator from generated framework code.
 * @param handle Process-local opaque operator identifier supplied by vecops.
 * @param call Positional metadata and framework-owned tensor addresses.
 * @param num_specialization_values Number of entries in @p specialization_values.
 * @param specialization_values Named compiler specialization values, or null when empty.
 * @param error Optional caller-owned diagnostic buffer.
 * @return A `VecopsStatusCode` value.
 *
 * The function validates the handle and metadata, then synchronously resolves
 * and invokes the selected executable.  It neither takes tensor ownership nor
 * transfers the handle to the extension.
 */
VECOPS_RUNTIME_EXPORT int32_t
vecops_operator_bridge_invoke_v1(uint64_t handle, const VecopsCall* call, uint32_t num_specialization_values,
                                 const VecopsSpecializationArgument* specialization_values, VecopsError* error);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // VECOPS_RUNTIME_OPERATOR_BRIDGE_ABI_H
