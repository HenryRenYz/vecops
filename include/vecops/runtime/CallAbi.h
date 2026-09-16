/**
 * @file CallAbi.h
 * @brief Shared C ABI values passed to kernels and process-local operator bridges.
 *
 * This is the lowest-level vecops ABI.  It is a C-only data contract: callers
 * initialize every `struct_size` to `sizeof(the record they provide)`, retain
 * every pointed-to object for the duration of the call, and use no C++ types
 * across this boundary.  All ranks, sizes, and strides are expressed in
 * elements, while `byte_offset` and workspace sizes are expressed in bytes.
 *
 * A kernel artifact consumes this format through `KernelAbi.h`; a generated
 * framework adapter uses it through `OperatorBridgeAbi.h`.  Neither protocol
 * owns tensor storage, workspace, stream, or `user_data`.  ABI functions are
 * synchronous unless their individual protocol documents otherwise.
 */
#ifndef VECOPS_RUNTIME_CALL_ABI_H
#define VECOPS_RUNTIME_CALL_ABI_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32)
#  define VECOPS_RUNTIME_EXPORT __declspec(dllexport)
#else
#  define VECOPS_RUNTIME_EXPORT __attribute__((visibility("default")))
#endif

typedef enum VecopsStatusCode {
  /** Operation completed successfully. */
  VECOPS_STATUS_OK = 0,
  /** Caller metadata or values violated a contract. */
  VECOPS_STATUS_INVALID_ARGUMENT = 1,
  /** A candidate recipe does not support this otherwise valid call. */
  VECOPS_STATUS_NOT_APPLICABLE = 2,
  /** No requested artifact, recipe, or specialization was found. */
  VECOPS_STATUS_NOT_FOUND = 3,
  /** ABI major version, record size, or descriptor contract differs. */
  VECOPS_STATUS_ABI_MISMATCH = 4,
  /** Dynamic loader or required-symbol lookup failed. */
  VECOPS_STATUS_LOAD_ERROR = 5,
  /** A validated kernel callback reported an execution failure. */
  VECOPS_STATUS_EXECUTION_ERROR = 6,
  /** The runtime encountered an unexpected internal failure. */
  VECOPS_STATUS_INTERNAL_ERROR = 7
} VecopsStatusCode;

typedef enum VecopsDeviceType {
  /** Host CPU storage. */
  VECOPS_DEVICE_CPU = 1,
  /** CUDA device storage; the core runtime only transports this identifier. */
  VECOPS_DEVICE_CUDA = 2,
  /** Framework-defined device type; its numeric meaning is embedding-specific. */
  VECOPS_DEVICE_PRIVATE = 255
} VecopsDeviceType;

typedef enum VecopsDType {
  /** Sentinel; never a valid tensor or scalar dtype. */
  VECOPS_DTYPE_INVALID = 0,
  /** One-byte boolean scalar. */
  VECOPS_DTYPE_BOOL = 1,
  /** Signed 8-bit integer. */
  VECOPS_DTYPE_I8 = 2,
  /** Unsigned 8-bit integer. */
  VECOPS_DTYPE_U8 = 3,
  /** Signed 16-bit integer. */
  VECOPS_DTYPE_I16 = 4,
  /** Unsigned 16-bit integer. */
  VECOPS_DTYPE_U16 = 5,
  /** Signed 32-bit integer. */
  VECOPS_DTYPE_I32 = 6,
  /** Unsigned 32-bit integer. */
  VECOPS_DTYPE_U32 = 7,
  /** Signed 64-bit integer. */
  VECOPS_DTYPE_I64 = 8,
  /** Unsigned 64-bit integer. */
  VECOPS_DTYPE_U64 = 9,
  /** IEEE binary16-compatible 16-bit float. */
  VECOPS_DTYPE_F16 = 10,
  /** bfloat16-compatible 16-bit float. */
  VECOPS_DTYPE_BF16 = 11,
  /** IEEE binary32-compatible 32-bit float. */
  VECOPS_DTYPE_F32 = 12,
  /** IEEE binary64-compatible 64-bit float. */
  VECOPS_DTYPE_F64 = 13
} VecopsDType;

typedef enum VecopsValueKind {
  /** Missing optional tensor value. */
  VECOPS_VALUE_NONE = 0,
  /** Active union member is `VecopsTensorView`. */
  VECOPS_VALUE_TENSOR = 1,
  /** Active union member is `VecopsScalar`. */
  VECOPS_VALUE_SCALAR = 2
} VecopsValueKind;

typedef enum VecopsTensorFlags {
  /** The callee may read the tensor allocation. */
  VECOPS_TENSOR_READ = UINT64_C(1) << 0,
  /** The callee may write the tensor allocation. */
  VECOPS_TENSOR_WRITE = UINT64_C(1) << 1
} VecopsTensorFlags;

/**
 * @brief Caller-provided storage for an ABI diagnostic.
 *
 * Callees set `code` and `message_required`.  When `message` has non-zero
 * `message_capacity`, they write a nul-terminated truncated diagnostic into
 * it.  A null `message` is valid and can be used to query the required size.
 */
typedef struct VecopsError {
  /** Size of this record known to the caller. */
  uint32_t struct_size;
  /** ABI status code written by a failing callee. */
  int32_t code;
  /** Optional caller-owned character buffer. */
  char* message;
  /** Capacity of `message`, including its nul terminator. */
  size_t message_capacity;
  /** Required diagnostic bytes including terminator, even when truncated. */
  size_t message_required;
} VecopsError;

/**
 * @brief Non-owning tensor metadata and storage address.
 *
 * `data + byte_offset` identifies logical index zero.  `sizes` and `strides`
 * each contain `rank` signed element counts; neither array is owned or copied
 * by the callee.  `flags` declares the access granted by the caller and does
 * not provide synchronization for aliased storage.
 */
typedef struct VecopsTensorView {
  /** Size of this record known to the caller. */
  uint32_t struct_size;
  /** Base allocation address, or null for a zero-sized tensor. */
  void* data;
  /** Byte displacement from `data` to logical element zero. */
  uint64_t byte_offset;
  /** `VecopsDeviceType` numeric identifier. */
  uint32_t device_type;
  /** Device ordinal, meaningful for non-generic device types. */
  int32_t device_index;
  /** `VecopsDType` numeric element representation. */
  uint32_t dtype;
  /** Number of entries in each of `sizes` and `strides`. */
  uint32_t rank;
  /** Non-owning array of non-negative element extents. */
  const int64_t* sizes;
  /** Non-owning array of element strides; negative strides are artifact-defined. */
  const int64_t* strides;
  /** Granted `VecopsTensorFlags` bitset. */
  uint64_t flags;
} VecopsTensorView;

/** @brief Storage union for a scalar, selected by `VecopsScalar::dtype`. */
typedef union VecopsScalarStorage {
  /** Active for signed integer dtypes. */
  int64_t i64;
  /** Active for unsigned integer and bool dtypes. */
  uint64_t u64;
  /** Active for floating-point dtypes. */
  double f64;
} VecopsScalarStorage;

/** @brief Non-owning scalar value encoded in the ABI's widest storage lane. */
typedef struct VecopsScalar {
  /** Size of this record known to the caller. */
  uint32_t struct_size;
  /** Concrete `VecopsDType` selecting `value`. */
  uint32_t dtype;
  /** Signed, unsigned, or floating payload selected by `dtype`. */
  VecopsScalarStorage value;
} VecopsScalar;

/**
 * @brief One positional call argument.
 *
 * `kind` selects the active union member.  `VECOPS_VALUE_NONE` is used only
 * for an omitted optional tensor and leaves the union unspecified.
 */
typedef struct VecopsValue {
  /** Size of this record known to the caller. */
  uint32_t struct_size;
  /** `VecopsValueKind` selecting the active union member. */
  uint32_t kind;
  union {
    /** Active tensor metadata for `VECOPS_VALUE_TENSOR`. */
    VecopsTensorView tensor;
    /** Active scalar payload for `VECOPS_VALUE_SCALAR`. */
    VecopsScalar scalar;
  } value;
} VecopsValue;

/**
 * @brief Optional opaque execution context supplied by the embedding runtime.
 *
 * `stream` and `user_data` have no core-runtime interpretation. Workspace
 * placement uses a separate extension pointer, so embedding state and memory
 * policy can be supplied together.
 */
typedef enum VecopsWorkspaceTier { VECOPS_WORKSPACE_TIER_FAST = 0, VECOPS_WORKSPACE_TIER_SLOW = 1 } VecopsWorkspaceTier;

#define VECOPS_WORKSPACE_ARENA_PROVIDER_FLAG_SHARED_ARENAS UINT64_C(1)

/** One provider-owned workspace arena returned across the process-local ABI. */
typedef struct VecopsWorkspaceArena {
  uint32_t struct_size;
  uint32_t reserved;
  void* data;
  uint64_t capacity;
  /** Opaque handle passed to `release`; it need not equal `data`. */
  void* owner;
} VecopsWorkspaceArena;

typedef uint64_t (*VecopsWorkspaceArenaCapacityFn)(void* context, uint32_t tier);
typedef int32_t (*VecopsWorkspaceArenaAllocateFn)(void* context, uint32_t tier, uint64_t bytes, uint64_t alignment,
                                                  VecopsWorkspaceArena* arena, VecopsError* error);
typedef void (*VecopsWorkspaceArenaReleaseFn)(void* context, void* owner);
typedef void (*VecopsWorkspaceArenaContextFn)(void* context);

/**
 * Versioned allocation callbacks passed to generated source-kernel DSOs.
 *
 * A consumer calls `retain(context)` before retaining this provider and pairs
 * it with `release_context(context)`. Every successful non-empty `allocate`
 * result is paired with `release(context, arena.owner)`. `identity` must be
 * stable and unique for the provider's placement policy lifetime.
 */
typedef struct VecopsWorkspaceArenaProvider {
  uint32_t struct_size;
  uint32_t reserved;
  uint64_t identity;
  void* context;
  VecopsWorkspaceArenaCapacityFn capacity;
  VecopsWorkspaceArenaAllocateFn allocate;
  VecopsWorkspaceArenaReleaseFn release;
  VecopsWorkspaceArenaContextFn retain;
  VecopsWorkspaceArenaContextFn release_context;
  /** Provider capabilities; consumers ignore unknown bits. */
  uint64_t flags;
} VecopsWorkspaceArenaProvider;

/** One logical task in a synchronous embedding-owned parallel region. */
typedef void (*VecopsParallelTaskFn)(void* body_context, uint32_t task_id, uint32_t task_count);
typedef uint32_t (*VecopsThreadPoolQueryFn)(void* context);
typedef int32_t (*VecopsThreadPoolParallelForFn)(void* context, uint32_t task_count, void* body_context,
                                                 VecopsParallelTaskFn body, VecopsError* error);
typedef void (*VecopsThreadPoolContextFn)(void* context);

/**
 * Versioned synchronous logical-task executor supplied by an embedding.
 *
 * `parallel_for` must invoke `body(body_context, task_id, task_count)` exactly
 * once for every dense task id in `[0, task_count)`, and must not return until
 * all callbacks have completed. It may execute several logical tasks on one
 * physical worker, which keeps compile-time vecops sharding independent of
 * the backend's scheduling policy.
 */
typedef struct VecopsThreadPoolV1 {
  uint32_t struct_size;
  uint16_t abi_major;
  uint16_t abi_minor;
  uint64_t identity;
  void* context;
  VecopsThreadPoolQueryFn max_parallelism;
  VecopsThreadPoolQueryFn in_parallel_region;
  VecopsThreadPoolParallelForFn parallel_for;
  VecopsThreadPoolContextFn retain;
  VecopsThreadPoolContextFn release;
  uint64_t flags;
} VecopsThreadPoolV1;

#define VECOPS_THREAD_POOL_ABI_MAJOR 1
#define VECOPS_THREAD_POOL_ABI_MINOR 0

/** Factory exported by the optional vecops OpenMP provider library. */
VECOPS_RUNTIME_EXPORT const VecopsThreadPoolV1* vecops_openmp_thread_pool_v1(void);

/**
 * Factory for the process-local persistent native thread pool.
 *
 * The first call fixes the physical worker count for the lifetime of the
 * process. Later calls return the same provider. Worker threads park between
 * submissions so an embedding's own parallel runtime can use the same CPU set
 * without competing with a permanently spinning vecops team.
 */
VECOPS_RUNTIME_EXPORT const VecopsThreadPoolV1* vecops_native_thread_pool_v1(uint32_t threads);

/**
 * Keep the process-local native workers active between submissions.
 *
 * Calls may be nested.  Every successful begin must be paired with an end.
 * The first begin also creates the pool when necessary, using @p threads as
 * the process-lifetime worker count.  This is intended for a bounded fused
 * region containing many short parallel phases; leaving it active around
 * work owned by another CPU runtime can cause oversubscription.
 */
VECOPS_RUNTIME_EXPORT void vecops_native_thread_pool_begin_active(uint32_t threads);

/** End one active native-pool region begun by the calling process. */
VECOPS_RUNTIME_EXPORT void vecops_native_thread_pool_end_active(void);

/**
 * @brief One workspace allocation performed inside a kernel invocation.
 *
 * String members are owned by the reporting context and remain valid only
 * for the duration of the observer callback; collectors must copy them.
 * `tier` reports where the bytes actually came from: 0 = fast arena,
 * 1 = slow arena, 2 = separately owned heap allocation.
 */
typedef struct VecopsWorkspaceAllocationV1 {
  /** Size of this record known to the reporter. */
  uint32_t struct_size;
  /** Kernel recipe name; static storage for the reporting DSO. */
  const char* recipe;
  /** Workspace-local site name; static or scope-lifetime storage. */
  const char* site;
  /** Hash of the enclosing scope chain, stable within a recipe. */
  uint64_t scope_hash;
  /** Requested logical bytes (before replica multiplication). */
  uint64_t bytes;
  /** Effective allocation alignment in bytes. */
  uint64_t alignment;
  /** Replica count for worker-local sites, otherwise one. */
  uint64_t replicas;
  /** `WorkspaceDomain` value: 0 global, 1 worker-local. */
  int32_t domain;
  /** `WorkspacePlacementPolicy` value. */
  int32_t placement;
  /** Actual backing: 0 fast, 1 slow, 2 heap. */
  int32_t tier;
  /** Reserved; zero. */
  int32_t reserved;
  /** Estimated bytes transferred through this buffer. */
  double estimated_traffic_bytes;
} VecopsWorkspaceAllocationV1;

/**
 * @brief Optional allocation observer attached to an execution context.
 *
 * The callback may be invoked concurrently from multiple worker threads
 * and must not acquire the Python GIL; push to a lock-free or mutex-held
 * native buffer instead. `record` is only valid during the call.
 */
typedef struct VecopsAllocationObserverV1 {
  /** Size of this record; must be `sizeof(VecopsAllocationObserverV1)`. */
  uint32_t struct_size;
  /** Opaque collector context passed back to the callback. */
  void* user_context;
  /** Allocation notification; never null when installed. */
  void (*on_workspace_allocation)(void* user_context,
                                  const VecopsWorkspaceAllocationV1* record);
} VecopsAllocationObserverV1;

typedef struct VecopsExecutionContext {
  /** Size of this record known to the caller. */
  uint32_t struct_size;
  /** Optional requested worker count; zero means embedding default. */
  uint32_t requested_threads;
  /** Optional framework stream pointer forwarded without interpretation. */
  void* stream;
  /** Opaque embedding-owned pointer. */
  void* user_data;
  /** Embedding-defined flags. */
  uint64_t flags;
  /** Optional physical workspace provider; absent from older short records. */
  const VecopsWorkspaceArenaProvider* workspace_provider;
  /** Optional synchronous logical-task executor; absent from older records. */
  const VecopsThreadPoolV1* thread_pool;
  /** Optional workspace allocation observer; absent from older records. */
  const VecopsAllocationObserverV1* allocation_observer;
} VecopsExecutionContext;

/**
 * @brief Complete transient argument frame for one workspace query or run.
 *
 * The value array is positional and follows the descriptor/schema order.
 * Workspace is optional and is not allocated by the runtime.  The caller
 * retains the call record, values, metadata arrays, workspace, and context
 * until the ABI function returns.
 */
typedef struct VecopsCall {
  /** Size of this record known to the caller. */
  uint32_t struct_size;
  /** Number of positional records in `values`. */
  uint32_t num_values;
  /** Non-owning positional ABI values. */
  const VecopsValue* values;
  /** Optional caller-owned external workspace allocation. */
  void* workspace;
  /** Available external workspace capacity in bytes. */
  uint64_t workspace_size;
  /** Optional non-owning execution context. */
  const VecopsExecutionContext* context;
} VecopsCall;

#ifdef __cplusplus
} // extern "C"
#endif

#endif // VECOPS_RUNTIME_CALL_ABI_H
