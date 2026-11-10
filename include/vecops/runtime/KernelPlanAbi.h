// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
/** @file KernelPlanAbi.h
 * @brief Optional C ABI for persistent source-kernel grids.
 *
 * The artifact and all external tensor owners outlive the prepared object.
 * prepare builds metadata and resources without executing tensor computation.
 * bind validates exact captured metadata and changes addresses/scalars without
 * allocation. replay submits one team. An embedding can instead enter once,
 * call run_lane on every real resident worker, then leave after all workers
 * finish; this composes several kernels without intermediate submissions.
 */
#ifndef VECOPS_RUNTIME_KERNEL_PLAN_ABI_H
#define VECOPS_RUNTIME_KERNEL_PLAN_ABI_H
#include "vecops/runtime/CallAbi.h"
#ifdef __cplusplus
extern "C" {
#endif

#define VECOPS_KERNEL_PLAN_QUERY_SYMBOL_V1 "vecops_kernel_plan_query_v1"
#define VECOPS_KERNEL_PLAN_ABI_MAJOR 1u

/** Optional author-audited input/output alias, indexed by VecopsCall value.
 * Permission is not donation: the graph must independently prove ownership,
 * alias-group deadness and exact dtype/layout before rebinding the output.
 */
typedef struct VecopsKernelInplaceV1 {
  uint32_t input;
  uint32_t output;
} VecopsKernelInplaceV1;

/** Buffer declaration owned by the prepared metadata. Offsets/lifetimes are
 * local grid stages; no allocator, physical pool or tier is selected here. */
typedef struct VecopsKernelBufferRequirementV1 {
  /** Unique deterministic slot symbol, including its descriptive label.
   * Consumers must preserve this exact string for activation; labels alone
   * need not be unique within a kernel. */
  const char* name;
  uint64_t bytes, alignment, replicas, replica_stride;
  uint32_t domain;
  /** Prepared state may retain initialized contents between calls. A graph
   * program must keep this range live until the prepared object is suspended. */
  uint32_t retained;
  double estimated_traffic_bytes;
  const VecopsBufferLifetimeV1* lifetimes;
  size_t lifetime_count;
} VecopsKernelBufferRequirementV1;
/** Explicit borrowed mapping plus a retained physical lifetime token. Owner
 * callbacks are infrequent activation/suspension operations, never grid work. */
typedef struct VecopsKernelBufferBindingV1 {
  const char* name;
  void* address;
  uint64_t bytes, replica_stride;
} VecopsKernelBufferBindingV1;
typedef struct VecopsBufferOwnerV1 {
  void* context;
  void (*retain)(void*);
  void (*release)(void*);
} VecopsBufferOwnerV1;

typedef struct VecopsPreparedKernelV1 {
  uint32_t struct_size;
  uint32_t abi_major;
  void* context;
  uint32_t workers;
  uint32_t grids;
  /** Actual provider; embeddings must check resident-team capability. */
  const VecopsThreadPoolV1* thread_pool;
  int32_t (*bind)(void*, const VecopsCall*, VecopsError*);
  /** Expert graph entry: all metadata/optional-presence invariants are proven. */
  int32_t (*bind_prevalidated)(void*, const VecopsCall*, VecopsError*);
  int32_t (*replay)(void*, VecopsError*);
  int32_t (*enter)(void*, VecopsError*);
  int32_t (*run_lane)(void*, const VecopsTeamContextV1*, VecopsError*);
  void (*leave)(void*, uint32_t successful);
  void (*destroy)(void*);
  /** Optional shared-ownership extension. destroy releases one retained ref. */
  void (*retain)(void*);
  /** Stable copy of the captured execution services, owned by context. */
  const VecopsExecutionContext* execution_context;
  /** Optional extension; inspect struct_size before reading these fields. */
  uint32_t inplace_count;
  const VecopsKernelInplaceV1* inplace;
  /** Planning metadata stays valid through suspension and has no physical
   * allocation side effect. Activation requires one exact complete binding set. */
  uint32_t buffer_count;
  const VecopsKernelBufferRequirementV1* buffers;
  int32_t (*activate_buffers)(void*, const VecopsKernelBufferBindingV1*, size_t,
                              const VecopsBufferOwnerV1*, VecopsError*);
  /** Caller proves all graph/peer accesses stopped before suspension. Native
   * running instances reject suspension. Destroy prepared state before unmap. */
  int32_t (*suspend_buffers)(void*, VecopsError*);

} VecopsPreparedKernelV1;

typedef struct VecopsKernelPlanDescriptorV1 {
  uint32_t struct_size;
  uint32_t abi_major;
  /** Nonzero only for the KernelBuilder source model. */
  uint32_t supported;
  uint32_t reserved;
  int32_t (*prepare)(const VecopsCall*, VecopsPreparedKernelV1*, VecopsError*);
} VecopsKernelPlanDescriptorV1;
typedef const VecopsKernelPlanDescriptorV1* (*VecopsKernelPlanQueryFnV1)(void);

#ifdef __cplusplus
}
#endif
#endif
