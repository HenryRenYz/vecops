// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
/**
 * @file Runtime.h
 * @brief Convenience include for the framework-neutral runtime public API.
 *
 * Including this header makes ABI declarations, C++ argument adaptation,
 * schema validation, dynamic executables, dispatch operators, statuses, and
 * `tensor::Tensor` adaptation available.  It owns no global runtime state and
 * has no side effects beyond normal C++ header inclusion.
 */
#ifndef VECOPS_RUNTIME_RUNTIME_H
#define VECOPS_RUNTIME_RUNTIME_H

#include "vecops/runtime/CallAbi.h"
#include "vecops/runtime/Argument.h"
#include "vecops/runtime/Executable.h"
#include "vecops/runtime/KernelAbi.h"
#include "vecops/runtime/OperatorBridgeAbi.h"
#include "vecops/runtime/KernelDefinition.h"
#include "vecops/runtime/Operator.h"
#include "vecops/runtime/Provider.h"
#include "vecops/runtime/Schema.h"
#include "vecops/runtime/Status.h"
#include "vecops/tensor/Tensor.h"

#endif // VECOPS_RUNTIME_RUNTIME_H
