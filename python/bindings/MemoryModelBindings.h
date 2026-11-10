// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
#pragma once
#include <pybind11/pybind11.h>

// Register pure resource-model data and queries in the caller's memory submodule.
// Runtime/native-owner adapters are registered separately by the module owner.
void bind_memory_model_types(pybind11::module_&);
