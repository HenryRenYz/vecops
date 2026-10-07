// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
//
//

#ifndef VECOPS_EXECUTION_DETAILS_ARM_RESOURCES_H
#define VECOPS_EXECUTION_DETAILS_ARM_RESOURCES_H

#include "vecops/execution/details/ResourceSet.h"

/**
 * @file vecops/execution/details/arm/Resources.h
 * @brief ARM manually-owned Streaming SVE and ZA resource tag.
 */

namespace vecops::execution::details::arm {

/** Resource proving that a scope owns both PSTATE.SM and destructive ZA. */
struct StreamingZA {};

/** Internal operation token used to open a manually-owned SME interval. */
struct StreamingZARegion {
  using ResourceRequirements = ResourceSet<StreamingZA>;
};

} // namespace vecops::execution::details::arm

#endif // VECOPS_EXECUTION_DETAILS_ARM_RESOURCES_H
