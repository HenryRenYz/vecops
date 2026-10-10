# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Small target-side packing/cache helpers, not matrix-product kernels."""

PREAMBLE = r"""
#include "vecops/platform/CacheInfo.h"
#include "vecops/vec/SVE.h"
namespace vecops_dsl {
// Codes: dense=0, AMX=1, SME=2, automatic=-1. The build host never selects
// the packed ABI of an AArch64 artifact.
template<class T, int Format> inline constexpr int packing_format = []() VECOPS_INLINE_LAMBDA {
  if constexpr (Format >= 0) return Format;
#if defined(HAS_AMX_TILE)
  if constexpr (sizeof(T) <= 2) return 1;
#elif defined(HAS_SME)
  return 2;
#endif
  return 0;
}();
template<class T, int Format> VECOPS_ALWAYS_INLINE vecops::nint_t pack_panel() noexcept {
  if constexpr (packing_format<T, Format> == 1) {
#if defined(HAS_AMX_TILE)
    return 16;
#else
    return 0;
#endif
  }
  else if constexpr (packing_format<T, Format> == 2) {
#if defined(HAS_SME)
    if constexpr (sizeof(T) == 8) return 2 * vecops::vec::sme::streaming_lanes<T>();
    else return 2 * vecops::vec::sme::streaming_lanes<vecops::float32_t>();
#else
    return 0;
#endif
  } else return 1;
}
template<class T, int Format> inline constexpr vecops::nint_t pack_group =
    packing_format<T, Format> == 0 ? 1 : (sizeof(T) > 4 ? 1 : 4 / sizeof(T));
template<class T, int Format> VECOPS_ALWAYS_INLINE vecops::nint_t pack_kblock() noexcept {
  if constexpr (packing_format<T, Format> == 1) return 64 / sizeof(T);
  else return pack_group<T, Format>;
}
template<class T, int Format> inline vecops::nint_t pack_size(vecops::nint_t s, vecops::nint_t k) {
  const auto panel = pack_panel<T, Format>(), block = pack_kblock<T, Format>();
  if (s < 0 || k < 0 || panel <= 0 || block <= 0) throw std::invalid_argument("invalid packed matrix extents/target");
  const __int128 count = ((static_cast<__int128>(s) + panel - 1) / panel) * panel
                      * ((static_cast<__int128>(k) + block - 1) / block) * block;
  if (count > std::numeric_limits<vecops::nint_t>::max() / sizeof(T))
    throw std::overflow_error("packed matrix extent overflow");
  return static_cast<vecops::nint_t>(count);
}
template<class T, int Format, bool SideB>
VECOPS_ALWAYS_INLINE vecops::nint_t pack_offset(vecops::nint_t s, vecops::nint_t k,
                                              vecops::nint_t spatial, vecops::nint_t logical_k) noexcept {
  if constexpr (packing_format<T, Format> == 0) return s * logical_k + k;
  else {
    const auto panel = pack_panel<T, Format>(), block = pack_kblock<T, Format>();
    const auto groups = (logical_k + block - 1) / block;
    constexpr auto kp = pack_group<T, Format>;
    if constexpr (packing_format<T, Format> == 1 && !SideB)
      return ((s / panel) * groups + k / block) * panel * block + (s % panel) * block + k % block;
    else
      return ((s / panel) * groups + k / block) * panel * block
           + (k % block / kp) * panel * kp + (s % panel) * kp + k % kp;
  }
}
} // namespace vecops_dsl
"""
