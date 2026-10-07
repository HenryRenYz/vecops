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
struct MatmulBlocks { vecops::nint_t m, n, k; };
VECOPS_ALWAYS_INLINE bool matmul_budget_part(vecops::nint_t rows, vecops::nint_t cols,
                                            vecops::nint_t bytes, vecops::nint_t& budget) noexcept {
  if (rows == 0 || cols == 0) return true;
  if (rows > budget / bytes / cols) return false;
  budget -= rows * cols * bytes; // The quotient test proved this fits.
  return true;
}
VECOPS_ALWAYS_INLINE bool matmul_fits(vecops::nint_t m, vecops::nint_t n, vecops::nint_t k,
                                     vecops::nint_t a_bytes, vecops::nint_t b_bytes,
                                     vecops::nint_t c_bytes, vecops::nint_t budget) noexcept {
  return matmul_budget_part(m,n,c_bytes,budget)
      && matmul_budget_part(m,k,a_bytes,budget)
      && matmul_budget_part(n,k,b_bytes,budget);
}
inline MatmulBlocks matmul_blocks(vecops::nint_t mr, vecops::nint_t nr, vecops::nint_t kr,
                                 vecops::nint_t a_bytes, vecops::nint_t b_bytes,
                                 vecops::nint_t c_bytes,vecops::nint_t logical_k,
                                 vecops::nint_t b_panel_divisor=0) {
  const auto& cache = vecops::platform::cache_info();
  auto align = [](vecops::nint_t n, vecops::nint_t a) VECOPS_INLINE_LAMBDA {
    return std::max(a, n / a * a);
  };
#if defined(HAS_AMX_TILE)
  // Cache arithmetic uses instruction-tile extents, not the 2x2 register
  // family span; confusing these doubles unnecessary K accumulator traffic.
  const auto atom_m=std::min<vecops::nint_t>(16,mr),atom_n=std::min<vecops::nint_t>(16,nr);
#else
  const auto atom_m=mr,atom_n=nr;
#endif
  auto kc = align(std::max<vecops::nint_t>(cache.l1d_bytes / 2, 1) / (atom_m*a_bytes + atom_n*b_bytes), kr);
  // Keep accumulators live for the whole K when the complete micro-panel
  // fits private L2. L1-only K sizing needlessly writes/reads C every phase.
  if (logical_k>0 && static_cast<__int128>(logical_k)*(mr*a_bytes+nr*b_bytes)<=cache.l2_bytes/2)
    kc=align(((logical_k+kr-1)/kr)*kr,kr);
  const auto budget = std::max<vecops::nint_t>(cache.l2_bytes * 3 / 4, cache.line_bytes);
  auto mc = align(budget / 3 / (kc*a_bytes), mr);
  auto nc = align((budget - budget/3) / (kc*b_bytes), nr);
  // Architecture policy may reserve room for B's producer footprint as well
  // as its packed panel and A. Also useful for external packed inputs: the
  // smaller traversal panel improves operand residency across M groups.
  // Explicit BN is applied by the caller after this automatic calculation.
  if (b_panel_divisor > 0)
    nc = std::min(nc, align(cache.l2_bytes / b_panel_divisor / (kc*b_bytes), nr));
  // L3 is optional. Cap the accumulator working set if a shared last-level
  // cache is present; leave L1/L2-derived values untouched otherwise.
  const auto outer_cache = cache.llc_bytes > 0 ? cache.llc_bytes :
                           cache.l3_present ? cache.l3_bytes : 0;
  if (outer_cache > 0) {
    const auto output_budget = std::max<vecops::nint_t>(mr*nr*c_bytes,outer_cache / 4);
    while (static_cast<__int128>(mc)*nc*c_bytes > output_budget && (mc > mr || nc > nr)) {
      if (nc >= mc && nc > nr) nc = align(nc/2, nr);
      else mc = align(mc/2, mr);
    }
  }
  return {mc, nc, kc};
}
} // namespace vecops_dsl
"""
