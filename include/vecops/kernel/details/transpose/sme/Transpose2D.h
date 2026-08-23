//
// Copyright (c) vecops contributors.
//

#ifndef VECOPS_KERNEL_DETAILS_TRANSPOSE_SME_TRANSPOSE2D_H
#define VECOPS_KERNEL_DETAILS_TRANSPOSE_SME_TRANSPOSE2D_H

#include <algorithm>
#include <cstdint>
#include <type_traits>

#include <arm_sme.h>

#include "vecops/CoreDefs.h"
#include "vecops/CoreTypes.h"
#include "vecops/execution/ExecutionSession.h"
#include "vecops/execution/details/arm/Resources.h"
#include "vecops/kernel/details/transpose/generic/Transpose2D.h"
#include "vecops/tensor/Layout.h"

/**
 * @file Transpose2D.h
 * @brief SME/ZA transpose leaf and its execution-resource backend wrapper.
 *
 * Rows are written horizontally into ZA and read vertically to obtain the
 * transpose. The outer execution scope owns streaming mode; each leaf declares
 * a private ZA lifetime. Element values are moved as unsigned bit patterns so
 * integer and floating-point types share the same ZA data path.
 */

namespace vecops::kernel::transpose2d_details::sme {

template <typename T>
/** Unsigned integer type with the same storage width as `T`. */
using UnsignedBits = std::conditional_t<
    sizeof(T) == 1, uint8_t,
    std::conditional_t<sizeof(T) == 2, uint16_t,
    std::conditional_t<sizeof(T) == 4, uint32_t, uint64_t>>>;

template <typename U>
/** Return an all-zero streaming SVE vector for unsigned element type `U`. */
VECOPS_ALWAYS_INLINE auto zero_sve() __arm_streaming;
template <> VECOPS_ALWAYS_INLINE auto zero_sve<uint8_t>() __arm_streaming {
  return svdup_u8(0);
}
template <> VECOPS_ALWAYS_INLINE auto zero_sve<uint16_t>() __arm_streaming {
  return svdup_u16(0);
}
template <> VECOPS_ALWAYS_INLINE auto zero_sve<uint32_t>() __arm_streaming {
  return svdup_u32(0);
}
template <> VECOPS_ALWAYS_INLINE auto zero_sve<uint64_t>() __arm_streaming {
  return svdup_u64(0);
}

template <typename U>
/** Return the current streaming vector length measured in `U` elements. */
VECOPS_ALWAYS_INLINE nint_t streaming_lanes() __arm_streaming {
  if constexpr (sizeof(U) == 1) return static_cast<nint_t>(svcntb());
  if constexpr (sizeof(U) == 2) return static_cast<nint_t>(svcnth());
  if constexpr (sizeof(U) == 4) return static_cast<nint_t>(svcntw());
  return static_cast<nint_t>(svcntd());
}

template <typename U>
/** Build a streaming predicate enabling the first `count` elements. */
VECOPS_ALWAYS_INLINE svbool_t first_predicate(nint_t count) __arm_streaming {
  if constexpr (sizeof(U) == 1) return svwhilelt_b8(nint_t{0}, count);
  if constexpr (sizeof(U) == 2) return svwhilelt_b16(nint_t{0}, count);
  if constexpr (sizeof(U) == 4) return svwhilelt_b32(nint_t{0}, count);
  return svwhilelt_b64(nint_t{0}, count);
}

template <typename U>
/** Read one vertical ZA slice, merging inactive lanes with zero. */
VECOPS_ALWAYS_INLINE auto read_ver(uint32_t slice, svbool_t pg)
    __arm_streaming __arm_inout("za") {
  if constexpr (sizeof(U) == 1) {
    return svread_ver_za8_u8_m(zero_sve<U>(), pg, 0, slice);
  } else if constexpr (sizeof(U) == 2) {
    return svread_ver_za16_u16_m(zero_sve<U>(), pg, 0, slice);
  } else if constexpr (sizeof(U) == 4) {
    return svread_ver_za32_u32_m(zero_sve<U>(), pg, 0, slice);
  } else {
    return svread_ver_za64_u64_m(zero_sve<U>(), pg, 0, slice);
  }
}

template <typename T>
/** Load `T` and reinterpret it as unsigned bits without numeric conversion. */
VECOPS_ALWAYS_INLINE auto load_value_bits(svbool_t pg, const T* pointer)
    __arm_streaming {
  if constexpr (std::same_as<T, float32_t>) {
    return svreinterpret_u32_f32(svld1_f32(pg, pointer));
  } else if constexpr (std::same_as<T, float64_t>) {
    return svreinterpret_u64_f64(svld1_f64(pg, pointer));
  } else if constexpr (std::same_as<T, int8_t>) {
    return svreinterpret_u8_s8(svld1_s8(pg, pointer));
  } else if constexpr (std::same_as<T, int16_t>) {
    return svreinterpret_u16_s16(svld1_s16(pg, pointer));
  } else if constexpr (std::same_as<T, int32_t>) {
    return svreinterpret_u32_s32(svld1_s32(pg, pointer));
  } else if constexpr (std::same_as<T, int64_t>) {
    return svreinterpret_u64_s64(svld1_s64(pg, pointer));
  } else {
    using U = UnsignedBits<T>;
    const auto* bits = reinterpret_cast<const U*>(pointer);
    if constexpr (sizeof(U) == 1) return svld1_u8(pg, bits);
    else if constexpr (sizeof(U) == 2) return svld1_u16(pg, bits);
    else if constexpr (sizeof(U) == 4) return svld1_u32(pg, bits);
    else return svld1_u64(pg, bits);
  }
}

template <typename T, typename Raw>
/** Store unsigned raw bits using the original element type `T`. */
VECOPS_ALWAYS_INLINE void store_value_bits(
    svbool_t pg, T* pointer, Raw value) __arm_streaming {
  if constexpr (std::same_as<T, float32_t>) {
    svst1_f32(pg, pointer, svreinterpret_f32_u32(value));
  } else if constexpr (std::same_as<T, float64_t>) {
    svst1_f64(pg, pointer, svreinterpret_f64_u64(value));
  } else if constexpr (std::same_as<T, int8_t>) {
    svst1_s8(pg, pointer, svreinterpret_s8_u8(value));
  } else if constexpr (std::same_as<T, int16_t>) {
    svst1_s16(pg, pointer, svreinterpret_s16_u16(value));
  } else if constexpr (std::same_as<T, int32_t>) {
    svst1_s32(pg, pointer, svreinterpret_s32_u32(value));
  } else if constexpr (std::same_as<T, int64_t>) {
    svst1_s64(pg, pointer, svreinterpret_s64_u64(value));
  } else {
    using U = UnsignedBits<T>;
    auto* bits = reinterpret_cast<U*>(pointer);
    if constexpr (sizeof(U) == 1) svst1_u8(pg, bits, value);
    else if constexpr (sizeof(U) == 2) svst1_u16(pg, bits, value);
    else if constexpr (sizeof(U) == 4) svst1_u32(pg, bits, value);
    else svst1_u64(pg, bits, value);
  }
}

template <typename U, typename Raw>
/** Write one horizontal ZA slice under `pg`. */
VECOPS_ALWAYS_INLINE void write_hor(
    uint32_t slice, svbool_t pg, Raw value)
    __arm_streaming __arm_inout("za") {
  if constexpr (sizeof(U) == 1) svwrite_hor_za8_u8_m(0, slice, pg, value);
  else if constexpr (sizeof(U) == 2)
    svwrite_hor_za16_u16_m(0, slice, pg, value);
  else if constexpr (sizeof(U) == 4)
    svwrite_hor_za32_u32_m(0, slice, pg, value);
  else svwrite_hor_za64_u64_m(0, slice, pg, value);
}

template <typename T, typename M, typename N,
          int SrcRank, int DstRank,
          int SrcRow, int SrcCol, int DstRow, int DstCol>
/**
 * @brief Transpose one logical plane through private ZA storage.
 * @tparam SrcRank Physical rank of the bound source.
 * @tparam DstRank Physical rank of the bound destination.
 * @tparam SrcRow Source axis forming logical rows.
 * @tparam SrcCol Source axis forming logical columns.
 * @tparam DstRow Destination axis receiving output rows.
 * @tparam DstCol Destination axis receiving output columns.
 * @param m Logical source rows; Const/constrained metadata is preserved.
 * @param n Logical source columns.
 * @param source_data Raw readable base pointer.
 * @param destination_data Raw writable base pointer.
 * @param src_strides Source element strides.
 * @param dst_strides Destination element strides.
 * @param src_origin Source logical plane origin.
 * @param dst_origin Destination logical plane origin.
 *
 * `__arm_new("za")` is intentional: the leaf clears/overwrites every used ZA
 * slice and does not consume caller ZA contents. Using `__arm_inout("za")`
 * would unnecessarily make the caller preserve a live ZA value. Current
 * compilers may still emit `__arm_tpidr2_save` as required by the SME ABI; that
 * helper is ABI state management, not a duplicate framework transition.
 */
__arm_new("za") VECOPS_NOINLINE void transpose(
    M m, N n,
    const T* source_data,
    tensor::Coord<SrcRank> src_strides,
    tensor::Coord<SrcRank> src_origin,
    T* destination_data,
    tensor::Coord<DstRank> dst_strides,
    tensor::Coord<DstRank> dst_origin)
    __arm_streaming {
  using U = UnsignedBits<T>;
  static_assert(sizeof(T) == sizeof(U));
  const nint_t lanes = streaming_lanes<U>();
  const nint_t m_extent = static_cast<nint_t>(m);
  const nint_t n_extent = static_cast<nint_t>(n);
  nint_t src_base = 0;
  VECOPS_UNROLL
  for (int d = 0; d < SrcRank; ++d) {
    src_base += src_origin[d] * src_strides[d];
  }
  nint_t dst_base = 0;
  VECOPS_UNROLL
  for (int d = 0; d < DstRank; ++d) {
    dst_base += dst_origin[d] * dst_strides[d];
  }
  for (nint_t mi = 0; mi < m_extent; mi += lanes) {
    const nint_t active_m = std::min(lanes, m_extent - mi);
    const svbool_t m_pg = first_predicate<U>(active_m);
    for (nint_t ni = 0; ni < n_extent; ni += lanes) {
      const nint_t active_n = std::min(lanes, n_extent - ni);
      const svbool_t n_pg = first_predicate<U>(active_n);

      for (nint_t r = 0; r < active_m; ++r) {
        const nint_t offset =
            src_base + (mi + r) * src_strides[SrcRow] +
            ni * src_strides[SrcCol];
        write_hor<U>(
            static_cast<uint32_t>(r), n_pg,
            load_value_bits<T>(n_pg, source_data + offset));
      }

      for (nint_t c = 0; c < active_n; ++c) {
        const auto raw = read_ver<U>(static_cast<uint32_t>(c), m_pg);
        const nint_t offset =
            dst_base + (ni + c) * dst_strides[DstRow] +
            mi * dst_strides[DstCol];
        store_value_bits<T>(m_pg, destination_data + offset, raw);
      }
    }
  }
}

} // namespace vecops::kernel::transpose2d_details::sme

namespace vecops::kernel::transpose2d_details {

/**
 * @brief SME transpose backend owning the streaming/ZA contract.
 *
 * The execution region owns PSTATE.SM, while each leaf invocation owns a
 * private ZA instance through `__arm_new("za")`. Only raw, unit-column-stride
 * accesses are accepted because conversion or layout materialization belongs
 * to DataAccess before entering this backend.
 */
struct SMEBackend {
  /** Execution resource required before entering the streaming leaf. */
  using ResourceRequirements = execution::details::ResourceSet<
      execution::details::arm::Streaming>;

  template <int SrcRow, int SrcCol, int DstRow, int DstCol,
            execution::ExecutionScope Scope,
            typename M, typename N, typename Source, typename Destination,
            typename Policy>
  /**
   * @brief Validate a direct DataAccess pair and invoke the SME leaf.
   * @param scope Active scope proving `arm::Streaming`.
   * @param m Logical source rows.
   * @param n Logical source columns.
   * @param source Bound direct readable access.
   * @param destination Bound direct writable access.
   * @param src_origin Source plane origin.
   * @param dst_origin Destination plane origin.
   */
  VECOPS_ALWAYS_INLINE static void run(
      Scope& scope, M m, N n,
      Source& source,
      tensor::Coord<std::remove_cvref_t<Source>::Rank> src_origin,
      Destination& destination,
      tensor::Coord<std::remove_cvref_t<Destination>::Rank> dst_origin,
      Policy) {
    static_assert(std::same_as<Policy, transpose2d_policy::Automatic>,
                  "SME transpose does not implement Gather policy");
    static_assert(
        execution::has_resource_v<execution::details::arm::Streaming, Scope>,
        "SME transpose requires an active streaming execution scope");
    static_assert(
        generic::RawDirectAccess<Source> &&
        generic::RawDirectAccess<Destination>,
        "SME transpose requires direct no-transform DataAccess with memory "
        "dtype equal to ComputeType");
    using SourceSpec = generic::SpecOf<Source>;
    using DestinationSpec = generic::SpecOf<Destination>;
    static_assert(std::same_as<
        tensor::stride_type_t<SrcCol, typename SourceSpec::InputLayout>,
        meta::Const<1>>);
    static_assert(std::same_as<
        tensor::stride_type_t<DstCol, typename DestinationSpec::OutputLayout>,
        meta::Const<1>>);
    using T = generic::ComputeOf<Source>;
    sme::transpose<
        T, M, N,
        std::remove_cvref_t<Source>::Rank,
        std::remove_cvref_t<Destination>::Rank,
        SrcRow, SrcCol, DstRow, DstCol>(
        m, n,
        reinterpret_cast<const T*>(source.raw_data()),
        source.raw_strides(), src_origin,
        reinterpret_cast<T*>(destination.raw_data()),
        destination.raw_strides(), dst_origin);
    (void)scope;
  }
};

} // namespace vecops::kernel::transpose2d_details

#endif // VECOPS_KERNEL_DETAILS_TRANSPOSE_SME_TRANSPOSE2D_H
