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
#include "vecops/vec/ConversionMemory.h"

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
using UnsignedBits = generic::UIntOfSize<sizeof(T)>;

template <typename U>
/** Read one vertical ZA slice, merging inactive lanes with zero. */
VECOPS_ALWAYS_INLINE auto read_ver(uint32_t slice, svbool_t pg)
    __arm_streaming __arm_inout("za") {
  using Tag = vec::ScalableTag<U, 0>;
  const auto zero = vec::details::sve_basic_raw_word(vec::zeros(Tag{}));
  if constexpr (sizeof(U) == 1) {
    return svread_ver_za8_u8_m(zero, pg, 0, slice);
  } else if constexpr (sizeof(U) == 2) {
    return svread_ver_za16_u16_m(zero, pg, 0, slice);
  } else if constexpr (sizeof(U) == 4) {
    return svread_ver_za32_u32_m(zero, pg, 0, slice);
  } else {
    return svread_ver_za64_u64_m(zero, pg, 0, slice);
  }
}

template <vec::VectorTag Tag>
/** Bridge a raw ZA/SVE predicate to the active vec backend representation. */
VECOPS_ALWAYS_INLINE vec::Mask<Tag> predicate_mask(svbool_t pg)
    __arm_streaming {
#if defined(HAS_FIXED_SVE_BITS)
  using WordMask = vec::NativeWordMask<Tag>;
  return vec::mask_from_words(Tag{}, static_cast<WordMask>(pg));
#else
  return pg;
#endif
}

template <typename Compute, typename Memory>
/** Load memory lanes as Compute and reinterpret them as unsigned ZA bits. */
VECOPS_ALWAYS_INLINE auto load_compute_bits(
    svbool_t pg, nint_t active_lanes, const Memory* pointer)
    __arm_streaming {
  using U = UnsignedBits<Compute>;
  using ValueTag = vec::ScalableTag<Compute, 0>;
  using BitsTag = vec::ScalableTag<U, 0>;
  const auto active = predicate_mask<ValueTag>(pg);
  const auto value = [&]() VECOPS_INLINE_LAMBDA {
    // Besides removing predicate work, the unmasked full-row form keeps a
    // width-changing Rebind on the SVE whole-operation lowering. The generic
    // masked oversized boundary must otherwise preserve lanes one by one.
    if (active_lanes == vec::size(ValueTag{})) {
      if constexpr (std::same_as<std::remove_cv_t<Memory>, Compute>) {
        return vec::load(ValueTag{}, pointer);
      } else {
        return vec::load_convert(ValueTag{}, pointer);
      }
    } else {
      if constexpr (std::same_as<std::remove_cv_t<Memory>, Compute>) {
        return vec::load(
            ValueTag{}, pointer, vec::opt::masked(active), vec::opt::zero);
      } else {
        return vec::load_convert(
            ValueTag{}, pointer, vec::opt::masked(active), vec::opt::zero);
      }
    }
  }();
  return vec::details::sve_basic_raw_word(
      vec::bitcast(BitsTag{}, ValueTag{}, value));
}

template <typename Compute, typename Memory, typename Raw>
/** Reinterpret unsigned ZA bits as Compute and store-convert if required. */
VECOPS_ALWAYS_INLINE void store_compute_bits(
    svbool_t pg, nint_t active_lanes, Memory* pointer, Raw value)
    __arm_streaming {
  using U = UnsignedBits<Compute>;
  using ValueTag = vec::ScalableTag<Compute, 0>;
  using BitsTag = vec::ScalableTag<U, 0>;
  const auto active = predicate_mask<ValueTag>(pg);
  const auto bits = vec::details::sve_basic_wrap_word<BitsTag>(value);
  const auto converted = vec::bitcast(ValueTag{}, BitsTag{}, bits);
  if (active_lanes == vec::size(ValueTag{})) {
    if constexpr (std::same_as<std::remove_cv_t<Memory>, Compute>) {
      vec::store(ValueTag{}, pointer, converted);
    } else {
      vec::store_convert(ValueTag{}, pointer, converted);
    }
  } else {
    if constexpr (std::same_as<std::remove_cv_t<Memory>, Compute>) {
      vec::store(
          ValueTag{}, pointer, converted, vec::opt::masked(active));
    } else {
      vec::store_convert(
          ValueTag{}, pointer, converted, vec::opt::masked(active));
    }
  }
}

template <typename U, typename Raw>
/** Write one horizontal ZA slice under `pg`. */
VECOPS_ALWAYS_INLINE void write_hor(
    uint32_t slice, svbool_t pg, Raw value)
    __arm_streaming __arm_inout("za") {
  const auto raw = vec::details::sve_basic_raw_word(value);
  if constexpr (sizeof(U) == 1) svwrite_hor_za8_u8_m(0, slice, pg, raw);
  else if constexpr (sizeof(U) == 2)
    svwrite_hor_za16_u16_m(0, slice, pg, raw);
  else if constexpr (sizeof(U) == 4)
    svwrite_hor_za32_u32_m(0, slice, pg, raw);
  else svwrite_hor_za64_u64_m(0, slice, pg, raw);
}

template <typename SourceMemory, typename DestinationMemory, typename Compute,
          typename M, typename N,
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
    const SourceMemory* source_data,
    tensor::Coord<SrcRank> src_strides,
    tensor::Coord<SrcRank> src_origin,
    DestinationMemory* destination_data,
    tensor::Coord<DstRank> dst_strides,
    tensor::Coord<DstRank> dst_origin)
    __arm_streaming {
  using U = UnsignedBits<Compute>;
  using BitsTag = vec::ScalableTag<U, 0>;
  constexpr bool SourceNative =
      std::same_as<std::remove_cv_t<SourceMemory>, Compute>;
  constexpr bool DestinationNative =
      std::same_as<std::remove_cv_t<DestinationMemory>, Compute>;
  // Two independent rows/columns let the core overlap memory and ZA moves
  // without carrying a vector across loop iterations. Keep this deliberately
  // type/shape selective: A/B measurements found register-pressure regressions
  // for load-convert, 8-byte store-convert, same-type 16-bit, Dynamic/Any, and
  // several smaller/rectangular matrices. Non-selected instantiations retain
  // the original single-row/single-column loops with no runtime dispatch.
  constexpr bool PipelinePairs = [=] {
    if constexpr (!M::is_const || !N::is_const || !SourceNative) {
      return false;
    } else {
      constexpr nint_t Elements = M::value * N::value;
      if constexpr (DestinationNative) {
        if constexpr (sizeof(Compute) == 1 || sizeof(Compute) == 4)
          return Elements >= 1024 * 1024;
        else if constexpr (sizeof(Compute) == 8)
          return M::value == N::value && Elements >= 1024 * 1024;
      } else if constexpr (
          sizeof(Compute) == 4 && sizeof(DestinationMemory) == 2) {
        return Elements >= 8000;
      }
      return false;
    }
  }();
  static_assert(sizeof(Compute) == sizeof(U));
  const nint_t lanes = vec::size(BitsTag{});
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
    const svbool_t m_pg = vec::mwhilelt(BitsTag{}, nint_t{0}, active_m);
    for (nint_t ni = 0; ni < n_extent; ni += lanes) {
      const nint_t active_n = std::min(lanes, n_extent - ni);
      const svbool_t n_pg = vec::mwhilelt(BitsTag{}, nint_t{0}, active_n);

      if constexpr (PipelinePairs) {
        nint_t r = 0;
        for (; r + 1 < active_m; r += 2) {
          const nint_t offset0 =
              src_base + (mi + r) * src_strides[SrcRow] +
              ni * src_strides[SrcCol];
          const nint_t offset1 = offset0 + src_strides[SrcRow];
          const auto value0 = load_compute_bits<Compute>(
              n_pg, active_n, source_data + offset0);
          const auto value1 = load_compute_bits<Compute>(
              n_pg, active_n, source_data + offset1);
          write_hor<U>(static_cast<uint32_t>(r), n_pg, value0);
          write_hor<U>(static_cast<uint32_t>(r + 1), n_pg, value1);
        }
        if (r < active_m) {
          const nint_t offset =
              src_base + (mi + r) * src_strides[SrcRow] +
              ni * src_strides[SrcCol];
          write_hor<U>(
              static_cast<uint32_t>(r), n_pg,
              load_compute_bits<Compute>(
                  n_pg, active_n, source_data + offset));
        }
      } else {
        for (nint_t r = 0; r < active_m; ++r) {
          const nint_t offset =
              src_base + (mi + r) * src_strides[SrcRow] +
              ni * src_strides[SrcCol];
          write_hor<U>(
              static_cast<uint32_t>(r), n_pg,
              load_compute_bits<Compute>(
                  n_pg, active_n, source_data + offset));
        }
      }

      if constexpr (PipelinePairs) {
        nint_t c = 0;
        for (; c + 1 < active_n; c += 2) {
          const auto raw0 = read_ver<U>(static_cast<uint32_t>(c), m_pg);
          const auto raw1 = read_ver<U>(static_cast<uint32_t>(c + 1), m_pg);
          const nint_t offset0 =
              dst_base + (ni + c) * dst_strides[DstRow] +
              mi * dst_strides[DstCol];
          const nint_t offset1 = offset0 + dst_strides[DstRow];
          store_compute_bits<Compute>(
              m_pg, active_m, destination_data + offset0, raw0);
          store_compute_bits<Compute>(
              m_pg, active_m, destination_data + offset1, raw1);
        }
        if (c < active_n) {
          const auto raw = read_ver<U>(static_cast<uint32_t>(c), m_pg);
          const nint_t offset =
              dst_base + (ni + c) * dst_strides[DstRow] +
              mi * dst_strides[DstCol];
          store_compute_bits<Compute>(
              m_pg, active_m, destination_data + offset, raw);
        }
      } else {
        for (nint_t c = 0; c < active_n; ++c) {
          const auto raw = read_ver<U>(static_cast<uint32_t>(c), m_pg);
          const nint_t offset =
              dst_base + (ni + c) * dst_strides[DstRow] +
              mi * dst_strides[DstCol];
          store_compute_bits<Compute>(
              m_pg, active_m, destination_data + offset, raw);
        }
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
 * private ZA instance through `__arm_new("za")`. Raw no-transform,
 * unit-column-stride accesses are accepted. Numeric conversion is fused into
 * the SME tile boundary: load-convert before ZA and/or store-convert after ZA.
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
        generic::RawNoTransformAccess<Source> &&
        generic::RawNoTransformAccess<Destination>,
        "SME transpose requires raw no-transform DataAccess");
    using SourceSpec = generic::SpecOf<Source>;
    using DestinationSpec = generic::SpecOf<Destination>;
    static_assert(std::same_as<
        tensor::stride_type_t<SrcCol, typename SourceSpec::InputLayout>,
        meta::Const<1>>);
    static_assert(std::same_as<
        tensor::stride_type_t<DstCol, typename DestinationSpec::OutputLayout>,
        meta::Const<1>>);
    using Compute = generic::ComputeOf<Source>;
    using SourceMemory =
        typename std::remove_cvref_t<Source>::MemoryElement;
    using DestinationMemory =
        typename std::remove_cvref_t<Destination>::MemoryElement;
    sme::transpose<
        SourceMemory, DestinationMemory, Compute, M, N,
        std::remove_cvref_t<Source>::Rank,
        std::remove_cvref_t<Destination>::Rank,
        SrcRow, SrcCol, DstRow, DstCol>(
        m, n,
        source.raw_data(),
        source.raw_strides(), src_origin,
        destination.raw_data(),
        destination.raw_strides(), dst_origin);
    (void)scope;
  }
};

} // namespace vecops::kernel::transpose2d_details

#endif // VECOPS_KERNEL_DETAILS_TRANSPOSE_SME_TRANSPOSE2D_H
