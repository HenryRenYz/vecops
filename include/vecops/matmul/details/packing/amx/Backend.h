//
// Copyright (c) vecops contributors.
//

/**
 * @file vecops/matmul/details/packing/amx/Backend.h
 * @brief The AMX format's single packing Backend: intrinsic packers when
 *        the compiled vector width allows, generic loops otherwise.
 *
 * On the AMX format the `Vector` implementation tag *is* the native
 * backend: this specialization upgrades to AVX-512 intrinsic packers
 * (dword transpose network, VPDPBUSD sidecar) whenever the access shape
 * permits and VEC_WIDTH >= 512, and falls back to the backend-independent
 * generic loops otherwise.  `eligible` is therefore false: this
 * specialization is the unconditional last link of the selection chain in
 * packing/Plan.h (on x86 the SME-family tags answer false through the
 * sentinel primary template), so it is never probed.  It is also the only
 * AMX-format specialization with `supports_column_compensation`
 * (run_compensated below).
 */

#ifndef VECOPS_MATMUL_DETAILS_PACK_AMX_BACKEND_H
#define VECOPS_MATMUL_DETAILS_PACK_AMX_BACKEND_H

#include "vecops/execution/ExecutionSession.h"
#include "vecops/matmul/details/packing/amx/Format.h"
#include "vecops/matmul/details/packing/amx/Pack.h"
#include "vecops/matmul/details/packing/generic/Pack.h"
#include "vecops/vec/Capabilities.h"

namespace vecops::kernel::matmul_pack_details {

/// ScalableTag scale factor whose vector holds exactly 16 elements of T —
/// one AMX panel row, the lane count the B access packers vectorize over.
/// Computed as a two-way power-of-two ratio between 16 * sizeof(T) bytes
/// and the native width (growing the native side for positive powers, the
/// requested side for negative ones), so it stays correct whether the
/// native vector is wider or narrower than one panel row.
template <typename T>
consteval int amx_panel_scale_power() {
#if VEC_WIDTH > 0
  nint_t native_bytes = VEC_WIDTH / 8;
#else
  nint_t native_bytes = 16;
#endif
  nint_t requested_bytes = 16 * static_cast<nint_t>(sizeof(T));
  int power = 0;
  while (requested_bytes > native_bytes) {
    native_bytes *= 2;
    ++power;
  }
  while (requested_bytes < native_bytes) {
    requested_bytes *= 2;
    --power;
  }
  return power;
}

/// AMX format under the Vector tag: intrinsic packers below the plan layer,
/// generic vector loops as the fallback.  Plain vector code throughout —
/// no execution resources beyond the default environment are needed.
template <>
struct Backend<
    ::vecops::matmul::details::amx::Format,
    matmul_pack_implementation::Vector> {
  using ResourceRequirements =
      typename execution::details::current_backend_t::DefaultRequirements;
  static constexpr bool supports_column_compensation = true;

  template <typename InputSpec, typename OutputSpec>
  static constexpr bool eligible = false;

  /**
   * @brief Pack `source` into the AMX block format for one operand.
   *
   * Dispatch tree, best path first:
   *
   * 1. Direct fast path — intrinsic packers over the raw pointer (A rows
   *    as 64-byte vector moves, B through the dword transpose).
   * 2. Access path, A side — pack_a_access vectorizes along K.
   * 3. Access path, B side — a K-contiguous input with a rowwise-
   *    compatible transform stages row-major and transposes; anything
   *    else loads spatial columns and interleaves (pack_b_access).
   * 4. Below AVX-512 — the backend-independent generic loops.
   */
  template <::vecops::matmul::Atom Atom, ::vecops::matmul::Operand Side,
            execution::ExecutionScope Scope,
            typename Source, typename Destination>
  VECOPS_ALWAYS_INLINE static void run(
      Scope&, const Source& source, Destination& destination) {
    using Packing = ::vecops::matmul::packing_t<Atom, Side>;
    using T = typename Packing::Element;
    static_assert(std::same_as<typename Source::ComputeType, T>);
    const auto& layout = source.spec().input_layout();
    const auto spatial = tensor::size_value<0>(layout);
    const auto k = tensor::size_value<1>(layout);
    using Spatial = std::remove_cvref_t<decltype(spatial)>;
    using K = std::remove_cvref_t<decltype(k)>;
    constexpr bool SpatialGuaranteed = Spatial::aligns(Packing::Panel);
    constexpr bool KGuaranteed = K::aligns(Packing::KTile);
    auto* output = reinterpret_cast<T*>(destination.raw_data());
    using Tag = vec::ScalableTag<T, amx_panel_scale_power<T>()>;
    // The direct fast path needs all three conditions: 512-bit vectors
    // (the transpose network and 64-byte row moves are AVX-512), a raw
    // pointer over the compute type (no transform, memory element ==
    // compute element — generic::RawDirectAccess), and a unit-stride K
    // axis so source rows are truly contiguous in memory.
    constexpr bool Direct = VEC_WIDTH >= 512 &&
        generic::RawDirectAccess<Source> &&
        std::same_as<
            tensor::stride_type_t<
                1, typename generic::SpecOf<Source>::InputLayout>,
            meta::Const<1>>;
    if constexpr (Direct) {
      const auto* input = reinterpret_cast<const T*>(source.raw_data());
      const nint_t row_stride = static_cast<nint_t>(
          tensor::stride_value<0>(layout));
      if constexpr (Side == ::vecops::matmul::Operand::A) {
        // Fully dynamic shapes keep the separate dynamic-ABI entry so GCC
        // retains its const-propagated clones (see the note in Pack.h).
        if constexpr (!SpatialGuaranteed && !KGuaranteed) {
          amx::pack_a_direct_dynamic(
              input, row_stride, output,
              static_cast<nint_t>(spatial), static_cast<nint_t>(k),
              Packing::Panel, Packing::KTile);
        } else {
          amx::pack_a_direct<SpatialGuaranteed, KGuaranteed>(
              input, row_stride, output,
              static_cast<nint_t>(spatial), static_cast<nint_t>(k));
        }
      } else {
        amx::pack_b_direct<
            Packing::KPack, SpatialGuaranteed, KGuaranteed>(
            input, row_stride, output,
            static_cast<nint_t>(spatial), static_cast<nint_t>(k));
      }
    } else if constexpr (Side == ::vecops::matmul::Operand::A) {
      if constexpr (VEC_WIDTH >= 512) {
        using ATag = vec::ScalableTag<T, 0>;
        amx::pack_a_access<Packing::KTile, ATag>(
            source, output, spatial, k);
      } else {
        generic::pack_blocked_rows<Tag>(
                                        source, output,
                                        static_cast<nint_t>(spatial),
                                        static_cast<nint_t>(k),
                                        Packing::Panel, Packing::KTile);
      }
    } else {
      if constexpr (VEC_WIDTH >= 512) {
        using Transform =
            typename generic::SpecOf<Source>::TransformType;
        // Packing a row-major B permutes element positions through the
        // dword transpose.  An elementwise, permutation-equivariant
        // transform commutes with that permutation, so transform-on-load
        // followed by the transpose equals transforming at the final
        // packed position — the row-major staging path is sound.  Any
        // other transform must go through the position-faithful column
        // loads of pack_b_access.
        constexpr bool RowwiseCompatibleTransform = [] {
          if constexpr (std::same_as<Transform, tensor::NoTransform>) {
            return true;
          } else {
            return Transform::is_elementwise &&
                Transform::permutation_equivariant;
          }
        }();
        constexpr bool RowMajorInput = std::same_as<
            tensor::stride_type_t<
                1, typename generic::SpecOf<Source>::InputLayout>,
            meta::Const<1>> && RowwiseCompatibleTransform;
        if constexpr (RowMajorInput) {
          amx::pack_b_row_major_access<
              Packing::KPack, Packing::KTile>(
                  source, output, spatial, k);
        } else {
          amx::pack_b_access<
              Packing::KPack, Packing::KTile, Tag>(
                  source, output, spatial, k);
        }
      } else {
        // ceil_div(Dynamic, Const) stays in the meta expression domain;
        // materialize to the runtime value before the narrowing multiply.
        const nint_t padded_groups =
            static_cast<nint_t>(ceil_div(k, Packing::KTile)) *
            (Packing::KTile / Packing::KPack);
        generic::pack_interleaved_panels<Packing::KPack, Tag>(
            source, output,
            static_cast<nint_t>(spatial), static_cast<nint_t>(k),
            Packing::Panel, padded_groups);
      }
    }
  }

  /**
   * @brief Pack signed-byte B and emit the asymmetric-A column sidecar.
   *
   * Two levels of fallback behind the direct path: a K-contiguous,
   * rowwise-compatible access packs row-major with the sidecar fused into
   * the transpose (pack_b_row_major_access_compensated); anything else —
   * including every sub-AVX-512 target — packs first on its fastest path
   * and derives the sidecar in a post-pass over the packed stream
   * (compensate_packed_s8_b).
   */
  template <::vecops::matmul::Atom Atom,
            execution::ExecutionScope Scope,
            typename Source, typename Destination,
            typename CompensationDestination>
  VECOPS_ALWAYS_INLINE static void run_compensated(
      Scope& scope, const Source& source, Destination& destination,
      CompensationDestination& compensation, int32_t a_zero_point) {
    using Packing = ::vecops::matmul::packing_t<Atom, ::vecops::matmul::Operand::B>;
    static_assert(std::same_as<typename Atom::TA, uint8_t>);
    static_assert(std::same_as<typename Atom::TB, int8_t>);
    static_assert(std::same_as<typename Atom::TAcc, int32_t>);
    static_assert(std::same_as<typename Source::ComputeType, int8_t>);
    static_assert(std::same_as<typename Destination::ComputeType, int8_t>);
    static_assert(std::same_as<
                  typename CompensationDestination::ComputeType, int32_t>);
    const auto& layout = source.spec().input_layout();
    const auto spatial = tensor::size_value<0>(layout);
    const auto k = tensor::size_value<1>(layout);
    using Spatial = std::remove_cvref_t<decltype(spatial)>;
    using K = std::remove_cvref_t<decltype(k)>;
    constexpr bool SpatialGuaranteed = Spatial::aligns(Packing::Panel);
    constexpr bool KGuaranteed = K::aligns(Packing::KTile);
    auto* output = reinterpret_cast<int8_t*>(destination.raw_data());
    auto* correction =
        reinterpret_cast<int32_t*>(compensation.raw_data());
    // The compensated vector paths (fused VPDPBUSD observers and the packed
    // post-pass) need AVX512_VNNI, which plain VEC_WIDTH >= 512 (e.g. the
    // skylake-avx512 capability baseline) does not imply.  Without it both
    // levels fall through to the portable pack-then-compensate fallback.
#if defined(HAS_AVX512_VNNI)
    constexpr bool CompensatedVectorPaths = VEC_WIDTH >= 512;
#else
    constexpr bool CompensatedVectorPaths = false;
#endif
    // Same three-condition probe as run()'s direct fast path.
    constexpr bool Direct = CompensatedVectorPaths &&
        generic::RawDirectAccess<Source> &&
        std::same_as<
            tensor::stride_type_t<
                1, typename generic::SpecOf<Source>::InputLayout>,
            meta::Const<1>>;
    if constexpr (Direct) {
      const auto* input =
          reinterpret_cast<const int8_t*>(source.raw_data());
      amx::pack_b_direct_compensated<
          Packing::KPack, SpatialGuaranteed, KGuaranteed>(
          input, static_cast<nint_t>(tensor::stride_value<0>(layout)),
          output, correction,
          static_cast<nint_t>(spatial), static_cast<nint_t>(k),
          a_zero_point);
    } else if constexpr (CompensatedVectorPaths) {
      using Transform = typename generic::SpecOf<Source>::TransformType;
      constexpr bool RowwiseCompatibleTransform = [] {
        if constexpr (std::same_as<Transform, tensor::NoTransform>) {
          return true;
        } else {
          return Transform::is_elementwise &&
              Transform::permutation_equivariant;
        }
      }();
      constexpr bool RowMajorInput = std::same_as<
          tensor::stride_type_t<
              1, typename generic::SpecOf<Source>::InputLayout>,
          meta::Const<1>> && RowwiseCompatibleTransform;
      if constexpr (RowMajorInput) {
        amx::pack_b_row_major_access_compensated<
            Packing::KPack, Packing::KTile>(
                source, output, correction, spatial, k, a_zero_point);
      } else {
        // Level two: reuse the ordinary pack, then compute the sidecar
        // from the packed stream (the packer stays on its fastest path).
        run<Atom, ::vecops::matmul::Operand::B>(scope, source, destination);
        amx::compensate_packed_s8_b(
            output, correction,
            static_cast<nint_t>(spatial), static_cast<nint_t>(k),
            a_zero_point);
      }
    } else {
      // Same pack-then-compensate fallback for sub-AVX-512 targets.
      run<Atom, ::vecops::matmul::Operand::B>(scope, source, destination);
      amx::compensate_packed_s8_b(
          output, correction,
          static_cast<nint_t>(spatial), static_cast<nint_t>(k),
          a_zero_point);
    }
  }
};

} // namespace vecops::kernel::matmul_pack_details

#endif // VECOPS_MATMUL_DETAILS_PACK_AMX_BACKEND_H
