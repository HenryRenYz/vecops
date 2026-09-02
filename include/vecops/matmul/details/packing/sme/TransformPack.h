//
// Copyright (c) vecops contributors.
//

#ifndef VECOPS_MATMUL_DETAILS_PACK_SME_TRANSFORM_PACK_H
#define VECOPS_MATMUL_DETAILS_PACK_SME_TRANSFORM_PACK_H

#include <type_traits>

#include "vecops/matmul/details/packing/sme/Pack.h"
#include "vecops/tensor/DataAccess.h"
#include "vecops/vec/Vec.h"

namespace vecops::kernel::matmul_pack_details::sme {

template <typename T>
inline constexpr int panel_scale_power =
    sizeof(T) == 4 ? 1 : (sizeof(T) == 2 ? 0 : -1);

template <typename T>
using PanelTag = vec::ScalableTag<T, panel_scale_power<T>>;

template <typename T>
using WordTag = vec::ScalableTag<T, 0>;

template <typename T, typename Mask0, typename Mask1>
VECOPS_ALWAYS_INLINE vec::Vec<PanelTag<T>> read_panel(
    uint32_t slice, Mask0 pg0, Mask1 pg1) noexcept {
  using U = Bits<T>;
  using BitsPanelTag = PanelTag<U>;
  if constexpr (sizeof(T) == 4) {
    const auto lo_bits = read_ver<0, U>(slice, pg0);
    const auto hi_bits = read_ver<1, U>(slice, pg1);
    const auto lo = vec::bitcast(
        WordTag<T>{}, WordTag<U>{}, lo_bits);
    const auto hi = vec::bitcast(
        WordTag<T>{}, WordTag<U>{}, hi_bits);
    return vec::concat(PanelTag<T>{}, lo, hi);
  } else {
    const auto bits = read_ver<0, U>(slice, pg0);
    return vec::bitcast(PanelTag<T>{}, BitsPanelTag{}, bits);
  }
}

template <typename T, typename Source>
VECOPS_ALWAYS_INLINE vec::Vec<PanelTag<T>> postprocess_panel(
    const Source& source,
    vec::Vec<PanelTag<typename Source::MemoryElement>> input) noexcept {
  using Memory = typename Source::MemoryElement;
  using Transform = typename Source::Transform;
  using MemoryTag = PanelTag<Memory>;
  using OutputTag = PanelTag<T>;
  if constexpr (std::same_as<Transform, tensor::NoTransform>) {
    if constexpr (std::same_as<Memory, T>) return input;
    else return vec::convert(OutputTag{}, MemoryTag{}, input);
  } else {
    using TransformInTag = vec::Rebind<typename Transform::TIn, MemoryTag>;
    using TransformOutTag = vec::Rebind<typename Transform::TOut, MemoryTag>;
    const auto transform_input = [&]() VECOPS_INLINE_LAMBDA_NOEXCEPT {
      if constexpr (std::same_as<Memory, typename Transform::TIn>) return input;
      else return vec::convert(TransformInTag{}, MemoryTag{}, input);
    }();
    const auto transformed = source.spec().transform()(
        TransformOutTag{}, transform_input);
    if constexpr (std::same_as<typename Transform::TOut, T>) {
      return transformed;
    } else {
      return vec::convert(OutputTag{}, TransformOutTag{}, transformed);
    }
  }
}

template <typename T>
VECOPS_ALWAYS_INLINE vec::Vec<PanelTag<T>> zero_inactive_spatial(
    vec::Vec<PanelTag<T>> value, nint_t active_spatial) noexcept {
  using Tag = PanelTag<T>;
  const nint_t panel =
      2 * vec::details::sme::streaming_lanes<uint32_t>();
  if (active_spatial >= panel) return value;
  return vec::blend(
      Tag{}, vec::zeros(Tag{}),
      vec::mwhilelt(Tag{}, 0, active_spatial), value);
}

template <typename T>
VECOPS_ALWAYS_INLINE void store_single_panel(
    T*& output, vec::Vec<PanelTag<T>> value) noexcept {
  static_assert(sizeof(T) == 4);
  using U = Bits<T>;
  using Tag = PanelTag<T>;
  using BitsWordTag = WordTag<U>;
  const auto lo = vec::bitcast(
      BitsWordTag{}, WordTag<T>{}, vec::lower(Tag{}, value));
  const auto hi = vec::bitcast(
      BitsWordTag{}, WordTag<T>{}, vec::upper(Tag{}, value));
  vec::store(BitsWordTag{}, reinterpret_cast<U*>(output), lo);
  output += vec::size(BitsWordTag{});
  vec::store(BitsWordTag{}, reinterpret_cast<U*>(output), hi);
  output += vec::size(BitsWordTag{});
}

template <typename T>
VECOPS_ALWAYS_INLINE void store_pair_panel(
    T*& output,
    vec::Vec<PanelTag<T>> v0,
    vec::Vec<PanelTag<T>> v1) noexcept {
  static_assert(sizeof(T) == 2);
  using U = Bits<T>;
  using BitsTag = PanelTag<U>;
  store_pair<U>(
      output,
      vec::bitcast(BitsTag{}, PanelTag<T>{}, v0),
      vec::bitcast(BitsTag{}, PanelTag<T>{}, v1));
  output += 2 * vec::size(WordTag<U>{});
}

template <typename T>
VECOPS_ALWAYS_INLINE void store_quad_panel(
    T*& output,
    vec::Vec<PanelTag<T>> v0,
    vec::Vec<PanelTag<T>> v1,
    vec::Vec<PanelTag<T>> v2,
    vec::Vec<PanelTag<T>> v3) noexcept {
  static_assert(sizeof(T) == 1);
  using U = Bits<T>;
  using BitsTag = PanelTag<U>;
  const nint_t panel =
      2 * vec::details::sme::streaming_lanes<uint32_t>();
  const auto pg = first<U>(panel);
  store_quad<U>(
      output, pg,
      vec::bitcast(BitsTag{}, PanelTag<T>{}, v0),
      vec::bitcast(BitsTag{}, PanelTag<T>{}, v1),
      vec::bitcast(BitsTag{}, PanelTag<T>{}, v2),
      vec::bitcast(BitsTag{}, PanelTag<T>{}, v3));
  output += 4 * panel;
}

template <::vecops::matmul::Atom Atom, ::vecops::matmul::Operand Side,
          typename Source,
          meta::ValueType Spatial, meta::ValueType K,
          meta::ValueType RowStride>
VECOPS_ALWAYS_INLINE void pack_postprocess(
    const Source& source, Spatial spatial_meta, K k_meta,
    RowStride row_stride_meta,
    typename ::vecops::matmul::packing_t<Atom, Side>::Element* output) noexcept {
  using Packing = ::vecops::matmul::packing_t<Atom, Side>;
  using T = typename Packing::Element;
  using Memory = typename Source::MemoryElement;
  using U = Bits<Memory>;
  const nint_t spatial = static_cast<nint_t>(spatial_meta);
  const nint_t k = static_cast<nint_t>(k_meta);
  const nint_t row_stride = static_cast<nint_t>(row_stride_meta);
  constexpr nint_t KPack = Packing::KPack;
  const nint_t panel =
      2 * vec::details::sme::streaming_lanes<uint32_t>();
  const nint_t lanes = vec::size(WordTag<Memory>{});
  const nint_t k_chunk = lanes;
  const nint_t panels = ceil_div(spatial, panel);
  const auto* input = source.raw_data();
  auto* out = output;

  for (nint_t sp = 0; sp < panels; ++sp) {
    const nint_t panel_base = sp * panel;
    const nint_t active_spatial = vec::details::sme::clamp_value(
        spatial - panel_base, nint_t{0}, panel);
    const nint_t active0 = vec::details::sme::min_value(active_spatial, lanes);
    const nint_t active1 = sizeof(Memory) == 4
        ? vec::details::sme::max_value(nint_t{0}, active_spatial - lanes)
        : 0;
    const auto pg0 = first<U>(active0);
    const auto pg1 = first<U>(active1);
    const auto write_pg = all<U>();

    auto pack_chunk = [&]<bool FullPanel, bool FullK>(
        nint_t kb, nint_t active_k)
        VECOPS_INLINE_LAMBDA_NOEXCEPT {
      const auto load_pg = FullK ? all<U>() : first<U>(active_k);
      const auto read_pg0 = FullPanel ? all<U>() : pg0;
      const auto read_pg1 = FullPanel ? all<U>() : pg1;
      const nint_t rows0 = FullPanel ? lanes : active0;
      const nint_t rows1 = FullPanel ? lanes : active1;
      auto load_row = [&](nint_t row) VECOPS_INLINE_LAMBDA_NOEXCEPT {
        return load_bits<U>(load_pg, input + row * row_stride + kb);
      };
      auto write_rows = [&]<int Tile>(nint_t begin, nint_t count)
          VECOPS_INLINE_LAMBDA_NOEXCEPT {
        for (nint_t r = 0; r < count; ++r) {
          const nint_t row = begin + r;
          write_hor<Tile, U>(
              static_cast<uint32_t>(r), write_pg, load_row(row));
          }
      };
      write_rows.template operator()<0>(panel_base, rows0);
      if constexpr (sizeof(Memory) == 4) {
        write_rows.template operator()<1>(panel_base + lanes, rows1);
      }

      auto process_slice = [&](nint_t slice)
          VECOPS_INLINE_LAMBDA_NOEXCEPT {
        using OutputTag = PanelTag<T>;
        if constexpr (!FullK) {
          if (slice >= active_k) return vec::zeros(OutputTag{});
        }
        const auto memory = read_panel<Memory>(
            static_cast<uint32_t>(slice), read_pg0, read_pg1);
        const auto processed = postprocess_panel<T>(source, memory);
        if constexpr (FullPanel) return processed;
        else return zero_inactive_spatial<T>(processed, active_spatial);
      };

      const nint_t groups = FullK
          ? k_chunk / KPack
          : ceil_div(active_k, KPack);
      if constexpr (KPack == 1) {
        for (nint_t g = 0; g < groups; ++g) {
          store_single_panel(out, process_slice(g));
        }
      } else if constexpr (KPack == 2) {
        for (nint_t g = 0; g < groups; ++g) {
          store_pair_panel(
              out, process_slice(g * KPack),
              process_slice(g * KPack + 1));
        }
      } else {
        for (nint_t g = 0; g < groups; ++g) {
          store_quad_panel(
              out, process_slice(g * KPack),
              process_slice(g * KPack + 1),
              process_slice(g * KPack + 2),
              process_slice(g * KPack + 3));
        }
      }
    };

    auto pack_panel = [&]<bool FullPanel>() VECOPS_INLINE_LAMBDA_NOEXCEPT {
      nint_t kb = 0;
      for (; kb + k_chunk <= k; kb += k_chunk) {
        pack_chunk.template operator()<FullPanel, true>(kb, k_chunk);
      }
      if (kb < k) {
        pack_chunk.template operator()<FullPanel, false>(kb, k - kb);
      }
    };
    if (active_spatial == panel) pack_panel.template operator()<true>();
    else pack_panel.template operator()<false>();
  }
}

template <typename T>
VECOPS_ALWAYS_INLINE void zero_contiguous(
    T* pointer, nint_t count) {
  using Tag = vec::ScalableTag<T, 0>;
  const nint_t lanes = vec::size(Tag{});
  const auto zero = vec::zeros(Tag{});
  for (nint_t offset = 0; offset < count; offset += lanes) {
    vec::store(
        Tag{}, pointer + offset, zero,
        vec::opt::first(vec::details::sme::min_value(lanes, count - offset)));
  }
}

template <typename T>
VECOPS_ALWAYS_INLINE void zero_strided(
    T* pointer, nint_t count, nint_t stride) {
  using Tag = vec::ScalableTag<T, 0>;
  const nint_t lanes = vec::size(Tag{});
  const auto zero = vec::zeros(Tag{});
  for (nint_t offset = 0; offset < count; offset += lanes) {
    vec::store(
        Tag{}, pointer + offset * stride, zero,
        vec::opt::first(vec::details::sme::min_value(lanes, count - offset)),
        vec::opt::strided(stride));
  }
}

template <nint_t KPack, typename T, typename Transform,
          meta::ValueType Spatial, meta::ValueType K,
          meta::ValueType Panel>
VECOPS_NOINLINE void transform_packed_inplace(
    T* output, Spatial spatial_meta, K k_meta,
    Panel panel_meta,
    const Transform& transform) {
  static_assert(KPack == 1 || KPack == 2 || KPack == 4);
  const nint_t spatial = static_cast<nint_t>(spatial_meta);
  const nint_t k = static_cast<nint_t>(k_meta);
  const nint_t panel = static_cast<nint_t>(panel_meta);
  using Tag = vec::ScalableTag<T, 0>;
  const nint_t spatial_panels = ceil_div(spatial, panel);
  const nint_t k_groups = ceil_div(k, KPack);
  const nint_t elements = spatial_panels * k_groups * panel * KPack;
  const nint_t lanes = vec::size(Tag{});
  nint_t offset = 0;
  for (; offset + lanes <= elements; offset += lanes) {
    const auto value = vec::load(Tag{}, output + offset);
    vec::store(Tag{}, output + offset, transform(Tag{}, value));
  }
  if (offset < elements) {
    const nint_t active = elements - offset;
    const auto value = vec::load(
        Tag{}, output + offset,
        vec::opt::first(active), vec::opt::zero);
    vec::store(
        Tag{}, output + offset, transform(Tag{}, value),
        vec::opt::first(active));
  }

  for (nint_t sp = 0; sp < spatial_panels; ++sp) {
    const nint_t active_spatial = vec::details::sme::min_value(panel, spatial - sp * panel);
    if (active_spatial < panel) {
      for (nint_t kg = 0; kg < k_groups; ++kg) {
        auto* group = output + (sp * k_groups + kg) * panel * KPack;
        zero_contiguous(
            group + active_spatial * KPack,
            (panel - active_spatial) * KPack);
      }
    }
    const nint_t active_k_pack = k - (k_groups - 1) * KPack;
    if (k_groups > 0 && active_k_pack < KPack) {
      auto* group = output +
          (sp * k_groups + k_groups - 1) * panel * KPack;
      for (nint_t ki = active_k_pack; ki < KPack; ++ki) {
        zero_strided(group + ki, active_spatial, KPack);
      }
    }
  }
}

template <typename T,
          meta::ValueType Spatial, meta::ValueType K,
          meta::ValueType Panel>
VECOPS_NOINLINE void expand_fp16_packed_to_fp32(
    const float16_t* source, T* output,
    Spatial spatial_meta, K k_meta, Panel panel_meta) {
  static_assert(std::same_as<T, float32_t>);
  using SourcePairTag = vec::ScalableTag<float16_t, 1>;
  using SourcePanelTag = vec::Half<SourcePairTag>;
  using OutputPanelTag = vec::Rebind<T, SourcePanelTag>;
  const nint_t spatial = static_cast<nint_t>(spatial_meta);
  const nint_t k = static_cast<nint_t>(k_meta);
  const nint_t panel = static_cast<nint_t>(panel_meta);
  const nint_t spatial_panels = ceil_div(spatial, panel);
  const nint_t source_groups = ceil_div(k, nint_t{2});
  for (nint_t sp = spatial_panels; sp-- > 0;) {
    for (nint_t kg = source_groups; kg-- > 0;) {
      const auto packed = vec::load(
          SourcePairTag{},
          source + (sp * source_groups + kg) * panel * 2);
      const auto even = vec::even(SourcePairTag{}, packed);
      const auto odd = vec::odd(SourcePairTag{}, packed);
      const auto out0 = vec::convert(
          OutputPanelTag{}, SourcePanelTag{}, even);
      const nint_t k0 = kg * 2;
      if (k0 + 1 < k) {
        const auto out1 = vec::convert(
            OutputPanelTag{}, SourcePanelTag{}, odd);
        vec::store(
            OutputPanelTag{},
            output + (sp * k + k0 + 1) * panel, out1);
      }
      vec::store(
          OutputPanelTag{}, output + (sp * k + k0) * panel, out0);
    }
  }
}

} // namespace vecops::kernel::matmul_pack_details::sme

#endif // VECOPS_MATMUL_DETAILS_PACK_SME_TRANSFORM_PACK_H
