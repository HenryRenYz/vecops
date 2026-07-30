#ifndef VECOPS_VEC_DETAILS_CONVERSION_MEMORY_H
#define VECOPS_VEC_DETAILS_CONVERSION_MEMORY_H

/**
 * @file ConversionMemory.h
 * @brief Memory-conversion infrastructure: options detection traits
 * (IsConversionMemoryLayoutOption etc.), options validation, and the
 * GenericImpl fallback for load_convert and store_convert operations.
 */

#include <type_traits>
#include <utility>

#include "vecops/vec/details/Options.h"

namespace vecops::vec::details {


/* **************************************************************************** */
//                  Option validation for memory conversions                  //
/* **************************************************************************** */

template <typename T>
struct IsConversionMemoryLayoutOption : std::bool_constant<
    IsOrderedOption<T>::value || IsUnorderedOption<T>::value> {};

template <typename T>
struct IsConversionMemoryValueOption : std::bool_constant<
    IsSaturateOption<T>::value || IsWrapOption<T>::value> {};

template <typename T>
struct IsConversionMemoryPackingOption : std::bool_constant<
    IsPackedOption<T>::value || IsSplitOption<T>::value> {};

template <VectorTag LogicalTag, Element Other, bool IsStore, typename Option>
inline constexpr bool is_memory_conversion_option_for = [] {
  using Clean = std::remove_cvref_t<Option>;
  using MemoryTag = Rebind<Other, LogicalTag>;
  if constexpr (
      IsConversionMemoryLayoutOption<Clean>::value ||
      IsConversionMemoryValueOption<Clean>::value ||
      IsMemoryAlignmentOption<Clean>::value ||
      IsMemoryTemporalityOption<Clean>::value ||
      IsConversionMemoryPackingOption<Clean>::value ||
      IsUnmaskedOption<Clean>::value ||
      IsFirstOption<Clean>::value || IsZeroOption<Clean>::value) {
    return true;
  } else if constexpr (IsMemoryAddressingOption<Clean>::value) {
    return is_memory_addressing_option_for<LogicalTag, Clean>;
  } else if constexpr (IsMaskedOption<Clean>::value) {
    using MaskType = typename IsMaskedOption<Clean>::Value;
    return std::same_as<MaskType, Mask<LogicalTag>> ||
        std::same_as<MaskType, Mask<MemoryTag>>;
  } else if constexpr (!IsStore && IsVectorMergeOption<Clean>::value) {
    return std::same_as<
        typename IsVectorMergeOption<Clean>::Value, Vec<LogicalTag>>;
  } else if constexpr (!IsStore && IsScalarMergeOption<Clean>::value) {
    return std::same_as<
        typename IsScalarMergeOption<Clean>::Value, ElementOf<LogicalTag>>;
  } else {
    return false;
  }
}();

template <VectorTag LogicalTag, Element Other, bool IsStore, typename... Options>
consteval bool valid_memory_conversion_options() {
  using Input = std::conditional_t<IsStore, ElementOf<LogicalTag>, Other>;
  using Output = std::conditional_t<IsStore, Other, ElementOf<LogicalTag>>;
  using MemoryTag = Rebind<Other, LogicalTag>;
  constexpr std::size_t layout_count =
      option_count<IsConversionMemoryLayoutOption, Options...>;
  constexpr std::size_t value_count =
      option_count<IsConversionMemoryValueOption, Options...>;
  constexpr std::size_t alignment_count =
      option_count<IsMemoryAlignmentOption, Options...>;
  constexpr std::size_t temporality_count =
      option_count<IsMemoryTemporalityOption, Options...>;
  constexpr std::size_t indexed_count =
      option_count<IsIndexedOption, Options...>;
  constexpr std::size_t strided_count =
      option_count<IsStridedOption, Options...>;
  constexpr std::size_t packing_count =
      option_count<IsConversionMemoryPackingOption, Options...>;
  constexpr std::size_t active_count =
      option_count<IsMemoryActiveOption, Options...>;
  constexpr std::size_t population_count =
      option_count<IsMemoryPopulationOption, Options...>;
  constexpr bool unordered = option_count<IsUnorderedOption, Options...> == 1;
  constexpr bool wraps = option_count<IsWrapOption, Options...> == 1;
  constexpr bool split = option_count<IsSplitOption, Options...> == 1;
  constexpr bool mask_type_matches = [] {
    bool valid = true;
    ([&]<typename Option>() {
      using Clean = std::remove_cvref_t<Option>;
      if constexpr (IsMaskedOption<Clean>::value) {
        using Actual = typename IsMaskedOption<Clean>::Value;
        using Expected = std::conditional_t<
            unordered, Mask<MemoryTag>, Mask<LogicalTag>>;
        valid = std::same_as<Actual, Expected>;
      }
    }.template operator()<Options>(), ...);
    return valid;
  }();

  if constexpr (!(is_memory_conversion_option_for<
                    LogicalTag, Other, IsStore, Options> && ...)) {
    return false;
  } else {
    const bool unique = layout_count <= 1 && value_count <= 1 &&
        alignment_count <= 1 && temporality_count <= 1 &&
        indexed_count <= 1 && strided_count <= 1 &&
        indexed_count + strided_count <= 1 &&
        ((indexed_count + strided_count == 0) || alignment_count == 0) &&
        packing_count <= 1 && active_count <= 1 && population_count <= 1;
    const bool valid_population = population_count == 0 ||
        (!IsStore && !unordered && active_count == 1);
    const bool valid_wrap = !wraps ||
        (std::integral<Input> && std::integral<Output> &&
         sizeof(Output) < sizeof(Input));
    const bool valid_packing = !split || (IsStore && !unordered && !wraps);
    return unique && mask_type_matches && valid_population &&
        valid_wrap && valid_packing;
  }
}

template <template <typename> typename Predicate, typename Default,
          typename... Options>
VECOPS_ALWAYS_INLINE constexpr decltype(auto) memory_conversion_option_or(
    Default&& default_value, Options&&... options) {
  if constexpr (option_count<Predicate, Options...> == 0)
    return std::forward<Default>(default_value);
  else
    return find_option<Predicate>(std::forward<Options>(options)...);
}

template <typename Backend, VectorTag ToTag>
struct GenericImpl<Backend, LoadConvertOp, ToTag> {
  template <Element From, typename Layout, typename ValuePolicy,
            typename Alignment, typename Temporality>
  static VECOPS_ALWAYS_INLINE Vec<ToTag> call(
      LoadConvertOp, ToTag to, const From* pointer, Layout layout,
      ValuePolicy value_policy, Alignment alignment,
      Temporality temporality) {
    using FromTag = Rebind<From, ToTag>;
    const auto loaded = execute_load_options(
        LoadOp{}, FromTag{}, pointer, alignment, temporality);
    return execute(
        ConvertOp{}, to, FromTag{}, loaded, layout, value_policy);
  }

  template <Element From, typename Layout, typename ValuePolicy,
            typename Alignment, typename Temporality>
  static VECOPS_ALWAYS_INLINE Vec<ToTag> call(
      LoadConvertOp, ToTag to, const From* pointer,
      Mask<Rebind<From, ToTag>> memory_mask, Layout layout,
      ValuePolicy value_policy, Alignment alignment,
      Temporality temporality) {
    using FromTag = Rebind<From, ToTag>;
    const auto loaded = execute_load_options(
        LoadOp{}, FromTag{}, pointer, opt::masked(memory_mask),
        opt::zero, alignment, temporality);
    return execute(
        ConvertOp{}, to, FromTag{}, loaded, layout, value_policy);
  }

  template <Element From, typename Layout, typename ValuePolicy,
            typename Alignment, typename Temporality>
  static VECOPS_ALWAYS_INLINE Vec<ToTag> call(
      LoadConvertOp op, ToTag to, const From* pointer,
      Mask<ToTag> output_mask, Vec<ToTag> inactive, Layout layout,
      ValuePolicy value_policy, Alignment alignment,
      Temporality temporality) {
    using FromTag = Rebind<From, ToTag>;
    const auto memory_mask = execute(
        ConvertOp{}, FromTag{}, to, output_mask);
    const auto converted = call(
        op, to, pointer, memory_mask, layout, value_policy,
        alignment, temporality);
    return execute(BlendOp{}, to, inactive, output_mask, converted);
  }
};

template <typename Backend, VectorTag FromTag>
struct GenericImpl<Backend, StoreConvertOp, FromTag> {
  template <Element To, typename Layout, typename ValuePolicy,
            typename Alignment, typename Temporality, typename Packing>
  static VECOPS_ALWAYS_INLINE void call(
      StoreConvertOp, FromTag from, To* pointer, Vec<FromTag> value,
      Layout layout, ValuePolicy value_policy, Alignment alignment,
      Temporality temporality, Packing) {
    using ToTag = Rebind<To, FromTag>;
    const auto converted = execute(
        ConvertOp{}, ToTag{}, from, value, layout, value_policy);
    execute_store_options(
        StoreOp{}, ToTag{}, pointer, converted, alignment, temporality);
  }

  template <Element To, MaskValue MaskValueType, typename Layout,
            typename ValuePolicy, typename Alignment, typename Temporality,
            typename Packing>
  static VECOPS_ALWAYS_INLINE void call(
      StoreConvertOp, FromTag from, To* pointer, Vec<FromTag> value,
      MaskValueType mask, Layout layout, ValuePolicy value_policy,
      Alignment alignment, Temporality temporality, Packing packing) {
    using ToTag = Rebind<To, FromTag>;
    const auto converted = execute(
        ConvertOp{}, ToTag{}, from, value, layout, value_policy);
    const auto memory_mask = [&] {
      if constexpr (std::same_as<
                        std::remove_cvref_t<MaskValueType>, Mask<ToTag>>)
        return mask;
      else
        return execute(ConvertOp{}, ToTag{}, from, mask);
    }();
    execute_store_options(
        StoreOp{}, ToTag{}, pointer, converted, opt::masked(memory_mask),
        alignment, temporality);
    (void)packing;
  }
};

template <VectorTag ToTag, Element From, typename... Options>
VECOPS_ALWAYS_INLINE Vec<ToTag> execute_load_convert_options(
    LoadConvertOp op, ToTag to, const From* pointer, Options&&... options) {
  constexpr std::size_t active_count =
      option_count<IsMemoryActiveOption, Options...>;
  constexpr std::size_t population_count =
      option_count<IsMemoryPopulationOption, Options...>;
  constexpr bool unordered = option_count<IsUnorderedOption, Options...> == 1;
  constexpr std::size_t addressing_count =
      option_count<IsMemoryAddressingOption, Options...>;
  using FromTag = Rebind<From, ToTag>;

  auto invoke = [&](auto layout, auto value_policy, auto alignment,
                    auto temporality) -> Vec<ToTag> {
    if constexpr (
        active_count == 0 ||
        option_count<IsUnmaskedOption, Options...> == 1) {
      return execute(
          op, to, pointer, layout, value_policy, alignment, temporality);
    } else if constexpr (unordered) {
      Mask<FromTag> mask;
      if constexpr (option_count<IsMaskedOption, Options...> == 1) {
        mask = find_option<IsMaskedOption>(
            std::forward<Options>(options)...).value;
      } else {
        const nint_t count = find_option<IsFirstOption>(
            std::forward<Options>(options)...).count;
        VECOPS_ASSERT(
            0 <= count && count <= size(FromTag{}),
            "load_convert count %zd !in 0..%zd", count, size(FromTag{}));
        mask = mwhilelt(FromTag{}, 0, count);
      }
      return execute(
          op, to, pointer, mask, layout, value_policy,
          alignment, temporality);
    } else {
      Mask<ToTag> mask;
      if constexpr (option_count<IsMaskedOption, Options...> == 1) {
        mask = find_option<IsMaskedOption>(
            std::forward<Options>(options)...).value;
      } else {
        const nint_t count = find_option<IsFirstOption>(
            std::forward<Options>(options)...).count;
        VECOPS_ASSERT(
            0 <= count && count <= size(to),
            "load_convert count %zd !in 0..%zd", count, size(to));
        mask = mwhilelt(to, 0, count);
      }
      Vec<ToTag> inactive;
      if constexpr (population_count == 0 ||
                    option_count<IsZeroOption, Options...> == 1) {
        inactive = zeros(to);
      } else if constexpr (
          option_count<IsVectorMergeOption, Options...> == 1) {
        inactive = find_option<IsVectorMergeOption>(
            std::forward<Options>(options)...).value;
      } else {
        inactive = fill(
            to, find_option<IsScalarMergeOption>(
                    std::forward<Options>(options)...).value);
      }
      return execute(
          op, to, pointer, mask, inactive, layout, value_policy,
          alignment, temporality);
    }
  };

  auto invoke_with_access = [&](auto access) -> Vec<ToTag> {
    return invoke(
        memory_conversion_option_or<IsConversionMemoryLayoutOption>(
            cvt::ordered, std::forward<Options>(options)...),
        memory_conversion_option_or<IsConversionMemoryValueOption>(
            cvt::saturate, std::forward<Options>(options)...),
        access,
        memory_conversion_option_or<IsMemoryTemporalityOption>(
            mem::temporal, std::forward<Options>(options)...));
  };
  if constexpr (addressing_count == 1) {
    const auto addressing = find_option<IsMemoryAddressingOption>(
        std::forward<Options>(options)...);
    if constexpr (IsStridedOption<
                      std::remove_cvref_t<decltype(addressing)>>::value) {
      auto indices = make_strided_indices<CurrentBackend>(
          to, static_cast<nint_t>(addressing.stride));
      return invoke_with_access(opt::indexed(indices));
    } else {
      return invoke_with_access(addressing);
    }
  } else {
    return invoke_with_access(
        memory_conversion_option_or<IsMemoryAlignmentOption>(
            mem::unaligned, std::forward<Options>(options)...));
  }
}

template <VectorTag FromTag, Element To, typename... Options>
VECOPS_ALWAYS_INLINE void execute_store_convert_options(
    StoreConvertOp op, FromTag from, To* pointer, Vec<FromTag> value,
    Options&&... options) {
  constexpr std::size_t active_count =
      option_count<IsMemoryActiveOption, Options...>;
  constexpr bool unordered = option_count<IsUnorderedOption, Options...> == 1;
  constexpr std::size_t addressing_count =
      option_count<IsMemoryAddressingOption, Options...>;
  using ToTag = Rebind<To, FromTag>;

  auto invoke = [&](auto layout, auto value_policy, auto alignment,
                    auto temporality, auto packing) {
    if constexpr (
        active_count == 0 ||
        option_count<IsUnmaskedOption, Options...> == 1) {
      execute(
          op, from, pointer, value, layout, value_policy,
          alignment, temporality, packing);
    } else if constexpr (unordered) {
      Mask<ToTag> mask;
      if constexpr (option_count<IsMaskedOption, Options...> == 1) {
        mask = find_option<IsMaskedOption>(
            std::forward<Options>(options)...).value;
      } else {
        const nint_t count = find_option<IsFirstOption>(
            std::forward<Options>(options)...).count;
        VECOPS_ASSERT(
            0 <= count && count <= size(ToTag{}),
            "store_convert count %zd !in 0..%zd", count, size(ToTag{}));
        mask = mwhilelt(ToTag{}, 0, count);
      }
      execute(
          op, from, pointer, value, mask, layout, value_policy,
          alignment, temporality, packing);
    } else {
      Mask<FromTag> mask;
      if constexpr (option_count<IsMaskedOption, Options...> == 1) {
        mask = find_option<IsMaskedOption>(
            std::forward<Options>(options)...).value;
      } else {
        const nint_t count = find_option<IsFirstOption>(
            std::forward<Options>(options)...).count;
        VECOPS_ASSERT(
            0 <= count && count <= size(from),
            "store_convert count %zd !in 0..%zd", count, size(from));
        mask = mwhilelt(from, 0, count);
      }
      execute(
          op, from, pointer, value, mask, layout, value_policy,
          alignment, temporality, packing);
    }
  };

  auto invoke_with_access = [&](auto access) {
    invoke(
        memory_conversion_option_or<IsConversionMemoryLayoutOption>(
            cvt::ordered, std::forward<Options>(options)...),
        memory_conversion_option_or<IsConversionMemoryValueOption>(
            cvt::saturate, std::forward<Options>(options)...),
        access,
        memory_conversion_option_or<IsMemoryTemporalityOption>(
            mem::temporal, std::forward<Options>(options)...),
        memory_conversion_option_or<IsConversionMemoryPackingOption>(
            mem::packed, std::forward<Options>(options)...));
  };
  if constexpr (addressing_count == 1) {
    const auto addressing = find_option<IsMemoryAddressingOption>(
        std::forward<Options>(options)...);
    if constexpr (IsStridedOption<
                      std::remove_cvref_t<decltype(addressing)>>::value) {
      auto indices = make_strided_indices<CurrentBackend>(
          from, static_cast<nint_t>(addressing.stride));
      invoke_with_access(opt::indexed(indices));
    } else {
      invoke_with_access(addressing);
    }
  } else {
    invoke_with_access(memory_conversion_option_or<IsMemoryAlignmentOption>(
        mem::unaligned, std::forward<Options>(options)...));
  }
}

} // namespace vecops::vec::details

#endif // VECOPS_VEC_DETAILS_CONVERSION_MEMORY_H
