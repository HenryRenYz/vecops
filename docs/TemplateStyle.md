# vecops Template Interface Style Guide

This guide governs the naming and usage of every template interface under
`include/vecops/`: concepts, boolean/value traits, producing-type
metafunctions, and vocabulary aliases. All new code must follow it. Test and
benchmark code may use local helpers freely, but any reference to a library
trait or concept must use the names defined here.

Each rule below states where a construct may appear, how it must be named,
and shows the accepted and rejected forms.

## R1 Concepts

- Name concepts as **CamelCase nouns** with no suffix. The suffix
  `XxxConcept` is forbidden.
- A concept expresses a *category of types* and appears in **constraint
  positions**: constrained template parameters and `requires` clauses.
- A concept name must not collide with an existing class template. When the
  natural name is taken, pick a domain name instead — e.g. the concept for
  input specs is `InputSpecLike`, because `InputSpec` is already a class
  template.

```cpp
// Definition: a concept is a single-line delegate to the trait (see R6).
template <typename T>
concept FloatingTag = VectorTag<T> && is_float_v<ElementOf<T>>;

// Use: constrained template parameter (preferred form, see R6).
template <FloatingTag Tag>
Vec<Tag> sqrt(Tag tag, Vec<Tag> value);

// Wrong: XxxConcept suffix.
template <typename T>
concept FloatingTagConcept = ...;          // rejected

// Wrong: concept naming that shadows a class template.
template <typename T>
concept InputSpec = ...;                    // rejected: InputSpec is a class
```

## R2 Boolean and value traits

- The public form is a **snake_case variable template with the `_v` suffix**,
  declared `inline constexpr`.
- Every boolean predicate and every value trait carries `_v`, following
  `std::tuple_size_v` / `std::extent_v`.
- Forbidden: bare predicates without `_v` (`is_shape`), the `V` suffix
  (`IsFloatV`), and uppercase letters inside snake_case names
  (`is_A_packed_layout`).

```cpp
// Accepted.
template <typename T>
inline constexpr bool is_float_v =
    std::is_same_v<T, float32_t> || std::is_same_v<T, float64_t> ||
    std::is_same_v<T, bfloat16_t> || std::is_same_v<T, float16_t>;

template <typename T>
inline constexpr nint_t fixed_lanes_v = ...;

// Rejected names: is_float, IsFloatV, is_Float_v
```

## R3 Trait implementation structs

- Structs that need pattern matching or partial specialization are the
  *implementation* of a trait: name them **CamelCase** and place them in a
  `details::` namespace.
- Their members are lowercase: `value` (following
  `std::integral_constant`) and `type` (following `std::enable_if`).
  Uppercase members `Value` / `Type` are forbidden, except inside the
  member-bundle traits of R8.
- The public layer exposes only the `_v` / `_t` form; the implementation
  struct itself never appears in a public header's interface.

```cpp
namespace details {
template <template <typename...> class Template, typename T>
struct IsSpecializationOf : std::false_type {};

template <template <typename...> class Template, typename... Args>
struct IsSpecializationOf<Template, Template<Args...>> : std::true_type {};
}  // namespace details

// Public form delegates to the struct.
template <template <typename...> class Template, typename T>
inline constexpr bool is_specialization_of_v =
    details::IsSpecializationOf<Template, std::remove_cvref_t<T>>::value;
```

## R4 Producing-type metafunctions

A metafunction whose name describes a *transformation, lookup, or selection*
that yields a type is named **snake_case with the `_t` suffix**. Existing
examples: `repeat_t`, `find_option_type_t`, `option_type_or_t`,
`slice_access_policy_t`, `suggested_factor_tag_t`, `to_value_t`,
`meta_element_t`, `stride_type_t`.

```cpp
template <int I, typename TLayout>
using stride_type_t = meta_element_t<
    I, typename std::remove_cvref_t<TLayout>::Strides>;

// Rejected: a CamelCase alias with a T suffix (VecToTagT-style spellings
// must not be introduced; `VecToTag` itself is a vocabulary alias, see R5).
```

## R5 Vocabulary type aliases — explicit allowlist

Whether an alias may keep a plain CamelCase name (no `_t`) is decided by its
*role*, not by its module: it must be a noun that recurs across modules or in
user code and whose meaning is a typed domain concept. The allowlist is:

- vec tag / representation vocabulary: `FixedTag` `ScalableTag` `ElementOf`
  `Rebind` `ViewAs` `Half` `Twice` `IndexElement` `IndexTag` `InferredTagOf`
  `Vec` `Mask` `NativeWordVec` `NativeWordMask` `VecToTag`
- Meta value vocabulary: `meta::Any`

Aliases naming a type without a computation role that are *not* on this list
take the `_t` form (R4). Adding an allowlist entry requires updating this
file in the same change.

## R5b Pre-configured instance aliases

An alias whose right-hand side is a **concrete instantiation** with no
selection semantics — a "type constant" — may be CamelCase without a suffix:
`GenericSoftmaxRecipe`, `DefaultAccessDefaults`, `Array`, `TraversalAxis`.
The test: `using X = SomeTemplate<concrete arguments>;` with no dependent
choice involved.

## R5c Exemptions

- Member aliases inside a class (`ElementType`, `TIn`, `Base`): part of the
  class interface, free naming.
- Function-local `using` declarations: free naming.
- Aliases internal to `details::` may be CamelCase (e.g. `SVEInteger`), but
  boolean judgments still use `_v` (R2).
- Probe concepts and aliases local to a test: allowed, still CamelCase.

## R6 Division of labor between concepts and traits

- **The `_v` trait is the single source of truth.** A concept is a
  constraint-facing wrapper whose definition must delegate to the trait in
  one line:

  ```cpp
  template <typename T>
  concept FloatingTag = VectorTag<T> && is_float_v<ElementOf<T>>;
  ```

- **Constraint positions** (constrained template parameters, `requires`
  clauses) express a type category with the concept. When a concept exists,
  writing the trait directly there is forbidden:

  ```cpp
  template <FloatingTag Tag> ...            // accepted
  template <VectorTag Tag>
    requires IsFloatV<ElementOf<Tag>>       // rejected: bypasses the concept
  ...                                        //         (and a banned name)
  ```

  Non-category conditions (counts, constant values, combinations of values)
  are written as plain boolean expressions in a `requires` clause.

- **Single-parameter, single-concept atoms are written as constrained
  template parameters `template<C T>`**, including inside macros. A doubled
  constraint merges:

  ```cpp
  template <VectorTag Tag>
    requires FloatingTag<Tag>               // rejected: redundant double
  Vec<Tag> sqrt(Tag, Vec<Tag>);              //         constraint

  template <FloatingTag Tag>                // accepted: concept
  Vec<Tag> sqrt(Tag, Vec<Tag>);              //         subsumption keeps
                                              //         the meaning
  ```

  Keep a `requires` clause only for: compound conditions (`&&` / `||` / `!`,
  spanning several parameters, mixing in value computations), atoms applied
  to a *transformation* of a parameter (`std::integral<std::decay_t<I>>`),
  and value-range conditions (`I < J`). Prefer a constrained parameter over
  a `static_assert` inside the body when validating template parameters —
  it diagnoses at the call site and participates in SFINAE.

- **Value positions** (`if constexpr`, `static_assert`, partial
  specialization conditions, metafunction composition) use the `_v` trait:

  ```cpp
  if constexpr (is_scalable_tag_v<Tag>) { ... }
  static_assert(is_layout_v<L>, "...");
  ```

- Not every judgment needs both forms. Bucket by where it appears:
  - only in value positions → provide only `_v` (e.g. the
    `is_*_option_for_v` family, dispatch atoms like `is_subword_v`);
  - in both → provide both, `_v` as the source of truth plus a one-line
    concept delegate (canonical pairs: `is_float_v`/`FloatingTag`,
    `is_layout_v`/`LayoutLike`, `is_array_meta_v`/`ArrayMetaLike`,
    `is_input_spec_v`/`InputSpecLike`);
  - a concept form is impossible for: non-boolean value templates
    (`option_count_v`, `fixed_lanes_v`), class-scope member predicates
    (concepts live at namespace scope only), and `constexpr` member
    functions taking runtime values (`conforms(nint_t)`).

- A `requires` guard on a value template's preconditions (e.g.
  `requires is_fixed_tag_v<Tag>` guarding `fixed_lanes_v`) may keep the
  `_v` spelling: it is an implementation guard, not an interface contract.

- When a concept's natural name collides with a type name, add the `Like`
  suffix (`LayoutLike`, `TransformContextLike`).

## R7 Choosing standard-library facilities

- **Where a std predicate has an equivalent concept, use the concept form —
  including in value positions:** `std::same_as`, `std::integral`,
  `std::signed_integral`, `std::convertible_to`, `std::floating_point`,
  `std::derived_from`. Constraint positions *must* use the concept form;
  spelling the `_v` trait there loses subsumption.

- Exceptions — keep the `_v` trait:
  - no concept counterpart exists: `std::is_const_v`,
    `std::is_trivially_*_v`, and similar;
  - **semantically not equivalent — do not swap blindly:** the traits
    `std::is_signed_v` / `std::is_unsigned_v` test the *signedness
    property* (`is_signed_v<float>` is true, `is_unsigned_v<bool>` is
    true), while the concepts `signed_integral` / `unsigned_integral`
    denote the sets *integral and signed / unsigned*;
  - `std::is_base_of_v` may remain in a `static_assert`; in constraint
    positions use `std::derived_from` (note they differ: `derived_from`
    additionally requires public inheritance — this project's hierarchies
    are all public).

  Project predicates keep their `_v` form in value positions (they combine
  into `(is_xxx_v<Opts> && ...)` fold arithmetic); std concepts are the
  terminal readability judgment and project `_v` the composition atom.

- **`std::enable_if_t` must not be introduced**; use `requires` clauses.

## R8 Member-bundle traits

A trait struct bundling several related facts (in the style of
`std::numeric_limits`) may use a CamelCase class with static members;
member names are lowercase. A single fact must not be turned into a bundle
— use R2/R3 forms for that.

```cpp
template <typename Backend, VectorTag Tag>
struct RepresentationTraits {      // member bundle: several facts together
  static constexpr nint_t word_count = ...;
  static constexpr nint_t word_lanes = ...;
  using RawVec = ...;
};
```

## Audit commands (keep regressions out)

```bash
# Banned forms: V suffix, T-suffix aliases, namespace-scope predicates
# without _v
grep -rnE "Is[A-Za-z0-9]+V\b" include/ | grep -v "//"
grep -rnE "^using [A-Za-z0-9]+T =" include/
grep -rnE "^(static |inline )?constexpr bool is_[a-z0-9_]+ =" include/ | grep -v "_v ="
# enable_if leftovers
grep -rn "enable_if" include/
# Uppercase Type / value members
grep -rnE "using Type =|static constexpr (bool|int|size_t) Value =" include/
```
