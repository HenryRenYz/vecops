# Lambda Vec Adapter Design

## Overview

Add `PositionalLambdaVecAdapter<Eo, Ei, Fn>` and `ElementwiseLambdaVecAdapter<Eo, Ei, Fn>` to `include/vecops/gemm/Attachment.h`. These wrap a lambda/functor `Fn` as a `PositionedVecFn` or `ElementwiseVecFn` facade, analogous to `ConversionVecAdapter`.

Unlike `PositionedVecFn` which requires implementations for ALL POW2 levels in range, `Fn` may only implement a subset. The adapter automatically finds the nearest supported POW2 via upward (bitcast-and-truncate) or downward (split-and-concat) dispatch.

## Architecture

```
PositionedVecFn<Eo, Ei>
├── ElementwiseVecFn<Eo, Ei>
├── ConversionVecAdapter<Eo, Ei, VecFn>        (existing)
├── PositionalLambdaVecAdapter<Eo, Ei, Fn>     (NEW)
└── ElementwiseLambdaVecAdapter<Eo, Ei, Fn>    (NEW, inherits PositionedVecFn directly)
```

## Components

### 1. SFINAE Detection Traits (`namespace vecops::gemm::details`)

Two traits detect at compile-time whether `Fn` supports a given Tag type:

- `can_call_positional<To, Fn, Ei>` — checks validity of `Fn(To, Vec<Rebind<Ei, To>>, nint_t, nint_t)`
- `can_call_elementwise<To, Fn, Ei>` — checks validity of `Fn(To, Vec<Rebind<Ei, To>>)`

Both use `std::void_t` + `decltype` for SFINAE. Only check callability (valid expression), not exact return type.

### 2. `PositionalLambdaVecAdapter<Eo, Ei, Fn>`

**Inherits:** `PositionedVecFn<Eo, Ei>` (gets all range traits)

**Constructor:** Copy and move from `Fn`.

**Public:** `call(To, Vec<Rebind<Ei, To>>, nint_t x, nint_t y)`

**Private helpers:**

- `_try_upward<To, Ti, OrigPow2, TryPow2>` — linear scan upward (TryPow2 = OrigPow2+1, +2, ..., max_input_pow2). On match: bitcast input up, call Fn, bitcast result down. On exhaustion: delegate to `_try_downward`.
- `_try_downward<To, Ti, OrigPow2>` — recursive downward split. Tries direct call at current POW2 first, if fails splits input into halves, recurses on each half, concats results. `static_assert` at `min_input_pow2` boundary.

### 3. `ElementwiseLambdaVecAdapter<Eo, Ei, Fn>`

**Inherits:** `PositionedVecFn<Eo, Ei>` directly (NOT `ElementwiseVecFn`).

**Override:** `static constexpr bool is_elementwise = true`

**Public:**

- `call(To, Vec<Rebind<Ei, To>>)` — elementwise entry point
- `call(To, Vec<Rebind<Ei, To>>, nint_t x, nint_t y)` — discard (x,y), delegate to coord-free call

**Private helpers:** Same `_try_upward`/`_try_downward` structure but use `can_call_elementwise` and call `_fn` without coordinate params.

## Dispatch Algorithm

For a request at POW2 `p`:

```
1. DIRECT:  if can_call<ScalableTag<Eo,p>> → return _fn(tag, vec, ...)
2. UPWARD:  for k = 1,2,... while p+k ≤ max_input_pow2:
              if can_call<ScalableTag<Eo,p+k>>:
                bitcast in:  Vec<Rebind<Ei, To>> → Vec<Rebind<Ei, TryOut>>
                call Fn → Vec<TryOut>
                bitcast out: Vec<TryOut> → Vec<To>
                return
3. DOWNWARD: try_downward(To):
              if can_call<To> → return _fn(tag, vec, ...)
              if p > min_input_pow2:
                split: lower(To), upper(To)
                recurse try_downward on each half
                concat results
              else: static_assert
```

Key: `try_downward` does NOT re-attempt upward search. Each POW2 is checked exactly once.

## API / Public Interface

### `PositionalLambdaVecAdapter<Eo, Ei, Fn>`

```cpp
template <typename Eo, typename Ei, typename Fn>
struct PositionalLambdaVecAdapter : public PositionedVecFn<Eo, Ei> {
    PositionalLambdaVecAdapter(Fn&& fn);
    PositionalLambdaVecAdapter(const Fn& fn);

    // SFINAE-constrained call for all valid To
    template <TLV_DECL_TAG(To), ...>
    Vec<To> call(To t, Vec<Rebind<Ei, To>> v_in, nint_t x, nint_t y) const;
};
```

### `ElementwiseLambdaVecAdapter<Eo, Ei, Fn>`

```cpp
template <typename Eo, typename Ei, typename Fn>
struct ElementwiseLambdaVecAdapter : public PositionedVecFn<Eo, Ei> {
    static constexpr bool is_elementwise = true;

    ElementwiseLambdaVecAdapter(Fn&& fn);
    ElementwiseLambdaVecAdapter(const Fn& fn);

    template <TLV_DECL_TAG(To), ...>
    Vec<To> call(To t, Vec<Rebind<Ei, To>> v_in) const;

    template <TLV_DECL_TAG(To), ...>
    Vec<To> call(To t, Vec<Rebind<Ei, To>> v_in, nint_t x, nint_t y) const;
};
```

## Error Handling

- `static_assert` in `_try_downward` base case: "No supported POW2 found for this lambda"
- Existing `PositionedVecFn` static_assert: `Ei`, `Eo` must be valid element types

## Testing

- Test with lambda implementing only a subset of POW2 levels
- Test upward dispatch (lambda only supports larger POW2)
- Test downward dispatch (lambda only supports smaller POW2)
- Test direct dispatch (lambda supports exact POW2)
- Test boundary: lambda supports none → compile error
- Test elementwise version with and without coordinate-ignoring adapter

## Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Upward search | Linear scan p→max | Matches spec, avoids interleaving up/down |
| Downward recursion | Dedicated `_try_downward` | No O(n²) compile overhead from re-scanning upward |
| Elementwise base class | `PositionedVecFn` directly | Explicit control, no inherited ElementwiseVecFn adapter |
| SFINAE return check | Callability only | Less restrictive, lambda can return convertible type |
| Detection trait location | `namespace vecops::gemm::details` | Internal, not exposed to users |
| Construction | Copy + move from `Fn` | Follows `ConversionVecAdapter` pattern |
