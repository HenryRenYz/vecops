# vecops Header Organization Conventions

## The `vec` module three-layer layout

Headers under `include/vecops/vec/` intentionally share names across three
layers. Same-named files are **not duplicates**; each layer owns a distinct
responsibility:

| Layer | Responsibility |
|---|---|
| `vec/X.h` | Public API contract: Op functor declarations, doxygen semantics, the single backend assembly point (`#if` arch guards), and the `operator()` entry points that forward to `details::execute*`. |
| `vec/details/X.h` | Backend-independent infrastructure for the operation family: option legality predicates, option-pack-to-Request dispatch helpers, and multi-word `GenericImpl` fallbacks. Never includes backend headers. |
| `vec/details/{scalar,sve,x86}/X.h` | ISA specializations (`NativeImpl` / `NativeWordImpl`) for that backend tag, plus backend-internal helpers. May include `vec/details/X.h` to reuse its predicates and fallbacks. |

Backend selection is fully compile-time: arch macros guard the backend
includes, `details/Backend.h` maps them to the `CurrentBackend` tag alias, and
`details/Dispatch.h` resolves calls through a three-level `requires` probe
(`NativeImpl` → `NativeWordImpl` → `GenericImpl`).

`execution/`, `kernel/`, and `ops/` follow the same top-level-plus-`details/`
pattern; keep new modules consistent with it.

## Include-ordering constraint

Every backend or `GenericImpl` specialization must be declared **before** the
top-level `operator()` definitions that instantiate `details::execute()`.
Top-level operation headers therefore follow this sandwich shape:

```cpp
// 1. Op functor declarations (namespace vecops::vec)
// 2. inline constexpr op entry variables (add{}, blend{}, exp{}, ...)
// 3. #include "vecops/vec/details/Dispatch.h"
// 4. #include "vecops/vec/details/X.h"        // shared layer first
// 5. #include "vecops/vec/details/scalar/X.h" // then backends behind arch guards
// 6. operator() class-external definitions
```

The entry variables sit at the **end of the declaration half** (step 2), so
the details layer — shared code and backends — can call ops by their short
public names (`mul(tag, a, b)`, `blend(tag, inactive, mask, computed)`) with
no `details::execute(XxxOp{}, ...)` spelling. The `operator()` definitions
stay after the backend includes: dispatch specializations must be visible
before those definitions instantiate them.

Keep the shared `details/X.h` include **before** the backend includes: backend
headers may reuse its predicates and fallbacks (the Conversion family already
does), and `details/Arithmetic.h::synthesize_fma_word` relies on this order to
be visible to backend FMA implementations.

## Internal short-call convention

Inside the details layer, call ops exactly like external users — through the
public entry variables. Each Op/token struct carries the extra entries it
needs **inline in its own definition** (directly, or via its family's
`VECOPS_VEC_DECLARE_*` macro): a word-level overload and, where dispatchers
use it, a low-layer masked overload. The shared vocabulary they rely on lives
at the end of `vec/VecBase.h`: forward declarations of
`details::execute` / `details::execute_word` (defined in `Dispatch.h`), the
inactive-lane policy tags with `arithmetic_inactive_policy`, and
`multi_word_tag_v`.

- **Word arguments** (`NativeWordVec<Tag>` / `NativeWordMask<Tag>`) resolve to
  the word-level overload, which dispatches word 0 of `CurrentBackend`. These
  entries are reserved for **index-insensitive** elementwise ops; a word
  implementation may never depend on its word ordinal for them.
- **Single-word Tags** have `Vec<Tag> == NativeWordVec<Tag>`, so the word
  entries are constrained away and word-shaped calls fall through to the
  public whole-Tag overloads; `details::execute()`'s `word_count == 1` branch
  then routes into `NativeWordImpl<0>` — both spellings behave identically.
- **Low-layer masked entries** `(tag, values..., mask, inactive, Policy{})`
  exist as `operator()` overloads for dispatchers and backends; `Policy` must
  satisfy `arithmetic_inactive_policy` (VecBase.h).
- **The forward declarations are load-bearing**: the Op functors are declared
  in the declaration half, long before `Dispatch.h` arrives in the sandwich
  middle, and the inline entries call through those declarations. Every such
  call site is itself a template, so it instantiates after the definitions
  are visible and links against them.
- **Math tokens** additionally expose token-level variable templates in
  `vecops::vec::details` — `exp<Base, A, NegativeOnly>(tag, v)`,
  `log<Base, A>(tag, v)`, `rsqrt<A>(tag, v)`, `rcp<A>(tag, v)` — which shadow
  the public variables inside details and force an explicit accuracy tier.
- **`fill_word(tag, value)`** (in `details/Basic.h`) is the word twin of
  `fill`: fill's `(Tag, scalar)` shape cannot distinguish the representations
  by overload resolution. **`masked_unary_word`** (same header) is the shared
  sanitize→compute→blend skeleton for masked unary math word implementations.

What deliberately keeps the explicit forms:

- `Dispatch.h` itself (the probe mechanism) and the public `operator()`
  definitions (execute's first callers).
- `GenericImpl` word-batching loops: they iterate a generic `Backend` template
  parameter, not `CurrentBackend`.
- Generic-op forwarding (`execute(op, tag, ...)` where `op` is a function
  parameter — no token literal to remove) and `synthesize_fma_word`.
- Index-sensitive word calls (conversions, memory): spell
  `execute_word<Index, Backend>(op, tag, words...)`; both template arguments
  are always explicit (see `Dispatch.h`).

## The `math/` family split inside backend Math headers

`vec/details/Math.h` stays the backend-independent layer (option legality,
`GenericImpl` fallbacks, per-family option dispatch), but a backend whose
`Math.h` would host several operation families splits each family into
`vec/details/<backend>/math/<Family>.h` (e.g. `sve/math/Exp.h`,
`sve/math/Reciprocal.h`). The backend `Math.h` becomes a thin aggregator
that only includes those siblings, so the public include chain keeps one
entry point per backend. A family file must stay self-contained (include
what it uses) and carry a file-header comment stating its design; where a
design or its coefficients derive from an external source such as the Arm
optimized-routines project, cite the source files and license there — see
`sve/math/Exp.h` for the expected form.

## Boundary rules

- Code outside the `vec` module must not include `vecops/vec/details/**`
  paths. Shared option-detection traits live at the end of the public
  `vecops/vec/Options.h` (still inside `namespace vecops::vec::details`);
  include that header instead.
- Known exception: `tensor/DataAccess.h` references `vec::details::` symbols
  (Request resolvers, execute entry points, `memory_rebind_supported`) that
  are reachable through the `vec/Vec.h` umbrella. Treat those as a controlled
  collaboration surface; do not grow it without promoting the symbols to a
  public header first.

## Keeping documentation in sync

When changing code, update its documentation in the same change:

1. **In-code doxygen comments** — header doxygen blocks are the primary
   reference for this library (semantics, constraints, pitfalls). If a
   signature, option, guarantee, or special behavior changes, the comment
   next to it changes too.
2. **`docs/` module guides, if one covers the changed area.** Current
   coverage:

   | Document | Covers |
   |---|---|
   | `docs/Vec.md` | `vecops/vec/*` (concepts, operations, options, implementation, pitfalls) |
   | `docs/Meta.md` | `vecops/Meta.h` (Value hierarchy, constraint propagation, PackedStorage) |
   | `docs/Layout.md` | `vecops/tensor/Layout.h` (Shape/Strides/Layout, continuity, transforms) |
   | `docs/Tensor.md` | `vecops/tensor/{Tensor,DataAccess,AccessPolicy,AccessOptions,Transform,OptionalOperand}.h` |
   | `docs/ElementTypes.md` | the 12 element types, `util/{Float16,BFloat16,SmallFloat}.h`, small-float compute paths |
   | `docs/Runtime.md` | `runtime/{CallAbi,KernelAbi,OperatorBridgeAbi,Argument,Schema,KernelDefinition,Executable,Operator,Provider}.h` |
   | `docs/Compiler.md` | `compiler/Compiler.h` and generated source-kernel artifact behavior |
   | `docs/Python.md` | public `python/vecops` JIT, schema, cache, dtype, and framework-registration API |

   If a change alters a documented API, semantic, or pitfall, update the
   matching guide; if no guide exists for the area, none needs creating.
   Note that `docs/*` is currently git-ignored (`/docs/*` in `.gitignore`)
   but still the maintained documentation set — edit it regardless.

## Known accepted duplication

The arch-selection `#if` block (`ARCH_X86_FAMILY` / `CPU_CAPABILITY_SVE`) is
repeated in each top-level operation header and `VecBase.h`. Converging it
into a computed-include macro was considered and deferred: the sandwich layout
keeps the guards local and readable, and each backend file needs its own
guard anyway.

## Licensing of new files and third-party material

- Add `SPDX-FileCopyrightText` and `SPDX-License-Identifier` near the top of
  each new project-owned source, test, script, build file, and document,
  using its comment syntax. Project-owned contributions use MIT; use the
  actual copyright holder and years, not the last Git author automatically.
  For a new project-owned C++ file, for example:

  ```cpp
  // SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
  // SPDX-License-Identifier: MIT
  ```

- Preserve imported copyright, license, patent, attribution, and disclaimer
  notices. For copied code, ports, coefficients, tables, or data, record the
  source path/URL, full upstream commit, original notices, applicable license,
  and local changes in `THIRD_PARTY_NOTICES.md`; retain the required full texts
  in `LICENSES/`. A comparison reference is not proof of the original import
  revision. Stop for review if rights or provenance are unresolved.
- The root MIT license does not relicense third-party portions. Keep upstream
  SPDX identifiers and scope them accurately; do not apply a project MIT
  header to an imported file or invent an `OR` choice between unrelated rights.
  Keep the Huawei/MulanPSL-2.0 notice for KUPL-derived material.
- For formats that cannot contain comments, record an exact-path attribution
  in `THIRD_PARTY_NOTICES.md` or a dedicated `.license` sidecar. Preserve
  upstream license files unchanged; they do not need project SPDX headers.
- Review new-file SPDX fields, changed third-party mappings, full-license
  availability, and notices in distributed artifacts before accepting a
  change. A future automated check should examine added files with
  `git diff --diff-filter=A --name-only <base>...HEAD`, require both fields or
  an explicit exact-path exception, and verify each referenced license file.
  Existing unmarked files need individual review before any bulk annotation.
