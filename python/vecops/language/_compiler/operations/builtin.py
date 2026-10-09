# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Single source of truth for built-in IR semantics and ordinary intrinsics."""

import operator

from .registry import OpDef, register_operation
from .registry import Trait as T

ELEMENTWISE = T.ELEMENTWISE | T.CSE | T.TRANSPOSE_STREAMABLE


def define(names, family, traits=ELEMENTWISE, **kwargs):
  for name in names.split():
    own_traits = traits | T.HARDWARE_SCALAR if name in _HARDWARE_SCALARS else traits
    register_operation(OpDef(name, family, own_traits, **kwargs))


# Integer/boolean metadata operations that need no allocation, exceptions or
# ordinary vector state. Backends still own their forced-inline spelling;
# integer divide/remainder additionally require a positive constant divisor.
_HARDWARE_SCALARS = frozenset(
  (
    "add",
    "sub",
    "mul",
    "min",
    "max",
    "lt",
    "le",
    "gt",
    "ge",
    "eq",
    "ne",
    "and",
    "or",
    "not",
    "neg",
    "identity",
    "cast",
    "literal",
    "local",
    "frozen",
    "raw",
    "symbol",
    "streaming_lanes",
  )
)


_BINARY_FOLDS = {
  "add": operator.add,
  "sub": operator.sub,
  "mul": operator.mul,
  "div": operator.truediv,
  "floordiv": operator.floordiv,
  "mod": operator.mod,
  "lt": operator.lt,
  "le": operator.le,
  "gt": operator.gt,
  "ge": operator.ge,
  "eq": operator.eq,
  "ne": operator.ne,
  "and": lambda a, b: a and b,
  "or": lambda a, b: a or b,
  "min": min,
  "max": max,
  "bit_and": operator.and_,
  "bit_or": operator.or_,
  "bit_xor": operator.xor,
  "shl": operator.lshift,
  "shr": operator.rshift,
}
for name in (
  "add",
  "sub",
  "mul",
  "div",
  "floordiv",
  "mod",
  "ceildiv",
  "min",
  "max",
  "bit_and",
  "bit_or",
  "bit_xor",
  "shl",
  "shr",
):
  define(name, "arithmetic", arity=2, fold=_BINARY_FOLDS.get(name))
for name in ("lt", "le", "gt", "ge", "eq", "ne", "mask_xor"):
  define(name, "comparison", ELEMENTWISE | T.COMPARISON, arity=2, fold=_BINARY_FOLDS.get(name))
for name in ("and", "or"):
  define(name, "logical", ELEMENTWISE | T.COMPARISON | T.CONDITIONAL, arity=2, fold=_BINARY_FOLDS[name])
define("not", "logical", ELEMENTWISE | T.COMPARISON, arity=1)
define("neg identity bit_not", "arithmetic", arity=1)
define("fma", "arithmetic", arity=3)
define("where", "select", ELEMENTWISE | T.CONDITIONAL, arity=3)
define(
  "exp exp2 exp_neg exp2_neg log log2 rsqrt rcp",
  "math",
  ELEMENTWISE | T.UNARY_MATH | T.ACCURACY,
  arity=1,
  keywords=("accuracy",),
)
define("sqrt abs", "math", ELEMENTWISE | T.UNARY_MATH, arity=1)
# Cast interning remains conservative until dtype/tag/rounding contracts are
# represented completely. A memory read is not pure across writes/aliasing.
define("cast", "conversion", T.ELEMENTWISE | T.TRANSPOSE_STREAMABLE, arity=1, keywords=("dtype",))
define(
  "load",
  "memory",
  T.MEMORY_READ | T.RUNTIME_SHAPE | T.CAPTURE_SCALAR | T.UNSTABLE_EXTENT | T.TRANSPOSE_STREAMABLE,
  effects="read",
)
define("packed_load", "memory", T.VECTOR_BOUNDARY, effects="read")
define("local", "reference", T.REFERENCE | T.RUNTIME_SHAPE | T.CAPTURE_SCALAR | T.UNSTABLE_EXTENT)
define("frozen", "reference", T.REFERENCE | T.RUNTIME_SHAPE | T.UNSTABLE_EXTENT)
define("raw symbol", "reference", T.RUNTIME_SHAPE)
define("literal fill", "structural", ELEMENTWISE)
define("arange", "index", T.ELEMENTWISE | T.CSE)
define("packing_size broadcast_dim schedule_parameter", "shape", T.CSE)
define("lanes", "shape", T.CSE | T.POSITIVE_EXTENT)
define("streaming_lanes", "shape", T.CSE | T.POSITIVE_EXTENT)
define("vector_ref", "reference", T.REFERENCE | T.TRANSPOSE_STREAMABLE)
define("state_ref", "reference", T.STATE_REFERENCE | T.VECTOR_BOUNDARY, effects="read")
define("grid", "control", T.NONE, effects="state", keywords=("schedule", "chunk", "parallel"))
define("wordwise_math", "math", ELEMENTWISE)
for name in ("add", "min", "max"):
  define(
    "reduce_" + name,
    "reduction",
    T.SCALAR_REDUCTION | T.REDUCTION | T.RUNTIME_SHAPE | T.CAPTURE_SCALAR | T.UNSTABLE_EXTENT,
    arity=1,
    keywords=("axis", "unroll"),
    reduction=name,
  )
define("axis_reduce map_rows", "reduction", T.REDUCTION | T.VECTOR_BOUNDARY)
define("region_argument", "structural", T.NONE)
define("register_input", "structural", T.VECTOR_BOUNDARY)
define("dtype_name itemsize static_log2", "metadata", T.NONE)
define("dot", "matrix", T.VECTOR_BOUNDARY)
define(
  "gather",
  "indexed_memory",
  T.MEMORY_READ | T.CAPTURE_SCALAR | T.UNSTABLE_EXTENT,
  effects="read",
  keywords=("dtype", "mask", "other"),
)
define("scatter", "indexed_memory", T.NONE, effects="write", keywords=("mask", "unique"))
define("value_transform", "transform", T.VECTOR_BOUNDARY)
define("shuf local_shuf", "register", T.REGISTER_TRANSFORM, keywords=())
define(
  "bitcast lower upper lower_half upper_half concat interleave local_interleave_lower local_interleave_upper block_shuffle",
  "register_geometry",
  T.REGISTER_TRANSFORM | T.VECTOR_BOUNDARY,
)
define("element_at", "index", T.VECTOR_BOUNDARY | T.CAPTURE_SCALAR)
define("expand_dims squeeze", "transform", T.NONE, keywords=("axis",))
define("indices", "index", T.NONE, keywords=("axis",))
define("workspace", "allocation", T.NONE, keywords=("dtype",))
define("worker_capacity", "execution", T.CSE)
define("view_nonoverlap", "memory_proof", T.NONE)
define(
  "mask_all mask_any mask_none mask_count mask_first mask_last",
  "mask_query",
  T.MASK_REDUCTION | T.REDUCTION,
  keywords=("unroll",),
)
define("parallel", "local_control", T.NONE, keywords=("layout", "unroll"))
define("simd", "local_control", T.NONE, keywords=("vector", "unroll"))
define("reduce", "generic_reduction", T.NONE, keywords=("reducer", "axis", "unroll"))
define("scan", "generic_reduction", T.NONE, keywords=("reducer", "axis", "exclusive", "reverse"))
define("team", "team", T.NONE, keywords=("workers",))
define("worker_id worker_count barrier", "team", T.NONE)
define("team_reduce", "team", T.NONE, keywords=("op",))
define("shared", "allocation", T.NONE, keywords=("dtype",))
define(
  "atomic_add atomic_min atomic_max atomic_exchange atomic_compare_exchange",
  "atomic",
  T.NONE,
  effects="state",
  keywords=("order",),
)
define("prefetch", "pipeline", T.NONE, effects="read", keywords=("level", "intent", "temporal"))
define("pipelined", "pipeline", T.NONE, keywords=("stages", "distance", "unroll", "buffering"))
define("view_disjoint", "metadata", T.NONE)
define("double_buffer", "allocation", T.NONE, keywords=("dtype",))
define("amx.zero", "architecture", T.NONE, effects="state", keywords=("dtype",))
define("sme.load sme.load_bits sme.read_hor sme.read_ver", "architecture", T.NONE, effects="state", keywords=("dtype",))
define("sme.mask", "architecture", T.NONE, effects="none", keywords=("dtype",))
define("sme.store sme.store_bits sme.write_hor sme.write_ver sme.zero_za", "architecture", T.NONE, effects="state")
define(
  "sme.load_hor sme.load_ver sme.store_hor sme.store_ver",
  "architecture",
  T.NONE,
  effects="state",
  keywords=("word_dtype",),
)
