"""Single source of truth for built-in IR semantics and ordinary intrinsics."""

import operator

from .registry import OpDef, register_operation
from .registry import Trait as T

ELEMENTWISE = T.ELEMENTWISE | T.CSE | T.TRANSPOSE_STREAMABLE


def define(names, family, traits=ELEMENTWISE, **kwargs):
  for name in names.split():
    register_operation(OpDef(name, family, traits, **kwargs))


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
  "load", "memory", T.RUNTIME_SHAPE | T.CAPTURE_SCALAR | T.UNSTABLE_EXTENT | T.TRANSPOSE_STREAMABLE, effects="read"
)
define("packed_load", "memory", T.VECTOR_BOUNDARY, effects="read")
define("local", "reference", T.REFERENCE | T.RUNTIME_SHAPE | T.CAPTURE_SCALAR | T.UNSTABLE_EXTENT)
define("frozen", "reference", T.REFERENCE | T.RUNTIME_SHAPE | T.UNSTABLE_EXTENT)
define("raw symbol", "reference", T.RUNTIME_SHAPE)
define("literal fill", "structural", ELEMENTWISE)
define("arange", "index", T.ELEMENTWISE | T.CSE)
define("lanes streaming_lanes packing_size broadcast_dim", "shape", T.CSE)
define("vector_ref", "reference", T.REFERENCE | T.TRANSPOSE_STREAMABLE)
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
define("axis_reduce row_normalize", "reduction", T.REDUCTION | T.VECTOR_BOUNDARY)
define("dot", "matrix", T.VECTOR_BOUNDARY)
