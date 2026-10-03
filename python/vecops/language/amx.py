"""AMX-specific compiled lexical ownership and tile memory primitives.

region(configuration) accepts up to eight (rows, column_bytes) descriptors.
Indices are constants. load/stream_load/store take a rank-two contiguous-row
view and use its byte row stride; bounds are checked before acquiring state.
There is no AMX transpose instruction on the current x86 target: transpose
lowering uses SIMD, not AMX tiles. T.dot owns its separate typed matrix
schedule/state scope; these expert memory regions do not implicitly admit dot.
"""

from .dsl import Intrinsic

region = Intrinsic("amx.region")
zero = Intrinsic("amx.zero")
load = Intrinsic("amx.load")
stream_load = Intrinsic("amx.stream_load")
store = Intrinsic("amx.store")
__all__ = ["region", "zero", "load", "stream_load", "store"]
