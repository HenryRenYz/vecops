"""SME-specific DSL state and instruction vocabulary.

`with T.sme.region():` is compiled lexical ownership, never a host Python
context manager. load(view) creates one region-local streaming vector;
mopa(tile, A, B) selects a typed outer-product instruction, store_tile(tile, C)
writes the clipped accumulator tile. T.dot owns a separately planned schedule.
Validation/alias handling is outside SM. Vectors cannot escape into Python;
16/8-bit inputs must already use the architectural grouped-K packing order.
"""

from .dsl import Intrinsic

region = Intrinsic("sme.region")
zero_za = Intrinsic("sme.zero_za")
streaming_lanes = Intrinsic("sme.streaming_lanes")
load = Intrinsic("sme.load")
mopa = Intrinsic("sme.mopa")
store_tile = Intrinsic("sme.store_tile")
__all__ = ["region", "zero_za", "streaming_lanes", "load", "mopa", "store_tile"]
