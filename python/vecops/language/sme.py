"""SME-specific DSL state and instruction vocabulary.

`with T.sme.region():` is compiled lexical ownership, never a host Python
context manager. Its expert body admits transpose_copy and zero_za only;
T.dot owns a separately planned typed mopa schedule and state scope.
Validation/alias handling is outside SM. Vectors cannot escape into Python.
"""

from .dsl import Intrinsic

region = Intrinsic("sme.region")
zero_za = Intrinsic("sme.zero_za")
streaming_lanes = Intrinsic("sme.streaming_lanes")
__all__ = ["region", "zero_za", "streaming_lanes"]
