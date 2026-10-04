"""Architecture operand checks and region-scoped streaming vector handles."""

from .. import ir as I


class HardwareFrontend:
  def sme_operation(self, operation, args, node):
    region = id(self.hardware_regions[-1])
    if operation == "sme.load":
      if len(args) != 1 or not isinstance(args[0], I.View) or len(args[0].shape) != 1:
        self.error(node, "sme.load expects a rank-one view")
      if args[0].dtype not in {"float16", "bfloat16", "float32", "float64", "int8", "uint8"}:
        self.error(node, "unsupported SME streaming operand dtype")
      handle = I.SMEVector(self.unique("streaming_vector"), args[0], region)
      self.block.statements.append(I.ArchitectureCall(operation, view=args[0], operands=(handle,)))
      return handle
    if operation == "sme.mopa":
      if len(args) != 3 or any(not isinstance(v, I.SMEVector) or v.region != region for v in args[1:]):
        self.error(node, "sme.mopa expects a ZA tile index and two sme.load handles in this region")
      tile = I.constant(self.value(args[0], node))
      a, b = args[1:]
      da, db = a.view.dtype, b.view.dtype
      integer = da in {"int8", "uint8"} and db in {"int8", "uint8"}
      floating = da == db and da in {"float16", "bfloat16", "float32", "float64"}
      if not (integer or floating):
        self.error(node, "unsupported SME mopa operand dtype combination")
      dtype = "int32" if integer else "float64" if da == "float64" else "float32"
      limit = 8 if dtype == "float64" else 4
      if type(tile) is not int or not 0 <= tile < limit:
        self.error(node, "SME accumulator tile index out of range")
      self.block.statements.append(I.ArchitectureCall(operation, tile, operands=(dtype, a, b)))
      return None
    if operation == "sme.store_tile":
      if len(args) != 2 or not isinstance(args[1], I.View) or len(args[1].shape) != 2 or not args[1].writable:
        self.error(node, "sme.store_tile expects a ZA index and writable rank-two view")
      view = args[1]
      tile = I.constant(self.value(args[0], node))
      if (
        view.dtype not in {"float32", "int32", "float64"}
        or type(tile) is not int
        or not 0 <= tile < (8 if view.dtype == "float64" else 4)
      ):
        self.error(node, "invalid SME store accumulator type/index")
      self.block.statements.append(I.ArchitectureCall(operation, tile, view))
      return None
    return NotImplemented
