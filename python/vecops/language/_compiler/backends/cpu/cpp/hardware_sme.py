"""Streaming operands are created and consumed within one SME region."""

from ....dtypes import ELEMENT_BYTES
from ..capabilities import CPP_TYPES


class SMEInstructionEmitter:
  def check_sme_instruction(self, statement):
    if statement.operation not in {"sme.load", "sme.store_tile"}:
      return
    view = statement.view
    lanes = f"vecops::vec::sme::streaming_lanes<{CPP_TYPES[view.dtype]}>()"
    guards = [f"({self.scalar(view.strides[-1])}) == 1"]
    guards += [f"({self.scalar(n)})>=0 && ({self.scalar(n)})<=({lanes})" for n in view.shape]
    self.line("#if defined(HAS_SME_FA64)")
    self.line(
      "if (!("
      + " && ".join(guards)
      + ')) throw std::invalid_argument("SME operand must fit one streaming word/tile and have unit column stride");'
    )
    if statement.operation == "sme.store_tile":
      self.line(
        f"vecops_dsl::check_writable<2>({{{', '.join(self.scalar(v) for v in view.shape)}}}, {{{', '.join(self.scalar(v) for v in view.strides)}}});"
      )
    self.line("#endif")

  def emit_sme_instruction(self, statement):
    if statement.operation == "sme.load":
      handle = statement.operands[0]
      tag = handle.name + "_tag"
      self.line(f"using {tag} = vecops::vec::ScalableTag<{CPP_TYPES[handle.view.dtype]}, 0>;")
      self.line(f"auto {handle.name}_mask = vecops::vec::mwhilelt({tag}{{}}, 0, {self.scalar(handle.view.shape[0])});")
      self.line(
        f"auto {handle.name} = vecops::vec::load_convert({tag}{{}}, {self.pointer(handle.view)}, vecops::vec::opt::masked({handle.name}_mask));"
      )
    elif statement.operation == "sme.mopa":
      dtype, a, b = statement.operands
      self.line(
        f"vecops::vec::sme::mopa<{statement.tile}, {CPP_TYPES[dtype]}, {CPP_TYPES[a.view.dtype]}, {CPP_TYPES[b.view.dtype]}>({a.name}_mask, {b.name}_mask, {a.name}, {b.name});"
      )
    elif statement.operation == "sme.store_tile":
      view = statement.view
      unsigned = "std::uint64_t" if ELEMENT_BYTES[view.dtype] == 8 else "std::uint32_t"
      row, tag, mask = self.unique("za_row"), self.unique("za_tag"), self.unique("za_mask")
      self.line(f"using {tag} = vecops::vec::ScalableTag<{unsigned}, 0>;")
      self.line(f"auto {mask} = vecops::vec::mwhilelt({tag}{{}}, 0, {self.scalar(view.shape[1])});")
      self.opening(f"for (vecops::nint_t {row}=0; {row}<({self.scalar(view.shape[0])}); ++{row})")
      self.line(
        f"vecops::vec::sme::store_hor<{statement.tile}>({row}, {mask}, reinterpret_cast<{unsigned}*>({self.pointer(view, (row, '0'))}));"
      )
      self.close()
    else:
      return False
    return True
