# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Streaming operands are created and consumed within one SME region."""

from .... import ir as I
from ....dtypes import ELEMENT_BYTES
from ..capabilities import CPP_TYPES


class SMEInstructionEmitter:
  def streaming_mask(self, tag, dtype, extent):
    """Use all lanes only for the captured or explicit streaming geometry.

    A CPU model, ordinary VL, or a larger logical view does not prove this.
    Partial/unknown prefixes keep the normal bounded mask operation.
    """
    native = self.environment.vector_state.sme_bytes
    if (
      native is not None
      and I.constant(extent) == native // ELEMENT_BYTES[dtype]
      or extent.op == "streaming_lanes"
      and ELEMENT_BYTES[extent.data] == ELEMENT_BYTES[dtype]
    ):
      return f"vecops::vec::mtrue({tag}{{}})"
    return f"vecops::vec::mwhilelt({tag}{{}}, 0, {self.scalar(extent)})"

  def emit_sme_instruction(self, statement):
    if statement.operation == "sme.mask":
      mask = statement.vector
      tag = mask.name + "_tag"
      unsigned = f"std::uint{ELEMENT_BYTES[mask.dtype] * 8}_t"
      self.line(f"using {tag} = vecops::vec::ScalableTag<{unsigned}, 0>;")
      self.line(f"const auto {mask.name} = {self.streaming_mask(tag, mask.dtype, mask.extent)};")
    elif statement.operation in {"sme.load_hor", "sme.load_ver", "sme.store_hor", "sme.store_ver"}:
      view, index = statement.view, statement.slice_index
      word_bytes = ELEMENT_BYTES[statement.operands[0]]
      unsigned = f"std::uint{word_bytes * 8}_t"
      tag, mask = self.unique("za_slice_tag"), self.unique("za_slice_mask")
      self.line(f"using {tag} = vecops::vec::ScalableTag<{unsigned}, 0>;")
      count = view.shape[0]
      if word_bytes != ELEMENT_BYTES[view.dtype]:
        count = I.binary(
          "floordiv", I.binary("mul", count, I.literal(ELEMENT_BYTES[view.dtype])), I.literal(word_bytes)
        )
      self.line(f"auto {mask} = {self.streaming_mask(tag, statement.operands[0], count)};")
      const = "const " if statement.operation.startswith("sme.load") else ""
      instruction = statement.operation.split(".")[1]
      pointer = self.pointer(view)
      if I.constant(view.shape[0]) == 0:
        pointer = self.storage_pointer(view.storage)
      elif I.constant(view.shape[0]) is None:
        pointer = f"({self.scalar(view.shape[0])}>0 ? {pointer} : {self.storage_pointer(view.storage)})"
      self.line(
        f"vecops::vec::sme::{instruction}<{statement.tile}>(static_cast<std::uint32_t>({self.scalar(index)}), {mask}, reinterpret_cast<{const}{unsigned}*>({pointer}));"
      )
    elif statement.operation in {"sme.load_bits", "sme.store_bits"}:
      handle, view = statement.vector, statement.view
      tag = handle.name + "_tag"
      memory_tag = self.unique("streaming_bits_tag")
      memory_type = f"std::uint{ELEMENT_BYTES[view.dtype] * 8}_t"
      self.line(f"using {memory_tag} = vecops::vec::ScalableTag<{memory_type}, 0>;")
      mask = self.unique("streaming_bits_mask")
      self.line(f"auto {mask} = {self.streaming_mask(memory_tag, view.dtype, view.shape[0])};")
      if statement.operation == "sme.load_bits":
        self.line(f"using {tag} = vecops::vec::ScalableTag<{CPP_TYPES[handle.dtype]}, 0>;")
        self.line(f"auto {handle.name}_mask = {self.streaming_mask(tag, handle.dtype, handle.extent)};")
        self.line(
          f"auto {handle.name} = vecops::vec::bitcast({tag}{{}}, {memory_tag}{{}}, "
          f"vecops::vec::load({memory_tag}{{}}, reinterpret_cast<const {memory_type}*>({self.pointer(view)}), "
          f"vecops::vec::opt::masked({mask})));"
        )
      else:
        self.line(
          f"vecops::vec::store({memory_tag}{{}}, reinterpret_cast<{memory_type}*>({self.pointer(view)}), "
          f"vecops::vec::bitcast({memory_tag}{{}}, {tag}{{}}, {handle.name}), vecops::vec::opt::masked({mask}));"
        )
    elif statement.operation == "sme.load":
      handle = statement.vector
      tag = handle.name + "_tag"
      self.line(f"using {tag} = vecops::vec::ScalableTag<{CPP_TYPES[handle.dtype]}, 0>;")
      self.line(f"auto {handle.name}_mask = {self.streaming_mask(tag, handle.dtype, handle.extent)};")
      stride = (
        f", vecops::vec::opt::strided({self.scalar(statement.view.strides[0])})"
        if statement.operands == (True,)
        else ""
      )
      pointer = self.pointer(statement.view)
      if I.constant(handle.extent) == 0:
        pointer = self.storage_pointer(statement.view.storage)
      elif I.constant(handle.extent) is None:
        pointer = f"({self.scalar(handle.extent)}>0 ? {pointer} : {self.storage_pointer(statement.view.storage)})"
      self.line(
        f"auto {handle.name} = vecops::vec::load_convert({tag}{{}}, {pointer}, vecops::vec::opt::masked({handle.name}_mask){stride});"
      )
    elif statement.operation == "sme.store":
      handle = statement.vector
      self.line(
        f"vecops::vec::store_convert({handle.name}_tag{{}}, {self.pointer(statement.view)}, {handle.name}, vecops::vec::opt::masked({handle.name}_mask));"
      )
    elif statement.operation in {"sme.read_hor", "sme.read_ver", "sme.write_hor", "sme.write_ver"}:
      handle, index = statement.vector, statement.slice_index
      tag, bits = handle.name + "_tag", self.unique("za_vector_bits")
      unsigned = f"std::uint{ELEMENT_BYTES[handle.dtype] * 8}_t"
      self.line(f"using {bits} = vecops::vec::ScalableTag<{unsigned}, 0>;")
      mask = self.unique("za_vector_mask")
      predicate = (
        statement.operands[0].name
        if statement.operands and isinstance(statement.operands[0], I.SMEMask)
        else self.streaming_mask(bits, handle.dtype, handle.extent)
      )
      self.line(f"auto {mask} = {predicate};")
      instruction = statement.operation.split(".")[1]
      slice_index = f"static_cast<std::uint32_t>({self.scalar(index)})"
      if instruction.startswith("read"):
        self.line(f"using {tag} = vecops::vec::ScalableTag<{CPP_TYPES[handle.dtype]}, 0>;")
        self.line(f"auto {handle.name}_mask = {mask};")
        self.line(
          f"auto {handle.name} = vecops::vec::bitcast({tag}{{}}, {bits}{{}}, vecops::vec::sme::{instruction}<{statement.tile}>({bits}{{}}, {slice_index}, {mask}));"
        )
      else:
        self.line(
          f"vecops::vec::sme::{instruction}<{statement.tile}>({slice_index}, {mask}, vecops::vec::bitcast({bits}{{}}, {tag}{{}}, {handle.name}));"
        )
    elif statement.operation == "sme.mopa":
      dtype, a, b = statement.operands
      self.line(
        f"vecops::vec::sme::mopa<{statement.tile}, {CPP_TYPES[dtype]}, {CPP_TYPES[a.dtype]}, {CPP_TYPES[b.dtype]}>({a.name}_mask, {b.name}_mask, {a.name}, {b.name});"
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
