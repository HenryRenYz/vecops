# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Streaming operands are created and consumed within one SME region."""

from ....dtypes import ELEMENT_BYTES
from ..capabilities import CPP_TYPES


class SMEInstructionEmitter:
  def emit_sme_instruction(self, statement):
    if statement.operation in {"sme.load_hor", "sme.load_ver", "sme.store_hor", "sme.store_ver"}:
      view, index = statement.view, statement.slice_index
      word_bytes = ELEMENT_BYTES[statement.operands[0]]
      unsigned = f"std::uint{word_bytes * 8}_t"
      tag, mask = self.unique("za_slice_tag"), self.unique("za_slice_mask")
      self.line(f"using {tag} = vecops::vec::ScalableTag<{unsigned}, 0>;")
      count = self.scalar(view.shape[0])
      if word_bytes != ELEMENT_BYTES[view.dtype]:
        count = f"(({count}) * {ELEMENT_BYTES[view.dtype]} / {word_bytes})"
      self.line(f"auto {mask} = vecops::vec::mwhilelt({tag}{{}}, 0, {count});")
      const = "const " if statement.operation.startswith("sme.load") else ""
      instruction = statement.operation.split(".")[1]
      self.line(
        f"vecops::vec::sme::{instruction}<{statement.tile}>(static_cast<std::uint32_t>({self.scalar(index)}), {mask}, reinterpret_cast<{const}{unsigned}*>({self.pointer(view)}));"
      )
    elif statement.operation in {"sme.load_bits", "sme.store_bits"}:
      handle, view = statement.vector, statement.view
      tag = handle.name + "_tag"
      memory_tag = self.unique("streaming_bits_tag")
      memory_type = f"std::uint{ELEMENT_BYTES[view.dtype] * 8}_t"
      self.line(f"using {memory_tag} = vecops::vec::ScalableTag<{memory_type}, 0>;")
      mask = self.unique("streaming_bits_mask")
      self.line(f"auto {mask} = vecops::vec::mwhilelt({memory_tag}{{}}, 0, {self.scalar(view.shape[0])});")
      if statement.operation == "sme.load_bits":
        self.line(f"using {tag} = vecops::vec::ScalableTag<{CPP_TYPES[handle.dtype]}, 0>;")
        self.line(f"auto {handle.name}_mask = vecops::vec::mwhilelt({tag}{{}}, 0, {self.scalar(handle.extent)});")
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
      self.line(f"auto {handle.name}_mask = vecops::vec::mwhilelt({tag}{{}}, 0, {self.scalar(handle.extent)});")
      self.line(
        f"auto {handle.name} = vecops::vec::load_convert({tag}{{}}, {self.pointer(statement.view)}, vecops::vec::opt::masked({handle.name}_mask));"
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
      self.line(f"auto {mask} = vecops::vec::mwhilelt({bits}{{}}, 0, {self.scalar(handle.extent)});")
      instruction = statement.operation.split(".")[1]
      slice_index = f"static_cast<std::uint32_t>({self.scalar(index)})"
      if instruction.startswith("read"):
        self.line(f"using {tag} = vecops::vec::ScalableTag<{CPP_TYPES[handle.dtype]}, 0>;")
        self.line(f"auto {handle.name}_mask = vecops::vec::mwhilelt({tag}{{}}, 0, {self.scalar(handle.extent)});")
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
