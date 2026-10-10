# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Physical packed-ABI decoding; no matrix algorithm or packing traversal."""

from dataclasses import replace

from ....dtypes import ELEMENT_BYTES
from ..capabilities import CPP_TYPES

FORMATS = {"auto": -1, "dense": 0, "amx": 1, "sme": 2}


class PackedMemoryEmitter:
  def packed_offset(self, packed, s, k):
    shape = packed.shape
    if packed.origins:
      os, ok = packed.origins if packed.side == "A" else packed.origins[::-1]
      s, k = f"(({s})+({self.scalar(os)}))", f"(({k})+({self.scalar(ok)}))"
    spatial, extent_k = shape if packed.side == "A" else shape[::-1]
    return f"vecops_dsl::pack_offset<{CPP_TYPES[packed.view.dtype]}, {FORMATS[packed.format]}, {'true' if packed.side == 'B' else 'false'}>({s}, {k}, {self.scalar(spatial)}, {self.scalar(extent_k)})"

  def packed_vector(self, value, tag, root_shape, coords, active, axis, *, stream=True):
    packed = value.data
    if packed.format == "auto":
      result = self.unique("automatic_packed_vector")
      self.line(f"vecops::vec::Vec<{tag}> {result};")

      def assign(format_):
        own = replace(value, data=replace(packed, format=format_))
        self.retain_expression(own)
        vector = self.packed_vector(own, tag, root_shape, coords, active, axis)
        self.line(f"{result}={vector};")

      self.line("#if defined(HAS_AMX_TILE)")
      self.opening(f"if constexpr (vecops_dsl::packing_format<{CPP_TYPES[packed.view.dtype]},-1> == 1)")
      assign("amx")
      self.close()
      self.opening("else")
      assign("dense")
      self.close()
      self.line("#elif defined(HAS_SME)")
      assign("sme")
      self.line("#else")
      assign("dense")
      self.line("#endif")
      return result
    own = self.coordinates(value.type.shape, root_shape, coords)
    own_axis = axis - (len(root_shape) - len(value.type.shape))
    along_k = own_axis == (1 if packed.side == "A" else 0)
    s, k = own if packed.side == "A" else own[::-1]
    pointer = self.pointer(packed.view)
    offset = self.packed_offset(packed, s, k)
    native = self.tag_dtypes[tag] == packed.view.dtype
    if (
      packed.format == "dense"
      or stream
      and along_k
      and packed.side == "A"
      and packed.format == "amx"
      and ELEMENT_BYTES[self.tag_dtypes[tag]] >= ELEMENT_BYTES[packed.view.dtype]
    ):

      def contiguous():
        name = self.unique("packed_stream")
        stride = ""
        if not along_k:
          extent_k = packed.shape[1 if packed.side == "A" else 0]
          stride = f",vecops::vec::opt::strided({self.scalar(extent_k)})"
        self.line(
          f"auto {name}=vecops::vec::load_convert({tag}{{}},{pointer}+({offset}),vecops::vec::opt::first({active}){stride});"
        )
        return name

      if packed.format == "dense":
        return contiguous()
      # AMX A is contiguous only inside one physical K block. A virtual
      # reduction vector may cross that block, including a shortened panel.
      origin = packed.origins[1] if packed.origins else None
      physical_k = f"(({k})+({self.scalar(origin)}))" if origin is not None else k
      block = f"vecops_dsl::pack_kblock<{CPP_TYPES[packed.view.dtype]},1>()"
      return self.conditional_vector(
        value.type,
        tag,
        f"({active}) <= ({block}) - (({physical_k}) % ({block}))",
        contiguous,
        lambda: self.packed_vector(value, tag, root_shape, coords, active, axis, stream=False),
      )
    kp = max(1, 4 // ELEMENT_BYTES[packed.view.dtype])
    # A fixed K tail may contain less than one packed instruction group.
    # Reinterpreting one bf16 lane as uint32 (or 1-3 i8 lanes as uint32)
    # is not a valid ViewAs. Preserve the grouped stream for whole groups;
    # use the scalar packed-offset path below for incomplete fixed vectors.
    complete_groups = self.fixed_tag_lanes.get(tag, kp) % kp == 0
    if (
      stream
      and along_k
      and complete_groups
      and (packed.format == "sme" or packed.side == "B" and packed.format == "amx")
    ):
      input_tag = f"vecops::vec::Rebind<{CPP_TYPES[packed.view.dtype]},{tag}>"
      word_type = "std::uint64_t" if ELEMENT_BYTES[packed.view.dtype] == 8 else "std::uint32_t"
      bits = f"vecops::vec::ViewAs<{word_type},{input_tag}>"
      panel = f"vecops_dsl::pack_panel<{CPP_TYPES[packed.view.dtype]},{FORMATS[packed.format]}>()"

      def grouped():
        name = self.unique("packed_k_stream")
        self.line(
          f"auto {name}=vecops::vec::bitcast({input_tag}{{}},{bits}{{}},vecops::vec::load({bits}{{}},reinterpret_cast<const {word_type}*>({pointer}+({offset})),vecops::vec::opt::first((({active})+{kp}-1)/{kp}),vecops::vec::opt::strided({panel})));"
        )
        if not native:
          converted = self.unique("packed_stream_conversion")
          self.line(f"auto {converted}=vecops::vec::convert({tag}{{}},{input_tag}{{}},{name});")
          name = converted
        return self.zero_packed_tail(tag, name, active)

      origin = packed.origins[1 if packed.side == "A" else 0] if packed.origins else None
      physical_k = f"(({k})+({self.scalar(origin)}))" if origin is not None else k
      return self.conditional_vector(
        value.type,
        tag,
        f"(({physical_k}) % {kp}) == 0",
        grouped,
        lambda: self.packed_vector(value, tag, root_shape, coords, active, axis, stream=False),
      )
    name, lane = self.unique("packed_vector"), self.unique("packed_lane")
    self.line(f"auto {name} = vecops::vec::zeros({tag}{{}});")
    self.opening(f"for (vecops::nint_t {lane}=0; {lane} < {active}; ++{lane})")
    indices = [f"({c}) + {lane}" if i == own_axis else c for i, c in enumerate(own)]
    s, k = indices if packed.side == "A" else indices[::-1]
    pointer = self.pointer(packed.view)
    offset = self.packed_offset(packed, s, k)
    scalar = self.scalar_convert(self.tag_dtypes[tag], f"{pointer}[{offset}]", packed.view.dtype)
    self.line(f"{name} = vecops::vec::set({tag}{{}}, {name}, {lane}, {scalar});")
    self.close()
    return name

  def zero_packed_tail(self, tag, value, active):
    name = self.unique("matrix_tail_zero")
    self.line(
      f"auto {name}=vecops::vec::blend({tag}{{}},vecops::vec::zeros({tag}{{}}),vecops::vec::mwhilelt({tag}{{}},0,{active}),{value});"
    )
    return name
