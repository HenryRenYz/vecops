# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Terminal register instruction spelling, shared by transpose and packing.

Algorithms are staged T.func expression DAGs. This printer does not choose a
butterfly, loop order or tile size, and never represents values as Vec arrays.
"""

from .... import ir as I
from ....dtypes import ELEMENT_BYTES
from ..capabilities import CPP_TYPES


class RegistersEmitter:
  def register_instruction(self, value, tag, arguments, argument_tags, *, native_bits=False):
    if value.op in {"bitcast", "bitreinterpret"}:
      if native_bits:
        # SVE prefix placement can cross fixed/scalable descriptor domains.
        # Both physical values are one sizeless native word; reinterpret that
        # word, leaving the prepared logical tags to select the valid prefix.
        target = f"vecops::vec::ScalableTag<{CPP_TYPES[value.type.dtype]}, 0>"
        source = f"vecops::vec::ScalableTag<{CPP_TYPES[value.args[0].type.dtype]}, 0>"
        return f"vecops::vec::bitcast({target}{{}}, {source}{{}}, {arguments[0]})"
      return f"vecops::vec::bitcast({tag}{{}}, {argument_tags[0]}{{}}, {arguments[0]})"
    if value.op in {"lower", "upper"}:
      return f"vecops::vec::{value.op}({argument_tags[0]}{{}}, {arguments[0]})"
    if value.op == "block_shuffle":
      # One physical 512-bit dword word; larger logical vectors are legalized
      # before reaching this ISA spelling. No ISA leaks into the algorithm.
      immediates = {(0, 2, 4, 6): "0x88", (1, 3, 5, 7): "0xdd"}
      immediate = immediates.get(value.data)
      if (
        value.type.dtype == "uint32"
        and I.constant(value.type.shape[0]) == 16
        and immediate is not None
        and self.environment.codegen_target.architecture != "arm"
      ):
        return (
          f"vecops::vec::Vec<{tag}>{{_mm512_shuffle_i32x4({arguments[0]}.value, {arguments[1]}.value, {immediate})}}"
        )
      block = 16 // ELEMENT_BYTES[value.type.dtype]
      blocks = I.constant(value.type.shape[0]) // block
      name = self.unique("block_shuffle")
      self.line(f"auto {name}=vecops::vec::zeros({tag}{{}});")
      for ordinal, chosen in enumerate(value.data):
        for lane in range(block):
          source = f"vecops::vec::get({argument_tags[chosen // blocks]}{{}}, {arguments[chosen // blocks]}, {chosen % blocks * block + lane})"
          self.line(f"{name}=vecops::vec::set({tag}{{}}, {name}, {ordinal * block + lane}, {source});")
      return name
    return f"vecops::vec::{value.op}({tag}{{}}, {', '.join(arguments)})"

  def register_program(self, program, load, *, prefix="register", physical_tags=None):
    """Print a closed typed program using a caller-provided memory boundary."""
    names, tags = {}, {}

    def tag(value):
      own_dtype = value.type.dtype
      lanes = I.constant(value.type.shape[0])
      descriptor = value.type.tag if physical_tags is None else physical_tags[id(value)]
      key = own_dtype, lanes, descriptor.pow, descriptor.basis_dtype
      if key not in tags:
        own = self.unique(prefix + "_tag")
        if descriptor.N is None:
          base = f"vecops::vec::ScalableTag<{CPP_TYPES[descriptor.basis_dtype]}, {descriptor.pow}>"
          spelling = (
            base if own_dtype == descriptor.basis_dtype else f"vecops::vec::ViewAs<{CPP_TYPES[own_dtype]}, {base}>"
          )
        else:
          spelling = f"vecops::vec::FixedTag<{CPP_TYPES[own_dtype]}, {descriptor.N}>"
        self.line(f"using {own} = {spelling};")
        self.tag_dtypes[own] = own_dtype
        if lanes is not None:
          self.fixed_tag_lanes[own] = lanes
        tags[key] = own
      return tags[key]

    ordinals = {id(value): index for index, value in enumerate(program.inputs)}
    for value in program.operations:
      own = tag(value)
      if id(value) in ordinals:
        names[id(value)] = load(ordinals[id(value)], own)
      else:
        name = self.unique(prefix + "_vec")
        arguments = tuple(names[id(a)] for a in value.args)
        expression = self.register_instruction(
          value,
          own,
          arguments,
          tuple(tag(a) for a in value.args),
          native_bits=physical_tags is not None and self.environment.codegen_target.architecture == "arm",
        )
        self.line(f"auto {name} = {expression};")
        names[id(value)] = name
    return tag(program.outputs[0]), tuple(names[id(value)] for value in program.outputs)

  def register_store(self, statement):
    plan = self.prepared.register_programs.get(id(statement))
    if plan is None:
      return False
    program = plan.program
    previous_full, cache = getattr(self, "full_vector", False), self.chunk_cache
    self.full_vector, self.chunk_cache = True, {}
    try:
      tag, values = self.register_program(
        program,
        lambda index, own: self.vector(
          program.inputs[index],
          own,
          program.inputs[index].type.shape,
          ("0",),
          str(I.constant(program.inputs[index].type.shape[0])),
        ),
        physical_tags=plan.tags,
      )
      stride = statement.destination.strides[0]
      active = self.scalar(statement.destination.shape[0])
      extent = self.scalar(program.outputs[0].type.shape[0])
      self.line(
        f'if (({active}) < 0 || ({active}) > ({extent})) throw std::invalid_argument("invalid register store prefix");'
      )
      options = f", vecops::vec::opt::first({active})"
      if I.constant(stride) != 1:
        options += f", vecops::vec::opt::strided({self.scalar(stride)})"
      self.line(f"vecops::vec::store_convert({tag}{{}}, {self.pointer(statement.destination)}, {values[0]}{options});")
      return True
    finally:
      self.full_vector, self.chunk_cache = previous_full, cache
