"""Tile permutation to explicit vector split/interleave/concat operations."""

from ..ir.transpose import PermutationOp, RegisterTranspose


def x86_dword_transpose():
  """AVX-512 16x16: two local unpack stages, two 128-bit lane stages.

  Keep the 512-bit words intact instead of recursively extracting XMM halves.
  This is an ISA alternative, not the portable/SVE permutation contract.
  """
  operations = []

  def op(kind, *inputs, width=0):
    operations.append(PermutationOp(kind, 16, inputs, width))
    return len(operations) - 1

  values = tuple(op("input") for _ in range(16))
  first = []
  for a, b in zip(values[::2], values[1::2]):
    first.extend((op("interleave_lower", a, b, width=4), op("interleave_upper", a, b, width=4)))
  values = []
  for start in range(0, 16, 4):
    for offset in range(2):
      a, b = first[start + offset], first[start + offset + 2]
      values.extend((op("interleave_lower", a, b, width=8), op("interleave_upper", a, b, width=8)))
  for span in (4, 8):
    output = [None] * 16
    for start in range(0, 16, span * 2):
      for offset in range(span):
        a, b = values[start + offset], values[start + offset + span]
        output[start + offset] = op("x86_shuffle_128_lower", a, b)
        output[start + offset + span] = op("x86_shuffle_128_upper", a, b)
    values = output
  return RegisterTranspose(16, 16, tuple(operations), tuple(values))


def register_transpose(rows, columns, element_bytes):
  """Build the butterfly before emission, including rectangular register tiles.

  The 128-bit base matches vec's local_interleave contract; larger tiles are
  split recursively. No backend operator, C++ helper or closure hides the DAG.
  """
  operations = []

  def op(kind, lanes, *inputs, width=0):
    operations.append(PermutationOp(kind, lanes, inputs, width))
    return len(operations) - 1

  def square(values, lanes):
    if lanes == 1:
      return values
    if lanes * element_bytes == 16:
      span = 1
      while span < lanes:
        output = [None] * lanes
        for block in range(0, lanes, span * 2):
          for j in range(span):
            a, b = values[block + j], values[block + span + j]
            output[block + 2 * j] = op("interleave_lower", lanes, a, b, width=element_bytes * span)
            output[block + 2 * j + 1] = op("interleave_upper", lanes, a, b, width=element_bytes * span)
        values, span = tuple(output), span * 2
      return values
    half = lanes // 2
    low = tuple(op("lower", half, v) for v in values)
    high = tuple(op("upper", half, v) for v in values)
    tl, bl = square(low[:half], half), square(low[half:], half)
    tr, br = square(high[:half], half), square(high[half:], half)
    return tuple(op("concat", lanes, a, b) for a, b in zip(tl, bl)) + tuple(
      op("concat", lanes, a, b) for a, b in zip(tr, br)
    )

  inputs = tuple(op("input", columns) for _ in range(rows))
  if columns == rows:
    outputs = square(inputs, rows)
  elif columns == rows * 2:
    low = square(tuple(op("lower", rows, v) for v in inputs), rows)
    high = square(tuple(op("upper", rows, v) for v in inputs), rows)
    outputs = low + high
  else:
    raise ValueError("register transpose requires square or double-width tiles")
  return RegisterTranspose(rows, columns, tuple(operations), outputs)
