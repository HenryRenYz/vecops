"""
GDB pretty-printer for vecops::vec::x86::RegType<ElemType, N>

Makes GDB display SIMD register types (RegType / WrapperType) as arrays
of their logical element type, e.g.:

    RegType<int16_t, 16>  -->  [0, 1, -1, 42, ...]   (16 elements)
    RegType<float32_t, 8> -->  [1.0, 2.0, 3.5, ...]  (8 elements)

Supports all element types defined by TL_DEFINE_MMREG:
  bfloat16_t, float16_t, float32_t, float64_t,
  int8_t, uint8_t, int16_t, uint16_t,
  int32_t, uint32_t, int64_t, uint64_t

Usage:
  This module is auto-loaded by GDB via the <executable>-gdb.py mechanism.
  It is registered through CMake: see CMakeLists.txt add_multiarch_executable().
"""

import re
import struct
import gdb
import gdb.printing

# ---------------------------------------------------------------------------
# Half-float conversion helpers
# ---------------------------------------------------------------------------

def _fp16_bits_to_float(bits):
    """
    Convert IEEE 754 half-precision (float16) bits to Python float.

    Handles all cases: +0/-0, normalised, denormal, +inf/-inf, NaN.
    The input `bits` is a 16-bit unsigned integer.
    """
    sign = -1.0 if (bits & 0x8000) else 1.0
    exp = (bits >> 10) & 0x1F
    mant = bits & 0x3FF

    if exp == 0x1F:                     # Inf or NaN
        if mant == 0:
            return sign * float('inf')
        else:
            # NaN: preserve sign in the float representation
            return float('nan')
    elif exp == 0:                      # Zero or denormal
        if mant == 0:
            return 0.0 if sign > 0 else -0.0
        # Denormal: value = (-1)^s * 2^(-14) * 0.mant
        value = mant / 1024.0
        value *= 2.0 ** (-14)
        return sign * value
    else:                               # Normal
        # value = (-1)^s * 2^(exp-15) * 1.mant
        value = 1.0 + mant / 1024.0
        value *= 2.0 ** (exp - 15)
        return sign * value


def _bf16_bits_to_float(bits):
    """
    Convert bfloat16 bits to Python float.

    bfloat16 is the upper 16 bits of an IEEE 754 float32.  We shift left by
    16 and reinterpret via struct.  This inherently handles +0/-0,
    normalised/denormal, +inf/-inf, and NaN.
    """
    packed = struct.pack('<I', bits << 16)
    return struct.unpack('<f', packed)[0]


# ---------------------------------------------------------------------------
# Element type registry
# ---------------------------------------------------------------------------
# Map the C++ typedef / struct name (as it appears in the template parameter)
# to (gdb_type_name, size_in_bytes, float_kind)
#
#   float_kind is one of:
#       None       — integer  (format with int())
#       'float32'  — standard 32-bit float  (format with float())
#       'float64'  — standard 64-bit double (format with float())
#       'float16'  — IEEE 754 half-precision (format via _fp16_bits_to_float)
#       'bfloat16' — bfloat16 (format via _bf16_bits_to_float)
#
# These names must match what GDB prints in template arguments. For types
# that are namespace-qualified in the source (e.g. ct::int16_t), GDB may
# print either the short form or the fully-qualified form depending on
# typedef stripping.  We handle both.

_ELEM_TABLE = {
    # ---- C++ typedef / struct name ----   (GDB lookup name)       bytes  float_kind
    'bfloat16_t':                        ('unsigned short',           2,  'bfloat16'),
    'ct::bfloat16_t':                    ('unsigned short',           2,  'bfloat16'),
    'float16_t':                         ('unsigned short',           2,  'float16'),
    'ct::float16_t':                     ('unsigned short',           2,  'float16'),
    'vecops::BFloat16':                  ('unsigned short',           2,  'bfloat16'),
    'vecops::bfloat16_t':                ('unsigned short',           2,  'bfloat16'),
    'vecops::Float16':                   ('unsigned short',           2,  'float16'),
    'vecops::float16_t':                 ('unsigned short',           2,  'float16'),
    'float32_t':                         ('float',                    4,  'float32'),
    'ct::float32_t':                     ('float',                    4,  'float32'),
    'vecops::float32_t':                 ('float',                    4,  'float32'),
    'float64_t':                         ('double',                   8,  'float64'),
    'ct::float64_t':                     ('double',                   8,  'float64'),
    'vecops::float64_t':                 ('double',                   8,  'float64'),
    'int8_t':                            ('signed char',              1,  None),
    'ct::int8_t':                        ('signed char',              1,  None),
    'uint8_t':                           ('unsigned char',            1,  None),
    'ct::uint8_t':                       ('unsigned char',            1,  None),
    'int16_t':                           ('short',                    2,  None),
    'ct::int16_t':                       ('short',                    2,  None),
    'uint16_t':                          ('unsigned short',           2,  None),
    'ct::uint16_t':                      ('unsigned short',           2,  None),
    'int32_t':                           ('int',                      4,  None),
    'ct::int32_t':                       ('int',                      4,  None),
    'uint32_t':                          ('unsigned int',             4,  None),
    'ct::uint32_t':                      ('unsigned int',             4,  None),
    'int64_t':                           ('long long',                8,  None),
    'ct::int64_t':                       ('long long',                8,  None),
    'uint64_t':                          ('unsigned long long',       8,  None),
    'ct::uint64_t':                      ('unsigned long long',       8,  None),
}

# Inverse map: gdb_type_name -> (c_typedef_name, bytes, float_kind) for display
_GDB_TO_TYPEDEF = {
    'float':              ('float32_t',  4, 'float32'),
    'double':             ('float64_t',  8, 'float64'),
    'signed char':        ('int8_t',     1, None),
    'unsigned char':      ('uint8_t',    1, None),
    'short':              ('int16_t',    2, None),
    'unsigned short':     ('uint16_t',   2, None),
    'int':                ('int32_t',    4, None),
    'unsigned int':       ('uint32_t',   4, None),
    'long long':          ('int64_t',    8, None),
    'unsigned long long': ('uint64_t',   8, None),
}


# ---------------------------------------------------------------------------
# Printer
# ---------------------------------------------------------------------------
class RegTypePrinter:
    """
    Pretty-printer for vecops::vec::x86::RegType<ElemType, N>.

    The template parameter N is the number of logical elements.
    We extract the element type name from the template args and cast the
    underlying intrinsic vector ('v') to an array of that element type.
    """

    def __init__(self, val):
        self._val = val
        self._elem_gdb_type = None
        self._float_kind = None  # None=int, 'float32','float64','float16','bfloat16'
        self._nelems = 0
        self._tag_name = None   # e.g. "int16_t"
        self._init()

    # ------------------------------------------------------------------
    def _init(self):
        typ = self._val.type.strip_typedefs()
        elem_name = None

        # Preferred: GDB's template_argument API correctly handles nested types
        try:
            elem_name = str(typ.template_argument(0))
            self._nelems = int(typ.template_argument(1))
        except (gdb.error, RuntimeError):
            # Fallback: regex from tag string
            tag = typ.tag
            if not tag:
                return
            m = re.search(r'RegType\s*<\s*(.+?)\s*,\s*(\d+)\s*>', tag)
            if not m:
                return
            elem_name = m.group(1).strip()
            self._nelems = int(m.group(2))

        if elem_name is None:
            return

        # Resolve element type
        info = _ELEM_TABLE.get(elem_name)
        if info is None:
            # Try stripping 'ct::' / 'vecops::' if present (or adding them if not)
            for ns in ('ct::', 'vecops::'):
                if elem_name.startswith(ns):
                    info = _ELEM_TABLE.get(elem_name[len(ns):])
                else:
                    info = _ELEM_TABLE.get(ns + elem_name)
                if info is not None:
                    break

        if info is None:
            # Last resort: try gdb.lookup_type directly
            try:
                t = gdb.lookup_type(elem_name)
                self._elem_gdb_type = t
                self._float_kind = \
                    'float32' if t.code == gdb.TYPE_CODE_FLT and t.sizeof == 4 else \
                    'float64' if t.code == gdb.TYPE_CODE_FLT and t.sizeof == 8 else \
                    None
                self._tag_name = elem_name
            except gdb.error:
                pass
            return

        gdb_name, _, float_kind = info
        try:
            self._elem_gdb_type = gdb.lookup_type(gdb_name)
            self._float_kind = float_kind
            self._tag_name = gdb_name
        except gdb.error:
            pass

    # ------------------------------------------------------------------
    def _get_v(self):
        """Access the 'v' member (may be in base class WrapperType)."""
        try:
            return self._val['v']
        except gdb.error:
            pass
        # Fallback: walk base classes
        try:
            typ = self._val.type.strip_typedefs()
            for field in typ.fields():
                if field.is_base_class:
                    try:
                        return self._val.cast(field.type)['v']
                    except gdb.error:
                        continue
        except gdb.error:
            pass
        return None

    # ------------------------------------------------------------------
    def _cast_to_array(self, v_val):
        """Cast the intrinsic vector to an element-type array."""
        return v_val.cast(self._elem_gdb_type.array(self._nelems))

    # ------------------------------------------------------------------
    def _format_elem(self, elem_val):
        if self._float_kind in ('float32', 'float64'):
            return f'{float(elem_val):.6g}'
        elif self._float_kind == 'float16':
            return f'{_fp16_bits_to_float(int(elem_val)):.6g}'
        elif self._float_kind == 'bfloat16':
            return f'{_bf16_bits_to_float(int(elem_val)):.6g}'
        else:
            return str(int(elem_val))

    # ------------------------------------------------------------------
    def to_string(self):
        if self._elem_gdb_type is None or self._nelems == 0:
            return str(self._val)

        v = self._get_v()
        if v is None:
            return str(self._val)

        try:
            arr = self._cast_to_array(v)
            # Summary: show first few elements, truncate if too long
            if self._nelems <= 8:
                parts = [self._format_elem(arr[i]) for i in range(self._nelems)]
            else:
                parts = [self._format_elem(arr[i]) for i in range(6)]
                parts.append('...')
                parts.append(self._format_elem(arr[self._nelems - 1]))
            return '[' + ', '.join(parts) + ']'
        except Exception as e:
            return f'<RegType printer error: {e}>'

    # ------------------------------------------------------------------
    def children(self):
        if self._elem_gdb_type is None or self._nelems == 0:
            return

        v = self._get_v()
        if v is None:
            return

        try:
            arr = self._cast_to_array(v)
            for i in range(self._nelems):
                raw = arr[i]
                if self._float_kind == 'float16':
                    yield (f'[{i}]', gdb.Value(_fp16_bits_to_float(int(raw))))
                elif self._float_kind == 'bfloat16':
                    yield (f'[{i}]', gdb.Value(_bf16_bits_to_float(int(raw))))
                else:
                    yield (f'[{i}]', raw)
        except gdb.error:
            yield ('raw', v)

    # ------------------------------------------------------------------
    def display_hint(self):
        return 'array'


# ---------------------------------------------------------------------------
# ScalarArray Printer
# ---------------------------------------------------------------------------
class ScalarArrayPrinter:
    """
    Pretty-printer for vecops::vec::ScalarArray<T, N>.

    Flat case (T is a scalar element type):
        ScalarArray<float32_t, 8>  -->  [1.0, 2.0, 3.5, ...]

    Multi-word case (T is RegType / another ScalarArray):
        ScalarArray<RegType<float32_t, 8>, 2>
            -->  [[1.0, 2.0, ..., 8.0], [9.0, 10.0, ..., 16.0]]
        children() flattens all inner elements into one sequence.
    """

    def __init__(self, val):
        self._val = val
        typ = val.type.strip_typedefs()
        self._inner_type_name = None
        self._num_words = 0

        # Preferred: GDB's template_argument API correctly handles nested
        # templates (e.g. ScalarArray<RegType<T, 8>, 2> -> num_words=2).
        try:
            self._inner_type_name = str(typ.template_argument(0))
            self._num_words = int(typ.template_argument(1))
            return
        except (gdb.error, RuntimeError):
            pass

        # Fallback: regex (may misparse nested template arguments)
        tag = typ.tag
        if tag:
            m = re.search(r'ScalarArray\s*<\s*(.+?)\s*,\s*(\d+)\s*>', tag)
            if m:
                self._inner_type_name = m.group(1).strip()
                self._num_words = int(m.group(2))

    # ------------------------------------------------------------------
    def _get_elem(self, index):
        """
        Access the index-th element of the underlying std::array.
        GDB may support operator[] directly, or we fall back to _M_elems.
        """
        try:
            return self._val[index]
        except gdb.error:
            pass
        return self._val['_M_elems'][index]

    # ------------------------------------------------------------------
    @staticmethod
    def _get_element_children(elem_val):
        """Try to get sub-children from an inner element via its printer."""
        pp = gdb.default_visualizer(elem_val)
        if pp is not None:
            try:
                return list(pp.children())
            except Exception:
                pass
        return None

    # ------------------------------------------------------------------
    def to_string(self):
        if self._num_words == 0:
            return str(self._val)

        try:
            first_elem = self._get_elem(0)
        except Exception:
            return str(self._val)

        has_compound_inner = gdb.default_visualizer(first_elem) is not None

        if has_compound_inner:
            # Multi-word case: show per-word summaries
            word_summaries = []
            max_words = self._num_words if self._num_words <= 4 else 3

            for i in range(self._num_words):
                if i == max_words and self._num_words > 4:
                    word_summaries.append('...')
                if i >= max_words and i != self._num_words - 1:
                    continue
                try:
                    elem = self._get_elem(i)
                    inner_pp = gdb.default_visualizer(elem)
                    if inner_pp is not None:
                        word_summaries.append(inner_pp.to_string())
                    else:
                        word_summaries.append(str(elem))
                except Exception as e:
                    word_summaries.append(f'<{e}>')

            return '[' + ', '.join(word_summaries) + ']'
        else:
            # Flat scalar case: show element values with truncation
            if self._num_words <= 8:
                parts = [str(self._get_elem(i)) for i in range(self._num_words)]
            else:
                parts = [str(self._get_elem(i)) for i in range(6)]
                parts.append('...')
                parts.append(str(self._get_elem(self._num_words - 1)))
            return '[' + ', '.join(parts) + ']'

    # ------------------------------------------------------------------
    def children(self):
        if self._num_words == 0:
            return

        idx = 0
        for i in range(self._num_words):
            try:
                elem = self._get_elem(i)
            except gdb.error:
                continue

            inner_children = self._get_element_children(elem)
            if inner_children is not None:
                for _name, child_val in inner_children:
                    yield (f'[{idx}]', child_val)
                    idx += 1
            else:
                yield (f'[{idx}]', elem)
                idx += 1

    # ------------------------------------------------------------------
    def display_hint(self):
        return 'array'


# ---------------------------------------------------------------------------
# SVE Vec Printer  (ARM SVE: Vec<Tag<elem, -1, POW2>>)
# ---------------------------------------------------------------------------
class VecSVEPrinter:
    """
    Pretty-printer for vecops::vec::Vec<vecops::vec::Tag<ElemType, -1, POW2>>
    on ARM SVE.

    Single word (POW2 = 0, underlying type is svfloat32_t etc.):
        Vec<Tag<float32_t, -1, 0>>  -->  [1.0, 2.0, 3.5, ...]

    Multi-word (POW2 > 0, underlying type is svfloat32x2_t / x4_t etc.):
        Vec<Tag<float32_t, -1, 1>>  -->  [[...], [...]]
        Vec<Tag<float32_t, -1, 2>>  -->  [[...], [...], [...], [...]]
    """

    def __init__(self, val):
        self._val = val
        self._is_sve = False
        self._elem_type_name = None
        self._pow2 = 0
        self._num_words = 1
        self._elems_per_word = 0
        self._float_kind = None
        self._gdb_name = None
        self._parse_tag()
        if self._is_sve:
            self._detect_elems_per_word()

    # ------------------------------------------------------------------
    def _parse_tag(self):
        """Extract element type and POW2 from the SVE vector type."""
        typ = self._val.type
        tag = typ.tag

        if tag and 'Vec<' in tag:
            # Template alias preserved: Vec<Tag<T, N, POW2>>
            success = self._parse_from_template(typ, tag)
            if success:
                return

        # Template alias resolved: try to determine info from the resolved
        # type (GCC vector extension, ACLE type, etc.)
        self._parse_from_resolved()

    # ------------------------------------------------------------------
    def _parse_from_template(self, typ, tag):
        """Parse element info from Vec<Tag<T, N, POW2>> via template_argument."""
        try:
            tag_type = typ.template_argument(0)          # Tag<T, N, POW2>
            elem_type = tag_type.template_argument(0)    # Element type
            self._elem_type_name = str(elem_type)
            self._pow2 = int(tag_type.template_argument(2))
        except (gdb.error, RuntimeError):
            m = re.search(
                r'Vec<.*Tag<\s*(.+?)\s*,\s*.+?\s*,\s*([-\d]+)\s*>>', tag
            )
            if not m:
                return False
            self._elem_type_name = m.group(1).strip()
            self._pow2 = int(m.group(2))

        self._is_sve = True
        self._num_words = 1 << self._pow2 if self._pow2 >= 0 else 1
        self._lookup_elem()
        return True

    # ------------------------------------------------------------------
    def _parse_from_resolved(self):
        """Parse element info from resolved type (GCC vector extension etc.)."""
        resolved = self._val.type.strip_typedefs()
        resolved_str = str(resolved.unqualified())

        # Single-word: vector extension like "float __attribute__((vector_size(64)))"
        if 'vector_size' in resolved_str:
            # Extract element type from the vector's target type
            try:
                self._elem_type_name = str(resolved.target())
            except gdb.error:
                return
            self._pow2 = 0
            self._is_sve = True
            self._num_words = 1
            # Compute element count from total size
            try:
                # For GCC vector extensions, sizeof returns the actual size
                self._elems_per_word = resolved.sizeof
                self._elems_per_word //= resolved.target().sizeof
            except gdb.error:
                pass
            self._lookup_elem()
            return

        # ACLE types: svfloat32_t, __SVFloat32_t
        if resolved_str.startswith('sv') or resolved_str.startswith('__SV'):
            self._elem_type_name = str(resolved.target()) if hasattr(resolved, 'target') else ''
            self._pow2 = 0
            self._is_sve = True
            self._num_words = 1
            self._lookup_elem()
            return

    # ------------------------------------------------------------------
    def _lookup_elem(self):
        """Resolve the element type to GDB type info using _ELEM_TABLE."""
        if not self._elem_type_name:
            return

        info = _ELEM_TABLE.get(self._elem_type_name)
        if info is None:
            for ns in ('ct::', 'vecops::'):
                if self._elem_type_name.startswith(ns):
                    info = _ELEM_TABLE.get(self._elem_type_name[len(ns):])
                else:
                    info = _ELEM_TABLE.get(ns + self._elem_type_name)
                if info is not None:
                    break

        if info is not None:
            self._gdb_name, _, self._float_kind = info
        else:
            try:
                t = gdb.lookup_type(self._elem_type_name)
                if t.code == gdb.TYPE_CODE_FLT:
                    if t.sizeof == 4:
                        self._float_kind = 'float32'
                    elif t.sizeof == 8:
                        self._float_kind = 'float64'
                self._gdb_name = self._elem_type_name
            except gdb.error:
                pass

    # ------------------------------------------------------------------
    def _get_word(self, word_idx):
        """Return the word_idx-th SVE word value (cast to resolved type)."""
        resolved = self._val.cast(self._val.type.strip_typedefs())
        if self._pow2 == 0:
            return resolved
        return resolved['__val'][word_idx]

    # ------------------------------------------------------------------
    def _detect_elems_per_word(self):
        """Count elements in one SVE word by probing subscript access."""
        try:
            word = self._get_word(0)
        except gdb.error:
            return

        count = 0
        while True:
            try:
                _ = word[count]
                count += 1
            except gdb.error:
                break
        self._elems_per_word = count

    # ------------------------------------------------------------------
    def _format_elem(self, elem_val):
        if self._float_kind in ('float32', 'float64'):
            return f'{float(elem_val):.6g}'
        elif self._float_kind == 'float16':
            return f'{_fp16_bits_to_float(int(elem_val)):.6g}'
        elif self._float_kind == 'bfloat16':
            return f'{_bf16_bits_to_float(int(elem_val)):.6g}'
        else:
            return str(int(elem_val))

    # ------------------------------------------------------------------
    def to_string(self):
        if not self._is_sve:
            return str(self._val)

        if self._elems_per_word == 0:
            return f'<SVE: cannot detect vector length, raw={str(self._val)}>'

        if self._pow2 == 0:
            # Single word — same format as RegType
            n = self._elems_per_word
            if n <= 8:
                parts = [self._format_elem(self._val[i]) for i in range(n)]
            else:
                parts = [self._format_elem(self._val[i]) for i in range(6)]
                parts.append('...')
                parts.append(self._format_elem(self._val[n - 1]))
            return '[' + ', '.join(parts) + ']'

        # Multi-word — per-word groups
        word_summaries = []
        max_words = self._num_words if self._num_words <= 4 else 3
        for w in range(self._num_words):
            if w == max_words and self._num_words > 4:
                word_summaries.append('...')
            if w >= max_words and w != self._num_words - 1:
                continue
            try:
                word = self._get_word(w)
                n = self._elems_per_word
                if n <= 8:
                    w_parts = [self._format_elem(word[i]) for i in range(n)]
                else:
                    w_parts = [self._format_elem(word[i]) for i in range(6)]
                    w_parts.append('...')
                    w_parts.append(self._format_elem(word[n - 1]))
                word_summaries.append('[' + ', '.join(w_parts) + ']')
            except Exception as e:
                word_summaries.append(f'<{e}>')

        return '[' + ', '.join(word_summaries) + ']'

    # ------------------------------------------------------------------
    def children(self):
        if not self._is_sve or self._elems_per_word == 0:
            return

        idx = 0
        for w in range(self._num_words):
            try:
                word = self._get_word(w)
            except gdb.error:
                continue

            for i in range(self._elems_per_word):
                try:
                    raw = word[i]
                except gdb.error:
                    continue

                if self._float_kind == 'float16':
                    val = gdb.Value(_fp16_bits_to_float(int(raw)))
                elif self._float_kind == 'bfloat16':
                    val = gdb.Value(_bf16_bits_to_float(int(raw)))
                else:
                    val = raw

                yield (f'[{idx}]', val)
                idx += 1

    # ------------------------------------------------------------------
    @staticmethod
    def _matches(val):
        """Return True if val is an SVE vector (Vec alias or resolved type)."""
        tag = val.type.tag
        if tag:
            # Match Vec<Tag<...>> (preserved alias in debug info)
            if re.match(r'^vecops::vec::Vec<\s*vecops::vec::Tag<.*>\s*>$', tag):
                return True
        # Match resolved SVE types: ACLE (svfloat32_t), GDB internal
        # (__SVFloat32_t), or GCC vector extensions
        try:
            resolved_str = str(val.type.strip_typedefs().unqualified())
        except Exception:
            return False
        return ('vector_size' in resolved_str or
                resolved_str.startswith('__SV'))

    # ------------------------------------------------------------------
    def display_hint(self):
        return 'array'


# ---------------------------------------------------------------------------
# Registration helpers
# ---------------------------------------------------------------------------

def _vecsve_lookup(val):
    """Custom pretty-printer lookup for SVE Vec types.
    We cannot use RegexpCollectionPrettyPrinter because Clang may resolve
    the Vec alias to a GCC vector extension whose tag is None.
    """
    if VecSVEPrinter._matches(val):
        return VecSVEPrinter(val)
    return None
def register_vecops_printers(objfile=None):
    """Register all vecops x86 RegType pretty-printers."""
    pp = gdb.printing.RegexpCollectionPrettyPrinter('vecops')

    # Match vecops::vec::x86::RegType<anything, number>
    # The tag may or may not include namespace qualifiers on the element type
    pp.add_printer(
        'RegType',
        r'^vecops::vec::x86::RegType<.*,\s*\d+\s*>$',
        RegTypePrinter,
    )
    # Also match WrapperType directly (e.g. when viewed through base pointer)
    pp.add_printer(
        'WrapperType',
        r'^vecops::vec::x86::WrapperType<.*>$',
        RegTypePrinter,  # same printer — it just reads 'v'
    )
    # Match vecops::vec::ScalarArray (flat scalar arrays and multi-word)
    pp.add_printer(
        'ScalarArray',
        r'^vecops::vec::ScalarArray<.*>$',
        ScalarArrayPrinter,
    )
    # Custom matcher for ARM SVE vectors (tag may be None for Clang vector
    # extension types, so RegexpCollection cannot match them)
    gdb.printing.register_pretty_printer(
        objfile or gdb.current_objfile(), _vecsve_lookup
    )

    gdb.printing.register_pretty_printer(objfile or gdb.current_objfile(), pp)


# Auto-register when this module is imported by the <exe>-gdb.py loader
register_vecops_printers()
