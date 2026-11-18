"""PDF objects and the file writer: every byte of a sample is written here, so its structure is exactly what the sample
says (classic tables, cross-reference streams, object streams, hybrid files, incremental updates, damaged tables).

Objects are Python values: None, bool, int, float, Name, Ref, String, Raw (bytes written as they are), list (array),
dict (dictionary, keys are names without the slash, insertion order kept) and Stream (only as an indirect object).
"""
import hashlib

WHITESPACE = b"\x00\t\n\x0c\r "
DELIMITERS = b"()<>[]{}/%"


class Name:
    """A name object; the value is text, written as UTF-8 with #xx for every byte that needs it."""

    __slots__ = ("value",)

    def __init__(self, value):
        self.value = value

    def __eq__(self, other):
        return isinstance(other, Name) and other.value == self.value

    def __hash__(self):
        return hash(("Name", self.value))

    def __repr__(self):
        return "/" + self.value


class Ref:
    """An indirect reference."""

    __slots__ = ("num", "gen")

    def __init__(self, num, gen=0):
        self.num = num
        self.gen = gen

    def __eq__(self, other):
        return isinstance(other, Ref) and (other.num, other.gen) == (self.num, self.gen)

    def __hash__(self):
        return hash(("Ref", self.num, self.gen))

    def __repr__(self):
        return f"{self.num} {self.gen} R"


class String:
    """A string object: its bytes, written as a literal (escaped) or a hex string."""

    __slots__ = ("data", "hex")

    def __init__(self, data, hex=False):
        self.data = bytes(data)
        self.hex = hex


class Raw:
    """Bytes written exactly as given (unusual syntax: balanced parentheses, a line continuation, odd hex digits)."""

    __slots__ = ("data",)

    def __init__(self, data):
        self.data = bytes(data)


class Stream:
    """A stream: its dictionary (without /Length, added when written), the bytes as stored in the file (encoded) and the
    bytes its filters decode to. `length` forces the written /Length (an int, possibly wrong, or a Ref to an object
    holding it); `crypt` False keeps it unencrypted in an encrypted file (metadata with /EncryptMetadata false)."""

    __slots__ = ("dict", "data", "decoded", "length", "crypt", "note")

    def __init__(self, d, data, decoded=None, length=None, crypt=True, note=None):
        self.dict = d
        self.data = bytes(data)
        self.decoded = self.data if decoded is None else bytes(decoded)
        self.length = length
        self.crypt = crypt
        self.note = note


def N(value):
    return Name(value)


def S(text):
    """A text string in PDFDocEncoding when it can be, else UTF-16BE with its byte order mark."""
    from .encodings import pdfdoc_encode

    data = pdfdoc_encode(text)
    return String(data if data is not None else b"\xfe\xff" + text.encode("utf-16-be"))


def U16(text, hex=False):
    return String(b"\xfe\xff" + text.encode("utf-16-be"), hex=hex)


def B(data):
    return String(data)


def H(data):
    return String(data, hex=True)


def fmt_num(x):
    if isinstance(x, bool):
        raise TypeError("bool is not a number")
    if isinstance(x, int):
        return str(x)
    if x != x or x in (float("inf"), float("-inf")):
        raise ValueError("not a finite number")
    text = f"{x:.4f}".rstrip("0").rstrip(".")
    return "0" if text in ("-0", "") else text


def escape_name(value):
    out = bytearray()
    for b in value.encode("utf-8"):
        if b < 0x21 or b > 0x7E or b in DELIMITERS or b == 0x23:
            out += b"#%02X" % b
        else:
            out.append(b)
    return bytes(out)


def escape_literal(data):
    out = bytearray(b"(")
    for b in data:
        if b in (0x28, 0x29, 0x5C):
            out += b"\\" + bytes([b])
        elif b == 0x0A:
            out += b"\\n"
        elif b == 0x0D:
            out += b"\\r"
        elif b == 0x09:
            out += b"\\t"
        elif b == 0x08:
            out += b"\\b"
        elif b == 0x0C:
            out += b"\\f"
        elif b < 0x20 or b > 0x7E:
            out += b"\\%03o" % b
        else:
            out.append(b)
    out += b")"
    return bytes(out)


def ser(v, crypt=None):
    """The bytes of an object; `crypt` encrypts each string (the object's own key)."""
    if v is None:
        return b"null"
    if v is True:
        return b"true"
    if v is False:
        return b"false"
    if isinstance(v, (int, float)):
        return fmt_num(v).encode("ascii")
    if isinstance(v, Name):
        return b"/" + escape_name(v.value)
    if isinstance(v, Ref):
        return b"%d %d R" % (v.num, v.gen)
    if isinstance(v, String):
        data = crypt(v.data) if crypt is not None else v.data
        if v.hex:
            return b"<" + data.hex().upper().encode("ascii") + b">"
        return escape_literal(data)
    if isinstance(v, Raw):
        return v.data
    if isinstance(v, (list, tuple)):
        return b"[" + b" ".join(ser(x, crypt) for x in v) + b"]"
    if isinstance(v, dict):
        out = bytearray(b"<<")
        for key, value in v.items():
            out += b"/" + escape_name(key) + b" " + ser(value, crypt)
        out += b">>"
        return bytes(out)
    raise TypeError(f"not a PDF object: {type(v).__name__}")


def sha256(data):
    return hashlib.sha256(data).hexdigest()


class PdfFile:
    """Objects by number, the trailer's entries and the writer for every cross-reference kind."""

    def __init__(self, version="1.7"):
        self.version = version
        self.objects = {}  # num -> [gen, obj]
        self.next_num = 1
        self.trailer = {}

    def alloc(self):
        ref = Ref(self.next_num)
        self.next_num += 1
        return ref

    def add(self, obj, gen=0):
        ref = Ref(self.next_num, gen)
        self.next_num += 1
        self.objects[ref.num] = [gen, obj]
        return ref

    def put(self, ref, obj):
        self.objects[ref.num] = [ref.gen, obj]
        if ref.num >= self.next_num:
            self.next_num = ref.num + 1

    def get(self, ref):
        return self.objects[ref.num][1]


class Writer:
    """Writes revisions into one buffer. Offsets in the cross-reference count from the start of `buf`; bytes put in
    front of it afterwards (junk before the header) leave them counting from the header."""

    def __init__(self, pdf, security=None, binary_comment=True, eol=b"\n"):
        self.pdf = pdf
        self.security = security
        self.eol = eol
        self.buf = bytearray(b"%PDF-" + pdf.version.encode("ascii") + eol)
        if binary_comment:
            self.buf += b"%\xe2\xe3\xcf\xd3" + eol
        self.entries = {}  # num -> (type, field2, field3) as the newest revision gives it
        self.streams = []  # what each written stream holds: object, filters, lengths, decoded hash
        self.last_xref = None
        self.revisions = []
        self.no_crypt = set()  # object numbers never encrypted (the encryption dictionary)

    # -- objects -----------------------------------------------------------------------------------------------------

    def _crypt_for(self, num, gen, stream=False):
        if self.security is None or num in self.no_crypt:
            return None
        return self.security.string_crypt(num, gen) if not stream else self.security.stream_crypt(num, gen)

    def _object_bytes(self, num, gen, obj, in_objstm=False):
        if isinstance(obj, Stream):
            data = obj.data
            crypt = None if not obj.crypt else self._crypt_for(num, gen, stream=True)
            if crypt is not None:
                data = crypt(data)
            d = dict(obj.dict)
            if obj.length is None:
                d["Length"] = len(data)
            else:
                d["Length"] = obj.length
            head = ser(d, None if in_objstm else self._crypt_for(num, gen))
            self.streams.append(stream_record(num, gen, obj, len(data), len(self.revisions)))
            return head + b"\nstream\n" + data + b"\nendstream"
        return ser(obj, None if in_objstm else self._crypt_for(num, gen))

    def write_object(self, num, gen, obj):
        offset = len(self.buf)
        self.buf += b"%d %d obj\n" % (num, gen) + self._object_bytes(num, gen, obj) + b"\nendobj\n"
        self.entries[num] = (1, offset, gen)
        return offset

    def write_objstm(self, nums, objstm_num, extends=None):
        """An object stream holding `nums` (each generation 0, no streams); written as object `objstm_num`."""
        from .filters import flate

        bodies = [ser(self.pdf.objects[n][1]) for n in nums]
        header, body, pos = [], bytearray(), 0
        for n, b in zip(nums, bodies):
            header.append(b"%d %d" % (n, pos))
            body += b + b"\n"
            pos = len(body)
        head = b" ".join(header) + b"\n"
        decoded = head + bytes(body)
        d = {"Type": N("ObjStm"), "N": len(nums), "First": len(head), "Filter": N("FlateDecode")}
        if extends is not None:
            d["Extends"] = extends
        stream = Stream(d, flate(decoded), decoded)
        self.write_object(objstm_num, 0, stream)
        for i, n in enumerate(nums):
            self.entries[n] = (2, objstm_num, i)
        return stream

    # -- cross-reference sections -------------------------------------------------------------------------------------

    def xref_table(self, entries, trailer, eol=b"\r\n"):
        """A classic table of `entries` (num -> (type, f2, f3); type 0 free, 1 in use), in contiguous subsections."""
        offset = len(self.buf)
        out = bytearray(b"xref\n")
        nums = sorted(entries)
        i = 0
        while i < len(nums):
            j = i
            while j + 1 < len(nums) and nums[j + 1] == nums[j] + 1:
                j += 1
            out += b"%d %d\n" % (nums[i], j - i + 1)
            for n in nums[i:j + 1]:
                t, f2, f3 = entries[n]
                out += b"%010d %05d %s" % (f2, f3, b"n" if t == 1 else b"f") + (eol if len(eol) == 2 else b" " + eol)
            i = j + 1
        out += b"trailer\n" + ser(trailer) + b"\nstartxref\n%d\n%%%%EOF\n" % offset
        self.buf += out
        self.last_xref = offset
        return offset

    def xref_stream(self, entries, trailer, num, predictor=True, widths=None, index=None):
        """A cross-reference stream (object `num`, listing itself) of `entries` (type 0, 1 or 2)."""
        from .filters import flate, png_predict

        offset = len(self.buf)
        entries = dict(entries)
        entries[num] = (1, offset, 0)
        nums = sorted(entries)
        if widths is None:
            big = max([f2 for t, f2, f3 in entries.values()] + [offset + 4096])
            w1 = 1 if big < 1 << 8 else 2 if big < 1 << 16 else 3 if big < 1 << 24 else 4
            w2 = 1 if max(f3 for t, f2, f3 in entries.values()) < 256 else 2
            widths = [1, w1, w2]
        rows = bytearray()
        subsections = []
        i = 0
        while i < len(nums):
            j = i
            while j + 1 < len(nums) and nums[j + 1] == nums[j] + 1:
                j += 1
            subsections += [nums[i], j - i + 1]
            for n in nums[i:j + 1]:
                t, f2, f3 = entries[n]
                rows += t.to_bytes(widths[0], "big") + f2.to_bytes(widths[1], "big") + f3.to_bytes(widths[2], "big")
            i = j + 1
        d = {"Type": N("XRef"), "Size": trailer["Size"]}
        if index is not False and (subsections != [0, trailer["Size"]] or index):
            d["Index"] = subsections
        d["W"] = widths
        for key in ("Root", "Info", "ID", "Encrypt", "Prev"):
            if key in trailer:
                d[key] = trailer[key]
        if predictor:
            columns = sum(widths)
            encoded = flate(png_predict(bytes(rows), 1, 8, columns, pattern=(2,)))
            d["Filter"] = N("FlateDecode")
            d["DecodeParms"] = {"Predictor": 12, "Columns": columns}
        else:
            encoded = flate(bytes(rows))
            d["Filter"] = N("FlateDecode")
        stream = Stream(d, encoded, bytes(rows), crypt=False)
        self.buf += b"%d 0 obj\n" % num
        head = ser({**d, "Length": len(encoded)})
        self.buf += head + b"\nstream\n" + encoded + b"\nendstream\nendobj\n"
        self.streams.append(stream_record(num, 0, stream, len(encoded), len(self.revisions)))
        self.buf += b"startxref\n%d\n%%%%EOF\n" % offset
        self.entries[num] = (1, offset, 0)
        self.last_xref = offset
        return offset

    # -- whole revisions ---------------------------------------------------------------------------------------------

    def revision(self, nums, kind="table", trailer=None, objstm=None, free=None, prev=None, objstm_size=50,
                 xref_num=None, predictor=True, table_eol=b"\r\n", first=False):
        """Writes the objects `nums` and one cross-reference section of `kind`: "table", "stream", "objstm" (object
        streams and a cross-reference stream) or "hybrid" (a table for the plain objects, the object streams listed only
        in the stream that /XRefStm points to). `free` maps deleted numbers to their next generation."""
        trailer = dict(trailer or {})
        if prev is not None:
            trailer["Prev"] = prev
        section = {}
        if first:
            section[0] = (0, 0, 65535)
        compressible = []
        if kind in ("objstm", "hybrid"):
            for n in nums:
                gen, obj = self.pdf.objects[n]
                if gen == 0 and not isinstance(obj, Stream) and n not in self.no_crypt and n not in (objstm or {}).get("exclude", ()):
                    compressible.append(n)
        plain = [n for n in nums if n not in compressible]
        for n in plain:
            gen, obj = self.pdf.objects[n]
            self.write_object(n, gen, obj)
            section[n] = self.entries[n]
        compressed = {}
        groups = [compressible[i:i + objstm_size] for i in range(0, len(compressible), objstm_size)]
        stm_nums = []
        for k, group in enumerate(groups):
            stm_ref = self.pdf.alloc()
            stm_nums.append(stm_ref.num)
            extends = Ref(stm_nums[0]) if (objstm or {}).get("extends") and k > 0 else None
            self.write_objstm(group, stm_ref.num, extends)
            section[stm_ref.num] = self.entries[stm_ref.num]
            for n in group:
                compressed[n] = self.entries[n]
        freed = sorted((free or {}).items())
        for k, (n, gen) in enumerate(freed):
            following = freed[k + 1][0] if k + 1 < len(freed) else 0
            section[n] = (0, following, gen)
            self.entries[n] = (0, following, gen)
        if freed and not first:
            section[0] = (0, freed[0][0], 65535)
        if xref_num is None and kind != "table":
            xref_num = self.pdf.alloc().num
        size = max([n for n in self.entries] + [xref_num or 0, self.pdf.next_num - 1]) + 1
        trailer["Size"] = size
        if kind == "table":
            offset = self.xref_table(section, trailer, table_eol)
        elif kind in ("stream", "objstm"):
            offset = self.xref_stream({**section, **compressed}, trailer, xref_num, predictor)
        elif kind == "hybrid":
            # The stream lists the compressed objects; the table lists the rest and points to the stream.
            stream_offset = len(self.buf)
            self.xref_stream(compressed, {"Size": size}, xref_num, predictor)
            # The stream's own startxref line is not the file's: the table below is (drop it from the buffer).
            tail = b"startxref\n%d\n%%%%EOF\n" % stream_offset
            assert self.buf.endswith(tail)
            del self.buf[len(self.buf) - len(tail):]
            section[xref_num] = (1, stream_offset, 0)
            trailer["XRefStm"] = stream_offset
            offset = self.xref_table(section, trailer, table_eol)
        else:
            raise ValueError(kind)
        self.revisions.append({"kind": kind, "offset": offset, "objects": sorted(nums), "objectStreams": stm_nums,
                               "compressed": sorted(compressed), "free": sorted((free or {}).keys())})
        return offset


def stream_record(num, gen, stream, stored_length, revision=0):
    from .filters import filter_names

    d = stream.dict
    return {
        "object": [num, gen],
        "revision": revision,
        "type": d["Type"].value if isinstance(d.get("Type"), Name) else None,
        "filters": filter_names(d.get("Filter")),
        "storedLength": stored_length,
        "lengthWritten": stream.length if isinstance(stream.length, int) else stored_length if stream.length is None else "indirect",
        "decodedLength": len(stream.decoded),
        "decodedSha256": sha256(stream.decoded),
        "note": stream.note,
    }
