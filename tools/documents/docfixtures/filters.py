"""The stream filters' encoders (the reader's decoders are checked against them through qpdf, which decodes every one):
Flate with and without the zlib header, ASCIIHex, ASCII85, LZW with and without early change, RunLength, and the PNG
and TIFF predictors. zlib is the interpreter's own (its version is in the manifest)."""
import base64
import zlib

from .pdf import Name


def filter_names(value):
    if value is None:
        return []
    if isinstance(value, Name):
        return [value.value]
    return [v.value for v in value]


def flate(data, header=True, level=9):
    if header:
        return zlib.compress(data, level)
    c = zlib.compressobj(level, zlib.DEFLATED, -15)
    return c.compress(data) + c.flush()


def ahx(data, line=64, lower=False, odd=False):
    """Hex digits in lines of `line`, ended by '>'; `odd` drops the last digit when it is 0 (a reader appends it)."""
    text = data.hex()
    if not lower:
        text = text.upper()
    if odd:
        assert text.endswith("0"), "odd ASCIIHex needs data whose last digit is 0"
        text = text[:-1]
    lines = [text[i:i + line] for i in range(0, len(text), line)]
    return "\n".join(lines).encode("ascii") + b">"


def a85(data):
    return base64.a85encode(data, wrapcol=72) + b"~>"


def rle(data):
    out = bytearray()
    i, n = 0, len(data)
    while i < n:
        run = 1
        while i + run < n and run < 128 and data[i + run] == data[i]:
            run += 1
        if run >= 2:
            out += bytes([257 - run, data[i]])
            i += run
            continue
        j = i
        while j < n and j - i < 128 and not (j + 1 < n and data[j + 1] == data[j]):
            j += 1
        if j == i:
            j = i + 1
        out += bytes([j - i - 1]) + data[i:j]
        i = j
    out.append(128)
    return bytes(out)


class _Bits:
    def __init__(self):
        self.out = bytearray()
        self.acc = 0
        self.n = 0

    def put(self, code, width):
        self.acc = (self.acc << width) | code
        self.n += width
        while self.n >= 8:
            self.n -= 8
            self.out.append((self.acc >> self.n) & 0xFF)
        self.acc &= (1 << self.n) - 1

    def done(self):
        if self.n:
            self.out.append((self.acc << (8 - self.n)) & 0xFF)
            self.n = 0
        return bytes(self.out)


def _width(next_code, early):
    # The decoder reads with 9 bits while its next code plus early change stays below 512, and so on to 12 bits; the
    # encoder is one table entry ahead of the decoder.
    n = next_code - 1 + early
    return 9 if n < 512 else 10 if n < 1024 else 11 if n < 2048 else 12


def lzw(data, early=1, clear_at=4094):
    bits = _Bits()
    table = {bytes([i]): i for i in range(256)}
    next_code = 258
    bits.put(256, 9)
    if not data:
        bits.put(257, 9)
        return bits.done()
    w = data[:1]
    for b in data[1:]:
        wc = w + bytes([b])
        if wc in table:
            w = wc
            continue
        bits.put(table[w], _width(next_code, early))
        table[wc] = next_code
        next_code += 1
        if next_code >= clear_at:
            bits.put(256, _width(next_code, early))
            table = {bytes([i]): i for i in range(256)}
            next_code = 258
        w = bytes([b])
    bits.put(table[w], _width(next_code, early))
    bits.put(257, _width(next_code + 1, early))
    return bits.done()


def lzw_decode(data, early=1):
    """The reference decoder (as pdf.js and pdfium read it), used to check the encoder."""
    out = bytearray()
    acc, nbits, pos = 0, 0, 0
    width = 9
    table = [bytes([i]) for i in range(256)] + [b"", b""]
    prev = None
    while True:
        while nbits < width and pos < len(data):
            acc = (acc << 8) | data[pos]
            pos += 1
            nbits += 8
        if nbits < width:
            break
        nbits -= width
        code = (acc >> nbits) & ((1 << width) - 1)
        if code == 256:
            table = table[:258]
            width = 9
            prev = None
            continue
        if code == 257:
            break
        if prev is None:
            entry = table[code]
        elif code < len(table):
            entry = table[code]
            table.append(prev + entry[:1])
        elif code == len(table):
            entry = prev + prev[:1]
            table.append(entry)
        else:
            raise ValueError(f"LZW code {code} past the table ({len(table)})")
        out += entry
        prev = entry
        if len(table) + early >= (1 << width) and width < 12:
            width += 1
    return bytes(out)


def _paeth(a, b, c):
    p = a + b - c
    pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    return b if pb <= pc else c


def png_predict(data, colors, bpc, columns, pattern=(0, 1, 2, 3, 4)):
    """PNG prediction (Predictor 10-15): each row gets the filter `pattern` gives it in turn, its type byte first."""
    row_len = (colors * bpc * columns + 7) // 8
    bpp = max(1, colors * bpc // 8)
    assert len(data) % row_len == 0, (len(data), row_len)
    out = bytearray()
    prev = bytes(row_len)
    for r in range(len(data) // row_len):
        row = data[r * row_len:(r + 1) * row_len]
        f = pattern[r % len(pattern)]
        enc = bytearray(row_len)
        for i in range(row_len):
            a = row[i - bpp] if i >= bpp else 0
            b = prev[i]
            c = prev[i - bpp] if i >= bpp else 0
            pred = (0, a, b, (a + b) // 2, _paeth(a, b, c))[f]
            enc[i] = (row[i] - pred) & 0xFF
        out.append(f)
        out += enc
        prev = row
    return bytes(out)


def _unpack(row, bpc, count):
    if bpc == 8:
        return list(row[:count])
    if bpc == 16:
        return [row[2 * i] << 8 | row[2 * i + 1] for i in range(count)]
    vals = []
    for i in range(count):
        bit = i * bpc
        vals.append((row[bit >> 3] >> (8 - bpc - (bit & 7))) & ((1 << bpc) - 1))
    return vals


def _pack(vals, bpc, row_len):
    if bpc == 8:
        return bytes(vals) + bytes(row_len - len(vals))
    if bpc == 16:
        out = bytearray()
        for v in vals:
            out += v.to_bytes(2, "big")
        return bytes(out) + bytes(row_len - len(out))
    out = bytearray(row_len)
    for i, v in enumerate(vals):
        bit = i * bpc
        out[bit >> 3] |= v << (8 - bpc - (bit & 7))
    return bytes(out)


def tiff_predict(data, colors, bpc, columns):
    """TIFF predictor 2: each sample less the same component of the pixel to its left, modulo 2^bpc."""
    row_len = (colors * bpc * columns + 7) // 8
    count = colors * columns
    out = bytearray()
    for r in range(len(data) // row_len):
        vals = _unpack(data[r * row_len:(r + 1) * row_len], bpc, count)
        diff = [vals[i] if i < colors else (vals[i] - vals[i - colors]) % (1 << bpc) for i in range(count)]
        out += _pack(diff, bpc, row_len)
    return bytes(out)
