"""A small PNG writer (grey or RGB, 8 or 16 bits, no filtering, zlib level 9): the expected pixels and the rendered
pages are written through it, so their bytes depend only on the pixels and zlib."""
import struct
import zlib


def _chunk(kind, data):
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)


def png_bytes(width, height, channels, depth, pixels):
    """`pixels`: interleaved samples, row by row (16-bit samples big-endian)."""
    colour = {1: 0, 3: 2, 4: 6}[channels]
    row = width * channels * depth // 8
    assert len(pixels) == row * height, (len(pixels), row, height)
    raw = bytearray()
    for y in range(height):
        raw.append(0)
        raw += pixels[y * row:(y + 1) * row]
    ihdr = struct.pack(">IIBBBBB", width, height, depth, colour, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + _chunk(b"IHDR", ihdr) + _chunk(b"IDAT", zlib.compress(bytes(raw), 9)) \
        + _chunk(b"IEND", b"")
