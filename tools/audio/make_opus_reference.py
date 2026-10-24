"""The reference for the Opus plug-in (plan 32, phase O0): xiph's libopus built from its C source, a C harness
(tools/audio/opus_reference.c) that encodes generated signals into Ogg Opus files with libopus' own encoder and decodes
them with libopus' decoder, and the fixtures, their decoded samples and a manifest of what a decoder must give.

    python3 tools/audio/make_opus_reference.py tests/Idrak.Tests/data/audio/opus
    python3 tools/audio/make_opus_reference.py --work /tmp/opus-work --keep tests/Idrak.Tests/data/audio/opus

Needs git, cmake, a C compiler (CC, default cc) and numpy; ffmpeg is optional (an independent decoder to compare
with, both its own Opus decoder and the system's libopus through it). Nothing is downloaded but libopus' source:
https://github.com/xiph/opus at tag v1.6.1 (BSD-3-Clause; the newest release when this was written, with every
decoder fix to date), and v1.5.2 (the release before it) only to say whether its float decoder gives the same samples.

libopus is built with cmake, static, three ways: float (the reference), fixed point (to compare) and 1.5.2 float (to
compare), each with -O2 -ffp-contract=off (no fused multiply-adds) and OPUS_DISABLE_INTRINSICS=ON (no SSE/AVX/NEON
paths and no run-time CPU detection: plain C, the same arithmetic in the same order on every machine); DRED, OSCE,
deep PLC, QEXT and custom modes stay off (their defaults), so the decoder is RFC 6716's with RFC 8251's fixes.

The plug-in decodes SILK only (speech; plan 32's scope), so the fixtures are SILK in every bandwidth, frame size,
packet code and channel layout, with one short CELT-only and one short hybrid file for its refusal.

Writes, into the folder (see the README's opus/ row and opus.json's "conventions"):
- the fixtures (*.opus, one .ogg): libopus' encoder (SILK NB/MB/WB, 10-120 ms, mono and stereo, CBR and VBR, FEC,
  DTX, voice notes from 16 kHz input, frame size, channel and bandwidth changes); libopus' repacketizer for packet
  codes 1-3 with padding; damaged copies and variants made here;
- their decoded samples as .npy, planar [channels, samples]: int16 where every sample is a multiple of 1/32768 (SILK
  frames: the int16 values / 32768 are opus_decode_float's samples exactly, and opus_decode's int16 output is the
  same values), else float32 (the output gain, and the redundant CELT frames of a SILK bandwidth switch);
- opus.json: each file's settings, header, tags, pages, packets (TOC fields, frame sizes, padding, final range per
  packet and per frame, where each frame's SILK data ended and its redundant CELT frames), decoded lengths, the
  SHA-256 of every file and of every decoding (each rate, float and int16, with and without the output gain), the
  checks below, and the mode survey: which modes and bandwidths libopus' encoder picks, unforced, for voice-note-like
  settings (16 and 48 kHz input, VOIP and AUDIO, 12-48 kbit/s, 20 and 60 ms, clean speech and speech in noise);
  symbols.json (the first range-decoder symbols of a few packets, with the decoder's state after each) and
  range-coder.json (seeded symbol sequences through libopus' range encoder and decoder).

The SILK end and the redundant frames come from a traced copy of libopus' src/opus_decoder.c (two lines added that
report them; built with the library's flags; its samples checked equal to the reference's).

Checks made on every run (the script stops if one fails): every kept fixture is SILK only (the refusal files CELT or
hybrid only); the decoder's final range equals the encoder's for every packet; the same at every output rate; every
frame decoded alone (its own code-0 packet) gives the packet's samples; decoding twice gives the same bytes; opus_demo
(libopus' own tool) decodes every file with the final ranges checked; the repacketized and chained files decode as
their sources; the bandwidth switch happens twice, inside its stored window. Measured and recorded (not required
equal): the fixed-point build, libopus 1.5.2 and ffmpeg against the reference.

Reproducible: the signals use only IEEE-exact operations (+, -, *, /, sqrt, floor, round) and PCG64 uniform doubles,
so they are the same bits everywhere; a rerun on the same machine, compiler and C library writes the same bytes. The
float encoder calls the C library's log/exp/sin, so on another machine or C library a rerun may encode a few packets
differently; the fixtures in the repository are then still the reference (their decodings come from the files).
"""
import argparse
import hashlib
import json
import math
import os
import platform
import shutil
import struct
import subprocess
import sys
import tempfile
from fractions import Fraction

import numpy as np

REPOSITORY = "https://github.com/xiph/opus.git"
TAG = "v1.6.1"
COMPARE_TAG = "v1.5.2"
C_FLAGS = "-O2 -ffp-contract=off"
CMAKE_OPTIONS = ["-DCMAKE_BUILD_TYPE=None", "-DCMAKE_C_FLAGS=" + C_FLAGS, "-DOPUS_DISABLE_INTRINSICS=ON",
                 "-DOPUS_BUILD_PROGRAMS=ON", "-DOPUS_BUILD_SHARED_LIBRARY=OFF", "-DOPUS_BUILD_TESTING=OFF"]
RATES = [48000, 24000, 16000, 12000, 8000]
HERE = os.path.dirname(os.path.abspath(__file__))

# ---- signals: IEEE-exact arithmetic only, so they are the same bits on every machine ----

TWO_PI_HI = 6.283185307179586            # 2*pi rounded to double
TWO_PI_LO = 2.4492935982947064e-16       # 2*pi - TWO_PI_HI
SIN_COEFFICIENTS = [float(Fraction((-1) ** n, math.factorial(2 * n + 1))) for n in range(14)]


def sin(x):
    """sin(x) to about 1e-15 from +, *, floor only (numpy's own sin may differ in the last bit between CPUs)."""
    k = np.floor(x / TWO_PI_HI + 0.5)
    r = (x - k * TWO_PI_HI) - k * TWO_PI_LO
    r2 = r * r
    acc = np.full_like(r, SIN_COEFFICIENTS[-1])
    for c in reversed(SIN_COEFFICIENTS[:-1]):
        acc = acc * r2 + c
    return acc * r


def generator(seed):
    return np.random.Generator(np.random.PCG64(seed))


def noise(rng, n):
    """Roughly normal noise (four uniforms, Irwin-Hall), unit variance."""
    u = rng.random(n) + rng.random(n) + rng.random(n) + rng.random(n)
    return (u - 2.0) * 1.7320508075688772


def smooth(x, taps):
    """A moving average over `taps` samples (a low-pass), summed in a fixed order."""
    y = np.zeros_like(x)
    for k in range(taps):
        y[k:] += x[:len(x) - k]
    return y / taps


def lorentz(f, centre, width):
    d = (f - centre) / width
    return 1.0 / (1.0 + d * d)


VOWELS = [(730, 1090, 2440), (270, 2290, 3010), (300, 870, 2240), (530, 1840, 2480), (570, 840, 2410)]


def voice_part(rate, n, rng, f0_range=(95, 230), start=0.05):
    """Voiced syllables (a gliding pitch's harmonics shaped by three formants), fricatives, pauses."""
    x = np.zeros(n)
    voiced = np.zeros(n, dtype=bool)
    at = start
    total = n / rate
    count = 0
    while at < total - 0.12:
        length = 0.12 + 0.13 * rng.random()
        a, b = int(at * rate), min(n, int((at + length) * rate))
        u = np.arange(b - a) / (b - a)
        f0a, f0b = f0_range[0] + (f0_range[1] - f0_range[0]) * rng.random(), f0_range[0] + (f0_range[1] - f0_range[0]) * rng.random()
        f0 = f0a + (f0b - f0a) * u
        phase = np.cumsum(f0 * (TWO_PI_HI / rate))
        f1, f2, f3 = VOWELS[int(rng.random() * len(VOWELS))]
        tone = np.zeros(b - a)
        for k in range(1, 400):
            fk = k * f0
            if fk[0] > rate / 2 - 300 or fk[-1] > rate / 2 - 300:
                break
            shape = 1.0 / (k ** 0.7) * (0.15 + lorentz(fk, f1, 90) + 0.7 * lorentz(fk, f2, 120) + 0.35 * lorentz(fk, f3, 160))
            tone += shape * sin(k * phase)
        envelope = np.sqrt(sin(np.pi * u) * (u < 1))
        x[a:b] += tone * envelope
        voiced[a:b] = True
        at += length
        count += 1
        gap = 0.04 + 0.1 * rng.random()
        if count % 3 == 0 and at + gap + 0.07 < total:   # a fricative in every third gap
            fa, fb = int((at + 0.01) * rate), int((at + 0.07) * rate)
            hiss = noise(rng, fb - fa)
            hiss = hiss - smooth(hiss, 3)
            hiss = hiss - smooth(hiss, 3)
            x[fa:fb] += 0.25 * hiss * sin(np.pi * np.arange(fb - fa) / (fb - fa))
            gap += 0.07
        at += gap
    return x, voiced


def quantize(x, level):
    """Scaled to `level` at the peak and rounded to 16-bit steps (exact in float32)."""
    peak = np.max(np.abs(x))
    if peak > 0:
        x = x * (level / peak)
    return (np.clip(np.round(x * 32767.0), -32768, 32767) / 32768.0).astype(np.float32)


def speech(rate, n, seed, channels=1, silence=None, level=0.35):
    """Speech-like syllables over a faint noise floor; with `silence` (start, end in seconds), digital zeros there and
    no floor anywhere (what DTX needs: 200 ms of silence before it stops sending)."""
    if n < rate // 2:   # a syllable needs more than a short file: made longer, then cut
        return speech(rate, rate // 2, seed, channels, silence, level)[:, :n]
    rng = generator(seed)
    a, _ = voice_part(rate, n, rng)
    floor = 0.0 if silence else 0.004 * smooth(noise(rng, n), 4) / np.max(np.abs(a))
    if channels == 1:
        x = a / np.max(np.abs(a)) + floor
        if silence:
            x[int(silence[0] * rate):int(silence[1] * rate)] = 0.0
        return quantize(x, level)[None, :]
    b, _ = voice_part(rate, n, rng, f0_range=(160, 300), start=0.11)
    left = a / np.max(np.abs(a)) + 0.45 * b / np.max(np.abs(b)) + floor
    right = 0.55 * a / np.max(np.abs(a)) + b / np.max(np.abs(b)) + floor
    both = np.stack([left, right])
    return np.stack([quantize(c, level * np.max(np.abs(c)) / np.max(np.abs(both))) for c in both])


SCALE = [220.0, 246.94, 261.63, 293.66, 329.63, 349.23, 392.0, 440.0, 493.88, 523.25, 587.33, 659.26]


def music(rate, n, seed, channels=1, level=0.5):
    """Plucked notes (harmonics, decaying) over a beat of kicks and hi-hats, panned across the channels (only for the
    CELT file the decoder refuses)."""
    rng = generator(seed)
    t = np.arange(n) / rate
    out = np.zeros((2, n))
    at = 0.0
    while at < n / rate:
        f0 = SCALE[int(rng.random() * len(SCALE))] * (1 + int(rng.random() * 2))
        pan = rng.random()
        a = int(at * rate)
        dt = t[a:] - at
        env = 1.0 / (1.0 + dt / 0.12) ** 2 * np.minimum(1.0, dt / 0.004)
        tone = np.zeros(n - a)
        for k in range(1, 16):
            if k * f0 > rate / 2 - 500:
                break
            tone += (1.0 / k) * sin((TWO_PI_HI * k * f0) * dt) / (1.0 + dt * (k / 0.6))
        out[0, a:] += (1 - pan) * tone * env
        out[1, a:] += pan * tone * env
        at += 0.06 + 0.12 * rng.random()
    beat = 0.0
    while beat < n / rate:
        a = int(beat * rate)
        dt = t[a:] - beat
        kick = sin(TWO_PI_HI * (55.0 * dt + 40.0 * dt / (1.0 + dt / 0.02))) / (1.0 + dt / 0.05) ** 2
        hat = noise(rng, n - a)
        hat = hat - smooth(hat, 2)
        hat = 0.25 * hat / (1.0 + dt / 0.008) ** 2
        out[0, a:] += 0.8 * kick + hat
        out[1, a:] += 0.8 * kick + 0.8 * hat
        beat += 0.125
    if channels == 1:
        return quantize(out[0] + out[1], level)[None, :]
    peak = np.max(np.abs(out))
    return np.stack([quantize(c, level * np.max(np.abs(c)) / peak) for c in out])


def speech_in_noise(rate, n, seed):
    """Speech over a steady background (low-passed noise and a quiet hum, about 20 dB under the speech's peaks): a voice
    note recorded in a room with a fan or traffic."""
    x = speech(rate, n, seed)[0].astype(np.float64)
    rng = generator(seed + 1000)
    t = np.arange(n) / rate
    background = 0.06 * smooth(noise(rng, n), 3) + 0.02 * sin(TWO_PI_HI * 100.0 * t)
    return quantize(x + background, 0.4)[None, :]


SIGNALS = {"speech": speech, "music": music, "speech-in-noise": speech_in_noise}

# ---- the fixtures ----


def F(name, seconds, source, seed, decode=(48000,), extra=(), **settings):
    return {"name": name, "seconds": seconds, "signal": source, "seed": seed, "npy": list(decode), "extra": list(extra), "settings": settings}


# settings: the harness' config keys (rate, channels, family, application, frame, bitrate, vbr, cvbr, complexity, dtx,
# fec, loss, gain, bandwidth, mode, signal, max_page, page_ms, force_channels); "changes": [(seconds, key, value)].
# The decoder reads SILK only (plan 32's scope): every SILK variant at length, and one short CELT-only and one short
# hybrid file for its refusal.
FIXTURES = [
    F("silk-nb-10ms-mono", 1.0, "speech", 1, (48000, 8000), rate=8000, application="voip", mode="silk", bandwidth="nb", frame=10, bitrate=8000),
    F("silk-nb-40ms-stereo", 0.6, "speech", 2, (48000,), channels=2, mode="silk", bandwidth="nb", frame=40, bitrate=16000, signal="voice"),
    F("silk-mb-20ms-mono", 0.6, "speech", 3, (48000, 16000), mode="silk", bandwidth="mb", frame=20, bitrate=12000, signal="voice"),
    F("silk-wb-20ms-stereo", 0.8, "speech", 4, (48000, 16000), channels=2, mode="silk", bandwidth="wb", frame=20, bitrate=28000, signal="voice"),
    F("silk-wb-40ms-mono", 1.0, "speech", 5, (48000,), mode="silk", bandwidth="wb", frame=40, bitrate=16000, signal="voice"),
    F("silk-wb-60ms-stereo", 0.72, "speech", 6, (48000,), channels=2, mode="silk", bandwidth="wb", frame=60, bitrate=24000, signal="voice"),
    F("silk-wb-20ms-mono-fec", 1.0, "speech", 7, (48000,), application="voip", mode="silk", bandwidth="wb", frame=20, bitrate=24000, fec=1, loss=20),
    F("silk-wb-20ms-mono-cbr", 0.6, "speech", 8, (48000,), mode="silk", bandwidth="wb", frame=20, bitrate=32000, vbr=0, signal="voice"),
    F("silk-nb-80ms-mono-cbr", 0.8, "speech", 9, (48000,), mode="silk", bandwidth="nb", frame=80, bitrate=12000, vbr=0, signal="voice"),
    F("silk-wb-100ms-mono", 1.0, "speech", 10, (48000,), mode="silk", bandwidth="wb", frame=100, bitrate=16000, signal="voice"),
    F("silk-wb-120ms-mono", 1.2, "speech", 11, (48000,), mode="silk", bandwidth="wb", frame=120, bitrate=16000, signal="voice"),
    F("silk-frame-changes-mono", 1.2, "speech", 12, (48000,), mode="silk", bandwidth="wb", frame=20, bitrate=20000, signal="voice",
      changes=[(0.2, "frame", 40), (0.44, "frame", 60), (0.68, "frame", 10), (0.88, "frame", 20)]),
    # SILK's own bandwidth switch: asked for MB at 0.06 s, libopus switches after its transition (counted in SILK frames:
    # about 2.6 s at 20 ms); asked for WB again at 2.9 s, it switches back at once. Each switch carries redundant CELT
    # frames (checked: two switches, inside the window). Stored: 16 kHz whole, and 48 kHz from 2.5 s to the end.
    F("silk-bandwidth-switch-mono", 3.2, "speech", 20, (16000,), mode="silk", bandwidth="wb", frame=20, bitrate=20000, signal="voice", window=(2.5, 3.2),
      changes=[(0.06, "bandwidth", "mb"), (0.06, "bitrate", 12000), (2.9, "bandwidth", "wb"), (2.9, "bitrate", 20000)]),
    F("silk-wb-20ms-stereo-to-mono", 0.8, "speech", 13, (48000,), channels=2, mode="silk", bandwidth="wb", frame=20, bitrate=28000, signal="voice",
      force_channels=2, changes=[(0.3, "force_channels", 1), (0.56, "force_channels", 2)]),
    F("dtx-silk-wb-20ms-mono", 1.0, "speech-silence", 14, (48000,), application="voip", mode="silk", bandwidth="wb", frame=20, bitrate=16000, dtx=1, silence=(0.25, 0.75)),
    F("voicenote-16k-20ms", 1.2, "speech", 15, (48000, 24000, 16000, 12000, 8000), rate=16000, application="voip", frame=20, bitrate=20000,
      comments=["TITLE=\u0645\u0644\u0627\u062d\u0638\u0629 \u0635\u0648\u062a\u064a\u0629", "LANGUAGE=ar"]),
    F("voicenote-16k-60ms", 1.2, "speech", 16, (48000, 16000), rate=16000, application="voip", frame=60, bitrate=16000),
    F("long-silk-wb-20ms-stereo-cbr", 4.2, "speech", 17, (), channels=2, mode="silk", bandwidth="wb", frame=20, bitrate=128000, vbr=0, signal="voice"),
    F("refuse-celt-fb-20ms-mono", 0.1, "music", 18, (), mode="celt", bandwidth="fb", frame=20, bitrate=48000),
    F("refuse-hybrid-swb-20ms-mono", 0.1, "speech", 19, (), mode="hybrid", bandwidth="swb", frame=20, bitrate=28000, signal="voice"),
]

# Files the voice notes must be: unforced, and SILK in every packet (checked).
SILK_ONLY_UNFORCED = ["voicenote-16k-20ms", "voicenote-16k-60ms"]

# libopus' repacketizer over a fixture's packets: (name, source, packets per group (-1: each packet's frames twice),
# padding bytes per packet). Grouped files decode as their sources; the doubled one is a new stream, decoded on its own.
REPACKED = [
    ("repack-code1-silk-wb-20ms-doubled", "silk-mb-20ms-mono", -1, 0),
    ("repack-code2-silk-wb-20ms-stereo", "silk-wb-20ms-stereo", 2, 0),
    ("repack-code3-cbr-pad10-silk-wb-20ms", "silk-wb-20ms-mono-cbr", 3, 10),
    ("repack-code3-12x10ms-pad300-silk-nb", "silk-nb-10ms-mono", 12, 300),
]

SYMBOL_FILES = ["silk-nb-10ms-mono", "silk-wb-20ms-stereo", "silk-wb-60ms-stereo", "silk-wb-100ms-mono", "voicenote-16k-20ms"]

# The mode survey: which modes and bandwidths libopus' encoder picks, unforced, for voice-note-like settings.
SURVEY_SECONDS = 3.0
SURVEY = ([(16000, app, kbps, frame) for app in ("voip", "audio") for kbps in (12, 16, 24, 32, 48) for frame in (20, 60)]
          + [(48000, app, kbps, frame) for app in ("voip", "audio") for kbps in (24, 32) for frame in (20, 60)])
SURVEY_SIGNALS = [("speech", 101), ("speech-in-noise", 102)]


def length_48k(seconds):
    """The input's length at 48 kHz: a multiple of 12 (so every Opus rate divides it) that no frame size divides."""
    return int(round(seconds * 100)) * 480 + 132


def serial_of(name):
    return int(hashlib.sha256(name.encode()).hexdigest()[:8], 16)


# ---- Ogg in Python (for the damaged copies and the variants): the same CRC as the harness' ----

CRC_TABLE = []
for i in range(256):
    r = i << 24
    for _ in range(8):
        r = ((r << 1) ^ 0x04C11DB7) & 0xFFFFFFFF if r & 0x80000000 else (r << 1) & 0xFFFFFFFF
    CRC_TABLE.append(r)


def ogg_crc(data):
    crc = 0
    for b in data:
        crc = ((crc << 8) & 0xFFFFFFFF) ^ CRC_TABLE[((crc >> 24) ^ b) & 0xFF]
    return crc


def read_pages(data):
    pages, off = [], 0
    while off < len(data):
        nseg = data[off + 26]
        lacing = list(data[off + 27:off + 27 + nseg])
        body = data[off + 27 + nseg:off + 27 + nseg + sum(lacing)]
        flags, granule, serial, seq = data[off + 5], *struct.unpack_from("<qII", data, off + 6)
        pages.append({"offset": off, "flags": flags, "granule": granule, "serial": serial, "seq": seq, "lacing": lacing, "body": bytes(body)})
        off += 27 + nseg + len(body)
    return pages


def page_bytes(flags, granule, serial, seq, lacing, body):
    head = b"OggS" + bytes([0, flags]) + struct.pack("<qIII", granule, serial, seq, 0) + bytes([len(lacing)]) + bytes(lacing)
    crc = ogg_crc(head + body)
    return head[:22] + struct.pack("<I", crc) + head[26:] + body


def lacing_of(packets):
    out = []
    for p in packets:
        out += [255] * (len(p) // 255) + [len(p) % 255]
    return out


def packets_of(page):
    """The packets of a page that holds whole packets only."""
    out, at, current = [], 0, b""
    for l in page["lacing"]:
        current += page["body"][at:at + l]
        at += l
        if l < 255:
            out.append(current)
            current = b""
    assert current == b"" and not page["flags"] & 1, "the page holds part of a packet"
    return out


def rewrite_page(page, packets):
    return page_bytes(page["flags"], page["granule"], page["serial"], page["seq"], lacing_of(packets), b"".join(packets))


def whole(pages, replace):
    """The file's pages again, with the pages in `replace` (index: bytes) swapped in."""
    return b"".join(replace.get(i, page_bytes(p["flags"], p["granule"], p["serial"], p["seq"], p["lacing"], p["body"])) for i, p in enumerate(pages))


MODE_NAMES = ["silk"] * 12 + ["hybrid"] * 4 + ["celt"] * 16
BANDWIDTH_NAMES = ["nb"] * 4 + ["mb"] * 4 + ["wb"] * 4 + ["swb"] * 2 + ["fb"] * 2 + ["nb"] * 4 + ["wb"] * 4 + ["swb"] * 4 + ["fb"] * 4
FRAME_MS = [10, 20, 40, 60] * 3 + [10, 20] * 2 + [2.5, 5, 10, 20] * 4


def ogg_packets(data):
    """Every packet of a one-stream file, in order."""
    packets, current = [], b""
    for p in read_pages(data):
        at = 0
        for l in p["lacing"]:
            current += p["body"][at:at + l]
            at += l
            if l < 255:
                packets.append(current)
                current = b""
    return packets


# ---- running things ----


def run(args, **kw):
    return subprocess.run(args, check=True, **kw)


def harness(program, *args):
    """Runs the harness; its result is the JSON on stdout (or the error on its last line)."""
    p = subprocess.run([program, *map(str, args)], capture_output=True, text=True)
    if p.returncode == 2:
        return {"error": json.loads(p.stdout.strip().splitlines()[-1])["error"]}
    if p.returncode != 0:
        sys.exit(f"{program} {' '.join(map(str, args))}: exit {p.returncode}\n{p.stderr}")
    return json.loads(p.stdout)


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha256(path):
    with open(path, "rb") as f:
        return sha256_bytes(f.read())


def checkout(given, tag, work):
    if given:
        return os.path.abspath(given)
    target = os.path.join(work, "opus-" + tag)
    if not os.path.isdir(target):
        run(["git", "-c", "advice.detachedHead=false", "clone", "--quiet", "--depth", "1", "--branch", tag, REPOSITORY, target])
    return target


def commit_of(folder):
    try:
        return run(["git", "-C", folder, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def build(source, work, name, fixed):
    """libopus static with cmake, then the harness against it; returns (harness, opus_demo, the compile definitions)."""
    folder = os.path.join(work, "build-" + name)
    options = CMAKE_OPTIONS + ["-DOPUS_FIXED_POINT=" + ("ON" if fixed else "OFF")]
    if not os.path.exists(os.path.join(folder, "libopus.a")):
        run(["cmake", "-S", source, "-B", folder, *options], stdout=subprocess.DEVNULL)
        run(["cmake", "--build", folder, "-j", str(os.cpu_count() or 2)], stdout=subprocess.DEVNULL)
    with open(os.path.join(folder, "CMakeFiles", "opus.dir", "flags.make"), encoding="utf-8") as f:
        flags = dict(line.split(" = ", 1) for line in f.read().splitlines() if " = " in line)
    program = os.path.join(work, "opus_reference-" + name)
    cc = os.environ.get("CC", "cc")
    run([cc, *C_FLAGS.split(), "-std=gnu99", *[a for d in ["include", "celt", "src", "silk"] for a in ("-I", os.path.join(source, d))],
         "-I", folder, os.path.join(HERE, "opus_reference.c"), os.path.join(folder, "libopus.a"), "-lm", "-o", program])
    return program, os.path.join(folder, "opus_demo"), {"cmake_options": options, "defines": flags.get("C_DEFINES", "").split(), "c_flags": flags.get("C_FLAGS", "")}


def build_traced(source, work):
    """The harness with OPUS_REFERENCE_TRACE against the float build and a copy of libopus' src/opus_decoder.c with two
    lines added (where the SILK part ended, and whether a redundant CELT frame follows), compiled with the library's own
    flags; its samples are checked equal to the reference's."""
    folder = os.path.join(work, "build-float")
    with open(os.path.join(folder, "CMakeFiles", "opus.dir", "flags.make"), encoding="utf-8") as f:
        flags = dict(line.split(" = ", 1) for line in f.read().splitlines() if " = " in line)
    with open(os.path.join(source, "src", "opus_decoder.c"), encoding="utf-8") as f:
        text = f.read()
    anchors = [("   start_band = 0;\n   if (!decode_fec && mode != MODE_CELT_ONLY && data != NULL",
                "   opus_reference_silk_tell = data != NULL ? ec_tell(&dec) : -1;\n   start_band = 0;\n   if (!decode_fec && mode != MODE_CELT_ONLY && data != NULL"),
               ("   if (mode != MODE_CELT_ONLY)\n      start_band = 17;",
                "   opus_reference_trace(mode, st->prev_mode, redundancy, celt_to_silk, redundancy_bytes, len);\n   if (mode != MODE_CELT_ONLY)\n      start_band = 17;")]
    for a, b in anchors:
        if text.count(a) != 1:
            sys.exit(f"opus_decoder.c: the trace anchor is not there once: {a!r}")
        text = text.replace(a, b)
    text = ("extern int opus_reference_silk_tell;\n"
            "void opus_reference_trace(int mode, int prev_mode, int redundancy, int celt_to_silk, int redundancy_bytes, int len);\n") + text
    patched = os.path.join(work, "opus_decoder_traced.c")
    with open(patched, "w", encoding="utf-8") as f:
        f.write(text)
    cc = os.environ.get("CC", "cc")
    obj = os.path.join(work, "opus_decoder_traced.o")
    run([cc, *flags["C_DEFINES"].split(), "-I", os.path.join(source, "src"), *flags["C_INCLUDES"].split(), *flags["C_FLAGS"].split(), "-w", "-c", patched, "-o", obj])
    program = os.path.join(work, "opus_reference-traced")
    run([cc, *C_FLAGS.split(), "-std=gnu99", "-DOPUS_REFERENCE_TRACE", *[a for d in ["include", "celt", "src", "silk"] for a in ("-I", os.path.join(source, d))],
         "-I", folder, os.path.join(HERE, "opus_reference.c"), obj, os.path.join(folder, "libopus.a"), "-lm", "-o", program])
    return program


def planar(raw, dtype, channels):
    return np.frombuffer(raw, dtype=dtype).reshape(-1, channels).T.copy()


def decode(program, path, rate, work, link=0, gain=1, frames=False):
    """The harness' decoding: (its JSON, float32 [C, N], int16 [C, N], frames-alone float32 or None)."""
    f32, i16, fr = (os.path.join(work, "out." + e) for e in ("f32", "i16", "frames"))
    info = harness(program, "decode", path, rate, link, gain, f32, i16, *([fr] if frames else []))
    if "error" in info:
        return info, None, None, None
    c = info["head"]["channels"]
    with open(f32, "rb") as f:
        x = planar(f.read(), "<f4", c)
    with open(i16, "rb") as f:
        y = planar(f.read(), "<i2", c)
    z = None
    if frames:
        with open(fr, "rb") as f:
            raw = f.read()
        z = planar(raw, "<f4", c) if raw else None
    return info, x, y, z


def max_diff(a, b):
    n = min(a.shape[-1], b.shape[-1])
    return float(np.max(np.abs(a[..., :n].astype(np.float64) - b[..., :n].astype(np.float64)))) if n else 0.0


def ffmpeg_decode(path, channels, decoder):
    """ffmpeg's decoding at 48 kHz as float32 [C, N] (its own Opus decoder, or libopus through it), or None."""
    if not shutil.which("ffmpeg"):
        return None
    request = ["-request_sample_fmt", "flt"] if decoder == "libopus" else []
    p = subprocess.run(["ffmpeg", "-v", "error", *request, "-c:a", decoder, "-i", path, "-f", "f32le", "-ar", "48000", "-"], capture_output=True)
    if p.returncode != 0 or not p.stdout:
        return None
    return planar(p.stdout, "<f4", channels)


def opus_demo_check(opus_demo, info, path, work, reference):
    """opus_demo -d on the file's packets with their final ranges (it stops on a mismatch); its 24-bit output (as float)
    against the reference; mono and stereo only."""
    packets = ogg_packets(open(path, "rb").read())
    bit = os.path.join(work, "demo.bit")
    with open(bit, "wb") as f:
        for p, row in zip(packets[2:], info["packets"]):
            f.write(struct.pack(">II", len(p), row["final_range"]) + p)
    out = os.path.join(work, "demo.f32")
    c = info["head"]["channels"]
    r = subprocess.run([opus_demo, "-d", "48000", str(c), "-f32", bit, out], capture_output=True, text=True)
    if r.returncode != 0 or "mismatch" in r.stderr:
        sys.exit(f"opus_demo on {path}: {r.stderr}")
    y = planar(open(out, "rb").read(), "<f4", c)[:, info["skip"]:info["skip"] + info["length"]]
    return {"ranges_checked": len(packets) - 2, "samples_equal_length": y.shape[1] == reference.shape[1], "max_abs_diff_24bit": max_diff(y, reference)}


def to_npy(path, x, channels):
    """float32 as is, or int16 when every sample is a multiple of 1/32768 (exact); returns the dtype written."""
    scaled = x.astype(np.float64) * 32768.0
    if np.all(scaled == np.round(scaled)) and np.all(np.abs(scaled) <= 32767):
        np.save(path, scaled.astype("<i2"))
        return "int16"
    np.save(path, x.astype("<f4"))
    return "float32"


# ---- the manifest's JSON: dictionaries indented, rows of numbers on one line ----


def has_dict(value):
    return isinstance(value, dict) or (isinstance(value, list) and any(has_dict(v) for v in value))


def dump(value, indent=0):
    pad = "  " * indent
    if isinstance(value, dict):
        if not value:
            return "{}"
        inline = json.dumps(value, ensure_ascii=False)
        if len(inline) < 160 and not any(has_dict(v) for v in value.values()):
            return inline
        items = []
        for k, v in value.items():
            if k == "rows" and isinstance(v, list):
                text = "[\n" + ",\n".join(pad + "    " + json.dumps(r, ensure_ascii=False) for r in v) + "\n" + pad + "  ]" if v else "[]"
            else:
                text = dump(v, indent + 1)
            items.append(f"{pad}  {json.dumps(k, ensure_ascii=False)}: {text}")
        return "{\n" + ",\n".join(items) + "\n" + pad + "}"
    if isinstance(value, list):
        if not has_dict(value):
            return json.dumps(value, ensure_ascii=False)
        return "[\n" + ",\n".join(pad + "  " + dump(v, indent + 1) for v in value) + "\n" + pad + "]"
    return json.dumps(value, ensure_ascii=False)


PACKET_FIELDS = ["page", "bytes", "config", "mode", "bandwidth", "frame_ms", "stereo", "code", "vbr", "padding", "frames", "samples_48k", "final_range", "frame_ranges",
                 "silk_tell", "redundancy"]


MS_PACKET_FIELDS = ["page", "bytes", "streams", "samples_48k", "final_range"]


def packet_rows(info, ratio):
    rows = []
    for p in info["packets"]:
        if "streams" in p:
            rows.append([p["page"], p["bytes"], [[s["config"], s["mode"], s["bandwidth"], s["frame_ms"], s["stereo"], s["code"], s["bytes"], s["padding"], s["frames"]] for s in p["streams"]],
                         p["samples"] * ratio, p["final_range"]])
            continue
        rows.append([p.get("page"), p.get("bytes"), p.get("config"), p.get("mode"), p.get("bandwidth"), p.get("frame_ms"), p.get("stereo"), p.get("code"),
                     p.get("vbr"), p.get("padding"), p.get("frames"), p["samples"] * ratio, p["final_range"],
                     p.get("frame_ranges", []) if len(p.get("frames", [])) > 1 else [],
                     [t[2] for t in p.get("trace", [])], [[i, t[4], t[5]] for i, t in enumerate(p.get("trace", [])) if t[3]]])
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("out", help="the folder to write the fixtures into")
    parser.add_argument("--opus", help=f"a checkout of xiph/opus at {TAG} (default: cloned from GitHub)")
    parser.add_argument("--opus-compare", help=f"a checkout at {COMPARE_TAG} (default: cloned)")
    parser.add_argument("--work", help="the folder for clones and builds (default: a temporary one, removed)")
    parser.add_argument("--keep", action="store_true", help="keep the work folder")
    args = parser.parse_args()

    work = os.path.abspath(args.work) if args.work else tempfile.mkdtemp(prefix="opus-")
    os.makedirs(work, exist_ok=True)
    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)
    for name in os.listdir(out):
        if name.endswith((".opus", ".ogg", ".npy", ".json")):
            os.remove(os.path.join(out, name))
    scratch = os.path.join(work, "scratch")
    os.makedirs(scratch, exist_ok=True)
    try:
        source = checkout(args.opus, TAG, work)
        compare_source = checkout(args.opus_compare, COMPARE_TAG, work)
        print(f"building libopus {TAG} float and fixed point, {COMPARE_TAG} float", flush=True)
        ref, opus_demo, ref_build = build(source, work, "float", False)
        traced = build_traced(source, work)
        fixed, _, fixed_build = build(source, work, "fixed", True)
        older, _, older_build = build(compare_source, work, "compare", False)
        cc = os.environ.get("CC", "cc")
        compiler = run([cc, "--version"], capture_output=True, text=True).stdout.splitlines()[0]

        files = {}
        decoded = {}       # fixture name -> {rate: (float32 [C, N], int16 [C, N])} at the reference

        def record_file(name):
            path = os.path.join(out, name)
            entry = files.setdefault(name, {})
            entry["bytes"] = os.path.getsize(path)
            entry["sha256"] = sha256(path)
            return entry

        def full_decode(name, path, npy_rates, hash_rates, gain_variants=(1,), link=0, base=None, window=None):
            """Decodes at every rate with the reference (and twice at 48 kHz), checks, and records it."""
            entry = files[os.path.basename(path)]
            info48, x48, y48, z48 = decode(ref, path, 48000, scratch, link=link, frames=True)
            if "error" in info48:
                sys.exit(f"{path}: {info48['error']}")
            again = decode(ref, path, 48000, scratch, link=link)
            if again[1].tobytes() != x48.tobytes() or again[2].tobytes() != y48.tobytes():
                sys.exit(f"{path}: two decodings differ")
            head = info48["head"]
            entry.update({"serial": info48["serial"], "head": head, "vendor": info48["vendor"], "comments": info48["comments"],
                          "pages": {"fields": ["offset", "size", "flags", "granule", "serial", "sequence", "segments", "packets_ended"], "rows": info48["pages"]},
                          "streams": info48["streams"], "last_granule": info48["last_granule"], "decoded_48k_untrimmed": info48["decoded_48k"],
                          "length_48k": info48["length"]})
            if head["family"] == 0:
                if z48 is None or z48.tobytes() != x48.tobytes():
                    sys.exit(f"{path}: the frames decoded alone differ from the packets")
                for p in info48["packets"]:
                    if p["frame_ranges"][-1] != p["final_range"]:
                        sys.exit(f"{path}: the last frame's range differs from the packet's")
            tinfo, tx, _, _ = decode(traced, path, 48000, scratch, link=link)
            if tx.tobytes() != x48.tobytes():
                sys.exit(f"{path}: the traced decoder's samples differ from the reference's")
            for p, t in zip(info48["packets"], tinfo["packets"]):
                p["trace"] = t["trace"]
            entry["packets"] = {"fields": PACKET_FIELDS if head["family"] == 0 else MS_PACKET_FIELDS, "rows": packet_rows(info48, 1)}
            entry["redundant_celt_frames"] = sum(1 for p in info48["packets"] for t in p["trace"] if t[3])
            if head["family"] != 0:
                entry["packets"]["stream_fields"] = ["config", "mode", "bandwidth", "frame_ms", "stereo", "code", "bytes", "padding", "frames"]
            decodes = []
            results = {}
            for rate in sorted(set(npy_rates) | set(hash_rates) | {48000}, reverse=True):
                for gain in gain_variants:
                    info, x, y, _ = (info48, x48, y48, None) if rate == 48000 and gain == 1 else decode(ref, path, rate, scratch, link=link, gain=gain)
                    ranges = [p["final_range"] for p in info["packets"]]
                    if ranges != [p["final_range"] for p in info48["packets"]]:
                        sys.exit(f"{path}: the final ranges at {rate} Hz differ from 48 kHz's")
                    d = {"rate": rate, "gain_applied": bool(gain), "skip": info["skip"], "samples": info["length"], "exact_scaling": bool(info["exact_scaling"]),
                         "float32_sha256": sha256_bytes(x.astype("<f4").tobytes()), "int16_sha256": sha256_bytes(y.astype("<i2").tobytes())}
                    scaled = x.astype(np.float64) * 32768.0
                    d["float_is_int16_exactly"] = bool(np.all(scaled == y.astype(np.float64)))
                    d["clipped_samples"] = int(np.sum(np.abs(x) > 1.0))
                    if window and rate == 48000 and gain == gain_variants[0]:
                        a, b = int(round(window[0] * rate)), min(x.shape[1], int(round(window[1] * rate)))
                        npy = f"{os.path.splitext(os.path.basename(path))[0]}-{rate}-window.npy"
                        d["npy"] = npy
                        d["npy_window"] = [a, b]
                        d["npy_dtype"] = to_npy(os.path.join(out, npy), x[:, a:b], head["channels"])
                    elif gain == gain_variants[0] and rate in npy_rates and base is None:
                        npy = f"{os.path.splitext(os.path.basename(path))[0]}-{rate}.npy"
                        d["npy"] = npy
                        d["npy_dtype"] = to_npy(os.path.join(out, npy), x, head["channels"])
                    elif base is not None:
                        d["same_as"] = base
                    decodes.append(d)
                    results[(rate, gain)] = (x, y)
            entry["decodes"] = decodes
            return info48, results

        def comparisons(name, path, info48, results):
            entry = files[os.path.basename(path)]
            x48, y48 = results[(48000, 1)]
            checks = {}
            c = info48["head"]["channels"]
            for label, program in [("fixed_point", fixed), ("libopus_" + COMPARE_TAG, older)]:
                row = {}
                for rate in (48000, 16000):
                    info, x, y, _ = decode(program, path, rate, scratch)
                    xr, yr = results[(rate, 1)] if (rate, 1) in results else decode(ref, path, rate, scratch)[1:3]
                    row[str(rate)] = {"float32_equal": x.tobytes() == xr.tobytes(), "int16_equal": y.tobytes() == yr.tobytes(),
                                      "max_abs_diff": max_diff(x, xr), "int16_max_diff": int(np.max(np.abs(y.astype(np.int32) - yr.astype(np.int32)))) if y.size else 0,
                                      "final_ranges_equal": [p["final_range"] for p in info["packets"]] == [p["final_range"] for p in info48["packets"]]}
                checks[label] = row
            if c <= 2:
                plain = x48 if info48["head"]["gain"] == 0 else results[(48000, 0)][0]   # opus_demo applies no output gain
                checks["opus_demo"] = opus_demo_check(opus_demo, info48, path, scratch, plain)
                for label, decoder in [("ffmpeg_native", "opus"), ("ffmpeg_libopus", "libopus")]:
                    z = ffmpeg_decode(path, c, decoder)
                    if z is not None:
                        checks[label] = {"samples": int(z.shape[1]), "length_equal": z.shape[1] == x48.shape[1], "max_abs_diff": max_diff(z, x48)}
            entry["checks"] = checks

        # 1. the encoded fixtures
        for fx in FIXTURES:
            name, s = fx["name"], dict(fx["settings"])
            rate = s.get("rate", 48000)
            channels = s.get("channels", 1)
            l48 = length_48k(fx["seconds"])
            n = l48 * rate // 48000
            kind = fx["signal"]
            silence = s.pop("silence", None)
            window = s.pop("window", None)
            if kind == "speech-silence":
                x = speech(rate, n, fx["seed"], silence=silence)
            elif kind == "speech":
                x = speech(rate, n, fx["seed"], channels=channels)
            elif kind == "music":
                x = music(rate, n, fx["seed"], channels=channels)
            else:
                x = SIGNALS[kind](rate, n, fx["seed"])
            assert x.shape == (channels, n)
            raw = os.path.join(scratch, name + ".f32")
            x.T.astype("<f4").tofile(raw)
            changes = s.pop("changes", [])
            comments = s.pop("comments", [])
            config = os.path.join(scratch, name + ".cfg")
            with open(config, "w", encoding="utf-8", newline="\n") as f:
                for k, v in s.items():
                    f.write(f"{k} {v}\n")
                f.write(f"length {n}\nserial {serial_of(name)}\ncomment TITLE={name}\ncomment ENCODER=tools/audio/make_opus_reference.py\n")
                for c in comments:
                    f.write(f"comment {c}\n")
                for at, k, v in changes:
                    f.write(f"at {int(round(at * rate))} {k} {v}\n")
            path = os.path.join(out, name + ".opus")
            enc = harness(ref, "encode", config, raw, path)
            if "error" in enc:
                sys.exit(f"{name}: {enc['error']}")
            entry = record_file(name + ".opus")
            entry.update({"kind": "encoded", "signal": kind, "seed": fx["seed"], "input_rate": rate, "input_samples": n, "input_sha256": sha256_bytes(x.tobytes()),
                          "settings": {k: v for k, v in s.items()}, **({"silence_seconds": list(silence)} if silence else {}), "changes": [[a, k, v] for a, k, v in changes], "lookahead": enc["lookahead"]})
            print(f"{name}: {len(enc['packets'])} packets, {entry['bytes']} bytes", flush=True)
            info48, results = full_decode(name, path, fx["npy"], [48000, 16000] + fx["extra"], window=window)
            dec_ranges = [p["final_range"] if isinstance(p["final_range"], int) else None for p in info48["packets"]]
            enc_ranges = [r for _, r in enc["packets"]]
            if channels <= 2 and dec_ranges != enc_ranges:
                bad = [i for i, (a, b) in enumerate(zip(enc_ranges, dec_ranges)) if a != b]
                sys.exit(f"{name}: the decoder's final range differs from the encoder's at packets {bad[:10]}")
            entry["encoder_final_ranges_equal"] = True if channels <= 2 else "per stream (multistream: the encoder gives their xor)"
            if channels > 2:
                xor = []
                for p in info48["packets"]:
                    v = 0
                    for r in p["final_range"]:
                        v ^= r
                    xor.append(v)
                if xor != enc_ranges:
                    sys.exit(f"{name}: the streams' final ranges do not xor to the encoder's")
                entry["encoder_final_ranges_equal"] = True
            decoded[name] = results
            comparisons(name, path, info48, results)
            if window:
                redundant_at = [sum(r[11] for r in entry["packets"]["rows"][:i]) for i, r in enumerate(entry["packets"]["rows"]) if r[15]]
                bandwidths = [r[4] for r in entry["packets"]["rows"]]
                switches = sum(1 for a, b in zip(bandwidths, bandwidths[1:]) if a != b)
                if switches < 2 or not redundant_at or min(redundant_at) - entry["head"]["pre_skip"] < window[0] * 48000:
                    sys.exit(f"{name}: {switches} bandwidth switches, redundant frames at 48 kHz samples {redundant_at}: not both inside the window")
                entry["bandwidth_switches"] = [[i + 1, a, b] for i, (a, b) in enumerate(zip(bandwidths, bandwidths[1:])) if a != b]
            modes = sorted({row[3] for row in entry["packets"]["rows"]})
            entry["modes"] = modes
            if name.startswith("refuse-"):
                entry["expected"] = f"NotSupportedException naming the {modes[0]} frames: the decoder reads SILK only"
                if modes != [name.split("-")[1]]:
                    sys.exit(f"{name}: its packets are {modes}, not {name.split('-')[1]} only")
            elif modes != ["silk"]:
                sys.exit(f"{name}: its packets are {modes}, not SILK only")

        # the output gain: silk-mb-20ms-mono's packets under an OpusHead whose gain is -1536 (-6 dB in Q7.8)
        name = "gain-minus6db-silk-mb-20ms-mono"
        sp = read_pages(open(os.path.join(out, "silk-mb-20ms-mono.opus"), "rb").read())
        h = bytearray(packets_of(sp[0])[0])
        h[16:18] = struct.pack("<h", -1536)
        path = os.path.join(out, name + ".opus")
        with open(path, "wb") as f:
            f.write(whole(sp, {0: rewrite_page(sp[0], [bytes(h)])}))
        entry = record_file(name + ".opus")
        entry.update({"kind": "variant", "source": "silk-mb-20ms-mono.opus", "what": "silk-mb-20ms-mono.opus with OpusHead's output gain -1536 (Q7.8: -6 dB); its pages otherwise the same"})
        info48, results = full_decode(name, path, [48000], [48000, 16000], gain_variants=(1, 0))
        decoded[name] = results
        comparisons(name, path, info48, results)
        for rate in (48000, 16000):
            if results[(rate, 0)][0].tobytes() != decoded["silk-mb-20ms-mono"][(rate, 1)][0].tobytes():
                sys.exit("the gain file decoded without its gain differs from its source")
        entry["without_gain_equals"] = "silk-mb-20ms-mono.opus"
        print(f"{name}: {entry['bytes']} bytes", flush=True)

        # the bandwidth switch without its redundant CELT frames: each packet that carries one cut where the frame's SILK data
        # ends (its length minus the redundant bytes), so libopus finds no redundancy: what a decoder that skips them gives
        name, source_name = "silk-bandwidth-switch-mono-stripped", "silk-bandwidth-switch-mono"
        src = files[source_name + ".opus"]
        fields = src["packets"]["fields"]
        cut = {}
        for i, r in enumerate(src["packets"]["rows"]):
            if r[fields.index("redundancy")]:
                (frame, _, red_bytes), = r[fields.index("redundancy")]
                if r[fields.index("code")] != 0 or frame != 0:
                    sys.exit(f"{name}: packet {i} is not a one-frame packet")
                keep = (r[fields.index("silk_tell")][0] + 1 + 7) >> 3
                if keep != r[fields.index("frames")][0] - red_bytes:
                    sys.exit(f"{name}: packet {i}: SILK ends at {keep} bytes, the frame less its redundancy is {r[fields.index('frames')][0] - red_bytes}")
                cut[i] = 1 + keep
        sp = read_pages(open(os.path.join(out, source_name + ".opus"), "rb").read())
        replace, at = {}, 0
        for k, pg in enumerate(sp[2:], start=2):
            pk = packets_of(pg)
            if any(at + j in cut for j in range(len(pk))):
                replace[k] = rewrite_page(pg, [q[:cut[at + j]] if at + j in cut else q for j, q in enumerate(pk)])
            at += len(pk)
        path = os.path.join(out, name + ".opus")
        with open(path, "wb") as f:
            f.write(whole(sp, replace))
        entry = record_file(name + ".opus")
        entry.update({"kind": "variant", "source": source_name + ".opus",
                      "what": "the source with each redundant CELT frame cut off (the packet ends where its SILK data ends): libopus then decodes no "
                              "redundancy, so this is what a SILK-only decoder that skips redundant frames gives for the source"})
        info48, results = full_decode(name, path, [], [48000, 16000])
        decoded[name] = results
        if entry["redundant_celt_frames"] != 0:
            sys.exit(f"{name}: redundancy left")
        differences = {}
        for rate in (48000, 16000):
            x, sx = results[(rate, 1)][0], decoded[source_name][(rate, 1)][0]
            where = np.nonzero(np.any(x != sx, axis=0))[0]
            differences[str(rate)] = {"samples": int(len(where)), "first": int(where[0]) if len(where) else None, "last": int(where[-1]) if len(where) else None,
                                      "max_abs_diff": max_diff(x, sx), "float_is_int16_exactly": bool(np.all(x.astype(np.float64) * 32768 == np.round(x.astype(np.float64) * 32768)))}
        entry["differs_from_source"] = differences
        print(f"{name}: differs from its source at 48 kHz in {differences['48000']['samples']} samples, by up to {differences['48000']['max_abs_diff']:.3g}", flush=True)

        # 2. libopus' repacketizer: the same frames in packets of codes 1-3 (their decodings are their sources')
        for name, source_name, group, pad in REPACKED:
            path = os.path.join(out, name + ".opus")
            rp = harness(ref, "repack", os.path.join(out, source_name + ".opus"), path, group, pad, serial_of(name))
            if "error" in rp:
                sys.exit(f"{name}: {rp['error']}")
            entry = record_file(name + ".opus")
            entry.update({"kind": "repacketized", "source": source_name + ".opus", "frames_per_packet": group if group > 0 else "each packet's frames twice",
                          "padding_per_packet": pad})
            if group < 0:
                entry["what"] = ("every packet of the source joined with itself by libopus' repacketizer: code 1 (two equal frames), a new stream "
                                 "twice as long (each frame decoded twice in a row), its end trimmed by as much as the source's")
                info48, results = full_decode(name, path, [48000], [48000, 16000])
                decoded[name] = results
                comparisons(name, path, info48, results)
                print(f"{name}: codes {sorted({row[7] for row in entry['packets']['rows']})}, {entry['bytes']} bytes", flush=True)
                continue
            info48, results = full_decode(name, path, [], [48000, 16000], base=source_name + ".opus")
            # The float samples must be the source's; opus_decode's int16 samples may not be where the decoding passes 1.0:
            # libopus soft-clips each opus_decode call's whole buffer (opus_pcm_soft_clip), so the packets' grouping moves
            # the clipping curve.
            for d in entry["decodes"]:
                x, y = results[(d["rate"], 1)]
                sx, sy = decoded[source_name][(d["rate"], 1)]
                if x.tobytes() != sx.tobytes():
                    sys.exit(f"{name}: its float decoding at {d['rate']} Hz differs from {source_name}'s")
                d["int16_equal_to_source"] = y.tobytes() == sy.tobytes()
                if not d["int16_equal_to_source"]:
                    d["int16_differences"] = {"samples": int(np.sum(y != sy)), "max": int(np.max(np.abs(y.astype(np.int32) - sy.astype(np.int32)))),
                                              "why": "opus_pcm_soft_clip runs over each packet's samples at once; the source's packets hold fewer frames"}
                    if d["clipped_samples"] == 0:
                        sys.exit(f"{name}: its int16 decoding at {d['rate']} Hz differs from {source_name}'s without clipping")
            codes = sorted({row[7] for row in entry["packets"]["rows"]})
            print(f"{name}: codes {codes}, {entry['bytes']} bytes", flush=True)

        # 3. variants and damaged copies, made here from the fixtures' pages
        def write(name, data, entry):
            with open(os.path.join(out, name), "wb") as f:
                f.write(data)
            e = record_file(name)
            e.update(entry)
            return e

        damaged_source = "silk-wb-40ms-mono"
        base = open(os.path.join(out, damaged_source + ".opus"), "rb").read()
        pages = read_pages(base)
        # a body byte of the fourth page flipped, its CRC left as it was
        p = pages[3]
        flipped = bytearray(base)
        flipped[p["offset"] + 27 + len(p["lacing"]) + 5] ^= 0x10
        write("bad-crc.opus", bytes(flipped), {"kind": "damaged", "source": damaged_source + ".opus", "what": "page 3: a body byte flipped (0x10), the CRC not updated"})
        # cut inside the last page's body
        last = pages[-1]
        cut = last["offset"] + 27 + len(last["lacing"]) + len(last["body"]) // 2
        write("bad-truncated.opus", base[:cut], {"kind": "damaged", "source": damaged_source + ".opus", "what": f"cut at byte {cut}, inside the last page's body"})
        # a packet with a code-1 TOC and an odd payload (two equal frames cannot fill it)
        target = 5
        index, at = None, 2
        for i, pg in enumerate(pages[2:], start=2):
            pk = packets_of(pg)
            if at <= target + 2 < at + len(pk):
                index, which, pk_list = i, target + 2 - at, pk
                break
            at += len(pk)
        pk = list(pk_list)
        q = bytes([pk[which][0] | 1]) + pk[which][1:]
        if (len(q) - 1) % 2 == 0:
            q += b"\x00"
        pk[which] = q
        write("bad-packet-code1-odd.opus", whole(pages, {index: rewrite_page(pages[index], pk)}),
              {"kind": "damaged", "source": damaged_source + ".opus", "what": f"packet {target}: TOC code 1 with an odd payload ({len(q) - 1} bytes): two equal frames cannot fill it"})
        # a code-3 packet of 4 frames of 40 ms: 160 ms, past the 120 ms a packet may hold
        pk = list(pk_list)
        frame = pk[which][1:]
        pk[which] = bytes([(pk[which][0] & 0xFC) | 3, 4]) + frame
        write("bad-packet-160ms.opus", whole(pages, {index: rewrite_page(pages[index], pk)}),
              {"kind": "damaged", "source": damaged_source + ".opus", "what": f"packet {target}: code 3 with 4 frames of 40 ms (160 ms; a packet holds at most 120 ms)"})
        # OpusHead with mapping family 255 (valid, not read)
        sp = read_pages(open(os.path.join(out, "silk-nb-10ms-mono.opus"), "rb").read())
        h = bytearray(packets_of(sp[0])[0])
        h[18] = 255
        h = bytes(h[:19]) + bytes([1, 0, 0])
        write("unsupported-family255.opus", whole(sp, {0: rewrite_page(sp[0], [h])}),
              {"kind": "variant", "source": "silk-nb-10ms-mono.opus", "what": "OpusHead's mapping family 255 (streams 1, coupled 0, mapping [0]): valid Ogg Opus, not read (NotSupportedException naming the family)"})
        # two links one after the other (chained), and two streams interleaved (grouped)
        a = open(os.path.join(out, "silk-mb-20ms-mono.opus"), "rb").read()
        b = open(os.path.join(out, "silk-nb-40ms-stereo.opus"), "rb").read()
        write("chained.opus", a + b, {"kind": "variant", "what": "two links: silk-mb-20ms-mono.opus then silk-nb-40ms-stereo.opus (each link decodes as its source)",
                                      "links": [{"serial": serial_of("silk-mb-20ms-mono"), "same_as": "silk-mb-20ms-mono.opus"}, {"serial": serial_of("silk-nb-40ms-stereo"), "same_as": "silk-nb-40ms-stereo.opus"}]})
        pa, pb = read_pages(open(os.path.join(out, "voicenote-16k-60ms.opus"), "rb").read()), read_pages(open(os.path.join(out, "silk-nb-10ms-mono.opus"), "rb").read())
        order = [pa[0], pb[0]]
        rest_a, rest_b = pa[1:], pb[1:]
        while rest_a or rest_b:
            if rest_a:
                order.append(rest_a.pop(0))
            if rest_b:
                order.append(rest_b.pop(0))
        write("multiplexed.opus", whole(order, {}), {"kind": "variant", "what": "two streams grouped (both BOS pages first, then their pages alternating): voicenote-16k-60ms.opus and silk-nb-10ms-mono.opus",
                                                     "links": [{"serial": serial_of("voicenote-16k-60ms"), "same_as": "voicenote-16k-60ms.opus"}, {"serial": serial_of("silk-nb-10ms-mono"), "same_as": "silk-nb-10ms-mono.opus"}]})
        # an Ogg Vorbis identification header: an Ogg file that is not Opus
        vorbis = b"\x01vorbis" + struct.pack("<IBIiiiBB", 0, 1, 44100, 0, 112000, 0, 0xB8, 1)
        write("not-opus-vorbis.ogg", page_bytes(2, 0, serial_of("vorbis"), 0, lacing_of([vorbis]), vorbis),
              {"kind": "variant", "what": "one BOS page holding a Vorbis identification header (mono, 44.1 kHz): an Ogg stream that is not Opus (ReadInfo answers null)"})

        # what the reference decoder says of each damaged or variant file, and the links' decodings
        for name in ["bad-crc.opus", "bad-truncated.opus", "bad-packet-code1-odd.opus", "bad-packet-160ms.opus", "unsupported-family255.opus", "not-opus-vorbis.ogg"]:
            info = decode(ref, os.path.join(out, name), 48000, scratch)[0]
            if "error" not in info:
                sys.exit(f"{name}: the reference decoded it")
            files[name]["reference_error"] = info["error"]
            print(f"{name}: {info['error']}", flush=True)
        for name in ["chained.opus", "multiplexed.opus"]:
            for link, l in enumerate(files[name]["links"]):
                info, x, y, _ = decode(ref, os.path.join(out, name), 48000, scratch, link=link)
                src = decoded[l["same_as"][:-5]][(48000, 1)]
                if "error" in info or x.tobytes() != src[0].tobytes() or info["serial"] != l["serial"]:
                    sys.exit(f"{name}: link {link} does not decode as {l['same_as']}")
                l["length_48k"] = info["length"]
            files[name]["links_checked"] = True

        # 4. the mode survey: libopus' encoder unforced (mode, bandwidth and signal type its own choice), VBR
        survey = []
        for signal, seed in SURVEY_SIGNALS:
            signals = {}
            for rate, app, kbps, frame in SURVEY:
                n = length_48k(SURVEY_SECONDS) * rate // 48000
                if rate not in signals:
                    signals[rate] = SIGNALS[signal](rate, n, seed)
                raw = os.path.join(scratch, f"survey-{rate}.f32")
                signals[rate].T.astype("<f4").tofile(raw)
                config = os.path.join(scratch, "survey.cfg")
                with open(config, "w", encoding="utf-8", newline="\n") as f:
                    f.write(f"rate {rate}\nchannels 1\napplication {app}\nframe {frame}\nbitrate {kbps * 1000}\nvbr 1\nlength {n}\nserial 1\n")
                path = os.path.join(scratch, "survey.opus")
                enc = harness(ref, "encode", config, raw, path)
                if "error" in enc:
                    sys.exit(f"survey: {enc['error']}")
                packets = ogg_packets(open(path, "rb").read())[2:]
                modes, bandwidths, both, frames = {}, {}, {}, {}
                others = []
                x = signals[rate][0].astype(np.float64)
                for i, p in enumerate(packets):
                    if MODE_NAMES[p[0] >> 3] != "silk":
                        # the input under this packet (the encoder's lookahead ahead of it): its peak level in dBFS
                        a = max(0, i * frame * rate // 1000 - enc["lookahead"])
                        b = min(len(x), a + frame * rate // 1000)
                        peak = float(np.max(np.abs(x[a:b]))) if b > a else 0.0
                        others.append([i, round(i * frame / 1000, 2), MODE_NAMES[p[0] >> 3], round(20 * math.log10(peak), 1) if peak > 0 else None])
                for p in packets:
                    config_ = p[0] >> 3
                    m, b = MODE_NAMES[config_], BANDWIDTH_NAMES[config_]
                    modes[m] = modes.get(m, 0) + 1
                    bandwidths[b] = bandwidths.get(b, 0) + 1
                    both[m + "-" + b] = both.get(m + "-" + b, 0) + 1
                    f_ms = FRAME_MS[config_]
                    frames[str(f_ms)] = frames.get(str(f_ms), 0) + 1
                tinfo = decode(traced, path, 48000, scratch)[0]
                redundant = [[i, t[4], t[5]] for i, p in enumerate(tinfo["packets"]) for t in p["trace"] if t[3]]
                survey.append({"signal": signal, "input_rate": rate, "application": app, "bitrate": kbps * 1000, "frame_ms": frame, "packets": len(packets),
                               "kbps": round(sum(len(p) for p in packets) * 8 / (len(packets) * frame) , 2), "silk_only": set(modes) == {"silk"},
                               "modes": modes, "bandwidths": bandwidths, "mode_bandwidth": both, "toc_frame_ms": frames,
                               "not_silk": others if len(others) <= 40 else f"{len(others)} packets",
                               "redundant_celt_frames": redundant if len(redundant) <= 40 else f"{len(redundant)} frames"})
                print(f"survey {signal} {rate} {app} {kbps}k {frame}ms: {both}", flush=True)

        # 5. range decoder traces
        symbols = {"note": "the first symbols of each packet's first frame, decoded with libopus' ec_dec as its SILK or CELT decoder reads them; "
                           "the state after each: rng, val (RFC 6716 4.1's), tell (ec_tell) and tell_frac (ec_tell_frac, 1/8 bits)", "files": {}}
        for name in SYMBOL_FILES:
            symbols["files"][name + ".opus"] = harness(ref, "symbols", os.path.join(out, name + ".opus"), 4)["packets"]
        with open(os.path.join(out, "symbols.json"), "w", encoding="utf-8", newline="\n") as f:
            f.write(dump(symbols) + "\n")
        rc = harness(ref, "rangecoder")
        rc["note"] = ("seeded symbol sequences encoded with libopus' ec_enc (ec_enc_bit_logp, ec_enc_icdf, ec_enc_uint, ec_enc_bits, ec_laplace_encode) "
                      "into a buffer of `storage` bytes (hex: the whole buffer, raw bits at its end), then decoded with ec_dec; the state after each symbol; "
                      "the decoder's final rng equals the encoder's (encoder_final_range)")
        with open(os.path.join(out, "range-coder.json"), "w", encoding="utf-8", newline="\n") as f:
            f.write(dump(rc) + "\n")
        for name in ["symbols.json", "range-coder.json"]:
            record_file(name)["kind"] = "debug"

        npys = sorted(n for n in os.listdir(out) if n.endswith(".npy"))
        for n in npys:
            e = record_file(n)
            e["kind"] = "decoded"
        manifest = {
            "libopus": {"repository": REPOSITORY, "tag": TAG, "commit": commit_of(source), "version": files["silk-nb-10ms-mono.opus"]["vendor"],
                        "build": ref_build, "fixed_point_build": fixed_build},
            "compare": {"libopus": {"tag": COMPARE_TAG, "commit": commit_of(compare_source), "build": older_build},
                        "ffmpeg": run(["ffmpeg", "-version"], capture_output=True, text=True).stdout.splitlines()[0] if shutil.which("ffmpeg") else None},
            "compiler": compiler,
            "machine": platform.machine(),
            "numpy": np.__version__,
            "harness": "tools/audio/opus_reference.c",
            "conventions": {
                "samples": "the .npy files are planar [channels, samples] in OpusHead's channel order; npy_dtype float32 is opus_decode_float's output as is; "
                           "int16 is used only where every float sample is a multiple of 1/32768 (SILK frames without redundant CELT frames): divide by 32768 "
                           "for the float samples, exactly; a decode with npy_window holds only samples [start, end) of the decoding (its hashes are the whole's)",
                "trimming": "pre-skip x rate / 48000 samples dropped from the start; (last granule - pre-skip) x rate / 48000 kept (every fixture divides exactly: exact_scaling)",
                "gain": "OpusHead's output gain applied inside libopus (OPUS_SET_GAIN), as opusfile does: every sample x 10^(gain / 5120), in float; gain_applied false decodes ignore it",
                "hashes": "float32_sha256 and int16_sha256 are of the planar little-endian samples (the .npy's data bytes for its dtype)",
                "int16": "opus_decode's output (the float build: the float samples soft-clipped by opus_pcm_soft_clip, then rounded to nearest-even and saturated)",
                "packets": "rows of fields: page (the page the packet ends on), bytes, TOC config/mode/bandwidth/frame_ms/stereo/code, vbr (code 2, or code 3's flag), "
                           "padding bytes (code 3), frame sizes (opus_packet_parse), samples at 48 kHz, final_range (OPUS_GET_FINAL_RANGE after the packet; a list "
                           "per stream for a multistream), frame_ranges (packets of several frames: each frame decoded alone as its own code-0 packet, its final "
                           "range; the last is the packet's; the samples are the packet's, checked), silk_tell (per frame: ec_tell in bits where the SILK part ended, as "
                           "libopus' decoder found it; -1 for a frame without data), redundancy ([frame, 1 if CELT to SILK else 0, bytes] for each frame that "
                           "carries a redundant CELT frame: in a SILK-only frame, 17 or more bits left after SILK mean one; both from a traced copy of "
                           "libopus' decoder whose samples are checked equal)",
                "pages": "rows: byte offset, size, header flags (1 continued, 2 BOS, 4 EOS), granule, serial, sequence, lacing values, packets ending on it",
                "checks": "fixed_point and libopus 1.5.2: the same file through those builds (float32_equal, max_abs_diff); opus_demo: libopus' tool on the packets "
                          "with their final ranges (it stops on a mismatch), its 24-bit output (opus_decode24, no output gain) against the reference; ffmpeg_native: "
                          "ffmpeg's own decoder (another implementation: its SILK is not libopus', so its samples are not close); ffmpeg_libopus: the system's libopus "
                          "(another build, with its SIMD paths) through ffmpeg, as float",
            },
            "files": files,
            "mode_survey": {
                "what": "libopus' encoder unforced (no mode, bandwidth or signal type set), mono, VBR, complexity 10, on 3 s of the speech-like signal and "
                        "of it over a steady background, from 16 kHz input (OpusHead input rate 16000, as voice notes) and 48 kHz; per encoding the packets "
                        "counted by their TOC's mode, bandwidth and both; kbps is the packets' mean rate; not_silk: [packet, its time in s, mode, the input's "
                        "peak in dBFS under it] for each packet that is not SILK (up to 40); redundant_celt_frames: [packet, 1 if CELT to SILK, bytes] for each frame "
                        "carrying a redundant CELT frame (RFC 6716 4.5.1). Not kept as files.",
                "rows": survey,
            },
        }
        with open(os.path.join(out, "opus.json"), "w", encoding="utf-8", newline="\n") as f:
            f.write(dump(manifest) + "\n")
        total = sum(os.path.getsize(os.path.join(out, n)) for n in os.listdir(out))
        print(f"wrote {out}: {len(files)} files, {total} bytes in all", flush=True)
    finally:
        if not args.keep and not args.work:
            shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
