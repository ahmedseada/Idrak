"""The constant tables of the MP3 plug-in's encoder (samples/Mp3Audio), derived so no value is typed by hand:

- the Huffman codes (ISO/IEC 11172-3 Annex B, tables 1-24 and the count1 table A), read back from the decoder's tables
  in Mp3Tables.cs (generated from minimp3 by make_mp3_tables.py) by decoding every bit pattern: each table must come
  out a complete prefix code with xlen * xlen entries;
- the analysis window C[i] = D[i] / 32 (11172-3 Table 3-C.1 and 3-B.3), from the synthesis window the decoder holds
  (minimp3's g_win and the constants its synth pair inlines) with D's symmetry D[512 - i] = -D[i] (i not a multiple of
  64); checked: the standard's analysis and synthesis filterbanks reconstruct white noise above 80 dB at a delay of 481;
- the matrixing cosines, the MDCT (sine window, 1/9 scale), the aliasing butterflies (from the standard's c[i]), 2^(k/16);
  the whole hybrid (polyphase, MDCT, butterflies) checked against the standard's decoder side above 80 dB at 1057;
- a small psychoacoustic model per sample rate: the spreading function across the scalefactor bands (Schroeder's, as
  in model 2) with an offset of 10 + 0.4 z dB, and the absolute threshold of hearing (Terhardt's) with full scale taken
  as 90 dB SPL, in the encoder's MDCT energy units.

    python tools/audio/make_mp3_encoder_tables.py

Writes samples/Mp3Audio/Idrak.Mp3Audio/Mp3EncoderTables.cs. Needs numpy.
"""
import math
import os
import re
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PLUGIN = os.path.join(ROOT, "samples", "Mp3Audio", "Idrak.Mp3Audio")
RATES = [44100, 48000, 32000, 22050, 24000, 16000]           # the encoder's rate index
SCF_ROW = {44100: 5, 48000: 6, 32000: 7, 22050: 2, 24000: 3, 16000: 4}   # minimp3's g_scf_long rows
ATH_FULL_SCALE_DB = 90.0


def read_array(source, name):
    m = re.search(r"\b" + name + r" =\s*\[(.*?)\];", source, re.S)
    if not m:
        sys.exit(f"{name} not found in Mp3Tables.cs")
    return [float(v.rstrip("f")) for v in re.findall(r"-?\d+(?:\.\d+)?(?:e-?\d+)?f?", m.group(1))]


def huffman(tabs, tabindex, tab32):
    """Every pair table's (code, length) by (x, y), and count1 table A's by v*8 + w*4 + x*2 + y."""
    def pair(codebook, cache):
        w, used = 5, 0
        leaf = tabs[codebook + (cache >> (32 - w))]
        while leaf < 0:
            cache = (cache << w) & 0xFFFFFFFF
            used += w
            w = leaf & 7
            leaf = tabs[codebook + (cache >> (32 - w)) - (leaf >> 3)]
        return leaf & 15, (leaf >> 4) & 15, used + (leaf >> 8)

    xlen = {1: 2, 2: 3, 3: 3, 5: 4, 6: 4, 7: 6, 8: 6, 9: 6, 10: 8, 11: 8, 12: 8, 13: 16, 15: 16, 16: 16, 24: 16}
    tables = {}
    for t, n in xlen.items():
        codes, p, bits = {}, 0, 24
        while p < (1 << bits):
            x, y, length = pair(int(tabindex[t]), p << (32 - bits))
            code = p >> (bits - length)
            if (x, y) in codes or x >= n or y >= n:
                sys.exit(f"table {t}: ({x}, {y}) decoded twice or outside {n} x {n}")
            codes[(x, y)] = (code, length)
            p = (code + 1) << (bits - length)
        if len(codes) != n * n or abs(sum(2.0 ** -l for _, l in codes.values()) - 1) > 1e-12:
            sys.exit(f"table {t}: {len(codes)} codes, not a complete code of {n * n}")
        tables[t] = (n, codes)

    count1, p, bits = {}, 0, 8
    while p < (1 << bits):
        cache = p << (32 - bits)
        leaf = tab32[cache >> 28]
        if (leaf & 8) == 0:
            leaf = tab32[(leaf >> 3) + (((cache << 4) & 0xFFFFFFFF) >> (32 - (leaf & 3)))]
        length, v = leaf & 7, (leaf >> 4) & 15
        code = p >> (bits - length)
        if v in count1:
            sys.exit("count1 table A: a value decoded twice")
        count1[v] = (code, length)
        p = (code + 1) << (bits - length)
    if len(count1) != 16:
        sys.exit("count1 table A: not 16 codes")
    return tables, count1


def synthesis_window(win, decoder):
    """The standard's D[512] (units of 2^-16) from minimp3's g_win and the two halves of its synth pair."""
    body = re.search(r"static void SynthPair\(.*?\n    \}", decoder, re.S)
    if not body:
        sys.exit("SynthPair not found in Mp3Decoder.cs")
    halves = body.group(0).split("zAt += 2;")

    def terms(text):
        out = []
        for m in re.finditer(r"a \+?= (.*?);", text):
            e = m.group(1)
            paired = re.match(r"\((.*)\) \* (-?\d+)$", e)
            if paired:
                idx = [int(i) for i in re.findall(r"z\[zAt \+ (\d+)", paired.group(1))]
                op = re.findall(r"\] ([+-]) z", paired.group(1))[0]
                c = int(paired.group(2))
                out += [(idx[0], c), (idx[1], c if op == "+" else -c)]
            else:
                single = re.match(r"z\[zAt \+ (\d+)(?: \* 64)?\] \* (-?\d+)$", e)
                out.append((int(single.group(1)), int(single.group(2))))
        return out

    d = np.full(512, np.nan)
    d[0] = 0
    for z, c in terms(halves[0]):
        d[32 * (z + 1)] = c                                   # D[32 m], m = 1..15
    for z, c in terms(halves[1]):
        d[16 + 32 * z] = c                                    # D[16 + 64 n]
    for r in range(15):                                       # rows j = 1..15: D[j + 32 k] and D[j + 32 (15 - k)]
        j = r + 1
        for k in range(8):
            d[j + 32 * k] = win[r * 16 + 2 * k]
            d[j + 32 * (15 - k)] = win[r * 16 + 2 * k + 1]
    for n in range(8):
        d[48 + 64 * n] = -d[464 - 64 * n]                     # weight zero in both filterbanks; by the symmetry
    for j in range(17, 32):
        for m in range(16):
            d[j + 32 * m] = -d[(32 - j) + 32 * (15 - m)]
    if np.isnan(d).any():
        sys.exit("D has gaps")
    for m in range(1, 16):
        n = 32 * m
        if d[512 - n] != (1 if n % 64 == 0 else -1) * d[n]:
            sys.exit(f"D[{512 - n}] breaks the window's symmetry")
    return d / 65536.0


def check_filterbanks(d, cs, ca, mdct):
    """The standard's filterbanks (11172-3 Annex C analysis, 2.4.3.2 synthesis) and the hybrid; returns (delay, snr, k)."""
    c = d / 32
    n_mat = np.array([[math.cos((16 + i) * (2 * k + 1) * math.pi / 64) for k in range(32)] for i in range(64)])
    m_mat = np.array([[math.cos((2 * i + 1) * (k - 16) * math.pi / 64) for k in range(64)] for i in range(32)])

    def analysis(x):
        xp = np.concatenate([np.zeros(512), x])
        out = []
        for t in range(len(x) // 32):
            z = c * xp[512 + 32 * t + 31 - np.arange(512)]
            out.append(m_mat @ np.array([z[k::64].sum() for k in range(64)]))
        return np.array(out)

    def synthesis(s):
        v, out = np.zeros(1024), []
        for row in s:
            v = np.roll(v, 64)
            v[:64] = n_mat @ row
            u = np.zeros(512)
            for i in range(8):
                u[i * 64:i * 64 + 32] = v[i * 128:i * 128 + 32]
                u[i * 64 + 32:i * 64 + 64] = v[i * 128 + 96:i * 128 + 128]
            w = u * d
            out.append(np.array([w[j::32].sum() for j in range(32)]))
        return np.concatenate(out)

    def encode(x):
        s = analysis(x)
        for sb in range(1, 32, 2):
            s[1::2, sb] *= -1
        prev, grs = np.zeros((32, 18)), []
        for g in range(s.shape[0] // 18):
            cur = s[18 * g:18 * g + 18, :].T
            xr = (np.concatenate([prev, cur], axis=1) @ mdct).reshape(576)
            prev = cur
            for sb in range(1, 32):
                for i in range(8):
                    bu, bd = xr[18 * sb - 1 - i], xr[18 * sb + i]
                    xr[18 * sb - 1 - i], xr[18 * sb + i] = bu * cs[i] + bd * ca[i], bd * cs[i] - bu * ca[i]
            grs.append(xr)
        return grs

    def decode(grs):
        window = np.sin(np.pi / 36 * (np.arange(36) + 0.5))
        cos = np.array([[math.cos(math.pi / 72 * (2 * n + 19) * (2 * k + 1)) for n in range(36)] for k in range(18)])
        prev, s = np.zeros((32, 18)), []
        for xr in grs:
            xr = xr.copy()
            for sb in range(1, 32):
                for i in range(8):
                    bu, bd = xr[18 * sb - 1 - i], xr[18 * sb + i]
                    xr[18 * sb - 1 - i], xr[18 * sb + i] = bu * cs[i] - bd * ca[i], bd * cs[i] + bu * ca[i]
            z = (xr.reshape(32, 18) @ cos) * window
            s.append((z[:, :18] + prev).T)
            prev = z[:, 18:]
        s = np.concatenate(s, axis=0)
        for sb in range(1, 32, 2):
            s[1::2, sb] *= -1
        return synthesis(s)

    def snr(x, y, delay):
        a, b = x[:len(x) - delay], y[delay:]
        return 10 * math.log10(np.sum(a * a) / np.sum((b - a) ** 2))

    rng = np.random.default_rng(1)
    x = rng.standard_normal(32 * 160) * 0.3
    y = synthesis(analysis(x))
    if snr(x, y, 481) < 80:
        sys.exit(f"the polyphase filterbanks reconstruct at {snr(x, y, 481):.1f} dB, not above 80")
    x = rng.standard_normal(576 * 16) * 0.3
    grs = encode(x)
    y = decode(grs)
    hybrid = snr(x, y, 1057)
    if hybrid < 80:
        sys.exit(f"the hybrid filterbank reconstructs at {hybrid:.1f} dB, not above 80")
    k = float(np.mean(np.concatenate(grs[2:]) ** 2) / np.mean(x * x))   # MDCT energy per unit of signal power
    return hybrid, k


def bark(f):
    return 13 * math.atan(0.00076 * f) + 3.5 * math.atan((f / 7500) ** 2)


def ath_db(f):
    f = max(f, 20.0) / 1000
    return 3.64 * f ** -0.8 - 6.5 * math.exp(-0.6 * (f - 3.3) ** 2) + 1e-3 * f ** 4


def psychoacoustics(scf_long, rate, k):
    widths = scf_long[SCF_ROW[rate] * 23:SCF_ROW[rate] * 23 + 22]
    if sum(widths) != 576:
        sys.exit(f"{rate} Hz: the band widths sum to {sum(widths)}")
    starts = np.concatenate([[0], np.cumsum(widths)])
    hz = rate / 2 / 576
    z = [bark((starts[b] + starts[b + 1]) / 2 * hz) for b in range(22)]
    spread = np.zeros((22, 22))
    for b in range(22):
        offset = 10 ** (-(10 + 0.4 * z[b]) / 10)
        for j in range(22):
            dz = z[b] - z[j] + 0.474
            sf = 15.81 + 7.5 * dz - 17.5 * math.sqrt(1 + dz * dz)
            spread[b, j] = 10 ** (sf / 10) * offset if sf > -60 else 0.0
    ath = [k * 576 * 0.5 * 10 ** ((min(ath_db((i + 0.5) * hz) for i in range(int(starts[b]), int(starts[b + 1]))) - ATH_FULL_SCALE_DB) / 10)
           for b in range(22)]
    return spread.reshape(-1), ath


def floats(values):
    return [str(np.float32(v)) for v in values]                # the shortest digits that give the same float32


def cs_array(name, kind, values, doc):
    if kind == "float":
        text = []
        for v in floats(values):
            v = v if ("e" in v or "." in v) else v + ".0"
            text.append(v + "f")
    else:
        text = [str(int(v)) for v in values]
    rows = [", ".join(text[i:i + 12]) for i in range(0, len(text), 12)]
    body = ",\n        ".join(rows)
    return f"    /// <summary>{doc}</summary>\n    internal static readonly {kind}[] {name} =\n    [\n        {body},\n    ];\n"


def main():
    tables_cs = open(os.path.join(PLUGIN, "Mp3Tables.cs"), encoding="utf-8").read()
    decoder_cs = open(os.path.join(PLUGIN, "Mp3Decoder.cs"), encoding="utf-8").read()
    tabs = [int(v) for v in read_array(tables_cs, "Tabs")]
    tabindex = [int(v) for v in read_array(tables_cs, "TabIndex")]
    tab32 = [int(v) for v in read_array(tables_cs, "Tab32")]
    win = read_array(tables_cs, "Win")
    scf_long = [int(v) for v in read_array(tables_cs, "ScfLong")]

    pairs, count1 = huffman(tabs, tabindex, tab32)
    d = synthesis_window(win, decoder_cs)
    c = d / 32
    # The window arranged for a forward walk of the newest 512 samples a[0..511] (a[511] the newest): the standard's
    # Y[63 - k] = sum over j of W[64 j + k] * a[64 j + k], W[64 j + k] = C[63 - k + 64 (7 - j)].
    window = [c[63 - k + 64 * (7 - j)] for j in range(8) for k in range(64)]
    matrix = [math.cos((2 * i + 1) * dd * math.pi / 64) for dd in range(32) for i in range(32)]   # [d][i], the folded M
    ci = [-0.6, -0.535, -0.33, -0.185, -0.095, -0.041, -0.0142, -0.0037]
    cs = [1 / math.sqrt(1 + x * x) for x in ci]
    ca = [x / math.sqrt(1 + x * x) for x in ci]
    aa = read_array(tables_cs, "Aa")
    if max(abs(a - b) for a, b in zip(aa, cs + [-x for x in ca])) > 1e-7:              # minimp3 holds -ca
        sys.exit("the aliasing butterflies disagree with the decoder's")
    mdct = np.array([[math.sin(math.pi / 36 * (n + 0.5)) * math.cos(math.pi / 72 * (2 * n + 19) * (2 * k + 1)) / 9
                      for k in range(18)] for n in range(36)])                          # [n][k]
    hybrid, k = check_filterbanks(d, cs, ca, mdct)
    print(f"hybrid filterbank: {hybrid:.1f} dB at 1057 samples; MDCT energy {k:.6f} per unit of power")

    order = [1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 24]
    offsets, codes, lengths, at = [], [], [], 0
    for t in order:
        n, table = pairs[t]
        offsets.append(at)
        for x in range(n):
            for y in range(n):
                codes.append(table[(x, y)][0])
                lengths.append(table[(x, y)][1])
        at += n * n
    psy_spread, psy_ath = [], []
    for rate in RATES:
        spread, ath = psychoacoustics(scf_long, rate, k)
        psy_spread += list(spread)
        psy_ath += ath

    out = [
        cs_array("HuffmanTables", "byte", order, "The pair tables held, in order (16 stands for 16-23, 24 for 24-31: their codes are the same)."),
        cs_array("HuffmanOffsets", "int", offsets, "Where each held table's codes start in <see cref=\"HuffmanCodes\"/>, x * xlen + y after it."),
        cs_array("HuffmanCodes", "int", codes, "The pair codes (11172-3 Annex B, Table B.7), by table then x * xlen + y."),
        cs_array("HuffmanLengths", "byte", lengths, "The pair codes' lengths in bits."),
        cs_array("Count1Codes", "byte", [count1[v][0] for v in range(16)], "Count1 table A's codes by v * 8 + w * 4 + x * 2 + y (table B is 15 - that value in 4 bits)."),
        cs_array("Count1Lengths", "byte", [count1[v][1] for v in range(16)], "Count1 table A's code lengths."),
        cs_array("AnalysisWindow", "float", window, "The analysis window C (11172-3 Table 3-C.1) arranged for a forward walk of the newest 512 samples: Y[63 - k] = sum over j of W[64 j + k] * a[64 j + k]."),
        cs_array("AnalysisMatrix", "float", matrix, "cos((2 i + 1) d pi / 64) by [d][i]: the matrixing of the folded Y (d 0..31) into the 32 subbands."),
        cs_array("MdctMatrix", "float", list(mdct.reshape(-1)), "The long-block MDCT by [n][k]: sin(pi / 36 (n + 1/2)) cos(pi / 72 (2 n + 19)(2 k + 1)) / 9."),
        cs_array("AliasCs", "float", cs, "The aliasing butterflies' cs[i] (11172-3 Table 3-B.9)."),
        cs_array("AliasCa", "float", ca, "The aliasing butterflies' ca[i]."),
        cs_array("Pow2Sixteenths", "float", [2 ** (i / 16) for i in range(16)], "2^(i / 16): the quantizer's steps."),
        cs_array("PsySpread", "float", psy_spread, "Per rate (44.1, 48, 32, 22.05, 24, 16 kHz) the 22 x 22 spreading of band energies into a band's masking threshold, the offset included."),
        cs_array("PsyAth", "float", psy_ath, "Per rate the 22 bands' absolute threshold of hearing, as MDCT energy."),
    ]
    cs_text = ("// Copyright (c) 2026 Ahmed Seada\n"
               "// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.\n\n"
               "// Generated by tools/audio/make_mp3_encoder_tables.py from the decoder's tables (Mp3Tables.cs) and the\n"
               "// formulas of ISO/IEC 11172-3: do not edit by hand.\n\n"
               "namespace Idrak.Mp3Audio;\n\n"
               "internal static class Mp3EncoderTables\n{\n" + "\n".join(out) + "}\n")
    path = os.path.join(PLUGIN, "Mp3EncoderTables.cs")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(cs_text)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
