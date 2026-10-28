#!/usr/bin/env python3
# Makes tests/Idrak.Tests.Library/Gcn/GcnInterpreter.Opcodes.cs: the gfx9 opcode table of the GCN interpreter's decoder
# (plan 35, G1b), read from what LLVM's disassembler prints for probe words. Development only: the tests never call
# LLVM, and the table is not the encoder's (the interpreter checks the encoder, so it keeps its own reading of the ISA).
#
#   python3 -I tests/Idrak.Tests/data/gcn/interpreter/make_opcode_table.py
#
# For every encoding and opcode, probe words are made with distinct register numbers in each operand field (a field may
# also be left zero, since LLVM refuses an unused field that is not), disassembled with llvm-mc for each CPU, and the
# probe that LLVM reads with the most fields gives the opcode's operand template: which fields it prints, in what
# order, of what width and register class. Made with Ubuntu LLVM 18.1.3.
import itertools
import os
import re
import struct
import subprocess
import sys

CPUS = ["gfx900", "gfx906", "gfx908", "gfx90c"]
LITERAL = 0x000000F0
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "..", "..", "..", "Idrak.Tests.Library", "Gcn", "GcnInterpreter.Opcodes.cs")


def le(words):
    return ",".join("0x%02x" % b for w in words for b in struct.pack("<I", w & 0xFFFFFFFF))


def disassemble(cpu, probes):
    """probes: list of word lists. Returns, per probe, (text, bytes consumed) or None."""
    lines = [le(p + [LITERAL]) for p in probes]
    run = subprocess.run(["llvm-mc", "--disassemble", "-triple=amdgcn", "-mcpu=" + cpu, "-show-encoding"],
                         input="\n".join(lines) + "\n", capture_output=True, text=True)
    warnings = set()
    for m in re.finditer(r"<stdin>:(\d+):(\d+): warning", run.stderr):
        warnings.add((int(m.group(1)), int(m.group(2))))
    decoded = []
    for line in run.stdout.splitlines():
        line = line.strip()
        if not line or line.startswith("."):
            continue
        m = re.match(r"(.*?)\s*; encoding: \[(.*)\]$", line)
        if not m:
            raise SystemExit("unexpected line: " + line)
        decoded.append((m.group(1).strip(), len(m.group(2).split(","))))
    results = []
    at = 0
    for number, probe in enumerate(probes, start=1):
        total = 4 * len(probe) + 4
        position = 0
        first = None
        while position < total:
            column = 1 + 5 * position
            if (number, column) in warnings:
                if position == 0:
                    first = False
                position += 4
                continue
            text, size = decoded[at]
            at += 1
            if position == 0:
                first = (text, size)
            position += size
        results.append(first if first else None)
    if at != len(decoded):
        raise SystemExit("alignment lost for " + cpu)
    return results


def split(text, separator):
    """Splits at the separator outside brackets and quotes, trimmed, empty parts dropped."""
    parts, depth, quoted, start = [], 0, False, 0
    for i, ch in enumerate(text):
        if ch == '"':
            quoted = not quoted
        elif not quoted and ch in "([":
            depth += 1
        elif not quoted and ch in ")]":
            depth -= 1
        elif not quoted and depth == 0 and ch == separator:
            parts.append(text[start:i])
            start = i + 1
    parts.append(text[start:])
    return [p.strip() for p in parts if p.strip()]


REG = re.compile(r"^(-?\|?)?([vsa])(?:(\d+)|\[(\d+):(\d+)\])\|?$")


def template(text, fields, sizes, consumed, base):
    """Maps a probe's printed operands to fields. fields: {start number: field name}. Returns (name, tokens) or None."""
    if "Invalid" in text or "<unknown>" in text or "/*" in text:
        return None
    parts = text.split(" ", 1)
    name = parts[0]
    rest = parts[1] if len(parts) > 1 else ""
    operands = split(rest, ",") if rest else []
    tokens = []
    modifiers = []
    if operands:
        last = split(operands[-1], " ")
        operands[-1] = last[0]
        modifiers = last[1:]
        # A hwreg(a, b, c) operand's own spaces.
        if operands[-1].count("(") > operands[-1].count(")"):
            operands[-1] = " ".join(last)
            modifiers = []
    for o in operands:
        m = REG.match(o)
        if m:
            cls = m.group(2)
            if m.group(3) is not None:
                start, width = int(m.group(3)), 1
            else:
                start, width = int(m.group(4)), int(m.group(5)) - int(m.group(4)) + 1
            if start not in fields:
                return None
            tokens.append("%s:%s%d" % (fields[start], cls, width))
        elif o in ("vcc", "exec", "off", "m0", "vcc_lo", "exec_lo", "scc"):
            tokens.append("=" + o)
        elif o.startswith("hwreg("):
            tokens.append("K:hwreg")
        elif o.startswith("sendmsg("):
            tokens.append("K:sendmsg")
        elif o.startswith("gpr_idx("):
            tokens.append("S1:gpridx")
        elif o in ("0x1234", "0xf0", "0x1"):
            if consumed > base and o == "0xf0":
                tokens.append("K:lit")
            elif o == "0x1":
                tokens.append("O:hex")
            else:
                tokens.append("K:hex16")
        elif re.match(r"^-?\d+$", o):
            tokens.append("K:dec")
        elif o.startswith("vmcnt("):
            tokens.append("K:waitcnt")
            modifiers = []
        else:
            return None
    for mod in modifiers:
        if mod.startswith("offset:") or mod.startswith("offset0:") or mod.startswith("offset1:"):
            kind = "O:8x2" if mod.startswith("offset0") or mod.startswith("offset1") else ("O:swz" if "swizzle" in mod else "O:16")
            if kind not in tokens:
                tokens.append(kind)
        elif mod.startswith("swizzle") or mod.startswith("offset:swizzle"):
            tokens.append("O:swz")
        elif mod.startswith("op_sel_hi:"):
            tokens.append("mix")
        elif mod in ("glc", "slc"):
            pass
        elif mod.startswith("vmcnt") or mod.startswith("expcnt") or mod.startswith("lgkmcnt"):
            pass
        else:
            return None
    return name, tokens


def probe_sets(encoding, op):
    """Yields (words, fields, base size) probes for one opcode."""
    if encoding == "sop1":
        for d, s in itertools.product([10, None], [20, None]):
            w = 0xBE800000 | ((d or 0) << 16) | (op << 8) | (s or 0)
            yield [w], {k: v for k, v in ((10, "D"), (20, "S0")) if (k == 10 and d) or (k == 20 and s)}, 4
    elif encoding == "sop2":
        for d, a, b in itertools.product([10, None], [20, None], [30, None]):
            w = 0x80000000 | (op << 23) | ((d or 0) << 16) | ((b or 0) << 8) | (a or 0)
            yield [w], dict(x for x in ((10, "D"), (20, "S0"), (30, "S1")) if x[0] in (d, a, b)), 4
    elif encoding == "sopk":
        for d in (10, None):
            w = 0xB0000000 | (op << 23) | ((d or 0) << 16) | 0x1234
            yield [w], ({10: "D"} if d else {}), 4
    elif encoding == "sopc":
        for a, b in itertools.product([20, None], [30, None]):
            w = 0xBF000000 | (op << 16) | ((b or 0) << 8) | (a or 0)
            yield [w], dict(x for x in ((20, "S0"), (30, "S1")) if x[0] in (a, b)), 4
    elif encoding == "sopp":
        for k in (3, 0):
            yield [0xBF800000 | (op << 16) | k], {}, 4
    elif encoding == "smem":
        for d, b in itertools.product([40, None], [10, None]):
            w = 0xC0000000 | (op << 18) | (1 << 17) | ((d or 0) << 6) | (b or 0)
            yield [w, 1], dict(x for x in ((40, "D"), (20, "B")) if (x[0] == 40 and d) or (x[0] == 20 and b)), 8
            w = 0xC0000000 | (op << 18) | ((d or 0) << 6) | (b or 0)
            yield [w, 0], dict(x for x in ((40, "D"), (20, "B")) if (x[0] == 40 and d) or (x[0] == 20 and b)), 8
    elif encoding == "vop1":
        for d, s in itertools.product([10, None], [276, 20, None]):
            w = 0x7E000000 | ((d or 0) << 17) | (op << 9) | (s or 0)
            yield [w], dict(x for x in ((10, "D"), (20, "S0")) if (x[0] == 10 and d) or (x[0] == 20 and s)), 4
    elif encoding == "vop2":
        for d, s in itertools.product([10, None], [276, 20, None]):
            w = (op << 25) | ((d or 0) << 17) | (30 << 9) | (s or 0)
            yield [w], dict(x for x in ((10, "D"), (20, "S0"), (30, "S1")) if (x[0] == 10 and d) or (x[0] == 20 and s) or x[0] == 30), 4
    elif encoding == "vopc":
        for s in (276, 20):
            w = 0x7C000000 | (op << 17) | (30 << 9) | s
            yield [w], {20: "S0", 30: "S1"}, 4
    elif encoding in ("vop3", "vop3p"):
        choices = [256 + 20, 20, None]
        for d, sd, a, b, c in itertools.product([10, None], [50, None] if encoding == "vop3" else [None], choices, [256 + 30, 30, None], [256 + 40, 40, None]):
            if encoding == "vop3":
                w0 = 0xD0000000 | (op << 16) | ((sd or 0) << 8) | (d or 0)
                w1 = (a or 0) | ((b or 0) << 9) | ((c or 0) << 18)
            else:
                w0 = 0xD3800000 | (op << 16) | (1 << 14) | (d or 0)
                w1 = (a or 0) | ((b or 0) << 9) | ((c or 0) << 18) | (3 << 27)
            fields = {}
            if d:
                fields[10] = "D"
            if sd:
                fields[50] = "SD"
            if a:
                fields[20] = "S0"
            if b:
                fields[30] = "S1"
            if c:
                fields[40] = "S2"
            yield [w0, w1], fields, 8
    elif encoding == "ds":
        for d, a, x, y in itertools.product([10, None], [20, None], [30, None], [40, None]):
            w0 = 0xD8000000 | (op << 17) | 1 | (2 << 8)
            w1 = (a or 0) | ((x or 0) << 8) | ((y or 0) << 16) | ((d or 0) << 24)
            yield [w0, w1], dict(f for f in ((10, "D"), (20, "A"), (30, "S0"), (40, "S1")) if f[0] in (d, a, x, y)), 8
    elif encoding in ("flat", "global", "scratch"):
        segment = {"flat": 0, "scratch": 1, "global": 2}[encoding]
        for glc in (0, 1):
            for d, a, x in itertools.product([10, None], [20, None], [30, None]):
                w0 = 0xDC000000 | (op << 18) | (segment << 14) | (glc << 16) | 1
                w1 = (a or 0) | ((x or 0) << 8) | ((0x7F if segment else 0) << 16) | ((d or 0) << 24)
                yield [w0, w1], dict(f for f in ((10, "D"), (20, "A"), (30, "S0")) if f[0] in (d, a, x)), 8 + glc * 1000


ENCODINGS = [("sop1", 256), ("sop2", 0x60), ("sopk", 0x1D), ("sopc", 128), ("sopp", 128), ("smem", 256), ("vop1", 256), ("vop2", 0x3E),
             ("vopc", 256), ("vop3", 0x380), ("vop3p", 128), ("ds", 256), ("flat", 128), ("global", 128), ("scratch", 128)]


def table(cpu):
    rows = {}
    for encoding, count in ENCODINGS:
        probes = []
        owners = []
        for op in range(count):
            for words, fields, base in probe_sets(encoding, op):
                probes.append(words)
                owners.append((op, fields, base))
        results = disassemble(cpu, probes)
        best = {}
        for (op, fields, base), result in zip(owners, results):
            if not result:
                continue
            text, consumed = result
            glc = base >= 1000
            base %= 1000
            if consumed < base:
                continue
            got = template(text, fields, None, consumed, base)
            if not got:
                continue
            name, tokens = got
            score = sum(1 for t in tokens if not t.startswith("=") and not t.startswith("K") and not t.startswith("O") and t != "mix")
            key = (op, glc)
            if key not in best or score > best[key][0]:
                best[key] = (score, name, tokens)
        for (op, glc), (_, name, tokens) in best.items():
            if glc:
                continue
            row = rows.setdefault((encoding, op), [name, " ".join(tokens), None])
            if (op, 1) in best and best[(op, 1)][2] != tokens:
                row[2] = " ".join(best[(op, 1)][2])
    return rows


def main():
    merged = {}
    for cpu in CPUS:
        print("probing", cpu, file=sys.stderr)
        for key, (name, tokens, glc) in table(cpu).items():
            entry = (key[0], key[1], name, tokens, glc)
            merged.setdefault(entry, []).append(cpu)
    version = subprocess.run(["llvm-mc", "--version"], capture_output=True, text=True).stdout
    version = re.search(r"LLVM version (\S+)", version).group(1)
    rows = sorted(merged.items(), key=lambda e: (ENCODINGS.index(next(x for x in ENCODINGS if x[0] == e[0][0])), e[0][1], e[0][2]))
    lines = []
    for (encoding, op, name, tokens, glc), cpus in rows:
        mask = "".join("1" if c in cpus else "0" for c in CPUS)
        text = "%s %03x %s %s %s" % (encoding, op, mask, name, tokens if tokens else "-")
        if glc:
            text += " | " + glc
        lines.append(text)
    with open(OUT, "w", encoding="utf-8", newline="\n") as f:
        f.write("// Copyright (c) 2026 Ahmed Seada\n")
        f.write("// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.\n\n")
        f.write("namespace Idrak.Tests.Gcn;\n\n")
        f.write("// Made by tests/Idrak.Tests/data/gcn/interpreter/make_opcode_table.py from LLVM %s's disassembler; do not edit, rerun the\n" % version)
        f.write("// script. One opcode a line: the encoding, the opcode (hex), the CPUs that have it (%s, 1 or 0 each), the mnemonic\n" % ", ".join(CPUS))
        f.write("// as LLVM prints it, and the operand template in print order (a field, its register class and width; '=' a fixed\n")
        f.write("// operand; K an immediate and its kind; O the offset's kind); after '|', FLAT's template when glc is set.\n")
        f.write("internal static partial class GcnOpcodeTable\n{\n")
        f.write("    internal const string Cpus = \"%s\";\n\n" % " ".join(CPUS))
        f.write("    internal const string Text = \"\"\"\n")
        for line in lines:
            f.write("        " + line + "\n")
        f.write("        \"\"\";\n}\n")
    print("wrote %d opcodes to %s" % (len(lines), os.path.normpath(OUT)), file=sys.stderr)


if __name__ == "__main__":
    main()
