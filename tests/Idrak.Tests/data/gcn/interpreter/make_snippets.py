#!/usr/bin/env python3
# Makes tests/Idrak.Tests/data/gcn/interpreter/snippets.golden: the hand-written GCN snippets of this folder (*.s) as
# LLVM assembles them for gfx900 (the same encodings as gfx90c), for the GCN interpreter's tests (plan 35, G1b).
# Development only: the tests read the golden, never LLVM.
#
#   python3 -I tests/Idrak.Tests/data/gcn/interpreter/make_snippets.py
#
# Each snippet is assembled alone (llvm-mc -filetype=obj, its labels resolved in the object) and disassembled
# (llvm-objdump -d, every byte accounted for): one record a line, tab-separated: the snippet, the offset (hex), the
# instruction as llvm-objdump prints it (a branch's target as its 16-bit dword count, unsigned, as LLVM prints it),
# and its words (hex, as llvm-objdump groups them).
import glob
import os
import re
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    version = re.search(r"LLVM version (\S+)", subprocess.run(["llvm-mc", "--version"], capture_output=True, text=True).stdout).group(1)
    lines = [
        "# The GCN interpreter's snippets (*.s in this folder) as LLVM %s assembles them for gfx900; made by" % version,
        "# make_snippets.py, do not edit. Tab-separated: snippet, offset (hex), instruction (llvm-objdump's text), words (hex).",
    ]
    with tempfile.TemporaryDirectory() as work:
        for source in sorted(glob.glob(os.path.join(HERE, "*.s"))):
            name = os.path.splitext(os.path.basename(source))[0]
            obj = os.path.join(work, name + ".o")
            subprocess.run(["llvm-mc", "-triple=amdgcn", "-mcpu=gfx900", "-filetype=obj", "-o", obj, source], check=True)
            relocations = subprocess.run(["llvm-readelf", "-r", obj], capture_output=True, text=True, check=True).stdout
            if "There are no relocations" not in relocations:
                sys.exit(name + ": the object has relocations; every label must be resolved within the snippet")
            out = subprocess.run(["llvm-objdump", "-d", "--no-leading-addr", obj], capture_output=True, text=True, check=True).stdout
            count = 0
            expected = 0
            for line in out.splitlines():
                m = re.match(r"^\s+(\S.*?)\s*// ([0-9A-F]{12}): ([0-9A-F ]+?)(?:\s+<.*>)?$", line)
                if not m:
                    continue
                text, offset, words = m.group(1), int(m.group(2), 16), m.group(3).split()
                if offset != expected:
                    sys.exit("%s: an instruction at 0x%x was not read (the next is at 0x%x)" % (name, expected, offset))
                expected = offset + 4 * len(words)
                lines.append("%s\t%x\t%s\t%s" % (name, offset, text, " ".join(w.lower() for w in words)))
                count += 1
            print("%s: %d instructions" % (name, count), file=sys.stderr)
    with open(os.path.join(HERE, "snippets.golden"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
